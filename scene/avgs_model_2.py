#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_model import GaussianModel
import einops
import torch.nn.functional as F
import math
import time

# from pykeops.torch import LazyTensor

def mydropout(tensor, p=0.5, training=True):
    if not training or p == 0:
        return tensor
    else:
        batch_size = tensor.shape[0]
        random_tensor = torch.rand(batch_size, device=tensor.device)
        new_tensor = [torch.zeros_like(tensor[i]) if random_tensor[i] <= p else tensor[i] for i in range(batch_size)]
        new_tensor = torch.stack(new_tensor, dim=0) # [B, ...]
        return new_tensor

class Embedding(nn.Module):
    def __init__(self, num_layer, num_embed, ch):
        super().__init__()
        self.embeds = nn.Parameter(torch.randn(num_embed, num_layer, ch) / math.sqrt(ch), requires_grad=True)
        self.num_embed = num_embed
    
    def forward(self, ori):
        embeds = torch.cat([self.embeds[-1:], self.embeds, self.embeds[:1]], dim=0)
        ori = (ori + 1) / 2 * self.num_embed
        t_value = torch.arange(-1, self.num_embed + 1, device=ori.device)
        right_idx = torch.searchsorted(t_value, ori, right=False)
        left_idx = right_idx - 1

        left_dis = ori - t_value[left_idx]
        right_dis = t_value[right_idx] - ori
        left_dis = torch.clamp(left_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]
        right_dis = torch.clamp(right_dis, 0, 1).unsqueeze(1).unsqueeze(2) # [B, 1, 1]

        left_embed = embeds[left_idx] # [B, l, c]
        right_embed = embeds[right_idx] # [B, l, c]

        output = left_embed * right_dis + right_embed * left_dis
        return output # [B, l, c]

class MLPwSkip(nn.Module):
    def __init__(self,
                 in_ch,
                 intermediate_ch=256,
                 layer_num=4,
                 ):
        super().__init__()
        self.residual_layer = nn.Linear(in_ch, intermediate_ch)
        self.layers = nn.ModuleList()
        for layer_idx in range(layer_num):
            in_ch_ = in_ch if layer_idx == 0 else intermediate_ch
            out_ch_ = intermediate_ch
            self.layers.append(nn.Sequential(nn.Linear(in_ch_, out_ch_),
                                             nn.ReLU(inplace=True)))

    def forward(self, x, embed=None):
        residual = self.residual_layer(x)
        for layer_idx in range(len(self.layers)):
            if embed is not None:
                # embed [B, l, c]
                x = self.layers[layer_idx](x) + embed[:, layer_idx].unsqueeze(1)
            else:
                x = self.layers[layer_idx](x)
            if layer_idx == len(self.layers) // 2 - 1:
                x = x + residual
        return x

class embedding_module_log(nn.Module):
    def __init__(self, funcs=[torch.sin, torch.cos], num_freqs=20, max_freq=10, ch_dim=-1, include_in=True):
        super().__init__()
        self.functions = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.funcs = funcs
        self.include_in = include_in

    def forward(self, x_input):
        if self.include_in:
            out_list = [x_input]
        else:
            out_list = []
        for func in self.funcs:
            for freq in self.freqs:
                out_list.append(func(x_input*freq))
        return torch.cat(out_list, dim=self.ch_dim)
    
# source, target 주변의 point들만 사용
def select_top_k_percent_points(source, target, points, k):
    # Calculate distances between points and both source and target
    distances_to_source = torch.norm(points - source, dim=1)
    distances_to_target = torch.norm(points - target, dim=1)
    
    # Determine the threshold for top k% distance for each distance set
    source_threshold = torch.quantile(distances_to_source, k / 100)
    target_threshold = torch.quantile(distances_to_target, k / 100)
    
    # Select points within the top k% vicinity for either source or target
    selected_indices = torch.where(
        torch.logical_or(distances_to_source <= source_threshold, distances_to_target <= target_threshold)
    )[0]
    
    return selected_indices

# class AVGS(GaussianModel):
class AVGS(GaussianModel,nn.Module):
    def __init__(self, xyz, acoustic, intermediate_ch=128, k=15):
        nn.Module.__init__(self)
        # self._xyz = nn.parameter(xyz.requires_grad_(True))
        self._xyz = xyz
        # self.percentile_xyz = xyz

        self._acoustic = nn.Parameter(acoustic.requires_grad_(True))
        # self._acoustic = acoustic
        self.acoustic_gradient_accum = torch.empty(0)
        self.acoustic_denom = torch.empty(0)
        # self.tau_g = 0.0004
        self.tau_g = 0.000004

        # acoustic field params
        self.A_F = nn.Sequential(
            nn.Linear(xyz.shape[1]+acoustic.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ).cuda()
        self.k = k

        # binauralizer params
        self.p = 0
        self.training = True
        self.freq_num = 257
        self.pos_embedder = embedding_module_log(num_freqs=10, ch_dim=1).cuda()
        self.freq_embedder = embedding_module_log(num_freqs=10, ch_dim=1).cuda()
        self.query_prj = nn.Sequential(nn.Linear(42 + 21, intermediate_ch), nn.ReLU(inplace=True)).cuda()
        
        self.mix_mlp = MLPwSkip(intermediate_ch, intermediate_ch).cuda()
        self.mix_prj = nn.Linear(intermediate_ch, 1).cuda()

        self.ori_embedder = Embedding(4, 4, intermediate_ch).cuda()
        self.diff_mlp = MLPwSkip(intermediate_ch, intermediate_ch).cuda()
        self.diff_prj = nn.Linear(intermediate_ch, 1).cuda()

        self.post_process = nn.Sequential(nn.Conv2d(4, 16, 7, 1, 3),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(16, 16, 3, 1, 1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(16, 1, 3, 1, 1)).cuda()
        
    @property
    def get_acoustic(self):
        return self._acoustic
    
    def acoustic_field(self, source, listener):
        idx = select_top_k_percent_points(source, listener, self._xyz, self.k)
        xyz = self._xyz[idx]
        acoustic = self._acoustic[idx]

        G_S = (xyz - source) / torch.norm(xyz - source, dim=-1, keepdim=True)
        G_L = (xyz - listener) / torch.norm(xyz - listener, dim=-1, keepdim=True)

        C_S = self.A_F(torch.cat([acoustic,G_S],dim=-1))
        C_L = self.A_F(torch.cat([acoustic,G_L],dim=-1))

        # G_S = (self._xyz - source) / torch.norm(self._xyz - source, dim=-1, keepdim=True)
        # G_L = (self._xyz - listener) / torch.norm(self._xyz - listener, dim=-1, keepdim=True)

        # C_S = self.A_F(torch.cat([self._acoustic,G_S],dim=-1))
        # C_L = self.A_F(torch.cat([self._acoustic,G_L],dim=-1))

        return torch.cat([C_S, C_L],dim=-1), idx
    

    def binauralizer(self,x):
        # import pdb;pdb.set_trace()
        B = x["pos"].unsqueeze(dim=0).shape[0]
        pos = self.pos_embedder(x["pos"].unsqueeze(dim=0)) # [B, 42]
        # pos = self.pos_embedder(x["pos"]) # [B, 42]

        ### normalized coordinates (add)
        pos = (pos - pos.min()) / (pos.max() - pos.min())

        pos = mydropout(pos, p=self.p, training=self.training)
        freq = torch.linspace(-0.99, 0.99, self.freq_num, device=x["pos"].device).unsqueeze(1) # [F, 1]
        freq = self.freq_embedder(freq) # [F, 21]

        pos = einops.repeat(pos, "b c -> b f c", f=self.freq_num)
        freq = einops.repeat(freq, "f c -> b f c", b=B)
        query = torch.cat([pos, freq], dim=2) # [B, F, ?]
        query = self.query_prj(query) # [B, F, ?]

        feats, idx = self.acoustic_field(x["source"],x["target"])
        if self.training:
            noise = torch.randn_like(feats) * 0.1
            feats = feats + noise
        
        # Mix Mask Prediction
        feats = self.mix_mlp(query + torch.mean(feats,dim=0).unsqueeze(dim=0))
        mask_mix = self.mix_prj(feats).squeeze(-1) # [B, F]

        # Diff Mask Prediction
        ori = self.ori_embedder(x["ori"])  # Encode orientation
        feats = self.diff_mlp(feats, ori)
        mask_diff = self.diff_prj(feats).squeeze(-1)  # [B, F]
        mask_diff = torch.sigmoid(mask_diff) * 2 - 1  # Scale to [-1, 1]


        # Reconstruct left and right channels
        time_dim = x["mag_sc"].shape[0]
        mask_mix = einops.repeat(mask_mix, "b f -> b t f", t=time_dim)
        mask_diff = einops.repeat(mask_diff, "b f -> b t f", t=time_dim)
        reconstr_mono = x["mag_sc"] * mask_mix
        reconstr_diff = reconstr_mono * mask_diff
        reconstr_left = reconstr_mono + reconstr_diff
        reconstr_right = reconstr_mono - reconstr_diff

        #
        left_input = torch.stack([x["mag_sc"].unsqueeze(dim=0), mask_mix, mask_diff, reconstr_left], dim=1) # [B, 4, T, F]
        right_input = torch.stack([x["mag_sc"].unsqueeze(dim=0), mask_mix, -mask_diff, reconstr_right], dim=1) # [B, 4, T, F]
        left_output = self.post_process(left_input).squeeze(1)
        right_output = self.post_process(right_input).squeeze(1)
        reconstr_left = reconstr_left + left_output
        reconstr_right = reconstr_right + right_output
        reconstr = torch.stack([reconstr_left, reconstr_right], dim=1) # [B, 2, T, F]
        reconstr = F.relu(reconstr)

        return {"reconstr_mono": reconstr_mono,
                "reconstr": reconstr,
                "idx": idx}


    def train_phase(self,training):
        self.training=training

    # def training_setup(self, training_args):
    def training_setup(self):
        # self.percent_dense = training_args.percent_dense
        self.acoustic_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.acoustic_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], "name": "xyz"},
            {'params': [self._acoustic], "name": "acoustic"},
            {'params': self.A_F.parameters(), "name": "A_F"},
            {'params': self.pos_embedder.parameters(), "name": "pos_embedder"},
            {'params': self.freq_embedder.parameters(), "name": "freq_embedder"},
            {'params': self.query_prj.parameters(), "name": "query_prj"},
            {'params': self.mix_mlp.parameters(), "name": "mix_mlp"},
            {'params': self.mix_prj.parameters(), "name": "mix_prj"},
            {'params': self.ori_embedder.parameters(), "name": "ori_embedder"},
            {'params': self.diff_mlp.parameters(), "name": "diff_mlp"},
            {'params': self.diff_prj.parameters(), "name": "diff_prj"},
            {'params': self.post_process.parameters(), "name": "post_process"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=5e-4, weight_decay=1e-4)

        # if self.pretrained_exposures is None:
        #     self.exposure_optimizer = torch.optim.Adam([self._exposure])

        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        
        # self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
        #                                                 lr_delay_steps=training_args.exposure_lr_delay_steps,
        #                                                 lr_delay_mult=training_args.exposure_lr_delay_mult,
        #                                                 max_steps=training_args.iterations)

    def acoustic_add_densification_stats(self, idx):
        # import pdb;pdb.set_trace()
        self.acoustic_gradient_accum[idx] += torch.norm(torch.cat([self._xyz.grad[idx],self._acoustic.grad[idx]],dim=1),dim=-1,keepdim=True)
        # self.acoustic_gradient_accum[idx] += torch.sum(torch.abs(torch.cat([self._xyz.grad[idx], self._acoustic.grad[idx]], dim=1)), dim=-1, keepdim=True)

        # torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.acoustic_denom[idx] += 1

    def acoustic_cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        # import pdb;pdb.set_trace()
        for group in self.optimizer.param_groups:
            # assert len(group["params"]) == 1
            if group["name"] not in ["xyz","acoustic"]:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    
    def acoustic_densification_postfix(self, new_xyz, new_acoustic):
        d = {"xyz": new_xyz,
        "acoustic": new_acoustic}

        optimizable_tensors = self.acoustic_cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._acoustic = optimizable_tensors["acoustic"]

        self.acoustic_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.acoustic_denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def acoustic_densify(self,):
        grads = self.acoustic_gradient_accum / self.acoustic_denom
        grads[grads.isnan()] = 0.0
        
        significant_pts_mask = torch.where(torch.norm(grads, dim=-1) >= self.tau_g, True, False)
        # significant_pts_mask = torch.where(torch.sum(grads, dim=-1) >= self.tau_g, True, False)

        significant_idx = torch.nonzero(significant_pts_mask, as_tuple=True)[0]
        significant_positions = self._xyz[significant_idx]  
        
        num_new_points = significant_positions.size(0)  
        # import pdb;pdb.set_trace()
        new_xyz = torch.normal(mean=significant_positions, std=0.1)  

        new_acoustic = torch.rand((num_new_points, self._acoustic.size(1))).cuda()

        self.acoustic_densification_postfix(new_xyz,new_acoustic)
        # print(self._xyz.shape)



    # def acoustic_add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.acoustic_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.acoustic_denom[update_filter] += 1


    # def remove_radius_outliers_torch(self, xyz, nb_points=8, radius=0.1):

    #     import pdb;pdb.set_trace()
    #     dist = torch.cdist(xyz, xyz)  # [N, N] 형태

    #     neighbor_count = (dist < radius).sum(dim=1)

    #     inliers = neighbor_count >= nb_points
    #     return xyz[inliers], inliers.nonzero(as_tuple=True)[0]
        
    

    # def remove_radius_outliers_keops(self, xyz, nb_points=8, radius=0.1):
    #     N, D = xyz.shape
    #     xyz_i = LazyTensor(xyz[:, None, :])  # (N, 1, D)
    #     xyz_j = LazyTensor(xyz[None, :, :])  # (1, N, D)
        
    #     dist = ((xyz_i - xyz_j) ** 2).sum(-1).sqrt().eval()
    #     neighbor_count = (dist < radius).sum(-1)
        
    #     inliers = neighbor_count >= nb_points
    #     return xyz[inliers], inliers.nonzero(as_tuple=True)[0]
        
    def remove_radius_outliers_batched(self, xyz, nb_points=8, radius=0.1, batch_size=4): # 512
        num_points = xyz.shape[0]
        inliers = torch.zeros(num_points, dtype=torch.bool, device=xyz.device)
        
        for i in range(0, num_points, batch_size):
            end_i = min(i + batch_size, num_points)
            batch_xyz = xyz[i:end_i]
            
            dist = torch.cdist(batch_xyz, xyz)
            neighbor_count = (dist < radius).sum(dim=1)
            
            inliers[i:end_i] = neighbor_count >= nb_points

        return xyz[inliers], inliers.nonzero(as_tuple=True)[0]

    def point_management(self):
        # import pdb;pdb.set_trace()
        # print(self._xyz.shape)
        # grads = self.xyz_gradient_accum / self.denom
        # grads[grads.isnan()] = 0.0

        # # Densification steps (retain as is)
        # self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        # Convert the points to an Open3D point cloud
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(self._xyz.detach().cpu().numpy())
        
        # import pdb;pdb.set_trace()
        # start = time.time()
        cl_,ind_ = self.remove_radius_outliers_batched(self._xyz)
        # print(time.time()-start,"sec")


        # Apply Open3D's remove_radius_outlier method
        # start = time.time()
        # cl, ind = pcd.remove_radius_outlier(nb_points=8, radius=0.1)
        # print(time.time()-start,"sec")
        # import pdb;pdb.set_trace()
        # Get the mask of valid points to keep
        valid_points_mask = np.zeros(len(self._xyz), dtype=bool)
        valid_points_mask[ind_.cpu().numpy()] = True
        
        # Apply the mask to prune points
        self.prune_points(torch.tensor(~valid_points_mask, device='cuda'))

        torch.cuda.empty_cache()

        # Optionally, still prune based on opacity and scaling if needed
        # prune_mask = (self.get_opacity < min_opacity).squeeze()
        # if max_screen_size:
        #     big_points_vs = self.max_radii2D > max_screen_size
        #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        # self.prune_points(prune_mask)

        

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in ["xyz","acoustic"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    
    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._acoustic = optimizable_tensors["acoustic"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        self.acoustic_gradient_accum = self.acoustic_gradient_accum[valid_points_mask]

        self.acoustic_denom = self.acoustic_denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]

        






    # def setup_functions(self):
    #     def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    #         L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    #         actual_covariance = L @ L.transpose(1, 2)
    #         symm = strip_symmetric(actual_covariance)
    #         return symm
        
    #     self.scaling_activation = torch.exp
    #     self.scaling_inverse_activation = torch.log

    #     self.covariance_activation = build_covariance_from_scaling_rotation

    #     self.opacity_activation = torch.sigmoid
    #     self.inverse_opacity_activation = inverse_sigmoid

    #     self.rotation_activation = torch.nn.functional.normalize


    # def __init__(self, sh_degree):
    #     self.active_sh_degree = 0
    #     self.max_sh_degree = sh_degree  
    #     self._xyz = torch.empty(0)
    #     self._features_dc = torch.empty(0)
    #     self._features_rest = torch.empty(0)
    #     self._scaling = torch.empty(0)
    #     self._rotation = torch.empty(0)
    #     self._opacity = torch.empty(0)
    #     self.max_radii2D = torch.empty(0)
    #     self.xyz_gradient_accum = torch.empty(0)
    #     self.denom = torch.empty(0)
    #     self.optimizer = None
    #     self.percent_dense = 0
    #     self.spatial_lr_scale = 0
    #     self.setup_functions()

    # def capture(self):
    #     return (
    #         self.active_sh_degree,
    #         self._xyz,
    #         self._features_dc,
    #         self._features_rest,
    #         self._scaling,
    #         self._rotation,
    #         self._opacity,
    #         self.max_radii2D,
    #         self.xyz_gradient_accum,
    #         self.denom,
    #         self.optimizer.state_dict(),
    #         self.spatial_lr_scale,
    #     )
    
    # def restore(self, model_args, training_args):
    #     (self.active_sh_degree, 
    #     self._xyz, 
    #     self._features_dc, 
    #     self._features_rest,
    #     self._scaling, 
    #     self._rotation, 
    #     self._opacity,
    #     self.max_radii2D, 
    #     xyz_gradient_accum, 
    #     denom,
    #     opt_dict, 
    #     self.spatial_lr_scale) = model_args
    #     self.training_setup(training_args)
    #     self.xyz_gradient_accum = xyz_gradient_accum
    #     self.denom = denom
    #     self.optimizer.load_state_dict(opt_dict)

    # @property
    # def get_scaling(self):
    #     return self.scaling_activation(self._scaling)
    
    # @property
    # def get_rotation(self):
    #     return self.rotation_activation(self._rotation)
    
    # @property
    # def get_xyz(self):
    #     return self._xyz
    
    # @property
    # def get_features(self):
    #     features_dc = self._features_dc
    #     features_rest = self._features_rest
    #     return torch.cat((features_dc, features_rest), dim=1)
    
    # @property
    # def get_features_dc(self):
    #     return self._features_dc
    
    # @property
    # def get_features_rest(self):
    #     return self._features_rest
    
    # @property
    # def get_opacity(self):
    #     return self.opacity_activation(self._opacity)
    
    # @property
    # def get_exposure(self):
    #     return self._exposure

    # def get_exposure_from_name(self, image_name):
    #     if self.pretrained_exposures is None:
    #         return self._exposure[self.exposure_mapping[image_name]]
    #     else:
    #         return self.pretrained_exposures[image_name]
    
    # def get_covariance(self, scaling_modifier = 1):
    #     return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # def oneupSHdegree(self):
    #     if self.active_sh_degree < self.max_sh_degree:
    #         self.active_sh_degree += 1

    # def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
    #     self.spatial_lr_scale = spatial_lr_scale
    #     fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
    #     fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
    #     features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
    #     features[:, :3, 0 ] = fused_color
    #     features[:, 3:, 1:] = 0.0

    #     print("Number of points at initialisation : ", fused_point_cloud.shape[0])

    #     dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
    #     scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
    #     rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
    #     rots[:, 0] = 1

    #     opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

    #     self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    #     self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
    #     self._scaling = nn.Parameter(scales.requires_grad_(True))
    #     self._rotation = nn.Parameter(rots.requires_grad_(True))
    #     self._opacity = nn.Parameter(opacities.requires_grad_(True))
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    #     self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
    #     self.pretrained_exposures = None
    #     exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
    #     self._exposure = nn.Parameter(exposure.requires_grad_(True))

    # def training_setup(self, training_args):
    #     self.percent_dense = training_args.percent_dense
    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

    #     l = [
    #         {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
    #         {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
    #         {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
    #         {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
    #         {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
    #         {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
    #     ]

    #     self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    #     if self.pretrained_exposures is None:
    #         self.exposure_optimizer = torch.optim.Adam([self._exposure])

    #     self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
    #                                                 lr_final=training_args.position_lr_final*self.spatial_lr_scale,
    #                                                 lr_delay_mult=training_args.position_lr_delay_mult,
    #                                                 max_steps=training_args.position_lr_max_steps)
        
    #     self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
    #                                                     lr_delay_steps=training_args.exposure_lr_delay_steps,
    #                                                     lr_delay_mult=training_args.exposure_lr_delay_mult,
    #                                                     max_steps=training_args.iterations)

    # def update_learning_rate(self, iteration):
    #     ''' Learning rate scheduling per step '''
    #     if self.pretrained_exposures is None:
    #         for param_group in self.exposure_optimizer.param_groups:
    #             param_group['lr'] = self.exposure_scheduler_args(iteration)

    #     for param_group in self.optimizer.param_groups:
    #         if param_group["name"] == "xyz":
    #             lr = self.xyz_scheduler_args(iteration)
    #             param_group['lr'] = lr
    #             return lr

    # def construct_list_of_attributes(self):
    #     l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    #     # All channels except the 3 DC
    #     for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
    #         l.append('f_dc_{}'.format(i))
    #     for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
    #         l.append('f_rest_{}'.format(i))
    #     l.append('opacity')
    #     for i in range(self._scaling.shape[1]):
    #         l.append('scale_{}'.format(i))
    #     for i in range(self._rotation.shape[1]):
    #         l.append('rot_{}'.format(i))
    #     return l

    # def save_ply(self, path):
    #     mkdir_p(os.path.dirname(path))

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    # def reset_opacity(self):
    #     opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
    #     optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
    #     self._opacity = optimizable_tensors["opacity"]

    # def load_ply(self, path, use_train_test_exp = False):
    #     plydata = PlyData.read(path)
    #     if use_train_test_exp:
    #         exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
    #         if os.path.exists(exposure_file):
    #             with open(exposure_file, "r") as f:
    #                 exposures = json.load(f)
    #             self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
    #             print(f"Pretrained exposures loaded.")
    #         else:
    #             print(f"No exposure to be loaded at {exposure_file}")
    #             self.pretrained_exposures = None

    #     xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
    #                     np.asarray(plydata.elements[0]["y"]),
    #                     np.asarray(plydata.elements[0]["z"])),  axis=1)
    #     opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    #     features_dc = np.zeros((xyz.shape[0], 3, 1))
    #     features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    #     features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    #     features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    #     extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    #     extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    #     assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    #     features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    #     for idx, attr_name in enumerate(extra_f_names):
    #         features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    #     # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    #     features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

    #     scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    #     scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    #     scales = np.zeros((xyz.shape[0], len(scale_names)))
    #     for idx, attr_name in enumerate(scale_names):
    #         scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    #     rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    #     rots = np.zeros((xyz.shape[0], len(rot_names)))
    #     for idx, attr_name in enumerate(rot_names):
    #         rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    #     self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    #     self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    #     self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    #     self.active_sh_degree = self.max_sh_degree

    # def replace_tensor_to_optimizer(self, tensor, name):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         if group["name"] == name:
    #             stored_state = self.optimizer.state.get(group['params'][0], None)
    #             stored_state["exp_avg"] = torch.zeros_like(tensor)
    #             stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors

    # def _prune_optimizer(self, mask):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:
    #             stored_state["exp_avg"] = stored_state["exp_avg"][mask]
    #             stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]
    #     return optimizable_tensors

    # def prune_points(self, mask):
    #     valid_points_mask = ~mask
    #     optimizable_tensors = self._prune_optimizer(valid_points_mask)

    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

    #     self.denom = self.denom[valid_points_mask]
    #     self.max_radii2D = self.max_radii2D[valid_points_mask]

    # def cat_tensors_to_optimizer(self, tensors_dict):
    #     optimizable_tensors = {}
    #     for group in self.optimizer.param_groups:
    #         assert len(group["params"]) == 1
    #         extension_tensor = tensors_dict[group["name"]]
    #         stored_state = self.optimizer.state.get(group['params'][0], None)
    #         if stored_state is not None:

    #             stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
    #             stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

    #             del self.optimizer.state[group['params'][0]]
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             self.optimizer.state[group['params'][0]] = stored_state

    #             optimizable_tensors[group["name"]] = group["params"][0]
    #         else:
    #             group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
    #             optimizable_tensors[group["name"]] = group["params"][0]

    #     return optimizable_tensors

    # def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
    #     d = {"xyz": new_xyz,
    #     "f_dc": new_features_dc,
    #     "f_rest": new_features_rest,
    #     "opacity": new_opacities,
    #     "scaling" : new_scaling,
    #     "rotation" : new_rotation}

    #     optimizable_tensors = self.cat_tensors_to_optimizer(d)
    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    # def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
    #     n_init_points = self.get_xyz.shape[0]
    #     # Extract points that satisfy the gradient condition
    #     padded_grad = torch.zeros((n_init_points), device="cuda")
    #     padded_grad[:grads.shape[0]] = grads.squeeze()
    #     selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

    #     stds = self.get_scaling[selected_pts_mask].repeat(N,1)
    #     means =torch.zeros((stds.size(0), 3),device="cuda")
    #     samples = torch.normal(mean=means, std=stds)
    #     rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
    #     new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
    #     new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
    #     new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
    #     new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
    #     new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
    #     new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    #     prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
    #     self.prune_points(prune_filter)

    # def densify_and_clone(self, grads, grad_threshold, scene_extent):
    #     # Extract points that satisfy the gradient condition
    #     selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
    #     selected_pts_mask = torch.logical_and(selected_pts_mask,
    #                                           torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
    #     new_xyz = self._xyz[selected_pts_mask]
    #     new_features_dc = self._features_dc[selected_pts_mask]
    #     new_features_rest = self._features_rest[selected_pts_mask]
    #     new_opacities = self._opacity[selected_pts_mask]
    #     new_scaling = self._scaling[selected_pts_mask]
    #     new_rotation = self._rotation[selected_pts_mask]

    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    # def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0

    #     self.densify_and_clone(grads, max_grad, extent)
    #     self.densify_and_split(grads, max_grad, extent)

    #     prune_mask = (self.get_opacity < min_opacity).squeeze()
    #     if max_screen_size:
    #         big_points_vs = self.max_radii2D > max_screen_size
    #         big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
    #         prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
    #     self.prune_points(prune_mask)

    #     torch.cuda.empty_cache()

    # def add_densification_stats(self, viewspace_point_tensor, update_filter):
    #     self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
    #     self.denom[update_filter] += 1