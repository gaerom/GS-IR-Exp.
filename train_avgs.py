import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Dict, List, Optional, Tuple, Union

import kornia
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm, trange

from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render, network_gui
from scene import GaussianModel, Scene, Camera
from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from scene.avgs_model_2 import AVGS
from AV_utils import Evaluator
import pickle
import json
import librosa
import soundfile as sf

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def get_tv_loss(gt_image: torch.Tensor, prediction: torch.Tensor, pad: int = 1, step: int = 1) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True))
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True))
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(-(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True))
            rgb_grad_w = torch.exp(-(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True))
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss

# ?
def get_masked_tv_loss(mask: torch.Tensor, gt_image: torch.Tensor, prediction: torch.Tensor, erosion: bool = False) -> torch.Tensor:
    rgb_grad_h = torch.exp(-(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True))
    rgb_grad_w = torch.exp(-(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True))
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)

    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]

    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss


def resize_tensorboard_img(img: torch.Tensor, max_res: int = 800) -> torch.Tensor:
    _, H, W = img.shape
    ratio = min(max_res / H, max_res / W)
    target_size = (int(H * ratio), int(W * ratio))
    transform = T.Resize(size=target_size)
    return transform(img) 


def material_acoustic_training(dataset: GroupParams, opt: GroupParams, pipe: GroupParams, testing_iterations: List[int], saving_iterations: List[int], checkpoint_iterations: int, checkpoint_path: Optional[str] = None, debug_from: int = -1, tone: bool = False, gamma: bool = False, normal_tv_weight: float = 1.0, brdf_tv_weight: float = 1.0, scene_num: Optional[str] = None) -> None:
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, scene_num)
    gaussians.training_setup(opt)
    tb_writer = prepare_output_and_logger(dataset)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    canonical_rays = scene.get_canonical_rays()

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model_params = checkpoint["gaussians"]
        first_iter = checkpoint["iteration"]
        gaussians.restore(model_params, opt)
        print(f"Loaded checkpoint from {checkpoint_path}")

    viewpoint_stack = scene.getTrainCameras().copy()
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    for iteration in range(first_iter + 1, opt.iterations + 1):
        iter_start.record()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random camera viewpoint
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        try:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T)
        except:
            continue

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        rendering_result = render(viewpoint_camera=viewpoint_cam, pc=gaussians, pipe=pipe, bg_color=bg, derive_normal=True)

        # Material Training Loss Calculations
        image = rendering_result["render"]
        depth_map = rendering_result["depth_map"]
        normal_map = rendering_result["normal_map"]
        albedo_map = rendering_result["albedo_map"]
        roughness_map = rendering_result["roughness_map"]
        metallic_map = rendering_result["metallic_map"]

        gt_image = viewpoint_cam.original_image.cuda()
        alpha_mask = viewpoint_cam.gt_alpha_mask.cuda()
        gt_image = (gt_image * alpha_mask + background[:, None, None] * (1.0 - alpha_mask)).clamp(0.0, 1.0)

        Ll1 = F.l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        normal_loss_weight = 1.0
        mask = rendering_result["normal_from_depth_mask"]
        normal_loss = F.l1_loss(normal_map[:, mask], normal_map[:, mask])
        loss += normal_loss_weight * normal_loss

        normal_tv_loss = get_tv_loss(gt_image, normal_map, pad=1, step=1)
        loss += normal_tv_loss * normal_tv_weight

        if (mask == 0).sum() > 0:
            brdf_tv_loss = get_masked_tv_loss(mask, gt_image, torch.cat([albedo_map, roughness_map, metallic_map], dim=0))
        else:
            brdf_tv_loss = get_tv_loss(gt_image, torch.cat([albedo_map, roughness_map, metallic_map], dim=0), pad=1, step=1)
        loss += brdf_tv_weight * brdf_tv_loss

        # Depth regularization
        depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = rendering_result["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((invDepth - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth

        loss.backward()

        # Acoustic Field Training
        avgs = AVGS(xyz=gaussians.get_xyz, acoustic=torch.cat([gaussians.get_features.flatten(start_dim=1), gaussians.get_rotation.flatten(start_dim=1)], dim=-1))
        avgs.training_setup()
        avgs.train()
        viewpoint_stack = scene.getTrainAudio().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))

        loss_list = []
        for audio_iter in range(1, 40001):
            if not viewpoint_stack:
                result = eval(avgs, scene.getTestAudio().copy())
                loss_list.append(result)
                viewpoint_stack = scene.getTrainAudio().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))

            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_audio = viewpoint_stack.pop(rand_idx)

            ret = avgs.binauralizer(viewpoint_audio)
            loss_vol = torch.sum(torch.prod(avgs._acoustic, dim=1))

            mag_bi_mean = viewpoint_audio["mag_bi"].mean(0)
            loss_mono = F.mse_loss(ret["reconstr_mono"][0], mag_bi_mean)
            loss_left = F.mse_loss(ret["reconstr"][0, 0], viewpoint_audio["mag_bi"][0])
            loss_right = F.mse_loss(ret["reconstr"][0, 1], viewpoint_audio["mag_bi"][1])
            loss_mag = loss_mono + loss_left + loss_right

            acoustic_loss = loss_mag + loss_vol
            loss_list.append(acoustic_loss.item())
            acoustic_loss.backward()

            # Densification
            if audio_iter < 40001:
                avgs.acoustic_add_densification_stats(ret["idx"])

                if audio_iter % 100 == 0:
                    avgs.acoustic_densify()

                if audio_iter % 3000 == 0:
                    avgs.point_management()

            avgs.optimizer.step()
            avgs.optimizer.zero_grad()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                progress_bar.update(10)

            if tb_writer:
                tb_writer.add_scalar("train/l1_loss", Ll1.item(), iteration)
                tb_writer.add_scalar("train/total_loss", loss.item(), iteration)

    with open(os.path.join(scene.model_path, "eval_loss.pkl"), 'wb') as f:
        pickle.dump(loss_list, f)

    progress_bar.close()
    print("Training complete.")


def prepare_output_and_logger(args: GroupParams) -> Optional[SummaryWriter]:
    if not args.model_path:
        unique_str = os.getenv("OAR_JOB_ID") if os.getenv("OAR_JOB_ID") else str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    if not TENSORBOARD_FOUND:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def eval(avgs, audio, save=False):
    avgs.eval()
    avgs.train_phase(False)
    evaluator = Evaluator()
    save_list = []

    audio_idx = list(range(len(audio)))
    with torch.no_grad():
        for i in range(len(audio_idx)):
            ret = avgs.binauralizer(audio[i])
            mag_prd = ret["reconstr"].cpu().numpy()
            phase_prd = audio[i]["phase_sc"]
            spec_prd = mag_prd * np.exp(1j * phase_prd[np.newaxis, :])
            wav_prd = librosa.istft(spec_prd.squeeze().transpose(0, 2, 1), length=22050)
            mag_gt = audio[i]["mag_bi"].cpu().numpy()
            wav_gt = audio[i]["wav_bi"]
            loss_list = evaluator.update(mag_prd, mag_gt, wav_prd, wav_gt)
            if save:
                save_list.append({"wav_prd": wav_prd, "wav_gt": wav_gt, "loss": loss_list, "img_idx": audio[i]["img_idx"]})

    result = evaluator.report()
    avgs.train()
    avgs.train_phase(True)
    return result


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 37_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 37_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--scene_num", type=str, default=None)
    parser.add_argument("--tone", action="store_true")
    parser.add_argument("--gamma", action="store_true")
    parser.add_argument("--normal_tv", default=5.0, type=float)
    parser.add_argument("--brdf_tv", default=1.0, type=float)
    args = parser.parse_args(sys.argv[1:])
    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    material_acoustic_training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_iterations=args.checkpoint_iterations,
        checkpoint_path=args.start_checkpoint,
        debug_from=args.debug_from,
        tone=args.tone,
        gamma=args.gamma,
        normal_tv_weight=args.normal_tv,
        brdf_tv_weight=args.brdf_tv,
        scene_num=args.scene_num
    )

    print("\nTraining complete.")
