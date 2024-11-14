import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.cameras import Camera
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

import torch
import torch.nn.functional as F

import librosa
import soundfile as sf
import numpy as np
import einops
import math


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, scene_num, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.depths, args.eval, args.train_test_exp)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            if self.model_path and not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
                
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

        # import pdb;pdb.set_trace()
        data_root = f"./release/{scene_num}"

        position = json.loads(open(os.path.join(os.path.dirname(data_root[:-1]), "position3.json"), "r").read())
        position = torch.tensor(position[scene_num]["source_position"][:3]).cuda()

        self.position = position

        clip_len = 0.5
        sr = 22050
        wav_len = int(2 * clip_len * sr)

        # audio
        if os.path.exists(os.path.join(data_root, "binaural_syn_re.wav")):
            audio_bi, _ = librosa.load(os.path.join(data_root, "binaural_syn_re.wav"), sr=sr, mono=False)
        else:
            print("Unavilable, re-process binaural...")
            audio_bi_path = os.path.join(data_root, "binaural_syn.wav")
            audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False) # [2, ?]
            audio_bi = audio_bi / np.abs(audio_bi).max()
            sf.write(os.path.join(data_root, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
        
        if os.path.exists(os.path.join(data_root, "source_syn_re.wav")):
            audio_sc, _ = librosa.load(os.path.join(data_root, "source_syn_re.wav"), sr=sr, mono=True)
        else:
            print("Unavilable, re-process source...")
            audio_sc_path = os.path.join(data_root, "source_syn.wav")
            audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True) # [?]
            audio_sc = audio_sc / np.abs(audio_sc).max()
            sf.write(os.path.join(data_root, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')

        self.train_audio = []
        self.test_audio = []
        
        # import pdb;pdb.set_trace()

        for i in range(len(self.train_cameras[1.0])):
            xyz = self.train_cameras[1.0][i].camera_center
            ori = self.train_cameras[1.0][i].camera_direction

            # data = {"pos": xyz}
            data = {"pos": xyz[:2]}

            data["dir"] = ori

            ori = relative_angle(position[:2],xyz[:2],ori[:2])
            data["ori"] = ori
            # data["ori"] = ori[:2]

    
            time = int(self.train_cameras[1.0][i].image_name.split('.')[0])
            # extract key frames at 1 fps
            # time = int(item["file_path"].split('/')[-1].split('.')[0])
            data["img_idx"] = time
            st_idx = max(0, int(sr * (time - clip_len)))
            ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
            if ed_idx - st_idx < int(clip_len * sr): continue
            audio_bi_clip = audio_bi[:, st_idx:ed_idx]
            audio_sc_clip = audio_sc[st_idx:ed_idx]

            # padding with zero
            if(ed_idx - st_idx < wav_len):
                pad_len = wav_len - (ed_idx - st_idx)
                audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
                audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
                print(f"padding from {ed_idx - st_idx} -> {wav_len}")
            elif(ed_idx - st_idx > wav_len):
                audio_bi_clip = audio_bi_clip[:, :wav_len]
                audio_sc_clip = audio_sc_clip[:wav_len]
                print(f"cutting from {ed_idx - st_idx} -> {wav_len}")

            # binaural
            spec_bi = stft(audio_bi_clip)
            mag_bi = np.abs(spec_bi) # [2, T, F]
            phase_bi = np.angle(spec_bi) # [2, T, F]
            data["mag_bi"] = torch.from_numpy(mag_bi).to(torch.float32).to(xyz.device)

            # source
            spec_sc = stft(audio_sc_clip)
            mag_sc = np.abs(spec_sc) # [T, F]
            phase_sc = np.angle(spec_sc) # [T, F]
            data["mag_sc"] = torch.from_numpy(mag_sc).to(torch.float32).to(xyz.device)

            data["source"] = position
            data["target"] = xyz
            
            self.train_audio.append(data)

        # import pdb;pdb.set_trace()

        for i in range(len(self.test_cameras[1.0])):
            xyz = self.test_cameras[1.0][i].camera_center
            ori = self.test_cameras[1.0][i].camera_direction

            # data = {"pos": xyz}
            data = {"pos": xyz[:2]}

            data["dir"] = ori

            ori = relative_angle(position[:2],xyz[:2],ori[:2]) # error
            data["ori"] = ori
            # data["ori"] = ori[:2]

    
            time = int(self.train_cameras[1.0][i].image_name.split('.')[0])
            # extract key frames at 1 fps
            # time = int(item["file_path"].split('/')[-1].split('.')[0])
            data["img_idx"] = time
            st_idx = max(0, int(sr * (time - clip_len)))
            ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))
            if ed_idx - st_idx < int(clip_len * sr): continue
            audio_bi_clip = audio_bi[:, st_idx:ed_idx]
            audio_sc_clip = audio_sc[st_idx:ed_idx]

            # padding with zero
            if(ed_idx - st_idx < wav_len):
                pad_len = wav_len - (ed_idx - st_idx)
                audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
                audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
                print(f"padding from {ed_idx - st_idx} -> {wav_len}")
            elif(ed_idx - st_idx > wav_len):
                audio_bi_clip = audio_bi_clip[:, :wav_len]
                audio_sc_clip = audio_sc_clip[:wav_len]
                print(f"cutting from {ed_idx - st_idx} -> {wav_len}")

            # binaural
            spec_bi = stft(audio_bi_clip)
            mag_bi = np.abs(spec_bi) # [2, T, F]
            phase_bi = np.angle(spec_bi) # [2, T, F]
            data["mag_bi"] = torch.from_numpy(mag_bi).to(torch.float32).to(xyz.device)

            # source
            spec_sc = stft(audio_sc_clip)
            mag_sc = np.abs(spec_sc) # [T, F]
            phase_sc = np.angle(spec_sc) # [T, F]
            data["mag_sc"] = torch.from_numpy(mag_sc,).to(torch.float32).to(xyz.device)

            data["wav_bi"] = audio_bi_clip
            data["phase_bi"] = phase_bi
            data["wav_sc"] = audio_sc_clip
            data["phase_sc"] = phase_sc


            data["source"] = position
            data["target"] = xyz
            
            self.test_audio.append(data)

        # import pdb;pdb.set_trace()



    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainAudio(self):
        return self.train_audio

    def getTestAudio(self):
        return self.test_audio
    
    def get_canonical_rays(self, scale: float = 1.0) -> torch.Tensor:
        # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
        # get reference camera
        ref_camera: Camera = self.train_cameras[scale][0]
        # TODO: inject intrinsic
        H, W = ref_camera.image_height, ref_camera.image_width
        cen_x = W / 2
        cen_y = H / 2
        tan_fovx = math.tan(ref_camera.FoVx * 0.5)
        tan_fovy = math.tan(ref_camera.FoVy * 0.5)
        focal_x = W / (2.0 * tan_fovx)
        focal_y = H / (2.0 * tan_fovy)

        x, y = torch.meshgrid(
            torch.arange(W),
            torch.arange(H),
            indexing="xy",
        )
        x = x.flatten()  # [H * W]
        y = y.flatten()  # [H * W]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - cen_x + 0.5) / focal_x,
                    (y - cen_y + 0.5) / focal_y,
                ],
                dim=-1,
            ),
            (0, 1),
            value=1.0,
        )  # [H * W, 3]
        # NOTE: it is not normalized
        return camera_dirs.cuda()


def relative_angle(source, xy, ori): # (-1, 1)
    # import pdb;pdb.set_trace()
    s = source - xy
    s = s.cpu().numpy()
    s = s / np.linalg.norm(s)
    
    ### invalid value encontered in divide
    d = ori.cpu().numpy() / np.linalg.norm(ori.cpu().numpy())
    theta = np.arccos(np.clip(np.dot(s, d), -1, 1)) / (1.01 * np.pi)
    rho = np.arcsin(np.clip(np.cross(s, d), -1, 1))
    if rho < 0:
        theta *= -1
    return torch.tensor([theta],dtype=torch.float32).to(xy.device)

def stft(signal):
    spec = librosa.stft(signal, n_fft=512)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c f t -> c t f")
    else:
        raise NotImplementedError
    return spec

