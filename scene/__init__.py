# modified
import json
import math
import os
import random
from typing import List, Optional

import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np
import einops

from arguments import GroupParams, ModelParams
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: GroupParams,
        gaussians: GaussianModel,
        scene_num: int = 0,
        load_iteration: Optional[int] = None,
        shuffle: bool = True,
        resolution_scales: List[float] = [1.0],
    ) -> None:
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.canonical_rays: torch.Tensor
        self.train_audio = []
        self.test_audio = []

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        # colmap / nerf synthetic dataset 구분
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        os.makedirs(self.model_path, exist_ok=True)
        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        
        data_root = f"./avnerf2/release/{scene_num}"
        position = json.loads(open(os.path.join(os.path.dirname(data_root[:-1]), "position3.json"), "r").read())
        position = torch.tensor(position[scene_num]["source_position"][:3]).cuda()
        self.position = position

        clip_len = 0.5
        sr = 22050
        wav_len = int(2 * clip_len * sr)

        
        audio_bi, audio_sc = self.load_audio(data_root, sr)

        for i in range(len(self.train_cameras[1.0])):
            self.process_audio_data(self.train_cameras[1.0][i], audio_bi, audio_sc, sr, wav_len, clip_len, is_test=False)
        for i in range(len(self.test_cameras[1.0])):
            self.process_audio_data(self.test_cameras[1.0][i], audio_bi, audio_sc, sr, wav_len, clip_len, is_test=True)

    def load_audio(self, data_root, sr):
        # Load binaural audio
        if os.path.exists(os.path.join(data_root, "binaural_syn_re.wav")):
            audio_bi, _ = librosa.load(os.path.join(data_root, "binaural_syn_re.wav"), sr=sr, mono=False)
        else:
            print("Unavailable, re-process binaural...")
            audio_bi_path = os.path.join(data_root, "binaural_syn.wav")
            audio_bi, _ = librosa.load(audio_bi_path, sr=sr, mono=False)
            audio_bi = audio_bi / np.abs(audio_bi).max()
            sf.write(os.path.join(data_root, "binaural_syn_re.wav"), audio_bi.T, sr, 'PCM_16')
        
        # Load source audio
        if os.path.exists(os.path.join(data_root, "source_syn_re.wav")):
            audio_sc, _ = librosa.load(os.path.join(data_root, "source_syn_re.wav"), sr=sr, mono=True)
        else:
            print("Unavailable, re-process source...")
            audio_sc_path = os.path.join(data_root, "source_syn.wav")
            audio_sc, _ = librosa.load(audio_sc_path, sr=sr, mono=True)
            audio_sc = audio_sc / np.abs(audio_sc).max()
            sf.write(os.path.join(data_root, "source_syn_re.wav"), audio_sc.T, sr, 'PCM_16')
        
        return audio_bi, audio_sc

    def process_audio_data(self, camera, audio_bi, audio_sc, sr, wav_len, clip_len, is_test):
        data = self.prepare_audio_data(camera, audio_bi, audio_sc, sr, wav_len, clip_len)
        
        if is_test:
            self.test_audio.append(data)
        else:
            self.train_audio.append(data)

    def prepare_audio_data(self, camera, audio_bi, audio_sc, sr, wav_len, clip_len):
        xyz = camera.camera_center
        ori = camera.camera_direction
        data = {"pos": xyz[:2], "dir": ori}

        # 상대 방향 계산
        ori = relative_angle(self.position[:2], xyz[:2], ori[:2])
        data["ori"] = ori

        # 오디오 클립 준비
        time = int(camera.image_name.split('.')[0])
        st_idx = max(0, int(sr * (time - clip_len)))
        ed_idx = min(audio_bi.shape[1]-1, int(sr * (time + clip_len)))

        # 오디오 클립 및 패딩 처리
        audio_bi_clip, audio_sc_clip = self.pad_audio_clips(audio_bi[:, st_idx:ed_idx], audio_sc[st_idx:ed_idx], wav_len)

        # STFT 변환 및 저장
        spec_bi = stft(audio_bi_clip)
        spec_sc = stft(audio_sc_clip)
        
        data["mag_bi"] = torch.from_numpy(np.abs(spec_bi)).to(torch.float32).to(xyz.device)
        data["mag_sc"] = torch.from_numpy(np.abs(spec_sc)).to(torch.float32).to(xyz.device)
        data["source"] = self.position
        data["target"] = xyz

        return data

    def pad_audio_clips(self, audio_bi_clip, audio_sc_clip, wav_len):
        if audio_bi_clip.shape[1] < wav_len:
            pad_len = wav_len - audio_bi_clip.shape[1]
            audio_bi_clip = np.concatenate((audio_bi_clip, np.zeros((2, pad_len))), axis=1)
            audio_sc_clip = np.concatenate((audio_sc_clip, np.zeros((pad_len))), axis=0)
        elif audio_bi_clip.shape[1] > wav_len:
            audio_bi_clip = audio_bi_clip[:, :wav_len]
            audio_sc_clip = audio_sc_clip[:wav_len]

        return audio_bi_clip, audio_sc_clip

    def save(self, iteration: int) -> None:
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale: float = 1.0) -> List[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = 1.0) -> List[Camera]:
        return self.test_cameras[scale]

    def getTrainAudio(self):
        return self.train_audio

    def getTestAudio(self):
        return self.test_audio


    def get_canonical_rays(self, scale: float = 1.0) -> torch.Tensor:
        ref_camera: Camera = self.train_cameras[scale][0] 
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
        return camera_dirs.cuda()


def relative_angle(source, xy, ori): 
    s = source - xy
    s = s.cpu().numpy()
    s = s / np.linalg.norm(s)
    
    d = ori.cpu().numpy() / np.linalg.norm(ori.cpu().numpy())
    theta = np.arccos(np.clip(np.dot(s, d), -1, 1)) / (1.01 * np.pi)
    rho = np.arcsin(np.clip(np.cross(s, d), -1, 1))
    if rho < 0:
        theta *= -1
    return torch.tensor([theta], dtype=torch.float32).to(xy.device)

def stft(signal):
    spec = librosa.stft(signal, n_fft=512)
    if spec.ndim == 2:
        spec = spec.T
    elif spec.ndim == 3:
        spec = einops.rearrange(spec, "c f t -> c t f")
    else:
        raise NotImplementedError
    return spec



# class Scene:
#     gaussians: GaussianModel

#     def __init__(
#         self,
#         args: GroupParams,
#         gaussians: GaussianModel,
#         load_iteration: Optional[int] = None,
#         shuffle: bool = True,
#         resolution_scales: List[float] = [1.0],
#     ) -> None:
#         """b
#         :param path: Path to colmap scene main folder.
#         """
#         self.model_path = args.model_path
#         self.loaded_iter = None
#         self.gaussians = gaussians
#         self.canonical_rays: torch.Tensor

#         if load_iteration:
#             if load_iteration == -1:
#                 self.loaded_iter = searchForMaxIteration(
#                     os.path.join(self.model_path, "point_cloud")
#                 )
#             else:
#                 self.loaded_iter = load_iteration
#             print(f"Loading trained model at iteration {self.loaded_iter}")

#         self.train_cameras = {}
#         self.test_cameras = {}

#         # colmap / nerf synthetic dataset 구분 
#         if os.path.exists(os.path.join(args.source_path, "sparse")):
#             scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
#         elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
#             print("Found transforms_train.json file, assuming Blender data set!")
#             scene_info = sceneLoadTypeCallbacks["Blender"](
#                 args.source_path, args.white_background, args.eval
#             )
#         else:
#             assert False, "Could not recognize scene type!"

#         os.makedirs(self.model_path, exist_ok=True)
#         if not self.loaded_iter:
#             with open(scene_info.ply_path, "rb") as src_file, open(
#                 os.path.join(self.model_path, "input.ply"), "wb"
#             ) as dest_file:
#                 dest_file.write(src_file.read())
#             json_cams = []
#             camlist = []
#             if scene_info.test_cameras:
#                 camlist.extend(scene_info.test_cameras)
#             if scene_info.train_cameras:
#                 camlist.extend(scene_info.train_cameras)
#             for id, cam in enumerate(camlist):
#                 json_cams.append(camera_to_JSON(id, cam))
#             with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
#                 json.dump(json_cams, file)

#         if shuffle:
#             random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
#             # # NOTE: I do not want to shuffle the test set
#             # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

#         self.cameras_extent = scene_info.nerf_normalization["radius"]

#         for resolution_scale in resolution_scales:
#             print("Loading Training Cameras")
#             self.train_cameras[resolution_scale] = cameraList_from_camInfos(
#                 scene_info.train_cameras, resolution_scale, args
#             )
#             print("Loading Test Cameras")
#             self.test_cameras[resolution_scale] = cameraList_from_camInfos(
#                 scene_info.test_cameras, resolution_scale, args
#             )

#         if self.loaded_iter:
#             self.gaussians.load_ply(
#                 os.path.join(
#                     self.model_path,
#                     "point_cloud",
#                     "iteration_" + str(self.loaded_iter),
#                     "point_cloud.ply",
#                 )
#             )
#         else:
#             self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

#     def save(self, iteration: int) -> None:
#         point_cloud_path = os.path.join(
#             self.model_path, f"point_cloud/iteration_{iteration}"
#         )
#         self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

#     def getTrainCameras(self, scale: float = 1.0) -> List[Camera]:
#         return self.train_cameras[scale]

#     def getTestCameras(self, scale: float = 1.0) -> List[Camera]:
#         return self.test_cameras[scale]

#     ### 추가된 부분
#     # 장면(2D image)의 각 pixel 위치에서 나오는 ray 방향(to 3D)을 계산하는 부분 (canonical ray를 계산)
#     # 특정 카메라에서 촬영된 이미지의 각 pixel에서 어떤 방향으로 광선이 뻗어 나가는지 알아야 카메라의 위치나 방향이 변경되더라도 일관된 방식으로 gaussian projection 가능
#     def get_canonical_rays(self, scale: float = 1.0) -> torch.Tensor:
#         # NOTE: some datasets do not share the same intrinsic (e.g. DTU)
#         # get reference camera
#         ref_camera: Camera = self.train_cameras[scale][0] # 1st camera 
#         # TODO: inject intrinsic
#         H, W = ref_camera.image_height, ref_camera.image_width
#         cen_x = W / 2
#         cen_y = H / 2
#         tan_fovx = math.tan(ref_camera.FoVx * 0.5)
#         tan_fovy = math.tan(ref_camera.FoVy * 0.5)
#         # focal length
#         focal_x = W / (2.0 * tan_fovx)
#         focal_y = H / (2.0 * tan_fovy)

#         x, y = torch.meshgrid( # 2D image plane -> 각 pixel
#             torch.arange(W),
#             torch.arange(H),
#             indexing="xy",
#         )
#         x = x.flatten()  # [H * W]
#         y = y.flatten()  # [H * W]
#         camera_dirs = F.pad( # camera direction
#             torch.stack(
#                 [
#                     (x - cen_x + 0.5) / focal_x,
#                     (y - cen_y + 0.5) / focal_y,
#                 ],
#                 dim=-1,
#             ),
#             (0, 1),
#             value=1.0,
#         )  # [H * W, 3]
#         # NOTE: it is not normalized
#         return camera_dirs.cuda()
