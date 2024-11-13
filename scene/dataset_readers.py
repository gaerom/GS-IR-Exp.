# modified

# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from scene.gaussian_model import BasicPointCloud
from utils.graphics_utils import focal2fov, fov2focal, getWorld2View2
from utils.sh_utils import SH2RGB


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray
    T: np.ndarray
    FovY: float
    FovX: float
    image: Image.Image
    image_path: str
    image_name: str
    depth_path: str
    depth_params: Optional[dict] = None
    width: int = 0
    height: int = 0
    is_test: bool = False


class SceneInfo(NamedTuple):
    point_cloud: Optional[BasicPointCloud]
    train_cameras: List
    test_cameras: List
    nerf_normalization: Dict
    ply_path: str
    is_nerf_synthetic: bool = False


def getNerfppNorm(cam_info: List[CameraInfo]) -> Dict:
    def get_center_and_diag(cam_centers: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics: Dict, cam_intrinsics: Dict, depths_params: Optional[Dict], images_folder: str,
    depths_folder: str, test_cam_names_list: List[str]
) -> List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        sys.stdout.write(f"Reading camera {idx + 1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height, width = intr.height, intr.width
        uid, R, T = intr.id, np.transpose(qvec2rotmat(extr.qvec)), np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x, focal_length_y = intr.params[:2]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Only undistorted PINHOLE or SIMPLE_PINHOLE cameras supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = depths_params.get(extr.name[:-n_remove]) if depths_params else None

        image_path = os.path.join(images_folder, extr.name)
        image = Image.open(image_path) # add
        depth_path = os.path.join(depths_folder, 'images', f"{extr.name[:-n_remove]}.png") if depths_folder else ""

        cam_info = CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
            image=image, image_path=image_path, image_name=extr.name, 
            depth_path=depth_path, width=width, height=height, is_test=extr.name in test_cam_names_list
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path: str) -> BasicPointCloud:
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("red", "u1"), ("green", "u1"), ("blue", "u1")]

    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path: str, images: str, depths: str, eval: bool, train_test_exp: bool, llffhold: int = 8 # eval, train_test_exp
) -> SceneInfo:
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    depths_params = None
    if depths:
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            med_scale = np.median(all_scales[all_scales > 0]) if (all_scales > 0).sum() else 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale
        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)

    cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
    test_cam_names_list = [name for idx, name in enumerate(sorted(cam_names)) if idx % llffhold == 0] if eval else []
    reading_dir = "images" if images is None else images

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir), depths_folder=os.path.join(path, depths),
        test_cam_names_list=test_cam_names_list
    )
    cam_infos = sorted(cam_infos_unsorted, key=lambda x: x.image_name)
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply")
        xyz, rgb, _ = read_points3D_binary(os.path.join(path, "sparse/0", "points3D.bin"))
        storePly(ply_path, xyz, rgb)
    pcd = fetchPly(ply_path)

    return SceneInfo(
        point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization, ply_path=ply_path, is_nerf_synthetic=False
    )


def readCamerasFromTransforms(
    path: str, transformsfile: str, depths_folder: str, white_background: bool, is_test: bool, extension: str = ".png"
) -> List[CameraInfo]:
    cam_infos = []
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1  # Change from OpenGL/Blender camera axes to COLMAP

            w2c = np.linalg.inv(c2w)
            R, T = np.transpose(w2c[:3, :3]), w2c[:3, 3]
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            depth_path = os.path.join(depths_folder, f"{image_name}.png") if depths_folder else ""
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            cam_infos.append(CameraInfo(
                uid=idx, R=R, T=T, FovY=fovy, FovX=fovx,
                image_path=image_path, image_name=image_name,
                width=image.size[0], height=image.size[1],
                depth_path=depth_path, depth_params=None, is_test=is_test
            ))
    return cam_infos


def readNerfSyntheticInfo(
    path: str, white_background: bool, depths: str, eval: bool, extension: str = ".png"
) -> SceneInfo:
    depths_folder = os.path.join(path, depths) if depths else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", depths_folder, white_background, True, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print(f"Generating random point cloud")
        num_pts = 100_000
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPly(ply_path)

    return SceneInfo(
        point_cloud=pcd, train_cameras=train_cam_infos, test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization, ply_path=ply_path, is_nerf_synthetic=True
    )


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
}


