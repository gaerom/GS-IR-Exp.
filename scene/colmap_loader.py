#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact george.drettakis@inria.fr
#

# modified
import collections
import struct
import numpy as np
from io import TextIOWrapper
from typing import Any, Dict, Tuple

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}

CAMERA_MODEL_IDS = {camera_model.model_id: camera_model for camera_model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {camera_model.model_name: camera_model for camera_model in CAMERA_MODELS}

def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]
    ])

def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self) -> np.ndarray:
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid: TextIOWrapper, num_bytes: int, format_char_sequence: str, endian_character: str = "<") -> Any:
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyzs, rgbs, errors = [], [], []
    with open(path, "r") as fid:
        for line in fid:
            if line and line[0] != "#":
                elems = line.strip().split()
                xyzs.append(list(map(float, elems[1:4])))
                rgbs.append(list(map(int, elems[4:7])))
                errors.append(float(elems[7]))
    return np.array(xyzs), np.array(rgbs), np.array(errors).reshape(-1, 1)

def read_points3D_binary(path_to_model_file: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        xyzs, rgbs, errors = [], [], []
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            xyzs.append(binary_point_line_properties[1:4])
            rgbs.append(binary_point_line_properties[4:7])
            errors.append(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, 8 * track_length, "ii" * track_length)
    return np.array(xyzs), np.array(rgbs), np.array(errors).reshape(-1, 1)

def read_intrinsics_text(path: str) -> Dict:
    cameras = {}
    with open(path, "r") as fid:
        for line in fid:
            if line and line[0] != "#":
                elems = line.strip().split()
                camera_id, model = int(elems[0]), elems[1]
                assert model == "PINHOLE", "Loader assumes PINHOLE model only."
                width, height = int(elems[2]), int(elems[3])
                params = np.array(list(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model, width=width, height=height, params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file: str) -> Dict:
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id, qvec, tvec, camera_id = binary_image_properties[0], binary_image_properties[1:5], binary_image_properties[5:8], binary_image_properties[8]
            image_name, current_char = "", read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
            xys = np.column_stack([x_y_id_s[0::3], x_y_id_s[1::3]])
            point3D_ids = np.array(x_y_id_s[2::3])
            images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_intrinsics_binary(path_to_model_file: str) -> Dict:
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id, model_id = camera_properties[0], camera_properties[1]
            model_name, width, height = CAMERA_MODEL_IDS[model_id].model_name, camera_properties[2], camera_properties[3]
            params = np.array(read_next_bytes(fid, 8 * CAMERA_MODEL_IDS[model_id].num_params, "d" * CAMERA_MODEL_IDS[model_id].num_params))
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=params)
    return cameras

def read_extrinsics_text(path: str) -> Dict:
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline().strip()
            if not line:
                break
            if line and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(list(map(float, elems[1:5])))
                tvec = np.array(list(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().strip().split()
                xys = np.column_stack([list(map(float, elems[0::3])), list(map(float, elems[1::3]))])
                point3D_ids = np.array(list(map(int, elems[2::3])))
                images[image_id] = Image(id=image_id, qvec=qvec, tvec=tvec, camera_id=camera_id, name=image_name, xys=xys, point3D_ids=point3D_ids)
    return images

def read_colmap_bin_array(path: str) -> np.ndarray:
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1, usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter, byte = 0, fid.read(1)
        while num_delimiter < 3:
            if byte == b"&":
                num_delimiter += 1
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
