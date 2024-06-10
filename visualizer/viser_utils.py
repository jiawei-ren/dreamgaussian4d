import struct
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, Union
from pathlib import Path
import torch
import math
from collections import defaultdict
from pprint import pprint
from kornia import create_meshgrid

@dataclass(frozen=True)
class CameraModel:
    model_id: int
    model_name: str
    num_params: int


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class BaseImage:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass(frozen=True)
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: Union[float, np.ndarray]
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


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
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path: Union[str, Path]) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(
                    id=camera_id, model=model, width=width, height=height, params=params
                )
    return cameras


def read_cameras_binary(path_to_model_file: Union[str, Path]) -> Dict[int, Camera]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras


def read_images_text(path: Union[str, Path]) -> Dict[int, Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def read_images_binary(path_to_model_file: Union[str, Path]) -> Dict[int, Image]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            x_y_id_s = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_text(path: Union[str, Path]):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    image_ids=image_ids,
                    point2D_idxs=point2D_idxs,
                )
    return points3D


def read_points3d_binary(path_to_model_file: Union[str, Path]) -> Dict[int, Point3D]:
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd"
            )
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[
                0
            ]
            track_elems = read_next_bytes(
                fid,
                num_bytes=8 * track_length,
                format_char_sequence="ii" * track_length,
            )
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def q2r(qvec):
    # qvec B x 4
    qvec = qvec / qvec.norm(dim=1, keepdim=True)
    rot = [
        1 - 2 * qvec[:, 2] ** 2 - 2 * qvec[:, 3] ** 2,
        2 * qvec[:, 1] * qvec[:, 2] - 2 * qvec[:, 0] * qvec[:, 3],
        2 * qvec[:, 3] * qvec[:, 1] + 2 * qvec[:, 0] * qvec[:, 2],
        2 * qvec[:, 1] * qvec[:, 2] + 2 * qvec[:, 0] * qvec[:, 3],
        1 - 2 * qvec[:, 1] ** 2 - 2 * qvec[:, 3] ** 2,
        2 * qvec[:, 2] * qvec[:, 3] - 2 * qvec[:, 0] * qvec[:, 1],
        2 * qvec[:, 3] * qvec[:, 1] - 2 * qvec[:, 0] * qvec[:, 2],
        2 * qvec[:, 2] * qvec[:, 3] + 2 * qvec[:, 0] * qvec[:, 1],
        1 - 2 * qvec[:, 1] ** 2 - 2 * qvec[:, 2] ** 2,
    ]
    rot = torch.stack(rot, dim=1).reshape(-1, 3, 3)
    return rot

def jacobian_torch(a):
    _rsqr = 1./(a[:, 0]**2 + a[:, 1]**2 + a[:, 2]**2).sqrt()
    _res = [
        1/a[:,2], torch.zeros_like(a[:,0]), -a[:,0]/(a[:,2]**2),
        torch.zeros_like(a[:,0]), 1/a[:,2], -a[:,1]/(a[:,2]**2),
        _rsqr * a[:, 0], _rsqr * a[:, 1], _rsqr * a[:, 2]
    ]
    return torch.stack(_res, dim=-1).reshape(-1, 3, 3)


def initialize_sh(rgbs):
    sh_coeff = torch.zeros(rgbs.shape[0], 3, 9, device=rgbs.device, dtype=rgbs.dtype)
    sh_coeff[:, :, 0] = rgbs / 0.28209479177387814
    return sh_coeff.flatten(1)

def inverse_sigmoid(y=0.001):
    return -math.log(1/y  - 1)

def inverse_sigmoid_torch(y):
    return -torch.log(1/y  - 1)


class Timer:
    recorder = defaultdict(list)

    def __init__(self, des="", verbose=False, record=True, debug=True) -> None:
        self.des = des
        self.verbose = verbose
        self.record = record
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            return self
        self.start = time.time()
        self.start_cuda = torch.cuda.Event(enable_timing=True)
        self.end_cuda = torch.cuda.Event(enable_timing=True)
        self.start_cuda.record()
        return self

    def __exit__(self, *args):
        if not self.debug:
            return 
        self.end = time.time()
        self.end_cuda.record()
        torch.cuda.synchronize()
        self.interval = self.start_cuda.elapsed_time(self.end_cuda)/1000.
        if self.verbose:
            print(f"[cudasync]{self.des} consuming {self.interval:.8f}")
        if self.record:
            Timer.recorder[self.des].append(self.interval)

    @staticmethod
    def show_recorder():
        pprint({k: np.mean(v) for k, v in Timer.recorder.items()})

def sample_two_point(gaussian_pos, gaussian_cov):
    # gaussian_cov: (..., 3, 3)
    # gaussian_pos: (..., 3)
    # n_samples: (...)
    # return: (..., n_samples, 3)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(
        gaussian_pos,
        gaussian_cov,
    )
    p1 = dist.sample()
    p2 = dist.sample()
    return p1, p2

def clamp(x):
    return torch.clamp(x, min=0, max=1)


def get_rays_direction_in_camera_space(H, W, focal):
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5
    i, j = grid.unbind(-1)
    cent = [W/2, H/2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)
    return directions

def get_rays_direction(w2c_rot, H, W, focal):
    c2w = torch.inverse(w2c_rot)
    directions = get_rays_direction_in_camera_space(H, W, focal)
    rays_d = directions @ c2w[:3, :3].T  # (H, W, 3)
    return rays_d