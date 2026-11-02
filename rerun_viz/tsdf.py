"""Open3D TSDF integration over depth-anchored, color-registered RGB-D frames.

Two backends behind one interface:

- ``tensor``: ``o3d.t.geometry.VoxelBlockGrid`` (hash-mapped voxel blocks, CPU or CUDA)
- ``legacy``: ``o3d.pipelines.integration.ScalableTSDFVolume`` (CPU, OpenMP)

Open3D assumes an ideal pinhole camera with depth and color pixel-aligned, so
frames are prepared by undistorting the color image and registering depth into
that pinhole color camera (:func:`prepare_rgbd`). The integrated camera is
therefore the color optical frame.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .alignment import RGBDCalibration, align_depth_to_color
from .camera import CameraIntrinsics, Undistorter
from .config import ReconstructionConfig
from .transforms import RigidTransform


def prepare_rgbd(
    depth_m: np.ndarray,
    color: np.ndarray,
    calib: RGBDCalibration,
    undistort: Undistorter | None,
    min_m: float,
    max_m: float,
) -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    """Return ``(depth_in_color, color, intrinsics)`` with depth registered to the (pinhole) color camera."""
    if undistort is not None:
        color = undistort(color)
        calib = RGBDCalibration(calib.depth, undistort.target, calib.T_color_depth)
    aligned = align_depth_to_color(depth_m, color, calib, min_m, max_m)
    return aligned.depth_in_color, color, calib.color


def resolve_device(name: str) -> o3c.Device:
    device = o3c.Device(name)
    if device.get_type() == o3c.Device.DeviceType.CUDA and not o3c.cuda.is_available():
        warnings.warn(f"{name} requested but CUDA is unavailable; using CPU:0", stacklevel=2)
        return o3c.Device("CPU:0")
    return device


@functools.cache
def legacy_backend_works() -> bool:
    """Whether ``ScalableTSDFVolume`` integrates anything in this Open3D build.

    The Open3D 0.20.0 / CPython 3.14 wheel returns empty volumes even for the
    canonical tutorial inputs, while the tensor ``VoxelBlockGrid`` is fine.
    """
    depth = o3d.geometry.Image(np.full((48, 64), 1.513, np.float32))
    color = o3d.geometry.Image(np.full((48, 64, 3), 128, np.uint8))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, 1.0, 3.0, False)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.02, sdf_trunc=0.08, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    vol.integrate(rgbd, o3d.camera.PinholeCameraIntrinsic(64, 48, 50.0, 50.0, 32.0, 24.0), np.eye(4))
    return len(vol.extract_point_cloud().points) > 0


class TSDFVolume:
    """Uniform wrapper over the tensor and legacy Open3D TSDF volumes."""

    def __init__(self, cfg: ReconstructionConfig, intr: CameraIntrinsics, depth_max: float):
        if intr.is_distorted:
            raise ValueError("TSDF integration requires pinhole intrinsics; undistort first (see prepare_rgbd)")
        self.cfg = cfg
        self.depth_max = depth_max
        self.n_integrated = 0

        if cfg.backend == "tensor":
            self.device = resolve_device(cfg.device)
            self._K = o3c.Tensor(intr.K, o3c.float64)
            self._vbg = o3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight", "color"),
                attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
                attr_channels=((1), (1), (3)),
                voxel_size=cfg.voxel_size,
                block_resolution=cfg.block_resolution,
                block_count=cfg.block_count,
                device=self.device,
            )
        elif cfg.backend == "legacy":
            if not legacy_backend_works():
                raise RuntimeError(
                    f"ScalableTSDFVolume produces empty volumes in this Open3D build ({o3d.__version__}); "
                    "use backend: tensor (VoxelBlockGrid, CPU:0 or CUDA:0)"
                )
            self._intr = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
            self._volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=cfg.voxel_size,
                sdf_trunc=cfg.sdf_trunc,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            )
        else:
            raise ValueError(f"Unknown TSDF backend: {cfg.backend!r}")

    def integrate(self, depth_m: np.ndarray, color: np.ndarray, T_world_cam: RigidTransform) -> None:
        """Integrate metric depth (pixel-aligned with ``color``) seen from camera pose ``T_world_cam``."""
        extrinsic = T_world_cam.inverse().as_matrix()  # Open3D wants world -> camera
        depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)

        if self.cfg.backend == "tensor":
            depth = o3d.t.geometry.Image(o3c.Tensor(depth_m)).to(self.device)
            rgb = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color, dtype=np.float32) / 255.0)).to(self.device)
            E = o3c.Tensor(extrinsic, o3c.float64)
            blocks = self._vbg.compute_unique_block_coordinates(depth, self._K, E, 1.0, self.depth_max)
            self._vbg.integrate(
                blocks, depth, rgb, self._K, E, 1.0, self.depth_max, self.cfg.sdf_trunc / self.cfg.voxel_size
            )
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(color, dtype=np.uint8)),
                o3d.geometry.Image(depth_m),
                depth_scale=1.0,
                depth_trunc=self.depth_max,
                convert_rgb_to_intensity=False,
            )
            self._volume.integrate(rgbd, self._intr, extrinsic)
        self.n_integrated += 1

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        if self.cfg.backend == "tensor":
            mesh = self._vbg.extract_triangle_mesh(weight_threshold=self.cfg.weight_threshold).to_legacy()
        else:
            mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def extract_point_cloud(self) -> o3d.geometry.PointCloud:
        if self.cfg.backend == "tensor":
            return self._vbg.extract_point_cloud(weight_threshold=self.cfg.weight_threshold).to_legacy()
        return self._volume.extract_point_cloud()
