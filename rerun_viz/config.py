"""Typed configuration, loaded from / saved to the YAML layout of ``config.yaml``.

Every field carries a ``help`` string in its metadata; the TUI config form is
generated from these dataclasses, so adding a field here is all it takes to
expose it on the command line form.
"""

from __future__ import annotations

import dataclasses
import types
import typing
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


def opt(default: Any = None, help: str = "", **kw):  # noqa: A002
    """Dataclass field with a help string (mutable defaults are copied per instance)."""
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda d=default: type(d)(d), metadata={"help": help}, **kw)
    return field(default=default, metadata={"help": help}, **kw)


@dataclass
class ViewerConfig:
    blueprint: str | None = opt("./rover_mcap_visualizer.rbl", "Rerun blueprint (.rbl) to load, if it exists")
    downsample: int = opt(32, "Process every Nth synchronized triplet (memory / speed)")
    spatial_type: str = opt("3D", "Whether to use proj (orbiting 3D) or ortho (2D locked)")


@dataclass
class TopicsConfig:
    pose: str = opt("/FitRosey_V1/pose", "geometry_msgs/PoseStamped rover pose (mocap)")
    imu: str | None = opt("/imu/data", "sensor_msgs/Imu")
    color: str = opt("/MastCam/Front/color/image_raw/compressed", "sensor_msgs/CompressedImage")
    color_info: str = opt("/MastCam/Front/color/camera_info", "sensor_msgs/CameraInfo")
    depth: str = opt("/MastCam/Front/depth/image_rect_raw/compressedDepth", "compressedDepth CompressedImage")
    depth_info: str = opt("/MastCam/Front/depth/camera_info", "sensor_msgs/CameraInfo")
    extrinsics: str = opt("/MastCam/Front/extrinsics/depth_to_color", "realsense2_camera_msgs/Extrinsics")
    joints: str | None = opt("/joint_states", "sensor_msgs/JointState")


@dataclass
class TemporalAlignmentConfig:
    method: Literal["slerp", "nearest"] = opt("slerp", "Pose lookup: linear pos + SLERP rot, or nearest sample")
    max_delta_sec: float = opt(0.05, "Max depth<->color (and 'nearest' pose) timestamp offset")
    max_pose_gap_sec: float | None = opt(None, "'slerp' only: max bracketing pose spacing (default 2x max_delta)")


@dataclass
class DepthConfig:
    scale: float = opt(0.001, "16UC1 depth units -> meters (RealSense default: mm)")
    min_m: float = opt(0.2, "Minimum valid depth (m)")
    max_m: float = opt(5.0, "Maximum valid depth (m)")


@dataclass
class OverlayConfig:
    enabled: bool = opt(True, "Log a colormapped depth-on-color overlay image")
    alpha: float = opt(0.4, "Opacity of depth overlay (0.0 to 1.0)")
    colormap: str = opt("COLORMAP_JET", "OpenCV colormap name (COLORMAP_JET, COLORMAP_TURBO, ...)")


@dataclass
class JointsPlotConfig:
    extra_fields: list[str] = opt([], "Extra per-joint array fields to plot (e.g. motor_currents)")


@dataclass
class PlotsConfig:
    joints: JointsPlotConfig = field(default_factory=JointsPlotConfig)


@dataclass
class ShimConfig:
    parent: str = opt("world/rover_base", "Rerun entity of the mocap-tracked rover base")
    child: str = opt("world/rover_base/mastcam_front", "Rerun entity of the MastCam camera_link")
    translation: list[float] = opt([0.25, 0.0, 0.235], "rover_base -> camera_link translation [x, y, z] (m)")
    rotation_xyzw: list[float] = opt(
        [0.541675, 0.541675, 0.454519, 0.454519], "rover_base -> camera_link rotation quaternion [x, y, z, w]"
    )


@dataclass
class AggregationConfig:
    enabled: bool = opt(True, "Accumulate a voxel-downsampled world point map in the viewer")
    stride: int = opt(1, "Pixel downsample factor (both image axes) for point cloud projection")
    voxel_size: float = opt(0.05, "Map voxel size (m)")
    keep: Literal["first", "latest"] = opt("first", "Per-voxel point/color: first observation or latest")
    max_points: int = opt(500000, "Max voxels in the map; least-recently-observed are evicted first")
    occlusion_tol_m: float | None = opt(None, "Drop depth points hidden from the color camera (m)")


@dataclass
class ReconstructionConfig:
    backend: Literal["tensor", "legacy"] = opt(
        "tensor", "tensor: VoxelBlockGrid (CPU/CUDA); legacy: ScalableTSDFVolume (CPU)"
    )
    device: str = opt("CPU:0", "Open3D device for the tensor backend (CPU:0, or CUDA:0 where available)")
    voxel_size: float = opt(0.01, "TSDF voxel size (m)")
    sdf_trunc: float = opt(0.04, "TSDF truncation distance (m)")
    block_resolution: int = opt(16, "tensor: voxels per block edge")
    block_count: int = opt(
        10000, "tensor: initial voxel-block capacity; grows on demand (~res^3 x 20 B each: 82 KB at 16^3)"
    )
    weight_threshold: float = opt(3.0, "tensor: min integration weight for extracted surface")
    frame_stride: int = opt(1, "Integrate every Nth synchronized frame")
    chunk_size: int = opt(256, "Frames per chunk; each chunk is a separate volume flushed to disk (0 = one volume)")
    workers: int = opt(2, "Parallel chunk worker processes")
    undistort_color: bool = opt(True, "Undistort color so the TSDF sees an ideal pinhole camera")
    output_dir: str = opt("recon_out", "Directory for chunk files and merged outputs")
    save_mesh: bool = opt(True, "Extract and merge triangle meshes")
    save_point_cloud: bool = opt(True, "Extract and merge point clouds")
    merge_voxel_size: float | None = opt(None, "Voxel size for downsampling the merged cloud (default: voxel_size)")
    keep_chunks: bool = opt(True, "Keep per-chunk files after merging")


@dataclass
class Config:
    bags: list[str] = opt(["../data/perception/091726/merged/merged_bag.mcap_0.mcap"], "Bag files/directories")
    viewer: ViewerConfig = field(default_factory=ViewerConfig)
    topics: TopicsConfig = field(default_factory=TopicsConfig)
    temporal_alignment: TemporalAlignmentConfig = field(default_factory=TemporalAlignmentConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    overlay: OverlayConfig = field(default_factory=OverlayConfig)
    plots: PlotsConfig = field(default_factory=PlotsConfig)
    shim: ShimConfig = field(default_factory=ShimConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    recon: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        with open(path) as f:
            return from_dict(cls, yaml.safe_load(f) or {})

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


def from_dict(cls, data: dict, prefix: str = ""):
    """Build dataclass ``cls`` from a (partial) dict; unknown keys warn, missing keys take defaults."""
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for key, value in data.items():
        if key not in hints:
            warnings.warn(f"Unknown config key '{prefix}{key}' ignored", stacklevel=2)
            continue
        hint = hints[key]
        if dataclasses.is_dataclass(hint):
            kwargs[key] = from_dict(hint, value or {}, f"{prefix}{key}.")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def unwrap_optional(hint) -> tuple[Any, bool]:
    """``X | None`` -> ``(X, True)``; anything else -> ``(hint, False)``."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return hint, False
