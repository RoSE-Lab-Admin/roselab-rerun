"""RGB-D + mocap pose alignment, interpolation, voxel / TSDF reconstruction and Rerun visualization.

Core modules are viewer-agnostic (no rerun imports) so they can be reused elsewhere:

- ``camera``         pinhole intrinsics, Brown-Conrady projection / deprojection
- ``transforms``     ``RigidTransform`` (``T_parent_child``, xyzw quaternions)
- ``interpolation``  int-ns timestamp matching, ``PoseTimeline`` (slerp / nearest)
- ``alignment``      depth-anchored depth->color registration, point coloring, overlays
- ``reconstruction`` voxel-grid downsampling and ``VoxelMap`` aggregation
- ``ros_io``         rosbags streaming, header stamps, compressed image/depth decoding
- ``dataset``        two-pass bag indexing / frame planning and bounded-memory frame streaming, ``Rig``
- ``tsdf``           Open3D TSDF volumes (tensor ``VoxelBlockGrid`` / legacy ``ScalableTSDFVolume``)
- ``config``         typed YAML config

``pipelines.visualize`` (Rerun) and ``pipelines.reconstruct`` (headless, chunked) build on these;
``cli`` / ``tui`` provide the ``rerun-viz`` command and its config form.
"""

from .alignment import AlignedRGBD, RGBDCalibration, align_depth_to_color, depth_overlay, depth_to_points
from .camera import CameraIntrinsics
from .interpolation import PoseTimeline, nearest_index, sec_to_ns
from .reconstruction import VoxelMap, voxel_downsample
from .transforms import RigidTransform

__all__ = [
    "AlignedRGBD",
    "CameraIntrinsics",
    "PoseTimeline",
    "RGBDCalibration",
    "RigidTransform",
    "VoxelMap",
    "align_depth_to_color",
    "depth_overlay",
    "depth_to_points",
    "nearest_index",
    "sec_to_ns",
    "voxel_downsample",
]
