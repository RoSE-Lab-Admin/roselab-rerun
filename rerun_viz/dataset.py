"""Two-pass RGB-D + pose dataset over ROS 2 bags.

Pass 1 (:func:`index_bags`) reads every relevant message once, keeping only
timestamps, poses and calibration, and plans the synchronized frames: each depth
image (the anchor) is matched to its nearest color image and an interpolated
rover pose. Pass 2 (:func:`stream_frames`) re-reads only the bag log-time window
covering a set of planned frames and yields decoded images, holding at most the
few messages still waiting for their partner. Memory therefore stays bounded no
matter how long the bag is, and parallel workers can each stream their own slice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .alignment import RGBDCalibration
from .camera import CameraIntrinsics
from .config import Config
from .interpolation import PoseTimeline, nearest_index, sec_to_ns
from .ros_io import BagMessage, decode_compressed_depth, decode_compressed_image, iter_messages
from .transforms import RigidTransform

# Depth optical frame in the MastCam camera_link (the shim's child). The stock RealSense tree puts the
# depth optical frame at the camera_link origin, rotated x-forward -> z-forward (xyzw [-0.5, 0.5, -0.5, 0.5]).
# Identity reproduces the original behavior, where depth-optical points were logged directly under the shim.
T_LINK_DEPTH = RigidTransform.identity()


@dataclass(frozen=True)
class Rig:
    """Static transform chain rover_base -> camera_link -> depth / color optical frames."""

    calib: RGBDCalibration
    T_rover_link: RigidTransform
    T_link_depth: RigidTransform = T_LINK_DEPTH

    @classmethod
    def from_config(cls, cfg: Config, calib: RGBDCalibration) -> Rig:
        return cls(calib, RigidTransform.from_quat(cfg.shim.translation, cfg.shim.rotation_xyzw))

    @property
    def T_link_color(self) -> RigidTransform:
        return self.T_link_depth @ self.calib.T_color_depth.inverse()

    @property
    def T_rover_depth(self) -> RigidTransform:
        return self.T_rover_link @ self.T_link_depth

    @property
    def T_rover_color(self) -> RigidTransform:
        return self.T_rover_link @ self.T_link_color


@dataclass(frozen=True)
class PlannedFrame:
    t_ns: int  # depth header stamp (the anchor)
    depth_log_ns: int
    color_log_ns: int
    T_world_rover: RigidTransform


@dataclass
class Frame:
    plan: PlannedFrame
    color: np.ndarray  # (H, W, 3) uint8 RGB
    depth_m: np.ndarray  # (H, W) float32 meters, 0 = invalid


@dataclass
class BagIndex:
    calib: RGBDCalibration | None
    poses: PoseTimeline
    frames: list[PlannedFrame]  # sorted by t_ns
    stats: dict[str, int] = field(default_factory=dict)


def _calibration(color_info, depth_info, extrinsics_msg) -> RGBDCalibration | None:
    if not (color_info and depth_info and extrinsics_msg):
        return None
    return RGBDCalibration(
        depth=CameraIntrinsics.from_camera_info(depth_info),
        color=CameraIntrinsics.from_camera_info(color_info),
        T_color_depth=RigidTransform.from_extrinsics_msg(extrinsics_msg),
    )


def index_bags(
    bags: Sequence[Path],
    cfg: Config,
    extra_topics: Iterable[str] = (),
    on_message: Callable[[BagMessage], None] | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> BagIndex:
    """Pass 1: collect calibration and poses, and plan synchronized depth/color/pose frames.

    Messages on ``extra_topics`` are handed to ``on_message`` (e.g. for plotting);
    ``on_progress(1)`` is called once per message read.
    """
    t = cfg.topics
    extra = {x for x in extra_topics if x}
    topics = {t.pose, t.color, t.color_info, t.depth, t.depth_info, t.extrinsics} | extra

    poses: list[tuple[int, RigidTransform]] = []
    color: list[tuple[int, int]] = []  # (stamp, log)
    depth: list[tuple[int, int]] = []
    color_info = depth_info = extrinsics_msg = None

    for m in iter_messages(bags, topics):
        if on_progress:
            on_progress(1)
        if m.topic == t.depth:
            depth.append((m.stamp_ns, m.log_ns))
        elif m.topic == t.color:
            color.append((m.stamp_ns, m.log_ns))
        elif m.topic == t.pose:
            poses.append((m.stamp_ns, RigidTransform.from_pose_msg(m.msg)))
        # Calibration is assumed static; keep the first message of each.
        elif m.topic == t.color_info:
            color_info = color_info or m.msg
        elif m.topic == t.depth_info:
            depth_info = depth_info or m.msg
        elif m.topic == t.extrinsics:
            extrinsics_msg = extrinsics_msg or m.msg
        if m.topic in extra and on_message:
            on_message(m)

    # Bag order need not match header-stamp order (e.g. merged bags), so sort before matching.
    color.sort()
    depth.sort()
    color_ts = np.array([s for s, _ in color], dtype=np.int64)
    timeline = PoseTimeline.from_samples(poses)

    sync = cfg.temporal_alignment
    max_delta_ns = sec_to_ns(sync.max_delta_sec)
    max_gap_ns = sec_to_ns(sync.max_pose_gap_sec) if sync.max_pose_gap_sec is not None else None

    frames, no_color, no_pose = [], 0, 0
    for t_depth, log_depth in depth:
        c = nearest_index(color_ts, t_depth, max_delta_ns)
        if c is None:
            no_color += 1
            continue
        T_world_rover = timeline.pose_at(t_depth, sync.method, max_delta_ns, max_gap_ns)
        if T_world_rover is None:
            no_pose += 1
            continue
        frames.append(PlannedFrame(t_depth, log_depth, color[c][1], T_world_rover))

    stats = {
        "depth": len(depth),
        "color": len(color),
        "poses": len(timeline),
        "synced": len(frames),
        "dropped_no_color": no_color,
        "dropped_no_pose": no_pose,
    }
    return BagIndex(_calibration(color_info, depth_info, extrinsics_msg), timeline, frames, stats)


def stream_frames(
    bags: Sequence[Path],
    cfg: Config,
    frames: Sequence[PlannedFrame],
    on_progress: Callable[[int], None] | None = None,
) -> Iterator[Frame]:
    """Pass 2: yield decoded :class:`Frame` s for ``frames`` (in bag order, not necessarily time order).

    Only the bag log-time window spanning ``frames`` is read; ``on_progress(1)``
    is called per yielded frame.
    """
    if not frames:
        return
    t = cfg.topics
    by_depth = {f.depth_log_ns: f for f in frames}
    users = defaultdict(list)  # color log time -> planned frames that use it
    for f in frames:
        users[f.color_log_ns].append(f)
    start = min(min(f.depth_log_ns, f.color_log_ns) for f in frames)
    stop = max(max(f.depth_log_ns, f.color_log_ns) for f in frames) + 1

    depth_msgs: dict[int, object] = {}  # depth log time -> msg, waiting for color
    color_msgs: dict[int, object] = {}  # color log time -> msg, waiting for depth
    remaining = {k: len(v) for k, v in users.items()}

    def emit(f: PlannedFrame, depth_msg, color_msg) -> Frame:
        remaining[f.color_log_ns] -= 1
        if remaining[f.color_log_ns] == 0:
            color_msgs.pop(f.color_log_ns, None)
        if on_progress:
            on_progress(1)
        return Frame(
            f,
            decode_compressed_image(color_msg),
            decode_compressed_depth(depth_msg, cfg.depth.scale),
        )

    for m in iter_messages(bags, {t.depth, t.color}, start, stop):
        if m.topic == t.depth and (f := by_depth.get(m.log_ns)) is not None:
            if (c := color_msgs.get(f.color_log_ns)) is not None:
                yield emit(f, m.msg, c)
            else:
                depth_msgs[m.log_ns] = m.msg
        elif m.topic == t.color and m.log_ns in users:
            color_msgs[m.log_ns] = m.msg
            for f in users[m.log_ns]:
                if (d := depth_msgs.pop(f.depth_log_ns, None)) is not None:
                    yield emit(f, d, m.msg)
