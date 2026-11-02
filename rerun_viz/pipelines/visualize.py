"""Rerun visualization: synchronized RGB-D + mocap pose, overlays, and a live voxel map."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import rerun as rr

from ..alignment import align_depth_to_color, depth_overlay
from ..config import Config
from ..dataset import Rig, index_bags, stream_frames
from ..progress import NullReporter, Reporter
from ..reconstruction import VoxelMap
from ..ros_io import BagMessage, count_messages
from ..transforms import RigidTransform


def set_time_ns(t_ns: int) -> None:
    rr.set_time("log_time", timestamp=np.datetime64(t_ns, "ns"))


def log_transform(entity: str, T: RigidTransform, static: bool = False) -> None:
    rr.log(entity, rr.Transform3D(translation=T.translation, rotation=rr.Quaternion(xyzw=T.quat_xyzw)), static=static)


def log_imu(msg) -> None:
    a, w = msg.linear_acceleration, msg.angular_velocity
    for axis in "xyz":
        rr.log(f"plots/imu/accel/{axis}", rr.Scalars(getattr(a, axis)))
        rr.log(f"plots/imu/gyro/{axis}", rr.Scalars(getattr(w, axis)))


def log_joints(msg, extra_fields) -> None:
    joint_names = getattr(msg, "name", [f"joint_{i}" for i in range(16)])
    for i, name in enumerate(joint_names):
        for field in ("position", "velocity", "effort", *extra_fields):
            arr = getattr(msg, field, None)
            if arr is not None and i < len(arr):
                rr.log(f"plots/joints/{name}/{field}", rr.Scalars(arr[i]))


def run_visualization(cfg: Config, reporter: Reporter | None = None, spawn: bool = True) -> dict[str, int]:
    """Stream the configured bags into a Rerun viewer. Returns frame / map statistics."""
    reporter = reporter or NullReporter()
    bags = [Path(p) for p in cfg.bags]
    topics, depth_cfg, agg_cfg, overlay_cfg = cfg.topics, cfg.depth, cfg.aggregation, cfg.overlay

    rr.init("rover_mcap_visualizer", spawn=spawn)
    if cfg.viewer.blueprint and Path(cfg.viewer.blueprint).exists():
        rr.log_file_from_path(cfg.viewer.blueprint)

    link_entity = cfg.shim.child
    T_rover_link = RigidTransform.from_quat(cfg.shim.translation, cfg.shim.rotation_xyzw)
    log_transform(link_entity, T_rover_link, static=True)

    def log_plot(m: BagMessage) -> None:
        set_time_ns(m.stamp_ns)
        if m.topic == topics.imu:
            log_imu(m.msg)
        elif m.topic == topics.joints:
            log_joints(m.msg, cfg.plots.joints.extra_fields)

    plot_topics = [topics.imu, topics.joints]
    all_topics = [*plot_topics, topics.pose, topics.color, topics.color_info, topics.depth, topics.depth_info, topics.extrinsics]
    task = reporter.task("Indexing bags + logging plots", count_messages(bags, {t for t in all_topics if t}))
    index = index_bags(bags, cfg, plot_topics, on_message=log_plot, on_progress=task.advance)
    task.finish()
    s = index.stats
    reporter.log(
        f"{s['depth']} depth / {s['color']} color frames, {s['poses']} poses -> {s['synced']} synchronized "
        f"(dropped: {s['dropped_no_color']} no color match, {s['dropped_no_pose']} no pose)"
    )

    rig = Rig.from_config(cfg, index.calib) if index.calib else None
    if rig is None:
        reporter.log("WARNING: missing camera_info/extrinsics; skipping overlay and point clouds.")
    else:
        color = rig.calib.color
        log_transform(f"{link_entity}/color", rig.T_link_color, static=True)
        rr.log(
            f"{link_entity}/color",
            rr.Pinhole(image_from_camera=color.K, width=color.width, height=color.height),
            static=True,
        )

    voxel_map = VoxelMap(agg_cfg.voxel_size, agg_cfg.max_points, agg_cfg.keep)
    cmap = getattr(cv2, overlay_cfg.colormap, cv2.COLORMAP_JET)
    frames = index.frames[:: max(1, cfg.viewer.downsample)]

    task = reporter.task("Streaming synchronized triplets", len(frames))
    n_logged = 0
    for frame in stream_frames(bags, cfg, frames, on_progress=task.advance):
        set_time_ns(frame.plan.t_ns)
        T_world_rover = frame.plan.T_world_rover
        log_transform(cfg.shim.parent, T_world_rover)
        rr.log(f"{link_entity}/color", rr.Image(frame.color))
        n_logged += 1
        if rig is None:
            continue

        aligned = align_depth_to_color(
            frame.depth_m,
            frame.color,
            rig.calib,
            depth_cfg.min_m,
            depth_cfg.max_m,
            stride=agg_cfg.stride,
            occlusion_tol_m=agg_cfg.occlusion_tol_m,
        )

        if overlay_cfg.enabled:
            overlay = depth_overlay(
                frame.color, aligned.depth_in_color, depth_cfg.min_m, depth_cfg.max_m, overlay_cfg.alpha, cmap
            )
            rr.log(f"{link_entity}/color_depth_overlay", rr.Image(overlay))

        if len(aligned.colors) == 0:
            continue
        pts_link = rig.T_link_depth.apply(aligned.points_depth)
        rr.log(f"{link_entity}/point_cloud", rr.Points3D(positions=pts_link, colors=aligned.colors))

        if agg_cfg.enabled:
            voxel_map.insert((T_world_rover @ T_rover_link).apply(pts_link), aligned.colors)
            rr.log("world/map/accumulated_cloud", rr.Points3D(positions=voxel_map.points, colors=voxel_map.colors))
    task.finish()

    reporter.log(f"Logged {n_logged} synchronized triplets; map has {len(voxel_map)} voxels.")
    return {**index.stats, "logged": n_logged, "map_voxels": len(voxel_map)}
