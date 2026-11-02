"""Writes a small synthetic ROS 2 (MCAP) bag with known geometry, for end-to-end tests.

Scene: textured ground plane (z = 0) and wall (x = WALL_X) in the world frame. A
rover drives along +y (with a little yaw) carrying the MastCam on the configured
shim; depth and color are ray-cast from the depth and (distorted) color cameras.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag2 import StoragePlugin, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from scipy.spatial.transform import Rotation

from rerun_viz.camera import CameraIntrinsics
from rerun_viz.config import Config
from rerun_viz.dataset import T_LINK_DEPTH
from rerun_viz.transforms import RigidTransform

WALL_X = 2.0
T0 = 1_700_000_000_000_000_000
DEPTH_INTR = CameraIntrinsics(160, 120, 120.0, 120.0, 80.0, 60.0)
COLOR_INTR = CameraIntrinsics(160, 120, 125.0, 124.0, 81.0, 59.0, np.array([-0.05, 0.06, 0.0006, 0.0005, -0.02]))
T_COLOR_DEPTH = RigidTransform(Rotation.from_euler("xyz", [0.3, -0.2, 0.1], degrees=True), np.array([-0.03, 0.0002, 0.0004]))


def rover_pose(t_sec: float) -> RigidTransform:
    return RigidTransform(Rotation.from_euler("z", 5.0 * np.sin(t_sec), degrees=True), np.array([0.0, 0.3 * t_sec, 0.0]))


def texture(points: np.ndarray) -> np.ndarray:
    """Checkerboard color (25 cm squares) on the wall (y, z) and ground (x, y)."""
    on_wall = np.abs(points[:, 0] - WALL_X) < 1e-6
    a = np.where(on_wall, points[:, 1], points[:, 0])
    b = np.where(on_wall, points[:, 2], points[:, 1])
    checker = (np.floor(a / 0.25) + np.floor(b / 0.25)).astype(int) % 2
    base = np.where(on_wall[:, None], [[200, 60, 40]], [[40, 160, 220]])
    return np.where(checker[:, None] == 1, base, base // 3).astype(np.uint8)


def raycast(intr: CameraIntrinsics, T_world_cam: RigidTransform) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel camera z-depth (0 = miss) and world hit points."""
    v, u = np.mgrid[0 : intr.height, 0 : intr.width]
    rays = intr.deproject(u.ravel(), v.ravel(), np.ones(u.size))  # z = 1 rays (handles distortion)
    d = T_world_cam.rotation.apply(rays)
    o = T_world_cam.translation
    with np.errstate(divide="ignore", invalid="ignore"):
        t_wall = np.where(d[:, 0] > 1e-9, (WALL_X - o[0]) / d[:, 0], np.inf)
        t_ground = np.where(d[:, 2] < -1e-9, -o[2] / d[:, 2], np.inf)
    t = np.minimum(t_wall, t_ground)
    hit = o + t[:, None] * d
    hit[np.isinf(t)] = 0
    hit[t_wall <= t_ground, 0] = WALL_X  # snap for exact texture lookup
    z = np.where(np.isfinite(t), t, 0.0)
    return z.reshape(intr.height, intr.width), hit.reshape(intr.height, intr.width, 3)


def _typestore():
    ts = get_typestore(Stores.ROS2_JAZZY)
    ts.register(get_types_from_msg("float64[9] rotation\nfloat64[3] translation", "realsense2_camera_msgs/msg/Extrinsics"))
    return ts


def write_bag(path: Path, cfg: Config, duration_s: float = 3.0, fps: float = 30.0, pose_hz: float = 120.0) -> Path:
    ts = _typestore()
    T = ts.types
    Time, Header = T["builtin_interfaces/msg/Time"], T["std_msgs/msg/Header"]

    def header(t_ns: int):
        return Header(stamp=Time(sec=t_ns // 1_000_000_000, nanosec=t_ns % 1_000_000_000), frame_id="")

    def cam_info(t_ns: int, intr: CameraIntrinsics):
        K = intr.K.ravel()
        P = np.hstack([intr.K, np.zeros((3, 1))]).ravel()
        roi = T["sensor_msgs/msg/RegionOfInterest"](x_offset=0, y_offset=0, height=0, width=0, do_rectify=False)
        return T["sensor_msgs/msg/CameraInfo"](
            header=header(t_ns), height=intr.height, width=intr.width, distortion_model="plumb_bob",
            d=intr.dist.astype(np.float64), k=K, r=np.eye(3).ravel(), p=P, binning_x=0, binning_y=0, roi=roi,
        )  # fmt: skip

    def vec3(x, y, z):
        return T["geometry_msgs/msg/Vector3"](x=float(x), y=float(y), z=float(z))

    T_rover_depth = RigidTransform.from_quat(cfg.shim.translation, cfg.shim.rotation_xyzw) @ T_LINK_DEPTH
    T_rover_color = T_rover_depth @ T_COLOR_DEPTH.inverse()
    tp = cfg.topics
    msgs = []  # (log_ns, topic, msgtype, msg)

    msgs.append((T0, tp.extrinsics, "realsense2_camera_msgs/msg/Extrinsics",
                 T["realsense2_camera_msgs/msg/Extrinsics"](
                     rotation=T_COLOR_DEPTH.rotation.as_matrix().ravel(order="F"),  # librealsense column-major
                     translation=T_COLOR_DEPTH.translation)))  # fmt: skip

    for k in range(int(duration_s * pose_hz) + 1):
        t_ns = T0 + int(k * 1e9 / pose_hz)
        P = rover_pose((t_ns - T0) / 1e9)
        q = P.quat_xyzw
        pose = T["geometry_msgs/msg/Pose"](
            position=T["geometry_msgs/msg/Point"](x=P.translation[0], y=P.translation[1], z=P.translation[2]),
            orientation=T["geometry_msgs/msg/Quaternion"](x=q[0], y=q[1], z=q[2], w=q[3]),
        )
        msgs.append((t_ns + 300_000, tp.pose, "geometry_msgs/msg/PoseStamped",
                     T["geometry_msgs/msg/PoseStamped"](header=header(t_ns), pose=pose)))  # fmt: skip

    for k in range(int(duration_s * fps)):
        t_ns = T0 + 20_000_000 + int(k * 1e9 / fps)
        T_world_rover = rover_pose((t_ns - T0) / 1e9)

        depth_z, _ = raycast(DEPTH_INTR, T_world_rover @ T_rover_depth)
        depth_mm = np.clip(np.round(depth_z * 1000), 0, 65535).astype(np.uint16)
        depth_mm[depth_z > 6.0] = 0
        png = cv2.imencode(".png", depth_mm)[1].tobytes()
        depth_msg = T["sensor_msgs/msg/CompressedImage"](
            header=header(t_ns), format="16UC1; compressedDepth png", data=np.frombuffer(b"\0" * 12 + png, np.uint8)
        )

        _, hits = raycast(COLOR_INTR, T_world_rover @ T_rover_color)
        rgb = texture(hits.reshape(-1, 3)).reshape(COLOR_INTR.height, COLOR_INTR.width, 3)
        jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 98])[1]
        t_color = t_ns + 40_000  # hardware sync offset
        color_msg = T["sensor_msgs/msg/CompressedImage"](
            header=header(t_color), format="rgb8; jpeg compressed bgr8", data=jpg.ravel()
        )

        # Latency: color logged before depth for odd frames, to exercise out-of-order pairing.
        lag_d, lag_c = (9_000_000, 4_000_000) if k % 2 else (4_000_000, 9_000_000)
        msgs.append((t_ns + lag_d, tp.depth, "sensor_msgs/msg/CompressedImage", depth_msg))
        msgs.append((t_color + lag_c, tp.color, "sensor_msgs/msg/CompressedImage", color_msg))
        msgs.append((t_ns + 1_000_000, tp.depth_info, "sensor_msgs/msg/CameraInfo", cam_info(t_ns, DEPTH_INTR)))
        msgs.append((t_color + 1_000_000, tp.color_info, "sensor_msgs/msg/CameraInfo", cam_info(t_color, COLOR_INTR)))

        imu = T["sensor_msgs/msg/Imu"](
            header=header(t_ns), orientation=T["geometry_msgs/msg/Quaternion"](x=0.0, y=0.0, z=0.0, w=1.0),
            orientation_covariance=np.zeros(9), angular_velocity=vec3(0, 0, 0.1), angular_velocity_covariance=np.zeros(9),
            linear_acceleration=vec3(0, 0, 9.81), linear_acceleration_covariance=np.zeros(9),
        )  # fmt: skip
        msgs.append((t_ns + 500_000, tp.imu, "sensor_msgs/msg/Imu", imu))
        joints = T["sensor_msgs/msg/JointState"](
            header=header(t_ns), name=["wheel_l", "wheel_r"], position=np.array([0.1 * k, 0.1 * k]),
            velocity=np.array([1.0, 1.0]), effort=np.zeros(0),
        )  # fmt: skip
        msgs.append((t_ns + 600_000, tp.joints, "sensor_msgs/msg/JointState", joints))

    msgs.sort(key=lambda m: m[0])
    with Writer(path, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        conns = {}
        for log_ns, topic, msgtype, msg in msgs:
            if topic not in conns:
                conns[topic] = writer.add_connection(topic, msgtype, typestore=ts)
            writer.write(conns[topic], log_ns, ts.serialize_cdr(msg, msgtype))
    return path
