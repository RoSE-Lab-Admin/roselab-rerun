"""Synthetic-data tests for projection, alignment, interpolation and reconstruction."""

from types import SimpleNamespace

import cv2
import numpy as np
import pyrealsense2 as rs
import pytest
from scipy.spatial.transform import Rotation

from rerun_viz.alignment import RGBDCalibration, align_depth_to_color, depth_overlay, depth_to_points
from rerun_viz.camera import CameraIntrinsics
from rerun_viz.interpolation import PoseTimeline, nearest_index, sec_to_ns
from rerun_viz.reconstruction import VoxelMap, voxel_downsample, voxel_keys
from rerun_viz.ros_io import decode_compressed_depth, stamp_ns
from rerun_viz.transforms import RigidTransform

rng = np.random.default_rng(0)

COLOR = CameraIntrinsics(640, 480, 610.0, 612.0, 321.5, 238.2, np.array([-0.05, 0.07, 0.001, 0.0005, -0.02]))
DEPTH = CameraIntrinsics(640, 480, 390.0, 390.0, 318.0, 241.0)


def to_rs_intrinsics(intr: CameraIntrinsics) -> rs.intrinsics:
    out = rs.intrinsics()
    out.width, out.height = intr.width, intr.height
    out.fx, out.fy, out.ppx, out.ppy = intr.fx, intr.fy, intr.cx, intr.cy
    out.model = rs.distortion.brown_conrady
    out.coeffs = [float(c) for c in intr.dist]
    return out


def random_points(n=500):
    return np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-0.7, 0.7, n), rng.uniform(0.5, 4.0, n)])


# --- camera ---


def test_project_matches_opencv_and_realsense():
    pts = random_points()
    u, v = COLOR.project(pts)
    cv_px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), COLOR.K, COLOR.dist)
    np.testing.assert_allclose(np.column_stack([u, v]), cv_px.reshape(-1, 2), atol=1e-6)

    rs_intr = to_rs_intrinsics(COLOR)
    rs_px = np.array([rs.rs2_project_point_to_pixel(rs_intr, p.tolist()) for p in pts])
    np.testing.assert_allclose(np.column_stack([u, v]), rs_px, atol=1e-2)  # rs is float32


@pytest.mark.parametrize("intr", [DEPTH, COLOR])
def test_deproject_roundtrip(intr):
    pts = random_points(2000)
    u, v = intr.project(pts)
    # The polynomial distortion model is only invertible over the imaged field of view.
    inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    pts, u, v = pts[inside], u[inside], v[inside]
    assert len(pts) > 200
    np.testing.assert_allclose(intr.deproject(u, v, pts[:, 2]), pts, atol=1e-9)


def test_project_behind_camera_is_nan():
    u, v = DEPTH.project(np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]))
    assert np.isnan(u).all() and np.isnan(v).all()


def test_from_camera_info():
    msg = SimpleNamespace(width=640, height=480, k=[600, 0, 320, 0, 601, 240, 0, 0, 1], d=[0.1, 0.2, 0, 0, 0.3])
    intr = CameraIntrinsics.from_camera_info(msg)
    assert (intr.fx, intr.fy, intr.cx, intr.cy) == (600, 601, 320, 240)
    np.testing.assert_array_equal(intr.dist, [0.1, 0.2, 0, 0, 0.3])


# --- transforms ---


def test_realsense_extrinsics_are_column_major():
    R = Rotation.from_euler("xyz", [3, -5, 10], degrees=True).as_matrix()
    t = np.array([0.015, 0.001, -0.002])
    msg = SimpleNamespace(rotation=R.flatten(order="F"), translation=t)  # as librealsense stores it
    T = RigidTransform.from_extrinsics_msg(msg)

    rs_ext = rs.extrinsics()
    rs_ext.rotation, rs_ext.translation = [float(x) for x in msg.rotation], t.tolist()
    p = np.array([0.3, -0.2, 1.5])
    np.testing.assert_allclose(T.apply(p), rs.rs2_transform_point_to_point(rs_ext, p.tolist()), atol=1e-6)
    np.testing.assert_allclose(T.apply(p), R @ p + t)


def test_transform_msg_dispatch():
    q = Rotation.from_euler("z", 30, degrees=True).as_quat()
    msg = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=1.0, y=2.0, z=3.0),
            rotation=SimpleNamespace(x=q[0], y=q[1], z=q[2], w=q[3]),
        )
    )
    T = RigidTransform.from_extrinsics_msg(msg)
    np.testing.assert_allclose(T.translation, [1, 2, 3])
    np.testing.assert_allclose(T.quat_xyzw, q)


def test_compose_and_inverse():
    A = RigidTransform(Rotation.random(random_state=1), rng.normal(size=3))
    B = RigidTransform(Rotation.random(random_state=2), rng.normal(size=3))
    pts = random_points(10)
    np.testing.assert_allclose((A @ B).apply(pts), A.apply(B.apply(pts)), atol=1e-12)
    np.testing.assert_allclose((A @ B).as_matrix(), A.as_matrix() @ B.as_matrix(), atol=1e-12)
    np.testing.assert_allclose(A.inverse().apply(A.apply(pts)), pts, atol=1e-12)


# --- interpolation ---


def make_timeline(rate_hz=120, n=50, t0=1_789_000_000_000_000_000, shuffle=True):
    dt = sec_to_ns(1 / rate_hz)
    stamps = t0 + np.arange(n) * dt
    pos = np.column_stack([np.arange(n) * 0.01, np.zeros(n), np.ones(n)])
    rots = Rotation.from_euler("z", np.arange(n)[:, None] * 1.0, degrees=True)
    order = rng.permutation(n) if shuffle else np.arange(n)
    return PoseTimeline(stamps[order], pos[order], rots.as_quat()[order]), stamps, pos


def test_slerp_midpoint_and_unsorted_input():
    tl, stamps, pos = make_timeline()
    t = (stamps[10] + stamps[11]) // 2
    T = tl.pose_at(t, "slerp", max_delta_ns=sec_to_ns(0.05))
    np.testing.assert_allclose(T.translation, (pos[10] + pos[11]) / 2, atol=1e-9)
    np.testing.assert_allclose(T.rotation.as_euler("xyz", degrees=True)[2], 10.5, atol=1e-6)


def test_slerp_rejects_extrapolation_and_gaps():
    tl, stamps, _ = make_timeline()
    d = sec_to_ns(0.05)
    assert tl.pose_at(int(stamps[0]) - 1, "slerp", d) is None
    assert tl.pose_at(int(stamps[-1]) + 1, "slerp", d) is None
    assert tl.pose_at(int(stamps[-1]), "slerp", d) is not None
    assert tl.pose_at(int(stamps[0]), "slerp", d) is not None
    # 120 Hz spacing (~8.3 ms) exceeds a 5 ms gap limit.
    assert tl.pose_at(int(stamps[5]) + 1000, "slerp", d, max_gap_ns=sec_to_ns(0.005)) is None


def test_nearest_respects_max_delta_in_ns():
    tl, stamps, pos = make_timeline()
    T = tl.pose_at(int(stamps[3]) + 1_000_000, "nearest", max_delta_ns=sec_to_ns(0.002))
    np.testing.assert_allclose(T.translation, pos[3])
    assert tl.pose_at(int(stamps[-1]) + sec_to_ns(0.051), "nearest", max_delta_ns=sec_to_ns(0.05)) is None


def test_duplicate_stamps_are_dropped():
    tl = PoseTimeline([0, 0, 10], np.zeros((3, 3)), np.tile([0, 0, 0, 1.0], (3, 1)))
    assert len(tl) == 2


def test_nearest_index():
    ts = np.array([0, 100, 200], dtype=np.int64)
    assert nearest_index(ts, 140, 50) == 1
    assert nearest_index(ts, 160, 50) == 2
    assert nearest_index(ts, 260, 50) is None
    assert nearest_index(np.empty(0, np.int64), 0, 50) is None


# --- alignment ---


def make_calib():
    T_color_depth = RigidTransform(Rotation.from_euler("xyz", [0.2, -0.3, 0.1], degrees=True), np.array([0.015, 0.0, 0.0]))
    return RGBDCalibration(depth=DEPTH, color=COLOR, T_color_depth=T_color_depth)


def test_align_matches_realsense_reference():
    """Vectorized pipeline reproduces the original per-pixel rs2_* loop."""
    calib = make_calib()
    depth = np.full((DEPTH.height, DEPTH.width), 2.0, dtype=np.float32)
    depth[100:200, 100:300] = 1.2
    depth[:, :20] = 0.0
    color = rng.integers(0, 255, (COLOR.height, COLOR.width, 3), dtype=np.uint8)

    out = align_depth_to_color(depth, color, calib, 0.2, 5.0, stride=7)

    rs_d, rs_c = to_rs_intrinsics(DEPTH), to_rs_intrinsics(COLOR)
    rs_ext = rs.extrinsics()
    rs_ext.rotation = [float(x) for x in calib.T_color_depth.rotation.as_matrix().flatten(order="F")]
    rs_ext.translation = calib.T_color_depth.translation.tolist()
    ref_pts, ref_cols = [], []
    for v in range(0, DEPTH.height, 7):
        for u in range(0, DEPTH.width, 7):
            z = float(depth[v, u])
            if not 0.2 <= z <= 5.0:
                continue
            p_d = rs.rs2_deproject_pixel_to_point(rs_d, [float(u), float(v)], z)
            uc, vc = rs.rs2_project_point_to_pixel(rs_c, rs.rs2_transform_point_to_point(rs_ext, p_d))
            uc, vc = int(np.floor(uc + 0.5)), int(np.floor(vc + 0.5))
            if 0 <= uc < COLOR.width and 0 <= vc < COLOR.height:
                ref_pts.append(p_d)
                ref_cols.append(color[vc, uc])

    assert len(out.points_depth) == len(ref_pts)
    np.testing.assert_allclose(out.points_depth, ref_pts, atol=1e-5)
    # float32 rounding in librealsense can flip a pixel right at a rounding boundary.
    assert np.mean(np.all(out.colors == np.array(ref_cols), axis=1)) > 0.99


def test_zbuffer_keeps_nearest_and_occlusion_filter():
    # Two depth pixels (x/z = -0.5, +0.5) that land on the same pixel of a 1x1 color camera
    # (u_c = 0.5 * x/z = +/-0.25 -> rounds to 0); the near one must win the z-buffer.
    depth_intr = CameraIntrinsics(2, 1, 1.0, 1.0, 0.5, 0.0)
    color_intr = CameraIntrinsics(1, 1, 0.5, 0.5, 0.0, 0.0)
    calib = RGBDCalibration(depth_intr, color_intr, RigidTransform.identity())
    depth = np.array([[1.0, 3.0]], dtype=np.float32)
    color = np.array([[[10, 20, 30]]], dtype=np.uint8)

    out = align_depth_to_color(depth, color, calib, 0.2, 5.0)
    assert len(out.points_depth) == 2
    assert out.depth_in_color[0, 0] == pytest.approx(1.0)

    out = align_depth_to_color(depth, color, calib, 0.2, 5.0, occlusion_tol_m=0.1)
    np.testing.assert_allclose(out.points_depth[:, 2], [1.0])


def test_depth_to_points_range_and_stride():
    depth = np.zeros((DEPTH.height, DEPTH.width), np.float32)
    depth[::2, ::2] = 1.0
    depth[0, 0] = 9.0
    pts = depth_to_points(depth, DEPTH, 0.2, 5.0, stride=2)
    assert len(pts) == (DEPTH.height // 2) * (DEPTH.width // 2) - 1
    assert np.all(pts[:, 2] == 1.0)


def test_depth_overlay_only_touches_valid_pixels():
    color = np.full((4, 4, 3), 100, np.uint8)
    d = np.zeros((4, 4), np.float32)
    d[1, 2] = 1.0
    out = depth_overlay(color, d, 0.2, 5.0, alpha=0.5)
    changed = np.any(out != color, axis=2)
    assert changed[1, 2] and changed.sum() == 1


# --- reconstruction ---


def test_voxel_keys_unique_per_voxel_and_signed():
    pts = np.array([[0.01, 0.01, 0.01], [0.04, 0.02, 0.0], [-0.01, 0.0, 0.0], [0.0, -0.01, 0.0], [0.06, 0.0, 0.0]])
    keys = voxel_keys(pts, 0.05)
    assert keys[0] == keys[1]
    assert len(set(keys.tolist())) == 4


def test_voxel_downsample_keep_policy():
    pts = np.array([[0.01, 0, 0], [0.02, 0, 0], [1.0, 0, 0]])
    cols = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], np.uint8)
    p, c = voxel_downsample(pts, cols, 0.05, keep="first")
    np.testing.assert_array_equal(c[:, 0], [1, 3])
    p, c = voxel_downsample(pts, cols, 0.05, keep="latest")
    np.testing.assert_array_equal(c[:, 0], [2, 3])


def test_voxel_map_eviction_is_by_recency():
    vm = VoxelMap(voxel_size=1.0, max_points=3, keep="first")
    col = lambda n, val: np.full((n, 3), val, np.uint8)  # noqa: E731
    vm.insert(np.array([[0.5, 0, 0], [1.5, 0, 0]]), col(2, 1))  # voxels A, B
    vm.insert(np.array([[2.5, 0, 0], [0.6, 0, 0]]), col(2, 2))  # C, re-observe A
    vm.insert(np.array([[3.5, 0, 0]]), col(1, 3))  # D -> evict B (least recent)
    assert len(vm) == 3
    xs = sorted(np.floor(vm.points[:, 0]).astype(int).tolist())
    assert xs == [0, 2, 3]
    # keep="first": voxel A retains its first color.
    assert vm.colors[np.floor(vm.points[:, 0]) == 0][0, 0] == 1


# --- ros_io ---


def test_decode_compressed_depth_16u():
    depth_mm = rng.integers(0, 6000, (20, 30), dtype=np.uint16)
    ok, png = cv2.imencode(".png", depth_mm)
    msg = SimpleNamespace(format="16UC1; compressedDepth png", data=np.frombuffer(b"\0" * 12 + png.tobytes(), np.uint8))
    np.testing.assert_allclose(decode_compressed_depth(msg), depth_mm.astype(np.float32) / 1000, rtol=1e-6)


def test_stamp_ns_prefers_header():
    msg = SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1_789_000_000, nanosec=123)))
    assert stamp_ns(msg, 5) == 1_789_000_000_000_000_123
    assert stamp_ns(SimpleNamespace(), 5) == 5
