import argparse
from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import rerun as rr
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation as R, Slerp
import yaml
from tqdm import tqdm

def decode_compressed_depth(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data[12:], np.uint8), cv2.IMREAD_UNCHANGED)


def ros_camerainfo_to_rs(msg) -> rs.intrinsics:
    intrin = rs.intrinsics()
    intrin.width, intrin.height = int(msg.width), int(msg.height)
    intrin.ppx, intrin.ppy = float(msg.k[2]), float(msg.k[5])
    intrin.fx, intrin.fy = float(msg.k[0]), float(msg.k[4])
    intrin.model = rs.distortion.brown_conrady
    intrin.coeffs = [float(c) for c in (msg.d[:5] if hasattr(msg, "d") else [0]*5)]
    return intrin


def ros_extrinsics_to_rs(msg) -> rs.extrinsics:
    extrin = rs.extrinsics()
    if hasattr(msg, "rotation") and hasattr(msg, "translation"):
        extrin.rotation = [float(r) for r in msg.rotation]
        extrin.translation = [float(t) for t in msg.translation]
    elif hasattr(msg, "transform"):
        q = msg.transform.rotation
        extrin.rotation = [float(r) for r in R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().flatten()]
        extrin.translation = [float(msg.transform.translation.x), float(msg.transform.translation.y), float(msg.transform.translation.z)]
    return extrin


def interpolate_pose(pose_timeline, target_ts, method="slerp", max_delta_ns=5e7):
    """Interpolates or selects nearest pose frame for a target timestamp."""
    if not pose_timeline:
        return None

    timestamps = np.array([p[0] for p in pose_timeline])
    idx = np.searchsorted(timestamps, target_ts)

    if method == "nearest":
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(timestamps)]
        if not candidates:
            return None
        best_idx = min(candidates, key=lambda i: abs(timestamps[i] - target_ts))
        if abs(timestamps[best_idx] - target_ts) > max_delta_ns:
            return None
        return pose_timeline[best_idx][1], R.from_quat(pose_timeline[best_idx][2])

    elif method == "slerp":
        if idx == 0 or idx == len(timestamps):
            return None
        t1, pos1, q1 = pose_timeline[idx - 1]
        t2, pos2, q2 = pose_timeline[idx]

        if (t2 - t1) > (2 * max_delta_ns) or not (t1 <= target_ts <= t2):
            return None

        alpha = (target_ts - t1) / (t2 - t1)
        pos_interp = (1.0 - alpha) * np.array(pos1) + alpha * np.array(pos2)

        slerp_obj = Slerp([t1, t2], R.from_quat([q1, q2]))
        rot_interp = slerp_obj([target_ts])[0]
        return pos_interp, rot_interp


def generate_depth_overlay(depth_m, color_img, depth_intrin, color_intrin, d2c_extrin, cfg):
    """Projects depth frame into color space and generates a blended colormap overlay."""
    h_c, w_c, _ = color_img.shape
    aligned_depth = np.zeros((h_c, w_c), dtype=np.float32)

    min_d = cfg.get("min_depth_m", 0.2)
    max_d = cfg.get("max_depth_m", 5.0)
    alpha = cfg.get("alpha", 0.4)

    v_grid, u_grid = np.indices(depth_m.shape)
    mask = (depth_m >= min_d) & (depth_m <= max_d)

    # Subsample pixel grid for fast overlay generation
    u_s, v_s, z_s = u_grid[mask][::2], v_grid[mask][::2], depth_m[mask][::2]

    for u, v, z in zip(u_s, v_s, z_s):
        pt_d = rs.rs2_deproject_pixel_to_point(depth_intrin, [float(u), float(v)], float(z))
        pt_c = rs.rs2_transform_point_to_point(d2c_extrin, pt_d)
        px_c = rs.rs2_project_point_to_pixel(color_intrin, pt_c)

        uc, vc = int(px_c[0]), int(px_c[1])
        if 0 <= uc < w_c and 0 <= vc < h_c:
            aligned_depth[vc, uc] = z

    depth_mask = aligned_depth > 0
    depth_norm = np.clip((aligned_depth - min_d) / (max_d - min_d), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)

    cmap_enum = getattr(cv2, cfg.get("colormap", "COLORMAP_JET"), cv2.COLORMAP_JET)
    cm_rgb = cv2.cvtColor(cv2.applyColorMap(depth_u8, cmap_enum), cv2.COLOR_BGR2RGB)

    overlay = color_img.copy()
    overlay[depth_mask] = cv2.addWeighted(
        color_img[depth_mask], 1.0 - alpha, cm_rgb[depth_mask], alpha, 0
    )
    return overlay


def voxel_grid_downsample(points, colors, voxel_size):
    if len(points) == 0:
        return points, colors
    voxel_coords = np.floor(points / voxel_size).astype(np.int32)
    _, idxs = np.unique(voxel_coords, axis=0, return_index=True)
    return points[idxs], colors[idxs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--bags", nargs="+")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bag_paths = [Path(p) for p in (args.bags or cfg.get("bags", []))]
    topics = cfg.get("topics", {})
    shim = cfg.get("shim", {})
    sync_cfg = cfg.get("temporal_alignment", {})
    overlay_cfg = cfg.get("overlay", {})
    plots_cfg = cfg.get("plots", {})
    agg_cfg = cfg.get("aggregation", {})
    view_cfg = cfg.get("viewer", {})

    rr.init("rover_mcap_visualizer", spawn=True)
    if blprnt := view_cfg.get("blueprint", None):
        rr.log_file_from_path(blprnt)

    rr.log(
        shim["child"],
        rr.Transform3D(
            translation=shim["translation"],
            rotation=rr.Quaternion(xyzw=shim["rotation_xyzw"]),
        ),
        static=True,
    )

    pose_timeline = []
    color_timeline = []
    depth_timeline = []

    rs_depth_intrin = None
    rs_color_intrin = None
    rs_d2c_extrin = None

    target_topics = set(topics.values())

    print("Indexing MCAP streams and logging timeseries plots...")
    with AnyReader(bag_paths) as reader:
        connections = [
            c for c in reader.connections if c.topic in target_topics
        ]

        for connection, timestamp, rawdata in reader.messages(
            connections=connections
        ):
            msg = reader.deserialize(rawdata, connection.msgtype)

            timestamp /= 1.0e9

            if hasattr(msg, "header"):
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9

            # print("Timestamp: ", timestamp)            
            rr.set_time("log_time", timestamp=timestamp)

            # --- Timeseries Plot Logging ---
            if connection.topic == topics.get("imu"):
                rr.log("plots/imu/accel/x", rr.Scalars(msg.linear_acceleration.x))
                rr.log("plots/imu/accel/y", rr.Scalars(msg.linear_acceleration.y))
                rr.log("plots/imu/accel/z", rr.Scalars(msg.linear_acceleration.z))
                rr.log("plots/imu/gyro/x", rr.Scalars(msg.angular_velocity.x))
                rr.log("plots/imu/gyro/y", rr.Scalars(msg.angular_velocity.y))
                rr.log("plots/imu/gyro/z", rr.Scalars(msg.angular_velocity.z))

            elif connection.topic == topics.get("joints"):
                extra_fields = plots_cfg.get("joints", {}).get("extra_fields", [])
                joint_names = getattr(msg, "name", [f"joint_{i}" for i in range(16)])

                for i, name in enumerate(joint_names):
                    if hasattr(msg, "position") and i < len(msg.position):
                        rr.log(f"plots/joints/{name}/position", rr.Scalars(msg.position[i]))
                    if hasattr(msg, "velocity") and i < len(msg.velocity):
                        rr.log(f"plots/joints/{name}/velocity", rr.Scalars(msg.velocity[i]))
                    if hasattr(msg, "effort") and i < len(msg.effort):
                        rr.log(f"plots/joints/{name}/effort", rr.Scalars(msg.effort[i]))
                    for field in extra_fields:
                        if hasattr(msg, field):
                            arr = getattr(msg, field)
                            if i < len(arr):
                                rr.log(f"plots/joints/{name}/{field}", rr.Scalars(arr[i]))

            # --- Calibration Metadata ---
            elif connection.topic == topics.get("color_info"):
                rs_color_intrin = ros_camerainfo_to_rs(msg)
                rr.log(
                    f"{shim['child']}/color",
                    rr.Pinhole(image_from_camera=np.array(msg.k).reshape(3, 3), width=msg.width, height=msg.height),
                    static=True,
                )
            elif connection.topic == topics.get("depth_info"):
                rs_depth_intrin = ros_camerainfo_to_rs(msg)
            elif connection.topic == topics.get("extrinsics"):
                rs_d2c_extrin = ros_extrinsics_to_rs(msg)

            # --- Buffer Image/Pose Stream Timelines ---
            elif connection.topic == topics.get("pose"):
                p, q = msg.pose.position, msg.pose.orientation
                pose_timeline.append((timestamp, [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]))
            elif connection.topic == topics.get("color"):
                color_timeline.append((timestamp, msg))
            elif connection.topic == topics.get("depth"):
                depth_timeline.append((timestamp, msg))

    print(f"Processing synchronized triplets...")

    color_ts_array = np.array([c[0] for c in color_timeline])
    
    print(f"Found {len(color_ts_array)} candidate color frames.")
    print(f"Found {len(depth_timeline)} candidate depth frames.")
    print(f"Found {len(pose_timeline)} candidate pose samples.")
    
    max_delta_ns = sync_cfg.get("max_delta_sec", 0.05) * 1e9
    method = sync_cfg.get("method", "slerp")

    accumulated_pts = np.empty((0, 3))
    accumulated_colors = np.empty((0, 3), dtype=np.uint8)

    DOWNSAMPLE_FACTOR = view_cfg.get("downsample", 256)

    for i, (t_depth, depth_msg) in enumerate(tqdm(depth_timeline)):
        # Only process every DOWNSAMPLE_FACTOR'th triplet
        if (i % DOWNSAMPLE_FACTOR):
            continue
        # print(f"Syncing triplet {i // DOWNSAMPLE_FACTOR}...")
        
        # 1. Temporal Sync: Nearest Color Frame
        if len(color_ts_array) == 0:
            continue
        c_idx = np.searchsorted(color_ts_array, t_depth)
        c_candidates = [i for i in (c_idx - 1, c_idx) if 0 <= i < len(color_ts_array)]
        if not c_candidates:
            continue
        best_c_idx = min(c_candidates, key=lambda i: abs(color_ts_array[i] - t_depth))

        if abs(color_ts_array[best_c_idx] - t_depth) > max_delta_ns:
            continue  # Drop unsynchronized frame

        color_msg = color_timeline[best_c_idx][1]

        # 2. Temporal Sync: Interpolated or Nearest Pose Frame
        pose_res = interpolate_pose(pose_timeline, t_depth, method=method, max_delta_ns=max_delta_ns)
        if pose_res is None:
            continue  # Drop frame due to missing/stale pose

        rover_pos, rover_rot = pose_res
        rr.set_time("log_time", timestamp=t_depth)

        # Log synchronized rover pose
        rr.log(
            shim["parent"],
            rr.Transform3D(translation=rover_pos, rotation=rr.Quaternion(xyzw=rover_rot.as_quat())),
        )

        # Decode Images
        np_color = cv2.imdecode(np.frombuffer(color_msg.data, np.uint8), cv2.IMREAD_COLOR)
        color_img = cv2.cvtColor(np_color, cv2.COLOR_BGR2RGB)

        depth_raw = decode_compressed_depth(depth_msg.data)
        depth_m = depth_raw.astype(np.float32) / 1000.0 if depth_raw.dtype == np.uint16 else depth_raw

        rr.log(f"{shim['child']}/color", rr.Image(color_img))

        # 3. Depth-Color Overlay Generation
        if overlay_cfg.get("enabled", True) and rs_depth_intrin and rs_color_intrin and rs_d2c_extrin:
            overlay_img = generate_depth_overlay(
                depth_m, color_img, rs_depth_intrin, rs_color_intrin, rs_d2c_extrin, overlay_cfg
            )
            rr.log(f"{shim['child']}/color_depth_overlay", rr.Image(overlay_img))

        # 4. Point Cloud Generation & World Map Aggregation
        if rs_depth_intrin and rs_color_intrin and rs_d2c_extrin:
            DEPTH_THRESH = agg_cfg.get("depth_threshold", 5.0) # Default to a depth threshold of 5m
            
            pts_local = []
            colors = []
            v_g, u_g = np.indices(depth_m.shape)
            mask = (depth_m > 0.2) & (depth_m < DEPTH_THRESH) # MIN / MAX DEPTH should be consistent for overlay and pc/agg
            stride = agg_cfg.get("stride", 4)

            for u, v, z in zip(u_g[mask][::stride], v_g[mask][::stride], depth_m[mask][::stride]):
                pt_d = rs.rs2_deproject_pixel_to_point(rs_depth_intrin, [float(u), float(v)], float(z))
                pt_c = rs.rs2_transform_point_to_point(rs_d2c_extrin, pt_d)
                px_c = rs.rs2_project_point_to_pixel(rs_color_intrin, pt_c)
                uc, vc = int(px_c[0]), int(px_c[1])

                if (0 <= uc < color_img.shape[1]) and (0 <= vc < color_img.shape[0]):
                    pts_local.append(pt_d)
                    colors.append(color_img[vc, uc])

            pts_local = np.array(pts_local)
            colors = np.array(colors)

            if len(pts_local) > 0:
                # For now we ALWAYS publish these raw points
                rr.log(f"{shim['child']}/point_cloud", rr.Points3D(positions=pts_local, colors=colors))

                if agg_cfg.get("enabled", True):
                    R_shim = R.from_quat(shim["rotation_xyzw"]).as_matrix()
                    t_shim = np.array(shim["translation"])

                    pts_rover = (R_shim @ pts_local.T).T + t_shim
                    pts_world = (rover_rot.as_matrix() @ pts_rover.T).T + rover_pos

                    accumulated_pts = np.vstack([accumulated_pts, pts_world])
                    accumulated_colors = np.vstack([accumulated_colors, colors])

                    accumulated_pts, accumulated_colors = voxel_grid_downsample(
                        accumulated_pts, accumulated_colors, agg_cfg.get("voxel_size", 0.05)
                    )

                    max_pts = agg_cfg.get("max_points", 250000)
                    if len(accumulated_pts) > max_pts:
                        accumulated_pts = accumulated_pts[-max_pts:]
                        accumulated_colors = accumulated_colors[-max_pts:]

                    rr.log(
                        "world/map/accumulated_cloud",
                        rr.Points3D(positions=accumulated_pts, colors=accumulated_colors),
                    )


if __name__ == "__main__":
    main()
