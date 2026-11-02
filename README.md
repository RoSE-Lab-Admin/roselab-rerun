# roselab-rerun
RGB-D + mocap rover bags (ROS 2 Jazzy, RealSense MastCam): Rerun visualization and headless
Open3D TSDF reconstruction. Tested on WSL Ubuntu 24.04.

```bash
uv sync
uv run rerun-viz viz   -c config.yaml            # Rerun viewer: synced RGB-D, poses, plots, live voxel map
uv run rerun-viz recon -c config.yaml            # headless chunked TSDF -> recon_out/{mesh,cloud}.ply
uv run rerun-viz recon                           # no config: pick options in a clickable form first
uv run rerun-viz viz -c config.yaml --edit       # review/edit a config in the form before running
uv run rerun-viz config my.yaml                  # create/edit a config YAML in the form
uv run rerun-viz recon --help                    # CLI overrides: --bag, --out, --workers, --chunk-size, ...
uv run pytest                                    # unit + end-to-end tests on a synthetic bag
```

`uv run main.py --config config.yaml --bags a.mcap` still works (runs `viz`).

## Reconstruction (chunk -> integrate -> flush -> evict -> merge)

Pass 1 indexes the bags once (timestamps, poses, calibration) and plans synchronized frames
(depth anchor, nearest color, interpolated mocap pose). Frames are split into `recon.chunk_size`
temporal chunks; each chunk runs in a spawned worker process that streams only its slice of the
bag, integrates a fresh TSDF volume, writes `chunk_XXXX_{mesh,cloud}.ply` and exits
(`max_tasks_per_child=1`), so peak memory is about `workers x one chunk volume`. Chunk outputs are
merged into `mesh.ply` / `cloud.ply` (the cloud is voxel-deduplicated; the mesh is concatenated,
so surfaces revisited across chunks may be layered; `chunk_size: 0` integrates one global volume).
A TUM trajectory of the color camera is written alongside.

Depth is registered into an undistorted pinhole color camera before integration, so Open3D
sees pixel-aligned RGB-D. Backends: `tensor` (`VoxelBlockGrid`, default, `device: CPU:0`) and
`legacy` (`ScalableTSDFVolume`, which returns empty volumes in the Open3D 0.20 / Python 3.14
wheel and is refused with an error there).

## Layout

| module | contents |
|---|---|
| `camera.py` | `CameraIntrinsics` (from `CameraInfo`), Brown-Conrady `project` / `deproject`, `Undistorter` |
| `transforms.py` | `RigidTransform` (`T_parent_child`, xyzw quats), from Pose / Transform / RealSense `Extrinsics` msgs |
| `interpolation.py` | int-ns timestamps, `nearest_index`, `PoseTimeline.pose_at` (`slerp` / `nearest`) |
| `alignment.py` | depth-anchored registration: `depth_to_points`, `align_depth_to_color` (z-buffered), `depth_overlay` |
| `reconstruction.py` | `voxel_downsample`, `VoxelMap` (one point per voxel, recency-based eviction) |
| `ros_io.py` | `iter_messages` (header stamp + log time, time windows), color / `compressedDepth` decoding |
| `dataset.py` | `index_bags` (pass 1 planning), `stream_frames` (pass 2, bounded memory), `Rig`, `T_LINK_DEPTH` |
| `tsdf.py` | `TSDFVolume` (tensor / legacy Open3D), `prepare_rgbd` |
| `config.py` | typed config dataclasses (YAML layout of `config.yaml`); field help drives the form |
| `pipelines/visualize.py` | Rerun pipeline |
| `pipelines/reconstruct.py` | headless chunked TSDF pipeline |
| `cli.py`, `tui.py`, `progress.py` | `rerun-viz` CLI (typer + rich progress), Textual config form |

Core modules (everything outside `pipelines/`, `cli.py`, `tui.py`) have no Rerun imports.
Conventions: timestamps are integer nanoseconds; `T_a_b` maps points in frame `b` into frame `a`;
camera frames are ROS/OpenCV optical frames (x right, y down, z forward).

## Frames in the viewer

- `shim.parent` (`world/rover_base`): mocap rover pose, interpolated at each depth stamp
- `shim.child` (`.../mastcam_front`): static shim from rover base to the MastCam camera_link
  (stop-gap until a URDF-based robot description provides the full tree)
- `shim.child/point_cloud`: depth-anchored points, placed in camera_link by `T_LINK_DEPTH` (`dataset.py`)
- `shim.child/color`: color camera pose in camera_link (`T_LINK_DEPTH` + depth->color extrinsics) + pinhole
