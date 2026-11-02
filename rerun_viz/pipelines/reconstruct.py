"""Headless chunked TSDF reconstruction (chunk -> integrate -> flush to disk -> evict -> merge).

Synchronized frames are split into temporal chunks. Each chunk is integrated
into its own TSDF volume by a worker process that streams only its slice of the
bag, extracts the surface, writes it to ``output_dir`` and exits, so peak memory
is bounded by ``workers x (one chunk volume)`` regardless of sequence length.
Chunk surfaces are merged at the end (the merged cloud is voxel-deduplicated;
the merged mesh is a concatenation, so surfaces seen by several chunks can
appear layered: use ``chunk_size: 0`` for a single, globally fused volume).
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import open3d as o3d

from ..camera import Undistorter
from ..config import Config
from ..dataset import PlannedFrame, Rig, index_bags, stream_frames
from ..progress import NullReporter, Reporter
from ..ros_io import count_messages
from ..tsdf import TSDFVolume, prepare_rgbd


@dataclass
class ChunkJob:
    index: int
    cfg: Config
    rig: Rig
    frames: list[PlannedFrame]
    out_dir: Path


@dataclass
class ChunkResult:
    index: int
    n_frames: int
    seconds: float
    mesh_path: Path | None = None
    cloud_path: Path | None = None
    n_vertices: int = 0
    n_points: int = 0


@dataclass
class ReconResult:
    output_dir: Path
    chunks: list[ChunkResult] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)
    mesh_path: Path | None = None
    cloud_path: Path | None = None
    trajectory_path: Path | None = None
    stats: dict[str, int] = field(default_factory=dict)


def integrate_chunk(job: ChunkJob, progress_queue=None) -> ChunkResult:
    """Integrate one chunk into a fresh volume and flush its surfaces to disk. Runs in a worker process."""
    t0 = time.perf_counter()
    cfg, rcfg, rig = job.cfg, job.cfg.recon, job.rig
    undistort = Undistorter(rig.calib.color) if rcfg.undistort_color else None
    intr = undistort.target if undistort else rig.calib.color.pinhole()
    volume = TSDFVolume(rcfg, intr, cfg.depth.max_m)

    def tick(n: int) -> None:
        if progress_queue is not None:
            progress_queue.put((job.index, n))

    bags = [Path(p) for p in cfg.bags]
    for frame in stream_frames(bags, cfg, job.frames, on_progress=tick):
        depth, color, _ = prepare_rgbd(
            frame.depth_m, frame.color, rig.calib, undistort, cfg.depth.min_m, cfg.depth.max_m
        )
        volume.integrate(depth, color, frame.plan.T_world_rover @ rig.T_rover_color)

    result = ChunkResult(job.index, volume.n_integrated, 0.0)
    stem = job.out_dir / f"chunk_{job.index:04d}"
    if rcfg.save_mesh and volume.n_integrated:
        mesh = volume.extract_mesh()
        result.n_vertices = len(mesh.vertices)
        if result.n_vertices:
            result.mesh_path = stem.with_name(stem.name + "_mesh.ply")
            o3d.io.write_triangle_mesh(str(result.mesh_path), mesh)
    if rcfg.save_point_cloud and volume.n_integrated:
        cloud = volume.extract_point_cloud()
        result.n_points = len(cloud.points)
        if result.n_points:
            result.cloud_path = stem.with_name(stem.name + "_cloud.ply")
            o3d.io.write_point_cloud(str(result.cloud_path), cloud)
    result.seconds = time.perf_counter() - t0
    return result


def write_tum_trajectory(path: Path, frames: list[PlannedFrame], rig: Rig) -> None:
    """Color-camera trajectory in TUM format (``t tx ty tz qx qy qz qw``, t in seconds)."""
    with open(path, "w") as f:
        for fr in sorted(frames, key=lambda x: x.t_ns):
            T = fr.T_world_rover @ rig.T_rover_color
            vals = [*T.translation, *T.quat_xyzw]
            f.write(f"{fr.t_ns / 1e9:.9f} " + " ".join(f"{v:.9f}" for v in vals) + "\n")


def merge_chunks(results: list[ChunkResult], out_dir: Path, cfg: Config, reporter: Reporter) -> tuple[Path | None, Path | None]:
    rcfg = cfg.recon
    mesh_path = cloud_path = None
    results = sorted(results, key=lambda r: r.index)

    meshes = [r.mesh_path for r in results if r.mesh_path and r.n_vertices]
    if meshes:
        task = reporter.task("Merging meshes", len(meshes))
        merged = o3d.geometry.TriangleMesh()
        for p in meshes:
            merged += o3d.io.read_triangle_mesh(str(p))
            task.advance()
        merged.merge_close_vertices(rcfg.voxel_size * 0.25)
        merged.remove_duplicated_triangles()
        merged.remove_degenerate_triangles()
        merged.remove_unreferenced_vertices()
        merged.compute_vertex_normals()
        mesh_path = out_dir / "mesh.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), merged)
        task.finish()
        reporter.log(f"Merged mesh: {len(merged.vertices)} vertices, {len(merged.triangles)} triangles -> {mesh_path}")

    clouds = [r.cloud_path for r in results if r.cloud_path and r.n_points]
    if clouds:
        task = reporter.task("Merging point clouds", len(clouds))
        merged = o3d.geometry.PointCloud()
        for p in clouds:
            merged += o3d.io.read_point_cloud(str(p))
            task.advance()
        merged = merged.voxel_down_sample(rcfg.merge_voxel_size or rcfg.voxel_size)
        cloud_path = out_dir / "cloud.ply"
        o3d.io.write_point_cloud(str(cloud_path), merged)
        task.finish()
        reporter.log(f"Merged cloud: {len(merged.points)} points -> {cloud_path}")

    if not cfg.recon.keep_chunks:
        for r in results:
            for p in (r.mesh_path, r.cloud_path):
                if p:
                    p.unlink(missing_ok=True)
    return mesh_path, cloud_path


def run_reconstruction(cfg: Config, reporter: Reporter | None = None) -> ReconResult:
    reporter = reporter or NullReporter()
    rcfg = cfg.recon
    bags = [Path(p) for p in cfg.bags]
    out_dir = Path(rcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ReconResult(out_dir)

    t = cfg.topics
    calib_topics = {t.pose, t.color, t.color_info, t.depth, t.depth_info, t.extrinsics}
    task = reporter.task("Indexing bags", count_messages(bags, calib_topics))
    index = index_bags(bags, cfg, on_progress=task.advance)
    task.finish()
    result.stats = dict(index.stats)
    if index.calib is None:
        raise RuntimeError("Missing camera_info / depth->color extrinsics; cannot reconstruct.")
    rig = Rig.from_config(cfg, index.calib)

    frames = index.frames[:: max(1, rcfg.frame_stride)]
    size = rcfg.chunk_size if rcfg.chunk_size > 0 else max(1, len(frames))
    jobs = [ChunkJob(i, cfg, rig, frames[s : s + size], out_dir) for i, s in enumerate(range(0, len(frames), size))]
    reporter.log(
        f"{index.stats['synced']} synchronized frames -> {len(frames)} to integrate in {len(jobs)} chunk(s) "
        f"of <= {size} ({rcfg.backend} backend, {min(rcfg.workers, len(jobs)) or 1} worker(s))"
    )
    write_tum_trajectory(out_dir / "trajectory_tum.txt", frames, rig)
    result.trajectory_path = out_dir / "trajectory_tum.txt"

    overall = reporter.task("Integrating frames", len(frames))
    chunk_tasks = {}

    def on_tick(idx: int, n: int) -> None:
        if idx not in chunk_tasks:
            chunk_tasks[idx] = reporter.task(f"  chunk {idx:04d}", len(jobs[idx].frames))
        chunk_tasks[idx].advance(n)
        overall.advance(n)

    def on_done(idx: int, res: ChunkResult | None, err: BaseException | None) -> None:
        if idx in chunk_tasks:
            chunk_tasks[idx].finish(hide=True)
        if err is not None:
            result.failed[idx] = repr(err)
            reporter.log(f"[red]x chunk {idx:04d} failed: {err!r}")
        else:
            result.chunks.append(res)
            reporter.log(
                f"[green]v[/] chunk {idx:04d}: {res.n_frames} frames, {res.n_vertices} mesh vertices, "
                f"{res.n_points} points in {res.seconds:.1f}s"
            )

    if rcfg.workers <= 1 or len(jobs) <= 1:
        _run_in_process(jobs, on_tick, on_done)
    else:
        _run_in_pool(jobs, rcfg.workers, on_tick, on_done)
    overall.finish()

    result.mesh_path, result.cloud_path = merge_chunks(result.chunks, out_dir, cfg, reporter)
    return result


def _run_in_process(jobs, on_tick, on_done) -> None:
    class _Q:
        def put(self, item):
            on_tick(*item)

    for job in jobs:
        try:
            on_done(job.index, integrate_chunk(job, _Q()), None)
        except Exception as e:  # noqa: BLE001 - report and continue with the remaining chunks
            on_done(job.index, None, e)


def _run_in_pool(jobs, workers: int, on_tick, on_done) -> None:
    # "spawn": CUDA and Open3D's thread pools do not survive fork(). max_tasks_per_child=1 retires each
    # worker after one chunk, returning all of its memory to the OS (the "evict" step).
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager:
        q = manager.Queue()

        def drain() -> None:
            while True:
                try:
                    on_tick(*q.get_nowait())
                except queue.Empty:
                    return

        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, max_tasks_per_child=1) as pool:
            pending: dict[Future, int] = {pool.submit(integrate_chunk, job, q): job.index for job in jobs}
            while pending:
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                drain()
                for fut in done:
                    idx = pending.pop(fut)
                    err = fut.exception()
                    on_done(idx, None if err else fut.result(), err)
            drain()
