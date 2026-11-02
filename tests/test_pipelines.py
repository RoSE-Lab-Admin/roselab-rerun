"""End-to-end tests on a synthetic bag with known geometry (see ``synthetic.py``)."""

import asyncio
import dataclasses

import numpy as np
import open3d as o3d
import pytest

from rerun_viz.config import Config
from rerun_viz.dataset import index_bags, stream_frames
from rerun_viz.pipelines.reconstruct import run_reconstruction
from rerun_viz.pipelines.visualize import run_visualization
from rerun_viz.tsdf import legacy_backend_works

from synthetic import WALL_X, texture, write_bag


@pytest.fixture(scope="session")
def bag_cfg(tmp_path_factory) -> Config:
    cfg = Config()
    tmp = tmp_path_factory.mktemp("synthetic")
    cfg.bags = [str(write_bag(tmp / "bag", cfg))]
    cfg.viewer.blueprint = None
    cfg.recon.output_dir = str(tmp / "recon")
    cfg.recon.voxel_size = 0.01
    cfg.recon.sdf_trunc = 0.04
    cfg.recon.weight_threshold = 2.0
    return cfg


def surface_residual(points: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest scene plane (wall or ground)."""
    return np.minimum(np.abs(points[:, 0] - WALL_X), np.abs(points[:, 2]))


def test_index_plans_all_frames(bag_cfg):
    index = index_bags([*map(str, bag_cfg.bags)], bag_cfg)
    assert index.calib is not None
    assert index.stats["depth"] == 90 and index.stats["synced"] == 90
    np.testing.assert_allclose(index.calib.T_color_depth.translation, [-0.03, 0.0002, 0.0004])


def test_stream_pairs_out_of_order_messages(bag_cfg):
    index = index_bags(bag_cfg.bags, bag_cfg)
    frames = list(stream_frames(bag_cfg.bags, bag_cfg, index.frames[10:40]))
    assert sorted(f.plan.t_ns for f in frames) == [f.t_ns for f in index.frames[10:40]]
    assert frames[0].depth_m.dtype == np.float32 and frames[0].color.shape == (120, 160, 3)


@pytest.mark.parametrize(
    "backend,workers,chunk_size",
    [("tensor", 1, 0), ("legacy", 1, 45), ("tensor", 2, 30)],
    ids=["tensor-single-volume", "legacy-chunked", "tensor-process-pool"],
)
def test_reconstruction_matches_scene(bag_cfg, tmp_path, backend, workers, chunk_size):
    cfg = dataclasses.replace(bag_cfg, recon=dataclasses.replace(
        bag_cfg.recon, backend=backend, device="CPU:0", workers=workers, chunk_size=chunk_size,
        output_dir=str(tmp_path), keep_chunks=False,
    ))  # fmt: skip
    if backend == "legacy" and not legacy_backend_works():
        pytest.skip("ScalableTSDFVolume is broken in this Open3D build")
    result = run_reconstruction(cfg)
    n_chunks = 1 if chunk_size == 0 else -(-90 // chunk_size)
    assert not result.failed and len(result.chunks) == n_chunks
    assert sum(c.n_frames for c in result.chunks) == 90
    assert not list(tmp_path.glob("chunk_*"))  # keep_chunks=False cleaned up

    cloud = o3d.io.read_point_cloud(str(result.cloud_path))
    pts, cols = np.asarray(cloud.points), (np.asarray(cloud.colors) * 255).astype(int)
    assert len(pts) > 1000
    # Geometry: the whole extrinsic chain (mocap pose -> shim -> depth -> color) lands surfaces in place.
    # (A deliberately inverted depth->color extrinsic gives ~7.6 mm median / 66% color hits; correct: 1.7 mm / >93%.)
    assert np.median(surface_residual(pts)) < 0.004
    assert np.mean(surface_residual(pts) < 0.03) > 0.95
    # Color: depth->color registration + undistortion samples the right texel (checker edges excluded).
    snapped = pts.copy()
    on_wall = np.abs(pts[:, 0] - WALL_X) < np.abs(pts[:, 2])
    snapped[on_wall, 0], snapped[~on_wall, 2] = WALL_X, 0.0
    a = np.where(on_wall, snapped[:, 1], snapped[:, 0]) / 0.25
    b = np.where(on_wall, snapped[:, 2], snapped[:, 1]) / 0.25
    interior = (np.abs(a - np.round(a)) > 0.15) & (np.abs(b - np.round(b)) > 0.15)
    err = np.abs(cols[interior] - texture(snapped[interior]).astype(int)).max(axis=1)
    assert np.mean(err < 30) > 0.9 and np.median(err) < 5

    mesh = o3d.io.read_triangle_mesh(str(result.mesh_path))
    assert len(mesh.triangles) > 1000
    assert np.median(surface_residual(np.asarray(mesh.vertices))) < 0.004
    assert result.trajectory_path.read_text().count("\n") == 90


def test_visualization_runs_and_builds_map(bag_cfg):
    cfg = dataclasses.replace(bag_cfg, viewer=dataclasses.replace(bag_cfg.viewer, downsample=10))
    stats = run_visualization(cfg, spawn=False)
    assert stats["logged"] == 9
    assert stats["map_voxels"] > 100


def test_config_form_roundtrip(tmp_path):
    from rerun_viz.tui import ConfigForm

    async def run():
        cfg = Config()
        app = ConfigForm(cfg, "recon", save_path=str(tmp_path / "out.yaml"))
        async with app.run_test(size=(120, 60)) as pilot:
            app.query_one("#f__recon__voxel_size").value = "0.015"
            app.query_one("#f__recon__save_mesh").value = False
            app.query_one("#f__recon__merge_voxel_size").value = ""
            app.query_one("#f__shim__translation").value = "0.3, 0, 0.2"
            await pilot.click("#save")
            await pilot.pause()
        return Config.load(tmp_path / "out.yaml")

    out = asyncio.run(run())
    assert out.recon.voxel_size == 0.015 and out.recon.save_mesh is False
    assert out.recon.merge_voxel_size is None
    assert out.shim.translation == [0.3, 0.0, 0.2]
    assert out.topics == Config().topics


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config.load("config.yaml")
    cfg.save(tmp_path / "c.yaml")
    assert Config.load(tmp_path / "c.yaml") == cfg
