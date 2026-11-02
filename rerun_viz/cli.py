"""Unified command line: ``rerun-viz viz | recon | config``.

Without ``--config`` (or with ``--edit``) an interactive, clickable form is
opened to choose options before running; in a non-interactive shell the
built-in defaults are used instead.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .config import Config
from .progress import RichReporter

app = typer.Typer(
    help="RGB-D + mocap rover bags: Rerun visualization and headless TSDF reconstruction.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", exists=True, dir_okay=False, help="YAML config (omit for the form)")
]
BagsOpt = Annotated[list[Path] | None, typer.Option("--bag", "-b", exists=True, help="Bag file/dir (repeatable; overrides config)")]
EditOpt = Annotated[bool, typer.Option("--edit", "-e", help="Review/edit options in the form before running")]


def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
        console=console,
    )


def resolve_config(config: Path | None, bags: list[Path] | None, edit: bool, mode: Literal["viz", "recon"]) -> Config:
    cfg = Config.load(config) if config else Config()
    if bags:
        cfg.bags = [str(b) for b in bags]
    if edit or config is None:
        if not sys.stdin.isatty():
            if edit:
                raise typer.BadParameter("--edit needs an interactive terminal")
            console.print("[yellow]No --config and no TTY: running with built-in defaults.")
            return cfg
        from .tui import edit_config  # Textual import is slow-ish; only pay for it when used

        edited = edit_config(cfg, mode, save_path=str(config or "config.yaml"))
        if edited is None:
            raise typer.Exit(1)
        cfg = edited
    return cfg


def summary(cfg: Config, mode: str) -> Table:
    t = Table(title=f"rerun-viz {mode}", show_header=False, box=None, title_justify="left")
    t.add_row("bags", "\n".join(cfg.bags))
    t.add_row("sync", f"{cfg.temporal_alignment.method}, max delta {cfg.temporal_alignment.max_delta_sec}s")
    t.add_row("depth", f"{cfg.depth.min_m}-{cfg.depth.max_m} m (scale {cfg.depth.scale})")
    if mode == "recon":
        r = cfg.recon
        t.add_row("tsdf", f"{r.backend} on {r.device if r.backend == 'tensor' else 'CPU'}, voxel {r.voxel_size} m, trunc {r.sdf_trunc} m")
        t.add_row("chunks", f"{r.chunk_size or 'single volume'} frames x {r.workers} worker(s), stride {r.frame_stride}")
        t.add_row("output", r.output_dir)
    else:
        t.add_row("frames", f"every {cfg.viewer.downsample}th triplet, map voxel {cfg.aggregation.voxel_size} m")
    return t


@app.command()
def viz(
    config: ConfigOpt = None,
    bag: BagsOpt = None,
    edit: EditOpt = False,
    spawn: Annotated[bool, typer.Option(help="Spawn a Rerun viewer")] = True,
):
    """Stream synchronized RGB-D, poses, plots and a live voxel map into Rerun."""
    from .pipelines.visualize import run_visualization

    cfg = resolve_config(config, bag, edit, "viz")
    console.print(summary(cfg, "viz"))
    with make_progress() as progress:
        run_visualization(cfg, RichReporter(progress), spawn=spawn)


@app.command()
def recon(
    config: ConfigOpt = None,
    bag: BagsOpt = None,
    edit: EditOpt = False,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Output directory")] = None,
    backend: Annotated[str | None, typer.Option(help="tensor | legacy")] = None,
    device: Annotated[str | None, typer.Option(help="Open3D device, e.g. CUDA:0 or CPU:0")] = None,
    workers: Annotated[int | None, typer.Option(help="Parallel chunk workers")] = None,
    chunk_size: Annotated[int | None, typer.Option(help="Frames per chunk (0 = single volume)")] = None,
    voxel_size: Annotated[float | None, typer.Option(help="TSDF voxel size (m)")] = None,
):
    """Headless chunked TSDF reconstruction (Open3D) to meshes / point clouds."""
    from .pipelines.reconstruct import run_reconstruction

    cfg = resolve_config(config, bag, edit, "recon")
    for name, value in [("output_dir", out and str(out)), ("backend", backend), ("device", device),
                        ("workers", workers), ("chunk_size", chunk_size), ("voxel_size", voxel_size)]:  # fmt: skip
        if value is not None:
            setattr(cfg.recon, name, value)
    console.print(summary(cfg, "recon"))
    with make_progress() as progress:
        result = run_reconstruction(cfg, RichReporter(progress))

    table = Table(title="Outputs", show_header=False, box=None, title_justify="left")
    for label, path in [("mesh", result.mesh_path), ("cloud", result.cloud_path), ("trajectory", result.trajectory_path)]:
        if path:
            table.add_row(label, str(path))
    console.print(table)
    if result.failed:
        console.print(f"[red]{len(result.failed)} chunk(s) failed: {sorted(result.failed)}")
        raise typer.Exit(2)


@app.command("config")
def config_cmd(
    path: Annotated[Path, typer.Argument(help="YAML to edit (created with defaults if missing)")] = Path("config.yaml"),
):
    """Create or edit a config YAML in the form."""
    from .tui import edit_config

    cfg = Config.load(path) if path.exists() else Config()
    edit_config(cfg, "config", save_path=str(path))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
