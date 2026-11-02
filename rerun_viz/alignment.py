"""Depth-anchored RGB-D alignment.

The depth camera is the anchor: depth pixels are back-projected to 3D in the
depth optical frame, moved into the color optical frame with the depth->color
extrinsics, and projected into the color image to (a) pick up a color per point
and (b) build a depth image registered to the color camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import CameraIntrinsics
from .transforms import RigidTransform


@dataclass(frozen=True)
class RGBDCalibration:
    depth: CameraIntrinsics
    color: CameraIntrinsics
    T_color_depth: RigidTransform  # depth optical frame -> color optical frame


@dataclass
class AlignedRGBD:
    points_depth: np.ndarray  # (N, 3) points in the depth optical frame
    points_color: np.ndarray  # (N, 3) same points in the color optical frame
    colors: np.ndarray  # (N, 3) uint8 RGB sampled from the color image
    depth_in_color: np.ndarray  # (H_c, W_c) float32 meters, 0 where empty


def depth_to_points(
    depth_m: np.ndarray, intr: CameraIntrinsics, min_m: float, max_m: float, stride: int = 1
) -> np.ndarray:
    """Back-project a metric depth image to ``(N, 3)`` points, keeping ``min_m <= z <= max_m``.

    ``stride`` subsamples the pixel grid uniformly in both image axes.
    """
    v, u = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    z = depth_m[::stride, ::stride]
    valid = np.isfinite(z) & (z >= min_m) & (z <= max_m)
    return intr.deproject(u[valid], v[valid], z[valid])


def project_to_pixels(points: np.ndarray, intr: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points to integer pixel indices; returns ``(u, v, in_image_mask)``."""
    u, v = intr.project(points)
    valid = np.isfinite(u) & np.isfinite(v)
    # Round to the pixel whose center is nearest; NaNs are masked out below.
    ui = np.where(valid, np.floor(np.nan_to_num(u) + 0.5), -1).astype(np.int64)
    vi = np.where(valid, np.floor(np.nan_to_num(v) + 0.5), -1).astype(np.int64)
    return ui, vi, valid & intr.in_bounds(ui, vi)


def zbuffer(u: np.ndarray, v: np.ndarray, z: np.ndarray, width: int, height: int) -> np.ndarray:
    """Splat depths into an image, keeping the nearest ``z`` per pixel (0 = empty)."""
    img = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(img, v * width + u, z.astype(np.float32))
    img[np.isinf(img)] = 0.0
    return img.reshape(height, width)


def align_depth_to_color(
    depth_m: np.ndarray,
    color_img: np.ndarray,
    calib: RGBDCalibration,
    min_m: float,
    max_m: float,
    stride: int = 1,
    occlusion_tol_m: float | None = None,
) -> AlignedRGBD:
    """Color the depth cloud and register depth into the color camera.

    If ``occlusion_tol_m`` is set, points that are more than that distance
    behind the nearest surface seen by the color camera along the same pixel
    ray are dropped (they would otherwise take the color of the occluder).
    """
    h_c, w_c = color_img.shape[:2]
    pts_d = depth_to_points(depth_m, calib.depth, min_m, max_m, stride)
    pts_c = calib.T_color_depth.apply(pts_d)
    u, v, valid = project_to_pixels(pts_c, calib.color)

    pts_d, pts_c, u, v = pts_d[valid], pts_c[valid], u[valid], v[valid]
    depth_in_color = zbuffer(u, v, pts_c[:, 2], w_c, h_c)

    if occlusion_tol_m is not None:
        visible = pts_c[:, 2] <= depth_in_color[v, u] + occlusion_tol_m
        pts_d, pts_c, u, v = pts_d[visible], pts_c[visible], u[visible], v[visible]

    return AlignedRGBD(
        points_depth=pts_d,
        points_color=pts_c,
        colors=color_img[v, u],
        depth_in_color=depth_in_color,
    )


def depth_overlay(
    color_img: np.ndarray,
    depth_in_color: np.ndarray,
    min_m: float,
    max_m: float,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Alpha-blend a colormapped registered depth image over an RGB image."""
    mask = depth_in_color > 0
    norm = np.clip((depth_in_color - min_m) / (max_m - min_m), 0.0, 1.0)
    cm_rgb = cv2.cvtColor(cv2.applyColorMap((norm * 255).astype(np.uint8), colormap), cv2.COLOR_BGR2RGB)

    overlay = color_img.copy()
    blended = (1.0 - alpha) * color_img[mask].astype(np.float32) + alpha * cm_rgb[mask].astype(np.float32)
    overlay[mask] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return overlay
