"""Pinhole camera intrinsics with Brown-Conrady (ROS ``plumb_bob``) distortion.

All projection functions are vectorized over ``(N, 3)`` point arrays and
``(N,)`` pixel arrays, and follow the OpenCV / ROS optical-frame convention:
+x right, +y down, +z forward (out of the lens), units in meters.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    # OpenCV ordering: [k1, k2, p1, p2, k3]
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @classmethod
    def from_camera_info(cls, msg) -> CameraIntrinsics:
        """Build from a ``sensor_msgs/msg/CameraInfo`` message."""
        k = np.asarray(msg.k, dtype=np.float64)
        d = np.zeros(5)
        coeffs = np.asarray(getattr(msg, "d", []), dtype=np.float64)[:5]
        d[: len(coeffs)] = coeffs
        return cls(
            width=int(msg.width),
            height=int(msg.height),
            fx=float(k[0]),
            fy=float(k[4]),
            cx=float(k[2]),
            cy=float(k[5]),
            dist=d,
        )

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])

    @property
    def is_distorted(self) -> bool:
        return bool(np.any(self.dist != 0.0))

    def pinhole(self) -> CameraIntrinsics:
        """Same camera with distortion removed (the target model of :class:`Undistorter`)."""
        return replace(self, dist=np.zeros(5))

    def in_bounds(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Mask of integer pixel coordinates that land inside the image."""
        return (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)

    def project(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project ``(N, 3)`` camera-frame points to float pixel coordinates ``(u, v)``.

        Points with ``z <= 0`` produce NaN pixels.
        """
        points = np.asarray(points, dtype=np.float64)
        z = points[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(z > 0, points[:, 0] / z, np.nan)
            y = np.where(z > 0, points[:, 1] / z, np.nan)
        x, y = distort(x, y, self.dist)
        return self.fx * x + self.cx, self.fy * y + self.cy

    def deproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Back-project pixels ``(u, v)`` at depth ``z`` (meters) to ``(N, 3)`` camera-frame points."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        x, y = undistort((u - self.cx) / self.fx, (v - self.cy) / self.fy, self.dist)
        return np.stack([x * z, y * z, z], axis=-1)


def distort(x: np.ndarray, y: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply Brown-Conrady distortion to normalized image coordinates (matches OpenCV)."""
    k1, k2, p1, p2, k3 = dist
    if not (k1 or k2 or p1 or p2 or k3):
        return x, y
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return xd, yd


def undistort(
    xd: np.ndarray, yd: np.ndarray, dist: np.ndarray, max_iter: int = 50, tol: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Invert :func:`distort` by fixed-point iteration (OpenCV's scheme, run to convergence).

    OpenCV's ``undistortPoints`` stops after 5 iterations, which leaves
    millimeter-level errors near the image corners for typical RealSense color lenses.
    """
    k1, k2, p1, p2, k3 = dist
    if not (k1 or k2 or p1 or p2 or k3):
        return xd, yd
    x, y = xd.copy(), yd.copy()
    for _ in range(max_iter):
        r2 = x * x + y * y
        radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x_new, y_new = (xd - dx) / radial, (yd - dy) / radial
        converged = np.nanmax(np.abs(x_new - x), initial=0.0) < tol and np.nanmax(np.abs(y_new - y), initial=0.0) < tol
        x, y = x_new, y_new
        if converged:
            break
    return x, y


class Undistorter:
    """Resamples images from a distorted camera onto its ideal pinhole model (same K).

    Remap tables are computed once per camera, so this is cheap per frame.
    """

    def __init__(self, intr: CameraIntrinsics):
        self.source = intr
        self.target = intr.pinhole()
        self._maps = None
        if intr.is_distorted:
            self._maps = cv2.initUndistortRectifyMap(
                intr.K, intr.dist, None, intr.K, (intr.width, intr.height), cv2.CV_32FC1
            )

    def __call__(self, img: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        if self._maps is None:
            return img
        return cv2.remap(img, *self._maps, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)
