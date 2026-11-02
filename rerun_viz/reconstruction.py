"""Naive numpy voxel-grid point aggregation."""

from __future__ import annotations

from typing import Literal

import numpy as np

KeepPolicy = Literal["first", "latest"]

# Voxel indices are packed into one int64 key, 21 bits per axis (signed via offset).
_BITS = 21
_OFFSET = 1 << (_BITS - 1)
_MASK = (1 << _BITS) - 1


def voxel_keys(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Pack the integer voxel coordinate of each point into a single int64 key.

    Covers +/- 2^20 voxels per axis (e.g. +/- 52 km at 5 cm voxels).
    """
    ijk = np.floor(points / voxel_size).astype(np.int64) + _OFFSET
    if ijk.size and (ijk.min() < 0 or ijk.max() > _MASK):
        raise ValueError("Points fall outside the representable voxel range; increase voxel_size.")
    return (ijk[:, 0] << (2 * _BITS)) | (ijk[:, 1] << _BITS) | ijk[:, 2]


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel_size: float, keep: KeepPolicy = "first"
) -> tuple[np.ndarray, np.ndarray]:
    """Keep one point per occupied voxel (the first or last one in input order)."""
    if len(points) == 0:
        return points, colors
    keys = voxel_keys(points, voxel_size)
    if keep == "latest":
        _, rev_idx = np.unique(keys[::-1], return_index=True)
        idx = len(keys) - 1 - rev_idx
    else:
        _, idx = np.unique(keys, return_index=True)
    idx.sort()  # preserve input order
    return points[idx], colors[idx]


class VoxelMap:
    """Accumulates colored points into a sparse voxel grid with one point per voxel.

    ``keep="first"`` keeps the earliest observation of each voxel, ``"latest"``
    replaces it with the newest. When ``max_points`` is exceeded, voxels that
    were least recently observed are evicted first.
    """

    def __init__(self, voxel_size: float = 0.05, max_points: int | None = None, keep: KeepPolicy = "first"):
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.keep = keep
        self.points = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self._keys = np.empty(0, dtype=np.int64)
        self._last_seen = np.empty(0, dtype=np.int64)
        self._frame = 0

    def __len__(self) -> int:
        return len(self.points)

    def insert(self, points_world: np.ndarray, colors: np.ndarray) -> None:
        if len(points_world) == 0:
            return
        self._frame += 1
        new_keys = voxel_keys(points_world, self.voxel_size)

        keys = np.concatenate([self._keys, new_keys])
        pts = np.concatenate([self.points, points_world])
        cols = np.concatenate([self.colors, np.asarray(colors, dtype=np.uint8)])

        # One entry per voxel, choosing which observation supplies the point/color.
        if self.keep == "latest":
            uniq, rev_idx = np.unique(keys[::-1], return_index=True)
            idx = len(keys) - 1 - rev_idx
        else:
            uniq, idx = np.unique(keys, return_index=True)

        # Any voxel touched this frame counts as seen now, regardless of keep policy.
        last_seen = np.full(len(uniq), -1, dtype=np.int64)
        old_pos = np.searchsorted(uniq, self._keys)
        last_seen[old_pos] = self._last_seen
        last_seen[np.searchsorted(uniq, new_keys)] = self._frame

        self._keys, self.points, self.colors, self._last_seen = uniq, pts[idx], cols[idx], last_seen

        if self.max_points is not None and len(self._keys) > self.max_points:
            keep_idx = np.argsort(-self._last_seen, kind="stable")[: self.max_points]
            keep_idx.sort()
            self._keys = self._keys[keep_idx]
            self.points = self.points[keep_idx]
            self.colors = self.colors[keep_idx]
            self._last_seen = self._last_seen[keep_idx]
