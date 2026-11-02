"""Timestamp matching and pose interpolation.

All timestamps are integer nanoseconds (``int`` / ``np.int64``). Float seconds
since the epoch only carry ~0.2 us of precision, and mixing units (seconds vs
ns thresholds) is an easy bug to make, so seconds appear only at the config edge.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .transforms import RigidTransform

NS_PER_SEC = 1_000_000_000

InterpMethod = Literal["slerp", "nearest"]


def sec_to_ns(t: float) -> int:
    return int(round(t * NS_PER_SEC))


def nearest_index(sorted_ts: np.ndarray, t: int, max_delta_ns: int) -> int | None:
    """Index of the sample in ``sorted_ts`` closest to ``t``, or None if none is within ``max_delta_ns``."""
    n = len(sorted_ts)
    if n == 0:
        return None
    i = int(np.searchsorted(sorted_ts, t))
    best = min((j for j in (i - 1, i) if 0 <= j < n), key=lambda j: abs(int(sorted_ts[j]) - t))
    return best if abs(int(sorted_ts[best]) - t) <= max_delta_ns else None


class PoseTimeline:
    """Time-indexed pose stream (e.g. 120 Hz mocap) supporting interpolated lookups.

    Samples are sorted and de-duplicated on construction, so poses can be added
    in bag order (which, for merged bags, need not be header-time order).
    """

    def __init__(self, stamps_ns, positions, quats_xyzw):
        stamps = np.asarray(stamps_ns, dtype=np.int64)
        order = np.argsort(stamps, kind="stable")
        stamps = stamps[order]
        keep = np.concatenate([[True], np.diff(stamps) > 0]) if len(stamps) else np.zeros(0, bool)

        self.stamps_ns = stamps[keep]
        self.positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)[order][keep]
        quats = np.asarray(quats_xyzw, dtype=np.float64).reshape(-1, 4)[order][keep]
        self.rotations = Rotation.from_quat(quats) if len(quats) else None

        # Slerp in seconds relative to the first sample to keep float precision.
        self._t0 = int(self.stamps_ns[0]) if len(self) else 0
        self._slerp = Slerp(self._rel_sec(self.stamps_ns), self.rotations) if len(self) >= 2 else None

    @classmethod
    def from_samples(cls, samples: list[tuple[int, RigidTransform]]) -> PoseTimeline:
        return cls(
            [t for t, _ in samples],
            [p.translation for _, p in samples],
            [p.quat_xyzw for _, p in samples],
        )

    def __len__(self) -> int:
        return len(self.stamps_ns)

    def _rel_sec(self, t_ns) -> np.ndarray:
        return (np.asarray(t_ns, dtype=np.int64) - self._t0) / NS_PER_SEC

    def pose_at(
        self, t_ns: int, method: InterpMethod = "slerp", max_delta_ns: int = 50_000_000, max_gap_ns: int | None = None
    ) -> RigidTransform | None:
        """Pose at ``t_ns``.

        ``nearest``: closest sample, rejected if further than ``max_delta_ns``.
        ``slerp``: linear translation + SLERP rotation between the bracketing
        samples, rejected if ``t_ns`` is outside the timeline or the bracketing
        samples are more than ``max_gap_ns`` apart (default ``2 * max_delta_ns``).
        """
        if len(self) == 0:
            return None

        if method == "nearest":
            i = nearest_index(self.stamps_ns, t_ns, max_delta_ns)
            if i is None:
                return None
            return RigidTransform(self.rotations[i], self.positions[i].copy())

        if method != "slerp":
            raise ValueError(f"Unknown interpolation method: {method!r}")

        max_gap_ns = 2 * max_delta_ns if max_gap_ns is None else max_gap_ns
        i = int(np.searchsorted(self.stamps_ns, t_ns, side="right"))
        if i == 0 or i == len(self):
            # Exact hit on the last sample is fine; anything else is extrapolation.
            if len(self) and int(self.stamps_ns[-1]) == t_ns:
                return RigidTransform(self.rotations[-1], self.positions[-1].copy())
            return None

        t1, t2 = int(self.stamps_ns[i - 1]), int(self.stamps_ns[i])
        if t2 - t1 > max_gap_ns:
            return None

        alpha = (t_ns - t1) / (t2 - t1)
        position = (1.0 - alpha) * self.positions[i - 1] + alpha * self.positions[i]
        rotation = self._slerp(self._rel_sec(t_ns))
        return RigidTransform(rotation, position)
