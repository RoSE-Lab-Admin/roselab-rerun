"""Rigid-body transforms.

Naming convention: ``T_a_b`` maps points expressed in frame ``b`` into frame ``a``
(i.e. it is the pose of ``b`` in ``a``), so ``T_a_c = T_a_b @ T_b_c``.
Quaternions are always scalar-last ``[x, y, z, w]`` (scipy / ROS / rerun ``xyzw``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RigidTransform:
    rotation: Rotation
    translation: np.ndarray  # (3,)

    @classmethod
    def identity(cls) -> RigidTransform:
        return cls(Rotation.identity(), np.zeros(3))

    @classmethod
    def from_quat(cls, translation, quat_xyzw) -> RigidTransform:
        return cls(Rotation.from_quat(quat_xyzw), np.asarray(translation, dtype=np.float64))

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> RigidTransform:
        T = np.asarray(T, dtype=np.float64)
        return cls(Rotation.from_matrix(T[:3, :3]), T[:3, 3].copy())

    @classmethod
    def from_pose_msg(cls, msg) -> RigidTransform:
        """``geometry_msgs/msg/Pose`` (or ``PoseStamped``)."""
        pose = getattr(msg, "pose", msg)
        p, q = pose.position, pose.orientation
        return cls.from_quat([p.x, p.y, p.z], [q.x, q.y, q.z, q.w])

    @classmethod
    def from_transform_msg(cls, msg) -> RigidTransform:
        """``geometry_msgs/msg/Transform`` (or ``TransformStamped``)."""
        tf = getattr(msg, "transform", msg)
        t, q = tf.translation, tf.rotation
        return cls.from_quat([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])

    @classmethod
    def from_realsense_extrinsics(cls, msg) -> RigidTransform:
        """``realsense2_camera_msgs/msg/Extrinsics``.

        librealsense stores the 3x3 rotation **column-major**, so the flat
        9-vector must be reshaped in Fortran order to get the matrix that maps
        points from the source stream frame into the target stream frame.
        """
        R = np.asarray(msg.rotation, dtype=np.float64).reshape(3, 3, order="F")
        return cls(Rotation.from_matrix(R), np.asarray(msg.translation, dtype=np.float64))

    @classmethod
    def from_extrinsics_msg(cls, msg) -> RigidTransform:
        """Dispatch on either RealSense ``Extrinsics`` or a ROS ``Transform[Stamped]``."""
        if hasattr(msg, "rotation") and hasattr(msg, "translation") and not hasattr(msg.rotation, "w"):
            return cls.from_realsense_extrinsics(msg)
        return cls.from_transform_msg(msg)

    @property
    def quat_xyzw(self) -> np.ndarray:
        return self.rotation.as_quat()

    def as_matrix(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.rotation.as_matrix()
        T[:3, 3] = self.translation
        return T

    def inverse(self) -> RigidTransform:
        inv = self.rotation.inv()
        return RigidTransform(inv, -inv.apply(self.translation))

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform ``(N, 3)`` (or ``(3,)``) points."""
        return self.rotation.apply(points) + self.translation

    def __matmul__(self, other: RigidTransform) -> RigidTransform:
        return RigidTransform(self.rotation * other.rotation, self.rotation.apply(other.translation) + self.translation)
