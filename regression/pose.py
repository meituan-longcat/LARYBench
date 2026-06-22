from enum import Enum
from typing import ClassVar, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


PoseT = TypeVar("PoseT", bound="Pose")


def invert_transformation(transform: NDArray[np.float64]) -> NDArray[np.float64]:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def relative_transformation(
    reference: NDArray[np.float64],
    target: NDArray[np.float64],
) -> NDArray[np.float64]:
    return invert_transformation(reference) @ target


class RotationType(Enum):
    QUAT = "quat"
    EULER = "euler"
    ROTVEC = "rotvec"
    MATRIX = "matrix"
    ROT6D = "rot6d"


class Pose:
    pose_type: ClassVar[str]

    def __sub__(self: PoseT, other: PoseT) -> PoseT:
        if type(self) is not type(other):
            raise TypeError(
                "Cannot compute a relative pose between "
                f"{type(self).__name__} and {type(other).__name__}."
            )
        return self._compute_relative(other)

    def _compute_relative(self: PoseT, other: PoseT) -> PoseT:
        raise NotImplementedError

    def copy(self: PoseT) -> PoseT:
        raise NotImplementedError


class EndPose(Pose):
    """End-effector pose represented internally as an SE(3) transform."""

    pose_type = "end_effector"

    def __init__(
        self,
        translation: list[float] | NDArray[np.float64] | None = None,
        rotation: list[float] | NDArray[np.float64] | None = None,
        rotation_type: str | RotationType | None = None,
        rotation_order: str | None = None,
        homogeneous: NDArray[np.float64] | None = None,
        degrees: bool = False,
    ) -> None:
        self._homogeneous_cache: NDArray[np.float64] | None = None
        self._cache_valid = False

        if homogeneous is not None:
            self._from_homogeneous(homogeneous)
            return

        self._translation = np.asarray(translation if translation is not None else np.zeros(3), dtype=np.float64)
        if self._translation.shape != (3,):
            raise ValueError(f"EndPose translation must have shape (3,), got {self._translation.shape}.")

        if rotation is None:
            self._rotation = Rotation.identity()
        else:
            if rotation_type is None:
                raise ValueError("rotation_type is required when rotation is provided.")
            self._rotation = self._make_rotation(rotation, rotation_type, rotation_order, degrees)

    def _from_homogeneous(self, homogeneous: NDArray[np.float64]) -> None:
        homogeneous = np.asarray(homogeneous, dtype=np.float64)
        if homogeneous.shape != (4, 4):
            raise ValueError(f"homogeneous must have shape (4, 4), got {homogeneous.shape}.")
        self._translation = homogeneous[:3, 3].copy()
        self._rotation = Rotation.from_matrix(homogeneous[:3, :3])

    @staticmethod
    def _make_rotation(
        rotation: list[float] | NDArray[np.float64],
        rotation_type: str | RotationType,
        rotation_order: str | None,
        degrees: bool,
    ) -> Rotation:
        rotation = np.asarray(rotation, dtype=np.float64)
        rot_type = rotation_type if isinstance(rotation_type, RotationType) else RotationType(rotation_type.lower())

        if rot_type == RotationType.QUAT:
            order = (rotation_order or "xyzw").lower()
            if order == "wxyz":
                rotation = np.array([rotation[1], rotation[2], rotation[3], rotation[0]], dtype=np.float64)
            elif order != "xyzw":
                raise ValueError(f"Unsupported quaternion order: {rotation_order}.")
            return Rotation.from_quat(rotation)
        if rot_type == RotationType.EULER:
            return Rotation.from_euler(rotation_order or "xyz", rotation, degrees=degrees)
        if rot_type == RotationType.ROTVEC:
            return Rotation.from_rotvec(rotation)
        if rot_type == RotationType.MATRIX:
            return Rotation.from_matrix(rotation)
        if rot_type == RotationType.ROT6D:
            return Rotation.from_matrix(EndPose.rot6d_to_matrix(rotation))
        raise ValueError(f"Unsupported rotation type: {rotation_type}.")

    @staticmethod
    def rot6d_to_matrix(rot6d: NDArray[np.float64]) -> NDArray[np.float64]:
        rot6d = np.asarray(rot6d, dtype=np.float64)
        a1 = rot6d[..., 0:3]
        a2 = rot6d[..., 3:6]

        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2, axis=-1)
        return np.stack([b1, b2, b3], axis=-1)

    @property
    def translation(self) -> NDArray[np.float64]:
        return self._translation.copy()

    @property
    def homogeneous(self) -> NDArray[np.float64]:
        if not self._cache_valid:
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = self._rotation.as_matrix()
            transform[:3, 3] = self._translation
            self._homogeneous_cache = transform
            self._cache_valid = True
        return self._homogeneous_cache.copy()

    @property
    def rotvec(self) -> NDArray[np.float64]:
        return self._rotation.as_rotvec()

    @property
    def quat_xyzw(self) -> NDArray[np.float64]:
        return self._rotation.as_quat()

    def to_vector(self, rotation_type: str = "rotvec", rotation_order: str | None = None) -> NDArray[np.float64]:
        rot_type = RotationType(rotation_type.lower())
        if rot_type == RotationType.ROTVEC:
            rotation = self._rotation.as_rotvec()
        elif rot_type == RotationType.EULER:
            rotation = self._rotation.as_euler(rotation_order or "xyz", degrees=False)
        elif rot_type == RotationType.QUAT:
            quat_xyzw = self._rotation.as_quat()
            if (rotation_order or "xyzw").lower() == "wxyz":
                rotation = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float64)
            else:
                rotation = quat_xyzw
        else:
            raise ValueError(f"Unsupported vector rotation type: {rotation_type}.")
        return np.concatenate([self._translation, rotation])

    def _compute_relative(self, other: "EndPose") -> "EndPose":
        return EndPose(homogeneous=relative_transformation(other.homogeneous, self.homogeneous))

    def copy(self) -> "EndPose":
        return EndPose(homogeneous=self.homogeneous)


EndEffectorPose = EndPose
