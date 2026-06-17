from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from regression.pose import EndPose


@dataclass(frozen=True)
class ArmSpec:
    position: slice
    rotation: slice
    rotation_type: str
    rotation_order: str | None = None


DATASET_ARM_SPECS = {
    "calvin": [ArmSpec(slice(0, 3), slice(3, 6), "euler", "xyz")],
    "vlabench": [ArmSpec(slice(0, 3), slice(3, 6), "euler", "xyz")],
    "vlabench_15": [ArmSpec(slice(0, 3), slice(3, 6), "euler", "xyz")],
    "vlabench_30": [ArmSpec(slice(0, 3), slice(3, 6), "euler", "xyz")],
    "agibotbeta": [
        ArmSpec(slice(0, 3), slice(6, 10), "quat", "xyzw"),
        ArmSpec(slice(3, 6), slice(10, 14), "quat", "xyzw"),
    ],
    "robocoin": [
        ArmSpec(slice(0, 3), slice(3, 6), "euler", "xyz"),
        ArmSpec(slice(6, 9), slice(9, 12), "euler", "xyz"),
    ],
}


DATASET_GRIPPER_SPECS = {
    "calvin": [slice(6, 7)],
    "vlabench": [slice(6, 7)],
    "vlabench_15": [slice(6, 7)],
    "vlabench_30": [slice(6, 7)],
    "agibotbeta": [slice(14, 16)],
}


class EndActionChunk:
    """A chunk of absolute end-effector actions with SE(3) relative conversion."""

    def __init__(
        self,
        actions: NDArray[np.float64],
        arm_specs: Sequence[ArmSpec],
        action_dim: int,
    ) -> None:
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2:
            raise ValueError(f"actions must have shape (T, D), got {actions.shape}.")
        if actions.shape[1] != action_dim:
            raise ValueError(f"actions width must be action_dim={action_dim}, got {actions.shape[1]}.")
        self.actions = actions
        self.arm_specs = list(arm_specs)
        self.action_dim = action_dim
        self.gripper_specs: list[slice] = []

    @classmethod
    def from_array(cls, array: NDArray[np.float64], dataset_name: str, action_dim: int) -> "EndActionChunk":
        flat = np.asarray(array, dtype=np.float64).reshape(-1)
        if flat.size % action_dim != 0:
            raise ValueError(f"Action size {flat.size} is not divisible by action_dim={action_dim}.")
        dataset_key = dataset_name.lower()
        if dataset_key not in DATASET_ARM_SPECS:
            raise ValueError(f"Delta action mode does not know the pose layout for dataset: {dataset_name}.")
        chunk = cls(flat.reshape(-1, action_dim), DATASET_ARM_SPECS[dataset_key], action_dim)
        chunk.gripper_specs = DATASET_GRIPPER_SPECS.get(dataset_key, [])
        return chunk

    @property
    def poses(self) -> list[list[EndPose]]:
        return [
            [
                EndPose(
                    translation=row[spec.position],
                    rotation=row[spec.rotation],
                    rotation_type=spec.rotation_type,
                    rotation_order=spec.rotation_order,
                    degrees=False,
                )
                for spec in self.arm_specs
            ]
            for row in self.actions
        ]

    @property
    def relative_action(self) -> NDArray[np.float32]:
        """Return a full chunk where each end-effector pose is relative to the first timestep."""
        reference_poses = self.poses[0]
        relative = self.actions.copy()

        for t, current_poses in enumerate(self.poses):
            for spec, current_pose, reference_pose in zip(self.arm_specs, current_poses, reference_poses):
                relative_pose = current_pose - reference_pose
                relative[t, spec.position] = relative_pose.translation
                relative[t, spec.rotation] = self._rotation_vector(relative_pose, spec)

        for gripper_slice in self.gripper_specs:
            relative[:, gripper_slice] = self.actions[:, gripper_slice] - self.actions[0, gripper_slice]

        return relative.astype(np.float32, copy=False).reshape(-1)

    @staticmethod
    def _rotation_vector(pose: EndPose, spec: ArmSpec) -> NDArray[np.float64]:
        if spec.rotation_type == "quat":
            return pose.to_vector("quat", spec.rotation_order)[3:]
        if spec.rotation_type == "euler":
            return pose.to_vector("euler", spec.rotation_order)[3:]
        if spec.rotation_type == "rotvec":
            return pose.rotvec
        raise ValueError(f"Unsupported output rotation type: {spec.rotation_type}.")


def to_delta_action_target(action: NDArray[np.float64], dataset_name: str, action_dim: int) -> NDArray[np.float32]:
    return EndActionChunk.from_array(action, dataset_name, action_dim).relative_action.reshape(-1, action_dim)[-1]


def delta_to_absolute_last_action(
    first_action: NDArray[np.float64],
    delta_action: NDArray[np.float64],
    dataset_name: str,
    action_dim: int,
) -> NDArray[np.float32]:
    """Compose a predicted final relative end-pose with the first absolute action."""
    first = np.asarray(first_action, dtype=np.float64).reshape(action_dim)
    delta = np.asarray(delta_action, dtype=np.float64).reshape(action_dim)
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_ARM_SPECS:
        raise ValueError(f"Delta action mode does not know the pose layout for dataset: {dataset_name}.")

    absolute = first.copy()
    for spec in DATASET_ARM_SPECS[dataset_key]:
        reference_pose = EndPose(
            translation=first[spec.position],
            rotation=first[spec.rotation],
            rotation_type=spec.rotation_type,
            rotation_order=spec.rotation_order,
            degrees=False,
        )
        relative_pose = EndPose(
            translation=delta[spec.position],
            rotation=delta[spec.rotation],
            rotation_type=spec.rotation_type,
            rotation_order=spec.rotation_order,
            degrees=False,
        )
        absolute_pose = EndPose(homogeneous=reference_pose.homogeneous @ relative_pose.homogeneous)
        absolute[spec.position] = absolute_pose.translation
        absolute[spec.rotation] = EndActionChunk._rotation_vector(absolute_pose, spec)

    for gripper_slice in DATASET_GRIPPER_SPECS.get(dataset_key, []):
        absolute[gripper_slice] = first[gripper_slice] + delta[gripper_slice]

    return absolute.astype(np.float32, copy=False)


def canonicalize_absolute_action(
    action: NDArray[np.float64],
    dataset_name: str,
    action_dim: int,
    reference_action: NDArray[np.float64] | None = None,
) -> NDArray[np.float32]:
    """Return an absolute action vector in the same pose representation used by delta composition."""
    absolute = np.asarray(action, dtype=np.float64).reshape(action_dim).copy()
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_ARM_SPECS:
        return absolute.astype(np.float32, copy=False)

    reference = None if reference_action is None else np.asarray(reference_action, dtype=np.float64).reshape(action_dim)
    for spec in DATASET_ARM_SPECS[dataset_key]:
        pose = EndPose(
            translation=absolute[spec.position],
            rotation=absolute[spec.rotation],
            rotation_type=spec.rotation_type,
            rotation_order=spec.rotation_order,
            degrees=False,
        )
        absolute[spec.position] = pose.translation
        absolute[spec.rotation] = EndActionChunk._rotation_vector(pose, spec)
        if spec.rotation_type == "quat" and reference is not None:
            if np.dot(absolute[spec.rotation], reference[spec.rotation]) < 0:
                absolute[spec.rotation] *= -1.0

    return absolute.astype(np.float32, copy=False)
