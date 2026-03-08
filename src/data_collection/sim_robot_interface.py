"""
Simulation robot interface wrapping IsaacLab's Articulation API.

Provides joint-level control for data collection, replacing
AutoDataCollector's FeetechController with sim equivalents.

This module runs INSIDE the IsaacLab conda subprocess (env_isaaclab).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .config import RobotSimConfig

# IsaacLab imports — only available inside the IsaacLab runtime.
# Guarded for type checking / static analysis outside IsaacLab.
try:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
except ImportError:
    if not TYPE_CHECKING:
        torch = None
        ManagerBasedRLEnv = None

logger = logging.getLogger(__name__)

_JOINT_NAME_ALIASES = {
    "left_inner_knuckle_joint": ("left_inner_finger_knuckle_joint",),
    "right_inner_knuckle_joint": ("right_inner_finger_knuckle_joint",),
    "left_inner_finger_knuckle_joint": ("left_inner_knuckle_joint",),
    "right_inner_finger_knuckle_joint": ("right_inner_knuckle_joint",),
}


class SimRobotInterface:
    """Joint-level robot control interface backed by IsaacLab Articulation.

    Wraps ``env.scene["robot"]`` to provide read/write access to arm joints,
    finger joints, and end-effector pose.  The action tensor sent to
    ``env.step()`` is assembled from the current joint-position targets.

    Args:
        env: A running ``ManagerBasedRLEnv`` instance.
        robot_cfg: Robot profile loaded from ``configs/robot_profiles/``.
        env_idx: Environment index for batched envs (default 0).
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: RobotSimConfig,
        env_idx: int = 0,
    ) -> None:
        self.env = env
        self.cfg = robot_cfg
        self.env_idx = env_idx

        # Resolve the articulation from the scene
        self._articulation = env.scene["robot"]

        # --- Resolve DOF indices from joint names ---
        self._arm_indices = self._resolve_joint_indices(robot_cfg.arm_joint_names)
        if robot_cfg.has_gripper_joints:
            self._finger_indices = self._resolve_joint_indices(robot_cfg.finger_joint_names)
        else:
            self._finger_indices = []

        self._all_indices = self._arm_indices + self._finger_indices

        # Action dimension from the env's action space
        self._action_dim = env.action_manager.total_action_dim

        # Pending joint targets (applied on next step_sim)
        self._pending_targets: np.ndarray | None = None

        # Track environment termination (success / timeout / failure)
        self._env_terminated: bool = False
        self._env_truncated: bool = False

        # Device (match the simulation tensor device)
        self._device = self._articulation.device

        # Resolve EE frame once. Prefer TCP frame when the profile provides one.
        self._ee_body_name, self._ee_body_idx = self._resolve_ee_body()

        logger.info(
            "SimRobotInterface initialized: %s | arm_dofs=%d finger_dofs=%d | action_dim=%d",
            robot_cfg.name,
            len(self._arm_indices),
            len(self._finger_indices),
            self._action_dim,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_joint_positions(self) -> np.ndarray:
        """Read all joint positions (arm + finger) in radians.

        Returns:
            Array of shape ``(total_dofs,)`` in the order defined by
            ``robot_cfg.all_joint_names``.
        """
        all_pos = self._articulation.data.joint_pos[self.env_idx]  # (num_joints,)
        return all_pos[self._all_indices].cpu().numpy()

    def read_target_positions(self) -> np.ndarray:
        """Read the most recently set joint position targets.

        Returns the pending targets if any have been set via
        :meth:`write_joint_positions` or :meth:`write_arm_joint_positions`,
        otherwise falls back to current joint positions (hold-position).

        Returns:
            Array of shape ``(total_dofs,)`` in radians.
        """
        return self._get_current_targets()

    def read_arm_joint_positions(self) -> np.ndarray:
        """Read arm joint positions in radians.

        Returns:
            Array of shape ``(arm_dofs,)``.
        """
        all_pos = self._articulation.data.joint_pos[self.env_idx]
        return all_pos[self._arm_indices].cpu().numpy()

    def read_ee_position(self) -> np.ndarray:
        """Read TCP/end-effector position in world frame.

        Returns:
            Array ``[x, y, z]`` in meters.
        """
        pos, _ = self.read_tcp_pose_world()
        return pos

    def read_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read TCP/end-effector pose in world frame.

        Returns:
            Tuple of (position ``[x,y,z]``, quaternion ``[w,x,y,z]``).
        """
        return self.read_tcp_pose_world()

    def read_tcp_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Read TCP pose in simulator world frame.

        Returns:
            Tuple of (position ``[x,y,z]``, quaternion ``[w,x,y,z]``).
        """
        return self._read_ee_frame()

    def read_tcp_pose_robot(self) -> tuple[np.ndarray, np.ndarray]:
        """Read TCP pose in the robot base frame."""
        pos_world, quat_world = self.read_tcp_pose_world()
        return self.world_pose_to_robot_pose(pos_world, quat_world)

    def read_root_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        """Read articulation root pose in simulator world frame."""
        root_pose = self._articulation.data.root_pose_w[self.env_idx]
        return (
            root_pose[:3].cpu().numpy(),
            root_pose[3:7].cpu().numpy(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_joint_positions(self, targets: np.ndarray) -> None:
        """Set position targets for all joints (arm + finger).

        The targets are stored internally and applied on the next
        ``step_sim()`` call.

        Args:
            targets: Array of shape ``(total_dofs,)`` in radians.
        """
        if len(targets) != len(self._all_indices):
            raise ValueError(
                f"Expected {len(self._all_indices)} joint targets, got {len(targets)}"
            )
        self._pending_targets = targets.copy()

    def write_arm_joint_positions(self, targets: np.ndarray) -> None:
        """Set position targets for arm joints only.

        Finger targets are preserved from the last call (or defaults to
        current positions if never set).

        Args:
            targets: Array of shape ``(arm_dofs,)`` in radians.
        """
        if len(targets) != len(self._arm_indices):
            raise ValueError(
                f"Expected {len(self._arm_indices)} arm targets, got {len(targets)}"
            )
        full = self._get_current_targets()
        full[: len(self._arm_indices)] = targets
        self._pending_targets = full

    def set_gripper(self, open: bool) -> None:
        """Open or close the gripper.

        Behavior depends on ``robot_cfg.gripper_type``:
        - ``parallel_jaw``: Sets both finger joints to open/close position.
        - ``claw``: Sets single revolute joint to open/close position.
        - ``suction``: No-op (logs warning).

        Args:
            open: True to open, False to close.
        """
        gripper_type = self.cfg.gripper_type

        if gripper_type == "suction":
            logger.warning(
                "Suction gripper control not implemented for %s — no-op",
                self.cfg.name,
            )
            return

        if not self.cfg.has_gripper_joints:
            logger.warning("Robot %s has no gripper joints", self.cfg.name)
            return

        pos_value = self.cfg.gripper_open_position if open else self.cfg.gripper_close_position

        full = self._get_current_targets()

        if gripper_type == "parallel_jaw":
            if self.cfg.gripper_mirrored:
                # Mirrored: set all finger joints to the same value
                for i, _ in enumerate(self._finger_indices):
                    full[len(self._arm_indices) + i] = float(pos_value)
            else:
                # Non-mirrored (e.g. OpenARM): pos_value applies to first finger,
                # second gets open/close independently
                if isinstance(pos_value, (list, tuple)):
                    for i, v in enumerate(pos_value[:len(self._finger_indices)]):
                        full[len(self._arm_indices) + i] = float(v)
                else:
                    for i, _ in enumerate(self._finger_indices):
                        full[len(self._arm_indices) + i] = float(pos_value)
        elif gripper_type == "claw":
            # Single revolute gripper joint
            full[len(self._arm_indices)] = float(pos_value)
        else:
            logger.warning("Unknown gripper type: %s", gripper_type)
            return

        self._pending_targets = full

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step_sim(self) -> dict:
        """Apply pending joint targets and step the environment.

        Constructs an action tensor from the pending targets and calls
        ``env.step(action)``.  If no targets have been set, the current
        joint positions are used (hold-position).

        Returns:
            Observation dict from ``env.step()``.
        """
        targets = self._get_current_targets()
        action = self._build_action_tensor(targets)
        obs, _, terminated, truncated, info = self.env.step(action)

        # Track termination state for success detection
        if terminated[self.env_idx].item():
            self._env_terminated = True
        if truncated[self.env_idx].item():
            self._env_truncated = True

        # Warn on unexpected episode reset (environment auto-resets on done)
        if terminated.any() or truncated.any():
            logger.warning(
                "Episode terminated/truncated during step_sim! "
                "Objects may have been reset. "
                f"terminated={terminated.cpu().numpy()}, truncated={truncated.cpu().numpy()}"
            )

        return obs

    @property
    def articulation(self):
        """Underlying IsaacLab articulation."""
        return self._articulation

    @property
    def arm_joint_indices(self) -> list[int]:
        """Resolved arm joint indices inside the articulation."""
        return list(self._arm_indices)

    @property
    def ee_body_name(self) -> str:
        """Resolved end-effector body name used for pose reads."""
        return self._ee_body_name

    @property
    def ee_body_index(self) -> int:
        """Resolved end-effector body index used for pose reads."""
        return self._ee_body_idx

    @property
    def env_terminated(self) -> bool:
        """Whether the environment fired a success/failure termination."""
        return self._env_terminated

    @property
    def env_truncated(self) -> bool:
        """Whether the environment timed out (truncation)."""
        return self._env_truncated

    def reset_termination_flags(self) -> None:
        """Reset termination tracking for a new episode."""
        self._env_terminated = False
        self._env_truncated = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_joint_indices(self, joint_names: list[str]) -> list[int]:
        """Map joint names to DOF indices in the articulation."""
        resolved_names: list[str] | None = None
        try:
            indices, _ = self._articulation.find_joints(joint_names)
            resolved_names = [self._articulation.joint_names[i] for i in indices]
        except ValueError:
            indices = []
            available = list(self._articulation.joint_names)
            available_to_index = {name: idx for idx, name in enumerate(available)}
            resolved_names = []
            missing: list[str] = []
            for name in joint_names:
                candidates = (name,) + _JOINT_NAME_ALIASES.get(name, ())
                matched_idx = next((available_to_index[c] for c in candidates if c in available_to_index), None)
                if matched_idx is None:
                    missing.append(name)
                else:
                    indices.append(matched_idx)
                    resolved_names.append(available[matched_idx])
            if missing:
                raise RuntimeError(
                    f"Could not find joints {missing} in articulation. "
                    f"Available: {self._articulation.joint_names}"
                ) from None
            logger.info("Resolved joint aliases for %s -> %s", joint_names, resolved_names)

        # find_joints returns (indices, names) — indices is a list[int]
        if len(indices) != len(joint_names):
            missing = set(joint_names) - set(resolved_names or [])
            raise RuntimeError(
                f"Could not find joints {missing} in articulation. "
                f"Available: {self._articulation.joint_names}"
            )
        return list(indices)

    def _get_current_targets(self) -> np.ndarray:
        """Return pending targets or fall back to current joint positions."""
        if self._pending_targets is not None:
            return self._pending_targets.copy()
        return self.read_joint_positions()

    def _build_action_tensor(self, joint_targets: np.ndarray) -> torch.Tensor:
        """Build the action tensor expected by ``env.step()``.

        The action manager may expect a tensor of size ``action_dim`` which
        can differ from the number of controlled joints (e.g. if the env
        uses a different action space parameterization).  We fill a
        zero-tensor and write joint targets at the corresponding indices.

        For standard joint-position action spaces the action_dim equals
        total_dofs and the mapping is direct.

        Special case: When action_dim < total_dofs (e.g. 8 vs 9 for Franka),
        the gripper uses BinaryJointPositionAction which expects sign-based
        encoding: positive = open, negative = close. The raw finger joint
        positions are converted to this convention.
        """
        action = torch.zeros(
            self.env.num_envs, self._action_dim, device=self._device
        )

        n_arm = len(self._arm_indices)

        # Write arm joint targets directly
        action[self.env_idx, :n_arm] = torch.tensor(
            joint_targets[:n_arm], dtype=torch.float32, device=self._device
        )

        # Handle gripper action dimension
        n_finger = len(self._finger_indices)
        if n_finger > 0 and self._action_dim > n_arm:
            gripper_dim = self._action_dim - n_arm  # typically 1 for binary
            if gripper_dim == 1 and n_finger >= 1:
                # Binary gripper action: BinaryJointPositionAction uses
                # sign convention: negative = close, positive/zero = open.
                finger_val = joint_targets[n_arm] if len(joint_targets) > n_arm else 0.0
                open_cfg = self.cfg.gripper_open_position
                close_cfg = self.cfg.gripper_close_position
                open_pos = float(open_cfg[0] if isinstance(open_cfg, (list, tuple)) else open_cfg)
                close_pos = float(close_cfg[0] if isinstance(close_cfg, (list, tuple)) else close_cfg)
                # Map: close_position → -1.0, open_position → +1.0
                if abs(finger_val - close_pos) < abs(finger_val - open_pos):
                    action[self.env_idx, n_arm] = -1.0  # close
                else:
                    action[self.env_idx, n_arm] = 1.0   # open
            else:
                # Multi-dim gripper: write directly
                n_g = min(gripper_dim, n_finger)
                action[self.env_idx, n_arm:n_arm + n_g] = torch.tensor(
                    joint_targets[n_arm:n_arm + n_g],
                    dtype=torch.float32, device=self._device,
                )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Action tensor built: arm=%s gripper=%s total_dim=%d",
                np.round(joint_targets[:n_arm], 4).tolist(),
                np.round(joint_targets[n_arm:], 4).tolist() if len(joint_targets) > n_arm else [],
                self._action_dim,
            )
        return action

    def expand_gripper_target_positions(
        self,
        target: float | list[float] | tuple[float, ...] | np.ndarray,
    ) -> np.ndarray:
        """Expand a gripper target into per-joint position targets."""
        if not self.cfg.has_gripper_joints:
            return np.zeros(0, dtype=np.float64)

        if isinstance(target, np.ndarray):
            target = target.tolist()

        if self.cfg.gripper_type == "parallel_jaw":
            if self.cfg.gripper_mirrored:
                scalar = float(target[0] if isinstance(target, (list, tuple)) else target)
                return np.full(len(self._finger_indices), scalar, dtype=np.float64)

            if isinstance(target, (list, tuple)):
                values = [float(v) for v in target[: len(self._finger_indices)]]
                if not values:
                    values = [0.0]
                while len(values) < len(self._finger_indices):
                    values.append(values[-1])
                return np.asarray(values, dtype=np.float64)

            return np.full(len(self._finger_indices), float(target), dtype=np.float64)

        scalar = float(target[0] if isinstance(target, (list, tuple)) else target)
        return np.asarray([scalar], dtype=np.float64)

    def _resolve_ee_body(self) -> tuple[str, int]:
        """Resolve the configured EE body once.

        Prefers ``tcp_frame`` when the robot profile provides it, and falls
        back to the legacy ``body`` name when the TCP frame is not a rigid body
        in the loaded articulation.
        """
        candidates = []
        if self.cfg.ee_frame_tcp:
            candidates.append(self.cfg.ee_frame_tcp)
        if self.cfg.ee_frame_body and self.cfg.ee_frame_body not in candidates:
            candidates.append(self.cfg.ee_frame_body)

        for body_name in candidates:
            body_indices, _ = self._articulation.find_bodies(body_name)
            if body_indices:
                return body_name, body_indices[0]

        raise RuntimeError(
            f"EE body not found. Tried {candidates}. "
            f"Available: {self._articulation.body_names}"
        )

    def world_pose_to_robot_pose(
        self,
        position_world: np.ndarray,
        quat_world_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a world-frame pose into the articulation root frame."""
        root_pos, root_quat = self.read_root_pose_world()
        root_rot = self._quat_to_rotation_matrix(root_quat)
        tcp_rot = self._quat_to_rotation_matrix(quat_world_wxyz)

        pos_robot = root_rot.T @ (np.asarray(position_world, dtype=np.float64) - root_pos)
        rot_robot = root_rot.T @ tcp_rot
        quat_robot = self._rotation_matrix_to_quat(rot_robot)
        return pos_robot, quat_robot

    def robot_pose_to_world_pose(
        self,
        position_robot: np.ndarray,
        quat_robot_wxyz: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform a robot-base-frame pose into simulator world frame."""
        root_pos, root_quat = self.read_root_pose_world()
        root_rot = self._quat_to_rotation_matrix(root_quat)
        tcp_rot_robot = self._quat_to_rotation_matrix(quat_robot_wxyz)

        pos_world = root_pos + root_rot @ np.asarray(position_robot, dtype=np.float64)
        rot_world = root_rot @ tcp_rot_robot
        quat_world = self._rotation_matrix_to_quat(rot_world)
        return pos_world, quat_world

    def _read_ee_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Read EE body frame from the articulation.

        Returns:
            (position [x,y,z], quaternion [w,x,y,z]) as numpy arrays.
        """
        # body_state_w shape: (num_envs, num_bodies, 13)
        # columns: pos(3), quat(4), lin_vel(3), ang_vel(3)
        body_state = self._articulation.data.body_state_w[self.env_idx, self._ee_body_idx]
        pos = body_state[:3].cpu().numpy()
        # IsaacLab body_state_w quaternion is already (w, x, y, z) format
        # See: ArticulationData.body_link_pose_w docstring
        quat_wxyz = body_state[3:7].cpu().numpy()
        # Some robots only expose a body-frame EE in sim while ADC/Pinocchio uses
        # a TCP offset from that body frame. Apply the same local offset here so
        # EE reads and IK targets stay in the same coordinate convention.
        if (
            self.cfg.ee_frame_offset_position
            and self._ee_body_name == self.cfg.ee_frame_body
            and not self.cfg.ee_frame_tcp
        ):
            offset_local = np.asarray(self.cfg.ee_frame_offset_position, dtype=np.float64)
            rot = self._quat_to_rotation_matrix(quat_wxyz)
            pos = pos + rot @ offset_local
        return pos, quat_wxyz

    @staticmethod
    def _quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotation_matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
        """Convert a 3x3 rotation matrix to quaternion [w, x, y, z]."""
        m = np.asarray(rotation, dtype=np.float64)
        trace = np.trace(m)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([w, x, y, z], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm
