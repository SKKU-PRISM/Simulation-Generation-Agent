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
        """Read end-effector position in world frame.

        Returns:
            Array ``[x, y, z]`` in meters.
        """
        pos, _ = self._read_ee_frame()
        return pos

    def read_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Read end-effector pose in world frame.

        Returns:
            Tuple of (position ``[x,y,z]``, quaternion ``[w,x,y,z]``).
        """
        return self._read_ee_frame()

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
        indices, _ = self._articulation.find_joints(joint_names)
        # find_joints returns (indices, names) — indices is a list[int]
        if len(indices) != len(joint_names):
            missing = set(joint_names) - set(
                self._articulation.joint_names[i] for i in indices
            )
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
                open_pos = float(self.cfg.gripper_open_position)
                close_pos = float(self.cfg.gripper_close_position)
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
        return action

    def _read_ee_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Read EE body frame from the articulation.

        Returns:
            (position [x,y,z], quaternion [w,x,y,z]) as numpy arrays.
        """
        body_name = self.cfg.ee_frame_body
        # Find the body index for the EE frame
        body_indices, _ = self._articulation.find_bodies(body_name)
        if not body_indices:
            raise RuntimeError(
                f"EE body '{body_name}' not found. "
                f"Available: {self._articulation.body_names}"
            )
        body_idx = body_indices[0]

        # body_state_w shape: (num_envs, num_bodies, 13)
        # columns: pos(3), quat(4), lin_vel(3), ang_vel(3)
        body_state = self._articulation.data.body_state_w[self.env_idx, body_idx]
        pos = body_state[:3].cpu().numpy()
        quat_xyzw = body_state[3:7].cpu().numpy()
        # IsaacLab uses xyzw internally — convert to wxyz
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return pos, quat_wxyz
