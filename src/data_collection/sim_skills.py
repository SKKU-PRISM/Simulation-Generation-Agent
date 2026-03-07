"""
Simulation skill primitives for multi-robot data collection.

Adapts AutoDataCollector's LeRobotSkills API for IsaacLab simulation.
Uses Pinocchio IK/FK (from ADC) when available, with IsaacLab DifferentialIK fallback.
Supports: Franka (7-DOF), OpenARM (7-DOF), UR10e (6-DOF), SO-101 (5-DOF).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from .config import RobotSimConfig

if TYPE_CHECKING:
    from .sim_camera import MultiCameraManager, SimCamera
    from .sim_detector import SimDetector
    from .sim_recorder import SimRecorder
    from .sim_robot_interface import SimRobotInterface

logger = logging.getLogger(__name__)

_CLOSED_GRIPPER_ANTI_DRIFT_GAIN = 5.0


# ============================================================
# Rotation matrix → RPY conversion (ZYX Euler, matches ADC convention)
# ============================================================


def _rotation_matrix_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Convert 3x3 rotation matrix to roll-pitch-yaw (ZYX Euler angles).

    Convention matches ADC ``_compute_ee_xyzrpy``:
    pitch = arcsin(-R[2,0]), with gimbal-lock guard.

    Returns:
        (roll, pitch, yaw) in radians.
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


# ============================================================
# Trajectory interpolation — import from ADC submodule with fallback
# Source: external/AutoDataCollector/src/lerobot_cap/planning/interpolation.py
# ============================================================

try:
    from .adc_imports import get_interpolation_module

    _interp = get_interpolation_module()
    smooth_linear_interpolation = _interp.smooth_linear_interpolation
    s_curve_interpolation = _interp.s_curve_interpolation
except ImportError:
    logger.debug("ADC interpolation not available, using inline fallback")

    def _smoothstep(t: float) -> float:
        return 3 * t**2 - 2 * t**3

    def smooth_linear_interpolation(
        start: np.ndarray, end: np.ndarray, num_points: int = 50
    ) -> np.ndarray:
        t_linear = np.linspace(0, 1, num_points)
        t_smooth = np.array([_smoothstep(t) for t in t_linear])
        return np.outer(1 - t_smooth, start) + np.outer(t_smooth, end)

    def s_curve_interpolation(
        start: np.ndarray, end: np.ndarray, num_points: int = 50, accel_ratio: float = 0.3
    ) -> np.ndarray:
        # Simplified fallback: use smoothstep
        return smooth_linear_interpolation(start, end, num_points)


# ============================================================
# Trajectory dataclass (adapted from ADC)
# ============================================================


@dataclass
class Trajectory:
    """Robot trajectory in joint space."""

    joint_positions: np.ndarray  # (num_points, arm_dofs) radians
    timestamps: np.ndarray  # (num_points,) seconds
    ee_positions: Optional[np.ndarray] = None  # (num_points, 3) meters
    ik_converged: bool = True
    expected_position_error: Optional[float] = None
    target_position: Optional[np.ndarray] = None

    @property
    def duration(self) -> float:
        return self.timestamps[-1] - self.timestamps[0]

    @property
    def num_points(self) -> int:
        return len(self.timestamps)

    def get_state_at_time(self, t: float) -> np.ndarray:
        """Get interpolated joint state at given time."""
        if t <= self.timestamps[0]:
            return self.joint_positions[0]
        if t >= self.timestamps[-1]:
            return self.joint_positions[-1]
        idx = np.searchsorted(self.timestamps, t)
        t0, t1 = self.timestamps[idx - 1], self.timestamps[idx]
        q0, q1 = self.joint_positions[idx - 1], self.joint_positions[idx]
        alpha = (t - t0) / (t1 - t0)
        return q0 + alpha * (q1 - q0)


# ============================================================
# IK Solver abstraction
# ============================================================


class IKSolver:
    """
    Abstract IK solver interface.

    Implementation: PinocchioIKSolver (ADC KinematicsEngine).
    """

    def solve_position(
        self,
        target_xyz: np.ndarray,
        current_joints: np.ndarray,
        fixed_joints: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve IK for target EE position (3-DOF, orientation free).

        Returns:
            (target_joints, success)
        """
        raise NotImplementedError

    def solve_pose(
        self,
        target_xyz: np.ndarray,
        target_rotation: np.ndarray,
        current_joints: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve IK for target EE pose (6-DOF, position + orientation).

        Args:
            target_xyz: Target position [x, y, z] meters.
            target_rotation: Target orientation as 3x3 rotation matrix.
            current_joints: Current arm joint positions.

        Returns:
            (target_joints, success)
        """
        raise NotImplementedError

    def solve_position_with_pitch(
        self,
        target_xyz: np.ndarray,
        target_pitch: float,
        current_joints: np.ndarray,
        fixed_joints: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, bool]:
        """
        Solve IK for target EE position + gripper pitch (4-DOF constraint).

        Ideal for 6-DOF robots: 4 constraints for 6 DOF leaves 2-DOF
        null-space freedom for self-motion optimization.

        Args:
            target_xyz: Target position [x, y, z] meters.
            target_pitch: Target gripper pitch in radians.
                          -π/2 = pointing down, 0 = horizontal.
            current_joints: Current arm joint positions.
            fixed_joints: Joint indices to keep fixed (optional).

        Returns:
            (target_joints, success)
        """
        raise NotImplementedError

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Compute EE position from joint angles. Returns [x,y,z]."""
        raise NotImplementedError

    def forward_kinematics_pose(
        self, joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute EE pose from joint angles in robot base frame.

        Returns:
            (position [3], rotation_matrix [3x3]) in robot base frame.
        """
        raise NotImplementedError

    def is_reachable(self, target_xyz: np.ndarray) -> bool:
        """Quick reachability check."""
        raise NotImplementedError


class PinocchioIKSolver(IKSolver):
    """
    Pinocchio-based IK solver using ADC's KinematicsEngine.

    Requires: pip install pin, URDF file for the robot.
    Reuses: AutoDataCollector/src/lerobot_cap/kinematics/engine.py
    """

    def __init__(self, urdf_path: str, ee_frame: str, joint_names: list[str]):
        try:
            import pinocchio  # noqa: F401
            from .adc_imports import get_kinematics_engine

            KinematicsEngine = get_kinematics_engine()
        except ImportError as e:
            raise ImportError(
                f"Pinocchio IK requires 'pinocchio' package and ADC submodule. "
                f"Run: git submodule update --init external/AutoDataCollector "
                f"Error: {e}"
            )

        self._engine = KinematicsEngine(
            urdf_path=urdf_path,
            end_effector_frame=ee_frame,
            joint_names=joint_names,
        )
        logger.info(f"PinocchioIKSolver initialized: URDF={urdf_path}, EE={ee_frame}")

    def solve_position(
        self,
        target_xyz: np.ndarray,
        current_joints: np.ndarray,
        fixed_joints: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, bool]:
        target_joints, success, _ = self._engine.inverse_kinematics_multi(
            target_xyz,
            current_joints=current_joints,
            fixed_joints=fixed_joints,
            num_random_samples=10,
        )
        return target_joints, success

    def solve_pose(
        self,
        target_xyz: np.ndarray,
        target_rotation: np.ndarray,
        current_joints: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Solve 6-DOF IK for target position + orientation.

        Args:
            target_xyz: Target EE position [x, y, z] meters.
            target_rotation: Target EE orientation as 3x3 rotation matrix.
            current_joints: Current arm joint positions.

        Returns:
            (target_joints, success)
        """
        target_joints, success, info = self._engine.inverse_kinematics_with_orientation_multi(
            target_position=target_xyz,
            target_orientation=target_rotation,
            current_joints=current_joints,
            num_random_samples=30,
            orientation_tolerance=0.15,  # ~8.6 degrees
        )
        if not success:
            logger.info(
                f"6-DOF IK: {info.get('num_valid', 0)} valid / "
                f"{info.get('num_solutions', 0)} converged / "
                f"{info.get('num_attempts', 0)} attempts"
            )
        return target_joints, success

    def solve_position_with_pitch(
        self,
        target_xyz: np.ndarray,
        target_pitch: float,
        current_joints: np.ndarray,
        fixed_joints: Optional[list[int]] = None,
    ) -> tuple[np.ndarray, bool]:
        target_joints, success, _ = self._engine.inverse_kinematics_multi(
            target_xyz,
            current_joints=current_joints,
            fixed_joints=fixed_joints,
            num_random_samples=15,
            target_pitch=target_pitch,
        )
        return target_joints, success

    def get_gripper_pitch(self, joints: np.ndarray) -> float:
        """Return gripper pitch (radians) from Pinocchio FK."""
        return self._engine.get_gripper_pitch(joints)

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        return self._engine.get_ee_position(joints)

    def forward_kinematics_pose(
        self, joints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (position [3], rotation_matrix [3x3]) in robot base frame."""
        return self._engine.forward_kinematics(joints)

    def is_reachable(self, target_xyz: np.ndarray) -> bool:
        return self._engine.is_position_reachable(target_xyz)


def create_ik_solver(
    robot_cfg: RobotSimConfig,
    urdf_path: Optional[str] = None,
) -> PinocchioIKSolver:
    """Create Pinocchio IK solver from robot config.

    Args:
        robot_cfg: Robot configuration (must have urdf_path).
        urdf_path: Override URDF path (optional).

    Raises:
        RuntimeError: If no URDF path available.
        ImportError: If pinocchio or ADC submodule not installed.
    """
    _urdf = urdf_path or robot_cfg.urdf_path
    if not _urdf:
        raise RuntimeError(
            f"No URDF path for robot '{robot_cfg.name}'. "
            "Add urdf_path to the robot profile YAML "
            "(e.g., urdf_path: '{{ISAACLAB_URDF_DIR}}/lula_franka_gen.urdf')."
        )
    return PinocchioIKSolver(
        urdf_path=_urdf,
        ee_frame=robot_cfg.ik_ee_frame or robot_cfg.ee_frame_body,
        joint_names=robot_cfg.arm_joint_names,
    )


# ============================================================
# SimSkills — Main skill class
# ============================================================


class SimSkills:
    """
    Skill-based manipulation primitives for IsaacLab simulation.

    Adapts ADC's LeRobotSkills for multi-robot sim execution.

    Args:
        robot_interface: SimRobotInterface for joint control
        robot_cfg: RobotSimConfig with robot specs
        detector: SimDetector for object positions
        recorder: SimRecorder for dataset recording (optional)
        camera: SimCamera for image capture during recording (optional)
        ik_solver: IKSolver instance (auto-created if None)
        interpolation_points: Points per trajectory segment
        use_s_curve: Use S-curve interpolation (jerk-limited)
    """

    def __init__(
        self,
        robot_interface: SimRobotInterface,
        robot_cfg: RobotSimConfig,
        detector: Optional[SimDetector] = None,
        recorder: Optional[SimRecorder] = None,
        camera: Optional[SimCamera] = None,
        cameras: Optional[MultiCameraManager] = None,
        ik_solver: Optional[IKSolver] = None,
        interpolation_points: int = 50,
        use_s_curve: bool = False,
        base_offset: Optional[np.ndarray] = None,
    ):
        self.robot = robot_interface
        self.cfg = robot_cfg
        # Robot base offset: IK solver works in base frame, sim works in world frame.
        # All IK targets must be converted: target_base = target_world - _base_offset
        self._base_offset = np.asarray(base_offset, dtype=np.float64) if base_offset is not None else np.zeros(3)
        if np.linalg.norm(self._base_offset) > 0.001:
            print(f"SimSkills: robot base offset = {self._base_offset} (IK targets will be converted)")
        self.detector = detector
        self.recorder = recorder
        # Multi-camera: prefer 'cameras' over legacy single 'camera'
        self.cameras = cameras
        self.camera = camera  # legacy single camera (backward compat)
        self.interpolation_points = interpolation_points
        self.use_s_curve = use_s_curve

        # Camera image capture decimation: only capture every Nth step
        # to avoid excessive render overhead (each capture triggers GPU render).
        self._image_capture_decimation = 5
        self._global_step_count = 0

        # IK solver
        if ik_solver is not None:
            self.ik = ik_solver
        else:
            self.ik = create_ik_solver(robot_cfg)

        # Velocity/acceleration limits for time parameterization
        vel_limits = [v for v in robot_cfg.velocity_limits.values() if v is not None]
        self._max_velocity = min(vel_limits) if vel_limits else 1.0
        self._max_acceleration = self._max_velocity * 2.0

        # Current skill tracking (for recording)
        self._current_skill_label = ""
        self._current_skill_type = ""
        self._skill_step_count = 0
        self._skill_total_steps = 0

        # Gripper state: when closed, contact forces create steady-state PD
        # errors on wrist joints.  IK strategy adapts accordingly.
        self._gripper_is_closed = False

        # Compute palm-down rotation from ready pose FK.
        # Each robot's EE frame convention differs (e.g., Franka panda_hand +Z =
        # tool direction, UR10e wrist_3_link +Y = tool direction). Using the
        # ready pose FK gives the correct grasp orientation for any robot.
        self._init_palm_down_rotation()

        # Orientation control strategy by DOF count:
        # - 6-DOF (UR10e): Lock all 3 wrist joints (idx 3,4,5).
        #   3 free DOF for 3D position. Orientation varies with position but is
        #   consistent for nearby positions. Grasp offset adapts to actual hand_z.
        # - 4-5 DOF: Lock wrist joints at ready-pose values.
        # - 7+ DOF (Franka, OpenARM): Full 6-DOF pose IK.
        self._palm_down_pitch = None  # Set for 6-DOF robots
        if self.cfg.arm_dofs == 6:
            self._orientation_lock_joints = [4, 5]  # lock wrist_2 + wrist_3 only; wrist_1 FREE for pitch control
            try:
                ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
                self._palm_down_pitch = self.ik.get_gripper_pitch(ready_joints)
                print(f"6-DOF palm-down pitch (Pinocchio): {np.degrees(self._palm_down_pitch):.1f}deg")
            except Exception as e:
                logger.warning(f"Palm-down pitch init failed: {e}, defaulting to -pi/2")
                self._palm_down_pitch = -np.pi / 2
        elif 4 <= self.cfg.arm_dofs <= 5:
            self._orientation_lock_joints = [
                self.cfg.arm_dofs - 3,
                self.cfg.arm_dofs - 2,
                self.cfg.arm_dofs - 1,
            ]
            logger.info(f"Orientation lock joints: {self._orientation_lock_joints}")
        else:
            self._orientation_lock_joints = []

    # ---- FK-based coordinate computation ----

    def _compute_goal_robot_xyzrpy(
        self, target_arm_joints: np.ndarray
    ) -> np.ndarray | None:
        """Compute EE pose in robot base frame via FK.

        Returns:
            [x, y, z, roll, pitch, yaw] float32, or None on failure.
        """
        try:
            pos, R = self.ik.forward_kinematics_pose(target_arm_joints)
            pos_world = pos + self._base_offset
            roll, pitch, yaw = _rotation_matrix_to_rpy(R)
            return np.array([*pos_world, roll, pitch, yaw], dtype=np.float32)
        except Exception:
            logger.debug("FK pose computation failed", exc_info=True)
            return None

    # ---- Position verification ----

    # Tolerance for EE position verification (meters)
    EE_POSITION_TOLERANCE = 0.03  # 3cm
    # Minimum Z-delta to consider object "lifted" after pick
    PICK_LIFT_THRESHOLD = 0.02  # 2cm
    # Tolerance for object at target position after place
    PLACE_POSITION_TOLERANCE = 0.05  # 5cm

    def _verify_ee_position(self, target_xyz: np.ndarray) -> tuple[bool, float]:
        """Verify EE reached target position after trajectory execution.

        Returns:
            (success, position_error_meters)
        """
        actual_ee = self.robot.read_ee_position()
        error = np.linalg.norm(actual_ee - target_xyz)
        success = error < self.EE_POSITION_TOLERANCE
        if not success:
            logger.warning(
                f"EE position error: {error:.4f}m (target={target_xyz}, actual={actual_ee})"
            )
        return success, float(error)

    def _verify_object_lifted(self, object_name: str, pre_z: float) -> tuple[bool, float]:
        """Verify object was lifted after pick (Z increased).

        Returns:
            (success, z_delta)
        """
        if self.detector is None:
            return True, 0.0
        try:
            post_pos = self.detector.get_object_position(object_name)
            z_delta = post_pos[2] - pre_z
            success = z_delta > self.PICK_LIFT_THRESHOLD
            if not success:
                logger.warning(
                    f"Object '{object_name}' not lifted: z_delta={z_delta:.4f}m "
                    f"(pre_z={pre_z:.4f}, post_z={post_pos[2]:.4f})"
                )
            return success, float(z_delta)
        except Exception as e:
            logger.warning(f"Cannot verify lift for '{object_name}': {e}")
            return True, 0.0  # assume success if can't verify

    def _verify_object_placed(
        self, object_name: str, target_xy: np.ndarray
    ) -> tuple[bool, float]:
        """Verify object is near target XY position after place.

        Returns:
            (success, xy_error)
        """
        if self.detector is None:
            return True, 0.0
        try:
            post_pos = self.detector.get_object_position(object_name)
            xy_error = np.linalg.norm(post_pos[:2] - target_xy[:2])
            success = xy_error < self.PLACE_POSITION_TOLERANCE
            if not success:
                logger.warning(
                    f"Object '{object_name}' placement error: {xy_error:.4f}m "
                    f"(target_xy={target_xy[:2]}, actual_xy={post_pos[:2]})"
                )
            return success, float(xy_error)
        except Exception as e:
            logger.warning(f"Cannot verify placement for '{object_name}': {e}")
            return True, 0.0

    # ---- Primitive Skills ----

    def move_to_ready(self, duration: float = 2.0, open_gripper: bool = True,
                      safe_retreat: bool = False) -> bool:
        """Move to robot's ready pose from profile.

        Args:
            duration: Movement duration in seconds.
            open_gripper: If True, opens gripper during move.
                Set False when holding an object (e.g. between pick and place).
            safe_retreat: If True, first lift the EE straight up to avoid
                colliding with nearby objects (e.g. a stacked tower).
        """
        if safe_retreat:
            # Lift EE above workspace first to avoid sweeping through objects
            ee_pos, _ = self.robot.read_ee_pose()
            retreat_pos = ee_pos.copy()
            retreat_pos[2] = max(retreat_pos[2] + 0.15, 0.45)  # at least 45cm above table
            self._move_palm_down(retreat_pos, duration=1.0)

        ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
        gripper_target = self.cfg.gripper_open_position if open_gripper else None
        return self._execute_joint_trajectory(
            target_joints=ready_joints,
            duration=duration,
            skill_label="move_to_ready",
            skill_type="move_initial",
            gripper_target=gripper_target,
        )

    # Default palm-down rotation (Franka panda_hand convention).
    # Overridden per-instance by _init_palm_down_rotation() using ready pose FK.
    PALM_DOWN_ROTATION = np.array([
        [-1.0, 0.0,  0.0],
        [ 0.0, 1.0,  0.0],
        [ 0.0, 0.0, -1.0],
    ])

    def _init_palm_down_rotation(self):
        """Compute palm-down rotation and URDF/USD frame offset.

        The URDF (used by Pinocchio IK) and USD (used by IsaacLab) may define
        the EE frame with different orientation conventions.  Joint positions
        match perfectly between the two, but the resulting EE *orientation*
        can differ by a constant rotation.

        At the ready pose (where Pinocchio FK and the live sim both agree on
        joint values), we measure both EE orientations and compute::

            R_offset = R_sim @ R_pinocchio.T

        Then for any world-frame target rotation R_target, we pass
        ``R_pinocchio = R_offset.T @ R_target`` to Pinocchio IK so that the
        joints it returns produce R_target in the simulation.

        Also sets ``self.PALM_DOWN_ROTATION`` to the *simulation's* EE
        orientation at ready pose (the world-frame palm-down target).
        """
        # R_local maps from Pinocchio EE frame to sim EE frame (local coords).
        # At any joint config q: R_sim(q) = R_pinocchio(q) @ R_local.
        # To achieve world-frame target R_target in sim:
        #   R_pinocchio_target = R_target @ R_local.T
        self._local_frame_offset = np.eye(3)  # default: no offset

        try:
            ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
            _, R_pinocchio = self.ik.forward_kinematics_pose(ready_joints)
            hand_z_pin = R_pinocchio[:, 2]
            print(f"Pinocchio FK at ready pose: hand_z={np.round(hand_z_pin, 3)}")

            # Query actual sim EE orientation at ready pose
            _, ee_quat = self.robot.read_ee_pose()
            R_sim = self._quat_to_rotation_matrix(ee_quat)
            hand_z_sim = R_sim[:, 2]
            print(f"Sim EE at ready pose:       hand_z={np.round(hand_z_sim, 3)}")

            # Local frame offset: R_local = R_pinocchio.T @ R_sim
            # This is the rotation from Pinocchio's EE frame to the sim's EE
            # frame, expressed in the Pinocchio EE frame (local coordinates).
            self._local_frame_offset = R_pinocchio.T @ R_sim

            # Rotation angle of the offset (for diagnostics)
            offset_angle = np.arccos(
                np.clip((np.trace(self._local_frame_offset) - 1) / 2, -1, 1)
            )
            print(f"URDF/USD frame offset: {np.degrees(offset_angle):.1f}°")

            if offset_angle > np.radians(5):
                # Significant mismatch — use compensation.
                # PALM_DOWN_ROTATION = the sim's EE orientation at ready pose
                # (the world-frame target for palm-down).
                self.PALM_DOWN_ROTATION = R_sim
                logger.info(
                    f"URDF/USD frame offset {np.degrees(offset_angle):.1f}° "
                    f"— orientation compensation active"
                )
            else:
                # Negligible offset — no compensation needed
                self._local_frame_offset = np.eye(3)
                self.PALM_DOWN_ROTATION = R_pinocchio
                print("  Negligible frame offset — no compensation needed")

        except Exception as e:
            logger.warning(f"Palm-down init failed: {e}, using defaults")
            print(f"Palm-down init failed: {e}")

    def move_to_position(
        self,
        target_xyz: np.ndarray,
        duration: Optional[float] = None,
        fixed_joints: Optional[list[int]] = None,
        lock_orientation: bool = False,
    ) -> bool:
        """
        Move end-effector to target [x,y,z] position via 3-DOF IK.

        Args:
            target_xyz: Target position in world frame [x, y, z] meters
            duration: Movement duration (auto if None)
            fixed_joints: Joint indices to keep fixed during IK
            lock_orientation: If True, fix orientation joints (wrist_1/wrist_2
                for 6-DOF robots) at ready-pose values to maintain palm-down.

        Returns:
            True if EE reached within EE_POSITION_TOLERANCE of target.
        """
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        # Convert world-frame target to robot base frame for IK
        target_base = target_xyz - self._base_offset
        current_arm_joints = self.robot.read_arm_joint_positions()

        # For 6-DOF robots: lock wrist orientation joints at ready-pose values
        # to prevent IK from choosing arbitrary orientations.
        if lock_orientation and self._orientation_lock_joints:
            if fixed_joints is None:
                fixed_joints = list(self._orientation_lock_joints)
            else:
                fixed_joints = list(fixed_joints) + list(self._orientation_lock_joints)
            ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
            for idx in self._orientation_lock_joints:
                current_arm_joints[idx] = ready_joints[idx]

        target_arm_joints, ik_success = self.ik.solve_position(
            target_base, current_arm_joints, fixed_joints=fixed_joints
        )

        if not ik_success:
            logger.warning(
                f"IK (3-DOF) failed for target {target_xyz}. "
                f"Attempting execution with best solution."
            )

        goal_xyz_world = self.ik.forward_kinematics(target_arm_joints) + self._base_offset
        fk_error = np.linalg.norm(goal_xyz_world - target_xyz)
        print(
            f"IK diagnostic: target={target_xyz}, FK={goal_xyz_world}, "
            f"fk_error={fk_error:.4f}m, ik_success={ik_success}"
        )

        self._execute_joint_trajectory(
            target_joints=target_arm_joints,
            duration=duration,
            skill_label=f"move_to [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
            skill_type="move",
        )

        actual_joints = self.robot.read_arm_joint_positions()
        joint_errors = target_arm_joints - actual_joints
        print(
            f"Joint convergence: max_err={np.max(np.abs(joint_errors)):.4f}rad, "
            f"per_joint_err={np.round(joint_errors, 4).tolist()}"
        )
        reached, error = self._verify_ee_position(target_xyz)
        return reached

    def move_to_position_with_pitch(
        self,
        target_xyz: np.ndarray,
        target_pitch: float,
        duration: Optional[float] = None,
    ) -> bool:
        """Move EE to target position while constraining gripper pitch.

        Uses 4-DOF IK (3D position + 1D pitch) — ideal for 6-DOF robots.
        Falls back to position-only IK if pitch-constrained IK is unavailable.

        Args:
            target_xyz: Target position [x, y, z] in world frame.
            target_pitch: Target pitch in Pinocchio frame (radians).
                          -π/2 = gripper pointing down.
            duration: Movement duration (auto if None).

        Returns:
            True if EE reached within EE_POSITION_TOLERANCE of target.
        """
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        # Convert world-frame target to robot base frame for IK
        target_base = target_xyz - self._base_offset
        current_arm_joints = self.robot.read_arm_joint_positions()

        # Lock wrist joints at ready-pose values to prevent flipping
        if self._orientation_lock_joints:
            ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
            for idx in self._orientation_lock_joints:
                current_arm_joints[idx] = ready_joints[idx]

        target_arm_joints, ik_success = self.ik.solve_position_with_pitch(
            target_base, target_pitch, current_arm_joints,
            fixed_joints=self._orientation_lock_joints,
        )

        if not ik_success:
            logger.warning(
                f"Pitch-constrained IK failed for target {target_xyz}, "
                f"pitch={np.degrees(target_pitch):.1f}deg. Using best solution."
            )

        # Diagnostics: position + achieved pitch
        goal_xyz_world = self.ik.forward_kinematics(target_arm_joints) + self._base_offset
        fk_error = np.linalg.norm(goal_xyz_world - target_xyz)
        achieved_pitch = self.ik.get_gripper_pitch(target_arm_joints)
        pitch_err = abs(np.degrees(achieved_pitch - target_pitch))
        print(
            f"IK pitch: target={target_xyz}, FK={goal_xyz_world}, "
            f"fk_err={fk_error:.4f}m, pitch={np.degrees(achieved_pitch):.1f}deg "
            f"(target={np.degrees(target_pitch):.1f}, err={pitch_err:.1f}deg), "
            f"ik_ok={ik_success}"
        )

        self._execute_joint_trajectory(
            target_joints=target_arm_joints,
            duration=duration,
            skill_label=f"move_to [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
            skill_type="move",
        )

        actual_joints = self.robot.read_arm_joint_positions()
        joint_errors = target_arm_joints - actual_joints
        print(
            f"Joint convergence: max_err={np.max(np.abs(joint_errors)):.4f}rad, "
            f"per_joint_err={np.round(joint_errors, 4).tolist()}"
        )
        reached, error = self._verify_ee_position(target_xyz)
        return reached

    # Tilted palm-down variants: small rotations around Y-axis (pitch).
    # Used as fallback when exact palm-down IK fails (joint limits / singularity).
    _TILT_ANGLES_DEG = [15, -15, 30, -30]

    @staticmethod
    def _rot_y(angle_rad: float) -> np.ndarray:
        """Rotation matrix around Y-axis."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def move_to_pose(
        self,
        target_xyz: np.ndarray,
        target_rotation: np.ndarray,
        duration: Optional[float] = None,
    ) -> bool:
        """
        Move end-effector to target [x,y,z] + orientation via 6-DOF IK.

        The ``target_rotation`` is in **world frame** (the orientation you want
        in the simulation).  If a URDF/USD frame offset was detected at init,
        it is automatically compensated so that Pinocchio IK produces the
        correct joint angles.

        If the exact orientation fails, tries tilted variants (±15°, ±30°)
        before falling back to 3-DOF position-only IK.

        Args:
            target_xyz: Target position in world frame [x, y, z] meters
            target_rotation: Target orientation as 3x3 rotation matrix (world frame)
            duration: Movement duration (auto if None)

        Returns:
            True if EE reached within tolerance of target position.
        """
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        # Convert world-frame target to robot base frame for IK
        target_base = target_xyz - self._base_offset
        current_arm_joints = self.robot.read_arm_joint_positions()

        # Transform target rotation from world/sim frame to Pinocchio frame.
        # R_sim(q) = R_pin(q) @ R_local, so to achieve R_target in sim:
        # R_pin_target = R_target @ R_local.T
        ik_rotation = target_rotation @ self._local_frame_offset.T

        # Try exact orientation first
        target_arm_joints, ik_success = self.ik.solve_pose(
            target_base, ik_rotation, current_arm_joints
        )

        # Fallback: try tilted orientation variants
        if not ik_success:
            for tilt_deg in self._TILT_ANGLES_DEG:
                tilt_rad = np.radians(tilt_deg)
                tilted_rot = ik_rotation @ self._rot_y(tilt_rad)
                target_arm_joints, ik_success = self.ik.solve_pose(
                    target_base, tilted_rot, current_arm_joints
                )
                if ik_success:
                    logger.info(f"6-DOF IK succeeded with {tilt_deg}° Y-tilt")
                    break

        # Last resort: 3-DOF position-only IK
        if not ik_success:
            logger.warning(
                f"IK (6-DOF + tilts) all failed for pos={target_xyz}. "
                f"Falling back to 3-DOF position-only IK."
            )
            target_arm_joints, ik_success = self.ik.solve_position(
                target_base, current_arm_joints
            )

        goal_xyz_world = self.ik.forward_kinematics(target_arm_joints) + self._base_offset
        fk_error = np.linalg.norm(goal_xyz_world - target_xyz)
        print(
            f"IK 6DOF: target={target_xyz}, FK={goal_xyz_world}, "
            f"fk_error={fk_error:.4f}m, ik_success={ik_success}"
        )

        self._execute_joint_trajectory(
            target_joints=target_arm_joints,
            duration=duration,
            skill_label=f"move_pose [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]",
            skill_type="move",
        )

        reached, error = self._verify_ee_position(target_xyz)
        return reached

    def gripper_open(self, duration: float = 0.5) -> bool:
        """Open gripper."""
        # FK for current arm pose → robot base frame
        current_arm = self.robot.read_arm_joint_positions()
        goal_robot_xyzrpy = self._compute_goal_robot_xyzrpy(current_arm)
        self._set_skill_meta(
            "gripper_open", "gripper",
            goal_robot_xyzrpy=goal_robot_xyzrpy, goal_gripper=0.0,
        )
        steps = max(1, int(duration * 20))  # 20Hz control

        for i in range(steps):
            self.robot.set_gripper(open=True)
            obs = self.robot.step_sim()
            self._record_step_if_active(
                progress=(i + 1) / steps,
                goal_robot_xyzrpy=goal_robot_xyzrpy, goal_gripper=0.0,
            )

        self._gripper_is_closed = False
        return True

    def gripper_close(self, duration: float = 0.5) -> bool:
        """Close gripper while actively holding arm position.

        Gripper contact forces (especially Robotiq 2F-85) can push wrist joints
        away from their targets. We explicitly re-command arm positions every
        step to maximise PD torque resistance.
        """
        # Snapshot arm targets BEFORE closing — these are the pre-grasp positions
        # that we want to hold steady throughout gripper closure.
        hold_arm = self.robot.read_arm_joint_positions().copy()
        goal_robot_xyzrpy = self._compute_goal_robot_xyzrpy(hold_arm)
        self._set_skill_meta(
            "gripper_close", "gripper",
            goal_robot_xyzrpy=goal_robot_xyzrpy, goal_gripper=1.0,
        )
        steps = max(1, int(duration * 20))

        # Anti-drift gain for orientation-lock joints (wrist_2/wrist_3).
        # Gripper contact forces push these joints, creating steady-state PD
        # error.  By mirroring the observed drift in the target (target =
        # hold - N*drift), we effectively multiply the PD gain by (N+1)
        # without changing the actuator stiffness.  N=2 → effective 3x gain
        # → steady-state drift reduced by factor of 3.
        for i in range(steps):
            actual_arm = self.robot.read_arm_joint_positions()
            corrected_arm = hold_arm.copy()
            # Anti-drift compensation on orientation-lock joints only
            if self._orientation_lock_joints:
                for idx in self._orientation_lock_joints:
                    drift = actual_arm[idx] - hold_arm[idx]
                    corrected_arm[idx] = hold_arm[idx] - _CLOSED_GRIPPER_ANTI_DRIFT_GAIN * drift
            self.robot.write_arm_joint_positions(corrected_arm)
            self.robot.set_gripper(open=False)
            obs = self.robot.step_sim()
            self._record_step_if_active(
                progress=(i + 1) / steps,
                goal_robot_xyzrpy=goal_robot_xyzrpy, goal_gripper=1.0,
            )

        # Extra settling with anti-drift compensation
        settle_steps = 40
        for _ in range(settle_steps):
            actual_arm = self.robot.read_arm_joint_positions()
            corrected_arm = hold_arm.copy()
            if self._orientation_lock_joints:
                for idx in self._orientation_lock_joints:
                    drift = actual_arm[idx] - hold_arm[idx]
                    corrected_arm[idx] = hold_arm[idx] - _CLOSED_GRIPPER_ANTI_DRIFT_GAIN * drift
            self.robot.write_arm_joint_positions(corrected_arm)
            self.robot.set_gripper(open=False)
            self.robot.step_sim()

        # Log joint convergence after gripper close + settling
        final_arm = self.robot.read_arm_joint_positions()
        joint_err = final_arm - hold_arm
        max_err_idx = np.argmax(np.abs(joint_err))
        print(f"  Joint convergence: max_err={joint_err[max_err_idx]:.4f}rad "
              f"(joint {max_err_idx}), per_joint_err={np.round(joint_err, 4)}")

        # Log final finger state
        if self.robot._finger_indices:
            finger_state = self.robot._articulation.data.joint_pos[
                self.robot.env_idx][self.robot._finger_indices].cpu().numpy()
            print(f"  gripper_close done: finger_state={np.round(finger_state, 4)}")

        self._gripper_is_closed = True
        return True

    # ---- Composite Skills ----

    def _move_palm_down(self, target_xyz, duration=None):
        """Move EE to target while maintaining palm-down orientation.

        Selects the appropriate IK strategy based on robot DOF:
        - 6 DOF: Position-only IK with wrist joints locked at ready-pose
          values. 3 free DOF for 3D position. Orientation depends on
          arm configuration — grasp formula adapts via hand_z.
        - 7+ DOF: Full 6-DOF pose IK with PALM_DOWN_ROTATION
        - 4-5 DOF: Position IK with wrist lock
        """
        if self.cfg.arm_dofs >= 7:
            return self.move_to_pose(target_xyz, self.PALM_DOWN_ROTATION, duration=duration)
        elif self.cfg.arm_dofs == 6 and self._palm_down_pitch is not None:
            # 4-DOF IK: 3D position + pitch constraint.
            # Joints 0-3 free (4 DOF), joints 4,5 locked at ready values.
            # Wrist_1 (joint 3) automatically compensates to maintain palm-down pitch.
            return self.move_to_position_with_pitch(
                target_xyz, self._palm_down_pitch, duration=duration
            )
        else:
            return self.move_to_position(
                target_xyz, duration=duration, lock_orientation=True
            )

    def _move_cartesian_steps(
        self,
        start_xyz: np.ndarray,
        end_xyz: np.ndarray,
        max_step: float = 0.03,
        fixed_joints: Optional[list[int]] = None,
    ) -> bool:
        """Move EE from start to end via intermediate Cartesian waypoints.

        Breaks large movements into small steps (default 3cm) so that each
        IK solve starts from joints close to the previous solution. This
        prevents joint discontinuities that cause wrist divergence on
        6-DOF robots with low-inertia wrist links.

        Args:
            start_xyz: Starting EE position (world frame).
            end_xyz: Target EE position (world frame).
            max_step: Maximum Cartesian step size in meters.
            fixed_joints: Optional arm-joint indices to keep fixed for each
                waypoint IK solve. Used for post-grasp lifts on 6-DOF arms.

        Returns:
            True if the final waypoint was reached within tolerance.
        """
        start_xyz = np.asarray(start_xyz, dtype=np.float64)
        end_xyz = np.asarray(end_xyz, dtype=np.float64)
        displacement = end_xyz - start_xyz
        distance = np.linalg.norm(displacement)

        if distance < max_step:
            # Small enough for a single move
            if fixed_joints is not None:
                return self.move_to_position(end_xyz, fixed_joints=fixed_joints)
            return self._move_palm_down(end_xyz)

        num_steps = max(2, int(np.ceil(distance / max_step)))
        reached = True
        for i in range(1, num_steps + 1):
            alpha = i / num_steps
            waypoint = start_xyz + alpha * displacement
            if fixed_joints is not None:
                ok = self.move_to_position(waypoint, fixed_joints=fixed_joints)
            else:
                ok = self._move_palm_down(waypoint)
            if not ok:
                # Check joint convergence — if wrist diverged badly, abort.
                # Skip this check when gripper is closed: wrist drift is
                # expected due to contact forces (steady-state PD error).
                actual_joints = self.robot.read_arm_joint_positions()
                ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
                if self._orientation_lock_joints and not self._gripper_is_closed:
                    for idx in self._orientation_lock_joints:
                        dev = abs(actual_joints[idx] - ready_joints[idx])
                        if dev > 0.5:
                            print(
                                f"  [ABORT] Wrist joint {idx} diverged by "
                                f"{dev:.3f}rad at step {i}/{num_steps}"
                            )
                            return False
                reached = False
        return reached

    @staticmethod
    def _quat_to_hand_z(quat_wxyz: np.ndarray) -> np.ndarray:
        """Compute the hand Z-axis (finger pointing direction) in world frame from quaternion."""
        w, x, y, z = quat_wxyz
        # Third column of rotation matrix = Z-axis of rotated frame
        return np.array([
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y),
        ])

    @staticmethod
    def _quat_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
        """Convert wxyz quaternion to 3x3 rotation matrix."""
        w, x, y, z = quat_wxyz
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ])

    def execute_pick(
        self,
        object_name: str,
        approach_offset: float = 0.10,
        grasp_offset: Optional[float] = None,
    ) -> bool:
        """
        Pick object by name using 6-DOF IK (position + palm-down orientation).

        Sequence: open gripper -> approach above -> descend -> close -> lift

        The 6-DOF IK constrains the hand to point straight down (palm-down),
        ensuring the finger tips align with the object and the ee_finger_offset
        maps directly to world-Z.

        Returns:
            True if object was successfully lifted (Z increased after pick).
        """
        if self.detector is None:
            raise RuntimeError("SimDetector required for execute_pick")

        if grasp_offset is None:
            grasp_offset = self.cfg.ee_finger_offset

        obj_pos = self.detector.get_object_position(object_name)
        pre_z = obj_pos[2]
        logger.info(f"Picking '{object_name}' at {obj_pos} (grasp_offset={grasp_offset:.3f})")

        self.gripper_open(duration=0.3)

        if self.cfg.arm_dofs == 6:
            # 6-DOF with pitch=-90° constraint: tool always points straight
            # down in the base frame.  Use VERTICAL grasp — EE directly above
            # object at ee_finger_offset height.
            #
            # Previously used measured hand_z (frame Z-axis) for the grasp
            # offset direction, but that produced tilted trajectories whose
            # lowest gripper link hit the table, causing collision forces that
            # destabilised the wrist PD controller.
            grasp_pos = obj_pos.copy()
            grasp_pos[2] += grasp_offset       # EE above object; fingertips at object surface

            approach_pos = grasp_pos.copy()
            approach_pos[2] += approach_offset  # Approach a bit higher

            reached = self._move_palm_down(approach_pos)
            if not reached:
                logger.warning(f"Failed to reach approach position for {object_name}")

            # Descend to grasp via Cartesian waypoints (1.5cm steps) to prevent
            # joint discontinuities on low-inertia wrist joints.
            self._move_cartesian_steps(approach_pos, grasp_pos, max_step=0.015)
        else:
            # 7+ DOF: full pose IK maintains palm-down → vertical approach
            grasp_pos = obj_pos.copy()
            grasp_pos[2] += grasp_offset
            approach_pos = grasp_pos.copy()
            approach_pos[2] += approach_offset

            reached = self._move_palm_down(approach_pos)
            if not reached:
                logger.warning(f"Failed to reach approach position for {object_name}")

            # Descend to grasp
            self._move_palm_down(grasp_pos)

        # Verify EE reached grasp position; recovery if not converged
        ee_pos, ee_quat = self.robot.read_ee_pose()
        grasp_error = np.linalg.norm(ee_pos - grasp_pos)
        if grasp_error > 0.015:
            logger.info(
                f"Grasp not converged ({grasp_error:.3f}m), re-solving IK..."
            )
            self._move_palm_down(grasp_pos)
            ee_pos, ee_quat = self.robot.read_ee_pose()
            grasp_error = np.linalg.norm(ee_pos - grasp_pos)

        hand_z = self._quat_to_hand_z(ee_quat)
        print(f"  PRE-GRASP '{object_name}': EE={np.round(ee_pos, 4)}, "
              f"hand_z={np.round(hand_z, 3)}, obj_z={obj_pos[2]:.4f}, "
              f"grasp_err={grasp_error:.4f}m")

        # Close gripper — use robot-specific duration from profile.
        # Franka (stiffness=2000 Nm/rad): 0.5s suffices.
        # UR10e Robotiq 2F-85 (stiffness=11.25 Nm/rad): needs ~2.0s for full closure.
        self.gripper_close(duration=self.cfg.gripper_close_duration)

        # Lift back to approach via Cartesian steps (6-DOF) or single move (7+)
        if self.cfg.arm_dofs == 6:
            ee_pos, _ = self.robot.read_ee_pose()
            self._move_cartesian_steps(
                ee_pos,
                approach_pos,
                max_step=0.015,
                fixed_joints=[3, 4, 5],
            )
        else:
            self._move_palm_down(approach_pos)

        # Verify object was lifted
        lifted, z_delta = self._verify_object_lifted(object_name, pre_z)
        if lifted:
            logger.info(f"Pick '{object_name}' SUCCESS (lifted {z_delta:.3f}m)")
        else:
            logger.warning(f"Pick '{object_name}' FAILED (z_delta={z_delta:.3f}m)")
        return lifted

    def execute_place(
        self,
        target_position: np.ndarray,
        approach_offset: float = 0.05,
        drop_offset: float = 0.005,
        _placed_object: Optional[str] = None,
    ) -> bool:
        """
        Place held object at target position using 6-DOF IK (palm-down).

        Sequence: approach above -> descend -> open gripper -> retreat up

        Returns:
            True if object landed near target XY (when _placed_object is given).
        """
        target_position = np.asarray(target_position, dtype=np.float64)
        finger_offset = self.cfg.ee_finger_offset

        if self.cfg.arm_dofs == 6:
            # 6-DOF with pitch=-90°: vertical placement.
            # EE directly above target at finger_offset + drop_offset height.
            place_pos = target_position.copy()
            place_pos[2] += drop_offset + finger_offset

            approach_pos = place_pos.copy()
            approach_pos[2] += approach_offset
        else:
            place_pos = target_position.copy()
            place_pos[2] += drop_offset + finger_offset
            approach_pos = place_pos.copy()
            approach_pos[2] += approach_offset

        # Move to approach
        self._move_palm_down(approach_pos)

        # Descend via Cartesian steps (6-DOF) or single move (7+)
        if self.cfg.arm_dofs == 6:
            self._move_cartesian_steps(approach_pos, place_pos, max_step=0.015)
        else:
            self._move_palm_down(place_pos)

        # Open gripper
        self.gripper_open(duration=0.3)

        # Retreat up via Cartesian steps (6-DOF)
        if self.cfg.arm_dofs == 6:
            ee_pos, _ = self.robot.read_ee_pose()
            self._move_cartesian_steps(ee_pos, approach_pos, max_step=0.015)
        else:
            self._move_palm_down(approach_pos)

        # Verify object placement if object name known
        if _placed_object and self.detector:
            placed_ok, xy_error = self._verify_object_placed(
                _placed_object, target_position
            )
            if placed_ok:
                logger.info(f"Place '{_placed_object}' SUCCESS (xy_error={xy_error:.3f}m)")
            else:
                logger.warning(f"Place '{_placed_object}' FAILED (xy_error={xy_error:.3f}m)")
            return placed_ok

        return True

    def execute_pick_and_place(
        self,
        pick_object: str,
        place_position: np.ndarray,
        approach_offset: float = 0.05,
        place_on_object: Optional[str] = None,
    ) -> bool:
        """Full pick-and-place sequence.

        Args:
            pick_object: Name of object to pick.
            place_position: Target position [x, y, z].
            approach_offset: Height above target for approach.
            place_on_object: If placing on top of another object, re-detect
                its position right before placing (accounts for objects shifted
                by previous manipulations).

        Returns:
            True if both pick and place succeeded verification.
        """
        pick_ok = self.execute_pick(pick_object, approach_offset=approach_offset)
        # Transit through ready pose while keeping gripper CLOSED (holding object)
        self.move_to_ready(duration=1.0, open_gripper=False)

        # Re-detect target object position right before placing
        if place_on_object and self.detector and pick_ok:
            try:
                updated_pos = self.detector.get_object_position(place_on_object)
                # Keep the original Z offset (stack height) but update XY
                height_offset = place_position[2] - 0.0  # approximate table level
                place_position = updated_pos.copy()
                place_position[2] += 0.05  # stack height above target object
                logger.info(
                    f"Re-detected '{place_on_object}' at {updated_pos}, "
                    f"updated place_position={place_position}"
                )
            except Exception as e:
                logger.warning(f"Failed to re-detect '{place_on_object}': {e}")

        place_ok = self.execute_place(
            place_position, approach_offset=approach_offset,
            _placed_object=pick_object,
        )
        success = pick_ok and place_ok
        if not success:
            logger.warning(
                f"Pick-and-place '{pick_object}': pick={'OK' if pick_ok else 'FAIL'}, "
                f"place={'OK' if place_ok else 'FAIL'}"
            )
        return success

    # ---- Internal Methods ----

    def _execute_joint_trajectory(
        self,
        target_joints: np.ndarray,
        duration: Optional[float] = None,
        skill_label: str = "",
        skill_type: str = "move",
        goal_world_xyzrpy: Optional[np.ndarray] = None,
        gripper_target: Optional[float] = None,
        goal_gripper: float = 0.0,
    ) -> bool:
        """
        Execute a joint-space trajectory from current to target.

        Uses smooth_linear or s_curve interpolation. Each waypoint is sent
        to the robot and the simulation is stepped.
        """
        current_joints = self.robot.read_arm_joint_positions()
        target_joints = np.asarray(target_joints, dtype=np.float64)

        # Compute goal_robot_xyzrpy via FK (robot base frame)
        goal_robot_xyzrpy = self._compute_goal_robot_xyzrpy(target_joints)

        # Interpolate
        if self.use_s_curve:
            trajectory = s_curve_interpolation(
                current_joints, target_joints, self.interpolation_points
            )
        else:
            trajectory = smooth_linear_interpolation(
                current_joints, target_joints, self.interpolation_points
            )

        # Compute duration if not specified
        if duration is None:
            max_joint_travel = np.max(np.abs(target_joints - current_joints))
            duration = max(0.5, max_joint_travel / self._max_velocity)

        timestamps = np.linspace(0, duration, self.interpolation_points)
        num_steps = len(trajectory)

        # Set skill metadata
        self._set_skill_meta(
            skill_label, skill_type,
            goal_joint=target_joints,
            goal_world_xyzrpy=goal_world_xyzrpy,
            goal_robot_xyzrpy=goal_robot_xyzrpy,
            goal_gripper=goal_gripper,
        )
        self._skill_total_steps = num_steps

        # Execute trajectory + settling steps
        settle_steps = 40  # extra steps holding final target for PD convergence
        total_steps = num_steps + settle_steps

        for i in range(total_steps):
            self._skill_step_count = min(i, num_steps - 1)

            # Use trajectory waypoint during interpolation, then hold final target
            waypoint = trajectory[min(i, num_steps - 1)]

            # Anti-drift compensation: when gripper is closed, contact forces
            # create steady-state PD error on wrist joints.  Mirror the drift
            # in the command to effectively increase PD gain on those joints.
            if self._gripper_is_closed and self._orientation_lock_joints:
                actual_arm = self.robot.read_arm_joint_positions()
                cmd = waypoint.copy()
                for idx in self._orientation_lock_joints:
                    drift = actual_arm[idx] - waypoint[idx]
                    cmd[idx] = waypoint[idx] - _CLOSED_GRIPPER_ANTI_DRIFT_GAIN * drift
                self.robot.write_arm_joint_positions(cmd)
            else:
                # Write arm joints
                self.robot.write_arm_joint_positions(waypoint)

            # Optionally set gripper
            if gripper_target is not None:
                if isinstance(gripper_target, (int, float)):
                    self.robot.set_gripper(open=(gripper_target > 0))

            # Step simulation
            obs = self.robot.step_sim()

            # Record if active (only during trajectory, not settling)
            if i < num_steps:
                self._record_step_if_active(
                    progress=(i + 1) / num_steps,
                    goal_joint=target_joints,
                    goal_world_xyzrpy=goal_world_xyzrpy,
                    goal_robot_xyzrpy=goal_robot_xyzrpy,
                    goal_gripper=goal_gripper,
                )

        return True

    def _pose_to_arm_array(self, pose_dict: dict[str, float]) -> np.ndarray:
        """Convert a pose dict {joint_name: value} to arm-only array."""
        arm_joints = np.zeros(self.cfg.arm_dofs)
        for i, name in enumerate(self.cfg.arm_joint_names):
            if name in pose_dict:
                arm_joints[i] = pose_dict[name]
        return arm_joints

    def _set_skill_meta(
        self,
        label: str,
        skill_type: str,
        goal_joint: Optional[np.ndarray] = None,
        goal_world_xyzrpy: Optional[np.ndarray] = None,
        goal_robot_xyzrpy: Optional[np.ndarray] = None,
        goal_gripper: float = 0.0,
    ):
        """Update current skill metadata for recording."""
        self._current_skill_label = label
        self._current_skill_type = skill_type
        self._current_goal_robot_xyzrpy = goal_robot_xyzrpy
        self._skill_step_count = 0
        self._skill_total_steps = 1

    def _record_step_if_active(
        self,
        progress: float = 0.0,
        goal_joint: Optional[np.ndarray] = None,
        goal_world_xyzrpy: Optional[np.ndarray] = None,
        goal_robot_xyzrpy: Optional[np.ndarray] = None,
        goal_gripper: float = 0.0,
    ):
        """Record a step if SimRecorder is active. Captures camera image.

        Camera images are captured every ``_image_capture_decimation`` steps
        to avoid excessive GPU render overhead. State/action are always recorded.
        """
        if self.recorder is None or not self.recorder.is_recording:
            self._global_step_count += 1
            return

        state = self.robot.read_joint_positions()
        action = self.robot.read_target_positions()  # pending targets = control command

        # Capture camera images with decimation (each capture triggers GPU render)
        images = None
        image = None
        should_capture = (self._global_step_count % self._image_capture_decimation == 0)
        if should_capture:
            if self.cameras is not None:
                try:
                    images = self.cameras.capture_all()
                except Exception as e:
                    logger.debug(f"Multi-camera capture failed: {e}")
            elif self.camera is not None:
                try:
                    image = self.camera.capture()
                except Exception as e:
                    logger.debug(f"Camera capture failed: {e}")

        self._global_step_count += 1

        self.recorder.record_step(
            state=state,
            action=action,
            image=image,
            images=images,
            skill_label=self._current_skill_label,
            skill_type=self._current_skill_type,
            skill_progress=progress,
            goal_joint=goal_joint,
            goal_world_xyzrpy=goal_world_xyzrpy,
            goal_robot_xyzrpy=goal_robot_xyzrpy,
            goal_gripper=goal_gripper,
        )
