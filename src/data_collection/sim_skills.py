"""
Simulation skill primitives for multi-robot data collection.

Adapts AutoDataCollector's LeRobotSkills API for IsaacLab simulation.
Uses Pinocchio IK/FK (from ADC) when available, with IsaacLab DifferentialIK fallback.
Supports: Franka (7-DOF), OpenARM (7-DOF), UR10 (6-DOF), SO-101 (5-DOF).
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

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        """Compute EE position from joint angles. Returns [x,y,z]."""
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

    def forward_kinematics(self, joints: np.ndarray) -> np.ndarray:
        return self._engine.get_ee_position(joints)

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
        ee_frame=robot_cfg.ee_frame_body,
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
    ):
        self.robot = robot_interface
        self.cfg = robot_cfg
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
            self.move_to_pose(retreat_pos, self.PALM_DOWN_ROTATION, duration=1.0)

        ready_joints = self._pose_to_arm_array(self.cfg.ready_pose)
        gripper_target = self.cfg.gripper_open_position if open_gripper else None
        return self._execute_joint_trajectory(
            target_joints=ready_joints,
            duration=duration,
            skill_label="move_to_ready",
            skill_type="move_initial",
            gripper_target=gripper_target,
        )

    # Palm-down rotation matrix: hand Z-axis points world -Z (straight down).
    # Franka panda_hand frame: Z = finger length direction, Y = finger opening.
    # Palm-down means hand X = world -X, hand Y = world Y, hand Z = world -Z.
    PALM_DOWN_ROTATION = np.array([
        [-1.0, 0.0,  0.0],
        [ 0.0, 1.0,  0.0],
        [ 0.0, 0.0, -1.0],
    ])

    def move_to_position(
        self,
        target_xyz: np.ndarray,
        duration: Optional[float] = None,
        fixed_joints: Optional[list[int]] = None,
    ) -> bool:
        """
        Move end-effector to target [x,y,z] position via 3-DOF IK.

        Args:
            target_xyz: Target position in world frame [x, y, z] meters
            duration: Movement duration (auto if None)
            fixed_joints: Joint indices to keep fixed during IK

        Returns:
            True if EE reached within EE_POSITION_TOLERANCE of target.
        """
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        current_arm_joints = self.robot.read_arm_joint_positions()

        target_arm_joints, ik_success = self.ik.solve_position(
            target_xyz, current_arm_joints, fixed_joints=fixed_joints
        )

        if not ik_success:
            logger.warning(
                f"IK (3-DOF) failed for target {target_xyz}. "
                f"Attempting execution with best solution."
            )

        goal_xyz = self.ik.forward_kinematics(target_arm_joints)
        fk_error = np.linalg.norm(goal_xyz - target_xyz)
        print(
            f"IK diagnostic: target={target_xyz}, FK={goal_xyz}, "
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

        If the exact orientation fails, tries tilted variants (±15°, ±30°)
        before falling back to 3-DOF position-only IK.

        Args:
            target_xyz: Target position in world frame [x, y, z] meters
            target_rotation: Target orientation as 3x3 rotation matrix
            duration: Movement duration (auto if None)

        Returns:
            True if EE reached within tolerance of target position.
        """
        target_xyz = np.asarray(target_xyz, dtype=np.float64)
        current_arm_joints = self.robot.read_arm_joint_positions()

        # Try exact orientation first
        target_arm_joints, ik_success = self.ik.solve_pose(
            target_xyz, target_rotation, current_arm_joints
        )

        # Fallback: try tilted orientation variants
        if not ik_success:
            for tilt_deg in self._TILT_ANGLES_DEG:
                tilt_rad = np.radians(tilt_deg)
                tilted_rot = target_rotation @ self._rot_y(tilt_rad)
                target_arm_joints, ik_success = self.ik.solve_pose(
                    target_xyz, tilted_rot, current_arm_joints
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
                target_xyz, current_arm_joints
            )

        goal_xyz = self.ik.forward_kinematics(target_arm_joints)
        fk_error = np.linalg.norm(goal_xyz - target_xyz)
        print(
            f"IK 6DOF: target={target_xyz}, FK={goal_xyz}, "
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
        self._set_skill_meta("gripper_open", "gripper", goal_gripper=0.0)
        steps = max(1, int(duration * 20))  # 20Hz control

        for i in range(steps):
            self.robot.set_gripper(open=True)
            obs = self.robot.step_sim()
            self._record_step_if_active(
                progress=(i + 1) / steps, goal_gripper=0.0
            )

        return True

    def gripper_close(self, duration: float = 0.5) -> bool:
        """Close gripper."""
        self._set_skill_meta("gripper_close", "gripper", goal_gripper=1.0)
        steps = max(1, int(duration * 20))

        for i in range(steps):
            self.robot.set_gripper(open=False)
            obs = self.robot.step_sim()
            self._record_step_if_active(
                progress=(i + 1) / steps, goal_gripper=1.0
            )

        # Log final finger state
        if self.robot._finger_indices:
            finger_state = self.robot._articulation.data.joint_pos[
                self.robot.env_idx][self.robot._finger_indices].cpu().numpy()
            print(f"  gripper_close done: finger_state={np.round(finger_state, 4)}")

        return True

    # ---- Composite Skills ----

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

    def execute_pick(
        self,
        object_name: str,
        approach_offset: float = 0.05,
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

        # Grasp position: EE directly above object, offset along world-Z
        # With palm-down orientation, hand Z = world -Z, so ee_finger_offset
        # maps directly to world Z distance.
        grasp_pos = obj_pos.copy()
        grasp_pos[2] += grasp_offset

        # Approach: higher above grasp
        approach_pos = grasp_pos.copy()
        approach_pos[2] += approach_offset

        # Use 6-DOF IK with palm-down orientation for all pick moves
        palm_down = self.PALM_DOWN_ROTATION

        self.gripper_open(duration=0.3)

        # Approach above object
        reached = self.move_to_pose(approach_pos, palm_down)
        if not reached:
            logger.warning(f"Failed to reach approach position for {object_name}")

        # Descend to grasp
        self.move_to_pose(grasp_pos, palm_down)

        # Verify EE reached grasp position; recovery if not converged
        ee_pos, ee_quat = self.robot.read_ee_pose()
        grasp_error = np.linalg.norm(ee_pos - grasp_pos)
        if grasp_error > 0.015:
            # EE didn't converge — re-solve IK from current actual joints
            # (the PD controller may track a different solution better)
            logger.info(
                f"Grasp not converged ({grasp_error:.3f}m), re-solving IK..."
            )
            self.move_to_pose(grasp_pos, palm_down)
            ee_pos, ee_quat = self.robot.read_ee_pose()
            grasp_error = np.linalg.norm(ee_pos - grasp_pos)

        hand_z = self._quat_to_hand_z(ee_quat)
        print(f"  PRE-GRASP '{object_name}': EE={np.round(ee_pos, 4)}, "
              f"hand_z={np.round(hand_z, 3)}, obj_z={obj_pos[2]:.4f}, "
              f"grasp_err={grasp_error:.4f}m")

        # Close gripper
        self.gripper_close(duration=0.5)

        # Lift back to approach
        self.move_to_pose(approach_pos, palm_down)

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

        # Apply ee_finger_offset: the target_position is where we want the
        # object to end up, but the EE frame is above the finger tips.
        finger_offset = self.cfg.ee_finger_offset
        place_pos = target_position.copy()
        place_pos[2] += drop_offset + finger_offset

        approach_pos = place_pos.copy()
        approach_pos[2] += approach_offset

        palm_down = self.PALM_DOWN_ROTATION

        # Move to approach
        self.move_to_pose(approach_pos, palm_down)

        # Descend
        self.move_to_pose(place_pos, palm_down)

        # Open gripper
        self.gripper_open(duration=0.3)

        # Retreat up
        self.move_to_pose(approach_pos, palm_down)

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
        goal_gripper: float = 0.0,
    ):
        """Update current skill metadata for recording."""
        self._current_skill_label = label
        self._current_skill_type = skill_type
        self._skill_step_count = 0
        self._skill_total_steps = 1

    def _record_step_if_active(
        self,
        progress: float = 0.0,
        goal_joint: Optional[np.ndarray] = None,
        goal_world_xyzrpy: Optional[np.ndarray] = None,
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
            goal_gripper=goal_gripper,
        )
