"""UR10e basic control and pick verification tests.

Runs 4 basic control checks plus an optional single-object pick diagnostic.

Usage:
  conda run -n env_isaaclab --no-capture-output \
    python3 tests/test_ur10e_basic_control.py --env-dir outputs/isaaclab/ur10estack_20260307_102720 \
      --headless --num_envs 1 --enable_cameras

  conda run -n env_isaaclab --no-capture-output \
    python3 tests/test_ur10e_basic_control.py --env-dir outputs/isaaclab/ur10estack_20260307_102720 \
      --only-pick-diagnostic --pick-object cube_2 --headless --num_envs 1 --enable_cameras
"""
import argparse
import math
import os
import sys
from pathlib import Path

import yaml

# === AppLauncher MUST be first (before any physics imports) ===
from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_env_dir() -> Path:
    candidates = sorted(
        (
            path for path in (PROJECT_ROOT / "outputs" / "isaaclab").glob("ur10estack_*")
            if (path / "env_cfg.py").exists()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return PROJECT_ROOT / "outputs" / "isaaclab" / "ur10estack_20260307_102720"
    return candidates[0]


parser = argparse.ArgumentParser()
parser.add_argument("--env-dir", default=str(_default_env_dir()), help="Path to generated IsaacLab env dir")
parser.add_argument(
    "--task-yaml",
    default=str(PROJECT_ROOT / "tasks" / "ur10e" / "stack" / "ur10e_stack.yaml"),
    help="Task YAML used for SimDetector-based diagnostics",
)
parser.add_argument("--pick-object", default="cube_2", help="Object name for pick diagnostic")
parser.add_argument(
    "--only-pick-diagnostic",
    action="store_true",
    help="Skip the 4 basic control tests and only run the pick diagnostic",
)
parser.add_argument(
    "--use-local-asset",
    action="store_true",
    help="Use the repo-local UR10e asset override instead of the default Isaac Sim asset",
)
parser.add_argument(
    "--skip-pad-collision-patch",
    action="store_true",
    help="Do not inject the UR10e inner-finger pad collision cubes",
)
parser.add_argument(
    "--gripper-close-mag",
    type=float,
    default=None,
    help="Override the absolute magnitude used by UR10e BinaryJointPositionAction close_command_expr",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not hasattr(args, 'headless') or args.headless is None:
    args.headless = True
if not hasattr(args, 'num_envs') or args.num_envs is None:
    args.num_envs = 1
args.enable_cameras = True
args.rendering_mode = "performance"
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# === Now safe to import physics modules ===
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv

# Add project paths
ENV_DIR = Path(args.env_dir).resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ENV_DIR))

from env_cfg import *  # noqa: F403
from src.data_collection.ur10e_patches import (
    apply_local_ur10e_asset_override,
    ensure_ur10e_pad_collisions,
    summarize_ur10e_pad_alignment,
    summarize_ur10e_pad_alignment_from_stage,
)

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #
EXPECTED_ARM_JOINTS = np.array([-0.4184, -1.9950, 2.3950, -1.9708, -1.5708, 0.0])
EXPECTED_BASE_POS = np.array([0.0, 0.0, 0.0])
EXPECTED_CUBE_POSITIONS = {
    "cube_1": np.array([0.4, 0.0, 0.0203]),
    "cube_2": np.array([0.55, 0.05, 0.0203]),
    "cube_3": np.array([0.60, -0.1, 0.0203]),
}
ARM_JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def build_action(env, arm_targets, gripper_open=True):
    """Build a (1, action_dim) action tensor.

    arm_targets: np.array of shape (6,) — absolute joint positions in radians.
    gripper_open: True → +1, False → -1.
    """
    action_dim = env.action_manager.total_action_dim
    action = torch.zeros(env.num_envs, action_dim, device=env.device)
    action[0, :6] = torch.tensor(arm_targets, dtype=torch.float32, device=env.device)
    if action_dim > 6:
        action[0, 6] = 1.0 if gripper_open else -1.0
    return action


def step_n(env, action, n):
    """Apply the same action for n steps."""
    for _ in range(n):
        obs, _, terminated, truncated, info = env.step(action)
    return obs


def print_header(test_num, title):
    print(f"\n{'='*60}")
    print(f"  Test {test_num}: {title}")
    print(f"{'='*60}")


def check(label, actual, expected, tol, unit=""):
    err = np.abs(actual - expected)
    max_err = np.max(err) if hasattr(err, '__len__') else err
    ok = max_err < tol
    status = "PASS" if ok else "FAIL"
    if hasattr(actual, '__len__'):
        print(f"  [{status}] {label}: max_err={max_err:.6f}{unit} (tol={tol}{unit})")
        print(f"         actual:   {np.round(actual, 4).tolist()}")
        print(f"         expected: {np.round(expected, 4).tolist()}")
        if not ok:
            print(f"         errors:   {np.round(err, 4).tolist()}")
    else:
        print(f"  [{status}] {label}: err={max_err:.6f}{unit} (tol={tol}{unit})")
    return ok


# --------------------------------------------------------------------------- #
#  Apply same patches as pipeline.py
# --------------------------------------------------------------------------- #
def apply_data_collection_patches(env_cfg):
    """Apply the same runtime patches that pipeline.py applies."""
    # Episode length
    env_cfg.episode_length_s = 600.0

    # Disable non-timeout terminations
    if hasattr(env_cfg, 'terminations'):
        for term_name in list(vars(env_cfg.terminations)):
            if term_name.startswith('_'):
                continue
            term = getattr(env_cfg.terminations, term_name, None)
            if term is None:
                continue
            if hasattr(term, 'time_out') and term.time_out:
                continue
            setattr(env_cfg.terminations, term_name, None)

    # Initial joint state: task-ready pose from robot profile
    if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'init_state'):
        ready_pose = {
            "shoulder_pan_joint": -0.4184, "shoulder_lift_joint": -1.9950,
            "elbow_joint": 2.3950, "wrist_1_joint": -1.9708,
            "wrist_2_joint": -1.5708, "wrist_3_joint": 0.0,
            "finger_joint": 0.0, "right_outer_knuckle_joint": 0.0,
            "left_inner_finger_joint": 0.0, "right_inner_finger_joint": 0.0,
            "left_inner_finger_knuckle_joint": 0.0, "right_inner_finger_knuckle_joint": 0.0,
        }
        env_cfg.scene.robot.init_state.joint_pos = ready_pose
        print(f"Patched initial joint_pos to task-ready pose")

    # Action space: absolute joint control
    if hasattr(env_cfg, 'actions') and hasattr(env_cfg.actions, 'arm_action'):
        env_cfg.actions.arm_action.scale = 1.0
        env_cfg.actions.arm_action.use_default_offset = False
        print("Patched arm_action: scale=1.0, use_default_offset=False")

    if (
        args.gripper_close_mag is not None
        and hasattr(env_cfg, "actions")
        and hasattr(env_cfg.actions, "gripper_action")
        and hasattr(env_cfg.actions.gripper_action, "close_command_expr")
    ):
        close_expr = getattr(env_cfg.actions.gripper_action, "close_command_expr", None) or {}
        patched_close_expr = {}
        for joint_name, value in close_expr.items():
            sign = 1.0 if float(value) >= 0.0 else -1.0
            patched_close_expr[joint_name] = sign * abs(args.gripper_close_mag)
        env_cfg.actions.gripper_action.close_command_expr = patched_close_expr
        print(f"Patched gripper close magnitude: {args.gripper_close_mag}")

    # Actuator PD gains
    if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'actuators'):
        for act_name, actuator in env_cfg.scene.robot.actuators.items():
            if 'hand' not in act_name and 'finger' not in act_name and 'gripper' not in act_name:
                orig_s = getattr(actuator, 'stiffness', 80.0)
                orig_d = getattr(actuator, 'damping', 4.0)
                orig_s = float(orig_s) if isinstance(orig_s, (int, float)) else 80.0
                orig_d = float(orig_d) if isinstance(orig_d, (int, float)) else 4.0
                STIFFNESS_MULT = 5.0
                MAX_STIFFNESS = 5000.0
                MAX_DAMPING = 400.0
                new_s = min(orig_s * STIFFNESS_MULT, MAX_STIFFNESS)
                new_d = min(orig_d * STIFFNESS_MULT, MAX_DAMPING)
                if act_name == "wrist":
                    new_s = max(new_s, 2000.0)
                    new_d = max(new_d, 200.0)
                actuator.stiffness = new_s
                actuator.damping = new_d
                actuator.effort_limit = 1e9
                actuator.effort_limit_sim = 1e9
                print(f"  {act_name}: stiffness={orig_s}->{new_s}, "
                      f"damping={orig_d}->{new_d}, effort_limit=1e9")
        print("Patched arm actuators: proportional PD (×5, cap 5000), effort_limit=1e9")

        # Gripper patches
        grasp_stiffness = 600.0
        if grasp_stiffness > 0:
            for act_name, actuator in env_cfg.scene.robot.actuators.items():
                if 'gripper_drive' in act_name or 'gripper_finger' in act_name:
                    actuator.stiffness = grasp_stiffness
                    actuator.damping = grasp_stiffness * 0.05
                    actuator.effort_limit_sim = max(100.0, grasp_stiffness * 0.2)
                elif 'gripper_passive' in act_name or 'passive' in act_name:
                    actuator.stiffness = grasp_stiffness * 0.5
                    actuator.damping = grasp_stiffness * 0.025

    # Disable gravity
    if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'spawn'):
        if hasattr(env_cfg.scene.robot.spawn, 'rigid_props') and env_cfg.scene.robot.spawn.rigid_props is not None:
            env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
            print("Patched: disable_gravity=True")


def add_robotiq_pad_collisions(env):
    """Mirror the UR10e data-collection finger-pad collision patch."""
    try:
        added = ensure_ur10e_pad_collisions(env.sim.stage, env_idx=0, verbose=True)
        print(f"Added {len(added)} Robotiq pad collision shapes")
    except Exception as exc:
        import traceback

        print(f"WARNING: failed to add Robotiq pad collisions: {exc}")
        traceback.print_exc()


def print_pad_alignment(env, object_name: str, object_pos: np.ndarray, label: str) -> None:
    """Print left/right pad centers and their midpoint relative to the object."""
    alignment = summarize_ur10e_pad_alignment(env.scene["robot"], object_pos, env_idx=0)
    print(
        f"  {label}: "
        f"left={np.round(alignment['left'], 4).tolist()} "
        f"right={np.round(alignment['right'], 4).tolist()} "
        f"mid={np.round(alignment['midpoint'], 4).tolist()} "
        f"mid-obj={np.round(alignment['midpoint_delta'], 4).tolist()} "
        f"sep={alignment['separation']:.4f}m"
    )


# --------------------------------------------------------------------------- #
#  Tests
# --------------------------------------------------------------------------- #
def test_1_initial_state(env):
    """Verify initial joint positions, base pos, EE, cubes, actuators."""
    print_header(1, "Initial State Verification")

    art = env.scene["robot"]
    all_pass = True

    # --- Joint positions ---
    joint_pos = art.data.joint_pos[0].cpu().numpy()
    # Find arm joint indices
    arm_indices = []
    for name in ARM_JOINT_NAMES:
        idx_list, _ = art.find_joints(name)
        arm_indices.append(idx_list[0])
    arm_pos = joint_pos[arm_indices]
    all_pass &= check("Arm joint positions (rad)", arm_pos, EXPECTED_ARM_JOINTS, 0.01, "rad")

    # --- Robot base position ---
    base_pos = art.data.root_pos_w[0].cpu().numpy()
    all_pass &= check("Robot base position", base_pos, EXPECTED_BASE_POS, 0.05, "m")

    # --- EE body pose ---
    body_indices, _ = art.find_bodies("wrist_3_link")
    if body_indices:
        ee_state = art.data.body_state_w[0, body_indices[0]]
        ee_pos = ee_state[:3].cpu().numpy()
        ee_quat_xyzw = ee_state[3:7].cpu().numpy()
        print(f"  [INFO] EE position (world): {np.round(ee_pos, 4).tolist()}")
        print(f"  [INFO] EE quaternion (xyzw): {np.round(ee_quat_xyzw, 4).tolist()}")
        # hand_z computation (from quaternion)
        x, y, z, w = ee_quat_xyzw  # IsaacLab uses xyzw internally
        hand_z = np.array([
            2 * (x * z + y * w),
            2 * (y * z - x * w),
            1 - 2 * (x * x + y * y),
        ])
        print(f"  [INFO] hand_z (finger direction): {np.round(hand_z, 4).tolist()}")
    else:
        print("  [FAIL] wrist_3_link not found!")
        all_pass = False

    # --- Cube positions ---
    for cube_name, expected_pos in EXPECTED_CUBE_POSITIONS.items():
        try:
            cube = env.scene[cube_name]
            actual_pos = cube.data.root_pos_w[0].cpu().numpy()
            # Cubes have randomization, so use relaxed tolerance
            all_pass &= check(f"{cube_name} position", actual_pos, expected_pos, 0.15, "m")
        except Exception as e:
            print(f"  [FAIL] {cube_name}: {e}")
            all_pass = False

    # --- Actuator properties ---
    print(f"\n  Actuator verification:")
    print(f"    Joint names:         {art.joint_names}")
    print(f"    Num joints:          {art.num_joints}")
    stiffness = art.data.joint_stiffness[0].cpu().numpy()
    damping = art.data.joint_damping[0].cpu().numpy()
    effort_limits = art.data.joint_effort_limits[0].cpu().numpy()
    for i, name in enumerate(art.joint_names):
        marker = " (ARM)" if any(n in name for n in ["shoulder", "elbow", "wrist_"]) else ""
        print(f"    [{i:2d}] {name:40s}: stiffness={stiffness[i]:8.1f}, "
              f"damping={damping[i]:8.3f}, effort_limit={effort_limits[i]:10.1f}{marker}")

    # --- Action space ---
    action_dim = env.action_manager.total_action_dim
    print(f"\n  Action dim: {action_dim} (expected: 7 = 6 arm + 1 binary gripper)")
    all_pass &= (action_dim == 7)
    if action_dim != 7:
        print(f"  [FAIL] Action dim mismatch! Got {action_dim}")

    return all_pass


def test_2_hold_position(env):
    """Hold current position for 100 steps, measure drift."""
    print_header(2, "Hold Position (100 steps)")

    art = env.scene["robot"]

    # Read initial joint positions
    arm_indices = []
    for name in ARM_JOINT_NAMES:
        idx_list, _ = art.find_joints(name)
        arm_indices.append(idx_list[0])

    initial_joints = art.data.joint_pos[0, arm_indices].cpu().numpy()
    print(f"  Initial arm joints: {np.round(initial_joints, 4).tolist()}")

    # Hold position: send current joints as action for 100 steps
    action = build_action(env, initial_joints, gripper_open=True)
    step_n(env, action, 100)

    # Read final joint positions
    final_joints = art.data.joint_pos[0, arm_indices].cpu().numpy()
    drift = final_joints - initial_joints
    max_drift = np.max(np.abs(drift))

    print(f"  Final arm joints:   {np.round(final_joints, 4).tolist()}")
    print(f"  Drift per joint:    {np.round(drift, 6).tolist()}")
    print(f"  Max drift:          {max_drift:.6f} rad")

    ok = max_drift < 0.01
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Max drift = {max_drift:.6f} rad (tolerance: 0.01 rad)")
    return ok


def test_3_single_joint(env):
    """Move shoulder_lift from -π/2 to -π/4, verify tracking."""
    print_header(3, "Single Joint Movement (shoulder_lift)")

    art = env.scene["robot"]

    # Reset env first
    env.reset()
    # Warmup
    action = build_action(env, EXPECTED_ARM_JOINTS, gripper_open=True)
    step_n(env, action, 20)

    arm_indices = []
    for name in ARM_JOINT_NAMES:
        idx_list, _ = art.find_joints(name)
        arm_indices.append(idx_list[0])

    # Read current
    current = art.data.joint_pos[0, arm_indices].cpu().numpy()
    print(f"  Before: {np.round(current, 4).tolist()}")

    # Target: change shoulder_lift (index 1) by +0.5 rad
    target = current.copy()
    target[1] = current[1] + 0.5

    print(f"  Target: {np.round(target, 4).tolist()}")
    print(f"  (shoulder_lift: {current[1]:.4f} → {target[1]:.4f}, delta={target[1]-current[1]:.4f} rad)")

    # Apply 50 trajectory steps + 40 settling
    action = build_action(env, target, gripper_open=True)

    # Log intermediate steps
    for i in range(90):
        obs, _, terminated, truncated, info = env.step(action)
        if i % 20 == 19:
            mid = art.data.joint_pos[0, arm_indices].cpu().numpy()
            err = np.abs(mid - target)
            print(f"    step {i+1:3d}: joints={np.round(mid, 4).tolist()}, max_err={np.max(err):.4f}")

    # Final check
    final = art.data.joint_pos[0, arm_indices].cpu().numpy()
    errors = np.abs(final - target)
    max_err = np.max(errors)

    print(f"  After (90 steps): {np.round(final, 4).tolist()}")
    print(f"  Errors:           {np.round(errors, 4).tolist()}")

    ok = max_err < 0.05
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Max tracking error = {max_err:.4f} rad (tolerance: 0.05 rad)")

    # Check if other joints stayed put
    other_drift = np.abs(final - current)
    other_drift[1] = 0  # exclude the moved joint
    max_other = np.max(other_drift)
    drift_ok = max_other < 0.05
    drift_status = "PASS" if drift_ok else "FAIL"
    print(f"  [{drift_status}] Other joints max drift = {max_other:.4f} rad (tolerance: 0.05 rad)")

    return ok and drift_ok


def test_4_ik_to_ee(env):
    """Use Pinocchio IK → action → verify EE reaches target position."""
    print_header(4, "IK → Action → EE Position")

    art = env.scene["robot"]

    # Reset env
    env.reset()
    action = build_action(env, EXPECTED_ARM_JOINTS, gripper_open=True)
    step_n(env, action, 20)

    arm_indices = []
    for name in ARM_JOINT_NAMES:
        idx_list, _ = art.find_joints(name)
        arm_indices.append(idx_list[0])

    # Read base position
    base_pos = art.data.root_pos_w[0].cpu().numpy()
    print(f"  Robot base (world): {np.round(base_pos, 4).tolist()}")

    # Read current EE position
    body_indices, _ = art.find_bodies("wrist_3_link")
    ee_state = art.data.body_state_w[0, body_indices[0]]
    ee_pos_before = ee_state[:3].cpu().numpy()
    print(f"  EE before (world):  {np.round(ee_pos_before, 4).tolist()}")

    # Try to create Pinocchio IK solver
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.data_collection.sim_skills import create_ik_solver
        from src.data_collection.config import load_robot_config

        robot_cfg = load_robot_config("ur10e")
        ik_solver = create_ik_solver(robot_cfg)
        print(f"  IK solver: {type(ik_solver).__name__}")

        # Target: 0.4m in front, 0.3m above base (robot at Z=0, so world Z=0.3)
        target_world = np.array([0.4, 0.0, 0.3])  # world frame
        target_base = target_world - base_pos       # robot base frame
        print(f"  IK target (world): {np.round(target_world, 4).tolist()}")
        print(f"  IK target (base):  {np.round(target_base, 4).tolist()}")

        current_joints = art.data.joint_pos[0, arm_indices].cpu().numpy()

        # Solve IK
        ik_joints, ik_success = ik_solver.solve_position(target_base, current_joints)
        print(f"  IK success: {ik_success}")
        print(f"  IK joints:  {np.round(ik_joints, 4).tolist()}")

        if not ik_success:
            print("  [FAIL] IK solver could not find solution")
            return False

        # FK verification
        fk_pos = ik_solver.forward_kinematics(ik_joints)
        fk_world = fk_pos + base_pos
        fk_err = np.linalg.norm(fk_world - target_world)
        print(f"  FK verify (world): {np.round(fk_world, 4).tolist()} (err={fk_err:.4f}m)")

        # Apply IK joints as action
        action = build_action(env, ik_joints, gripper_open=True)

        for i in range(90):
            obs, _, terminated, truncated, info = env.step(action)
            if i % 20 == 19:
                mid = art.data.joint_pos[0, arm_indices].cpu().numpy()
                err = np.abs(mid - ik_joints)
                ee_mid = art.data.body_state_w[0, body_indices[0], :3].cpu().numpy()
                pos_err = np.linalg.norm(ee_mid - target_world)
                print(f"    step {i+1:3d}: joint_max_err={np.max(err):.4f}, ee_pos_err={pos_err:.4f}m")

        # Final check
        ee_state_after = art.data.body_state_w[0, body_indices[0]]
        ee_pos_after = ee_state_after[:3].cpu().numpy()
        final_joints = art.data.joint_pos[0, arm_indices].cpu().numpy()

        ee_error = np.linalg.norm(ee_pos_after - target_world)
        joint_errors = np.abs(final_joints - ik_joints)
        max_joint_err = np.max(joint_errors)

        print(f"\n  EE after (world):   {np.round(ee_pos_after, 4).tolist()}")
        print(f"  EE target (world):  {np.round(target_world, 4).tolist()}")
        print(f"  Joint errors:       {np.round(joint_errors, 4).tolist()}")

        ee_ok = check("EE position error", ee_error, 0.0, 0.03, "m")
        joint_ok = check("Joint tracking error", max_joint_err, 0.0, 0.05, "rad")

        return ee_ok and joint_ok

    except Exception as e:
        import traceback
        print(f"  [FAIL] IK test error: {e}")
        traceback.print_exc()
        return False


def test_5_pick_diagnostic(env):
    """Run a single-object pick with SimSkills and report lift/contact behavior."""
    print_header(5, f"Single Pick Diagnostic ({args.pick_object})")

    try:
        from src.data_collection.config import load_robot_config
        from src.data_collection.sim_detector import SimDetector
        from src.data_collection.sim_robot_interface import SimRobotInterface
        from src.data_collection.sim_skills import SimSkills, create_ik_solver

        with open(args.task_yaml) as f:
            task_doc = yaml.safe_load(f)

        robot_cfg = load_robot_config("ur10e")
        robot_interface = SimRobotInterface(env, robot_cfg, env_idx=0)
        detector = SimDetector(env, task_doc, env_idx=0)
        ik_solver = create_ik_solver(robot_cfg)
        robot_base_pos = env.scene["robot"].data.root_pos_w[0].cpu().numpy()
        skills = SimSkills(
            robot_interface=robot_interface,
            robot_cfg=robot_cfg,
            detector=detector,
            ik_solver=ik_solver,
            base_offset=robot_base_pos,
        )

        object_name = args.pick_object
        pre_pos = detector.get_object_position(object_name)
        print(f"  Object before pick: {object_name} @ {np.round(pre_pos, 4).tolist()}")
        grasp_pos = pre_pos.copy()
        grasp_pos[2] += robot_cfg.ee_finger_offset
        approach_pos = grasp_pos.copy()
        approach_pos[2] += 0.10

        skills.gripper_open(duration=0.3)
        skills._move_palm_down(approach_pos)
        skills._move_cartesian_steps(approach_pos, grasp_pos, max_step=0.015)

        ee_pos, _ = robot_interface.read_ee_pose()
        print(f"  Pre-grasp EE:       {np.round(ee_pos, 4).tolist()}")
        print_pad_alignment(env, object_name, pre_pos, "PRE-GRASP pads")

        skills.gripper_close(duration=robot_cfg.gripper_close_duration)
        post_close_pos = detector.get_object_position(object_name)
        print(f"  Object after close: {np.round(post_close_pos, 4).tolist()}")
        print_pad_alignment(env, object_name, post_close_pos, "POST-CLOSE pads")

        ee_before_lift, _ = robot_interface.read_ee_pose()
        skills._move_cartesian_steps(
            ee_before_lift,
            approach_pos,
            max_step=0.015,
            fixed_joints=[3, 4, 5],
        )

        post_pos = detector.get_object_position(object_name)
        z_delta = float(post_pos[2] - pre_pos[2])
        ok = z_delta > 0.02
        finger_state = []
        if robot_interface._finger_indices:
            finger_state = (
                robot_interface._articulation.data.joint_pos[robot_interface.env_idx][robot_interface._finger_indices]
                .cpu()
                .numpy()
                .round(4)
                .tolist()
            )
        ee_pos, _ = robot_interface.read_ee_pose()
        arm_joints = robot_interface.read_arm_joint_positions().round(4).tolist()
        print(f"  Object after pick:  {object_name} @ {np.round(post_pos, 4).tolist()} (z_delta={z_delta:.4f}m)")
        print(f"  EE after pick:      {np.round(ee_pos, 4).tolist()}")
        print(f"  Arm joints:         {arm_joints}")
        print(f"  Finger state:       {finger_state}")
        print(f"  [{'PASS' if ok else 'FAIL'}] Pick diagnostic")
        return ok
    except Exception as e:
        import traceback
        print(f"  [FAIL] Pick diagnostic error: {e}")
        traceback.print_exc()
        return False


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 60)
    print("  UR10e Basic Control Verification")
    print("=" * 60)
    print(f"Using env dir: {ENV_DIR}")
    if not ENV_DIR.exists():
        print(f"ERROR: env dir not found: {ENV_DIR}")
        sys.exit(1)

    # Find env cfg class
    env_cfg_class = None
    for name, obj in globals().items():
        if isinstance(obj, type) and name.endswith("EnvCfg") and name != "ManagerBasedRLEnvCfg":
            env_cfg_class = obj
            break

    if env_cfg_class is None:
        print("ERROR: Could not find EnvCfg class")
        sys.exit(1)

    print(f"Using env_cfg class: {env_cfg_class.__name__}")

    # Create env_cfg and apply patches
    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = 1
    apply_data_collection_patches(env_cfg)
    using_local_ur10e_asset = False
    if args.use_local_asset:
        using_local_ur10e_asset = apply_local_ur10e_asset_override(env_cfg, verbose=True)
    else:
        print("Using default Isaac Sim UR10e asset (local override disabled)")

    # Create environment
    print("\nCreating environment...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    if not using_local_ur10e_asset and not args.skip_pad_collision_patch:
        add_robotiq_pad_collisions(env)
    elif args.skip_pad_collision_patch:
        print("Skipping UR10e pad collision injection")
    print("Environment created successfully.")

    # Reset
    print("Resetting environment...")
    obs, info = env.reset()

    # Warmup (10 steps) — hold initial joint positions, NOT zeros!
    # With scale=1.0 + offset=False, zero action = target 0.0 (wrong).
    warmup_action = build_action(env, EXPECTED_ARM_JOINTS, gripper_open=True)
    step_n(env, warmup_action, 10)
    print("Warmup complete.\n")

    # Run tests
    results = {}
    if args.only_pick_diagnostic:
        results[5] = test_5_pick_diagnostic(env)
    else:
        results[1] = test_1_initial_state(env)
        results[2] = test_2_hold_position(env)
        results[3] = test_3_single_joint(env)
        results[4] = test_4_ik_to_ee(env)
        results[5] = test_5_pick_diagnostic(env)

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for num, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  Test {num}: {status}")
        all_pass &= ok

    overall = "ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED"
    print(f"\n  {overall}")
    print(f"{'='*60}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
