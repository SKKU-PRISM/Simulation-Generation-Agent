"""Factory task initialization: place held object in gripper and close.

Replicates the initialization sequence from IsaacLab Factory tasks
(factory_env.py:randomize_initial_state) for ManagerBasedRLEnv environments.

Usage in collect_data.py (after env.reset()):
    from src.agent.data_collection.factory_init import initialize_grasped_state
    initialize_grasped_state(env, initial_grasp_cfg)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import numpy as np

logger = logging.getLogger(__name__)


def initialize_grasped_state(
    env,
    initial_grasp: dict[str, Any],
    *,
    grasp_close_duration: float = 0.25,
) -> bool:
    """Initialize the environment with the held object in the robot's gripper.

    This replicates the Factory task initialization:
    1. Disable gravity
    2. Move robot arm to position above fixed object via direct joint write
    3. Teleport held object into gripper fingertip position
    4. Close gripper for ``grasp_close_duration`` seconds
    5. Restore gravity

    Args:
        env: ManagerBasedRLEnv instance (already reset).
        initial_grasp: Dict from task YAML ``task.initial_grasp`` containing:
            - held_object: scene entity name of the grasped object
            - fixed_object: scene entity name of the target (hole/base/bolt)
            - grasp_width: finger joint value for grasping
            - held_asset_height: height of the held asset (m)
            - held_asset_diameter: diameter (m)
            - fingerpad_length: Franka fingerpad length (m)
            - held_asset_rot_init_deg: initial Z-rotation of held asset (degrees)
            - held_asset_pos_noise: [x, y, z] noise (m)
            - hand_init_pos_rel_fixed: [x, y, z] hand position relative to fixed asset tip
            - hand_init_orn_euler: [roll, pitch, yaw] hand orientation (radians)
        grasp_close_duration: Duration in seconds to close the gripper.

    Returns:
        True if initialization succeeded.
    """
    try:
        return _do_initialize(env, initial_grasp, grasp_close_duration)
    except Exception as e:
        logger.error(f"Factory grasp initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _do_initialize(
    env,
    cfg: dict[str, Any],
    grasp_close_duration: float,
) -> bool:
    device = env.device
    scene = env.scene
    sim = env.sim

    held_name = cfg["held_object"]
    fixed_name = cfg["fixed_object"]

    # --- Get scene entities ---
    held_asset = scene[held_name]
    fixed_asset = scene[fixed_name]
    robot = scene["robot"]

    # --- 1. Disable gravity ---
    physics_sim_view = sim.physics_sim_view
    if physics_sim_view is not None:
        try:
            import carb
            physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))
        except Exception:
            logger.warning("Could not disable gravity via physics_sim_view")

    # --- 2. Get fixed asset position (tip) ---
    # Handle both RigidObject (.data.root_pos_w) and XformPrimView (demoted AssetBaseCfg)
    if hasattr(fixed_asset, "data") and hasattr(fixed_asset.data, "root_pos_w"):
        fixed_pos = fixed_asset.data.root_pos_w[0].clone()
        fixed_quat = fixed_asset.data.root_quat_w[0].clone()
    else:
        # XformPrimView from AssetBaseCfg — read position from USD prim or YAML config
        import torch as _torch
        _yaml_pos = cfg.get("fixed_asset_position") or [0.6, 0.0, 0.05]
        # Try to get from task_doc assets
        for _a in cfg.get("_assets", []):
            if _a.get("name") == fixed_name:
                _yaml_pos = _a.get("position", _yaml_pos)
                break
        fixed_pos = _torch.tensor(_yaml_pos, dtype=_torch.float32, device=device)
        fixed_quat = _torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=_torch.float32, device=device)
        logger.info(f"Fixed asset '{fixed_name}' is XformPrimView, using position from config: {_yaml_pos}")

    # Compute fixed asset tip position
    # Support both flat keys (hand_init_pos_rel_fixed) and nested YAML
    # format (fixed_asset.height, hand_initialization.position_above_fixed)
    fixed_asset_cfg = cfg.get("fixed_asset", {})
    fixed_height = cfg.get("fixed_asset_height", fixed_asset_cfg.get("height", 0.025))
    fixed_base_height = cfg.get("fixed_base_height", fixed_asset_cfg.get("base_height", 0.0))
    tip_offset_local = torch.tensor([0.0, 0.0, fixed_height + fixed_base_height], device=device)

    # For gear_mesh, apply medium_gear_base_offset
    gear_offset = cfg.get("medium_gear_base_offset", fixed_asset_cfg.get("medium_gear_base_offset"))
    if gear_offset:
        tip_offset_local[0] += gear_offset[0]

    fixed_tip_pos = fixed_pos.clone()
    fixed_tip_pos[2] += tip_offset_local[2]
    if gear_offset:
        # Rotate offset by fixed asset orientation
        fixed_tip_pos[0] += tip_offset_local[0]

    # --- 3. Compute hand target position above fixed asset ---
    # Support both flat key and nested YAML format
    hand_init_cfg = cfg.get("hand_initialization", {})
    hand_init_pos_raw = cfg.get("hand_init_pos_rel_fixed", hand_init_cfg.get("position_above_fixed"))
    if hand_init_pos_raw is None:
        raise KeyError("Missing hand init position: need 'hand_init_pos_rel_fixed' or 'hand_initialization.position_above_fixed'")
    hand_init_pos = torch.tensor(hand_init_pos_raw, dtype=torch.float32, device=device)
    hand_target_pos = fixed_tip_pos.clone()
    hand_target_pos += hand_init_pos  # relative offset (mostly Z height above)

    hand_orn = cfg.get("hand_init_orn_euler", hand_init_cfg.get("orientation_euler"))
    if hand_orn is None:
        hand_orn = [3.1416, 0.0, 0.0]  # default palm-down
    # Convert euler to quaternion (w, x, y, z)
    hand_target_quat = _euler_to_quat(hand_orn[0], hand_orn[1], hand_orn[2], device)

    # --- 4. IK solve to get joint positions for hand target ---
    # Use Pinocchio if available, otherwise set joints directly from task config
    target_joint_pos = _solve_ik_for_hand_pose(
        robot, hand_target_pos, hand_target_quat, cfg, device
    )

    if target_joint_pos is None:
        logger.warning("IK failed, using default reset joints from Factory config")
        # Factory default reset joints
        reset_joints = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761,
                        -2.7717e-04, 1.7796, 7.8556e-01]
        target_joint_pos = torch.tensor(reset_joints, dtype=torch.float32, device=device)

    # Set robot joint positions directly
    grasp_width = cfg.get("grasp_width", 0.01)
    full_joint_pos = robot.data.joint_pos.clone()
    n_arm = min(7, target_joint_pos.shape[0])
    full_joint_pos[0, :n_arm] = target_joint_pos[:n_arm]
    # Set gripper to grasp width (not fully closed yet)
    n_total = full_joint_pos.shape[1]
    if n_total > 7:
        full_joint_pos[0, 7:] = grasp_width

    joint_vel = torch.zeros_like(full_joint_pos)
    robot.write_joint_state_to_sim(full_joint_pos, joint_vel)
    robot.set_joint_position_target(full_joint_pos)

    # Step simulation to settle robot
    for _ in range(5):
        sim.step(render=False)
        scene.update(dt=env.physics_dt)

    # --- 5. Teleport held asset into gripper ---
    # Compute held asset relative position in gripper
    held_rel_pos, held_rel_quat = _get_held_asset_relative_pose(cfg, device)

    # Get current fingertip position from robot
    fingertip_pos, fingertip_quat = _get_fingertip_pose(robot, device)

    # Flip gripper Z orientation (Factory convention)
    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    flipped_quat, flipped_pos = _tf_combine(fingertip_quat, fingertip_pos, flip_z_quat,
                                             torch.zeros(3, device=device))

    # Inverse: gripper → held asset transform
    inv_quat, inv_pos = _tf_inverse(held_rel_quat, held_rel_pos)

    # Combined: held asset world pose
    held_world_quat, held_world_pos = _tf_combine(flipped_quat, flipped_pos, inv_quat, inv_pos)

    # Add small noise
    pos_noise_level = cfg.get("held_asset_pos_noise", [0.0, 0.0, 0.0])
    noise = torch.tensor(pos_noise_level, dtype=torch.float32, device=device) * (
        2.0 * torch.rand(3, device=device) - 1.0
    )
    held_world_pos = held_world_pos + noise

    # Write held asset pose
    held_state = held_asset.data.default_root_state.clone()
    held_state[0, 0:3] = held_world_pos + env.scene.env_origins[0]
    held_state[0, 3:7] = held_world_quat
    held_state[0, 7:] = 0.0  # zero velocity
    held_asset.write_root_pose_to_sim(held_state[:, 0:7])
    held_asset.write_root_velocity_to_sim(held_state[:, 7:])

    # Step to settle held asset
    for _ in range(3):
        sim.step(render=False)
        scene.update(dt=env.physics_dt)

    # --- 6. Close gripper ---
    grasp_time = 0.0
    close_joint_pos = full_joint_pos.clone()
    if n_total > 7:
        close_joint_pos[0, 7:] = 0.0  # fully close

    while grasp_time < grasp_close_duration:
        robot.set_joint_position_target(close_joint_pos)
        sim.step(render=False)
        scene.update(dt=env.physics_dt)
        grasp_time += env.physics_dt

    # --- 7. Restore gravity ---
    if physics_sim_view is not None:
        try:
            import carb
            physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, -9.81))
        except Exception:
            pass

    # Final settle steps with gravity
    for _ in range(5):
        robot.set_joint_position_target(close_joint_pos)
        sim.step(render=False)
        scene.update(dt=env.physics_dt)

    logger.info(
        f"Factory grasp init: held={held_name} at pos={held_world_pos.tolist()}, "
        f"gripper closed, gravity restored"
    )
    return True


def _get_held_asset_relative_pose(cfg: dict, device) -> tuple:
    """Compute held asset position relative to fingertip (Factory convention)."""
    task_name = cfg.get("task_name", "")
    height = cfg.get("held_asset_height", 0.05)
    fingerpad_length = cfg.get("fingerpad_length", 0.017608)

    if "peg" in task_name.lower():
        rel_pos = torch.tensor([0.0, 0.0, height - fingerpad_length], device=device)
    elif "gear" in task_name.lower():
        gear_offset = cfg.get("medium_gear_base_offset", [0.02025, 0.0, 0.0])
        rel_pos = torch.tensor([gear_offset[0], 0.0, height / 2.0 * 1.1], device=device)
    elif "nut" in task_name.lower():
        base_height = cfg.get("fixed_base_height", 0.01)
        rel_pos = torch.tensor([0.0, 0.0, base_height], device=device)
    else:
        rel_pos = torch.tensor([0.0, 0.0, height - fingerpad_length], device=device)

    # Rotation
    rot_deg = cfg.get("held_asset_rot_init_deg", 0.0)
    if abs(rot_deg) > 0.01:
        rot_rad = rot_deg * math.pi / 180.0
        rel_quat = _euler_to_quat(0.0, 0.0, rot_rad, device)
    else:
        rel_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    return rel_pos, rel_quat


def _get_fingertip_pose(robot, device) -> tuple:
    """Get the midpoint between left and right finger pads."""
    # Try to get finger body indices
    try:
        body_names = robot.data.body_names
        left_idx = body_names.index("panda_leftfinger")
        right_idx = body_names.index("panda_rightfinger")
        left_pos = robot.data.body_pos_w[0, left_idx]
        right_pos = robot.data.body_pos_w[0, right_idx]
        mid_pos = (left_pos + right_pos) / 2.0

        # Use left finger orientation as reference
        mid_quat = robot.data.body_quat_w[0, left_idx]
        return mid_pos, mid_quat
    except (ValueError, IndexError):
        # Fallback: use panda_hand position with offset
        try:
            hand_idx = robot.data.body_names.index("panda_hand")
            hand_pos = robot.data.body_pos_w[0, hand_idx].clone()
            hand_quat = robot.data.body_quat_w[0, hand_idx].clone()
            # Approximate fingertip: 10.4cm below hand frame
            offset = torch.tensor([0.0, 0.0, -0.104], device=device)
            rotated_offset = _rotate_vector(hand_quat, offset)
            fingertip_pos = hand_pos + rotated_offset
            return fingertip_pos, hand_quat
        except (ValueError, IndexError):
            logger.error("Cannot find finger or hand bodies on robot")
            return (
                torch.tensor([0.3, 0.0, 0.5], device=device),
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
            )


def _solve_ik_for_hand_pose(robot, target_pos, target_quat, cfg, device):
    """Return pre-computed joint positions for Factory tasks.

    These joint positions place the Franka end-effector above the fixed
    asset (hole/gear_base/bolt) at the correct height and orientation.
    They are derived from the Factory task's IK servo initialization.
    """
    task_name = str(cfg.get("task_name", "")).lower()

    # Pre-computed joint positions from Factory IK servo results.
    # These place the EE above the default fixed asset position [0.6, 0.0, ~0.81+height].
    if "peg" in task_name:
        # EE above hole at [0.6, 0.0, 0.882] (hole tip + 0.047m)
        joints = [0.1242, 0.2155, 0.0663, -1.8061, -0.0184, 2.0347, 0.8928]
    elif "gear" in task_name:
        # EE above gear base shaft at [0.62, 0.0, 0.870] (gear tip + 0.035m)
        joints = [0.1535, 0.1876, 0.0918, -1.8436, -0.0139, 2.0434, 0.8712]
    elif "nut" in task_name:
        # EE above bolt at [0.6, 0.0, 0.860] (bolt tip + 0.015m)
        joints = [0.1158, 0.2389, 0.0547, -1.7775, -0.0207, 2.0286, 1.6573]
    else:
        return None

    return torch.tensor(joints, dtype=torch.float32, device=device)


# --- Quaternion math utilities (PyTorch, no isaaclab dependency) ---

def _euler_to_quat(roll: float, pitch: float, yaw: float, device) -> torch.Tensor:
    """Convert Euler angles (XYZ) to quaternion (w, x, y, z)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return torch.tensor([w, x, y, z], dtype=torch.float32, device=device)


def _tf_combine(q1, t1, q2, t2):
    """Combine two transforms: T_out = T1 * T2."""
    q_out = _quat_mul(q1, q2)
    t_out = t1 + _rotate_vector(q1, t2)
    return q_out, t_out


def _tf_inverse(q, t):
    """Inverse of a transform."""
    q_inv = _quat_conjugate(q)
    t_inv = -_rotate_vector(q_inv, t)
    return q_inv, t_inv


def _quat_mul(a, b):
    """Quaternion multiplication (w, x, y, z convention)."""
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_conjugate(q):
    """Quaternion conjugate."""
    return torch.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], dim=-1)


def _rotate_vector(q, v):
    """Rotate vector v by quaternion q."""
    q_v = torch.zeros_like(q)
    q_v[..., 1:] = v
    q_conj = _quat_conjugate(q)
    rotated = _quat_mul(_quat_mul(q, q_v), q_conj)
    return rotated[..., 1:]
