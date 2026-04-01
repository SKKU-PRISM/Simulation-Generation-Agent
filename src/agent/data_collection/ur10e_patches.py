"""Shared UR10e runtime fixes and diagnostics.

These helpers centralize the Robotiq 2F-85 pad-collision patch and the
geometry calculations used by UR10e grasp diagnostics so the main pipeline and
the debug harness do not drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_UR10E_ASSET_PATH = PROJECT_ROOT / "assets" / "robots" / "ur10e_robotiq_2f85" / "ur10e_robotiq_2f85.usd"
LOCAL_UR10E_JOINT_RENAMES = {
    "left_inner_finger_knuckle_joint": "left_inner_knuckle_joint",
    "right_inner_finger_knuckle_joint": "right_inner_knuckle_joint",
    "inner_finger_knuckle_joint": "inner_knuckle_joint",
}
UR10E_ROBOTIQ_BASE_TEMPLATE = "/World/envs/env_{env_idx}/Robot/ee_link/Robotiq_2F_85"
UR10E_PAD_LINK_NAMES = ("left_inner_finger", "right_inner_finger")
UR10E_PAD_LOCAL_CENTER = np.array([0.0, 0.0457554, -0.0272203], dtype=np.float64)
UR10E_PAD_HALF_SIZE = np.array([0.015, 0.035, 0.00375], dtype=np.float64)


def should_use_local_ur10e_asset() -> bool:
    """Return whether the repo-local UR10e asset override is explicitly enabled."""
    value = os.environ.get("SGA_USE_LOCAL_UR10E_ASSET", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _quat_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _ensure_xform_ops(collision_prim: Any, center: np.ndarray, half_size: np.ndarray) -> None:
    """Set translate/scale ops without creating duplicate ops on repeated calls."""
    from pxr import Gf, UsdGeom

    xformable = UsdGeom.Xformable(collision_prim)
    translate_op = None
    scale_op = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale and scale_op is None:
            scale_op = op

    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if scale_op is None:
        scale_op = xformable.AddScaleOp()

    translate_op.Set(Gf.Vec3d(*center.tolist()))
    scale_op.Set(Gf.Vec3d(*half_size.tolist()))


def apply_local_ur10e_asset_override(env_cfg: Any, verbose: bool = False) -> bool:
    """Swap the UR10e spawn asset to the repo-local USD when available."""
    if not LOCAL_UR10E_ASSET_PATH.exists():
        return False

    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None)
    if spawn_cfg is None or not hasattr(spawn_cfg, "usd_path"):
        return False

    spawn_cfg.usd_path = str(LOCAL_UR10E_ASSET_PATH)
    if hasattr(spawn_cfg, "variants"):
        spawn_cfg.variants = None
    _patch_local_ur10e_joint_names(env_cfg)
    if verbose:
        print(f"Patched UR10e spawn asset: {LOCAL_UR10E_ASSET_PATH}")
    return True


def _rename_joint_key(name: str) -> str:
    for old_name, new_name in LOCAL_UR10E_JOINT_RENAMES.items():
        name = name.replace(old_name, new_name)
    return name


def _patch_joint_mapping(mapping: Any) -> Any:
    if isinstance(mapping, dict):
        return {_rename_joint_key(str(key)): value for key, value in mapping.items()}
    return mapping


def _patch_joint_name_list(names: Any) -> Any:
    if isinstance(names, str):
        return _rename_joint_key(names)
    if isinstance(names, (list, tuple)):
        return [_rename_joint_key(str(name)) for name in names]
    return names


def _patch_local_ur10e_joint_names(env_cfg: Any) -> None:
    """Align the env config with the joint names emitted by the local URDF-converted USD."""
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    if robot_cfg is not None:
        init_state = getattr(robot_cfg, "init_state", None)
        if init_state is not None and hasattr(init_state, "joint_pos"):
            init_state.joint_pos = _patch_joint_mapping(init_state.joint_pos)

        actuators = getattr(robot_cfg, "actuators", None)
        if isinstance(actuators, dict):
            for actuator in actuators.values():
                joint_names_expr = getattr(actuator, "joint_names_expr", None)
                if joint_names_expr is not None:
                    actuator.joint_names_expr = _patch_joint_name_list(joint_names_expr)

    actions = getattr(env_cfg, "actions", None)
    gripper_action = getattr(actions, "gripper_action", None)
    if gripper_action is not None:
        if hasattr(gripper_action, "joint_names"):
            gripper_action.joint_names = _patch_joint_name_list(gripper_action.joint_names)
        if hasattr(gripper_action, "open_command_expr"):
            gripper_action.open_command_expr = _patch_joint_mapping(gripper_action.open_command_expr)
        if hasattr(gripper_action, "close_command_expr"):
            gripper_action.close_command_expr = _patch_joint_mapping(gripper_action.close_command_expr)

    ee_frame = getattr(getattr(env_cfg, "scene", None), "ee_frame", None)
    if ee_frame is not None:
        if hasattr(ee_frame, "prim_path"):
            ee_frame.prim_path = "{ENV_REGEX_NS}/Robot/ur"
        target_frames = getattr(ee_frame, "target_frames", None)
        if isinstance(target_frames, list):
            for frame_cfg in target_frames:
                if hasattr(frame_cfg, "name") and frame_cfg.name == "ee_frame" and hasattr(frame_cfg, "prim_path"):
                    frame_cfg.prim_path = "{ENV_REGEX_NS}/Robot/wrist_3_link"


def ensure_ur10e_pad_collisions(stage: Any, env_idx: int = 0, verbose: bool = False) -> list[str]:
    """Attach minimal collision cubes to the two Robotiq inner-finger pads."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    robotiq_base = UR10E_ROBOTIQ_BASE_TEMPLATE.format(env_idx=env_idx)
    robotiq_prim = stage.GetPrimAtPath(robotiq_base)
    if not robotiq_prim.IsValid():
        if verbose:
            print(f"WARNING: Robotiq gripper not found at {robotiq_base}")
        return []

    added_paths: list[str] = []
    for link_name in UR10E_PAD_LINK_NAMES:
        link_path = f"{robotiq_base}/{link_name}"
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim.IsValid():
            if verbose:
                print(f"WARNING: Robotiq link not found: {link_path}")
            continue

        collision_path = f"{link_path}/collision_shape"
        collision_prim = stage.GetPrimAtPath(collision_path)
        if not collision_prim.IsValid():
            collision_prim = stage.DefinePrim(collision_path, "Cube")
        _ensure_xform_ops(collision_prim, UR10E_PAD_LOCAL_CENTER, UR10E_PAD_HALF_SIZE)

        if not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(collision_prim)
        if not collision_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            physx_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
        else:
            physx_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
        physx_api.GetContactOffsetAttr().Set(0.005)
        physx_api.GetRestOffsetAttr().Set(0.0)
        UsdGeom.Imageable(collision_prim).MakeInvisible()

        added_paths.append(collision_path)
        if verbose:
            print(
                "  ensured collision: "
                f"{collision_path} center={UR10E_PAD_LOCAL_CENTER.tolist()} "
                f"half={UR10E_PAD_HALF_SIZE.tolist()}"
            )

    return added_paths


def get_ur10e_pad_world_centers(articulation: Any, env_idx: int = 0) -> dict[str, np.ndarray]:
    """Compute world-space centers of the injected pad boxes from articulation state."""
    centers: dict[str, np.ndarray] = {}
    for link_name in UR10E_PAD_LINK_NAMES:
        body_indices, _ = articulation.find_bodies(f"^{link_name}$")
        if len(body_indices) != 1:
            raise RuntimeError(f"Expected one body for {link_name}, got {body_indices}")
        body_state = articulation.data.body_state_w[env_idx, body_indices[0]]
        body_pos = body_state[:3].cpu().numpy().astype(np.float64)
        body_quat = body_state[3:7].cpu().numpy().astype(np.float64)
        centers[link_name] = body_pos + _quat_wxyz_to_rotation_matrix(body_quat) @ UR10E_PAD_LOCAL_CENTER
    return centers


def get_ur10e_pad_world_centers_from_stage(stage: Any, env_idx: int = 0) -> dict[str, np.ndarray]:
    """Read pad centers from the actual collision prim world transforms."""
    from pxr import Usd, UsdGeom

    robotiq_base = UR10E_ROBOTIQ_BASE_TEMPLATE.format(env_idx=env_idx)
    centers: dict[str, np.ndarray] = {}
    for link_name in UR10E_PAD_LINK_NAMES:
        collision_path = f"{robotiq_base}/{link_name}/collision_shape"
        collision_prim = stage.GetPrimAtPath(collision_path)
        if not collision_prim.IsValid():
            raise RuntimeError(f"Missing collision prim: {collision_path}")
        world_transform = UsdGeom.Xformable(collision_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centers[link_name] = np.array(world_transform.ExtractTranslation(), dtype=np.float64)
    return centers


def summarize_ur10e_pad_alignment(
    articulation: Any,
    object_position: np.ndarray,
    env_idx: int = 0,
) -> dict[str, Any]:
    """Return pad-center geometry relative to an object center."""
    centers = get_ur10e_pad_world_centers(articulation, env_idx=env_idx)
    midpoint = 0.5 * (centers["left_inner_finger"] + centers["right_inner_finger"])
    separation = float(np.linalg.norm(centers["left_inner_finger"] - centers["right_inner_finger"]))
    object_position = np.asarray(object_position, dtype=np.float64)
    return {
        "left": centers["left_inner_finger"],
        "right": centers["right_inner_finger"],
        "midpoint": midpoint,
        "midpoint_delta": midpoint - object_position,
        "separation": separation,
    }


def summarize_ur10e_pad_alignment_from_stage(
    stage: Any,
    object_position: np.ndarray,
    env_idx: int = 0,
) -> dict[str, Any]:
    """Return pad-center geometry using the actual collision prim transforms."""
    centers = get_ur10e_pad_world_centers_from_stage(stage, env_idx=env_idx)
    midpoint = 0.5 * (centers["left_inner_finger"] + centers["right_inner_finger"])
    separation = float(np.linalg.norm(centers["left_inner_finger"] - centers["right_inner_finger"]))
    object_position = np.asarray(object_position, dtype=np.float64)
    return {
        "left": centers["left_inner_finger"],
        "right": centers["right_inner_finger"],
        "midpoint": midpoint,
        "midpoint_delta": midpoint - object_position,
        "separation": separation,
    }
