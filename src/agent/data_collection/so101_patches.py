"""Runtime SO-101 collision helpers.

The stock SO-101 USD exposes instanceable mesh collisions for the claw links.
For tiny 2 cm cubes, those imported mesh collisions are not reliable enough for
ADC pick/stack. This helper adds two simple cuboid collision proxies at runtime:
one on the fixed jaw and one on the moving jaw.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SO101_ROBOT_BASE_TEMPLATE = "/World/envs/env_{env_idx}/Robot"
SO101_FIXED_LINK = "gripper_frame_link"
SO101_MOVING_LINK = "moving_jaw_so101_v1_link"

# Local-frame collision proxies tuned against the runtime SO-101 transforms:
# - gripper_frame_link local +X points toward the cube at grasp height
# - gripper_frame_link local +Y is the lateral pinch axis
# The fixed pad is expressed directly in gripper_frame_link; the moving pad is
# expressed in moving_jaw_so101_v1_link and back-computed from the desired
# gripper-frame target location near the claw mouth.
SO101_FIXED_PAD_LOCAL_CENTER = np.array([0.0400, -0.0105, 0.0], dtype=np.float64)
SO101_MOVING_PAD_LOCAL_CENTER = np.array([-0.0998, 0.0164, 0.0090], dtype=np.float64)
SO101_PAD_HALF_SIZE = np.array([0.008, 0.0025, 0.010], dtype=np.float64)


def _ensure_xform_ops(collision_prim: Any, center: np.ndarray, half_size: np.ndarray) -> None:
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


def _ensure_proxy(stage: Any, link_path: str, name: str, center: np.ndarray, half_size: np.ndarray) -> str:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    collision_path = f"{link_path}/{name}"
    collision_prim = stage.GetPrimAtPath(collision_path)
    if not collision_prim.IsValid():
        collision_prim = stage.DefinePrim(collision_path, "Cube")
    _ensure_xform_ops(collision_prim, center, half_size)

    if not collision_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(collision_prim)
    if not collision_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        physx_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_prim)
    else:
        physx_api = PhysxSchema.PhysxCollisionAPI(collision_prim)
    physx_api.GetContactOffsetAttr().Set(0.005)
    physx_api.GetRestOffsetAttr().Set(0.0)
    UsdGeom.Imageable(collision_prim).MakeInvisible()
    return collision_path


def ensure_so101_claw_collisions(stage: Any, env_idx: int = 0, verbose: bool = False) -> list[str]:
    """Attach minimal cuboid collision pads to the SO-101 claw at runtime."""
    robot_base = SO101_ROBOT_BASE_TEMPLATE.format(env_idx=env_idx)
    fixed_link_path = f"{robot_base}/{SO101_FIXED_LINK}"
    moving_link_path = f"{robot_base}/{SO101_MOVING_LINK}"

    fixed_link = stage.GetPrimAtPath(fixed_link_path)
    moving_link = stage.GetPrimAtPath(moving_link_path)
    if not fixed_link.IsValid():
        raise RuntimeError(f"SO-101 fixed jaw link not found: {fixed_link_path}")
    if not moving_link.IsValid():
        raise RuntimeError(f"SO-101 moving jaw link not found: {moving_link_path}")

    added_paths = [
        _ensure_proxy(
            stage,
            fixed_link_path,
            "fixed_pad_proxy",
            SO101_FIXED_PAD_LOCAL_CENTER,
            SO101_PAD_HALF_SIZE,
        ),
        _ensure_proxy(
            stage,
            moving_link_path,
            "moving_pad_proxy",
            SO101_MOVING_PAD_LOCAL_CENTER,
            SO101_PAD_HALF_SIZE,
        ),
    ]
    if verbose:
        for path in added_paths:
            print(f"  ensured SO-101 collision proxy: {path}")
    return added_paths


def get_so101_pad_world_centers_from_stage(stage: Any, env_idx: int = 0) -> dict[str, np.ndarray]:
    """Return world-space proxy centers from the active stage."""
    from pxr import Usd, UsdGeom

    robot_base = SO101_ROBOT_BASE_TEMPLATE.format(env_idx=env_idx)
    proxy_paths = {
        "fixed": f"{robot_base}/{SO101_FIXED_LINK}/fixed_pad_proxy",
        "moving": f"{robot_base}/{SO101_MOVING_LINK}/moving_pad_proxy",
    }
    centers: dict[str, np.ndarray] = {}
    for key, path in proxy_paths.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Missing SO-101 proxy prim: {path}")
        world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centers[key] = np.array(world_transform.ExtractTranslation(), dtype=np.float64)
    return centers


def summarize_so101_pad_alignment_from_stage(stage: Any, object_position: np.ndarray, env_idx: int = 0) -> dict[str, Any]:
    """Summarize current proxy geometry relative to an object center."""
    centers = get_so101_pad_world_centers_from_stage(stage, env_idx=env_idx)
    midpoint = 0.5 * (centers["fixed"] + centers["moving"])
    separation = float(np.linalg.norm(centers["fixed"] - centers["moving"]))
    object_position = np.asarray(object_position, dtype=np.float64)
    return {
        "fixed": centers["fixed"],
        "moving": centers["moving"],
        "midpoint": midpoint,
        "midpoint_delta": midpoint - object_position,
        "separation": separation,
    }
