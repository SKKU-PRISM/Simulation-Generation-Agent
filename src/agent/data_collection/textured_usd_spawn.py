"""Helpers for spawning textured USD props with child-prim physics.

These helpers are used for Isaac Sim assets whose root prim is a visual Xform
without rigid-body schemas. We preserve the original textured USD reference and
apply rigid/collision/mass schemas onto a resolved mesh child prim so
``RigidObjectCfg`` can still initialize against the asset tree.
"""

from __future__ import annotations

from copy import deepcopy

from pxr import Usd, UsdGeom, UsdShade

from isaaclab.sim import schemas
from isaaclab.sim.schemas import ConvexHullPropertiesCfg
from isaaclab.sim.spawners.from_files import from_files as from_files_impl
from isaaclab.sim.utils import clone, get_current_stage


TEXTURED_USD_CHILD_PHYSICS = {
    "sm_mug_a2.usd": {
        "physics_target_prim_basename": "SM_Mug_A2",
        "mesh_collision": "convexHull",
    },
    "007_tuna_fish_can.usd": {
        "physics_target_prim_basename": "_07_tuna_fish_can",
        "mesh_collision": "convexHull",
    },
}


def _resolve_mesh_child_prim(stage: Usd.Stage, root_path: str, cfg) -> Usd.Prim:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Failed to resolve textured USD root prim: {root_path}")

    usd_path = str(getattr(cfg, "usd_path", "") or "").lower()
    target_meta = None
    for suffix, meta in TEXTURED_USD_CHILD_PHYSICS.items():
        if suffix in usd_path:
            target_meta = meta
            break

    preferred_name = getattr(cfg, "physics_target_prim_basename", None)
    if not preferred_name and target_meta is not None:
        preferred_name = target_meta.get("physics_target_prim_basename")

    mesh_prims: list[Usd.Prim] = []
    mesh_with_material: list[Usd.Prim] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prims.append(prim)
        binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        if str(binding.GetMaterialPath()):
            mesh_with_material.append(prim)

    if preferred_name:
        for prim in mesh_prims:
            if prim.GetName() == preferred_name:
                return prim

    if len(mesh_with_material) == 1:
        return mesh_with_material[0]
    if len(mesh_prims) == 1:
        return mesh_prims[0]

    candidate_paths = [str(prim.GetPath()) for prim in mesh_prims]
    raise RuntimeError(
        f"Could not uniquely resolve textured USD physics target under '{root_path}'. "
        f"Candidates: {candidate_paths}"
    )


@clone
def spawn_textured_usd_with_child_physics(
    prim_path: str,
    cfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a textured USD and apply physics schemas to a resolved mesh child prim."""
    cfg_copy = deepcopy(cfg)
    rigid_props = getattr(cfg, "rigid_props", None)
    collision_props = getattr(cfg, "collision_props", None)
    mass_props = getattr(cfg, "mass_props", None)
    mesh_collision_props = getattr(cfg, "mesh_collision_props", None)

    cfg_copy.rigid_props = None
    cfg_copy.collision_props = None
    cfg_copy.mass_props = None
    if hasattr(cfg_copy, "mesh_collision_props"):
        cfg_copy.mesh_collision_props = None

    prim = from_files_impl._spawn_from_usd_file(
        prim_path,
        cfg_copy.usd_path,
        cfg_copy,
        translation,
        orientation,
        **kwargs,
    )

    stage = get_current_stage()
    target_prim = _resolve_mesh_child_prim(stage, prim_path, cfg)
    target_path = target_prim.GetPath().pathString

    if rigid_props is not None:
        schemas.define_rigid_body_properties(target_path, rigid_props, stage=stage)
    if collision_props is not None:
        schemas.define_collision_properties(target_path, collision_props, stage=stage)
    if mass_props is not None:
        schemas.define_mass_properties(target_path, mass_props, stage=stage)

    if mesh_collision_props is None:
        mesh_collision_props = ConvexHullPropertiesCfg()
    schemas.define_mesh_collision_properties(target_path, mesh_collision_props, stage=stage)

    print(
        f"Applied textured USD child physics: root={prim_path}, "
        f"target={target_path}, usd={getattr(cfg, 'usd_path', '')}"
    )
    return prim
