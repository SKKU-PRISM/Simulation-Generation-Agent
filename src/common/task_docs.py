"""Shared task-document loading and normalization helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import math
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UR10_ASSET_PATH = "{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10.usd"
UR10E_ASSET_PATH = "{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd"


def load_task_document(yaml_path: str | Path) -> dict[str, Any]:
    """Load a task YAML and normalize repo-specific conventions."""
    yaml_path = Path(yaml_path).resolve()
    with open(yaml_path) as f:
        task_doc = yaml.safe_load(f)
    return resolve_task_document(task_doc, yaml_path)


def resolve_task_document(task_doc: dict[str, Any], yaml_path: str | Path) -> dict[str, Any]:
    """Resolve repo-local path, assembly semantics, and legacy robot aliases."""
    resolved = deepcopy(task_doc)
    yaml_path = Path(yaml_path).resolve()
    _normalize_asset_paths(resolved)

    if _detect_category(yaml_path) != "assembly":
        return resolved

    _canonicalize_ur10_assembly(resolved)
    _resolve_assembly_conditions(resolved, yaml_path)
    return resolved


def dump_task_document(task_doc: dict[str, Any]) -> str:
    """Render a task document to YAML for prompt-grounding/debugging."""
    return yaml.safe_dump(task_doc, sort_keys=False, allow_unicode=False)


def _detect_category(yaml_path: Path) -> str | None:
    parts = yaml_path.parts
    for idx, part in enumerate(parts):
        if part == "tasks" and idx + 2 < len(parts):
            return parts[idx + 2]
    return None


def _normalize_asset_paths(task_doc: dict[str, Any]) -> None:
    """Normalize repo-local asset paths and stale assembly asset URLs."""
    assets = task_doc.get("assets", [])
    for asset in assets:
        asset_path = asset.get("asset_path")
        if isinstance(asset_path, str) and asset_path.startswith("assets/"):
            local_asset = PROJECT_ROOT / asset_path
            if local_asset.exists():
                asset["asset_path"] = str(local_asset.resolve())

        asset_url = asset.get("asset_url")
        if not asset_url or "assembling_kits" not in asset_url:
            continue
        filename = Path(asset_url).name
        local_asset = PROJECT_ROOT / "assets" / "assembling_kits" / filename
        if local_asset.exists():
            asset.pop("asset_url", None)
            asset["asset_path"] = f"assets/assembling_kits/{filename}"


def _canonicalize_ur10_assembly(task_doc: dict[str, Any]) -> None:
    """Backfill external legacy UR10 assembly specs to the canonical UR10e form."""
    for asset in task_doc.get("assets", []):
        if asset.get("type") != "articulation":
            continue
        asset_path = str(asset.get("asset_path", ""))
        if UR10_ASSET_PATH not in asset_path and "/ur10/ur10.usd" not in asset_path:
            continue

        asset["robot_type"] = "ur10e"
        asset["asset_path"] = UR10E_ASSET_PATH
        asset["variant_sets"] = {"Gripper": "Robotiq_2f_85"}

        ee_frame = asset.setdefault("ee_frame", {})
        ee_frame["body"] = "wrist_3_link"
        ee_frame["offset_position"] = [0.0, 0.0, 0.0]

        task = task_doc.get("task", {})
        task_name = task.get("name")
        if isinstance(task_name, str) and task_name.startswith("UR10"):
            task["name"] = task_name.replace("UR10", "UR10e", 1)

        task_desc = task.get("description")
        if isinstance(task_desc, str) and "UR10 robot" in task_desc:
            task["description"] = task_desc.replace(
                "UR10 robot with suction gripper",
                "UR10e robot with Robotiq 2F-85 gripper",
            )

        notes = task_doc.setdefault("notes", [])
        note = "Backward-compatibility: external legacy UR10 assembly spec canonicalized to UR10e + Robotiq 2F-85."
        if note not in notes:
            notes.append(note)


def _resolve_assembly_conditions(task_doc: dict[str, Any], yaml_path: Path) -> None:
    """Attach concrete targets for assembly-specific placeholders and relations."""
    if "assembling_kits" in yaml_path.stem:
        _resolve_assembling_kits(task_doc)

    assets_by_name = {
        asset.get("name"): asset for asset in task_doc.get("assets", []) if asset.get("name")
    }
    for cond in task_doc.get("goal", {}).get("conditions", []):
        relation = cond.get("relation")
        target_name = cond.get("target")
        target_asset = assets_by_name.get(target_name)
        if relation == "inserted_into" and target_asset:
            if "target_position" not in cond and target_asset.get("position"):
                cond["target_position"] = list(target_asset["position"])
            if "target_rotation" not in cond and target_asset.get("rotation"):
                cond["target_rotation"] = list(target_asset["rotation"])


def _resolve_assembling_kits(task_doc: dict[str, Any]) -> None:
    """Materialize the active movable shape and its slot target pose."""
    assets = task_doc.get("assets", [])
    tray = next((asset for asset in assets if asset.get("name") == "kit_tray"), None)
    active_shape = next(
        (
            asset
            for asset in assets
            if asset.get("type") == "rigid" and asset.get("name", "").startswith("shape_")
        ),
        None,
    )
    if tray is None or active_shape is None:
        return

    _merge_shape_to_place_config(task_doc, active_shape)
    target_position, target_rotation = _compute_slot_target_pose(tray, active_shape)

    episode = task_doc.setdefault("episode", {})
    episode["resolved_shape_to_place"] = active_shape["name"]
    if active_shape.get("object_id") is not None:
        episode["resolved_shape_to_place_id"] = active_shape["object_id"]

    for cond in task_doc.get("goal", {}).get("conditions", []):
        if cond.get("subject") == "shape_to_place":
            cond["subject"] = active_shape["name"]
        if cond.get("target") == "matching_cutout":
            cond["target"] = tray["name"]
            cond["target_position"] = target_position
            cond["target_rotation"] = target_rotation
            cond["resolved_target_shape_name"] = active_shape.get("shape_name")
            cond["resolved_target_object_id"] = active_shape.get("object_id")


def _merge_shape_to_place_config(task_doc: dict[str, Any], active_shape: dict[str, Any]) -> None:
    """Apply the shared shape-to-place config to the active rigid shape."""
    config = task_doc.get("shape_to_place_config")
    if not isinstance(config, dict):
        return

    if "spawn_z" in config:
        pos = list(active_shape.get("position", [0.0, 0.0, 0.0]))
        while len(pos) < 3:
            pos.append(0.0)
        pos[2] = float(config["spawn_z"])
        active_shape["position"] = pos

    physics = active_shape.setdefault("physics", {})
    for key, value in config.get("physics", {}).items():
        physics[key] = deepcopy(value)

    config_randomize = config.get("randomize", {})
    if isinstance(config_randomize, dict) and config_randomize:
        randomize = active_shape.setdefault("randomize", {})
        _deep_update(randomize, config_randomize)


def _deep_update(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Recursively merge source into target."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def _compute_slot_target_pose(tray: dict[str, Any], shape_asset: dict[str, Any]) -> tuple[list[float], list[float]]:
    """Return world-space target pose for an assembling-kits cutout."""
    tray_pos = tray.get("position", [0.0, 0.0, 0.0])
    tray_scale = tray.get("scale", [1.0, 1.0, 1.0])
    tray_rot = tray.get("rotation", [1.0, 0.0, 0.0, 0.0])

    local_pos = shape_asset.get("slot_position_local", [0.0, 0.0, 0.0])
    local_pos = [
        float(local_pos[0]) * float(tray_scale[0]),
        float(local_pos[1]) * float(tray_scale[1]),
        float(local_pos[2]) * float(tray_scale[2]),
    ]
    world_offset = _quat_apply(tray_rot, local_pos)
    world_pos = [float(tray_pos[i]) + world_offset[i] for i in range(3)]

    yaw = float(shape_asset.get("slot_rotation_z", 0.0))
    slot_rot = [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]
    world_rot = _quat_multiply(tray_rot, slot_rot)

    return _round_vec(world_pos), _round_vec(world_rot)


def _quat_multiply(q1: list[float], q2: list[float]) -> list[float]:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def _quat_apply(quat_wxyz: list[float], vec_xyz: list[float]) -> list[float]:
    q_vec = [0.0, float(vec_xyz[0]), float(vec_xyz[1]), float(vec_xyz[2])]
    q_conj = [float(quat_wxyz[0]), -float(quat_wxyz[1]), -float(quat_wxyz[2]), -float(quat_wxyz[3])]
    rotated = _quat_multiply(_quat_multiply(quat_wxyz, q_vec), q_conj)
    return rotated[1:]


def _round_vec(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(v), digits) for v in values]
