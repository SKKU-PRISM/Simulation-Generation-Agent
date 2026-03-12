"""ADC-style forward-only CaP generator for IsaacLab data collection."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import math
from pathlib import PurePosixPath
from pprint import pformat
import re
from typing import Any

import numpy as np

from .config import RobotSimConfig

SUPPORTED_RELATIONS = {
    "on_top_of",
    "stacked",
    "at_position",
    "on_surface",
    "inside_tray",
    "inside_drawer",
    "above",
    "height_above",
    "position_above",
    "position_below",
    "inserted_into",
    "height_below",
    "upright",
    "in_slot",
}
REFUSAL_MARKERS = (
    "i'm sorry",
    "i cannot assist",
    "i can’t assist",
    "cannot assist with that request",
    "can't assist with that request",
)


def _collect_goal_conditions(task_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize goal conditions across task schemas."""

    goal = task_doc.get("goal", {})
    success_criteria = goal.get("success_criteria", {})
    conditions: list[dict[str, Any]] = []

    if isinstance(success_criteria, dict):
        for cond in success_criteria.get("conditions", []):
            if isinstance(cond, dict):
                conditions.append(dict(cond))
        stacking_order = success_criteria.get("stacking_order")
        if isinstance(stacking_order, list):
            for idx in range(1, len(stacking_order)):
                top = stacking_order[idx]
                bottom = stacking_order[idx - 1]
                if isinstance(top, str) and isinstance(bottom, str):
                    conditions.append(
                        {
                            "subject": top,
                            "target": bottom,
                            "relation": "on_top_of",
                        }
                    )

    for cond in goal.get("conditions", []):
        if isinstance(cond, dict):
            conditions.append(dict(cond))
    return conditions


def _is_primary_robot_asset(asset: dict[str, Any]) -> bool:
    """Best-effort detection for the task's main robot articulation."""

    if asset.get("robot_type") is not None:
        return True

    name = str(asset.get("name", "")).lower()
    prim_path = str(asset.get("prim_path", "")).lower()
    return (
        name == "robot"
        or prim_path.endswith("/robot")
        or "initial_joints" in asset
        or "ee_frame" in asset
    )


def _goal_target_names(task_doc: dict[str, Any]) -> set[str]:
    """Collect goal target asset names referenced by the task."""

    targets: set[str] = set()
    for cond in _collect_goal_conditions(task_doc):
        target = cond.get("target")
        if isinstance(target, str):
            targets.add(target)
    return targets


def _is_named_placement_target(name: str, asset: dict[str, Any]) -> bool:
    """Return whether an asset should be exposed as a named placement target."""

    name_lower = name.lower()
    if "wall" in name_lower:
        return False
    if any(
        token in name_lower
        for token in (
            "zone",
            "marker",
            "tray",
            "bin",
            "anchor",
            "target",
            "collection",
            "command_pose",
            "handle",
            "hole",
            "slot",
            "cutout",
            "receptacle",
        )
    ):
        return True
    if "pose" in name_lower and asset.get("type") in {"rigid", "primitive", "static"}:
        return True
    return False


def _support_target_names(task_doc: dict[str, Any]) -> set[str]:
    """Collect static or quasi-static placement targets that should reach the prompt."""

    targets = _goal_target_names(task_doc)
    success_criteria = task_doc.get("goal", {}).get("success_criteria", {})

    for asset in task_doc.get("assets", []):
        name = asset.get("name")
        if not isinstance(name, str):
            continue
        if _is_named_placement_target(name, asset):
            targets.add(name)

    if isinstance(success_criteria, dict) and success_criteria.get("inside_tray"):
        for asset in task_doc.get("assets", []):
            name = asset.get("name")
            if not isinstance(name, str):
                continue
            if "tray_base" in name.lower():
                targets.add(name)
    return targets


def _goal_subject_names(task_doc: dict[str, Any]) -> set[str]:
    """Collect goal subject asset names referenced by the task."""

    subjects: set[str] = set()
    for cond in _collect_goal_conditions(task_doc):
        subject = cond.get("subject", cond.get("object"))
        if isinstance(subject, str):
            subjects.add(subject)
    return subjects


def _infer_ordered_objects(
    task_description: str,
    translated_positions: dict[str, dict[str, Any]],
) -> list[str]:
    """Infer manipulation order from object names mentioned in the task description."""

    def _candidate_index(description: str, candidate: str) -> int:
        pattern = rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])"
        match = re.search(pattern, description)
        return match.start() if match else -1

    description = task_description.lower()
    ordered: list[tuple[int, str]] = []
    seen: set[str] = set()
    for name, info in translated_positions.items():
        task_role = str(info.get("task_role", ""))
        if task_role in {"placement_target", "target_marker", "support_surface", "goal_target"}:
            continue
        candidates = [name.lower()]
        for key in ("color_name", "asset_label"):
            value = info.get(key)
            if isinstance(value, str):
                candidates.append(value.lower())
        for alias in info.get("aliases") or []:
            if isinstance(alias, str):
                candidates.append(alias.lower())

        candidate_indices = [_candidate_index(description, candidate) for candidate in candidates if candidate]
        candidate_indices = [idx for idx in candidate_indices if idx >= 0]
        if candidate_indices:
            key = (min(candidate_indices), name)
            if name not in seen:
                ordered.append(key)
                seen.add(name)
    ordered.sort()
    return [name for _, name in ordered]


def _infer_color_name(asset: dict[str, Any], name: str) -> str | None:
    """Infer a coarse color label from asset metadata."""

    searchable = " ".join(
        str(x)
        for x in (
            name,
            asset.get("asset_path", ""),
            asset.get("semantic_tag", ""),
            asset.get("semantic_label", ""),
        )
    ).lower()
    for color in (
        "blue",
        "red",
        "green",
        "yellow",
        "orange",
        "purple",
        "rose",
        "pink",
        "gold",
        "white",
        "black",
        "gray",
        "grey",
        "brown",
    ):
        if color in searchable:
            return color
    return None


def _infer_asset_label(asset: dict[str, Any], name: str, shape_type: str) -> str:
    """Build a compact natural-language label for an asset."""

    color_name = _infer_color_name(asset, name)
    tokens = [color_name, shape_type if shape_type != "unknown" else None]
    asset_label = " ".join(token for token in tokens if token).strip()
    if asset_label and not (color_name and shape_type == "unknown"):
        return asset_label

    asset_path = str(asset.get("asset_path", "")).strip()
    if asset_path:
        stem = asset_path.rsplit("/", 1)[-1].replace(".usd", "").replace("_", " ")
        if stem:
            return stem
    return name.replace("_", " ")


def _infer_task_role(
    name: str,
    asset: dict[str, Any],
    subject_names: set[str],
    target_names: set[str],
) -> str:
    """Infer a coarse task role for prompt grounding."""

    name_lower = name.lower()
    asset_type = str(asset.get("type", "")).lower()

    if name in subject_names and name in target_names:
        return "subject_and_target"
    if name in subject_names:
        return "goal_subject"
    if "handle" in name_lower:
        return "handle_target"
    if any(token in name_lower for token in ("hole", "slot", "cutout", "receptacle")):
        return "insertion_target"
    if "tray" in name_lower or "bin" in name_lower or "zone" in name_lower or "marker" in name_lower:
        return "placement_target"
    if name in target_names:
        if "tray" in name_lower or "bin" in name_lower or "zone" in name_lower:
            return "placement_target"
        if "marker" in name_lower or "pose" in name_lower:
            return "target_marker"
        if asset_type == "static" or "table" in name_lower:
            return "support_surface"
        return "goal_target"
    if asset_type == "static":
        return "support_surface"
    return "movable_object"


def _summarize_goal_roles(name: str, conditions: list[dict[str, Any]]) -> list[str]:
    """Summarize how an object participates in goal conditions."""

    roles: list[str] = []
    for cond in conditions:
        relation = str(cond.get("relation", cond.get("type", "")) or "").lower()
        subject = cond.get("subject", cond.get("object"))
        target = cond.get("target")
        if subject == name and relation:
            roles.append(f"subject:{relation}")
        if target == name and relation:
            roles.append(f"target:{relation}")
    return sorted(set(roles))


def _attach_task_role_metadata(
    task_doc: dict[str, Any],
    translated: dict[str, dict[str, Any]],
    assets_by_name: dict[str, dict[str, Any]],
) -> None:
    """Attach task-role and alias metadata derived from the YAML goal."""

    conditions = _collect_goal_conditions(task_doc)
    capability = assess_task_capability(task_doc)
    subject_names = _goal_subject_names(task_doc)
    target_names = _goal_target_names(task_doc)

    for name, info in translated.items():
        asset = assets_by_name.get(name, {})
        color_name = _infer_color_name(asset, name)
        shape_type = str(info.get("shape_type", "unknown"))
        aliases = [
            alias
            for alias in (
                name.replace("_", " "),
                _infer_asset_label(asset, name, shape_type),
                f"{color_name} {shape_type}" if color_name and shape_type != "unknown" else None,
            )
            if alias
        ]
        deduped_aliases: list[str] = []
        for alias in aliases:
            if alias not in deduped_aliases:
                deduped_aliases.append(alias)

        info["color_name"] = color_name
        info["asset_label"] = _infer_asset_label(asset, name, shape_type)
        info["aliases"] = deduped_aliases
        info["task_role"] = _infer_task_role(name, asset, subject_names, target_names)
        info["goal_roles"] = _summarize_goal_roles(name, conditions)


@dataclass(frozen=True)
class CaPProfile:
    """Embodiment-specific code generation profile."""

    robot_name: str
    class_name: str
    module_name: str
    ready_method: str
    closing_method: str
    approach_height: float
    api_doc: str


@dataclass
class CaPGenerationResult:
    """Generated code plus the translated positions used for prompting."""

    profile: CaPProfile
    translated_positions: dict[str, dict[str, Any]]
    generated_code: str
    raw_response: str


@dataclass(frozen=True)
class TaskCapability:
    """High-level affordance family inferred from the YAML goal."""

    family: str
    supported: bool
    reason: str


@dataclass(frozen=True)
class TaskSkillPreflight:
    """Decision-complete skill usage contract for a task family."""

    task_name: str
    family: str
    supported: bool
    reason: str
    primary_skill: str | None
    secondary_skills: tuple[str, ...]
    required_relations: tuple[str, ...]
    required_targets: tuple[str, ...]


_PRIMARY_SKILL_BY_FAMILY: dict[str, str | None] = {
    "tabletop_transfer": "execute_pick_and_place_on_target",
    "container_transfer": "execute_pick_and_place_in_container",
    "container_stack": "execute_pick_and_place_in_container",
    "stack": "execute_pick_and_stack_on_object",
    "lift_hold": "execute_pick_and_lift_to_pose",
    "support_surface_transfer": "execute_pick_and_place_on_support",
    "articulated_container_transfer": "execute_pull_handle_open",
    "articulated_pull": "execute_pull_handle_open",
    "upright_placement": "execute_pick_and_place_upright",
    "axial_insertion": "execute_pick_and_insert_into_target",
    "slot_fit": "execute_pick_and_fit_into_slot",
    "unsupported": None,
}

_SECONDARY_SKILLS_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "tabletop_transfer": ("execute_pick_object", "execute_place_object"),
    "container_transfer": ("execute_place_in_container",),
    "container_stack": ("execute_pick_and_stack_on_object", "execute_stack_on_object"),
    "stack": ("execute_stack_on_object",),
    "lift_hold": ("execute_lift_to_pose",),
    "support_surface_transfer": ("execute_pick_and_place_on_target",),
    "articulated_container_transfer": ("execute_pick_and_place_in_container", "execute_push_handle_closed"),
    "articulated_pull": (),
    "upright_placement": ("execute_pick_and_place_on_support",),
    "axial_insertion": (),
    "slot_fit": (),
    "unsupported": (),
}


def _infer_required_target_names(task_doc: dict[str, Any], family: str) -> tuple[str, ...]:
    """Return the named targets a task family expects to see in translated positions."""

    target_names = set(_goal_target_names(task_doc))
    success_criteria = task_doc.get("goal", {}).get("success_criteria", {})
    asset_names = {
        str(asset.get("name"))
        for asset in task_doc.get("assets", [])
        if isinstance(asset.get("name"), str)
    }

    if family == "lift_hold":
        if any(name.lower() == "command_pose" for name in target_names):
            target_names.add("command_pose")
    elif family in {"container_transfer", "container_stack"}:
        if isinstance(success_criteria, dict) and success_criteria.get("inside_tray"):
            target_names.add("tray_anchor")
    elif family == "articulated_container_transfer":
        target_names.add("drawer_handle_top")
        target_names.add("drawer_container")
    elif family == "support_surface_transfer":
        if "table" in target_names:
            target_names.add("table_surface")
        if any("drawer" in name.lower() for name in asset_names | target_names):
            target_names.add("drawer_top_surface")
    elif family == "articulated_pull":
        if any("drawer" in name.lower() or "cabinet" in name.lower() for name in asset_names):
            target_names.add("drawer_handle_top")
    elif family == "upright_placement":
        if "table" in target_names or any(name.lower() == "table" for name in asset_names):
            target_names.add("table_surface")
    elif family == "slot_fit":
        target_names.add("matching_cutout")

    return tuple(sorted(name for name in target_names if isinstance(name, str) and name))


def build_task_skill_preflight(task_doc: dict[str, Any]) -> TaskSkillPreflight:
    """Return the preferred skill usage contract for the task."""

    capability = assess_task_capability(task_doc)
    relations = sorted(
        {
            str(cond.get("type", cond.get("relation", "")) or "").lower()
            for cond in _collect_goal_conditions(task_doc)
            if isinstance(cond, dict)
        }
    )
    task_name = str(task_doc.get("task", {}).get("name", "UnknownTask"))
    family = capability.family
    return TaskSkillPreflight(
        task_name=task_name,
        family=family,
        supported=capability.supported,
        reason=capability.reason,
        primary_skill=_PRIMARY_SKILL_BY_FAMILY.get(family),
        secondary_skills=_SECONDARY_SKILLS_BY_FAMILY.get(family, ()),
        required_relations=tuple(relations),
        required_targets=_infer_required_target_names(task_doc, family),
    )


def extract_skill_calls_from_code(generated_code: str) -> tuple[str, ...]:
    """Return unique `skills.<method>(...)` calls in source order."""

    class _SkillCallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.calls: list[str] = []

        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            func = node.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "skills":
                    self.calls.append(func.attr)
            self.generic_visit(node)

    tree = ast.parse(generated_code)
    visitor = _SkillCallVisitor()
    visitor.visit(tree)

    ordered: list[str] = []
    seen: set[str] = set()
    for call in visitor.calls:
        if call not in seen:
            seen.add(call)
            ordered.append(call)
    return tuple(ordered)


def _format_preferred_skill_usage(preflight: TaskSkillPreflight) -> str:
    """Render the primary/secondary skill contract for prompt grounding."""

    lines = [f"- Capability family: `{preflight.family}`"]
    if preflight.primary_skill:
        lines.append(f"- Primary skill: `{preflight.primary_skill}`")
    if preflight.secondary_skills:
        secondary = ", ".join(f"`{name}`" for name in preflight.secondary_skills)
        lines.append(f"- Secondary skills: {secondary}")
    if preflight.required_targets:
        target_text = ", ".join(f"`{name}`" for name in preflight.required_targets)
        lines.append(f"- Named targets expected in `positions`: {target_text}")
    if preflight.required_relations:
        relation_text = ", ".join(f"`{name}`" for name in preflight.required_relations)
        lines.append(f"- Goal relations to satisfy: {relation_text}")
    return "\n".join(lines)


def _composite_insertion_aliases(
    translated_positions: dict[str, dict[str, Any]],
) -> list[tuple[str, str]]:
    """Return ``(alias_name, subject_name)`` pairs for composite insertion objects."""

    aliases: list[tuple[str, str]] = []
    for name, info in translated_positions.items():
        if not isinstance(info, dict):
            continue
        subject_name = info.get("insertion_subject_name")
        if isinstance(subject_name, str) and subject_name:
            aliases.append((name, subject_name))
    return aliases


def select_cap_profile(robot_cfg: RobotSimConfig) -> CaPProfile:
    """Return the CaP profile for the requested robot."""

    tabletop_api = """
- `connect()` / `disconnect()` — lifecycle no-ops in simulation, but always call them.
- `move_to_ready(skill_description=None)` — move to the robot's standard ready pose.
- `move_to_position(position, target_name=None, skill_description=None)` — move the tool center to a world-frame XYZ target.
- `gripper_open(skill_description=None, ratio=1.0)` — open the gripper. Use `ratio=1.0`.
- `gripper_close(skill_description=None)` — close the gripper while holding arm pose.
- `execute_pick_object(object_position, object_name=None, skill_description=None)` — descend to the object's grasp height and grasp it. Always pass the explicit `object_name`.
- `execute_place_object(place_position, target_name=None, is_table=True, skill_description=None)` — descend to the placement height and release the held object. Always pass `target_name` for named targets; it is required when `is_table=False`.
- `execute_place_on_target(target_name, skill_description=None)` — place the currently held object onto a named tabletop target such as a marker, zone, anchor, or support surface.
- `execute_place_in_container(container_name, skill_description=None)` — place the currently held object into a named tray, bin, or container anchor.
- `execute_stack_on_object(bottom_object_name, skill_description=None)` — place the currently held object on top of another object.
- `execute_pick_and_place_on_target(object_name, target_name, skill_description=None)` — pick a named object and place it onto a named tabletop target.
- `execute_pick_and_place_in_container(object_name, container_name, skill_description=None)` — pick a named object and place it into a tray/bin/container target.
- `execute_pick_and_stack_on_object(object_name, bottom_object_name, skill_description=None)` — pick a named object and stack it onto another named object.
- `execute_lift_to_pose(target_name, skill_description=None)` — move the currently held object to a named commanded pose while keeping the gripper closed.
- `execute_pick_and_lift_to_pose(object_name, target_name, skill_description=None)` — pick a named object and lift/hold it at a named commanded pose.
- `execute_pick_and_place_on_support(object_name, support_name, skill_description=None)` — pick a named object and place it onto a named elevated support surface.
- `execute_pull_handle_open(handle_name, open_fraction=None, skill_description=None)` — grasp a named handle and pull it along its configured open axis.
- `execute_set_handle_open_fraction(handle_name, open_fraction, skill_description=None)` — convenience handle API for articulated joints; drawer tasks should prefer explicit open/close skills.
- `execute_push_handle_closed(handle_name, skill_description=None)` — grasp a named handle and push it along its configured closing axis until the drawer closes.
- `execute_pick_and_place_upright(object_name, support_name, skill_description=None)` — pick a named object and place it upright on a named support surface.
- `execute_pick_and_insert_into_target(object_name, target_name, skill_description=None)` — pick a named object, align it with a named insertion target, and insert it along the target axis.
- `execute_pick_and_fit_into_slot(object_name, target_name, skill_description=None)` — pick a named object and place it flush into a named slot or cutout target.
"""
    if robot_cfg.name == "franka":
        return CaPProfile(
            robot_name="franka",
            class_name="FrankaSkills",
            module_name="skills.skills_franka",
            ready_method="move_to_ready",
            closing_method="move_to_ready",
            approach_height=0.18,
            api_doc=tabletop_api,
        )
    if robot_cfg.name == "openarm":
        return CaPProfile(
            robot_name="openarm",
            class_name="OpenArmSkills",
            module_name="skills.skills_openarm",
            ready_method="move_to_ready",
            closing_method="move_to_ready",
            approach_height=0.16,
            api_doc=tabletop_api,
        )
    if robot_cfg.name == "ur10e":
        return CaPProfile(
            robot_name="ur10e",
            class_name="UR10eSkills",
            module_name="skills.skills_ur10e",
            ready_method="move_to_ready",
            closing_method="move_to_ready",
            approach_height=0.18,
            api_doc=tabletop_api,
        )
    if robot_cfg.name == "so101":
        return CaPProfile(
            robot_name="so101",
            class_name="LeRobotSkills",
            module_name="skills.skills_lerobot",
            ready_method="move_to_initial_state",
            closing_method="move_to_free_state",
            approach_height=0.15,
            api_doc="""
- `connect()` / `disconnect()` — lifecycle calls.
- `move_to_initial_state(skill_description=None)` — move to the robot's initial pose.
- `move_to_position(position, target_name=None, skill_description=None)` — move the tool center to a world-frame XYZ target.
- `gripper_open(skill_description=None, ratio=1.0)` — open the gripper. Use `ratio=1.0`.
- `gripper_close(skill_description=None)` — close the gripper while holding arm pose.
- `rotate_90degree(direction=1, skill_description=None)` — rotate the wrist by 90 degrees. Use only when strictly needed.
- `move_to_free_state(skill_description=None)` — move to the robot's safe free pose.
- `execute_pick_object(object_position, object_name=None, skill_description=None)` — descend to the object's grasp height and grasp it. Always pass the explicit `object_name`.
- `execute_place_object(place_position, target_name=None, is_table=True, skill_description=None)` — descend to the placement height and release the held object. Always pass `target_name` for named targets; it is required when `is_table=False`.
""",
        )
    raise ValueError(f"Unsupported CaP profile for robot '{robot_cfg.name}'")


def assess_task_capability(task_doc: dict[str, Any]) -> TaskCapability:
    """Infer the task's primary affordance family and current support status."""

    goal = task_doc.get("goal", {})
    conditions = _collect_goal_conditions(task_doc)
    success_criteria = goal.get("success_criteria", {})
    task_info = task_doc.get("task", {})
    goal_target_names = {name.lower() for name in _goal_target_names(task_doc)}
    condition_relations = {
        str(cond.get("type", cond.get("relation", "")) or "").lower()
        for cond in conditions
        if isinstance(cond, dict)
    }
    description_text = " ".join(
        str(value)
        for value in (
            task_info.get("name", ""),
            task_info.get("description", ""),
            goal.get("description", ""),
        )
        if value
    ).lower()
    mentions_stack = "stack" in description_text
    has_articulated_container = any(
        asset.get("type") == "articulation"
        and isinstance(asset.get("handle"), dict)
        and isinstance(asset.get("container"), dict)
        for asset in task_doc.get("assets", [])
    )

    if has_articulated_container and (
        "inside_drawer" in condition_relations
        or ("position_below" in condition_relations and any("container" in target_name for target_name in goal_target_names))
    ):
        return TaskCapability(
            "articulated_container_transfer",
            True,
            "supported articulated open-place-close transfer",
        )

    if isinstance(success_criteria, dict) and success_criteria.get("inside_tray"):
        if mentions_stack:
            return TaskCapability("container_stack", True, "supported tray/container stacking")
        if condition_relations & {"on_top_of", "stacked"}:
            return TaskCapability("container_stack", True, "supported tray/container stacking")
        return TaskCapability("container_transfer", True, "supported tray/bin placement")

    if any(
        any(token in target_name for token in ("tray", "bin", "container"))
        for target_name in goal_target_names
    ):
        if mentions_stack or condition_relations & {"on_top_of", "stacked"}:
            return TaskCapability("container_stack", True, "supported tray/container stacking")
        return TaskCapability("container_transfer", True, "supported tray/bin placement")

    if "in_slot" in condition_relations:
        return TaskCapability("slot_fit", True, "supported slot/cutout fitting")

    if any(rel in {"inserted_into", "height_below"} for rel in condition_relations):
        return TaskCapability("axial_insertion", True, "supported alignment/insertion")

    if "upright" in condition_relations:
        return TaskCapability("upright_placement", True, "supported upright placement")

    if "position_above" in condition_relations:
        return TaskCapability("articulated_pull", True, "supported articulated pull")

    if "height_above" in condition_relations:
        return TaskCapability("lift_hold", True, "supported lift/hold relation")

    if "above" in condition_relations:
        return TaskCapability("support_surface_transfer", True, "supported elevated support placement")

    if mentions_stack:
        return TaskCapability("stack", True, "supported stacking order")

    if condition_relations & {"on_top_of", "stacked"}:
        return TaskCapability("stack", True, "supported stacking order")

    if not conditions:
        return TaskCapability("tabletop_transfer", True, "no explicit unsupported conditions")

    for cond in conditions:
        relation = str(cond.get("type", cond.get("relation", "")) or "").lower()
        if relation and relation not in SUPPORTED_RELATIONS:
            return TaskCapability("unsupported", False, f"unsupported goal relation '{relation}'")

    return TaskCapability("tabletop_transfer", True, "supported tabletop relations")


def is_supported_tabletop_task(task_doc: dict[str, Any]) -> tuple[bool, str]:
    """Backward-compatible support check used by the generator and tests."""

    capability = assess_task_capability(task_doc)
    return capability.supported, capability.reason


def translate_scene_state(
    task_doc: dict[str, Any],
    scene_state: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Translate scene graph poses into the ADC-style extended positions format."""

    assets_by_name = {
        asset.get("name"): asset
        for asset in task_doc.get("assets", [])
        if asset.get("name")
    }

    translated: dict[str, dict[str, Any]] = {}
    goal_targets = _support_target_names(task_doc)
    goal_subjects = _goal_subject_names(task_doc)
    names_to_include = set(scene_state.keys()) | goal_targets | goal_subjects
    capability = assess_task_capability(task_doc)
    if capability.family == "slot_fit":
        names_to_include |= {
            name
            for name, asset in assets_by_name.items()
            if asset.get("type") == "rigid"
        }

    for name in names_to_include:
        asset = assets_by_name.get(name, {})
        asset_type = asset.get("type", "")
        if name == "table" or _is_primary_robot_asset(asset):
            continue
        if asset_type not in {"rigid", "primitive", "static"}:
            continue
        if asset_type == "static" and name not in goal_targets:
            continue

        info = scene_state.get(name)
        if info is None:
            asset_position = asset.get("position")
            if not isinstance(asset_position, (list, tuple)) or len(asset_position) != 3:
                continue
            info = {
                "position": list(asset_position),
                "quaternion": asset.get("rotation"),
            }

        position = list(info.get("position", [0.0, 0.0, 0.0]))
        if len(position) != 3:
            continue
        asset_position = asset.get("position")
        if (
            isinstance(asset_position, (list, tuple))
            and len(asset_position) == 3
            and float(asset_position[2]) >= 0.0
            and float(position[2]) < max(float(asset_position[2]) - 0.05, -0.02)
        ):
            position = [
                float(asset_position[0]),
                float(asset_position[1]),
                float(asset_position[2]),
            ]

        half_height = _estimate_half_height(asset, position)
        half_extents = _estimate_half_extents(asset, position, half_height)
        translated[name] = {
            "position": [
                float(position[0]),
                float(position[1]),
                float(position[2] + half_height),
            ],
            "gripper_offset": 0.0,
            "estimated_half_height": float(half_height),
            "estimated_half_extents": [float(v) for v in half_extents],
            "shape_type": _infer_shape_type(asset),
            "quaternion": info.get("quaternion"),
        }

    _attach_topdown_affordances(translated)
    _attach_task_role_metadata(task_doc, translated, assets_by_name)
    _inject_synthetic_targets(task_doc, translated, assets_by_name)
    return translated


def _inject_synthetic_targets(
    task_doc: dict[str, Any],
    translated: dict[str, dict[str, Any]],
    assets_by_name: dict[str, dict[str, Any]],
) -> None:
    """Inject synthetic target anchors needed for prompt grounding and verification."""

    goal = task_doc.get("goal", {})
    success_criteria = task_doc.get("goal", {}).get("success_criteria", {})
    capability = assess_task_capability(task_doc)
    if isinstance(success_criteria, dict) and success_criteria.get("inside_tray"):
        tray_center = success_criteria.get("tray_center")
        if isinstance(tray_center, (list, tuple)) and len(tray_center) == 2:
            tray_half_extent = float(success_criteria.get("tray_half_extent", 0.0))

            tray_top_z = 0.0
            for asset_name, asset in assets_by_name.items():
                if "tray_base" not in asset_name.lower():
                    continue
                asset_position = asset.get("position")
                if not isinstance(asset_position, (list, tuple)) or len(asset_position) != 3:
                    continue
                half_height = _estimate_half_height(asset, list(asset_position))
                tray_top_z = float(asset_position[2] + half_height)
                break

            translated["tray_anchor"] = {
                "position": [float(tray_center[0]), float(tray_center[1]), tray_top_z],
                "gripper_offset": 0.0,
                "estimated_half_height": 0.0,
                "estimated_half_extents": [tray_half_extent, tray_half_extent, 0.0],
                "shape_type": "target_anchor",
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "color_name": None,
                "asset_label": "tray anchor",
                "aliases": ["tray anchor", "tray center"],
                "task_role": "placement_target",
                "goal_roles": ["target:inside_tray"],
                "affordances": {},
            }

    for target_name in _goal_target_names(task_doc):
        if not isinstance(target_name, str):
            continue
        lowered = target_name.lower()
        if not any(token in lowered for token in ("tray", "bin", "container", "collection")):
            continue
        anchor_name = f"{target_name}_anchor"
        if anchor_name in translated:
            continue
        target_entry = translated.get(target_name)
        if not isinstance(target_entry, dict):
            continue
        anchor_pos = list(target_entry["position"])
        translated[anchor_name] = {
            "position": [float(anchor_pos[0]), float(anchor_pos[1]), float(anchor_pos[2])],
            "gripper_offset": 0.0,
            "estimated_half_height": 0.0,
            "estimated_half_extents": [0.0, 0.0, 0.0],
            "shape_type": "target_anchor",
            "quaternion": [1.0, 0.0, 0.0, 0.0],
            "color_name": None,
            "asset_label": f"{target_name.replace('_', ' ')} anchor",
            "aliases": [
                f"{target_name.replace('_', ' ')} anchor",
                f"{target_name.replace('_', ' ')} center",
            ],
            "task_role": "placement_target",
            "goal_roles": ["target:inside_tray", "target:at_position"],
            "affordances": {},
            "support_asset_name": target_name,
            "support_alignment_target": target_name,
            "support_top_z": float(anchor_pos[2]),
        }

    needs_command_pose = any(
        isinstance(cond, dict) and str(cond.get("target", "")).lower() == "command_pose"
        for cond in _collect_goal_conditions(task_doc)
    )
    command_ranges = goal.get("command_ranges", {})
    position_ranges = command_ranges.get("position", {}) if isinstance(command_ranges, dict) else {}
    if needs_command_pose and isinstance(position_ranges, dict):
        axes = []
        for axis in ("x", "y", "z"):
            axis_range = position_ranges.get(axis)
            if isinstance(axis_range, (list, tuple)) and len(axis_range) == 2:
                axes.append((float(axis_range[0]) + float(axis_range[1])) * 0.5)
            else:
                axes = []
                break
        if len(axes) == 3:
            translated["command_pose"] = {
                "position": axes,
                "gripper_offset": 0.0,
                "estimated_half_height": 0.0,
                "estimated_half_extents": [0.0, 0.0, 0.0],
                "shape_type": "target_anchor",
                "quaternion": [1.0, 0.0, 0.0, 0.0],
                "color_name": None,
                "asset_label": "command pose",
                "aliases": ["command pose", "target pose"],
                "task_role": "placement_target",
                "goal_roles": ["target:command_pose"],
                "affordances": {},
            }

    conditions = _collect_goal_conditions(task_doc)

    def _quat_to_rotation_matrix(quat_wxyz) -> np.ndarray:
        quat = np.asarray(quat_wxyz if quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _rotation_matrix_to_quat(rotation: np.ndarray) -> list[float]:
        rot = np.asarray(rotation, dtype=np.float64)
        trace = float(np.trace(rot))
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rot[2, 1] - rot[1, 2]) / s
            y = (rot[0, 2] - rot[2, 0]) / s
            z = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
        quat = np.array([w, x, y, z], dtype=np.float64)
        quat /= max(np.linalg.norm(quat), 1e-9)
        return quat.tolist()

    def _rotate_vector(quat_wxyz, vector) -> np.ndarray:
        rotation = _quat_to_rotation_matrix(quat_wxyz)
        return rotation @ np.asarray(vector, dtype=np.float64)

    def _rotation_with_tool_z(z_axis_world: np.ndarray) -> np.ndarray:
        """Build a right-handed world rotation whose tool-Z points along ``z_axis_world``."""

        z_axis = np.asarray(z_axis_world, dtype=np.float64)
        z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-9)
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = up_hint - z_axis * float(np.dot(up_hint, z_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y_axis = up_hint - z_axis * float(np.dot(up_hint, z_axis))
        y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-9)
        return np.column_stack((x_axis, y_axis, z_axis))

    def _make_anchor(
        name: str,
        position: list[float],
        *,
        aliases: list[str],
        asset_label: str,
        task_role: str,
        goal_roles: list[str],
        quaternion: list[float] | None = None,
        affordances: dict[str, Any] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        translated[name] = {
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "gripper_offset": 0.0,
            "estimated_half_height": 0.0,
            "estimated_half_extents": [0.0, 0.0, 0.0],
            "shape_type": "target_anchor",
            "quaternion": quaternion or [1.0, 0.0, 0.0, 0.0],
            "color_name": None,
            "asset_label": asset_label,
            "aliases": aliases,
            "task_role": task_role,
            "goal_roles": goal_roles,
            "affordances": affordances or {},
        }
        if extra_fields:
            translated[name].update(extra_fields)

    support_subject_to_alignment_target: dict[str, str] = {}
    for cond in conditions:
        if str(cond.get("relation", cond.get("type", "")) or "").lower() == "at_position":
            subject = cond.get("subject", cond.get("object"))
            target = cond.get("target")
            if isinstance(subject, str) and isinstance(target, str):
                support_subject_to_alignment_target[subject] = target

    needs_articulated_handle_target = capability.family == "articulated_pull" or any(
        str(cond.get("relation", cond.get("type", "")) or "").lower() in {"position_above", "position_below"}
        for cond in conditions
        if isinstance(cond, dict)
    )

    if needs_articulated_handle_target:
        cabinet_asset = next(
            (
                asset for asset in assets_by_name.values()
                if asset.get("type") == "articulation" and isinstance(asset.get("handle"), dict)
            ),
            None,
        )
        if cabinet_asset:
            handle_cfg = cabinet_asset.get("handle", {})
            handle_name = handle_cfg.get("name")
            cabinet_pos = cabinet_asset.get("position")
            if (
                isinstance(handle_name, str)
                and isinstance(cabinet_pos, (list, tuple))
                and len(cabinet_pos) == 3
            ):
                cabinet_quat = cabinet_asset.get("rotation", [1.0, 0.0, 0.0, 0.0])
                world_offset = handle_cfg.get("interaction_offset_world_position")
                if isinstance(world_offset, (list, tuple)) and len(world_offset) == 3:
                    handle_pos = np.asarray(cabinet_pos, dtype=np.float64) + np.asarray(world_offset, dtype=np.float64)
                else:
                    offset = handle_cfg.get(
                        "interaction_offset_position",
                        handle_cfg.get("offset_position", [0.0, 0.0, 0.0]),
                    )
                    handle_pos = np.asarray(cabinet_pos, dtype=np.float64) + _rotate_vector(cabinet_quat, offset)
                cabinet_rot = _quat_to_rotation_matrix(cabinet_quat)
                pull_axis = cabinet_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
                handle_rot_local = _quat_to_rotation_matrix(handle_cfg.get("offset_rotation", [1.0, 0.0, 0.0, 0.0]))
                handle_rot_world = cabinet_rot @ handle_rot_local
                handle_joint_name = handle_cfg.get("joint_name")
                if not isinstance(handle_joint_name, str) or not handle_joint_name:
                    for goal_cond in conditions:
                        relation = str(goal_cond.get("relation", goal_cond.get("type", "")) or "").lower()
                        subject_name = goal_cond.get("subject", goal_cond.get("object"))
                        if relation in {"position_above", "position_below"} and isinstance(subject_name, str):
                            handle_joint_name = subject_name
                            break
                if not isinstance(handle_joint_name, str) or not handle_joint_name:
                    handle_joint_name = str((cabinet_asset.get("container", {}) or {}).get("joint_name", "") or "")
                max_open = 0.0
                for goal_cond in conditions:
                    goal_subject = goal_cond.get("subject", goal_cond.get("object"))
                    relation = str(goal_cond.get("relation", goal_cond.get("type", "")) or "").lower()
                    if goal_subject == handle_joint_name and relation == "position_above":
                        max_open = max(max_open, float(goal_cond.get("value", 0.0) or 0.0))
                if max_open <= 0.0:
                    max_open = float(
                        handle_cfg.get(
                            "target_joint_position",
                            (cabinet_asset.get("container", {}) or {}).get("open_target_joint_position", 0.0),
                        )
                        or 0.0
                    )
                pull_distance = float(handle_cfg.get("pull_distance", max(max_open, 0.25)) or max(max_open, 0.25))
                _make_anchor(
                    handle_name,
                    handle_pos.tolist(),
                    aliases=[handle_name.replace("_", " "), "drawer handle", "top drawer handle"],
                    asset_label="drawer handle",
                    task_role="handle_target",
                    goal_roles=["target:position_above", "target:position_below"],
                    quaternion=_rotation_matrix_to_quat(handle_rot_world),
                    extra_fields={
                        "articulation_name": str(cabinet_asset.get("name", "")),
                        "joint_name": handle_joint_name or None,
                        "target_rotation_world": handle_rot_world.tolist(),
                        "closed_position": handle_pos.tolist(),
                        "slide_axis_world": (pull_axis / max(np.linalg.norm(pull_axis), 1e-9)).tolist(),
                        "open_target_joint_position": max(max_open, 0.0),
                        "close_target_joint_position": 0.0,
                        "pull_axis_world": (pull_axis / max(np.linalg.norm(pull_axis), 1e-9)).tolist(),
                        "pull_distance": pull_distance,
                        "target_joint_position": max(max_open, 0.0),
                    },
                )

    needs_articulated_container_target = capability.family == "articulated_container_transfer" or any(
        str(cond.get("relation", cond.get("type", "")) or "").lower() == "inside_drawer"
        for cond in conditions
        if isinstance(cond, dict)
    )

    if needs_articulated_container_target:
        container_asset = next(
            (
                asset for asset in assets_by_name.values()
                if asset.get("type") == "articulation" and isinstance(asset.get("container"), dict)
            ),
            None,
        )
        if container_asset:
            container_cfg = container_asset.get("container", {})
            container_name = container_cfg.get("name")
            asset_position = container_asset.get("position")
            if (
                isinstance(container_name, str)
                and isinstance(asset_position, (list, tuple))
                and len(asset_position) == 3
            ):
                asset_quat = container_asset.get("rotation", [1.0, 0.0, 0.0, 0.0])
                closed_floor_offset = np.asarray(container_cfg.get("closed_floor_offset_position", [0.0, 0.0, 0.0]), dtype=np.float64)
                closed_center_offset = np.asarray(
                    container_cfg.get("closed_center_offset_position", closed_floor_offset),
                    dtype=np.float64,
                )
                slide_axis_local = np.asarray(container_cfg.get("slide_axis_local", [1.0, 0.0, 0.0]), dtype=np.float64)
                slide_axis_world = _rotate_vector(asset_quat, slide_axis_local)
                slide_axis_world = slide_axis_world / max(np.linalg.norm(slide_axis_world), 1e-9)
                open_target = float(container_cfg.get("open_target_joint_position", 0.0) or 0.0)
                close_target = float(container_cfg.get("close_target_joint_position", 0.0) or 0.0)
                asset_pos_vec = np.asarray(asset_position, dtype=np.float64)
                closed_floor_world = asset_pos_vec + _rotate_vector(asset_quat, closed_floor_offset)
                closed_center_world = asset_pos_vec + _rotate_vector(asset_quat, closed_center_offset)
                open_delta = open_target - close_target
                open_floor_world = closed_floor_world + slide_axis_world * open_delta
                open_center_world = closed_center_world + slide_axis_world * open_delta
                half_extents = [float(v) for v in container_cfg.get("half_extents", [0.0, 0.0, 0.0])]
                half_extents_vec = np.asarray(half_extents, dtype=np.float64)
                insertion_axis_world = -slide_axis_world
                insertion_axis_world = insertion_axis_world / max(np.linalg.norm(insertion_axis_world), 1e-9)
                target_rotation_world = _rotation_with_tool_z(insertion_axis_world)
                insertion_extent = max(float(np.dot(np.abs(insertion_axis_world), half_extents_vec)), 0.0)
                entry_clearance = max(insertion_extent + 0.03, 0.06)
                entry_position = open_center_world - insertion_axis_world * entry_clearance
                _make_anchor(
                    container_name,
                    open_center_world.tolist(),
                    aliases=[container_name.replace("_", " "), "drawer interior", "drawer container"],
                    asset_label="drawer container",
                    task_role="placement_target",
                    goal_roles=["target:inside_drawer"],
                    extra_fields={
                        "support_asset_name": str(container_asset.get("name", "")),
                        "support_top_z": float(open_floor_world[2]),
                        "articulation_name": str(container_asset.get("name", "")),
                        "joint_name": str(container_cfg.get("joint_name", "")) if container_cfg.get("joint_name") is not None else None,
                        "open_target_joint_position": open_target,
                        "close_target_joint_position": close_target,
                        "closed_floor_position": closed_floor_world.tolist(),
                        "closed_center_position": closed_center_world.tolist(),
                        "slide_axis_world": slide_axis_world.tolist(),
                        "entry_position": entry_position.tolist(),
                        "target_position": open_center_world.tolist(),
                        "insertion_axis_world": insertion_axis_world.tolist(),
                        "target_rotation_world": target_rotation_world.tolist(),
                        "entry_clearance": entry_clearance,
                        "container_half_extents": half_extents,
                    },
                )

    for cond in conditions:
        relation = str(cond.get("relation", cond.get("type", "")) or "").lower()
        subject = cond.get("subject", cond.get("object"))
        target = cond.get("target")

        if relation == "above" and isinstance(target, str):
            anchor_name = f"{target}_top_surface"
            if anchor_name in translated:
                continue
            align_target_name = (
                support_subject_to_alignment_target.get(subject)
                if isinstance(subject, str)
                else None
            )
            anchor_pos: list[float] | None = None
            if isinstance(align_target_name, str) and align_target_name in translated:
                anchor_pos = list(translated[align_target_name]["position"])
            else:
                target_entry = translated.get(target)
                if isinstance(target_entry, dict):
                    anchor_pos = list(target_entry["position"])
                else:
                    asset = assets_by_name.get(target, {})
                    asset_position = asset.get("position")
                    if isinstance(asset_position, (list, tuple)) and len(asset_position) == 3:
                        top_z = float(asset_position[2])
                        if isinstance(asset.get("scale"), (list, tuple)) and len(asset["scale"]) == 3:
                            top_z += float(asset["scale"][2]) / 2.0
                        anchor_pos = [float(asset_position[0]), float(asset_position[1]), top_z]
            if anchor_pos is None:
                continue
            _make_anchor(
                anchor_name,
                anchor_pos,
                aliases=[f"{target.replace('_', ' ')} top surface", f"{target.replace('_', ' ')} support"],
                asset_label=f"{target.replace('_', ' ')} top surface",
                task_role="placement_target",
                goal_roles=[f"target:{relation}"],
                extra_fields={
                    "support_asset_name": target,
                    "support_alignment_target": align_target_name,
                    "support_top_z": float(anchor_pos[2]),
                },
            )

        if relation in {"inserted_into", "height_below"} and isinstance(target, str) and target in translated:
            entry = translated[target]
            target_pos = np.asarray(entry["position"], dtype=np.float64)
            extra_fields: dict[str, Any] = {}
            if target == "hole":
                hole_asset = assets_by_name.get(target, {})
                hole_asset_pos = hole_asset.get("position")
                if isinstance(hole_asset_pos, (list, tuple)) and len(hole_asset_pos) == 3:
                    target_pos = np.asarray(hole_asset_pos, dtype=np.float64)
                    entry["position"] = target_pos.tolist()
                extra_fields = {
                    "entry_position": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2] + 0.05)],
                    "target_position": [float(target_pos[0]), float(target_pos[1]), float(target_pos[2] - 0.02)],
                    "insertion_axis_world": [0.0, 0.0, -1.0],
                    "target_rotation_world": [
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                        [-1.0, 0.0, 0.0],
                    ],
                    "insertion_depth": 0.05,
                    "required_depth": 0.02,
                }
            elif target == "box_wall_back":
                box_back_asset = assets_by_name.get(target, {})
                box_back_pos = box_back_asset.get("position")
                if isinstance(box_back_pos, (list, tuple)) and len(box_back_pos) == 3:
                    target_pos = np.asarray(box_back_pos, dtype=np.float64)
                    entry["position"] = [
                        float(target_pos[0]),
                        float(target_pos[1]),
                        float(target_pos[2] + float(entry.get("estimated_half_height", 0.0) or 0.0)),
                    ]
                box_center_x = None
                if "box_wall_top" in assets_by_name:
                    asset = assets_by_name["box_wall_top"]
                    asset_pos = asset.get("position")
                    scale = asset.get("scale")
                    if (
                        isinstance(asset_pos, (list, tuple))
                        and len(asset_pos) == 3
                        and isinstance(scale, (list, tuple))
                        and len(scale) == 3
                    ):
                        box_center_x = float(asset_pos[0])
                        front_x = box_center_x - float(scale[0]) / 2.0
                    else:
                        front_x = float(target_pos[0] - 0.13)
                else:
                    front_x = float(target_pos[0] - 0.13)
                extra_fields = {
                    "entry_position": [front_x, float(target_pos[1]), float(target_pos[2])],
                    "target_position": target_pos.tolist(),
                    "insertion_axis_world": [1.0, 0.0, 0.0],
                    "target_rotation_world": [
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                    ],
                    "insertion_depth": float(target_pos[0] - front_x),
                    "required_depth": float(target_pos[0] - front_x) * 0.7,
                }
            elif target == "receptacle_back":
                receptacle_asset = assets_by_name.get(target, {})
                receptacle_pos = receptacle_asset.get("position")
                if isinstance(receptacle_pos, (list, tuple)) and len(receptacle_pos) == 3:
                    target_pos = np.asarray(receptacle_pos, dtype=np.float64)
                    entry["position"] = [
                        float(target_pos[0]),
                        float(target_pos[1]),
                        float(target_pos[2] + float(entry.get("estimated_half_height", 0.0) or 0.0)),
                    ]
                front_x = float(target_pos[0] - 0.010)
                if "receptacle_frame_left" in assets_by_name:
                    frame_pos = assets_by_name["receptacle_frame_left"].get("position")
                    if isinstance(frame_pos, (list, tuple)) and len(frame_pos) == 3:
                        front_x = float(frame_pos[0])
                extra_fields = {
                    "entry_position": [front_x, float(target_pos[1]), float(target_pos[2])],
                    "target_position": target_pos.tolist(),
                    "insertion_axis_world": [1.0, 0.0, 0.0],
                    "target_rotation_world": [
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, -1.0],
                    ],
                    "insertion_depth": float(target_pos[0] - front_x),
                    "required_depth": max(float(target_pos[0] - front_x) * 0.8, 0.005),
                }
            if extra_fields:
                entry.update(extra_fields)
            if (
                isinstance(subject, str)
                and subject == "peg_head"
                and "peg_tail" in translated
                and "peg" not in translated
            ):
                head_pos = np.asarray(translated["peg_head"]["position"], dtype=np.float64)
                tail_pos = np.asarray(translated["peg_tail"]["position"], dtype=np.float64)
                peg_mid = 0.5 * (head_pos + tail_pos)
                peg_alias = dict(translated["peg_tail"])
                peg_alias["position"] = peg_mid.tolist()
                peg_alias["pick_position"] = tail_pos.tolist()
                peg_alias["asset_label"] = "peg"
                peg_alias["aliases"] = list(dict.fromkeys((peg_alias.get("aliases") or []) + ["peg"]))
                peg_alias["grasp_object_name"] = "peg_tail"
                peg_alias["insertion_subject_name"] = "peg_head"
                peg_alias["insertion_subject_offset_local"] = (head_pos - tail_pos).tolist()
                peg_alias["goal_roles"] = [f"subject:{relation}"]
                translated["peg"] = peg_alias

        if relation == "in_slot":
            dynamic_shapes = [
                name
                for name, asset in assets_by_name.items()
                if name.startswith("shape_") and asset.get("type") == "rigid" and name in translated
            ]
            if isinstance(subject, str) and subject not in translated and len(dynamic_shapes) == 1:
                source_name = dynamic_shapes[0]
                source_asset = assets_by_name.get(source_name, {})
                slot_yaw_deg = math.degrees(float(source_asset.get("slot_rotation_z", 0.0)))

                def _normalize_yaw_deg(deg: float) -> float:
                    return ((float(deg) + 180.0) % 360.0) - 180.0

                source_info = translated[source_name]
                shape_position = np.asarray(source_info.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
                shape_quaternion = source_info.get("quaternion", [1.0, 0.0, 0.0, 0.0])
                shape_rotation = _quat_to_rotation_matrix(shape_quaternion)
                half_extents = np.asarray(
                    source_info.get("estimated_half_extents", [0.03, 0.03, 0.01]),
                    dtype=np.float64,
                )
                grasp_radius_x = max(min(float(half_extents[0]) * 0.75, 0.024), 0.012)
                grasp_radius_y = max(min(float(half_extents[1]) * 0.75, 0.024), 0.012)
                grasp_drop = min(max(float(half_extents[2]) * 0.40, 0.0035), 0.008)
                local_candidates = _asset_grasp_points_local(source_asset)
                if not local_candidates:
                    local_candidates = [
                        np.array([grasp_radius_x, 0.0, 0.0], dtype=np.float64),
                        np.array([-grasp_radius_x, 0.0, 0.0], dtype=np.float64),
                        np.array([0.0, grasp_radius_y, 0.0], dtype=np.float64),
                        np.array([0.0, -grasp_radius_y, 0.0], dtype=np.float64),
                    ]
                pick_position_candidates = [
                    (
                        shape_position
                        + shape_rotation @ offset
                        + np.array([0.0, 0.0, -grasp_drop], dtype=np.float64)
                    ).tolist()
                    for offset in local_candidates
                ]
                pick_yaw_candidates_deg = []
                for candidate_deg in (
                    slot_yaw_deg,
                    slot_yaw_deg + 180.0,
                    slot_yaw_deg + 90.0,
                    slot_yaw_deg - 90.0,
                ):
                    normalized = _normalize_yaw_deg(candidate_deg)
                    if all(abs(normalized - existing) > 1e-6 for existing in pick_yaw_candidates_deg):
                        pick_yaw_candidates_deg.append(normalized)

                slot_fit_metadata = {
                    "preferred_grasp_region": "outer_rim",
                    "pick_required_tool_axis_world": [0.0, 0.0, -1.0],
                    "pick_position_candidates": pick_position_candidates,
                    "pick_position": list(pick_position_candidates[0]) if pick_position_candidates else source_info.get("position"),
                    "pick_yaw_candidates_deg": pick_yaw_candidates_deg,
                    "grasp_surface_clearance_override": -0.010,
                    "gripper_close_duration_override": 1.0,
                    "pick_breakout_lift_override": 0.05,
                    "place_approach_offset_override": 0.03,
                    "place_drop_offset_override": 0.0,
                    "slot_fit_debug": True,
                    "grasp_points_local": [offset.tolist() for offset in local_candidates],
                }
                translated[source_name].update(slot_fit_metadata)
                source_entry = dict(translated[source_name])
                source_entry["aliases"] = list(dict.fromkeys((source_entry.get("aliases") or []) + ["shape to place"]))
                source_entry["asset_label"] = "shape to place"
                source_entry["grasp_object_name"] = source_name
                source_entry["source_object_name"] = source_name
                source_entry["goal_roles"] = [f"subject:{relation}"]
                source_entry["preferred_grasp_region"] = "outer_rim"
                translated[subject] = source_entry

                tray_asset = assets_by_name.get("kit_tray", {})
                tray_pos = np.asarray(tray_asset.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
                tray_rot = _quat_to_rotation_matrix(tray_asset.get("rotation", [1.0, 0.0, 0.0, 0.0]))
                local_slot = source_asset.get("slot_position_local", [0.0, 0.0, 0.0])
                slot_world = tray_pos + tray_rot @ np.asarray(local_slot, dtype=np.float64)
                _make_anchor(
                    str(target),
                    slot_world.tolist(),
                    aliases=["matching cutout", "matching slot"],
                    asset_label="matching cutout",
                    task_role="insertion_target",
                    goal_roles=[f"target:{relation}"],
                    extra_fields={
                        "slot_position": slot_world.tolist(),
                        "slot_yaw_deg": slot_yaw_deg,
                        "support_top_z": float(tray_pos[2]),
                        "slot_yaw_tolerance_deg": 25.0,
                        "slot_z_tolerance": 0.012,
                    },
                )

        if relation == "upright" and isinstance(subject, str) and subject in translated:
            translated[subject]["upright_axis_local"] = [1.0, 0.0, 0.0]
            translated[subject]["upright_target_rotation_world"] = (
                np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
            ).tolist()
            half_extents = np.asarray(
                translated[subject].get("estimated_half_extents", [0.02, 0.02, 0.02]),
                dtype=np.float64,
            )
            axis_local = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            vertical_extent = max(float(np.dot(np.abs(axis_local), half_extents)), 0.02)
            perpendicular_extents = half_extents[np.abs(axis_local) < 0.5]
            perpendicular_extent = (
                float(np.max(perpendicular_extents))
                if perpendicular_extents.size
                else max(float(np.min(half_extents)), 0.02)
            )
            grasp_offset_mag = min(
                max(vertical_extent * 0.45, 0.035),
                max(vertical_extent - max(perpendicular_extent * 2.5, 0.05), 0.035),
            )
            upright_grasp_offset_local = axis_local * grasp_offset_mag
            translated[subject]["upright_grasp_offset_local"] = upright_grasp_offset_local.tolist()
            translated[subject]["upright_grasp_offset_candidates_local"] = [
                (axis_local * grasp_offset_mag).tolist(),
                (-axis_local * grasp_offset_mag).tolist(),
                (axis_local * max(grasp_offset_mag * 0.65, 0.03)).tolist(),
                (-axis_local * max(grasp_offset_mag * 0.65, 0.03)).tolist(),
            ]
            translated[subject]["upright_release_depth"] = min(
                max(perpendicular_extent * 0.25, 0.0),
                0.008,
            )
            translated[subject]["upright_settle_steps"] = 10
            quat = translated[subject].get("quaternion", [1.0, 0.0, 0.0, 0.0])
            obj_rot = _quat_to_rotation_matrix(quat)
            pick_position = np.asarray(translated[subject]["position"], dtype=np.float64) + (
                obj_rot @ upright_grasp_offset_local
            )
            translated[subject]["pick_position"] = pick_position.tolist()
            translated[subject]["approach_offset_override"] = 0.08
            translated[subject]["gripper_close_duration_override"] = 0.8
            translated[subject]["pick_breakout_lift_override"] = 0.06
            translated[subject]["place_approach_offset_override"] = 0.08
            translated[subject]["place_drop_offset_override"] = 0.0

    constraints = task_doc.get("constraints", [])
    if isinstance(constraints, list):
        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            if str(constraint.get("type", "")).lower() != "fixed_joint":
                continue
            def _resolve_translated_name(raw_name: Any) -> str | None:
                basename = PurePosixPath(str(raw_name or "")).name
                candidates = [basename, basename.lower()]
                normalized = basename.lower().replace("_", "")
                for candidate_name in translated:
                    if candidate_name in candidates:
                        return candidate_name
                    if candidate_name.lower() == basename.lower():
                        return candidate_name
                    if candidate_name.lower().replace("_", "") == normalized:
                        return candidate_name
                return None

            parent_name = _resolve_translated_name(constraint.get("parent"))
            child_name = _resolve_translated_name(constraint.get("child"))
            if parent_name not in translated or child_name not in translated:
                continue
            parent_entry = translated[parent_name]
            child_entry = translated[child_name]
            parent_pos = np.asarray(parent_entry["position"], dtype=np.float64)
            child_pos = np.asarray(child_entry["position"], dtype=np.float64)
            midpoint = 0.5 * (parent_pos + child_pos)
            parent_entry["asset_label"] = "two-tone peg"
            parent_entry["aliases"] = list(
                dict.fromkeys((parent_entry.get("aliases") or []) + ["peg", "two-tone peg"])
            )
            parent_entry["composite_pick_position"] = midpoint.tolist()
            parent_entry["pick_position"] = midpoint.tolist()
            parent_entry["composite_object_names"] = [parent_name, child_name]

    for info in translated.values():
        if not isinstance(info, dict):
            continue
        if str(info.get("shape_type", "")).lower() != "gear":
            continue
        position = np.asarray(info.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
        extents = np.asarray(info.get("estimated_half_extents", [0.03, 0.03, 0.008]), dtype=np.float64)
        lowered_position = position.copy()
        lowered_position[2] -= min(max(float(extents[2]) * 0.35, 0.002), 0.004)
        info["pick_position"] = [float(lowered_position[0]), float(lowered_position[1]), float(lowered_position[2])]
        info["pick_position_candidates"] = [info["pick_position"]]
        info["preferred_grasp_region"] = "center"
        info["pick_yaw_candidates_deg"] = [0.0, 90.0, -90.0]
        info["pick_required_tool_axis_world"] = [0.0, 0.0, -1.0]
        info["approach_offset_override"] = 0.065
        info["gripper_close_duration_override"] = 1.0
        info["pick_breakout_lift_override"] = 0.05
        info["place_drop_offset_override"] = 0.001

    for name, info in translated.items():
        if not isinstance(info, dict):
            continue
        asset = assets_by_name.get(name, {})
        asset_path = str(asset.get("asset_path", "")).lower()
        if "assets/assembling_kits/shape_" not in asset_path:
            continue
        slot_yaw_deg = math.degrees(float(asset.get("slot_rotation_z", 0.0)))
        yaw_candidates_deg: list[float] = []
        for candidate_deg in (
            slot_yaw_deg,
            slot_yaw_deg + 180.0,
            slot_yaw_deg + 90.0,
            slot_yaw_deg - 90.0,
        ):
            normalized = ((float(candidate_deg) + 180.0) % 360.0) - 180.0
            if all(abs(normalized - existing) > 1e-6 for existing in yaw_candidates_deg):
                yaw_candidates_deg.append(normalized)
        info["pick_yaw_candidates_deg"] = yaw_candidates_deg
        info["pick_required_tool_axis_world"] = [0.0, 0.0, -1.0]
        info["preferred_grasp_region"] = "outer_rim"
        info["approach_offset_override"] = 0.060
        quaternion = info.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        rotation = _quat_to_rotation_matrix(quaternion)
        extents = np.asarray(info.get("estimated_half_extents", [0.03, 0.03, 0.01]), dtype=np.float64)
        grasp_radius_x = max(min(float(extents[0]) * 0.75, 0.024), 0.012)
        grasp_radius_y = max(min(float(extents[1]) * 0.75, 0.024), 0.012)
        grasp_drop = min(max(float(extents[2]) * 0.35, 0.0025), 0.0045)
        local_candidates = _asset_grasp_points_local(asset)
        if not local_candidates:
            local_candidates = [
                np.array([grasp_radius_x, 0.0, 0.0], dtype=np.float64),
                np.array([-grasp_radius_x, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, grasp_radius_y, 0.0], dtype=np.float64),
                np.array([0.0, -grasp_radius_y, 0.0], dtype=np.float64),
            ]
        position = np.asarray(info.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
        info["pick_position_candidates"] = [
            (position + rotation @ offset + np.array([0.0, 0.0, -grasp_drop], dtype=np.float64)).tolist()
            for offset in local_candidates
        ]
        if info["pick_position_candidates"]:
            info["pick_position"] = list(info["pick_position_candidates"][0])
        info["grasp_points_local"] = [offset.tolist() for offset in local_candidates]
        info["gripper_close_duration_override"] = 1.0
        info["pick_breakout_lift_override"] = 0.05
        info["place_approach_offset_override"] = 0.03
        info["place_drop_offset_override"] = 0.0
        info["slot_fit_debug"] = True

    for name, info in translated.items():
        if not isinstance(info, dict):
            continue
        source_object_name = info.get("source_object_name") or info.get("grasp_object_name")
        if not isinstance(source_object_name, str) or source_object_name == name:
            continue
        source_info = translated.get(source_object_name)
        source_asset = assets_by_name.get(source_object_name, {})
        if not isinstance(source_info, dict):
            continue
        if "assets/assembling_kits/shape_" not in str(source_asset.get("asset_path", "")).lower():
            continue
        for field in (
            "pick_required_tool_axis_world",
            "pick_position_candidates",
            "pick_position",
            "pick_yaw_candidates_deg",
            "approach_offset_override",
            "grasp_offset_override",
            "gripper_close_duration_override",
            "pick_breakout_lift_override",
            "place_approach_offset_override",
            "place_drop_offset_override",
            "preferred_grasp_region",
            "slot_fit_debug",
        ):
            if field in source_info:
                value = source_info[field]
                if isinstance(value, list):
                    info[field] = list(value)
                else:
                    info[field] = value

    if "table" in _goal_target_names(task_doc) and "table_surface" not in translated:
        table_asset = assets_by_name.get("table", {})
        table_pos = table_asset.get("position")
        if isinstance(table_pos, (list, tuple)) and len(table_pos) == 3:
            _make_anchor(
                "table_surface",
                [float(table_pos[0]), float(table_pos[1]), 0.0],
                aliases=["table", "table surface", "table top"],
                asset_label="table surface",
                task_role="support_surface",
                goal_roles=["target:table"],
                extra_fields={"support_top_z": 0.0},
            )


def _footprint_radius_xy(info: dict[str, Any]) -> float:
    """Return a coarse XY footprint radius from translated object metadata."""

    extents = info.get("estimated_half_extents")
    if isinstance(extents, (list, tuple)) and len(extents) >= 2:
        return max(float(extents[0]), float(extents[1]), 0.001)
    half_height = info.get("estimated_half_height", 0.02)
    return max(float(half_height), 0.001)


def _attach_topdown_affordances(translated: dict[str, dict[str, Any]]) -> None:
    """Attach simple crowding and top-down affordance metadata for prompt grounding."""

    object_names = list(translated.keys())
    crowded_gap_threshold = 0.06
    nominal_top_clearance = 0.25

    for name in object_names:
        info = translated[name]
        position = np.asarray(info.get("position", [0.0, 0.0, 0.0]), dtype=float)
        radius_xy = _footprint_radius_xy(info)
        half_height = max(float(info.get("estimated_half_height", 0.02)), 0.001)
        nearest_gap = math.inf
        nearest_neighbor: str | None = None
        crowded_neighbors: list[str] = []
        top_clearance = math.inf

        for other_name in object_names:
            if other_name == name:
                continue
            other = translated[other_name]
            other_pos = np.asarray(other.get("position", [0.0, 0.0, 0.0]), dtype=float)
            other_radius = _footprint_radius_xy(other)
            xy_dist = float(np.linalg.norm(other_pos[:2] - position[:2]))
            boundary_gap = xy_dist - (radius_xy + other_radius)
            if boundary_gap < nearest_gap:
                nearest_gap = boundary_gap
                nearest_neighbor = other_name
            if boundary_gap < crowded_gap_threshold:
                crowded_neighbors.append(other_name)

            xy_overlap = xy_dist <= max(radius_xy, other_radius)
            if xy_overlap and other_pos[2] > position[2]:
                other_half_height = max(float(other.get("estimated_half_height", 0.02)), 0.001)
                other_bottom_z = float(other_pos[2]) - (2.0 * other_half_height)
                clearance = other_bottom_z - float(position[2])
                top_clearance = min(top_clearance, clearance)

        if not math.isfinite(nearest_gap):
            nearest_gap = nominal_top_clearance
        if not math.isfinite(top_clearance):
            top_clearance = nominal_top_clearance

        shape_type = str(info.get("shape_type", "unknown")).lower()
        stable_top_grasp = shape_type in {"cube", "box", "cylinder", "gear", "unknown"}
        blocked_side_grasp = nearest_gap < crowded_gap_threshold
        preferred_grasp = "top_down" if stable_top_grasp else "auto"

        info["affordances"] = {
            "preferred_grasp": preferred_grasp,
            "stable_top_grasp_feasible": bool(stable_top_grasp and top_clearance > 0.02),
            "blocked_side_grasp": bool(blocked_side_grasp),
            "reachable_from_above": True,
            "top_clearance": float(max(top_clearance, 0.0)),
            "nearest_neighbor_distance": float(max(nearest_gap, 0.0)),
            "nearest_neighbor": nearest_neighbor,
            "crowded_neighbors": crowded_neighbors,
        }


def _format_affordance_summary(translated_positions: dict[str, dict[str, Any]]) -> str:
    """Format object affordance metadata as compact prompt text."""

    lines: list[str] = []
    for name in sorted(translated_positions):
        affordances = translated_positions[name].get("affordances", {})
        if not affordances:
            continue
        crowded_neighbors = affordances.get("crowded_neighbors") or []
        crowded_text = ", ".join(crowded_neighbors) if crowded_neighbors else "none"
        lines.append(
            "- "
            f"{name}: preferred_grasp={affordances.get('preferred_grasp', 'auto')}, "
            f"stable_top_grasp_feasible={str(bool(affordances.get('stable_top_grasp_feasible'))).lower()}, "
            f"blocked_side_grasp={str(bool(affordances.get('blocked_side_grasp'))).lower()}, "
            f"top_clearance={float(affordances.get('top_clearance', 0.0)):.3f}m, "
            f"nearest_neighbor_distance={float(affordances.get('nearest_neighbor_distance', 0.0)):.3f}m, "
            f"crowded_neighbors={crowded_text}"
        )
    return "\n".join(lines) if lines else "- No affordance metadata available."


def _format_task_role_summary(translated_positions: dict[str, dict[str, Any]]) -> str:
    """Format task-role metadata as compact prompt text."""

    lines: list[str] = []
    for name in sorted(translated_positions):
        info = translated_positions[name]
        aliases = ", ".join(info.get("aliases") or []) or "none"
        goal_roles = ", ".join(info.get("goal_roles") or []) or "none"
        color_name = info.get("color_name") or "unknown"
        lines.append(
            "- "
            f"{name}: task_role={info.get('task_role', 'unknown')}, "
            f"asset_label={info.get('asset_label', name)}, "
            f"color_name={color_name}, "
            f"aliases={aliases}, "
            f"goal_roles={goal_roles}"
        )
    return "\n".join(lines) if lines else "- No task-role metadata available."


def _format_goal_condition_lines(task_doc: dict[str, Any]) -> str:
    """Render a compact goal-condition summary for the prompt."""

    lines: list[str] = []
    for cond in _collect_goal_conditions(task_doc):
        subject = cond.get("subject", cond.get("object", "?"))
        target = cond.get("target")
        relation = str(cond.get("relation", cond.get("type", "")) or "unknown")
        tolerance = cond.get("tolerance")
        value = cond.get("value", cond.get("height"))
        if target is not None:
            detail = f"{subject} {relation} {target}"
        elif value is not None:
            detail = f"{subject} {relation} {value}"
        else:
            detail = f"{subject} {relation}"
        if tolerance is not None:
            detail += f" (tol={tolerance})"
        lines.append(f"- {detail}")

    success_criteria = task_doc.get("goal", {}).get("success_criteria", {})
    if isinstance(success_criteria, dict):
        extras = []
        for key in ("xy_threshold", "height_diff", "height_threshold", "inside_tray", "tray_center", "tray_half_extent"):
            if key in success_criteria:
                extras.append(f"{key}={success_criteria[key]}")
        if extras:
            lines.append("- success_criteria: " + ", ".join(extras))

    return "\n".join(lines) if lines else "- No explicit goal conditions."


def _build_task_brief(
    task_doc: dict[str, Any],
    translated_positions: dict[str, dict[str, Any]],
    task_description: str,
) -> str:
    """Build a structured task brief from the YAML goal and translated scene."""

    task_info = task_doc.get("task", {})
    goal = task_doc.get("goal", {})
    ordered_objects = _infer_ordered_objects(task_description, translated_positions)
    ordered_text = ", ".join(ordered_objects) if ordered_objects else "none"

    subjects = sorted(_goal_subject_names(task_doc))
    targets = sorted(_goal_target_names(task_doc))

    return "\n".join(
        [
            f"- Task name: {task_info.get('name', 'UnknownTask')}",
            f"- Task description: {task_description}",
            f"- Goal description: {goal.get('description', '')}",
            "- Treat the YAML goal mappings and task-role metadata below as authoritative.",
            "- Do not invent alternative object-color remaps when `color_name`, `asset_label`, or aliases already identify the object.",
            f"- Goal subjects: {', '.join(subjects) if subjects else 'none'}",
            f"- Goal targets: {', '.join(targets) if targets else 'none'}",
            f"- Ordered object mentions from the task text: {ordered_text}",
        ]
    )


def _estimate_half_height(asset: dict[str, Any], position: list[float]) -> float:
    """Estimate half-height for tabletop objects from task metadata."""

    primitive = str(asset.get("primitive", "")).lower()
    scale = asset.get("scale")
    if primitive and isinstance(scale, (list, tuple)) and len(scale) == 3:
        return max(float(scale[2]) / 2.0, 0.001)

    if asset.get("type") == "static":
        return 0.0

    asset_path = str(asset.get("asset_path", "")).lower()
    if "factory_peg_8mm" in asset_path:
        return 0.004
    if "factory_hole_8mm" in asset_path:
        return 0.025
    if "assets/assembling_kits/shape_" in asset_path:
        return 0.010
    if "factory_gear_small" in asset_path:
        return 0.008
    if "factory_gear_medium" in asset_path:
        return 0.010
    if "m16_nut" in asset_path:
        return 0.008

    position_z = float(position[2])
    asset_position = asset.get("position")
    if position_z < 0.0 and isinstance(asset_position, (list, tuple)) and len(asset_position) == 3:
        position_z = float(asset_position[2])

    # For current supported tabletop tasks, rigid objects start on the table,
    # so center_z approximates half-height.
    return max(position_z, 0.001)


def _estimate_half_extents(
    asset: dict[str, Any],
    position: list[float],
    half_height: float,
) -> list[float]:
    """Estimate coarse half extents for object-relative grasp planning."""

    primitive = str(asset.get("primitive", "")).lower()
    scale = asset.get("scale")
    if primitive and isinstance(scale, (list, tuple)) and len(scale) == 3:
        return [
            max(float(scale[0]) / 2.0, 0.001),
            max(float(scale[1]) / 2.0, 0.001),
            max(float(scale[2]) / 2.0, 0.001),
        ]

    if primitive == "sphere":
        radius = max(float(half_height), 0.001)
        return [radius, radius, radius]

    shape_type = _infer_shape_type(asset)
    asset_path = str(asset.get("asset_path", "")).lower()
    hz = max(float(half_height), 0.001)
    if "factory_peg_8mm" in asset_path:
        return [0.025, 0.004, 0.004]
    if "factory_hole_8mm" in asset_path:
        return [0.025, 0.025, 0.025]
    if "assets/assembling_kits/shape_" in asset_path:
        return [0.030, 0.030, hz]
    if shape_type == "gear":
        if "factory_gear_small" in asset_path:
            return [0.030, 0.030, hz]
        if "factory_gear_medium" in asset_path:
            return [0.040, 0.040, hz]
        return [0.035, 0.035, hz]
    if shape_type in {"cube", "unknown"}:
        return [hz, hz, hz]
    if shape_type in {"cylinder", "can", "bottle", "mug"}:
        radius = max(min(hz, 0.045), 0.012)
        return [radius, radius, hz]
    if shape_type == "box":
        return [max(min(hz * 0.6, 0.05), 0.015), max(min(hz * 0.4, 0.04), 0.012), hz]
    return [hz, hz, hz]


def _infer_shape_type(asset: dict[str, Any]) -> str:
    """Infer a coarse shape label for grasp planning."""

    primitive = str(asset.get("primitive", "")).lower()
    if primitive in {"cube", "sphere", "cylinder"}:
        return primitive

    asset_path = str(asset.get("asset_path", "")).lower()
    if "can" in asset_path:
        return "cylinder"
    if "box" in asset_path or "sugar_box" in asset_path:
        return "box"
    if "mug" in asset_path:
        return "mug"
    if "bottle" in asset_path:
        return "bottle"
    if "factory_peg_8mm" in asset_path:
        return "box"
    if "gear" in asset_path:
        return "gear"
    return "unknown"


def _asset_grasp_points_local(asset: dict[str, Any]) -> list[np.ndarray]:
    """Return explicit per-asset local grasp anchors, if provided."""

    raw_points = asset.get("grasp_points_local")
    if not isinstance(raw_points, (list, tuple)):
        return []

    anchors: list[np.ndarray] = []
    for raw_point in raw_points:
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) < 2:
            continue
        anchors.append(
            np.array(
                [float(raw_point[0]), float(raw_point[1]), 0.0],
                dtype=np.float64,
            )
        )
    return anchors


class SimCaPGenerator:
    """Generate ADC-style Python code for the simulation runtime."""

    def __init__(self, llm_client, robot_cfg: RobotSimConfig, retry_max: int = 1):
        self.llm = llm_client
        self.robot_cfg = robot_cfg
        self.profile = select_cap_profile(robot_cfg)
        self.retry_max = max(int(retry_max), 1)

    def generate_code(
        self,
        task_doc: dict[str, Any],
        scene_state: dict[str, dict[str, Any]],
        task_description: str | None = None,
    ) -> CaPGenerationResult:
        """Generate forward execution code and translated scene positions."""

        capability = assess_task_capability(task_doc)
        if not capability.supported:
            raise ValueError(f"Task is outside the current CaP capability profile: {capability.reason}")

        translated_positions = translate_scene_state(task_doc, scene_state)
        if not translated_positions:
            raise ValueError("No manipulable rigid/primitive objects found for CaP code generation")

        user_prompt = self._build_user_prompt(
            task_doc,
            translated_positions,
            task_description,
            capability_family=capability.family,
        )
        last_error: ValueError | None = None
        for _attempt_idx in range(self.retry_max):
            raw_response = self.llm.generate(
                system_prompt=self._build_system_prompt(),
                user_prompt=user_prompt,
            )
            generated_code = self._extract_code(raw_response)
            try:
                self._validate_generated_code(generated_code)
                self._validate_task_specific_code(
                    generated_code,
                    translated_positions=translated_positions,
                    family=capability.family,
                )
            except ValueError as exc:
                last_error = exc
                continue
            return CaPGenerationResult(
                profile=self.profile,
                translated_positions=translated_positions,
                generated_code=generated_code,
                raw_response=raw_response,
            )

        message = "LLM did not return executable CaP code"
        if last_error is not None:
            raise ValueError(f"{message}: {last_error}") from last_error
        raise ValueError(message)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for ADC-style code generation."""

        return (
            "You are a robot Code-as-Policies generator.\n"
            "Return only executable Python code. No markdown fences, no prose.\n"
            "Generate a complete file with imports, execute_task(), and the __main__ block.\n"
            "Always use the provided robot skill API exactly as documented.\n"
            "Always include unique natural-language skill_description strings for every skill call.\n"
            "Stay within the supported tabletop manipulation API. Do not invent extra helpers.\n"
        )

    def _build_user_prompt(
        self,
        task_doc: dict[str, Any],
        translated_positions: dict[str, dict[str, Any]],
        task_description: str | None = None,
        capability_family: str | None = None,
    ) -> str:
        """Build the user prompt with task, robot, and scene context."""

        task_info = task_doc.get("task", {})
        goal = task_doc.get("goal", {})
        if task_description is None:
            task_description = task_info.get("description") or task_info.get("name", "Unknown task")

        goal_text = json.dumps(goal, indent=2, ensure_ascii=False)
        positions_text = pformat(translated_positions, sort_dicts=True, width=100)
        task_brief = _build_task_brief(task_doc, translated_positions, task_description)
        goal_condition_text = _format_goal_condition_lines(task_doc)
        role_summary = _format_task_role_summary(translated_positions)
        affordance_summary = _format_affordance_summary(translated_positions)
        family = capability_family or assess_task_capability(task_doc).family
        preflight = build_task_skill_preflight(task_doc)
        preferred_skill_usage = _format_preferred_skill_usage(preflight)
        example = self._build_code_example(family=family)
        execution_hints = self._build_execution_hints(task_description, task_doc, translated_positions, family=family)

        return f"""
### Task
{task_description}

### Goal Definition
```json
{goal_text}
```

### Structured Task Brief
{task_brief}

### Goal Condition Summary
{goal_condition_text}

### Robot
- Name: {self.robot_cfg.full_name}
- Arm DOFs: {self.robot_cfg.arm_dofs}
- Gripper Type: {self.robot_cfg.gripper_type}
- Practical Tabletop Reach: {self.robot_cfg.tabletop_reach:.3f} m
- Skill Class: {self.profile.class_name} from `{self.profile.module_name}`

### Scene Positions
The dictionary below is available at runtime as `positions`.
Each object's `position` is `[x, y, z_top]` in world frame meters.
Each object may also provide `quaternion`, `shape_type`, `estimated_half_extents`, `color_name`,
`asset_label`, `aliases`, `task_role`, `goal_roles`, and `affordances`.
`quaternion` is object local-frame orientation as `[w, x, y, z]`.
For OpenArm, treat the affordance metadata as geometry facts about crowding and top-down feasibility.

```python
positions = {positions_text}
```

### Task-Role Summary
{role_summary}

### Affordance Summary
{affordance_summary}

### Preferred Skill Usage
{preferred_skill_usage}

### Composite Semantic Object Aliases
{self._format_composite_semantic_aliases(translated_positions)}

### Execution Hints
{execution_hints}

### Inferred Capability Family
- `{family}`

### Available Skill API
{self.profile.api_doc}

### Code Structure Requirements
1. Import exactly `from {self.profile.module_name} import {self.profile.class_name}`.
2. Create the skill object inside `execute_task()`.
3. Always call `skills.connect()` before execution.
4. Start with `skills.{self.profile.ready_method}(skill_description="...")`.
5. End with `skills.{self.profile.closing_method}(skill_description="...")` unless the YAML goal requires holding an object at a commanded pose (for example a lift task with `height_above`).
6. Prefer the highest-level affordance method that matches the task family before falling back to raw pick/place calls.
7. Do not add extra approach/lift/retreat `move_to_position()` calls unless the task explicitly needs an intermediate waypoint.
8. For stacking or placing on another object, call `execute_place_object(..., is_table=False, ...)`.
9. For placing on a table target marker, tray, bin, zone, or absolute tabletop location, call `execute_place_object(..., is_table=True, ...)`.
10. Every skill call must include a unique `skill_description`.
11. Always pass the explicit `object_name` to `execute_pick_object(...)`.
12. Always pass the explicit `target_name` when placing relative to a named object or support surface. It is mandatory for `is_table=False`.
13. The YAML goal mappings are authoritative. Use `subject -> target` mappings exactly as given.
14. If an object already has `color_name`, `asset_label`, or aliases, do not remap it to a different color or role.
15. Use only Python code. No comments explaining the answer outside the code itself.

### Example Skeleton
{example}

Generate the complete executable Python file now.
""".strip()

    def _build_execution_hints(
        self,
        task_description: str,
        task_doc: dict[str, Any],
        translated_positions: dict[str, dict[str, Any]],
        family: str | None = None,
    ) -> str:
        """Build task-specific hints to keep generated code on-policy."""

        goal = task_doc.get("goal", {})
        conditions = _collect_goal_conditions(task_doc)
        family = family or assess_task_capability(task_doc).family
        preflight = build_task_skill_preflight(task_doc)
        hints: list[str] = [
            "- Finish the full task. Do not stop after the first successful pick-and-place.",
            "- Treat the YAML goal conditions as the source of truth if the free-form description is ambiguous.",
        ]
        if preflight.primary_skill:
            hints.append(
                f"- The primary affordance method for this task is `{preflight.primary_skill}`. Prefer it over lower-level pick/place calls."
            )
        if preflight.secondary_skills:
            secondary = ", ".join(f"`{name}`" for name in preflight.secondary_skills)
            hints.append(f"- Only fall back to secondary skills if the primary affordance method truly does not fit: {secondary}.")
        if family == "stack":
            hints.append(
                "- Use `execute_pick_and_stack_on_object(...)` for stack steps whenever possible."
            )
        elif family == "support_surface_transfer":
            hints.append(
                "- Use `execute_pick_and_place_on_support(...)` for elevated support-surface placement tasks."
            )
        elif family == "articulated_container_transfer":
            hints.append(
                "- First open the drawer with `execute_pull_handle_open(...)`, then place the object into `drawer_container`, then close it with `execute_push_handle_closed(...)`."
            )
        elif family == "articulated_pull":
            hints.append(
                "- Use `execute_pull_handle_open(...)` with the named handle target instead of inventing manual waypoint code."
            )
        elif family == "upright_placement":
            hints.append(
                "- Use `execute_pick_and_place_upright(...)` so the runtime handles the upright placement controller."
            )
        elif family == "axial_insertion":
            hints.append(
                "- Use `execute_pick_and_insert_into_target(...)` so the runtime handles pre-alignment and guarded insertion."
            )
            composite_aliases = _composite_insertion_aliases(translated_positions)
            if composite_aliases:
                alias_text = ", ".join(
                    f"`{alias_name}` (tip=`{subject_name}`)"
                    for alias_name, subject_name in composite_aliases
                )
                hints.append(
                    "- For composite insertion objects, use the semantic alias instead of the tip sub-part so the runtime can grasp the tail and align the tip correctly: "
                    + alias_text
                    + "."
                )
        elif family == "slot_fit":
            hints.append(
                "- Use `execute_pick_and_fit_into_slot(...)` for shape-to-cutout or slot-fitting tasks."
            )
        elif family == "container_stack":
            hints.append(
                "- Use `execute_pick_and_place_in_container(...)` to place the base object into the tray/container, then `execute_pick_and_stack_on_object(...)` for the remaining stack steps."
            )
        elif family == "container_transfer":
            hints.append(
                "- Use `execute_pick_and_place_in_container(...)` for tray/bin/container placement tasks whenever possible."
            )
        elif family == "lift_hold":
            hints.append(
                "- Use `execute_pick_and_lift_to_pose(..., target_name=\"command_pose\")` for commanded lift tasks."
            )
        else:
            hints.append(
                "- Use `execute_pick_and_place_on_target(...)` for simple marker/zone/support placement tasks whenever possible."
            )
        if conditions:
            ordered_subjects = [
                str(cond.get("subject", cond.get("object")))
                for cond in conditions
                if isinstance(cond, dict) and cond.get("subject", cond.get("object"))
            ]
            if ordered_subjects:
                hints.append(
                    "- Execute the goal conditions in the same order they are listed unless the task text explicitly requires a different order."
                )

        ordered_objects = _infer_ordered_objects(task_description, translated_positions)
        description_lower = task_description.lower()
        if "stack" in description_lower and len(ordered_objects) >= 2:
            stack_steps = [
                f"`{ordered_objects[idx]}` on `{ordered_objects[idx - 1]}`"
                for idx in range(1, len(ordered_objects))
            ]
            hints.append(
                "- Stack in this exact order: " + " then ".join(stack_steps) + "."
            )
            tray_targets = [
                name for name in translated_positions
                if name not in ordered_objects and "tray" in name.lower()
            ]
            if "inside a tray" in description_lower or "tray_anchor" in translated_positions or tray_targets:
                if "tray_anchor" in translated_positions:
                    base_target = "tray_anchor"
                else:
                    base_target = tray_targets[0] if tray_targets else "the tray base"
                hints.append(
                    f"- Move the first listed object into `{base_target}` to form the base, "
                    "then stack the remaining objects on top of it."
                )
            else:
                hints.append(
                    f"- Treat `{ordered_objects[0]}` as the base object already on the table. "
                    "Do not pick or relocate it unless the task explicitly requires a new base location."
                )
        if conditions:
            goal_mappings = []
            for cond in conditions:
                subject = cond.get("subject")
                target = cond.get("target")
                relation = cond.get("relation", cond.get("type", ""))
                if not isinstance(subject, str) or not isinstance(target, str):
                    continue
                if relation in {"at_position", "on_surface"}:
                    goal_mappings.append(f"`{subject}` -> `{target}`")
                elif relation in {"on_top_of", "stacked"}:
                    goal_mappings.append(f"`{subject}` on `{target}`")
            if goal_mappings:
                hints.append("- Respect these exact goal mappings: " + ", ".join(goal_mappings) + ".")
                hints.append(
                    "- When an object name has matching aliases or color metadata, use that object directly instead of reassigning roles."
                )

        if not conditions and len(ordered_objects) > 1:
            hints.append(
                "- Manipulate each named object in the order implied by the task description."
            )

        if any(
            str(cond.get("relation", cond.get("type", ""))).lower() == "height_above"
            for cond in conditions
        ):
            hints.append(
                "- For lift goals, keep the gripper closed at the commanded pose; do not place or release the object."
            )
            hints.append(
                "- Do not call `execute_place_object()` or reopen the gripper for a lift task unless the YAML explicitly asks for a release."
            )

        if self.robot_cfg.name == "openarm":
            hints.append(
                "- OpenArm grasp orientation is selected automatically by the runtime. Do not add orientation-specific pick or place arguments."
            )
            hints.append(
                "- For OpenArm, use the standard pick/place primitives directly and let the runtime preserve the carried grasp during placement."
            )
            hints.append(
                "- If an object's affordances say `preferred_grasp=top_down` or `blocked_side_grasp=true`, rely on the standard pick primitive and assume the runtime will execute a vertical top-down grasp."
            )

        return "\n".join(hints)

    @staticmethod
    def _format_composite_semantic_aliases(
        translated_positions: dict[str, dict[str, Any]],
    ) -> str:
        """Render composite semantic aliases for prompt grounding."""

        aliases = _composite_insertion_aliases(translated_positions)
        if not aliases:
            return "- No composite semantic aliases available."
        lines = []
        for alias_name, subject_name in aliases:
            info = translated_positions.get(alias_name, {})
            grasp_name = info.get("grasp_object_name", alias_name)
            lines.append(
                f"- `{alias_name}`: use this semantic object for insertion tasks; the runtime grasps `{grasp_name}` and treats `{subject_name}` as the insertion subject."
            )
        return "\n".join(lines)

    def _build_code_example(self, family: str | None = None) -> str:
        """Build a short API-faithful example for the current embodiment."""

        ready_desc = "move to the ready pose before starting the task"
        close_desc = "return to the ready pose after completing the task"
        family = family or "tabletop_transfer"
        if family == "stack":
            action_call = """        skills.execute_pick_and_stack_on_object(
            object_name=\"object_name\",
            bottom_object_name=\"target_name\",
            skill_description=\"stack object_name on target_name\",
        )"""
        elif family == "support_surface_transfer":
            action_call = """        skills.execute_pick_and_place_on_support(
            object_name=\"object_name\",
            support_name=\"target_name\",
            skill_description=\"pick object_name and place it on the named support surface\",
        )"""
        elif family == "articulated_pull":
            action_call = """        skills.execute_pull_handle_open(
            handle_name=\"target_name\",
            skill_description=\"grasp the named handle and pull it open\",
        )"""
        elif family == "articulated_container_transfer":
            action_call = """        skills.execute_pull_handle_open(
            handle_name="drawer_handle_top",
            skill_description="grasp the drawer handle and pull it open before storing the object",
        )
        skills.execute_pick_and_place_in_container(
            object_name="object_name",
            container_name="drawer_container",
            skill_description="pick object_name and place it into the opened drawer interior",
        )
        skills.execute_push_handle_closed(
            handle_name="drawer_handle_top",
            skill_description="grasp the drawer handle and push it closed after storing the object",
        )"""
        elif family == "upright_placement":
            action_call = """        skills.execute_pick_and_place_upright(
            object_name=\"object_name\",
            support_name=\"target_name\",
            skill_description=\"pick object_name and place it upright on target_name\",
        )"""
        elif family == "axial_insertion":
            action_call = """        skills.execute_pick_and_insert_into_target(
            object_name=\"object_name\",
            target_name=\"target_name\",
            skill_description=\"pick object_name and insert it into target_name\",
        )"""
        elif family == "slot_fit":
            action_call = """        skills.execute_pick_and_fit_into_slot(
            object_name=\"object_name\",
            target_name=\"target_name\",
            skill_description=\"pick object_name and fit it into target_name\",
        )"""
        elif family == "container_transfer":
            action_call = """        skills.execute_pick_and_place_in_container(
            object_name=\"object_name\",
            container_name=\"target_name\",
            skill_description=\"place object_name into the target container\",
        )"""
        elif family == "container_stack":
            action_call = """        skills.execute_pick_and_place_in_container(
            object_name=\"object_name\",
            container_name=\"target_name\",
            skill_description=\"place the base object into the target container\",
        )"""
        elif family == "lift_hold":
            action_call = """        skills.execute_pick_and_lift_to_pose(
            object_name=\"object_name\",
            target_name=\"target_name\",
            skill_description=\"pick object_name and hold it at target_name\",
        )"""
        else:
            action_call = """        skills.execute_pick_and_place_on_target(
            object_name=\"object_name\",
            target_name=\"target_name\",
            skill_description=\"pick object_name and place it on target_name\",
        )"""
        return f"""from {self.profile.module_name} import {self.profile.class_name}

def execute_task():
    skills = {self.profile.class_name}(frame="world")
    skills.connect()
    try:
        skills.{self.profile.ready_method}(skill_description="{ready_desc}")

        pick_obj = positions["object_name"]
        pick_pos = pick_obj["position"]

        target_obj = positions["target_name"]
        target_pos = target_obj["position"]

{action_call}
        skills.{self.profile.closing_method}(skill_description="{close_desc}")
    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
"""

    @staticmethod
    def _extract_code(response: str) -> str:
        """Extract Python code from an LLM response."""

        text = response.strip()
        fenced_match = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if fenced_match:
            return max(fenced_match, key=len).strip()

        lines = text.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("from ") or stripped.startswith("import ") or stripped.startswith("def execute_task"):
                return "\n".join(lines[idx:]).strip()
        return text

    def _validate_generated_code(self, generated_code: str) -> None:
        """Reject malformed codegen responses before execution."""

        lowered = generated_code.strip().lower()
        if not lowered:
            raise ValueError("empty model response")
        if any(marker in lowered for marker in REFUSAL_MARKERS):
            raise ValueError("model refusal")

        expected_import = f"from {self.profile.module_name} import {self.profile.class_name}"
        if expected_import not in generated_code:
            raise ValueError(f"missing expected import '{expected_import}'")
        if "def execute_task" not in generated_code:
            raise ValueError("missing execute_task() definition")
        if "skills.connect()" not in generated_code or "skills.disconnect()" not in generated_code:
            raise ValueError("missing connect()/disconnect() calls")
        if "__main__" not in generated_code:
            raise ValueError("missing __main__ block")
        if self.profile.robot_name == "openarm":
            forbidden_markers = ("grasp_face", "grasp_yaw_deg", "approach_angle_deg")
            forbidden = [marker for marker in forbidden_markers if marker in generated_code]
            if forbidden:
                raise ValueError(
                    "OpenArm CaP must use the standard pick/place API without policy-level grasp overrides: "
                    + ", ".join(forbidden)
                )
        try:
            ast.parse(generated_code)
        except SyntaxError as exc:
            raise ValueError(f"invalid Python syntax: {exc.msg}") from exc

    def _validate_task_specific_code(
        self,
        generated_code: str,
        translated_positions: dict[str, dict[str, Any]],
        family: str,
    ) -> None:
        """Reject family-specific code patterns that violate runtime contracts."""

        if family != "axial_insertion":
            return

        alias_by_subject = {
            subject_name: alias_name
            for alias_name, subject_name in _composite_insertion_aliases(translated_positions)
        }
        if not alias_by_subject:
            return

        tree = ast.parse(generated_code)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "skills"
                and func.attr == "execute_pick_and_insert_into_target"
            ):
                continue
            object_name: str | None = None
            for kw in node.keywords:
                if kw.arg != "object_name":
                    continue
                if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    object_name = kw.value.value
            if object_name is None and node.args:
                first_arg = node.args[0]
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                    object_name = first_arg.value
            if object_name in alias_by_subject.values():
                continue
            replacement_alias = alias_by_subject.get(object_name or "")
            if replacement_alias:
                raise ValueError(
                    f"axial insertion must use semantic alias '{replacement_alias}' instead of insertion subject '{object_name}'"
                )
