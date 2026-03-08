"""ADC-style forward-only CaP generator for IsaacLab data collection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pprint import pformat
import re
from typing import Any

from .config import RobotSimConfig

SUPPORTED_RELATIONS = {"on_top_of", "stacked", "at_position", "on_surface"}
UNSUPPORTED_KEYWORDS = (
    "cabinet",
    "drawer",
    "peg",
    "plug",
    "insert",
    "upright",
    "receptacle",
)


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

    goal = task_doc.get("goal", {})
    targets: set[str] = set()

    conditions: list[dict[str, Any]] = []
    success_criteria = goal.get("success_criteria", {})
    if isinstance(success_criteria, dict):
        conditions.extend(success_criteria.get("conditions", []))
    conditions.extend(goal.get("conditions", []))

    for cond in conditions:
        target = cond.get("target")
        if isinstance(target, str):
            targets.add(target)
    return targets


def _infer_ordered_objects(
    task_description: str,
    translated_positions: dict[str, dict[str, Any]],
) -> list[str]:
    """Infer manipulation order from object names mentioned in the task description."""

    description = task_description.lower()
    ordered: list[tuple[int, str]] = []
    for name in translated_positions:
        idx = description.find(name.lower())
        if idx >= 0:
            ordered.append((idx, name))
    ordered.sort()
    return [name for _, name in ordered]


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


def is_supported_tabletop_task(task_doc: dict[str, Any]) -> tuple[bool, str]:
    """Return whether the task fits the current v1 tabletop CaP profile."""

    task_info = task_doc.get("task", {})
    goal = task_doc.get("goal", {})
    haystack = " ".join(
        str(x)
        for x in (
            task_info.get("name", ""),
            task_info.get("description", ""),
            goal.get("description", ""),
        )
    ).lower()

    for keyword in UNSUPPORTED_KEYWORDS:
        if keyword in haystack:
            return False, f"unsupported keyword '{keyword}' in task description"

    # Any non-robot articulated asset typically implies cabinet/drawer/door style tasks.
    for asset in task_doc.get("assets", []):
        if asset.get("type") == "articulation" and not _is_primary_robot_asset(asset):
            return False, f"unsupported articulated asset '{asset.get('name', 'unknown')}'"

    conditions: list[dict[str, Any]] = []
    success_criteria = goal.get("success_criteria", {})
    if isinstance(success_criteria, dict):
        conditions.extend(success_criteria.get("conditions", []))
        if "stacking_order" in success_criteria:
            return True, "supported stacking order"
    conditions.extend(goal.get("conditions", []))

    if not conditions:
        return True, "no explicit unsupported conditions"

    for cond in conditions:
        relation = cond.get("type", cond.get("relation", "")).lower()
        if relation and relation not in SUPPORTED_RELATIONS:
            return False, f"unsupported goal relation '{relation}'"

    return True, "supported tabletop relations"


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
    goal_targets = _goal_target_names(task_doc)
    names_to_include = set(scene_state.keys()) | goal_targets

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

        half_height = _estimate_half_height(asset, position)
        translated[name] = {
            "position": [
                float(position[0]),
                float(position[1]),
                float(position[2] + half_height),
            ],
            "gripper_offset": 0.0,
            "estimated_half_height": float(half_height),
            "quaternion": info.get("quaternion"),
        }

    return translated


def _estimate_half_height(asset: dict[str, Any], position: list[float]) -> float:
    """Estimate half-height for tabletop objects from task metadata."""

    if asset.get("type") == "static":
        return 0.0

    scale = asset.get("scale")
    if asset.get("type") == "primitive" and isinstance(scale, (list, tuple)) and len(scale) == 3:
        return max(float(scale[2]) / 2.0, 0.001)

    # For current supported tabletop tasks, rigid objects start on the table,
    # so center_z approximates half-height.
    return max(float(position[2]), 0.001)


class SimCaPGenerator:
    """Generate ADC-style Python code for the simulation runtime."""

    def __init__(self, llm_client, robot_cfg: RobotSimConfig):
        self.llm = llm_client
        self.robot_cfg = robot_cfg
        self.profile = select_cap_profile(robot_cfg)

    def generate_code(
        self,
        task_doc: dict[str, Any],
        scene_state: dict[str, dict[str, Any]],
        task_description: str | None = None,
    ) -> CaPGenerationResult:
        """Generate forward execution code and translated scene positions."""

        supported, reason = is_supported_tabletop_task(task_doc)
        if not supported:
            raise ValueError(f"Task is outside the current CaP tabletop profile: {reason}")

        translated_positions = translate_scene_state(task_doc, scene_state)
        if not translated_positions:
            raise ValueError("No manipulable rigid/primitive objects found for CaP code generation")

        user_prompt = self._build_user_prompt(task_doc, translated_positions, task_description)
        raw_response = self.llm.generate(
            system_prompt=self._build_system_prompt(),
            user_prompt=user_prompt,
        )
        generated_code = self._extract_code(raw_response)
        return CaPGenerationResult(
            profile=self.profile,
            translated_positions=translated_positions,
            generated_code=generated_code,
            raw_response=raw_response,
        )

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
    ) -> str:
        """Build the user prompt with task, robot, and scene context."""

        task_info = task_doc.get("task", {})
        goal = task_doc.get("goal", {})
        if task_description is None:
            task_description = task_info.get("description") or task_info.get("name", "Unknown task")

        goal_text = json.dumps(goal, indent=2, ensure_ascii=False)
        positions_text = pformat(translated_positions, sort_dicts=True, width=100)
        example = self._build_code_example()
        execution_hints = self._build_execution_hints(task_description, task_doc, translated_positions)

        return f"""
### Task
{task_description}

### Goal Definition
```json
{goal_text}
```

### Robot
- Name: {self.robot_cfg.full_name}
- Arm DOFs: {self.robot_cfg.arm_dofs}
- Gripper Type: {self.robot_cfg.gripper_type}
- Practical Tabletop Reach: {self.robot_cfg.tabletop_reach:.3f} m
- Skill Class: {self.profile.class_name} from `{self.profile.module_name}`

### Scene Positions
The dictionary below is available at runtime as `positions`.
Each object's `position` is `[x, y, z_top]` in world frame meters.
For grasped objects, `z_top / 2` approximates the object's half-height.

```python
positions = {positions_text}
```

### Execution Hints
{execution_hints}

### Available Skill API
{self.profile.api_doc}

### Code Structure Requirements
1. Import exactly `from {self.profile.module_name} import {self.profile.class_name}`.
2. Create the skill object inside `execute_task()`.
3. Always call `skills.connect()` before execution.
4. Start with `skills.{self.profile.ready_method}(skill_description="...")`.
5. End with `skills.{self.profile.closing_method}(skill_description="...")`.
6. Prefer `execute_pick_object()` and `execute_place_object()` as the primary manipulation primitives.
7. Do not add extra approach/lift/retreat `move_to_position()` calls unless the task explicitly needs an intermediate waypoint.
8. For stacking or placing on another object, call `execute_place_object(..., is_table=False, ...)`.
9. For placing on a table target marker, tray, bin, zone, or absolute tabletop location, call `execute_place_object(..., is_table=True, ...)`.
10. Every skill call must include a unique `skill_description`.
11. Always pass the explicit `object_name` to `execute_pick_object(...)`.
12. Always pass the explicit `target_name` when placing relative to a named object or support surface. It is mandatory for `is_table=False`.
13. Use only Python code. No comments explaining the answer outside the code itself.

### Example Skeleton
{example}

Generate the complete executable Python file now.
""".strip()

    def _build_execution_hints(
        self,
        task_description: str,
        task_doc: dict[str, Any],
        translated_positions: dict[str, dict[str, Any]],
    ) -> str:
        """Build task-specific hints to keep generated code on-policy."""

        goal = task_doc.get("goal", {})
        hints: list[str] = [
            "- Finish the full task. Do not stop after the first successful pick-and-place."
        ]

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
            if "inside a tray" in description_lower or tray_targets:
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

        conditions = goal.get("conditions", [])
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

        if not conditions and len(ordered_objects) > 1:
            hints.append(
                "- Manipulate each named object in the order implied by the task description."
            )

        return "\n".join(hints)

    def _build_code_example(self) -> str:
        """Build a short API-faithful example for the current embodiment."""

        ready_desc = "move to the ready pose before starting the task"
        close_desc = "return to the ready pose after completing the task"
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

        skills.execute_pick_object(
            pick_pos,
            object_name="object_name",
            skill_description="pick object_name from the tabletop",
        )
        skills.execute_place_object(
            target_pos,
            target_name="target_name",
            is_table=False,
            skill_description="place object_name onto target_name",
        )
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
