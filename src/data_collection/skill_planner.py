"""
LLM-based skill sequence planner for simulation data collection.

Given a task description and scene state (from SimDetector),
uses the LLM to generate a sequence of skill calls.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import RobotSimConfig

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).parent / "prompts"


# Skill API documentation (included in LLM prompt)
SKILL_API_DOC = """
## Available Skills

### Primitive Skills
- `move_to_ready()` — Move robot to ready (home) pose
- `move_to_position(target_xyz=[x, y, z])` — Move end-effector to [x,y,z] in world frame (meters)
- `gripper_open()` — Open the gripper
- `gripper_close()` — Close the gripper

### Composite Skills
- `execute_pick(object_name="cube_1", approach_offset=0.05)` — Pick an object by name
  Sequence: open gripper → approach above → descend → close gripper → lift
- `execute_place(target_position=[x, y, z], approach_offset=0.05)` — Place held object at position
  Sequence: approach above → descend → open gripper → retreat
- `execute_pick_and_place(pick_object="cube_1", place_position=[x, y, z])` — Full pick-and-place

### Important Notes
- All positions are in meters, world frame
- approach_offset is height above target for safe approach (default 0.05m)
- Object names must match exactly as listed in scene state
- Always start with move_to_ready() and end with move_to_ready()
"""


class SkillPlanner:
    """
    Uses LLM to generate skill execution sequences from task descriptions.

    Args:
        llm_client: AzureOpenAIClient instance (from src/common/llm_client.py)
        robot_cfg: RobotSimConfig for robot-specific context
    """

    def __init__(self, llm_client, robot_cfg: RobotSimConfig):
        self.llm = llm_client
        self.robot_cfg = robot_cfg
        self._system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load skill generation system prompt."""
        prompt_path = PROMPT_DIR / "skill_generation.md"
        if prompt_path.exists():
            return prompt_path.read_text()

        # Default system prompt if file doesn't exist
        return (
            "You are a robot skill sequence planner. Given a task description "
            "and scene state, generate a JSON list of skill calls to accomplish the task.\n\n"
            "Output ONLY valid JSON. No explanation, no markdown code blocks.\n\n"
            f"{SKILL_API_DOC}"
        )

    def plan_skills(
        self,
        task_doc: dict,
        scene_state: dict[str, dict],
        task_description: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Generate a skill sequence for the task.

        Args:
            task_doc: Parsed YAML task document
            scene_state: Object positions from SimDetector.get_all_objects()
            task_description: Override task description (default from YAML)

        Returns:
            List of skill dicts:
            [
                {"skill": "move_to_ready"},
                {"skill": "execute_pick", "params": {"object_name": "cube_1"}},
                {"skill": "execute_place", "params": {"target_position": [0.4, 0.0, 0.1]}},
                {"skill": "move_to_ready"},
            ]
        """
        # Build user prompt
        user_prompt = self._build_user_prompt(task_doc, scene_state, task_description)

        # Call LLM
        response = self.llm.generate(
            system_prompt=self._system_prompt,
            user_prompt=user_prompt,
        )

        # Parse response
        return self._parse_skill_sequence(response)

    def _build_user_prompt(
        self,
        task_doc: dict,
        scene_state: dict[str, dict],
        task_description: Optional[str] = None,
    ) -> str:
        """Build the user prompt with task and scene context."""
        # Task description
        if task_description is None:
            task_info = task_doc.get("task", {})
            task_description = task_info.get("description", "")
            if not task_description:
                task_description = task_info.get("name", "Unknown task")

        # Goal conditions
        goal = task_doc.get("goal", {})
        goal_desc = ""
        if "success_criteria" in goal:
            goal_desc = f"\nSuccess Criteria: {json.dumps(goal['success_criteria'], indent=2)}"
        elif "conditions" in goal:
            goal_desc = f"\nGoal Conditions: {json.dumps(goal['conditions'], indent=2)}"

        # Scene state
        scene_desc = "Current Scene State:\n"
        for name, info in scene_state.items():
            pos = info.get("position", [0, 0, 0])
            pos_str = f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
            scene_desc += f"  - {name}: position {pos_str}\n"

        # Robot info
        robot_desc = (
            f"\nRobot: {self.robot_cfg.full_name} "
            f"({self.robot_cfg.arm_dofs}-DOF arm, "
            f"gripper type: {self.robot_cfg.gripper_type})\n"
            f"Workspace reach: {self.robot_cfg.tabletop_reach}m\n"
        )

        prompt = (
            f"## Task\n{task_description}\n"
            f"{goal_desc}\n"
            f"\n## Robot\n{robot_desc}\n"
            f"\n## Scene\n{scene_desc}\n"
            f"\n## Instructions\n"
            f"Generate a JSON list of skill calls to accomplish this task.\n"
            f"Use the available skills API. Output ONLY valid JSON array.\n"
            f"Always start and end with move_to_ready.\n"
            f"Example format:\n"
            f'[{{"skill": "move_to_ready"}}, '
            f'{{"skill": "execute_pick", "params": {{"object_name": "cube_1"}}}}, '
            f'{{"skill": "move_to_ready"}}]'
        )

        return prompt

    def _parse_skill_sequence(self, response: str) -> list[dict[str, Any]]:
        """Parse LLM response into skill sequence."""
        # Strip markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last ``` lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        # Try parsing as JSON
        try:
            skills = json.loads(text)
            if isinstance(skills, list):
                return self._validate_skills(skills)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from response
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            try:
                skills = json.loads(text[start : end + 1])
                if isinstance(skills, list):
                    return self._validate_skills(skills)
            except json.JSONDecodeError:
                pass

        logger.error(f"Failed to parse skill sequence from LLM response:\n{response}")

        # Return minimal fallback
        return [{"skill": "move_to_ready"}]

    def _validate_skills(self, skills: list[dict]) -> list[dict]:
        """Validate and clean skill sequence."""
        valid_skills = {
            "move_to_ready",
            "move_to_position",
            "gripper_open",
            "gripper_close",
            "execute_pick",
            "execute_place",
            "execute_pick_and_place",
        }

        validated = []
        for skill in skills:
            if not isinstance(skill, dict):
                continue
            name = skill.get("skill", "")
            if name not in valid_skills:
                logger.warning(f"Unknown skill '{name}', skipping")
                continue
            validated.append(skill)

        return validated

    def execute_skill_sequence(
        self,
        skills_instance,
        skill_sequence: list[dict[str, Any]],
    ) -> tuple[bool, list[str]]:
        """
        Execute a skill sequence on a SimSkills instance.

        Args:
            skills_instance: SimSkills object
            skill_sequence: List of skill dicts from plan_skills()

        Returns:
            (all_success, error_messages)
        """
        errors = []

        for i, skill_cmd in enumerate(skill_sequence):
            skill_name = skill_cmd.get("skill", "")
            params = skill_cmd.get("params", {})

            logger.info(f"Executing skill {i + 1}/{len(skill_sequence)}: {skill_name}")

            try:
                method = getattr(skills_instance, skill_name, None)
                if method is None:
                    errors.append(f"Skill '{skill_name}' not found")
                    continue

                # Convert list params to numpy arrays where needed
                for key in ("target_xyz", "target_position", "place_position"):
                    if key in params and isinstance(params[key], list):
                        import numpy as np
                        params[key] = np.array(params[key], dtype=np.float64)

                success = method(**params)
                if not success:
                    errors.append(f"Skill '{skill_name}' returned False")

            except Exception as e:
                error_msg = f"Skill '{skill_name}' failed: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return len(errors) == 0, errors
