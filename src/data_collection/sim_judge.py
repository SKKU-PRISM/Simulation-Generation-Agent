"""
VLM-based task success judge for simulation data collection.

Uses ADC's chain-of-thought judge prompts with SGA's Azure OpenAI (gpt-5)
for multimodal before/after image evaluation.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _condition_subject_name(condition: dict[str, Any]) -> str | None:
    subject = condition.get("subject", condition.get("object"))
    return subject if isinstance(subject, str) and subject else None


def _describe_object(name: str, entry: dict[str, Any] | None) -> str:
    if not isinstance(entry, dict):
        return name
    label = entry.get("asset_label")
    if isinstance(label, str) and label.strip() and label.strip().lower() != name.lower():
        return f"{name} ({label.strip()})"
    return name


def _describe_target(name: str, entry: dict[str, Any] | None) -> str:
    if name == "command_pose":
        return "the commanded target pose"
    return _describe_object(name, entry)


def _format_position_hint(entry: dict[str, Any] | None) -> str:
    if not isinstance(entry, dict):
        return ""
    position = entry.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return ""
    try:
        x, y, z = (float(position[0]), float(position[1]), float(position[2]))
    except (TypeError, ValueError):
        return ""
    return f" approximately at ({x:.2f}, {y:.2f}, {z:.2f})"


def _render_condition_for_vlm(
    condition: dict[str, Any],
    object_positions: dict[str, dict[str, Any]] | None,
) -> str:
    object_positions = object_positions or {}
    relation = str(condition.get("relation", condition.get("type", ""))).lower()
    subject_name = _condition_subject_name(condition) or "the object"
    subject_entry = object_positions.get(subject_name)
    subject_text = _describe_object(subject_name, subject_entry)
    target_name = condition.get("target")
    target_entry = object_positions.get(target_name) if isinstance(target_name, str) else None
    target_text = _describe_target(target_name, target_entry) if isinstance(target_name, str) else None
    tol = condition.get("tolerance")
    tol_text = ""
    if isinstance(tol, (int, float)):
        tol_text = f" within {float(tol):.2f} m"

    if relation == "height_above":
        value = condition.get("value")
        if isinstance(value, (int, float)):
            return f"{subject_text} is lifted at least {float(value):.2f} m above its start or support surface."
        return f"{subject_text} is clearly lifted above its start or support surface."

    if relation == "at_position" and target_text:
        return f"{subject_text} is near {target_text}{_format_position_hint(target_entry)}{tol_text}."

    if relation == "on_surface" and target_text:
        return f"{subject_text} is resting on {target_text} and not dropped or floating."

    if relation in {"on_top_of", "stacked"} and target_text:
        return f"{subject_text} is visibly stacked on top of {target_text}."

    if relation == "above" and target_text:
        description = condition.get("description")
        if isinstance(description, str) and description.strip():
            return description.strip()
        return f"{subject_text} is visibly above {target_text}, consistent with being placed on it."

    if relation == "inside_tray":
        objects = condition.get("objects") or []
        if isinstance(objects, list) and objects:
            object_text = ", ".join(str(obj) for obj in objects)
            return f"{object_text} all remain inside the tray boundary in the final state."
        return "All required objects remain inside the tray boundary in the final state."

    if relation == "inside_drawer" and target_text:
        return f"{subject_text} is inside {target_text}, not on top of it and not outside the opening."

    if relation == "position_below":
        value = condition.get("value")
        if isinstance(value, (int, float)):
            return f"{subject_text} is at or below joint position {float(value):.3f}, meaning the drawer or door appears closed."
        return f"{subject_text} is in the closed position."

    if relation == "position_above":
        value = condition.get("value")
        if isinstance(value, (int, float)):
            return f"{subject_text} is at or above joint position {float(value):.3f}, meaning the articulated object appears open."
        return f"{subject_text} is in the open position."

    if relation == "height_below":
        value = condition.get("value")
        if isinstance(value, (int, float)):
            return f"{subject_text} stays below {float(value):.2f} m in height."

    if relation == "inserted_into" and target_text:
        return f"{subject_text} is inserted into {target_text}."

    if relation == "upright":
        return f"{subject_text} remains upright in the final state."

    if relation == "in_slot" and target_text:
        return f"{subject_text} is seated in {target_text}."

    description = condition.get("description")
    if isinstance(description, str) and description.strip():
        return description.strip()

    if target_text:
        return f"{subject_text} satisfies relation '{relation}' with {target_text}."
    return f"{subject_text} satisfies relation '{relation}'."


def build_task_aware_judge_prompt(
    *,
    task_description: str,
    goal_description: str = "",
    goal_conditions: list[dict[str, Any]] | None = None,
    relevant_objects: dict[str, dict[str, Any]] | None = None,
    object_positions: dict[str, dict[str, Any]] | None = None,
    executed_code: str = "",
    image_resolution: tuple[int, int] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a task-aware VLM judging prompt from structured goal data."""

    relevant_objects = relevant_objects or {}
    object_positions = object_positions or {}
    goal_conditions = goal_conditions or []

    checklist = [
        _render_condition_for_vlm(condition, object_positions)
        for condition in goal_conditions
        if isinstance(condition, dict)
    ]

    object_lines = []
    for name in sorted(relevant_objects):
        entry = relevant_objects.get(name, {})
        label = _describe_object(name, entry)
        role = entry.get("task_role")
        role_text = f" [{role}]" if isinstance(role, str) and role else ""
        object_lines.append(f"- {label}{role_text}{_format_position_hint(entry)}")

    code_preview = ""
    if executed_code:
        compact_lines = [line.rstrip() for line in executed_code.strip().splitlines() if line.strip()]
        code_preview = "\n".join(compact_lines[:20])
        if len(compact_lines) > 20:
            code_preview += "\n# ... truncated ..."

    resolution_text = ""
    if image_resolution is not None:
        resolution_text = f"{int(image_resolution[0])}x{int(image_resolution[1])}"

    sections = [
        "You are judging whether the task succeeded from BEFORE and AFTER images.",
        "Use the structured success criteria below as the primary source of truth.",
        "Only mark TRUE if the AFTER images clearly satisfy the final success state.",
        "If the images are ambiguous or occluded, use UNCERTAIN instead of guessing.",
        "",
        "Task Summary:",
        f"- {task_description.strip() or 'No task description provided.'}",
    ]
    if goal_description:
        sections.extend(["", "Goal Description:", f"- {goal_description.strip()}"])

    if checklist:
        sections.extend(["", "Structured Success Criteria:"])
        sections.extend([f"{idx}. {item}" for idx, item in enumerate(checklist, start=1)])

    if object_lines:
        sections.extend(["", "Relevant Visible Objects and Targets:"])
        sections.extend(object_lines)

    sections.extend(
        [
            "",
            "Judging Notes:",
            "- Ignore invisible synthetic helper targets and internal planning anchors.",
            "- Use BEFORE images only as the initial-state reference.",
            "- Judge success from the AFTER images against the structured criteria.",
        ]
    )

    if code_preview:
        sections.extend(["", "Executed Policy Summary:", "```python", code_preview, "```"])

    if resolution_text:
        sections.extend(["", f"Image Resolution: {resolution_text}"])

    sections.extend(
        [
            "",
            "Return exactly two lines:",
            "PREDICTION: TRUE, FALSE, or UNCERTAIN",
            "REASONING: one concise sentence grounded in the final visual evidence",
        ]
    )

    prompt_context = {
        "task_description": task_description,
        "goal_description": goal_description,
        "goal_conditions": goal_conditions,
        "relevant_objects": relevant_objects,
        "image_resolution": image_resolution,
    }
    return "\n".join(sections), prompt_context


# ------------------------------------------------------------------ #
# SimJudge
# ------------------------------------------------------------------ #


class SimJudge:
    """VLM judge for simulation data collection episodes.

    Uses ADC's 6-step chain-of-thought prompts with SGA's Azure OpenAI
    API (gpt-5) for multimodal evaluation of before/after images.

    Args:
        model: Azure OpenAI model name (default: gpt-5).
    """

    def __init__(self, model: str = "gpt-5"):
        self._model = model
        self._client = None
        self._system_prompt: str | None = None
        self._build_prompt = None
        self._available = False
        self._availability_reason = "SimJudge not initialized"
        self._initialize()

    def _initialize(self):
        """Initialize OpenAI client and load prompts."""
        try:
            from .adc_imports import get_judge_prompts

            self._system_prompt, self._build_prompt = get_judge_prompts()
            logger.info("SimJudge: loaded ADC judge prompts")
        except Exception as exc:
            self._availability_reason = f"ADC judge prompts unavailable: {exc}"
            logger.warning("SimJudge: %s", self._availability_reason)
            return

        try:
            from dotenv import load_dotenv
            from openai import OpenAI

            project_root = Path(__file__).resolve().parents[2]
            load_dotenv(project_root / ".env")

            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            base_url = os.environ.get("AZURE_OPENAI_BASE_URL")

            if not api_key or not base_url:
                self._availability_reason = (
                    "AZURE_OPENAI_API_KEY or AZURE_OPENAI_BASE_URL not set"
                )
                logger.warning("SimJudge: %s", self._availability_reason)
                return

            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._available = True
            self._availability_reason = "ready"
            logger.info(f"SimJudge initialized: model={self._model}")
        except ImportError as e:
            self._availability_reason = f"OpenAI package not available ({e})"
            logger.warning("SimJudge: %s", self._availability_reason)

    @property
    def availability_reason(self) -> str:
        """Human-readable reason why the judge is or is not available."""
        return self._availability_reason

    @property
    def is_available(self) -> bool:
        """Whether the VLM judge is ready to use."""
        return self._available

    def judge_episode(
        self,
        task_description: str,
        initial_images: dict[str, np.ndarray] | np.ndarray,
        final_images: dict[str, np.ndarray] | np.ndarray,
        object_positions: dict,
        executed_code: str = "",
        executed_skills: str = "",
        goal_description: str = "",
        goal_conditions: list[dict[str, Any]] | None = None,
        relevant_objects: dict[str, dict[str, Any]] | None = None,
        # Legacy single-image aliases
        initial_image: np.ndarray | None = None,
        final_image: np.ndarray | None = None,
    ) -> dict:
        """Judge whether an episode successfully completed the task.

        Supports multi-image (dict of camera views) and single-image (legacy).

        Args:
            task_description: Natural language task instruction.
            initial_images: Dict ``{"wrist": array, "front": array}`` or single RGB array.
            final_images: Dict ``{"wrist": array, "front": array}`` or single RGB array.
            object_positions: ``{name: {"position": [x,y,z]}}`` from SimDetector.
            executed_code: Executed generated code (preferred).
            executed_skills: Legacy string description of executed skills.
            initial_image: Legacy single-image (used if initial_images is None).
            final_image: Legacy single-image (used if final_images is None).

        Returns:
            Dict with keys:
            - ``prediction``: "TRUE", "FALSE", or "UNCERTAIN"
            - ``reasoning``: Explanation string
            - ``success``: Boolean (True if prediction == "TRUE")
            - ``raw_response``: Full VLM response text
        """
        if not self._available:
            return {
                "prediction": "UNCERTAIN",
                "reasoning": self._availability_reason,
                "success": False,
                "raw_response": "",
                "user_prompt": "",
                "prompt_context": {},
            }

        executed_code = executed_code or executed_skills

        # Normalize inputs: accept both dict and single-image
        if initial_images is None and initial_image is not None:
            initial_images = initial_image
        if final_images is None and final_image is not None:
            final_images = final_image

        # Convert single images to dict format
        if isinstance(initial_images, np.ndarray):
            initial_images = {"view": initial_images}
        if isinstance(final_images, np.ndarray):
            final_images = {"view": final_images}

        # Get resolution from first image
        first_img = next(iter(initial_images.values()))
        image_resolution = (first_img.shape[1], first_img.shape[0])
        if self._build_prompt is None or self._system_prompt is None:
            return {
                "prediction": "UNCERTAIN",
                "reasoning": "ADC judge prompts unavailable",
                "success": False,
                "raw_response": "",
                "user_prompt": "",
                "prompt_context": {},
            }
        if goal_description or goal_conditions or relevant_objects is not None:
            user_prompt, prompt_context = build_task_aware_judge_prompt(
                task_description=task_description,
                goal_description=goal_description,
                goal_conditions=goal_conditions,
                relevant_objects=relevant_objects,
                object_positions=object_positions,
                executed_code=executed_code,
                image_resolution=image_resolution,
            )
        else:
            user_prompt = self._build_prompt(
                instruction=task_description,
                object_positions=object_positions,
                executed_code=executed_code,
                image_resolution=image_resolution,
            )
            prompt_context = {
                "task_description": task_description,
                "goal_description": "",
                "goal_conditions": [],
                "relevant_objects": object_positions or {},
                "image_resolution": image_resolution,
            }

        # Build multimodal content: text + interleaved before/after images per view
        content = [{"type": "input_text", "text": user_prompt}]

        # Add labeled images: "BEFORE (view_name)" then "AFTER (view_name)"
        for view_name in initial_images:
            if initial_images[view_name] is not None:
                content.append({
                    "type": "input_text",
                    "text": f"BEFORE ({view_name} camera):",
                })
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self._encode_image(initial_images[view_name])}",
                })
            if view_name in final_images and final_images[view_name] is not None:
                content.append({
                    "type": "input_text",
                    "text": f"AFTER ({view_name} camera):",
                })
                content.append({
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self._encode_image(final_images[view_name])}",
                })

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]

        # Call VLM via Responses API
        try:
            response = self._client.responses.create(
                model=self._model,
                input=messages,
            )
            raw_text = response.output_text
        except Exception as e:
            logger.error(f"SimJudge VLM call failed: {e}")
            return {
                "prediction": "UNCERTAIN",
                "reasoning": f"VLM call error: {e}",
                "success": False,
                "raw_response": "",
                "user_prompt": user_prompt,
                "prompt_context": prompt_context,
            }

        # Parse prediction from response
        prediction = self._parse_prediction(raw_text)
        reasoning = self._parse_reasoning(raw_text)

        result = {
            "prediction": prediction,
            "reasoning": reasoning,
            "success": prediction == "TRUE",
            "raw_response": raw_text,
            "user_prompt": user_prompt,
            "prompt_context": prompt_context,
        }

        logger.info(
            f"SimJudge verdict: {prediction} — {reasoning[:80]}"
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        """Encode RGB numpy array to base64 PNG string."""
        try:
            import cv2

            # RGB → BGR for cv2
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".png", bgr)
            return base64.b64encode(buf.tobytes()).decode("utf-8")
        except ImportError:
            # Fallback: use PIL
            from io import BytesIO

            from PIL import Image

            img = Image.fromarray(image)
            buf = BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_prediction(text: str) -> str:
        """Extract PREDICTION: TRUE/FALSE/UNCERTAIN from VLM response."""
        match = re.search(r"PREDICTION:\s*(TRUE|FALSE|UNCERTAIN)", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Fallback: look for standalone TRUE/FALSE
        if "TRUE" in text.upper().split("\n")[-5:]:
            return "TRUE"
        if "FALSE" in text.upper().split("\n")[-5:]:
            return "FALSE"
        return "UNCERTAIN"

    @staticmethod
    def _parse_reasoning(text: str) -> str:
        """Extract REASONING line from VLM response."""
        match = re.search(r"REASONING:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Return last non-empty line as reasoning
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        return lines[-1] if lines else "No reasoning provided"
