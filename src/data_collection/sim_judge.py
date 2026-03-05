"""
VLM-based task success judge for simulation data collection.

Uses ADC's chain-of-thought judge prompts with SGA's Azure OpenAI (gpt-5)
for multimodal before/after image evaluation.

Falls back to a built-in prompt if ADC submodule is not available.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Fallback prompt (used when ADC submodule is not available)
# ------------------------------------------------------------------ #

_FALLBACK_SYSTEM_PROMPT = """You are a Task Completion Judge for robotic manipulation tasks.

Evaluate whether the robot successfully completed the task by comparing the initial and final images.

Steps:
1. Identify objects in the initial image
2. Identify objects in the final image
3. Analyze if the target object moved
4. Verify the goal was achieved
5. Check spatial relationships match the instruction

Output format:
PREDICTION: [TRUE/FALSE/UNCERTAIN]
REASONING: [One sentence explanation]
"""


def _fallback_build_prompt(
    instruction: str,
    object_positions: dict,
    executed_code: str,
    image_resolution: tuple[int, int] | None = None,
) -> str:
    """Minimal prompt builder when ADC is not available."""
    positions_text = "\n".join(
        f"- {name}: {info}" for name, info in object_positions.items()
    )
    return (
        f"## Task Evaluation\n\n"
        f"### Goal: {instruction}\n\n"
        f"### Object Positions (initial):\n{positions_text}\n\n"
        f"### Executed Skills:\n```\n{executed_code}\n```\n\n"
        f"Analyze the before/after images and provide:\n"
        f"PREDICTION: [TRUE/FALSE/UNCERTAIN]\n"
        f"REASONING: [explanation]"
    )


# ------------------------------------------------------------------ #
# SimJudge
# ------------------------------------------------------------------ #


class SimJudge:
    """VLM judge for simulation data collection episodes.

    Uses ADC's 6-step chain-of-thought prompts with SGA's Azure OpenAI
    API (gpt-5) for multimodal evaluation of before/after images.

    Falls back to built-in prompts if ADC submodule is not available.

    Args:
        model: Azure OpenAI model name (default: gpt-5).
    """

    def __init__(self, model: str = "gpt-5"):
        self._model = model
        self._client = None
        self._system_prompt: str = _FALLBACK_SYSTEM_PROMPT
        self._build_prompt = _fallback_build_prompt
        self._available = False
        self._initialize()

    def _initialize(self):
        """Initialize OpenAI client and load prompts."""
        # Load ADC prompts if available
        try:
            from .adc_imports import get_judge_prompts

            self._system_prompt, self._build_prompt = get_judge_prompts()
            logger.info("SimJudge: loaded ADC judge prompts")
        except ImportError:
            logger.info("SimJudge: ADC prompts not available, using fallback")

        # Initialize Azure OpenAI client
        try:
            from dotenv import load_dotenv
            from openai import OpenAI

            project_root = Path(__file__).resolve().parents[2]
            load_dotenv(project_root / ".env")

            api_key = os.environ.get("AZURE_OPENAI_API_KEY")
            base_url = os.environ.get("AZURE_OPENAI_BASE_URL")

            if not api_key or not base_url:
                logger.warning(
                    "SimJudge: AZURE_OPENAI_API_KEY or AZURE_OPENAI_BASE_URL not set. "
                    "VLM judging disabled."
                )
                return

            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._available = True
            logger.info(f"SimJudge initialized: model={self._model}")
        except ImportError as e:
            logger.warning(f"SimJudge: OpenAI package not available ({e})")

    @property
    def is_available(self) -> bool:
        """Whether the VLM judge is ready to use."""
        return self._available

    def judge_episode(
        self,
        task_description: str,
        initial_image: np.ndarray,
        final_image: np.ndarray,
        object_positions: dict,
        executed_skills: str = "",
    ) -> dict:
        """Judge whether an episode successfully completed the task.

        Args:
            task_description: Natural language task instruction.
            initial_image: RGB (H, W, 3) uint8 before execution.
            final_image: RGB (H, W, 3) uint8 after execution.
            object_positions: ``{name: {"position": [x,y,z]}}`` from SimDetector.
            executed_skills: String description of executed skill sequence.

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
                "reasoning": "VLM judge not available",
                "success": False,
                "raw_response": "",
            }

        # Build user prompt from ADC template (or fallback)
        image_resolution = (initial_image.shape[1], initial_image.shape[0])
        user_prompt = self._build_prompt(
            instruction=task_description,
            object_positions=object_positions,
            executed_code=executed_skills,
            image_resolution=image_resolution,
        )

        # Encode images as base64
        initial_b64 = self._encode_image(initial_image)
        final_b64 = self._encode_image(final_image)

        # Build multimodal input for OpenAI Responses API
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{initial_b64}",
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{final_b64}",
                    },
                ],
            },
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
            }

        # Parse prediction from response
        prediction = self._parse_prediction(raw_text)
        reasoning = self._parse_reasoning(raw_text)

        result = {
            "prediction": prediction,
            "reasoning": reasoning,
            "success": prediction == "TRUE",
            "raw_response": raw_text,
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
