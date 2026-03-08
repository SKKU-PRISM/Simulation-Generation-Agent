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
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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
            }
        user_prompt = self._build_prompt(
            instruction=task_description,
            object_positions=object_positions,
            executed_code=executed_code,
            image_resolution=image_resolution,
        )

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
