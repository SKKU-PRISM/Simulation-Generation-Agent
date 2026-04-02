"""VLM-based scene verification for IsaacLab generated environments.

Two-stage verification:
1. Code-based: LLM checks env_cfg.py against YAML task spec
2. Image-based: VLM scores front/top screenshots against task spec

Usage:
    verifier = SceneVerifier()
    result = verifier.verify(front_img, top_img, env_cfg_text, yaml_text)
"""

import base64
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class SceneVerifier:
    """Verify generated IsaacLab environments using LLM code analysis + VLM image scoring."""

    def __init__(self, model: str = "gpt-5", threshold: int = 70):
        self._model = model
        self.threshold = threshold
        self._client = None
        self._initialize()

    def _initialize(self):
        try:
            from dotenv import load_dotenv
            from openai import OpenAI

            project_root = Path(__file__).resolve().parent.parent.parent.parent
            load_dotenv(project_root / ".env")

            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_BASE_URL")

            if not api_key:
                logger.warning("SceneVerifier: OPENAI_API_KEY not set")
                return

            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            self._client = OpenAI(**client_kwargs)
            logger.info("SceneVerifier initialized: model=%s, threshold=%d", self._model, self.threshold)
        except Exception as e:
            logger.warning("SceneVerifier init failed: %s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_llm(self, messages: list) -> str:
        response = self._client.responses.create(
            model=self._model,
            input=messages,
        )
        if hasattr(response, "usage") and response.usage:
            try:
                from src.agent.common.token_tracker import tracker
                tracker.record(
                    step="scene_verify",
                    model=self._model,
                    input_tokens=getattr(response.usage, "input_tokens", 0),
                    output_tokens=getattr(response.usage, "output_tokens", 0),
                )
            except Exception:
                pass
        return response.output_text

    def verify_code(self, env_cfg_text: str, yaml_text: str) -> dict:
        """Code-based verification: check env_cfg.py against YAML spec using LLM."""
        if not self.available:
            return {"pass": False, "score": 0, "reasoning": "SceneVerifier not available", "issues": []}

        system_prompt = """You are an expert IsaacLab environment code reviewer.
You will receive:
1. A YAML task specification (the intended task)
2. A generated env_cfg.py (the IsaacLab environment code)

Analyze whether the generated code correctly implements the YAML specification.

Check these items:
- All assets from YAML are present in SceneCfg
- Robot type matches YAML specification
- Physics parameters (timestep, decimation, episode_length) match
- Observation, reward, and termination functions align with YAML goals
- Object positions roughly match YAML asset positions

Respond in this exact JSON format:
{"pass": true/false, "score": 0-100, "reasoning": "brief explanation", "issues": ["issue1", "issue2"]}

Return ONLY the JSON, no other text."""

        user_prompt = f"""=== YAML Task Specification ===
{yaml_text[:3000]}

=== Generated env_cfg.py ===
{env_cfg_text[:6000]}"""

        try:
            raw = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ])
            return self._parse_json_response(raw, default_pass=False)
        except Exception as e:
            logger.error("Code verification failed: %s", e)
            return {"pass": False, "score": 0, "reasoning": str(e), "issues": []}

    def verify_image(self, image_path: Path, view_name: str, yaml_text: str) -> dict:
        """VLM image verification: score a single screenshot against task spec."""
        if not self.available:
            return {"score": 0, "reasoning": "SceneVerifier not available", "issues": []}

        if not image_path.exists():
            return {"score": 0, "reasoning": f"Image not found: {image_path}", "issues": []}

        system_prompt = f"""You are an expert robotics simulation evaluator.
You will see a {view_name} view screenshot of an IsaacLab simulation environment.
You will also receive the YAML task specification that describes the intended scene.

Score the environment from 0 to 100 based on:
- Is the correct robot visible? (e.g., Franka Panda with parallel jaw gripper)
- Are the objects from the YAML spec present and roughly positioned correctly?
- Is there a table/ground plane and proper lighting?
- Does the overall scene match the task description?

Respond in this exact JSON format:
{{"score": 0-100, "reasoning": "brief explanation", "issues": ["issue1", "issue2"]}}

Return ONLY the JSON, no other text."""

        user_prompt_content = [
            {"type": "input_text", "text": f"=== YAML Task Specification ===\n{yaml_text[:2000]}"},
            {"type": "input_text", "text": f"\n=== {view_name.upper()} VIEW Screenshot ==="},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{self._encode_image(image_path)}",
            },
        ]

        try:
            raw = self._call_llm([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_content},
            ])
            return self._parse_json_response(raw, default_pass=None)
        except Exception as e:
            logger.error("Image verification (%s) failed: %s", view_name, e)
            return {"score": 0, "reasoning": str(e), "issues": []}

    def verify(
        self,
        front_path: Path,
        top_path: Path,
        env_cfg_text: str,
        yaml_text: str,
    ) -> dict:
        """Full verification: code check + VLM image scoring."""
        # 1. Code-based verification
        logger.info("SceneVerifier: running code verification...")
        code_result = self.verify_code(env_cfg_text, yaml_text)
        logger.info("  Code verification: pass=%s, score=%s", code_result.get("pass"), code_result.get("score"))

        # 2. VLM image verification (front and top separately)
        logger.info("SceneVerifier: running front image verification...")
        front_result = self.verify_image(front_path, "front", yaml_text)
        logger.info("  Front score: %s", front_result.get("score"))

        logger.info("SceneVerifier: running top image verification...")
        top_result = self.verify_image(top_path, "top", yaml_text)
        logger.info("  Top score: %s", top_result.get("score"))

        # 3. VLM pass: either front OR top >= threshold
        front_score = front_result.get("score", 0)
        top_score = top_result.get("score", 0)
        vlm_pass = front_score >= self.threshold or top_score >= self.threshold

        overall_pass = code_result.get("pass", False) and vlm_pass

        logger.info("SceneVerifier: vlm_pass=%s (front=%s, top=%s, threshold=%d), overall=%s",
                     vlm_pass, front_score, top_score, self.threshold, overall_pass)

        return {
            "code_verification": code_result,
            "front_verification": front_result,
            "top_verification": top_result,
            "vlm_pass": vlm_pass,
            "overall_pass": overall_pass,
            "threshold": self.threshold,
        }

    @staticmethod
    def _parse_json_response(raw: str, default_pass=None) -> dict:
        """Parse JSON from LLM/VLM response, with fallback."""
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Find JSON boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            result = json.loads(cleaned)
            if "score" not in result and "pass" not in result:
                result["score"] = 0
            return result
        except json.JSONDecodeError:
            # Try to extract score from text
            score_match = re.search(r'"?score"?\s*:\s*(\d+)', raw)
            score = int(score_match.group(1)) if score_match else 0
            result = {"score": score, "reasoning": raw[:200], "issues": ["Failed to parse JSON response"]}
            if default_pass is not None:
                result["pass"] = default_pass
            return result
