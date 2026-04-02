"""VLM-based scene verification for IsaacLab generated environments.

Two independent evaluation systems:
1. Code-based (4-category, 100 points): LLM analyzes env_cfg.py against YAML spec
   - Scene Fidelity (30), MDP Correctness (25), Task Alignment (25), Runtime Validity (20)
2. Image-based (separate): VLM scores front/top screenshots against task spec
   - front/top each 0-100, pass if either >= threshold

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

    # ---- Code-based evaluation (4-category, 100 points) ----

    def verify_code(self, env_cfg_text: str, yaml_text: str) -> dict:
        """Code-based verification using 4-category 100-point scoring system.

        Returns:
            {
                "scene_fidelity": {"score": int, "max": 30, "details": str},
                "mdp_correctness": {"score": int, "max": 25, "details": str},
                "task_alignment": {"score": int, "max": 25, "details": str},
                "runtime_validity": {"score": int, "max": 20, "details": str},
                "total_score": int,
                "max_score": 100,
                "issues": [str]
            }
        """
        if not self.available:
            return self._empty_code_result("SceneVerifier not available")

        system_prompt = """You are an expert IsaacLab environment code reviewer.
You will receive a YAML task specification and a generated env_cfg.py.
Evaluate the code using the following 4-category scoring system (100 points total):

## Scene Fidelity (30 points)
- asset_completeness (10): Are ALL assets from YAML present in SceneCfg?
- asset_config (8): Do positions, USD paths, and physics match YAML?
- physics_config (5): Do timestep, decimation, episode_length match?
- robot_config (4): Is the correct robot type used?
- scene_structure (3): Are ground plane, lighting, env_spacing present?

## MDP Correctness (25 points)
- observation_coverage (7): Are relevant observations defined?
- observation_validity (3): Do observation functions exist and make sense?
- action_space (5): Is action space properly configured for the robot?
- reward_structure (5): Are reward terms defined and meaningful?
- termination_coverage (3): Are termination conditions appropriate?
- event_coverage (2): Are reset/randomization events defined?

## Task Alignment (25 points)
- goal_condition_mapping (10): Do rewards/terminations reflect YAML goal conditions?
- threshold_preservation (8): Are YAML thresholds preserved in code?
- custom_mdp_validity (7): Are custom MDP functions (mdp/*.py) valid?

## Runtime Validity (20 points)
- env_creation (5): Will the environment instantiate without errors?
- reset_step_cycle (5): Will reset/step loop work correctly?
- reward_computation (5): Will rewards compute without NaN/errors?
- physics_stability (5): Are physics parameters stable (no extreme values)?

Respond in this exact JSON format:
{
  "scene_fidelity": {"score": 0-30, "details": "brief explanation"},
  "mdp_correctness": {"score": 0-25, "details": "brief explanation"},
  "task_alignment": {"score": 0-25, "details": "brief explanation"},
  "runtime_validity": {"score": 0-20, "details": "brief explanation"},
  "issues": ["issue1", "issue2"]
}

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
            parsed = self._parse_json_response(raw)
            return self._normalize_code_result(parsed)
        except Exception as e:
            logger.error("Code verification failed: %s", e)
            return self._empty_code_result(str(e))

    def _empty_code_result(self, reason: str) -> dict:
        return {
            "scene_fidelity": {"score": 0, "max": 30, "details": reason},
            "mdp_correctness": {"score": 0, "max": 25, "details": reason},
            "task_alignment": {"score": 0, "max": 25, "details": reason},
            "runtime_validity": {"score": 0, "max": 20, "details": reason},
            "total_score": 0,
            "max_score": 100,
            "issues": [reason],
        }

    def _normalize_code_result(self, parsed: dict) -> dict:
        categories = {
            "scene_fidelity": 30,
            "mdp_correctness": 25,
            "task_alignment": 25,
            "runtime_validity": 20,
        }
        result = {}
        total = 0
        for cat, max_score in categories.items():
            entry = parsed.get(cat, {})
            if isinstance(entry, dict):
                score = min(int(entry.get("score", 0)), max_score)
                details = entry.get("details", "")
            elif isinstance(entry, (int, float)):
                score = min(int(entry), max_score)
                details = ""
            else:
                score = 0
                details = ""
            result[cat] = {"score": score, "max": max_score, "details": details}
            total += score

        result["total_score"] = total
        result["max_score"] = 100
        result["issues"] = parsed.get("issues", [])
        return result

    # ---- Image-based evaluation (separate scoring) ----

    def verify_image(self, image_path: Path, view_name: str, yaml_text: str) -> dict:
        """VLM image verification: score a single screenshot against task spec.

        Returns:
            {"score": 0-100, "reasoning": str, "issues": [str]}
        """
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
            return self._parse_json_response(raw)
        except Exception as e:
            logger.error("Image verification (%s) failed: %s", view_name, e)
            return {"score": 0, "reasoning": str(e), "issues": []}

    # ---- Full verification ----

    def verify(
        self,
        front_path: Path,
        top_path: Path,
        env_cfg_text: str,
        yaml_text: str,
    ) -> dict:
        """Full verification: code-based 4-category + VLM image scoring (separate)."""
        # 1. Code-based evaluation (4-category, 100 points)
        logger.info("SceneVerifier: running code evaluation (4-category)...")
        code_eval = self.verify_code(env_cfg_text, yaml_text)
        logger.info("  Code total: %s/100 (SF=%s MDP=%s TA=%s RV=%s)",
                     code_eval["total_score"],
                     code_eval["scene_fidelity"]["score"],
                     code_eval["mdp_correctness"]["score"],
                     code_eval["task_alignment"]["score"],
                     code_eval["runtime_validity"]["score"])

        # 2. VLM image evaluation (separate, front and top)
        logger.info("SceneVerifier: running front image evaluation...")
        front_result = self.verify_image(front_path, "front", yaml_text)
        logger.info("  Front score: %s/100", front_result.get("score"))

        logger.info("SceneVerifier: running top image evaluation...")
        top_result = self.verify_image(top_path, "top", yaml_text)
        logger.info("  Top score: %s/100", top_result.get("score"))

        # 3. VLM pass: either front OR top >= threshold
        front_score = front_result.get("score", 0)
        top_score = top_result.get("score", 0)
        vlm_pass = front_score >= self.threshold or top_score >= self.threshold

        # 4. Overall pass: code total >= 60 AND vlm pass
        code_pass = code_eval["total_score"] >= 60
        overall_pass = code_pass and vlm_pass

        logger.info("SceneVerifier: code_pass=%s (%s/100), vlm_pass=%s (front=%s, top=%s), overall=%s",
                     code_pass, code_eval["total_score"], vlm_pass, front_score, top_score, overall_pass)

        return {
            "code_evaluation": code_eval,
            "image_evaluation": {
                "front": front_result,
                "top": top_result,
                "vlm_pass": vlm_pass,
                "threshold": self.threshold,
            },
            "overall_pass": overall_pass,
        }

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Parse JSON from LLM/VLM response, with fallback."""
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            score_match = re.search(r'"?score"?\s*:\s*(\d+)', raw)
            score = int(score_match.group(1)) if score_match else 0
            return {"score": score, "reasoning": raw[:200], "issues": ["Failed to parse JSON response"]}
