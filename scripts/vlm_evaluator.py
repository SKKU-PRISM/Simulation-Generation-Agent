#!/usr/bin/env python3
"""
VLM Evaluator - Evaluates Isaac Sim scenes using Vision Language Models

This module supports multiple VLM backends:
1. Claude Vision (Anthropic API) - requires ANTHROPIC_API_KEY
2. Ollama + LLaVA (local) - requires Ollama running with llava model
3. Mock Evaluator (testing) - generates simulated responses for testing

The module auto-selects the best available backend.
"""

import base64
import json
import logging
import os
import random
import re
import subprocess
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import requests
except ImportError:
    requests = None

try:
    import google.generativeai as genai
    from PIL import Image
except ImportError:
    genai = None
    Image = None

logger = logging.getLogger(__name__)


class BaseVLMEvaluator(ABC):
    """Abstract base class for VLM evaluators."""

    def __init__(self):
        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "prompts" / "vlm_evaluation.md"
        if prompt_path.exists():
            self.prompt_template = prompt_path.read_text()
        else:
            self.prompt_template = self._default_prompt()

    def _default_prompt(self) -> str:
        """Default evaluation prompt if file not found."""
        return """# VLM Scene Evaluation

You are evaluating a robot simulation scene based ONLY on the task document requirements.

## Task Document Requirements
{task_document}

## Evaluation Criteria

1. **Asset Presence (30 points)**: Are ALL assets in the document visible?
2. **Position/Layout (30 points)**: Do positions match the document?
3. **Visual Correctness (20 points)**: Do colors, sizes match?
4. **Scene Completeness (20 points)**: Is the scene ready for the task?

## Output Format (JSON only)

```json
{
  "score": 0-100,
  "breakdown": {
    "asset_presence": {"score": 0-30, "details": "..."},
    "position_layout": {"score": 0-30, "details": "..."},
    "visual_correctness": {"score": 0-20, "details": "..."},
    "scene_completeness": {"score": 0-20, "details": "..."}
  },
  "feedback": {
    "missing_assets": [],
    "position_errors": [],
    "visual_issues": [],
    "general_issues": []
  },
  "document_improvements": []
}
```

Respond with ONLY the JSON object, no markdown or explanation."""

    @abstractmethod
    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """Evaluate a screenshot against task document requirements."""
        pass

    def _parse_response(self, response_text: str) -> dict:
        """Parse the JSON response from VLM."""
        # Try to extract JSON from response
        try:
            # Try direct JSON parse first
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find any JSON object
        json_match = re.search(r'\{[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                # This might be incomplete, try to find the full object
                start = response_text.find('{')
                if start >= 0:
                    # Find matching closing brace
                    depth = 0
                    for i, char in enumerate(response_text[start:]):
                        if char == '{':
                            depth += 1
                        elif char == '}':
                            depth -= 1
                            if depth == 0:
                                json_str = response_text[start:start + i + 1]
                                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If all parsing fails, return error result
        logger.warning(f"Failed to parse VLM response: {response_text[:500]}")
        return {
            "score": 0,
            "breakdown": {
                "asset_presence": {"score": 0, "details": "Failed to parse response"},
                "position_layout": {"score": 0, "details": ""},
                "visual_correctness": {"score": 0, "details": ""},
                "scene_completeness": {"score": 0, "details": ""}
            },
            "feedback": {
                "missing_assets": [],
                "position_errors": [],
                "visual_issues": [],
                "general_issues": ["VLM response parsing failed"]
            },
            "document_improvements": [],
            "raw_response": response_text
        }

    def evaluate_with_retry(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> dict:
        """Evaluate with retry on failure."""
        import time

        last_error = None
        for attempt in range(max_retries):
            result = self.evaluate(screenshot_path, task_document)

            if result.get("success"):
                return result

            last_error = result.get("error", "Unknown error")
            logger.warning(f"Attempt {attempt + 1} failed: {last_error}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        return {
            "success": False,
            "error": f"All {max_retries} attempts failed. Last error: {last_error}"
        }


class MockVLMEvaluator(BaseVLMEvaluator):
    """
    Mock VLM evaluator for testing pipeline flow without real VLM API.

    Generates plausible evaluation results with configurable behavior:
    - Simulates gradual improvement over iterations
    - Can be configured for specific test scenarios
    """

    def __init__(
        self,
        base_score: int = 50,
        improvement_per_iteration: int = 10,
        variance: int = 5
    ):
        super().__init__()
        self.base_score = base_score
        self.improvement_per_iteration = improvement_per_iteration
        self.variance = variance
        self.iteration_count = 0

    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """Generate a mock evaluation result."""
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {
                "success": False,
                "error": f"Screenshot not found: {screenshot_path}"
            }

        # Calculate score with gradual improvement and variance
        base = self.base_score + (self.iteration_count * self.improvement_per_iteration)
        variance = random.randint(-self.variance, self.variance)
        score = min(100, max(0, base + variance))

        # Distribute score across categories
        asset_score = min(30, int(score * 0.30))
        position_score = min(30, int(score * 0.30))
        visual_score = min(20, int(score * 0.20))
        completeness_score = min(20, int(score * 0.20))

        self.iteration_count += 1

        logger.info(f"[MOCK VLM] Generated score: {score} (iteration {self.iteration_count})")

        return {
            "success": True,
            "score": score,
            "breakdown": {
                "asset_presence": {
                    "score": asset_score,
                    "details": f"[MOCK] Assets appear present ({asset_score}/30)"
                },
                "position_layout": {
                    "score": position_score,
                    "details": f"[MOCK] Positions approximated ({position_score}/30)"
                },
                "visual_correctness": {
                    "score": visual_score,
                    "details": f"[MOCK] Visual check ({visual_score}/20)"
                },
                "scene_completeness": {
                    "score": completeness_score,
                    "details": f"[MOCK] Scene completeness ({completeness_score}/20)"
                }
            },
            "feedback": {
                "missing_assets": ["[MOCK] Could not verify specific assets"],
                "position_errors": ["[MOCK] Position verification simulated"],
                "visual_issues": [],
                "general_issues": ["[MOCK] This is a simulated evaluation - no real VLM analysis"]
            },
            "document_improvements": [
                "[MOCK] Consider adding more specific position constraints",
                "[MOCK] Color specifications could be more explicit"
            ],
            "_mock": True,
            "_iteration": self.iteration_count
        }


class OllamaVLMEvaluator(BaseVLMEvaluator):
    """
    VLM evaluator using Ollama with LLaVA or similar vision models.

    Requires Ollama running locally with a vision model (e.g., llava, llava:13b).
    """

    def __init__(
        self,
        model: str = "llava:7b",
        base_url: str = "http://localhost:11434",
        timeout: int = 300
    ):
        super().__init__()
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        if requests is None:
            raise ImportError("requests package is required for Ollama. Install with: pip install requests")

    @staticmethod
    def is_available(base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama is running and has a vision model available."""
        if requests is None:
            return False

        try:
            # Check if Ollama is running
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False

            # Check if a vision model is available
            models = response.json().get("models", [])
            vision_models = ["llava", "bakllava", "llava-llama3", "moondream"]
            for model in models:
                model_name = model.get("name", "").lower()
                if any(vm in model_name for vm in vision_models):
                    return True

            return False
        except Exception:
            return False

    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """Evaluate using Ollama vision model."""
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {
                "success": False,
                "error": f"Screenshot not found: {screenshot_path}"
            }

        # Read and encode image
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Format task document
        if isinstance(task_document, dict):
            doc_str = yaml.dump(task_document, default_flow_style=False, sort_keys=False)
        else:
            doc_str = task_document

        # Build prompt
        prompt = self.prompt_template.replace("{task_document}", doc_str)

        # Call Ollama API
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                    "options": {
                        "temperature": 0.1
                    }
                },
                timeout=self.timeout
            )

            if response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code} - {response.text}"
                }

            response_text = response.json().get("response", "")
            result = self._parse_response(response_text)
            result["success"] = True
            result["_backend"] = "ollama"
            return result

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": f"Ollama request timed out after {self.timeout}s"
            }
        except Exception as e:
            logger.error(f"Ollama VLM evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class ClaudeVLMEvaluator(BaseVLMEvaluator):
    """Evaluates scenes using Claude Vision API."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        temperature: float = 0
    ):
        super().__init__()

        if anthropic is None:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def is_available() -> bool:
        """Check if Claude API is available."""
        return anthropic is not None and os.environ.get("ANTHROPIC_API_KEY") is not None

    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """
        Evaluate a screenshot against task document requirements.

        Args:
            screenshot_path: Path to the screenshot image
            task_document: Task document (dict or YAML string)
            config: Optional evaluation config

        Returns:
            Evaluation result dict with score, breakdown, feedback
        """
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {
                "success": False,
                "error": f"Screenshot not found: {screenshot_path}"
            }

        # Read and encode image
        with open(screenshot_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        # Get media type
        suffix = screenshot_path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_type_map.get(suffix, "image/png")

        # Format task document
        if isinstance(task_document, dict):
            doc_str = yaml.dump(task_document, default_flow_style=False, sort_keys=False)
        else:
            doc_str = task_document

        # Build prompt
        prompt = self.prompt_template.replace("{task_document}", doc_str)

        # Call Claude Vision API
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )

            # Parse response
            response_text = response.content[0].text
            result = self._parse_response(response_text)
            result["success"] = True
            result["_backend"] = "claude"
            return result

        except Exception as e:
            logger.error(f"VLM evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class GeminiVLMEvaluator(BaseVLMEvaluator):
    """
    VLM evaluator using Google Gemini Vision API.

    Requires GOOGLE_API_KEY environment variable.
    Free tier: 15 requests per minute, 1500 requests per day.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0
    ):
        super().__init__()

        if genai is None:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai pillow"
            )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    @staticmethod
    def is_available() -> bool:
        """Check if Gemini API is available."""
        return genai is not None and os.environ.get("GOOGLE_API_KEY") is not None

    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """
        Evaluate a screenshot against task document requirements using Gemini.

        Args:
            screenshot_path: Path to the screenshot image
            task_document: Task document (dict or YAML string)
            config: Optional evaluation config

        Returns:
            Evaluation result dict with score, breakdown, feedback
        """
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {
                "success": False,
                "error": f"Screenshot not found: {screenshot_path}"
            }

        # Load image using PIL
        try:
            image = Image.open(screenshot_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load image: {e}"
            }

        # Format task document
        if isinstance(task_document, dict):
            doc_str = yaml.dump(task_document, default_flow_style=False, sort_keys=False)
        else:
            doc_str = task_document

        # Build prompt
        prompt = self.prompt_template.replace("{task_document}", doc_str)

        # Call Gemini Vision API
        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                )
            )

            # Parse response
            response_text = response.text
            result = self._parse_response(response_text)
            result["success"] = True
            result["_backend"] = "gemini"
            return result

        except Exception as e:
            logger.error(f"Gemini VLM evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class GeminiGTComparisonEvaluator(BaseVLMEvaluator):
    """
    Evaluates scenes by comparing against GT images using Gemini Vision.

    Free alternative to Claude GT comparison.
    """

    def __init__(
        self,
        gt_image_path: str | Path,
        model: str = "gemini-2.5-flash",
        temperature: float = 0
    ):
        self.gt_image_path = Path(gt_image_path)

        if not self.gt_image_path.exists():
            raise FileNotFoundError(f"Ground truth image not found: {gt_image_path}")

        if genai is None:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai pillow"
            )

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def _build_comparison_prompt(self, task_document: dict | str) -> str:
        """Build the prompt for GT vs generated image comparison."""
        if isinstance(task_document, dict):
            doc_str = yaml.dump(task_document, default_flow_style=False, sort_keys=False)
        else:
            doc_str = task_document

        return f"""Compare these two images of robot simulation scenes.

IMAGE 1 is the Ground Truth (Reference) - this is what we want to achieve.
IMAGE 2 is the Generated Scene - this is what we created.

## Task Document (for context)
{doc_str}

## Evaluation Criteria

1. **Scene Layout (30 points)**: Are major elements (robot, table, objects) positioned similarly?
2. **Object Presence (25 points)**: Are all objects from GT visible in generated scene?
3. **Visual Appearance (25 points)**: Do colors, sizes, and shapes match?
4. **Overall Similarity (20 points)**: How close is the overall scene composition?

## Important Notes
- Focus on STRUCTURAL similarity, not pixel-perfect matching
- Different lighting conditions are acceptable
- Minor position variations are acceptable

## Output Format (JSON only)

{{
  "score": 0-100,
  "breakdown": {{
    "scene_layout": {{"score": 0-30, "details": "..."}},
    "object_presence": {{"score": 0-25, "details": "..."}},
    "visual_appearance": {{"score": 0-25, "details": "..."}},
    "overall_similarity": {{"score": 0-20, "details": "..."}}
  }},
  "feedback": {{
    "missing_elements": [],
    "position_differences": [],
    "appearance_differences": [],
    "suggestions": []
  }},
  "document_improvements": []
}}

Respond with ONLY the JSON object, no markdown or explanation."""

    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """Evaluate by comparing generated screenshot against GT image."""
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {
                "success": False,
                "error": f"Screenshot not found: {screenshot_path}"
            }

        # Load both images
        try:
            gt_image = Image.open(self.gt_image_path)
            generated_image = Image.open(screenshot_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to load images: {e}"
            }

        # Build comparison prompt
        prompt = self._build_comparison_prompt(task_document)

        # Call Gemini Vision API with both images
        try:
            response = self.model.generate_content(
                [
                    "Ground Truth (Reference) Image:",
                    gt_image,
                    "Generated Scene Image:",
                    generated_image,
                    prompt
                ],
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                )
            )

            # Parse response
            response_text = response.text
            result = self._parse_response(response_text)
            result["success"] = True
            result["_backend"] = "gemini_gt_comparison"
            result["_gt_image"] = str(self.gt_image_path)
            return result

        except Exception as e:
            logger.error(f"Gemini GT comparison evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


class ClaudeGTComparisonEvaluator(BaseVLMEvaluator):
    """
    Evaluates scenes by comparing generated screenshots against Ground Truth images.

    Sends both GT and generated images to Claude Vision for comparison evaluation.
    This provides more meaningful feedback than pure pixel-based similarity metrics.
    """

    def __init__(
        self,
        gt_image_path: str | Path,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        temperature: float = 0
    ):
        # Don't call super().__init__() to avoid loading the standard prompt template
        self.gt_image_path = Path(gt_image_path)

        if not self.gt_image_path.exists():
            raise FileNotFoundError(f"Ground truth image not found: {gt_image_path}")

        if anthropic is None:
            raise ImportError("anthropic package is required. Install with: pip install anthropic")

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _load_image_base64(self, image_path: Path) -> tuple[str, str]:
        """Load image and return (base64_data, media_type)."""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = image_path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_type_map.get(suffix, "image/png")

        return image_data, media_type

    def _build_comparison_prompt(self, task_document: dict | str) -> str:
        """Build the prompt for GT vs generated image comparison."""
        if isinstance(task_document, dict):
            doc_str = yaml.dump(task_document, default_flow_style=False, sort_keys=False)
        else:
            doc_str = task_document

        return f"""Compare the Generated Scene Image against the Ground Truth (Reference) Image.

## Task Document (for context)
{doc_str}

## Evaluation Criteria

1. **Scene Layout (30 points)**: Are major elements (robot, table, objects) positioned similarly?
   - Robot position and orientation
   - Table position and size
   - Overall scene arrangement

2. **Object Presence (25 points)**: Are all objects from GT visible in generated scene?
   - Robot arm visible and properly loaded
   - Table/surface present
   - Cubes/objects present with correct count

3. **Visual Appearance (25 points)**: Do colors, sizes, and shapes match?
   - Cube colors (should be blue, red, green)
   - Table color/appearance
   - Lighting and overall visual quality

4. **Overall Similarity (20 points)**: How close is the overall scene composition?
   - Camera angle similarity
   - Scene depth and perspective
   - Background and environment

## Important Notes
- Focus on STRUCTURAL similarity, not pixel-perfect matching
- Different lighting conditions are acceptable
- Minor position variations are acceptable
- The key is: could this scene be used for the same robot task?

## Output Format (JSON only)

```json
{{
  "score": 0-100,
  "breakdown": {{
    "scene_layout": {{"score": 0-30, "details": "..."}},
    "object_presence": {{"score": 0-25, "details": "..."}},
    "visual_appearance": {{"score": 0-25, "details": "..."}},
    "overall_similarity": {{"score": 0-20, "details": "..."}}
  }},
  "feedback": {{
    "missing_elements": ["list of elements in GT but missing in generated"],
    "position_differences": ["list of position/layout differences"],
    "appearance_differences": ["list of color/size differences"],
    "suggestions": ["actionable suggestions to improve the generated scene"]
  }},
  "document_improvements": [
    "Specific changes to the task document that would help match GT better"
  ]
}}
```

Respond with ONLY the JSON object, no markdown or explanation."""

    def evaluate(
        self,
        screenshot_path: str | Path,
        task_document: dict | str,
        config: Optional[dict] = None
    ) -> dict:
        """
        Evaluate by comparing generated screenshot against GT image.

        Args:
            screenshot_path: Path to the generated screenshot
            task_document: Task document (for context)
            config: Optional evaluation config

        Returns:
            Evaluation result dict with score, breakdown, feedback
        """
        screenshot_path = Path(screenshot_path)

        if not screenshot_path.exists():
            return {
                "success": False,
                "error": f"Screenshot not found: {screenshot_path}"
            }

        # Load both images
        gt_data, gt_media_type = self._load_image_base64(self.gt_image_path)
        screenshot_data, screenshot_media_type = self._load_image_base64(screenshot_path)

        # Build comparison prompt
        prompt = self._build_comparison_prompt(task_document)

        # Call Claude Vision API with both images
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "## Ground Truth Image (Reference):"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": gt_media_type,
                                "data": gt_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "\n## Generated Scene Image:"
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": screenshot_media_type,
                                "data": screenshot_data
                            }
                        },
                        {
                            "type": "text",
                            "text": f"\n{prompt}"
                        }
                    ]
                }]
            )

            # Parse response
            response_text = response.content[0].text
            result = self._parse_response(response_text)
            result["success"] = True
            result["_backend"] = "gt_comparison"
            result["_gt_image"] = str(self.gt_image_path)
            return result

        except Exception as e:
            logger.error(f"GT comparison evaluation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


def create_evaluator(
    backend: str = "auto",
    gt_image_path: str | Path = None,
    **kwargs
) -> BaseVLMEvaluator:
    """
    Factory function to create the appropriate VLM evaluator.

    Args:
        backend: "auto", "claude", "gemini", "ollama", "mock", "gt_comparison", or "gemini_gt"
            - "auto": Try Gemini, Claude, Ollama, then Mock
            - "claude": Use Claude Vision API (requires ANTHROPIC_API_KEY)
            - "gemini": Use Google Gemini Vision API (requires GOOGLE_API_KEY) - FREE
            - "ollama": Use Ollama with LLaVA (requires Ollama running)
            - "mock": Use mock evaluator for testing
            - "gt_comparison": Compare against GT image using Claude Vision
            - "gemini_gt": Compare against GT image using Gemini Vision - FREE
        gt_image_path: Path to ground truth image (required for gt_comparison/gemini_gt)
        **kwargs: Additional arguments passed to the evaluator

    Returns:
        An instance of a VLM evaluator
    """
    if backend == "mock":
        logger.info("Using Mock VLM evaluator")
        return MockVLMEvaluator(**kwargs)

    if backend == "gemini_gt":
        if gt_image_path is None:
            raise ValueError("gt_image_path is required for gemini_gt backend")
        logger.info(f"Using Gemini GT Comparison evaluator with GT: {gt_image_path}")
        return GeminiGTComparisonEvaluator(gt_image_path=gt_image_path, **kwargs)

    if backend == "gt_comparison":
        if gt_image_path is None:
            raise ValueError("gt_image_path is required for gt_comparison backend")
        logger.info(f"Using Claude GT Comparison evaluator with GT: {gt_image_path}")
        return ClaudeGTComparisonEvaluator(gt_image_path=gt_image_path, **kwargs)

    if backend == "gemini":
        logger.info("Using Gemini Vision API (FREE)")
        return GeminiVLMEvaluator(**kwargs)

    if backend == "claude":
        logger.info("Using Claude Vision API")
        return ClaudeVLMEvaluator(**kwargs)

    if backend == "ollama":
        logger.info("Using Ollama VLM evaluator")
        return OllamaVLMEvaluator(**kwargs)

    # Auto-detect best available backend
    if backend == "auto":
        # If GT image provided, prefer GT comparison with available backend
        if gt_image_path:
            if GeminiVLMEvaluator.is_available():
                logger.info("Auto-selected: Gemini GT Comparison (FREE, GT image provided)")
                return GeminiGTComparisonEvaluator(gt_image_path=gt_image_path, **kwargs)
            if ClaudeVLMEvaluator.is_available():
                logger.info("Auto-selected: Claude GT Comparison (GT image provided)")
                return ClaudeGTComparisonEvaluator(gt_image_path=gt_image_path, **kwargs)

        # Without GT image, use regular evaluator
        if GeminiVLMEvaluator.is_available():
            logger.info("Auto-selected: Gemini Vision API (FREE)")
            return GeminiVLMEvaluator(**kwargs)

        if ClaudeVLMEvaluator.is_available():
            logger.info("Auto-selected: Claude Vision API")
            return ClaudeVLMEvaluator(**kwargs)

        if OllamaVLMEvaluator.is_available():
            logger.info("Auto-selected: Ollama VLM")
            return OllamaVLMEvaluator(**kwargs)

        logger.warning("No VLM backend available, using Mock evaluator")
        return MockVLMEvaluator(**kwargs)

    raise ValueError(f"Unknown backend: {backend}")


def load_task_document(path: str | Path) -> dict:
    """Load task document from YAML file."""
    path = Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Isaac Sim scene with VLM")
    parser.add_argument("--screenshot", "-s", type=str, required=True,
                        help="Path to screenshot image")
    parser.add_argument("--document", "-d", type=str, required=True,
                        help="Path to task document YAML")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output path for evaluation results (JSON)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load document
    document = load_task_document(args.document)

    # Create evaluator
    evaluator = ClaudeVLMEvaluator(model=args.model)

    # Evaluate
    print(f"Evaluating {args.screenshot} against {args.document}...")
    result = evaluator.evaluate_with_retry(args.screenshot, document)

    if result["success"]:
        print(f"\n=== Evaluation Results ===")
        print(f"Overall Score: {result['score']}/100")
        print(f"\nBreakdown:")
        for category, data in result.get("breakdown", {}).items():
            print(f"  {category}: {data['score']} - {data['details']}")

        if result.get("feedback"):
            print(f"\nFeedback:")
            feedback = result["feedback"]
            if feedback.get("missing_assets"):
                print(f"  Missing: {', '.join(feedback['missing_assets'])}")
            if feedback.get("position_errors"):
                print(f"  Position errors:")
                for err in feedback["position_errors"]:
                    print(f"    - {err}")
            if feedback.get("visual_issues"):
                print(f"  Visual issues:")
                for issue in feedback["visual_issues"]:
                    print(f"    - {issue}")

        if result.get("document_improvements"):
            print(f"\nSuggested Document Improvements:")
            for improvement in result["document_improvements"]:
                print(f"  - {improvement}")
    else:
        print(f"Evaluation failed: {result.get('error')}")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
