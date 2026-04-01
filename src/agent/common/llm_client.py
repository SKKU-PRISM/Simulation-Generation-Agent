"""Azure OpenAI LLM client for IsaacLab code generation."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


class AzureOpenAIClient:
    """Thin wrapper around Azure OpenAI Responses API."""

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. "
                "Add it to .env file or set the environment variable.\n"
                "  echo 'OPENAI_API_KEY=your-key-here' >> .env"
            )

        base_url = os.environ.get("OPENAI_BASE_URL")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-5-mini")

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 8000,
        temperature: float = 0.1,
        step: str = "llm_generate",
    ) -> str:
        """Generate a response from the LLM.

        Args:
            system_prompt: System-level instructions.
            user_prompt: User query / task content.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature (low = deterministic).
            step: Pipeline step name for token tracking.

        Returns:
            The generated text.
        """
        kwargs = dict(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        # Azure GPT-5 family rejects temperature on the Responses API.
        if temperature is not None and self._supports_temperature():
            kwargs["temperature"] = temperature
        try:
            response = self.client.responses.create(**kwargs)
        except Exception as e:
            if self._is_unsupported_temperature_error(e):
                kwargs.pop("temperature", None)
                response = self.client.responses.create(**kwargs)
            else:
                raise
        if hasattr(response, "usage") and response.usage:
            from src.agent.common.token_tracker import tracker
            tracker.record(
                step=step,
                model=self.model,
                input_tokens=getattr(response.usage, "input_tokens", 0),
                output_tokens=getattr(response.usage, "output_tokens", 0),
            )
        return response.output_text

    def _supports_temperature(self) -> bool:
        model_name = str(self.model or "").lower()
        return not model_name.startswith("gpt-5")

    @staticmethod
    def _is_unsupported_temperature_error(error: Exception) -> bool:
        body = getattr(error, "body", None) or {}
        if isinstance(body, dict) and body.get("param") == "temperature":
            return True
        return "temperature" in str(error).lower()
