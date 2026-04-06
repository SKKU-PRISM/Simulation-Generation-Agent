"""Config screen — pipeline settings with API key management."""

from __future__ import annotations

import os
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Placeholder shown when a key is already set (never reveal actual value)
_KEY_PLACEHOLDER = "\u2022" * 12  # ••••••••••••


def _has_env_key(var_name: str) -> bool:
    """Check if an API key env var is set and non-empty."""
    val = os.environ.get(var_name, "")
    return bool(val and val != "your-key" and not val.startswith("your-"))


def _update_env_file(env_path: Path, key: str, value: str) -> None:
    """Update or append a key=value pair in a .env file."""
    lines: list[str] = []
    found = False
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
                lines.append(f"{key}={value}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key}={value}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ConfigScreen(Screen):
    BINDINGS = [
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, config: dict) -> None:
        self.config = dict(config)
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="config-scroll"):
            yield Static("\n  [bold cyan]⚙️  Settings[/]\n")

            # Pipeline
            yield Static("  [bold]── Pipeline ──────────────────────────[/]")
            yield Label("  Mode")
            with RadioSet(id="mode-select"):
                yield RadioButton("Full Pipeline (Stage 1→2→3 + Dataset)", value=self.config.get("mode") == "full", id="mode-full")
                yield RadioButton("Stage 2 Only (YAML → IsaacLab)", value=self.config.get("mode") == "isaac-lab", id="mode-isaaclab")
                yield RadioButton("Stage 3 Only (Data Collection)", value=self.config.get("mode") == "data-collection", id="mode-dc")
            yield Label("  Episodes")
            yield Input(value=str(self.config.get("episodes", 1)), id="episodes-input", type="integer")
            yield Label("  Max Attempts")
            yield Input(value=str(self.config.get("max_attempts", 3)), id="max-attempts-input", type="integer")

            # LLM
            yield Static("\n  [bold]── LLM ──────────────────────────────[/]")
            yield Label("  Provider")
            with RadioSet(id="provider-select"):
                yield RadioButton("OpenAI", value=self.config.get("provider") == "openai", id="prov-openai")
                yield RadioButton("Azure OpenAI", value=self.config.get("provider") == "azure", id="prov-azure")
                yield RadioButton("AWS Bedrock", value=self.config.get("provider") == "bedrock", id="prov-bedrock")
            yield Label("  Model")
            yield Input(value=self.config.get("model", "gpt-5"), id="model-input")
            yield Label("  Temperature")
            yield Input(value=str(self.config.get("temperature", 0.1)), id="temp-input")
            yield Label("  Max Tokens")
            yield Input(value=str(self.config.get("max_tokens", 16000)), id="max-tokens-input", type="integer")

            # API Keys
            yield Static("\n  [bold]── API Keys ─────────────────────────[/]")
            yield Static("  [dim]Keys are saved to .env and masked for security.[/]")

            yield Label("  OpenAI API Key")
            yield Input(
                value=_KEY_PLACEHOLDER if _has_env_key("OPENAI_API_KEY") else "",
                id="openai-key-input",
                password=True,
                placeholder="sk-...",
            )

            yield Label("  Azure OpenAI API Key")
            yield Input(
                value=_KEY_PLACEHOLDER if _has_env_key("AZURE_OPENAI_API_KEY") else "",
                id="azure-key-input",
                password=True,
                placeholder="Azure API key...",
            )
            yield Label("  Azure OpenAI Base URL")
            yield Input(
                value=os.environ.get("AZURE_OPENAI_BASE_URL", ""),
                id="azure-url-input",
                placeholder="https://your-resource.openai.azure.com/openai/v1/",
            )

            yield Label("  HuggingFace Token")
            yield Input(
                value=_KEY_PLACEHOLDER if _has_env_key("HF_TOKEN") else "",
                id="hf-token-input",
                password=True,
                placeholder="hf_...",
            )

            # Robot
            yield Static("\n  [bold]── Robot ─────────────────────────────[/]")
            yield Label("  Embodiment")
            with RadioSet(id="robot-select"):
                yield RadioButton("Franka Panda", value=self.config.get("robot") == "franka", id="robot-franka")
                yield RadioButton("UR10e", value=self.config.get("robot") == "ur10e", id="robot-ur10e")
                yield RadioButton("OpenARM", value=self.config.get("robot") == "openarm", id="robot-openarm")
                yield RadioButton("SO-101", value=self.config.get("robot") == "so101", id="robot-so101")

            # VLM Judge
            yield Static("\n  [bold]── VLM Judge ────────────────────────[/]")
            yield Checkbox("VLM Judge Enabled", value=self.config.get("vlm_enabled", True), id="vlm-toggle")
            yield Label("  VLM Model")
            yield Input(value=self.config.get("vlm_model", "gpt-5"), id="vlm-model-input")

            # Paths
            yield Static("\n  [bold]── Paths ─────────────────────────────[/]")
            yield Label("  Output Root")
            yield Input(value=self.config.get("output_root", "outputs/"), id="output-root-input")
            yield Label("  Dataset Dir")
            yield Input(value=self.config.get("dataset_dir", "outputs/data_collection/"), id="dataset-dir-input")

            # Advanced
            yield Static("\n  [bold]── Advanced ──────────────────────────[/]")
            yield Label("  Execution Timeout (s)")
            yield Input(value=str(self.config.get("execution_timeout", 300)), id="timeout-input", type="integer")
            yield Label("  Max Retries (Stage 2)")
            yield Input(value=str(self.config.get("max_retries", 5)), id="retries-input", type="integer")
            yield Label("  Recording FPS")
            yield Input(value=str(self.config.get("recording_fps", 20)), id="fps-input", type="integer")
            yield Checkbox("Headless Mode", value=self.config.get("headless", True), id="headless-toggle")
            yield Checkbox("Auto LeRobot Conversion", value=self.config.get("auto_lerobot", True), id="lerobot-toggle")

            yield Static("")
            yield Button("Save", variant="primary", id="save-btn")
            yield Static("")

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self._apply_config()
            self._save_api_keys()
            self.app.pop_screen()

    def _apply_config(self) -> None:
        # Mode
        mode_set = self.query_one("#mode-select", RadioSet)
        if mode_set.pressed_index == 0:
            self.config["mode"] = "full"
        elif mode_set.pressed_index == 1:
            self.config["mode"] = "isaac-lab"
        elif mode_set.pressed_index == 2:
            self.config["mode"] = "data-collection"

        # Provider
        prov_set = self.query_one("#provider-select", RadioSet)
        if prov_set.pressed_index == 0:
            self.config["provider"] = "openai"
        elif prov_set.pressed_index == 1:
            self.config["provider"] = "azure"
        elif prov_set.pressed_index == 2:
            self.config["provider"] = "bedrock"

        # Robot
        robot_set = self.query_one("#robot-select", RadioSet)
        robots = ["franka", "ur10e", "openarm", "so101"]
        idx = robot_set.pressed_index
        if idx is not None and 0 <= idx < len(robots):
            self.config["robot"] = robots[idx]

        # Text inputs (with validation)
        def _int(widget_id: str, default: int) -> int:
            try:
                return int(self.query_one(widget_id, Input).value or default)
            except (ValueError, TypeError):
                return default

        def _float(widget_id: str, default: float) -> float:
            try:
                return float(self.query_one(widget_id, Input).value or default)
            except (ValueError, TypeError):
                return default

        self.config["episodes"] = _int("#episodes-input", 1)
        self.config["max_attempts"] = _int("#max-attempts-input", 3)
        self.config["model"] = self.query_one("#model-input", Input).value or "gpt-5"
        self.config["temperature"] = _float("#temp-input", 0.1)
        self.config["max_tokens"] = _int("#max-tokens-input", 16000)
        self.config["vlm_model"] = self.query_one("#vlm-model-input", Input).value or "gpt-5"
        self.config["output_root"] = self.query_one("#output-root-input", Input).value or "outputs/"
        self.config["dataset_dir"] = self.query_one("#dataset-dir-input", Input).value or "outputs/data_collection/"
        self.config["execution_timeout"] = _int("#timeout-input", 300)
        self.config["max_retries"] = _int("#retries-input", 5)
        self.config["recording_fps"] = _int("#fps-input", 20)

        # Toggles
        self.config["vlm_enabled"] = self.query_one("#vlm-toggle", Checkbox).value
        self.config["headless"] = self.query_one("#headless-toggle", Checkbox).value
        self.config["auto_lerobot"] = self.query_one("#lerobot-toggle", Checkbox).value

        # Propagate to app
        self.app.config.update(self.config)

    def _save_api_keys(self) -> None:
        """Save API keys to .env file and set in current process env."""
        env_path = PROJECT_ROOT / ".env"

        key_fields = [
            ("openai-key-input", "OPENAI_API_KEY"),
            ("azure-key-input", "AZURE_OPENAI_API_KEY"),
            ("hf-token-input", "HF_TOKEN"),
        ]

        for widget_id, env_var in key_fields:
            value = self.query_one(f"#{widget_id}", Input).value
            # Skip if empty, unchanged placeholder, or masked
            if not value or value == _KEY_PLACEHOLDER:
                continue
            _update_env_file(env_path, env_var, value)
            os.environ[env_var] = value

        # Azure Base URL (not a secret, no masking needed)
        azure_url = self.query_one("#azure-url-input", Input).value
        if azure_url:
            _update_env_file(env_path, "AZURE_OPENAI_BASE_URL", azure_url)
            os.environ["AZURE_OPENAI_BASE_URL"] = azure_url

    def action_cancel(self) -> None:
        self.app.pop_screen()
