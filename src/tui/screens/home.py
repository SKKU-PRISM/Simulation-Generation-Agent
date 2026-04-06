"""Home screen — task input + status bar."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static


class StatusBar(Static):
    """Shows current config summary — reads from app.config directly."""

    def render(self) -> str:
        c = self.app.config if hasattr(self.app, "config") else {}
        robot = c.get("robot", "franka")
        provider = c.get("provider", "openai")
        model = c.get("model", "gpt-5")
        mode = c.get("mode", "full")
        mode_label = {
            "full": "Full Pipeline (Stage 1→2→3 + Dataset)",
            "isaac-lab": "Stage 2 Only (YAML → IsaacLab)",
            "data-collection": "Stage 3 Only (Data Collection)",
        }.get(mode, mode)

        return (
            f"  [bold cyan]Robot:[/] {robot}  │  "
            f"[bold cyan]Provider:[/] {provider}  │  "
            f"[bold cyan]Model:[/] {model}\n"
            f"  [bold cyan]Mode:[/] {mode_label}"
        )


class HomeScreen(Screen):
    BINDINGS = [
        ("f1", "settings", "Settings"),
        ("f2", "history", "History"),
        ("f3", "help_screen", "Help"),
        ("escape", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="home-layout"):
            yield Static(
                "\n  [bold cyan]🤖 RAPIDS — Simulation Generation Agent[/]\n",
                id="title",
            )
            yield StatusBar(id="status-bar")
            yield Static("─" * 60, classes="separator")
            # Docker mount warning
            import os
            from pathlib import Path
            outputs_dir = Path("outputs")
            in_docker = os.path.exists("/.dockerenv") or os.path.exists("/workspace/IsaacLab")
            if in_docker and not outputs_dir.is_mount():
                yield Static(
                    "  [bold yellow]⚠ Warning:[/] [yellow]outputs/ is not mounted. "
                    "Results will be lost when the container exits.[/]\n"
                    "  [dim]Use: docker run -it --rm --gpus all "
                    "-v $(pwd)/outputs:/workspace/Simulation-Generation-Agent/outputs "
                    "--env-file .env simgen-agent[/]\n",
                    id="mount-warning",
                )

            yield Static(
                "\n  Describe a task for the robot to perform.\n"
                "  Type [bold]/config[/] for settings, [bold]/history[/] for past runs.\n",
                id="hint",
            )
            yield Input(placeholder="Enter task description...", id="task-input")
            yield Static("", id="feedback")
        yield Footer()

    def on_screen_resume(self) -> None:
        """Refresh status bar when returning from another screen."""
        try:
            self.query_one("#status-bar", StatusBar).refresh()
        except Exception:
            pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return

        event.input.clear()

        if value.startswith("/"):
            cmd = value.lower().split()[0]
            if cmd == "/config":
                self.app.action_settings()
            elif cmd == "/history":
                self.app.action_history()
            elif cmd == "/help":
                self.app.action_help_screen()
            elif cmd in ("/quit", "/exit"):
                self.app.exit()
            else:
                self.query_one("#feedback", Static).update(
                    f"  [dim]Unknown command: {cmd}[/]"
                )
            return

        # Run pipeline with current app config
        self.app.run_task(value)

    def action_settings(self) -> None:
        self.app.action_settings()

    def action_history(self) -> None:
        self.app.action_history()

    def action_help_screen(self) -> None:
        self.app.action_help_screen()

    def action_quit(self) -> None:
        self.app.exit()
