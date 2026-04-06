"""RAPIDS Interactive TUI — Main Application."""

from __future__ import annotations

import sys
from pathlib import Path

from textual.app import App

from .screens.config import ConfigScreen
from .screens.detail import DetailScreen
from .screens.history import HistoryScreen
from .screens.home import HomeScreen
from .screens.running import RunningScreen

# Default config
DEFAULT_CONFIG = {
    "mode": "full",
    "robot": "franka",
    "provider": "openai",
    "model": "gpt-5",
    "temperature": 0.1,
    "max_tokens": 16000,
    "episodes": 1,
    "max_attempts": 3,
    "vlm_enabled": True,
    "vlm_model": "gpt-5",
    "output_root": "outputs/",
    "dataset_dir": "outputs/data_collection/",
    "execution_timeout": 300,
    "max_retries": 5,
    "recording_fps": 20,
    "headless": True,
    "auto_lerobot": True,
}


class RapidsApp(App):
    """RAPIDS Simulation Generation Agent — Interactive TUI."""

    TITLE = "RAPIDS"
    CSS = """
    Screen {
        background: $surface;
    }
    #home-layout {
        padding: 0 2;
    }
    #config-scroll {
        padding: 0 2;
    }
    #detail-scroll {
        padding: 0 2;
    }
    #running-layout {
        padding: 0 2;
    }
    .separator {
        color: $text-muted;
    }
    Input {
        margin: 0 2 1 2;
    }
    Label {
        margin: 0 2;
    }
    RadioSet {
        margin: 0 2 1 2;
    }
    Checkbox {
        margin: 0 2;
    }
    Button {
        margin: 1 2;
    }
    DataTable {
        margin: 0 2;
        height: 1fr;
    }
    RichLog {
        margin: 0 2;
        height: 1fr;
    }
    """

    BINDINGS = [
        ("f1", "settings", "Settings"),
        ("f2", "history", "History"),
        ("f3", "help_screen", "Help"),
    ]

    def __init__(self) -> None:
        self.config = dict(DEFAULT_CONFIG)
        super().__init__()

    def on_mount(self) -> None:
        self.push_screen(HomeScreen(self.config))

    def action_settings(self) -> None:
        self.push_screen(ConfigScreen(self.config))

    def action_history(self) -> None:
        self.push_screen(HistoryScreen())

    def action_help_screen(self) -> None:
        from textual.screen import Screen
        from textual.widgets import Static, Footer, Header
        from textual.app import ComposeResult

        class HelpScreen(Screen):
            BINDINGS = [("escape", "back", "Back")]

            def compose(self) -> ComposeResult:
                yield Header()
                yield Static(
                    "\n"
                    "  [bold cyan]RAPIDS Help[/]\n\n"
                    "  [bold]Commands:[/]\n"
                    "  Type a task description and press Enter to run the pipeline.\n\n"
                    "  [bold]Slash Commands:[/]\n"
                    "  /config   — Open settings\n"
                    "  /history  — View past runs\n"
                    "  /help     — This help screen\n"
                    "  /quit     — Exit\n\n"
                    "  [bold]Keyboard Shortcuts:[/]\n"
                    "  F1        — Settings\n"
                    "  F2        — Run History\n"
                    "  F3        — Help\n"
                    "  Esc       — Back / Quit\n"
                    "  Ctrl+C    — Cancel running pipeline\n\n"
                    "  [bold]Examples:[/]\n"
                    '  "Stack the blocks inside the tray on the table"\n'
                    '  "Pick up the cube and place it on the target"\n'
                    '  "Sort the colored blocks into matching bins"\n'
                )
                yield Footer()

            def action_back(self) -> None:
                self.app.pop_screen()

        self.push_screen(HelpScreen())

    def run_task(self, task_desc: str) -> None:
        self.push_screen(RunningScreen(task_desc, self.config))


def main() -> None:
    # Ensure project root is on PYTHONPATH
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    app = RapidsApp()
    app.run()


if __name__ == "__main__":
    main()
