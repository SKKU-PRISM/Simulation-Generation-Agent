"""Running screen — shows pipeline execution with live progress and log."""

from __future__ import annotations

import re
import time
from threading import Thread

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static

from ..runner import run_pipeline
from ..run_index import append_run, build_run_entry


SPINNER_CHARS = set("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


def _is_noise(line: str) -> bool:
    """Return True if line is spinner/cursor noise that should not be logged."""
    if not line.strip():
        return True
    # Cursor control sequences
    if "\x1b[2K" in line or "\x1b[1A" in line:
        return True
    # Spinner characters
    if any(c in line for c in SPINNER_CHARS):
        return True
    return False


class RunningScreen(Screen):
    BINDINGS = [
        ("ctrl+c", "cancel_run", "Cancel"),
        ("escape", "cancel_run", "Back"),
    ]

    def __init__(self, task_desc: str, config: dict) -> None:
        self.task_desc = task_desc
        self.config = config
        self._start_time = time.time()
        self._result: dict | None = None
        self._process_holder: list = []
        self._finished = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="running-layout"):
            yield Static(self._build_header(), id="run-header")
            yield Static(self._build_initial_stages(), id="stage-status")
            yield Static("", id="elapsed-bar")
            yield Static("  [bold]── Log ──────────────────────────────────[/]\n")
            yield RichLog(id="run-log", highlight=True, markup=True, wrap=True, max_lines=200)
        yield Footer()

    def _build_header(self) -> str:
        c = self.config
        return (
            f"\n  [bold cyan]🚀 Running Pipeline[/]\n\n"
            f"  [bold]Task:[/]  \"{self.task_desc}\"\n"
            f"  [bold]Robot:[/] {c.get('robot', 'franka')}  │  "
            f"[bold]Episodes:[/] {c.get('episodes', 1)}  │  "
            f"[bold]Model:[/] {c.get('model', 'gpt-5')}\n"
        )

    def _build_initial_stages(self) -> str:
        return (
            "  [dim]○  Stage 1: NL → YAML[/]\n"
            "  [dim]○  Stage 2: YAML → IsaacLab[/]\n"
            "  [dim]○  Stage 3: Data Collection[/]"
        )

    def on_mount(self) -> None:
        self.set_interval(1.0, self._update_elapsed)
        thread = Thread(target=self._run_in_thread, daemon=True)
        thread.start()

    def _update_elapsed(self) -> None:
        if self._finished:
            return
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        try:
            bar = self.query_one("#elapsed-bar", Static)
            bar.update(f"  [dim]Elapsed: {m}m {s}s[/]")
        except Exception:
            pass

    def _run_in_thread(self) -> None:
        log = self.query_one("#run-log", RichLog)
        stage_status = self.query_one("#stage-status", Static)

        stages = {1: "pending", 2: "pending", 3: "pending"}
        stage_times: dict[int, str] = {}

        def _build_stage_display() -> str:
            names = {1: "NL → YAML", 2: "YAML → IsaacLab", 3: "Data Collection"}
            lines = []
            for i in [1, 2, 3]:
                s = stages[i]
                t = stage_times.get(i, "")
                time_str = f"  {t}" if t else ""
                if s == "done":
                    lines.append(f"  [green]✅ Stage {i}: {names[i]}[/]{time_str}")
                elif s == "fail":
                    lines.append(f"  [red]❌ Stage {i}: {names[i]}[/]{time_str}")
                elif s == "running":
                    lines.append(f"  [cyan]⏳ Stage {i}: {names[i]}  ...[/]")
                else:
                    lines.append(f"  [dim]○  Stage {i}: {names[i]}[/]")
            return "\n".join(lines)

        def on_line(line: str) -> None:
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

            # Detect stage transitions
            if "Stage 1" in clean and "✅" in clean:
                stages[1] = "done"
                m = re.search(r"(\d+m?\s*\d+s)", clean)
                if m:
                    stage_times[1] = m.group(1)
            elif "Stage 2" in clean and "✅" in clean:
                stages[2] = "done"
                m = re.search(r"(\d+m?\s*\d+s)", clean)
                if m:
                    stage_times[2] = m.group(1)
            elif "Stage 3" in clean and "✅" in clean:
                stages[3] = "done"
                m = re.search(r"(\d+m?\s*\d+s)", clean)
                if m:
                    stage_times[3] = m.group(1)
            elif "Stage 1" in clean and "❌" in clean:
                stages[1] = "fail"
            elif "Stage 2" in clean and "❌" in clean:
                stages[2] = "fail"
            elif "Stage 3" in clean and "❌" in clean:
                stages[3] = "fail"
            elif "Stage 1" in clean and stages[1] == "pending":
                stages[1] = "running"
            elif "Stage 2" in clean and stages[2] == "pending":
                stages[2] = "running"
            elif "Stage 3" in clean and stages[3] == "pending":
                stages[3] = "running"

            self.app.call_from_thread(stage_status.update, _build_stage_display())

            # Log meaningful lines only
            if not _is_noise(line):
                self.app.call_from_thread(log.write, clean)

        self._result = run_pipeline(
            task_desc=self.task_desc,
            robot=self.config.get("robot", "franka"),
            episodes=self.config.get("episodes", 1),
            max_attempts=self.config.get("max_attempts", 3),
            model=self.config.get("model", "gpt-5"),
            provider=self.config.get("provider", "openai"),
            output_dir=self.config.get("output_root", "outputs"),
            on_line=on_line,
            process_holder=self._process_holder,
        )

        self._finished = True

        if self._result:
            status = self._result.get("status", "unknown")
            if status == "completed":
                self.app.call_from_thread(
                    log.write, "\n[bold green]Pipeline completed successfully![/]"
                )
            else:
                self.app.call_from_thread(
                    log.write, f"\n[bold yellow]Pipeline finished: {status}[/]"
                )

            try:
                entry = build_run_entry(
                    result=self._result,
                    task_desc=self.task_desc,
                    robot=self.config.get("robot", "franka"),
                    model=self.config.get("model", "gpt-5"),
                    provider=self.config.get("provider", "openai"),
                    mode=self.config.get("mode", "full"),
                    episodes_target=self.config.get("episodes", 1),
                )
                append_run(entry)
            except Exception:
                pass

            self.app.call_from_thread(
                log.write, "\n[dim]Press Esc to return to home.[/]"
            )

    def action_cancel_run(self) -> None:
        # Kill subprocess if running
        for proc in self._process_holder:
            try:
                proc.kill()
            except Exception:
                pass
        self.app.pop_screen()
