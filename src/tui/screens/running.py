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
PROGRESS_FRAMES = ["◐", "◓", "◑", "◒"]


def _is_noise(line: str) -> bool:
    """Return True if line is spinner/cursor noise that should not be logged."""
    if not line.strip():
        return True
    if "\x1b[2K" in line or "\x1b[1A" in line:
        return True
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
        self._tick = 0
        # Shared state for stage tracking
        self._stages = {1: "pending", 2: "pending", 3: "pending"}
        self._stage_times: dict[int, str] = {}
        self._stage_start: dict[int, float] = {}
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="running-layout"):
            yield Static(self._build_header(), id="run-header")
            yield Static(self._build_stage_display(), id="stage-status")
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

    def _build_stage_display(self) -> str:
        names = {1: "NL → YAML", 2: "YAML → IsaacLab", 3: "Data Collection"}
        lines = []
        frame = PROGRESS_FRAMES[self._tick % len(PROGRESS_FRAMES)]

        for i in [1, 2, 3]:
            s = self._stages[i]
            t = self._stage_times.get(i, "")
            time_str = f"  {t}" if t else ""

            if s == "done":
                lines.append(f"  [green]✅ Stage {i}: {names[i]}[/]{time_str}")
            elif s == "fail":
                lines.append(f"  [red]❌ Stage {i}: {names[i]}[/]{time_str}")
            elif s == "running":
                # Show animated progress with elapsed time
                stage_elapsed = ""
                if i in self._stage_start:
                    se = int(time.time() - self._stage_start[i])
                    sm, ss = divmod(se, 60)
                    stage_elapsed = f" ({sm}m {ss}s)" if sm else f" ({ss}s)"
                bar_width = 20
                filled = (self._tick % bar_width)
                progress_bar = "░" * filled + "█" + "░" * (bar_width - filled - 1)
                lines.append(
                    f"  [cyan]{frame} Stage {i}: {names[i]}[/]"
                    f"  [cyan]{progress_bar}[/]"
                    f"[dim]{stage_elapsed}[/]"
                )
            else:
                lines.append(f"  [dim]○  Stage {i}: {names[i]}[/]")

        return "\n".join(lines)

    def on_mount(self) -> None:
        self.set_interval(0.5, self._update_display)
        thread = Thread(target=self._run_in_thread, daemon=True)
        thread.start()

    def _update_display(self) -> None:
        self._tick += 1
        if self._finished:
            return
        # Update elapsed
        elapsed = int(time.time() - self._start_time)
        m, s = divmod(elapsed, 60)
        try:
            self.query_one("#elapsed-bar", Static).update(
                f"  [dim]Total elapsed: {m}m {s}s[/]"
            )
            # Refresh stage display with animation
            self.query_one("#stage-status", Static).update(
                self._build_stage_display()
            )
        except Exception:
            pass

    def _run_in_thread(self) -> None:
        log = self.query_one("#run-log", RichLog)

        def on_line(line: str) -> None:
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

            # Detect stage transitions
            if "Stage 1" in clean and "✅" in clean:
                self._stages[1] = "done"
                m = re.search(r"(\d+m?\s*\d+s)", clean)
                if m:
                    self._stage_times[1] = m.group(1)
            elif "Stage 2" in clean and "✅" in clean:
                self._stages[2] = "done"
                m = re.search(r"(\d+m?\s*\d+s)", clean)
                if m:
                    self._stage_times[2] = m.group(1)
            elif "Stage 3" in clean and "✅" in clean:
                self._stages[3] = "done"
                m = re.search(r"(\d+m?\s*\d+s)", clean)
                if m:
                    self._stage_times[3] = m.group(1)
            elif "Stage 1" in clean and "❌" in clean:
                self._stages[1] = "fail"
            elif "Stage 2" in clean and "❌" in clean:
                self._stages[2] = "fail"
            elif "Stage 3" in clean and "❌" in clean:
                self._stages[3] = "fail"
            elif "Stage 1" in clean and self._stages[1] == "pending":
                self._stages[1] = "running"
                self._stage_start[1] = time.time()
            elif "Stage 2" in clean and self._stages[2] == "pending":
                self._stages[2] = "running"
                self._stage_start[2] = time.time()
            elif "Stage 3" in clean and self._stages[3] == "pending":
                self._stages[3] = "running"
                self._stage_start[3] = time.time()

            # Update stage display immediately on transitions
            self.app.call_from_thread(
                self.query_one("#stage-status", Static).update,
                self._build_stage_display(),
            )

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
        for proc in self._process_holder:
            try:
                proc.kill()
            except Exception:
                pass
        self.app.pop_screen()
