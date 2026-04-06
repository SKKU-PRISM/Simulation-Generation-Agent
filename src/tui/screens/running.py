"""Running screen — shows pipeline execution with live log."""

from __future__ import annotations

import re
import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, RichLog, Static
from textual.worker import Worker, get_current_worker

from ..runner import run_pipeline
from ..run_index import append_run, build_run_entry


class RunningScreen(Screen):
    BINDINGS = [
        ("ctrl+c", "cancel_run", "Cancel"),
    ]

    def __init__(self, task_desc: str, config: dict) -> None:
        self.task_desc = task_desc
        self.config = config
        self._start_time = time.time()
        self._result: dict | None = None
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="running-layout"):
            yield Static(self._build_header(), id="run-header")
            yield Static("", id="stage-status")
            yield Static("  [bold]── Log ──────────────────────────────────[/]\n")
            yield RichLog(id="run-log", highlight=True, markup=True, wrap=True, max_lines=200)
            yield Static("", id="run-footer")
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

    def on_mount(self) -> None:
        self.run_worker(self._execute_pipeline(), exclusive=True)

    async def _execute_pipeline(self) -> None:
        log = self.query_one("#run-log", RichLog)
        stage_status = self.query_one("#stage-status", Static)
        footer = self.query_one("#run-footer", Static)

        stages = {1: "pending", 2: "pending", 3: "pending"}

        def on_line(line: str) -> None:
            # Strip ANSI codes for parsing
            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)

            # Update stage status
            if "Stage 1" in clean and "✅" in line:
                stages[1] = "done"
            elif "Stage 2" in clean and "✅" in line:
                stages[2] = "done"
            elif "Stage 3" in clean and "✅" in line:
                stages[3] = "done"
            elif "Stage 1" in clean and "❌" in line:
                stages[1] = "fail"
            elif "Stage 2" in clean and "❌" in line:
                stages[2] = "fail"
            elif "Stage 3" in clean and "❌" in line:
                stages[3] = "fail"

            # Build stage display
            stage_lines = []
            for i in [1, 2, 3]:
                s = stages[i]
                names = {1: "NL → YAML", 2: "YAML → IsaacLab", 3: "Data Collection"}
                if s == "done":
                    stage_lines.append(f"  [green]✅ Stage {i}: {names[i]}[/]")
                elif s == "fail":
                    stage_lines.append(f"  [red]❌ Stage {i}: {names[i]}[/]")
                elif "Stage " + str(i) in clean:
                    stages[i] = "running"
                    stage_lines.append(f"  [cyan]⏳ Stage {i}: {names[i]}[/]")
                elif s == "running":
                    stage_lines.append(f"  [cyan]⏳ Stage {i}: {names[i]}[/]")
                else:
                    stage_lines.append(f"  [dim]○  Stage {i}: {names[i]}[/]")

            self.call_from_thread(stage_status.update, "\n".join(stage_lines))

            # Update elapsed
            elapsed = int(time.time() - self._start_time)
            m, s = divmod(elapsed, 60)
            self.call_from_thread(
                footer.update,
                f"\n  [dim]Elapsed: {m}m {s}s[/]"
            )

            # Log line (skip spinner updates)
            if not clean.startswith("�") and clean.strip():
                self.call_from_thread(log.write, line)

        self._result = await self.app.run_in_thread(
            lambda: run_pipeline(
                task_desc=self.task_desc,
                robot=self.config.get("robot", "franka"),
                episodes=self.config.get("episodes", 1),
                max_attempts=self.config.get("max_attempts", 3),
                model=self.config.get("model", "gpt-5"),
                provider=self.config.get("provider", "openai"),
                output_dir=self.config.get("output_root", "outputs"),
                on_line=on_line,
            )
        )

        # Pipeline finished
        if self._result:
            status = self._result.get("status", "unknown")
            if status == "completed":
                self.call_from_thread(
                    log.write, "\n[bold green]Pipeline completed successfully![/]"
                )
            else:
                self.call_from_thread(
                    log.write, f"\n[bold yellow]Pipeline finished: {status}[/]"
                )

            # Save to run index
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

            self.call_from_thread(
                log.write, "\n[dim]Press Esc to return to home.[/]"
            )

    def action_cancel_run(self) -> None:
        self.app.pop_screen()
