"""History screen — run list with navigation."""

from __future__ import annotations

from datetime import datetime

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Static

from ..run_index import import_existing_runs, load_runs


class HistoryScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("\n  [bold cyan]📋 Run History[/]\n")
        yield DataTable(id="history-table", cursor_type="row")
        yield Static("", id="history-summary")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.add_columns("#", "Status", "Task", "Robot", "Ep", "Score", "Model", "Date")

        runs = load_runs()
        if not runs:
            try:
                runs = import_existing_runs()
            except Exception:
                pass

        completed = partial = failed = 0

        for i, run in enumerate(runs):
            status = run.get("status", "unknown")
            if status == "completed":
                completed += 1
                status_str = "[green]completed[/]"
            elif status == "partial":
                partial += 1
                status_str = "[yellow]partial[/]"
            else:
                failed += 1
                status_str = "[red]failed[/]"

            task = run.get("task", "unknown")
            if len(task) > 25:
                task = task[:22] + "..."

            score = run.get("eval_score")
            score_str = str(score) if score is not None else "-"

            model = run.get("model", "-")

            date_str = ""
            started = run.get("started_at", "")
            if started:
                try:
                    dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                    date_str = dt.strftime("%m-%d %H:%M")
                except Exception:
                    date_str = started[:16]

            table.add_row(
                str(i + 1),
                status_str,
                task,
                run.get("robot", "-"),
                str(run.get("episodes_success", run.get("episodes_target", "-"))),
                score_str,
                model,
                date_str,
                key=str(i),
            )

        summary = (
            f"  [green]✅ completed: {completed}[/]  "
            f"[yellow]⚠️  partial: {partial}[/]  "
            f"[red]❌ failed: {failed}[/]"
        )
        self.query_one("#history-summary", Static).update(f"\n{summary}\n")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        runs = load_runs()
        try:
            idx = int(str(event.row_key.value))
            if 0 <= idx < len(runs):
                from .detail import DetailScreen
                self.app.push_screen(DetailScreen(runs[idx]))
        except (ValueError, IndexError):
            pass

    def action_back(self) -> None:
        self.app.pop_screen()
