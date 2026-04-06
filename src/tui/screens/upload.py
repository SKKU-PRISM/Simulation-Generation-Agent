"""Upload screen — select completed run and upload dataset to HuggingFace."""

from __future__ import annotations

import os
from datetime import datetime
from threading import Thread

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Static,
)

from ..run_index import load_runs


class UploadSelectScreen(Screen):
    """Step 1: Select a completed run to upload."""

    BINDINGS = [
        ("escape", "back", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("\n  [bold cyan]📤 Upload Dataset to HuggingFace[/]\n")
        yield Static("  [dim]Select a completed run to upload its dataset.[/]\n")
        yield DataTable(id="upload-table", cursor_type="row")
        yield Static("", id="upload-hint")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#upload-table", DataTable)
        table.add_columns("#", "Task", "Robot", "Episodes", "Score", "Date")

        runs = load_runs()
        self._completed_runs = [r for r in runs if r.get("status") == "completed"]

        if not self._completed_runs:
            self.query_one("#upload-hint", Static).update(
                "\n  [yellow]No completed runs available for upload.[/]\n"
            )
            return

        for i, run in enumerate(self._completed_runs):
            task = run.get("task", "unknown")
            if len(task) > 30:
                task = task[:27] + "..."

            score = run.get("eval_score")
            score_str = str(score) if score is not None else "-"

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
                task,
                run.get("robot", "-"),
                str(run.get("episodes_success", "-")),
                score_str,
                date_str,
                key=str(i),
            )

        self.query_one("#upload-hint", Static).update(
            f"\n  [dim]{len(self._completed_runs)} completed runs available. "
            f"Select one and press Enter.[/]\n"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            idx = int(str(event.row_key.value))
            if 0 <= idx < len(self._completed_runs):
                self.app.push_screen(
                    UploadConfigScreen(self._completed_runs[idx])
                )
        except (ValueError, IndexError):
            pass

    def action_back(self) -> None:
        self.app.pop_screen()


class UploadConfigScreen(Screen):
    """Step 2: Configure upload settings and execute."""

    BINDINGS = [
        ("escape", "back", "Back"),
    ]

    def __init__(self, run: dict) -> None:
        self.run = run
        self._uploading = False
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="upload-config-scroll"):
            yield Static("\n  [bold cyan]📤 Upload Configuration[/]\n")

            # Selected run info
            task = self.run.get("task", "unknown")
            robot = self.run.get("robot", "-")
            score = self.run.get("eval_score", "-")
            yield Static(
                f"  [bold]Selected:[/] {task}\n"
                f"  [bold]Robot:[/] {robot}  │  [bold]Score:[/] {score}\n"
            )

            # HuggingFace settings
            yield Static("  [bold]── HuggingFace Settings ──────────────[/]")

            yield Label("  HuggingFace Username / Organization")
            yield Input(
                value=os.environ.get("HF_USERNAME", ""),
                id="hf-namespace-input",
                placeholder="your-username or your-org",
            )

            yield Label("  Dataset Name")
            task_slug = self.run.get("task", "dataset").replace(" ", "-").lower()[:30]
            robot_name = self.run.get("robot", "franka")
            yield Input(
                value=f"{robot_name}-{task_slug}",
                id="hf-dataset-input",
                placeholder="franka-stack-sim",
            )

            yield Static("")
            yield Checkbox("Private Dataset", value=True, id="hf-private-toggle")

            yield Static("\n  [bold]── HuggingFace Token ─────────────────[/]")
            yield Static("  [dim]Token from Settings (F1) is used. Set it there if not configured.[/]")

            hf_set = bool(os.environ.get("HF_TOKEN"))
            if hf_set:
                yield Static("  [green]✓ HF_TOKEN is configured[/]")
            else:
                yield Static("  [red]✗ HF_TOKEN not set — configure in Settings (F1) first[/]")

            yield Static("")
            yield Button(
                "Upload" if hf_set else "Upload (token required)",
                variant="primary" if hf_set else "warning",
                id="upload-btn",
                disabled=not hf_set,
            )
            yield Static("")

            yield RichLog(id="upload-log", highlight=True, markup=True, max_lines=50)

        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "upload-btn" and not self._uploading:
            self._uploading = True
            event.button.disabled = True
            event.button.label = "Uploading..."
            thread = Thread(target=self._do_upload, daemon=True)
            thread.start()

    def _do_upload(self) -> None:
        log = self.query_one("#upload-log", RichLog)

        namespace = self.query_one("#hf-namespace-input", Input).value.strip()
        dataset_name = self.query_one("#hf-dataset-input", Input).value.strip()
        private = self.query_one("#hf-private-toggle", Checkbox).value

        if not namespace or not dataset_name:
            self.app.call_from_thread(
                log.write, "[red]Error: Username and dataset name are required.[/]"
            )
            self._uploading = False
            return

        repo_id = f"{namespace}/{dataset_name}"
        dataset_path = self.run.get("dataset_path", "")

        self.app.call_from_thread(
            log.write, f"[cyan]Preparing upload to {repo_id}...[/]"
        )

        # Ensure path points to raw_dataset/ subdirectory
        if dataset_path:
            from pathlib import Path
            dp = Path(dataset_path)
            if dp.is_dir() and (dp / "raw_dataset").is_dir():
                dataset_path = str(dp / "raw_dataset")
            elif dp.is_dir() and not (dp / "metadata.json").exists():
                # Search for raw_dataset inside
                candidates = list(dp.glob("**/raw_dataset/metadata.json"))
                if candidates:
                    dataset_path = str(candidates[0].parent)

        if not dataset_path:
            self.app.call_from_thread(
                log.write, "[yellow]No raw_dataset path found. Searching...[/]"
            )
            # Try to find from work_dir
            import glob
            from pathlib import Path
            work_dir = self.run.get("work_dir", "")
            if work_dir:
                candidates = sorted(
                    Path(work_dir).parent.glob("data_collection/*/raw_dataset"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ) if Path(work_dir).parent.exists() else []
                if candidates:
                    dataset_path = str(candidates[0])

        if not dataset_path:
            self.app.call_from_thread(
                log.write, "[red]Error: Could not find dataset path for this run.[/]"
            )
            self._uploading = False
            return

        self.app.call_from_thread(
            log.write, f"[dim]Dataset: {dataset_path}[/]"
        )

        # Step 1: Convert to LeRobot
        self.app.call_from_thread(
            log.write, "\n[cyan]Step 1/3: Converting to LeRobot format...[/]"
        )
        try:
            from src.agent.data_collection.lerobot_tools import (
                convert_raw_dataset_to_lerobot,
            )
            local_repo_id = f"local/{dataset_name}"
            lerobot_path = convert_raw_dataset_to_lerobot(
                dataset_path, repo_id=local_repo_id
            )
            self.app.call_from_thread(
                log.write, f"[green]✓ Converted: {lerobot_path}[/]"
            )
        except Exception as e:
            self.app.call_from_thread(
                log.write, f"[red]Error converting: {e}[/]"
            )
            self._uploading = False
            return

        # Step 2: Validate
        self.app.call_from_thread(
            log.write, "\n[cyan]Step 2/3: Validating dataset...[/]"
        )
        try:
            from src.agent.data_collection.lerobot_tools import (
                check_lerobot_dataset,
            )
            report = check_lerobot_dataset(lerobot_path, repo_id=local_repo_id)
            if report.get("pass"):
                frames = report.get("num_frames", 0)
                self.app.call_from_thread(
                    log.write,
                    f"[green]✓ Validation passed ({frames:,} frames)[/]",
                )
            else:
                missing = report.get("missing_features", [])
                self.app.call_from_thread(
                    log.write,
                    f"[red]Validation failed. Missing: {missing}[/]",
                )
                self._uploading = False
                return
        except Exception as e:
            self.app.call_from_thread(
                log.write, f"[red]Validation error: {e}[/]"
            )
            self._uploading = False
            return

        # Step 3: Upload
        self.app.call_from_thread(
            log.write,
            f"\n[cyan]Step 3/3: Uploading to HuggingFace ({repo_id})...[/]",
        )
        try:
            from src.agent.data_collection.lerobot_tools import (
                publish_lerobot_dataset,
            )
            result = publish_lerobot_dataset(
                lerobot_path,
                repo_id=repo_id,
                private=private,
                local_repo_id=local_repo_id,
            )
            repo_url = result.get("repo_url", repo_id)
            self.app.call_from_thread(
                log.write,
                f"\n[bold green]✅ Upload complete![/]\n"
                f"  [bold]URL:[/] {repo_url}\n"
                f"  [bold]Private:[/] {private}\n",
            )
        except Exception as e:
            self.app.call_from_thread(
                log.write, f"[red]Upload failed: {e}[/]"
            )

        self._uploading = False
        self.app.call_from_thread(
            log.write, "\n[dim]Press Esc to go back.[/]"
        )

    def action_back(self) -> None:
        if not self._uploading:
            self.app.pop_screen()
