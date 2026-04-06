"""Detail screen — shows full run details with score bars."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from ..theme import score_bar, score_color, status_icon


def _fmt_time(seconds: int) -> str:
    if seconds <= 0:
        return "-"
    m, s = divmod(seconds, 60)
    return f"{m}m {s}s" if m else f"{s}s"


class DetailScreen(Screen):
    BINDINGS = [
        ("escape", "back", "Back"),
    ]

    def __init__(self, run: dict) -> None:
        self.run = run
        super().__init__()

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="detail-scroll"):
            yield Static(self._build_detail())
        yield Footer()

    def _build_detail(self) -> str:
        r = self.run
        status = r.get("status", "unknown")
        icon = status_icon(status)

        lines = [
            "",
            f"  [bold cyan]🔍 Run Detail[/]",
            "",
            f"  [bold]Task:[/]     {r.get('task', 'unknown')}",
            f"  [bold]Robot:[/]    {r.get('robot', '-')}  │  "
            f"[bold]Model:[/] {r.get('model', '-')}  │  "
            f"[bold]Provider:[/] {r.get('provider', '-')}",
            f"  [bold]Status:[/]   {icon} {status}",
        ]

        # Date
        started = r.get("started_at", "")
        if started:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
                lines.append(f"  [bold]Date:[/]     {dt.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                lines.append(f"  [bold]Date:[/]     {started}")

        # Duration, cost, tokens
        duration = r.get("duration_s", 0)
        cost = r.get("cost", "")
        tokens = r.get("tokens", 0)
        api_calls = r.get("api_calls", 0)
        parts = []
        if duration:
            parts.append(f"Duration: {_fmt_time(duration)}")
        if cost:
            parts.append(f"Cost: ~{cost}")
        if tokens:
            parts.append(f"Tokens: {tokens:,}")
        if api_calls:
            parts.append(f"({api_calls} calls)")
        if parts:
            lines.append(f"  {' │ '.join(parts)}")

        # Environment Score
        breakdown = r.get("eval_breakdown", {})
        eval_score = r.get("eval_score")
        if eval_score is not None or breakdown:
            lines.append("")
            lines.append(f"  [bold]── Environment Score ──────────────────────────[/]")

            categories = [
                ("Scene Fidelity", "SF", 40),
                ("MDP Correctness", "MDP", 20),
                ("Task Alignment", "TA", 15),
                ("Runtime Validity", "RV", 25),
            ]
            total = 0
            max_total = 100
            for label, key, max_val in categories:
                val = breakdown.get(key, 0)
                total += val
                pct = int(val / max_val * 100) if max_val > 0 else 0
                color = score_color(val, max_val)
                bar = score_bar(val, max_val)
                lines.append(
                    f"  {label:20s} {val:3d}/{max_val}  {bar}  [{color}]{pct:3d}%[/]"
                )

            if eval_score is not None:
                lines.append(f"  {'─' * 50}")
                color = score_color(eval_score, max_total)
                lines.append(
                    f"  [bold]{'Total':20s} {eval_score:3d}/{max_total}[/]"
                )

        # VLM Image Score (SceneVerifier)
        image_score = r.get("image_score")
        if image_score:
            lines.append("")
            lines.append(f"  [bold]── VLM Scene Verification ────────────────────[/]")
            front = image_score.get("front", 0)
            top = image_score.get("top", 0)
            front_bar = score_bar(front, 100)
            top_bar = score_bar(top, 100)
            front_color = score_color(front, 100)
            top_color = score_color(top, 100)
            lines.append(f"  {'Front View':20s} {front:3d}/100  {front_bar}  [{front_color}]{front:3d}%[/]")
            lines.append(f"  {'Top View':20s} {top:3d}/100  {top_bar}  [{top_color}]{top:3d}%[/]")

        # Data Collection Results
        geo_ok = r.get("geometry_successful", 0)
        vlm_ok = r.get("vlm_successful", 0)
        target_met = r.get("target_met")
        if geo_ok or vlm_ok:
            lines.append("")
            lines.append(f"  [bold]── Data Collection Judge ─────────────────────[/]")
            lines.append(f"  Geometry pass:  {geo_ok} episodes")
            lines.append(f"  VLM pass:       {vlm_ok} episodes")
            if target_met is not None:
                target_icon = "[green]✅[/]" if target_met else "[yellow]⚠️[/]"
                lines.append(f"  Target met:     {target_icon} {target_met}")

        # Dataset
        dataset_path = r.get("dataset_path", "")
        ep_target = r.get("episodes_target", 0)
        ep_success = r.get("episodes_success", 0)
        if dataset_path or ep_target:
            lines.append("")
            lines.append(f"  [bold]── Dataset ───────────────────────────────────[/]")
            if dataset_path:
                lines.append(f"  [bold]Path:[/]     {dataset_path}")
            if ep_target or ep_success:
                lines.append(
                    f"  [bold]Episodes:[/] {ep_success} successful / {ep_target} target"
                )

        # Work dir
        work_dir = r.get("work_dir", "")
        if work_dir:
            lines.append("")
            lines.append(f"  [dim]Logs: {work_dir}[/]")

        lines.append("")
        return "\n".join(lines)

    def action_back(self) -> None:
        self.app.pop_screen()
