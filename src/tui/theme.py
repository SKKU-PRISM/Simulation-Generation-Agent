"""Color theme constants for the RAPIDS TUI."""

# Status colors
STATUS_COMPLETED = "green"
STATUS_PARTIAL = "yellow"
STATUS_FAILED = "red"
STATUS_RUNNING = "cyan"
STATUS_PENDING = "dim"

# Score bar colors
SCORE_HIGH = "green"       # 90+
SCORE_MEDIUM = "yellow"    # 60-89
SCORE_LOW = "red"          # <60

# UI elements
HEADER_STYLE = "bold cyan"
SECTION_STYLE = "bold white"
LABEL_STYLE = "dim"
VALUE_STYLE = "bold"
PROMPT_STYLE = "bold white"
FOOTER_STYLE = "dim white"

# App name
APP_TITLE = "RAPIDS — Simulation Generation Agent"


def score_color(score: int, max_score: int = 100) -> str:
    pct = (score / max_score * 100) if max_score > 0 else 0
    if pct >= 90:
        return SCORE_HIGH
    elif pct >= 60:
        return SCORE_MEDIUM
    return SCORE_LOW


def status_color(status: str) -> str:
    return {
        "completed": STATUS_COMPLETED,
        "partial": STATUS_PARTIAL,
        "failed": STATUS_FAILED,
        "running": STATUS_RUNNING,
    }.get(status, STATUS_PENDING)


def status_icon(status: str) -> str:
    return {
        "completed": "[green]✅[/]",
        "partial": "[yellow]⚠️[/]",
        "failed": "[red]❌[/]",
        "running": "[cyan]⏳[/]",
    }.get(status, "○")


def score_bar(score: int, max_score: int, width: int = 20) -> str:
    pct = score / max_score if max_score > 0 else 0
    filled = int(pct * width)
    empty = width - filled
    color = score_color(score, max_score)
    return f"[{color}]{'█' * filled}[/]{'░' * empty}"
