"""Pipeline execution wrapper for TUI — runs src/main.py as subprocess."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def run_pipeline(
    task_desc: str,
    robot: str = "franka",
    episodes: int = 1,
    max_attempts: int = 3,
    model: str = "gpt-5",
    provider: str = "openai",
    output_dir: str = "outputs",
    on_line: Callable[[str], None] | None = None,
) -> dict:
    """Run the full pipeline and stream output lines via callback.

    Returns the parsed results/output.json dict.
    """
    python_bin = os.environ.get("PYTHON_BIN", sys.executable)
    output_json = PROJECT_ROOT / "results" / "output.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Create temp input JSON
    input_data = {
        "tasks": [{"task_description": task_desc, "robot": robot}],
        "config": {"target_success": episodes, "max_attempts": max_attempts},
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="simgen_tui_", delete=False
    ) as f:
        json.dump(input_data, f)
        input_path = f.name

    env = {
        **os.environ,
        "PYTHONPATH": str(PROJECT_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    if model:
        env["OPENAI_MODEL"] = model

    cmd = [
        python_bin,
        str(PROJECT_ROOT / "src" / "main.py"),
        "--input", input_path,
        "--output", str(output_json),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
            env=env,
        )

        for line in iter(process.stdout.readline, ""):
            stripped = line.rstrip()
            if stripped and on_line:
                on_line(stripped)

        process.wait()

        if output_json.exists():
            return json.loads(output_json.read_text(encoding="utf-8"))
        return {"status": "error", "message": f"exit code {process.returncode}"}

    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass
