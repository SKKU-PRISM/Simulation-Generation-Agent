"""Manage run history index at outputs/.run_index.json."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INDEX_PATH = "outputs/.run_index.json"


def _resolve_index_path() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / DEFAULT_INDEX_PATH


def load_runs(index_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(index_path) if index_path else _resolve_index_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("runs", [])
    except (json.JSONDecodeError, KeyError):
        return []


def save_runs(runs: list[dict[str, Any]], index_path: str | Path | None = None) -> None:
    path = Path(index_path) if index_path else _resolve_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"runs": runs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def append_run(run: dict[str, Any], index_path: str | Path | None = None) -> None:
    runs = load_runs(index_path)
    runs.insert(0, run)  # newest first
    save_runs(runs, index_path)


def build_run_entry(
    result: dict[str, Any],
    task_desc: str,
    robot: str,
    model: str,
    provider: str,
    mode: str,
    episodes_target: int,
) -> dict[str, Any]:
    """Build a run index entry from pipeline result dict."""
    tasks = result.get("tasks", [])
    first_task = tasks[0] if tasks else {}
    steps = first_task.get("steps", {})

    dc = steps.get("data_collection", {})
    il = steps.get("yaml_to_isaaclab", {})

    eval_score = il.get("eval_score")
    eval_breakdown = {}
    code_score = il.get("last_code_score", {})
    if code_score:
        eval_breakdown = {
            "SF": code_score.get("scene_fidelity", 0),
            "MDP": code_score.get("mdp_correctness", 0),
            "TA": code_score.get("task_alignment", 0),
            "RV": code_score.get("runtime_validity", 0),
        }

    return {
        "id": Path(result.get("work_dir", "")).name,
        "task": task_desc,
        "robot": robot,
        "model": model,
        "provider": provider,
        "mode": mode,
        "episodes_target": episodes_target,
        "episodes_success": dc.get("success_episodes", 0),
        "status": result.get("status", "unknown"),
        "started_at": result.get("started_at", ""),
        "finished_at": result.get("finished_at", ""),
        "duration_s": _calc_duration(result.get("started_at"), result.get("finished_at")),
        "cost": result.get("estimated_cost", ""),
        "tokens": _count_tokens(result),
        "api_calls": _count_api_calls(result),
        "eval_score": eval_score,
        "eval_breakdown": eval_breakdown,
        "dataset_path": dc.get("output_dir", ""),
        "work_dir": result.get("work_dir", ""),
    }


def _calc_duration(started: str | None, finished: str | None) -> int:
    if not started or not finished:
        return 0
    try:
        fmt = "%Y-%m-%dT%H:%M:%S"
        s = datetime.fromisoformat(started.replace("Z", "+00:00"))
        f = datetime.fromisoformat(finished.replace("Z", "+00:00"))
        return int((f - s).total_seconds())
    except Exception:
        return 0


def _count_tokens(result: dict) -> int:
    report = result.get("token_usage_report", "")
    if isinstance(report, str) and "tokens" in report.lower():
        import re
        m = re.search(r"Total:\s*([\d,]+)\s*tokens", report)
        if m:
            return int(m.group(1).replace(",", ""))
    return 0


def _count_api_calls(result: dict) -> int:
    report = result.get("token_usage_report", "")
    if isinstance(report, str):
        import re
        m = re.search(r"(\d+)\s*API\s*calls", report)
        if m:
            return int(m.group(1))
    return 0


def import_existing_runs(outputs_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Scan outputs/challenge_run_*/ directories and import into run index."""
    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = Path(outputs_dir) if outputs_dir else project_root / "outputs"

    existing = load_runs()
    existing_ids = {r.get("id") for r in existing}

    new_runs = []
    for run_dir in sorted(out_dir.glob("challenge_run_*"), reverse=True):
        run_id = run_dir.name
        if run_id in existing_ids:
            continue

        # Try to find results
        result_files = list(run_dir.glob("*/step2_isaaclab.log"))
        if not result_files:
            continue

        # Try to read output.json if available
        for output_json in [project_root / "results" / "output.json"]:
            if output_json.exists():
                try:
                    data = json.loads(output_json.read_text())
                    if data.get("work_dir", "").endswith(run_id):
                        tasks = data.get("tasks", [])
                        if tasks:
                            entry = {
                                "id": run_id,
                                "task": tasks[0].get("description", "unknown"),
                                "robot": "franka",
                                "model": "gpt-5",
                                "provider": "openai",
                                "mode": "full",
                                "status": data.get("status", "unknown"),
                                "started_at": data.get("started_at", ""),
                                "finished_at": data.get("finished_at", ""),
                                "work_dir": str(run_dir),
                            }
                            new_runs.append(entry)
                except Exception:
                    pass

    if new_runs:
        all_runs = new_runs + existing
        save_runs(all_runs)

    return load_runs()
