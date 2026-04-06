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

        # Find task subdirectories (each has step logs)
        task_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not task_dirs:
            continue

        task_dir = task_dirs[0]  # first (usually only) task
        task_name = task_dir.name

        # Try to extract info from logs
        entry: dict[str, Any] = {
            "id": run_id,
            "task": task_name,
            "robot": "franka",
            "model": "gpt-5",
            "provider": "openai",
            "mode": "full",
            "status": "unknown",
            "started_at": "",
            "finished_at": "",
            "work_dir": str(run_dir),
        }

        # Check step logs to determine status
        step1_ok = (task_dir / "step1_nl_to_yaml.log").exists()
        step2_log = task_dir / "step2_isaaclab.log"
        step3_log = task_dir / "step3_data_collection.log"

        if step3_log.exists():
            entry["status"] = "completed"
        elif step2_log.exists():
            entry["status"] = "partial"
        elif step1_ok:
            entry["status"] = "partial"

        # Try to get eval score from isaaclab result.json
        isaaclab_dirs = sorted(
            (out_dir / "isaaclab").glob("*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ) if (out_dir / "isaaclab").exists() else []

        for idir in isaaclab_dirs:
            result_json = idir / "result.json"
            if result_json.exists():
                try:
                    rdata = json.loads(result_json.read_text())
                    sv = rdata.get("scene_verification", {})
                    ce = sv.get("code_evaluation", {})
                    if ce:
                        entry["eval_score"] = ce.get("total_score")
                        entry["eval_breakdown"] = {
                            "SF": ce.get("scene_fidelity", {}).get("score", 0),
                            "MDP": ce.get("mdp_correctness", {}).get("score", 0),
                            "TA": ce.get("task_alignment", {}).get("score", 0),
                            "RV": ce.get("runtime_validity", {}).get("score", 0),
                        }
                        break
                except Exception:
                    pass

        # Extract timestamp from dir name (challenge_run_YYYYMMDD_HHMMSS)
        import re
        m = re.search(r"(\d{8})_(\d{6})$", run_id)
        if m:
            try:
                dt_str = f"{m.group(1)}T{m.group(2)[:2]}:{m.group(2)[2:4]}:{m.group(2)[4:6]}"
                entry["started_at"] = dt_str
            except Exception:
                pass

        new_runs.append(entry)

    if new_runs:
        all_runs = new_runs + existing
        save_runs(all_runs)

    return load_runs()
