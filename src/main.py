#!/usr/bin/env python3
"""Simulation Generation Agent — Main Entry Point.

Runs the full NL → YAML → IsaacLab → DataCollection pipeline
using a JSON input file and writes structured results to a JSON output file.

Usage:
    python src/main.py --input data/input_sample.json --output results/output.json
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console

console = Console(force_terminal=True)

PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)

# Cost per 1K tokens (USD)
COST_PER_1K = {
    "gpt-5-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-5": {"input": 0.003, "output": 0.015},
}

# Estimated stage durations (seconds) for progress hints
STAGE_EST = {
    "Stage 1: NL → YAML": 90,
    "Stage 2: YAML → IsaacLab": 600,
    "Stage 3: Data Collection": 420,
}


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _run_step(
    cmd: list[str], label: str, log_path: Path | None = None, est_seconds: int = 0
) -> tuple[bool, str, float]:
    """Run a subprocess step with spinner. Returns (success, output, elapsed)."""
    start = time.time()
    est_hint = f", est. ~{_fmt_time(est_seconds)}" if est_seconds else ""

    try:
        with console.status(
            f"  [cyan]⏳ {label}[/] ... (elapsed: 0s{est_hint})",
            spinner="dots",
        ):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONUNBUFFERED": "1"},
            )
        elapsed = time.time() - start
        combined = result.stdout + "\n" + result.stderr

        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(combined, encoding="utf-8")

        if result.returncode != 0:
            console.print(f"  [red]❌ {label}[/]  {_fmt_time(elapsed)}")
            return False, combined, elapsed
        console.print(f"  [green]✅ {label}[/]  {_fmt_time(elapsed)}")
        return True, combined, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        console.print(f"  [red]⏰ {label} (timeout)[/]  {_fmt_time(elapsed)}")
        return False, "TIMEOUT", elapsed
    except Exception as e:
        elapsed = time.time() - start
        console.print(f"  [red]💥 {label} ({e})[/]  {_fmt_time(elapsed)}")
        return False, str(e), elapsed


def _sanitize_task_name(description: str) -> str:
    words = re.sub(r"[^a-zA-Z0-9\s]", "", description).split()
    return "".join(w.capitalize() for w in words[:8]) or "Task"


def _find_latest_result_json() -> Path | None:
    isaaclab_out = PROJECT_ROOT / "outputs" / "isaaclab"
    if not isaaclab_out.exists():
        return None
    results = sorted(isaaclab_out.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return results[0] if results else None


def _estimate_cost(token_file: Path) -> str:
    """Estimate API cost from token usage JSONL."""
    if not token_file.exists():
        return "$0.00"
    total_cost = 0.0
    try:
        for line in token_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            model = entry.get("model", "gpt-5-mini")
            inp = entry.get("input_tokens", 0)
            out = entry.get("output_tokens", 0)
            rates = COST_PER_1K.get(model, COST_PER_1K["gpt-5-mini"])
            total_cost += (inp / 1000) * rates["input"] + (out / 1000) * rates["output"]
    except Exception:
        pass
    return f"${total_cost:.2f}"


def run_task(
    task_desc: str,
    robot: str,
    target_success: int,
    max_attempts: int,
    work_dir: Path,
) -> dict:
    """Run a single task through the 3-step pipeline."""
    task_name = _sanitize_task_name(task_desc)
    task_dir = work_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = task_dir / "task.yaml"

    result = {"name": task_name, "description": task_desc, "steps": {}}

    # Stage 1: NL → YAML
    step1_cmd = [
        PYTHON_BIN,
        "scripts/task_spec_agent/task_spec_agent.py",
        task_desc,
        "--robot", robot,
        "--output", str(yaml_path),
    ]
    ok, _, _ = _run_step(
        step1_cmd, "Stage 1: NL → YAML",
        task_dir / "step1_nl_to_yaml.log",
        est_seconds=STAGE_EST["Stage 1: NL → YAML"],
    )
    result["steps"]["nl_to_yaml"] = {
        "success": ok and yaml_path.exists(),
        "yaml_path": str(yaml_path) if yaml_path.exists() else None,
    }
    if not ok or not yaml_path.exists():
        return result

    # Stage 2: YAML → IsaacLab
    step2_cmd = [PYTHON_BIN, "scripts/run_isaac_lab.py", str(yaml_path)]
    ok, _, _ = _run_step(
        step2_cmd, "Stage 2: YAML → IsaacLab",
        task_dir / "step2_isaaclab.log",
        est_seconds=STAGE_EST["Stage 2: YAML → IsaacLab"],
    )

    env_dir = None
    lab_success = False
    latest_result = _find_latest_result_json()
    if latest_result:
        try:
            rdata = json.loads(latest_result.read_text())
            env_dir = rdata.get("output_dir")
            lab_success = rdata.get("success", False)
        except Exception:
            pass

    result["steps"]["yaml_to_isaaclab"] = {"success": bool(lab_success), "env_dir": env_dir}
    if not lab_success:
        return result

    # Stage 3: Data Collection
    step3_cmd = [
        PYTHON_BIN,
        "scripts/run_data_collection.py",
        str(yaml_path),
        "--env-dir", str(env_dir),
        "--target-success", str(target_success),
        "--max-attempts", str(max_attempts),
    ]
    ok, _, _ = _run_step(
        step3_cmd, f"Stage 3: Data Collection (target: {target_success} episodes)",
        task_dir / "step3_data_collection.log",
        est_seconds=STAGE_EST["Stage 3: Data Collection"],
    )

    dc_result = {"success": False, "success_episodes": 0, "total_episodes": 0}
    dc_out = PROJECT_ROOT / "outputs" / "data_collection"
    if dc_out.exists():
        all_results = sorted(
            dc_out.glob("*/collection_results.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if all_results:
            coll_json = all_results[0]
            try:
                cdata = json.loads(coll_json.read_text())
                dc_result["success_episodes"] = cdata.get("geometry_successful_episodes", 0)
                dc_result["total_episodes"] = cdata.get("total_episodes", 0)
                dc_result["output_dir"] = str(coll_json.parent)
                dc_result["success"] = cdata.get("pipeline_completed", False)
            except Exception:
                pass

    result["steps"]["data_collection"] = dc_result
    return result


def run_pipeline(input_path: Path, output_path: Path) -> dict:
    """Execute the full pipeline from a JSON input spec."""
    input_data = json.loads(input_path.read_text(encoding="utf-8"))
    tasks = input_data.get("tasks", [])
    config = input_data.get("config", {})
    target_success = config.get("target_success", 1)
    max_attempts = config.get("max_attempts", 3)

    if not tasks:
        return {"status": "error", "message": "No tasks specified in input file"}

    pipeline_start = time.time()
    started_at = datetime.now(timezone.utc).isoformat()
    work_dir = PROJECT_ROOT / "outputs" / f"challenge_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Token tracking
    token_file = work_dir / "token_usage.jsonl"
    os.environ["TOKEN_USAGE_FILE"] = str(token_file)
    os.environ["TOKEN_USAGE_LOG"] = "1"

    task_results = []
    for task_spec in tasks:
        desc = task_spec.get("task_description", "")
        robot = task_spec.get("robot", "franka")
        if not desc:
            task_results.append({"name": "unknown", "steps": {}, "error": "empty task_description"})
            continue

        console.print(f"\n[bold]🚀 RAPIDS Pipeline[/] — \"{desc}\"")
        console.print(f"   Robot: {robot} | Target: {target_success} episodes\n")

        tr = run_task(desc, robot, target_success, max_attempts, work_dir)
        task_results.append(tr)

    finished_at = datetime.now(timezone.utc).isoformat()
    total_elapsed = time.time() - pipeline_start

    all_ok = all(
        t.get("steps", {}).get("data_collection", {}).get("success", False)
        for t in task_results
    )

    # Token summary
    token_summary = None
    token_count = 0
    api_calls = 0
    if token_file.exists():
        try:
            from src.agent.common.token_tracker import TokenTracker
            token_summary = TokenTracker.report_from_file(str(token_file))
            for line in token_file.read_text().strip().split("\n"):
                if line.strip():
                    e = json.loads(line)
                    token_count += e.get("input_tokens", 0) + e.get("output_tokens", 0)
                    api_calls += 1
        except Exception:
            pass

    cost_str = _estimate_cost(token_file)

    # Final summary
    status_icon = "✅" if all_ok else "⚠️"
    status_text = "completed" if all_ok else "partial"
    console.print(f"\n{'─' * 54}")
    console.print(f"  📊 Result: {status_icon} {status_text}")
    console.print(f"  ⏱️  Total: {_fmt_time(total_elapsed)}")
    if token_count:
        console.print(f"  🔤 Tokens: {token_count:,} ({api_calls} API calls)")
        console.print(f"  💰 Cost: ~{cost_str}")
    console.print(f"  📄 Output: {output_path}")
    console.print(f"  📁 Logs: {work_dir}/")
    console.print(f"{'─' * 54}\n")

    output = {
        "status": status_text,
        "started_at": started_at,
        "finished_at": finished_at,
        "work_dir": str(work_dir),
        "tasks": task_results,
    }
    if token_summary:
        output["token_usage_report"] = token_summary
    if token_file.exists():
        output["token_usage_file"] = str(token_file)
    if cost_str:
        output["estimated_cost"] = cost_str

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Simulation Generation Agent — Main Entry Point"
    )
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        console.print(f"[red]Error:[/] Input file not found: {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(input_path, output_path)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
