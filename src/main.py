#!/usr/bin/env python3
"""Challenge submission entry point.

Runs the full NL -> YAML -> IsaacLab -> DataCollection pipeline
using a JSON input file and writes structured results to a JSON output file.

Usage:
    python src/main.py --input data/input_sample.json --output results/output.json

Input JSON format:
    {
      "tasks": [
        {"task_description": "Stack the blocks inside the tray", "robot": "franka"}
      ],
      "config": {"target_success": 1, "max_attempts": 3}
    }
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("main")

PYTHON_BIN = os.environ.get("PYTHON_BIN", sys.executable)


def _run_step(cmd: list[str], label: str, log_path: Path | None = None) -> tuple[bool, str]:
    """Run a subprocess step. Returns (success, stdout_text)."""
    logger.info("Step [%s]: %s", label, " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONUNBUFFERED": "1"},
        )
        combined = result.stdout + "\n" + result.stderr
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(combined, encoding="utf-8")
        if result.returncode != 0:
            logger.error("Step [%s] failed (rc=%d):\n%s", label, result.returncode, combined[-2000:])
            return False, combined
        return True, combined
    except subprocess.TimeoutExpired:
        logger.error("Step [%s] timed out", label)
        return False, "TIMEOUT"
    except Exception as e:
        logger.error("Step [%s] exception: %s", label, e)
        return False, str(e)


def _sanitize_task_name(description: str) -> str:
    """Derive a CamelCase task name from NL description."""
    words = re.sub(r"[^a-zA-Z0-9\s]", "", description).split()
    return "".join(w.capitalize() for w in words[:8]) or "Task"


def _find_latest_result_json() -> Path | None:
    """Find the most recently created result.json under outputs/isaaclab/."""
    isaaclab_out = PROJECT_ROOT / "outputs" / "isaaclab"
    if not isaaclab_out.exists():
        return None
    results = sorted(isaaclab_out.glob("*/result.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return results[0] if results else None


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

    result = {
        "name": task_name,
        "description": task_desc,
        "steps": {},
    }

    # Step 1: NL -> YAML
    logger.info("=== Step 1: NL -> YAML for %s ===", task_name)
    step1_cmd = [
        PYTHON_BIN,
        "scripts/task_spec_agent/task_spec_agent.py",
        task_desc,
        "--robot", robot,
        "--output", str(yaml_path),
    ]
    ok, _ = _run_step(step1_cmd, f"{task_name}/nl_to_yaml", task_dir / "step1_nl_to_yaml.log")
    result["steps"]["nl_to_yaml"] = {
        "success": ok and yaml_path.exists(),
        "yaml_path": str(yaml_path) if yaml_path.exists() else None,
    }
    if not ok or not yaml_path.exists():
        logger.error("Step 1 failed for %s", task_name)
        return result

    # Step 2: YAML -> IsaacLab environment
    logger.info("=== Step 2: YAML -> IsaacLab for %s ===", task_name)
    step2_cmd = [
        PYTHON_BIN,
        "scripts/run_isaac_lab.py",
        str(yaml_path),
    ]
    ok, _ = _run_step(step2_cmd, f"{task_name}/yaml_to_isaaclab", task_dir / "step2_isaaclab.log")

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

    result["steps"]["yaml_to_isaaclab"] = {
        "success": bool(lab_success),
        "env_dir": env_dir,
    }
    if not lab_success:
        logger.error("Step 2 failed for %s", task_name)
        return result

    # Step 3: Data Collection
    logger.info("=== Step 3: Data Collection for %s ===", task_name)
    step3_cmd = [
        PYTHON_BIN,
        "scripts/run_data_collection.py",
        str(yaml_path),
        "--env-dir", str(env_dir),
        "--target-success", str(target_success),
        "--max-attempts", str(max_attempts),
    ]
    ok, _ = _run_step(step3_cmd, f"{task_name}/data_collection", task_dir / "step3_data_collection.log")

    dc_result = {"success": ok, "success_episodes": 0, "total_episodes": 0}

    # Parse collection results
    dc_out = PROJECT_ROOT / "outputs" / "data_collection"
    if dc_out.exists():
        candidates = sorted(
            [d for d in dc_out.iterdir() if d.is_dir() and task_name in d.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            coll_json = candidates[0] / "collection_results.json"
            if coll_json.exists():
                try:
                    cdata = json.loads(coll_json.read_text())
                    dc_result["success_episodes"] = cdata.get("geometry_successful_episodes", 0)
                    dc_result["total_episodes"] = cdata.get("total_episodes", 0)
                    dc_result["success"] = dc_result["success_episodes"] > 0
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

    started_at = datetime.now(timezone.utc).isoformat()
    work_dir = PROJECT_ROOT / "outputs" / f"challenge_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    work_dir.mkdir(parents=True, exist_ok=True)

    task_results = []
    for task_spec in tasks:
        desc = task_spec.get("task_description", "")
        robot = task_spec.get("robot", "franka")
        if not desc:
            task_results.append({"name": "unknown", "steps": {}, "error": "empty task_description"})
            continue
        tr = run_task(desc, robot, target_success, max_attempts, work_dir)
        task_results.append(tr)

    finished_at = datetime.now(timezone.utc).isoformat()

    all_ok = all(
        t.get("steps", {}).get("data_collection", {}).get("success", False)
        for t in task_results
    )

    return {
        "status": "completed" if all_ok else "partial",
        "started_at": started_at,
        "finished_at": finished_at,
        "work_dir": str(work_dir),
        "tasks": task_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Simulation Generation Agent — Challenge Submission Entry Point"
    )
    parser.add_argument("--input", required=True, help="Input JSON file path")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Pipeline start — input: %s, output: %s", input_path, output_path)
    result = run_pipeline(input_path, output_path)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Pipeline done — status: %s, output: %s", result["status"], output_path)


if __name__ == "__main__":
    main()
