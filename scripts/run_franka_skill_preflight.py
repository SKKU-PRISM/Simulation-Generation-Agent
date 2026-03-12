#!/usr/bin/env python3
"""Run skill-usage preflight checks for the Franka task suite."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.llm_client import AzureOpenAIClient
from src.common.task_docs import load_task_document
from src.data_collection.cap_generator import (
    SimCaPGenerator,
    build_task_skill_preflight,
    extract_skill_calls_from_code,
    translate_scene_state,
)
from src.data_collection.config import load_robot_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local_docs" / "adc"
FRANKA_TASK_ROOT = REPO_ROOT / "tasks" / "franka"
ISAACLAB_OUTPUT_ROOT = REPO_ROOT / "outputs" / "isaaclab"
DATA_COLLECTION_OUTPUT_ROOT = REPO_ROOT / "outputs" / "data_collection"

FAMILY_SMOKE_TASKS = {
    "tabletop_transfer": FRANKA_TASK_ROOT / "pick_place" / "franka_pick_place.yaml",
    "container_transfer": FRANKA_TASK_ROOT / "pick_place" / "franka_pick_place_gears.yaml",
    "container_stack": FRANKA_TASK_ROOT / "stack" / "franka_stack_tray.yaml",
    "stack": FRANKA_TASK_ROOT / "stack" / "franka_stack.yaml",
    "lift_hold": FRANKA_TASK_ROOT / "lift" / "franka_lift.yaml",
    "support_surface_transfer": FRANKA_TASK_ROOT / "pick_place" / "franka_pick_place_drawer.yaml",
    "articulated_pull": FRANKA_TASK_ROOT / "cabinet" / "franka_cabinet.yaml",
    "upright_placement": FRANKA_TASK_ROOT / "assembly" / "franka_lift_peg_upright.yaml",
    "axial_insertion": FRANKA_TASK_ROOT / "peg_insert" / "franka_peg_insert.yaml",
    "slot_fit": FRANKA_TASK_ROOT / "assembly" / "franka_assembling_kits.yaml",
}


def _discover_tasks(tasks_root: Path) -> list[Path]:
    return sorted(tasks_root.rglob("*.yaml"))


def _build_scene_state_from_assets(task_doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    scene_state: dict[str, dict[str, Any]] = {}
    for asset in task_doc.get("assets", []):
        name = asset.get("name")
        position = asset.get("position")
        if not isinstance(name, str):
            continue
        if asset.get("type") not in {"rigid", "primitive", "static"}:
            continue
        if not isinstance(position, (list, tuple)) or len(position) != 3:
            continue
        rotation = asset.get("rotation")
        if not (isinstance(rotation, (list, tuple)) and len(rotation) == 4):
            rotation = [1.0, 0.0, 0.0, 0.0]
        scene_state[name] = {
            "position": [float(position[0]), float(position[1]), float(position[2])],
            "quaternion": [float(v) for v in rotation],
        }
    return scene_state


def _slugify_task_name(task_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", task_name.lower())


def _resolve_latest_env_dir(task_name: str) -> Path | None:
    slug = _slugify_task_name(task_name)
    candidates = sorted(
        [path for path in ISAACLAB_OUTPUT_ROOT.glob(f"{slug}_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _resolve_fresh_data_collection_output(task_name: str, started_at: float) -> Path | None:
    prefix = f"{task_name}_"
    candidates = sorted(
        [
            path
            for path in DATA_COLLECTION_OUTPUT_ROOT.glob(f"{prefix}*")
            if path.is_dir() and path.stat().st_mtime >= started_at - 1.0
        ],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _run_smoke(task_path: Path, timeout_s: int) -> dict[str, Any]:
    task_doc = load_task_document(task_path)
    task_name = str(task_doc.get("task", {}).get("name", task_path.stem))
    env_dir = _resolve_latest_env_dir(task_name)
    if env_dir is None:
        return {
            "task": task_name,
            "task_path": str(task_path),
            "status": "missing_env_dir",
            "output_dir": None,
            "target_met": None,
            "geometry_target_met": None,
            "overall_target_met": None,
            "used_skill_calls": (),
        }

    cmd = [
        sys.executable,
        "scripts/run_data_collection.py",
        str(task_path),
        "--env-dir",
        str(env_dir),
        "--episodes",
        "1",
        "--target-success",
        "1",
        "--max-attempts",
        "1",
        "--no-vlm-judge",
    ]
    started_at = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
        status = "ok" if returncode == 0 else "run_failed"
    except subprocess.TimeoutExpired as exc:
        return {
            "task": task_name,
            "task_path": str(task_path),
            "status": "timeout",
            "output_dir": None,
            "target_met": None,
            "geometry_target_met": None,
            "overall_target_met": None,
            "used_skill_calls": (),
            "stdout_tail": (exc.stdout or "")[-2000:],
            "stderr_tail": (exc.stderr or "")[-2000:],
        }

    output_dir = _resolve_fresh_data_collection_output(task_name, started_at)
    results: dict[str, Any] = {}
    if output_dir is not None:
        results_path = output_dir / "collection_results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text())

    used_skill_calls: tuple[str, ...] = ()
    if output_dir is not None:
        generated_paths = sorted(output_dir.glob("cap_runs/episode_*/generated_code.py"))
        if generated_paths:
            used_skill_calls = extract_skill_calls_from_code(generated_paths[0].read_text())

    return {
        "task": task_name,
        "task_path": str(task_path),
        "status": status,
        "returncode": returncode,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "target_met": results.get("target_met"),
        "geometry_target_met": results.get("geometry_target_met"),
        "overall_target_met": results.get("overall_target_met"),
        "used_skill_calls": used_skill_calls,
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-2000:],
    }


def _render_markdown_report(
    rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    output_dir: Path,
) -> str:
    lines = [
        "# Franka Skill Preflight",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output dir: `{output_dir}`",
        f"- Task count: {len(rows)}",
        "",
        "## Task Matrix",
        "",
        "| Task | Family | Primary skill | Secondary skills | Used skills | Match |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        secondary = ", ".join(row["secondary_skills"]) or "-"
        used = ", ".join(row["used_skill_calls"]) or "-"
        lines.append(
            f"| {row['task_name']} | `{row['family']}` | "
            f"`{row['primary_skill'] or '-'}` | {secondary} | {used} | {row['codegen_match']} |"
        )

    if smoke_rows:
        lines.extend(["", "## Family Smoke", "", "| Task | Status | Geometry | Output | Used skills |", "| --- | --- | --- | --- | --- |"])
        for row in smoke_rows:
            used = ", ".join(row["used_skill_calls"]) or "-"
            geometry = row["geometry_target_met"]
            lines.append(
                f"| {row['task']} | {row['status']} | {geometry} | "
                f"`{row['output_dir'] or '-'}` | {used} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=FRANKA_TASK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-codegen", action="store_true")
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--smoke-timeout", type=int, default=900)
    args = parser.parse_args()

    tasks = _discover_tasks(args.tasks_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"franka_skill_preflight_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_cfg = load_robot_config("franka")
    llm = None if args.skip_codegen else AzureOpenAIClient()
    generator = None if llm is None else SimCaPGenerator(llm, robot_cfg)

    rows: list[dict[str, Any]] = []
    for task_path in tasks:
        task_doc = load_task_document(task_path)
        preflight = build_task_skill_preflight(task_doc)
        scene_state = _build_scene_state_from_assets(task_doc)
        translated = translate_scene_state(task_doc, scene_state)
        row: dict[str, Any] = {
            "task_path": str(task_path),
            "task_name": preflight.task_name,
            "family": preflight.family,
            "supported": preflight.supported,
            "reason": preflight.reason,
            "primary_skill": preflight.primary_skill,
            "secondary_skills": list(preflight.secondary_skills),
            "required_relations": list(preflight.required_relations),
            "required_targets": list(preflight.required_targets),
            "translated_targets": sorted(translated.keys()),
            "codegen_match": "not_run",
            "used_skill_calls": [],
            "codegen_error": None,
        }
        if generator is not None:
            try:
                result = generator.generate_code(task_doc, scene_state)
                used_skill_calls = list(extract_skill_calls_from_code(result.generated_code))
                row["used_skill_calls"] = used_skill_calls
                if preflight.primary_skill is None:
                    row["codegen_match"] = "n/a"
                else:
                    row["codegen_match"] = "yes" if preflight.primary_skill in used_skill_calls else "no"
            except Exception as exc:  # noqa: BLE001
                row["codegen_match"] = "error"
                row["codegen_error"] = str(exc)
        rows.append(row)

    smoke_rows: list[dict[str, Any]] = []
    if args.run_smoke:
        smoke_paths = []
        for family in sorted({row["family"] for row in rows if row["supported"]}):
            task_path = FAMILY_SMOKE_TASKS.get(family)
            if task_path is not None and task_path.exists():
                smoke_paths.append(task_path)
        seen: set[Path] = set()
        for task_path in smoke_paths:
            if task_path in seen:
                continue
            seen.add(task_path)
            smoke_rows.append(_run_smoke(task_path, timeout_s=args.smoke_timeout))

    summary = {"tasks": rows, "smoke": smoke_rows}
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    report_text = _render_markdown_report(rows, smoke_rows, output_dir)
    report_path = output_dir / "report.md"
    report_path.write_text(report_text)

    print(f"Saved summary: {summary_path}")
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
