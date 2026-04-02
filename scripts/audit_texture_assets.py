#!/usr/bin/env python3
"""Audit texture-sensitive Isaac Sim assets used by task YAMLs and latest collection runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.data_collection.pipeline import build_task_texture_audit


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_latest_run(outputs_root: Path, task_name: str) -> Path | None:
    runs = sorted(outputs_root.glob(f"{task_name}_*"))
    return runs[-1] if runs else None


def _infer_run_loader_policy(run_dir: Path | None, record: dict[str, object]) -> str:
    if run_dir is None:
        return "no_run"

    collect_data = run_dir / "collect_data.py"
    if not collect_data.exists():
        return "no_collect_data"

    text = collect_data.read_text(encoding="utf-8", errors="ignore")
    suffix = str(record.get("path_suffix", ""))
    recommended = str(record.get("loader_policy", ""))

    if recommended == "preserve_visual_usd_with_injected_physics":
        if "replaced rootless USD with primitive proxy" in text:
            return "proxy_replaced"
        if "preserved textured USD visual and injected rigid/collision props" in text:
            return "preserve_visual_usd_with_injected_physics"
        if suffix and suffix in text:
            return "direct_usd_unclassified"

    if recommended == "audit_axis_aligned_variant":
        if "replaced rootless USD with primitive proxy" in text and suffix and suffix in text:
            return "proxy_replaced"
        return "direct_usd_audit_candidate"

    return "unknown"


def _collect_records(tasks_root: Path, outputs_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for yaml_path in sorted(tasks_root.rglob("*.yaml")):
        doc = _load_yaml(yaml_path)
        task_records = build_task_texture_audit(doc)
        if not task_records:
            continue
        task_name = str(doc.get("task", {}).get("name", "unknown"))
        latest_run = _find_latest_run(outputs_root, task_name)
        robot = yaml_path.parts[-3] if len(yaml_path.parts) >= 3 else "unknown"
        category = yaml_path.parts[-2] if len(yaml_path.parts) >= 2 else "unknown"
        for record in task_records:
            row = dict(record)
            row["robot"] = robot
            row["category"] = category
            row["yaml_path"] = str(yaml_path.relative_to(PROJECT_ROOT))
            row["latest_run"] = str(latest_run.relative_to(PROJECT_ROOT)) if latest_run else None
            row["latest_run_loader_policy"] = _infer_run_loader_policy(latest_run, record)
            rows.append(row)
    return rows


def _render_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Texture Asset Audit",
        "",
        "| Robot | Task | Asset | Family | Recommended | Latest Run Policy | YAML | Latest Run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {robot} | {task_name} | {asset_name} | {asset_family} | {loader_policy} | {latest_run_loader_policy} | {yaml_path} | {latest_run} |".format(
                robot=row.get("robot", ""),
                task_name=row.get("task_name", ""),
                asset_name=row.get("asset_name", ""),
                asset_family=row.get("asset_family", ""),
                loader_policy=row.get("loader_policy", ""),
                latest_run_loader_policy=row.get("latest_run_loader_policy", ""),
                yaml_path=row.get("yaml_path", ""),
                latest_run=row.get("latest_run", ""),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit texture-sensitive Isaac Sim assets used by tasks.")
    parser.add_argument("--tasks-root", type=str, default="tasks", help="Task YAML root to scan.")
    parser.add_argument(
        "--outputs-root",
        type=str,
        default="outputs/data_collection",
        help="Collection outputs root for latest-run inspection.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".internal/local_docs/adc",
        help="Directory for audit artifacts.",
    )
    args = parser.parse_args()

    tasks_root = (PROJECT_ROOT / args.tasks_root).resolve()
    outputs_root = (PROJECT_ROOT / args.outputs_root).resolve()
    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _collect_records(tasks_root, outputs_root)
    json_path = output_dir / "texture_asset_audit.json"
    md_path = output_dir / "texture_asset_audit.md"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(rows), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
