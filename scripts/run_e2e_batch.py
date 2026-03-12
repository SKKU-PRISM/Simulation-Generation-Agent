#!/usr/bin/env python3
"""Run the full E2E batch pipeline from a YAML config."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.e2e_orchestrator import run_e2e_batch


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run config-driven E2E collection: task list -> collection -> success-only raw "
            "-> export -> preprocess -> LeRobot -> optional HF upload -> reports."
        )
    )
    parser.add_argument("config", help="Path to E2E batch config YAML")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    report = run_e2e_batch(args.config)
    print(json.dumps(report.get("summary", {}), indent=2, ensure_ascii=False))
    print(f"Markdown report: {report.get('markdown_report_path')}")
    print(f"JSON report: {report.get('json_report_path')}")
    print(f"VLM audit Markdown: {report.get('vlm_audit_markdown_path')}")
    print(f"VLM audit JSON: {report.get('vlm_audit_json_path')}")
    return 0 if report.get("batch_completed", report.get("pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
