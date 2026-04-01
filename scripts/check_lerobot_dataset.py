#!/usr/bin/env python3
"""Validate a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.data_collection.lerobot_tools import check_lerobot_dataset, write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a local LeRobot dataset and emit a JSON report"
    )
    parser.add_argument("dataset_root", help="Path to local LeRobot dataset root")
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Explicit repo_id for local LeRobot loading (recommended)",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for the JSON validation report",
    )
    args = parser.parse_args()

    report = check_lerobot_dataset(args.dataset_root, repo_id=args.repo_id)
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if args.report_path:
        report_path = write_json_report(report, args.report_path)
    else:
        report_path = write_json_report(
            report,
            dataset_root.parent / f"{dataset_root.name}_check_report.json",
        )
    print(str(report_path))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
