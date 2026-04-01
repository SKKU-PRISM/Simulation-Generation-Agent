#!/usr/bin/env python3
"""Convert a raw simulator dataset into a local LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.data_collection.lerobot_tools import (
    convert_raw_dataset_to_lerobot,
    write_json_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raw simulator dataset into a local LeRobot dataset"
    )
    parser.add_argument("raw_dataset_dir", help="Path to raw_dataset directory")
    parser.add_argument(
        "--repo-id",
        default="local/sim_dataset",
        help="Target LeRobot dataset repo ID / local folder layout",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for the local LeRobot dataset",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for a JSON conversion report",
    )
    args = parser.parse_args()

    dataset_root = convert_raw_dataset_to_lerobot(
        args.raw_dataset_dir,
        repo_id=args.repo_id,
        output_root=args.output_root,
    )
    report = {
        "raw_dataset_dir": str(Path(args.raw_dataset_dir).expanduser().resolve()),
        "repo_id": args.repo_id,
        "dataset_root": str(dataset_root),
        "pass": True,
    }
    if args.report_path:
        report_path = write_json_report(report, args.report_path)
    else:
        report_path = write_json_report(
            report,
            dataset_root.parent / f"{dataset_root.name}_conversion_report.json",
        )
    print(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "report_path": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
