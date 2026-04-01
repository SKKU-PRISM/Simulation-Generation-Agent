#!/usr/bin/env python3
"""Upload a local LeRobot dataset to a Hugging Face dataset repo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.data_collection.lerobot_tools import publish_lerobot_dataset, write_json_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a local LeRobot dataset to the Hugging Face Hub"
    )
    parser.add_argument("dataset_root", help="Path to local LeRobot dataset root")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Hugging Face dataset repo ID, e.g. org/my_dataset",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the dataset repo as private",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable to read for the Hugging Face token",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Explicit Hugging Face token override",
    )
    parser.add_argument(
        "--local-repo-id",
        default=None,
        help="Optional local repo_id for validating the local LeRobot dataset before upload",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path for the JSON upload report",
    )
    args = parser.parse_args()

    report = publish_lerobot_dataset(
        args.dataset_root,
        repo_id=args.repo_id,
        private=args.private,
        token_env=args.token_env,
        token=args.token,
        local_repo_id=args.local_repo_id,
    )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    if args.report_path:
        report_path = write_json_report(report, args.report_path)
    else:
        report_path = write_json_report(
            report,
            dataset_root.parent / f"{dataset_root.name}_publish_report.json",
        )
    print(str(report_path))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
