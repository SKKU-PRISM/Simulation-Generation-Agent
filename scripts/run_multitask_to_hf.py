#!/usr/bin/env python3
"""Run multi-task data collection, merge by robot, convert to LeRobot, and optionally upload."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_collection.config import load_pipeline_config
from src.data_collection.multitask_orchestrator import run_multitask_to_hf


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run multi-task IsaacLab data collection, merge successful raw datasets "
            "by robot, convert to LeRobot, and optionally upload to Hugging Face."
        )
    )
    parser.add_argument("tasks", nargs="+", help="Task YAML paths to collect")
    parser.add_argument("--config", default=None, help="Path to pipeline config YAML")
    parser.add_argument("--episodes", type=int, default=None, help="Override max episodes per task")
    parser.add_argument(
        "--target-success",
        type=int,
        default=None,
        help="Target number of successful episodes per task",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum total attempts per task",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Batch output root for reports, merged raw datasets, and LeRobot datasets",
    )
    parser.add_argument(
        "--local-repo-prefix",
        default="local/multitask",
        help="Prefix for local LeRobot repo IDs, e.g. local/mybatch",
    )
    parser.add_argument(
        "--hf-repo-prefix",
        default=None,
        help="Prefix for HF dataset repo IDs, e.g. org/mybatch",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload each robot-specific LeRobot dataset to Hugging Face after validation",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update HF dataset repos as private",
    )
    parser.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable to read for the Hugging Face token",
    )
    parser.add_argument("--token", default=None, help="Explicit Hugging Face token override")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch immediately on the first task/merge failure",
    )
    parser.add_argument(
        "--no-vlm-judge",
        action="store_true",
        help="Disable VLM success judging during collection",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run IsaacLab collection with GUI (not headless)",
    )
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

    config = load_pipeline_config(args.config)
    if args.episodes is not None:
        config.max_episodes = args.episodes
    if args.target_success is not None:
        config.target_successful_episodes = args.target_success
    if args.max_attempts is not None:
        config.max_total_attempts = args.max_attempts
    if args.no_vlm_judge:
        config.use_vlm_judge = False
    if args.gui:
        config.env_headless = False

    report = run_multitask_to_hf(
        args.tasks,
        config=config,
        output_root=args.output_root,
        local_repo_prefix=args.local_repo_prefix,
        hf_repo_prefix=args.hf_repo_prefix,
        upload_to_hf=args.upload_to_hf,
        hf_private=args.private,
        token_env=args.token_env,
        token=args.token,
        continue_on_failure=not args.fail_fast,
    )

    print(json.dumps(report["summary"], indent=2))
    print(f"Batch report: {report['report_path']}")
    for robot_name, robot_report in sorted(report["robot_datasets"].items()):
        print(
            f"[{robot_name}] merged={robot_report.get('merged_raw_dataset')} "
            f"lerobot={robot_report.get('lerobot_dataset_root')} "
            f"upload={'yes' if robot_report.get('publish_report_path') else 'no'}"
        )

    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
