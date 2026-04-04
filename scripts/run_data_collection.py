#!/usr/bin/env python3
"""
CLI entry point for simulation data collection.

Usage:
    # Basic: generate env + collect data
    python scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml

    # Use existing generated environment
    python scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
        --env-dir outputs/isaaclab/frankastack_20260219_160916

    # Configure episodes and output
    python scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml \
        --episodes 50 --repo-id "local/franka_stack_sim"

    # Skip VLM judging
    python scripts/run_data_collection.py tasks/franka/stack/franka_stack.yaml --no-vlm-judge

    # Batch collection
    python scripts/run_data_collection.py --batch tasks/franka/ --episodes 10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.data_collection.config import load_pipeline_config
from src.agent.data_collection.pipeline import DataCollectionPipeline, run_batch


def main():
    parser = argparse.ArgumentParser(
        description="Collect LeRobot datasets from IsaacLab simulation"
    )
    parser.add_argument(
        "yaml_path",
        nargs="?",
        help="Path to task YAML document",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Directory for batch processing (all YAMLs)",
    )
    parser.add_argument(
        "--env-dir",
        type=str,
        default=None,
        help="Path to pre-generated IsaacLab environment code",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to collect (default: from config)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Dataset repository ID (default: local/sim_dataset)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Recording FPS (default: match control rate)",
    )
    parser.add_argument(
        "--no-vlm-judge",
        action="store_true",
        help="Disable VLM success judging (record all episodes)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with GUI (not headless, enables camera rendering)",
    )
    parser.add_argument(
        "--target-success",
        type=int,
        default=None,
        help="Target number of successful episodes (loop until reached)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Maximum total episode attempts (default: target * 5)",
    )
    parser.add_argument(
        "--skip-auto-convert",
        action="store_true",
        help="Skip per-task raw -> LeRobot conversion after collection",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for collected data",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = load_pipeline_config(args.config)

    # Apply CLI overrides
    if args.episodes is not None:
        config.max_episodes = args.episodes
    if args.repo_id is not None:
        config.dataset_repo_id = args.repo_id
    if args.fps is not None:
        config.recording_fps = args.fps
    if args.no_vlm_judge:
        config.use_vlm_judge = False
    if args.gui:
        config.env_headless = False
    if args.target_success is not None:
        config.target_successful_episodes = args.target_success
    if args.max_attempts is not None:
        config.max_total_attempts = args.max_attempts
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # Batch mode
    if args.batch:
        results = run_batch(
            task_dir=args.batch,
            config=config,
            episodes_per_task=config.max_episodes,
            auto_convert_to_lerobot=not args.skip_auto_convert,
        )
        # Summary
        total = len(results)
        completed = sum(1 for r in results if r.get("pipeline_completed"))
        target_met = sum(1 for r in results if r.get("target_met"))
        print(f"\n{'='*60}")
        print(
            f"Batch complete: {completed}/{total} completed, "
            f"{target_met}/{total} met target"
        )
        for r in results:
            if r.get("target_met"):
                status = "OK"
            elif r.get("pipeline_completed"):
                status = "PARTIAL"
            else:
                status = "FAIL"
            task = r.get("task", r.get("yaml", "unknown"))
            print(f"  [{status}] {task}")
        return 0

    # Single task mode
    if args.yaml_path is None:
        parser.error("yaml_path is required (or use --batch)")

    pipeline = DataCollectionPipeline(
        yaml_path=args.yaml_path,
        config=config,
        env_dir=args.env_dir,
        auto_convert_to_lerobot=not args.skip_auto_convert,
    )

    result = pipeline.run()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Data Collection Complete")
    print(f"{'='*60}")
    print(f"  Task:    {result.get('task', 'unknown')}")
    print(f"  Robot:   {result.get('robot', 'unknown')}")
    print(f"  Completed: {result.get('pipeline_completed', False)}")
    print(f"  Target met: {result.get('target_met', False)}")
    print(f"  Output:  {result.get('output_dir', 'N/A')}")

    results_data = result.get("results", {})
    if results_data:
        total = result.get("total_episodes", results_data.get("total_episodes", 0))
        ok = result.get(
            "successful_episodes",
            results_data.get("successful_episodes", 0),
        )
        print(f"  Episodes: {ok}/{total} successful")

    if result.get("raw_dataset"):
        print(f"  Dataset: {result['raw_dataset']}")
    if result.get("lerobot_dataset"):
        print(f"  LeRobot: {result['lerobot_dataset']}")

    return 0 if result.get("pipeline_completed", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
