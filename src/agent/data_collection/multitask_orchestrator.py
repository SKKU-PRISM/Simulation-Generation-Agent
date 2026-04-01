"""Multi-task data collection orchestration for LeRobot + HF export."""

from __future__ import annotations

import copy
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import DataCollectionConfig, PROJECT_ROOT, load_pipeline_config
from .lerobot_tools import (
    check_lerobot_dataset,
    convert_raw_dataset_to_lerobot,
    ensure_huggingface_hub_available,
    ensure_lerobot_available,
    publish_lerobot_dataset,
    write_json_report,
)
from .pipeline import DataCollectionPipeline
from .raw_dataset_merge import load_raw_dataset_metadata, merge_raw_datasets

logger = logging.getLogger(__name__)


def run_multitask_to_hf(
    task_paths: list[str | Path],
    *,
    config: DataCollectionConfig | None = None,
    output_root: str | Path | None = None,
    local_repo_prefix: str = "local/multitask",
    hf_repo_prefix: str | None = None,
    upload_to_hf: bool = False,
    hf_private: bool = False,
    token_env: str = "HF_TOKEN",
    token: str | None = None,
    continue_on_failure: bool = True,
) -> dict[str, Any]:
    """Collect multiple tasks, merge by robot, convert to LeRobot, and optionally upload."""

    resolved_tasks = [Path(task).expanduser().resolve() for task in task_paths]
    if not resolved_tasks:
        raise ValueError("run_multitask_to_hf requires at least one task YAML path")

    missing_tasks = [str(task) for task in resolved_tasks if not task.exists()]
    if missing_tasks:
        raise FileNotFoundError(f"Task YAML(s) not found: {missing_tasks}")

    ensure_lerobot_available()
    if upload_to_hf:
        ensure_huggingface_hub_available()
        if not hf_repo_prefix:
            raise ValueError("hf_repo_prefix is required when upload_to_hf=True")

    cfg = copy.deepcopy(config or load_pipeline_config())
    batch_root = _resolve_batch_root(output_root)
    task_runs_root = batch_root / "task_runs"
    task_reports_root = batch_root / "task_reports"
    merged_raw_root = batch_root / "merged_raw"
    lerobot_root = batch_root / "lerobot"
    publish_reports_root = batch_root / "publish_reports"
    robot_reports_root = batch_root / "robot_reports"

    for path in (
        batch_root,
        task_runs_root,
        task_reports_root,
        merged_raw_root,
        lerobot_root,
        publish_reports_root,
        robot_reports_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    batch_report: dict[str, Any] = {
        "batch_root": str(batch_root),
        "task_count": len(resolved_tasks),
        "tasks": [],
        "robot_datasets": {},
        "summary": {},
        "pass": False,
    }

    task_results: list[dict[str, Any]] = []
    for idx, task_path in enumerate(resolved_tasks):
        task_cfg = copy.deepcopy(cfg)
        task_cfg.output_dir = str(task_runs_root.resolve())
        task_report_path = task_reports_root / f"{idx:03d}_{_slugify_task_path(task_path)}.json"
        task_result: dict[str, Any]
        try:
            pipeline = DataCollectionPipeline(
                yaml_path=str(task_path),
                config=task_cfg,
                auto_convert_to_lerobot=False,
            )
            task_result = pipeline.run()
            task_result["yaml_path"] = str(task_path)
            task_result["merge_eligible"] = _is_merge_eligible(task_result)
            if task_result.get("raw_dataset"):
                metadata = load_raw_dataset_metadata(task_result["raw_dataset"])
                task_result["raw_metadata"] = {
                    "robot_name": metadata.get("robot_name"),
                    "total_episodes": metadata.get("total_episodes", 0),
                    "successful_episodes": metadata.get("successful_episodes", 0),
                    "camera_names": metadata.get("camera_names", []),
                }
        except Exception as exc:
            task_result = {
                "success": False,
                "pipeline_completed": False,
                "yaml_path": str(task_path),
                "error": str(exc),
                "merge_eligible": False,
            }
            logger.error("Multi-task collection failed for %s: %s", task_path, exc)
            if not continue_on_failure:
                task_result["report_path"] = str(task_report_path)
                write_json_report(task_result, task_report_path)
                raise

        task_result["report_path"] = str(task_report_path)
        write_json_report(task_result, task_report_path)
        task_results.append(task_result)
        batch_report["tasks"].append(task_result)

    eligible_results = [result for result in task_results if result.get("merge_eligible")]
    grouped_results = _group_results_by_robot(eligible_results)

    for robot_name, robot_results in grouped_results.items():
        robot_report_path = robot_reports_root / f"{robot_name}.json"
        robot_report: dict[str, Any] = {
            "robot": robot_name,
            "task_count": len(robot_results),
            "tasks": [result.get("task") for result in robot_results],
            "raw_datasets": [result.get("raw_dataset") for result in robot_results],
            "merged_raw_dataset": None,
            "local_repo_id": _build_robot_repo_id(local_repo_prefix, robot_name),
            "lerobot_dataset_root": None,
            "validation_report": None,
            "hf_repo_id": None,
            "publish_report": None,
            "publish_report_path": None,
            "pass": False,
        }
        try:
            merged_raw_dir = merged_raw_root / robot_name / "raw_dataset"
            merge_raw_datasets(
                [result["raw_dataset"] for result in robot_results],
                merged_raw_dir,
                source_records=robot_results,
            )
            robot_report["merged_raw_dataset"] = str(merged_raw_dir)

            local_repo_id = robot_report["local_repo_id"]
            dataset_root = convert_raw_dataset_to_lerobot(
                merged_raw_dir,
                repo_id=local_repo_id,
                output_root=lerobot_root,
            )
            robot_report["lerobot_dataset_root"] = str(dataset_root)

            validation_report = check_lerobot_dataset(dataset_root, repo_id=local_repo_id)
            robot_report["validation_report"] = validation_report

            if upload_to_hf:
                hf_repo_id = _build_robot_repo_id(hf_repo_prefix or "", robot_name)
                robot_report["hf_repo_id"] = hf_repo_id
                publish_report = publish_lerobot_dataset(
                    dataset_root,
                    repo_id=hf_repo_id,
                    private=hf_private,
                    token_env=token_env,
                    token=token,
                    local_repo_id=local_repo_id,
                )
                publish_report_path = publish_reports_root / f"{robot_name}.json"
                write_json_report(publish_report, publish_report_path)
                robot_report["publish_report"] = publish_report
                robot_report["publish_report_path"] = str(publish_report_path)

            robot_report["pass"] = True
        except Exception as exc:
            robot_report["error"] = str(exc)
            logger.error("Multi-task merge/convert failed for %s: %s", robot_name, exc)
            if not continue_on_failure:
                robot_report["report_path"] = str(robot_report_path)
                write_json_report(robot_report, robot_report_path)
                raise

        robot_report["report_path"] = str(robot_report_path)
        write_json_report(robot_report, robot_report_path)
        batch_report["robot_datasets"][robot_name] = robot_report

    merged_robot_count = len(batch_report["robot_datasets"])
    batch_report["summary"] = {
        "task_count": len(resolved_tasks),
        "task_success_count": sum(1 for result in task_results if result.get("pipeline_completed")),
        "merge_eligible_task_count": len(eligible_results),
        "robot_dataset_count": merged_robot_count,
        "robot_dataset_success_count": sum(
            1 for result in batch_report["robot_datasets"].values() if result.get("pass")
        ),
        "upload_to_hf": bool(upload_to_hf),
    }
    batch_report["pass"] = (
        merged_robot_count > 0
        and all(result.get("pass") for result in batch_report["robot_datasets"].values())
    )

    batch_report_path = batch_root / "batch_report.json"
    batch_report["report_path"] = str(batch_report_path)
    write_json_report(batch_report, batch_report_path)
    return batch_report


def _resolve_batch_root(output_root: str | Path | None) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / "data_collection" / "batches" / timestamp).resolve()


def _is_merge_eligible(task_result: dict[str, Any]) -> bool:
    if not task_result.get("pipeline_completed"):
        return False
    raw_dataset = task_result.get("raw_dataset")
    if not raw_dataset:
        return False
    try:
        metadata = load_raw_dataset_metadata(raw_dataset)
    except Exception:
        return False
    return int(metadata.get("total_episodes", 0)) > 0


def _group_results_by_robot(task_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in task_results:
        robot_name = str(result.get("robot") or "unknown")
        grouped.setdefault(robot_name, []).append(result)
    return grouped


def _build_robot_repo_id(prefix: str, robot_name: str) -> str:
    normalized_prefix = prefix.rstrip("/")
    if not normalized_prefix:
        return robot_name
    if normalized_prefix.endswith(f"-{robot_name}") or normalized_prefix.endswith(f"/{robot_name}"):
        return normalized_prefix
    return f"{normalized_prefix}-{robot_name}"


def _slugify_task_path(task_path: Path) -> str:
    rel = task_path
    try:
        rel = task_path.relative_to(PROJECT_ROOT)
    except ValueError:
        pass
    return str(rel).replace("/", "_").replace(".yaml", "")
