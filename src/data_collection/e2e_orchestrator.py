"""Config-driven end-to-end batch orchestration for collection/export/LeRobot/HF."""

from __future__ import annotations

import copy
import gc
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.task_search.catalog import TaskCatalog
from src.task_search.taxonomy import TaskClassification, classify_task_yaml

from .config import DataCollectionConfig, PROJECT_ROOT, load_pipeline_config
from .dataset_export import ADC_COMPATIBLE_SCHEMA, export_dataset
from .dataset_preprocess import preprocess_exported_dataset
from .lerobot_tools import (
    check_lerobot_dataset,
    convert_raw_dataset_to_lerobot,
    ensure_huggingface_hub_available,
    ensure_lerobot_available,
    publish_lerobot_dataset,
    write_json_report,
)
from .pipeline import DataCollectionPipeline
from .raw_dataset_merge import (
    filter_raw_dataset_episodes,
    load_raw_dataset_metadata,
    merge_raw_datasets,
)

logger = logging.getLogger(__name__)

GOAL_POSITION_KEYS = (
    "target_position",
    "slot_position",
    "entry_position",
    "closed_position",
    "closed_center_position",
)
OBJECT_TASK_ROLES = {"movable_object", "goal_subject"}
GOAL_TASK_ROLES = {"placement_target", "insertion_target", "handle_target"}


@dataclass(frozen=True)
class E2ERunConfig:
    output_root: str | None = None
    continue_on_failure: bool = True
    export_schema: str = ADC_COMPATIBLE_SCHEMA
    build_export_artifacts: bool = True
    build_preprocessed_artifacts: bool = True
    preprocess_success_only: bool = True
    generate_report: bool = True
    resume: bool = False
    cleanup_intermediates: bool = True
    require_all_targets_for_robot_dataset: bool = True


@dataclass(frozen=True)
class E2ECollectionConfig:
    config_path: str | None = None
    llm_model: str = "gpt-5-mini"
    vlm_model: str = "gpt-5"
    use_vlm_judge: bool = True
    discard_failed_episodes: bool | None = None
    keep_failed_raw_dataset: bool | None = None
    gui: bool = False
    execution_timeout: int | None = None
    default_max_attempts_multiplier: int = 5
    task_retry_limit: int = 0
    require_target: bool = False


@dataclass(frozen=True)
class E2EHFConfig:
    upload: bool = False
    namespace: str = ""
    dataset_name: str = "sim_dataset"
    private: bool = False
    token_env: str = "HF_TOKEN"


@dataclass(frozen=True)
class E2ETaskConfig:
    requested_ref: str
    yaml_path: Path
    demos: int
    enabled: bool = True
    max_attempts: int | None = None
    baseline_tag: str | None = None
    task_retry_limit: int | None = None


@dataclass(frozen=True)
class E2EBatchConfig:
    source_path: Path
    run: E2ERunConfig = field(default_factory=E2ERunConfig)
    collection: E2ECollectionConfig = field(default_factory=E2ECollectionConfig)
    hf: E2EHFConfig = field(default_factory=E2EHFConfig)
    tasks: tuple[E2ETaskConfig, ...] = ()


def load_e2e_batch_config(config_path: str | Path) -> E2EBatchConfig:
    """Load a YAML config for the full E2E batch orchestrator."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"E2E batch config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    run_section = data.get("run", {})
    collection_section = data.get("collection", {})
    hf_section = data.get("hf", {})
    task_entries = data.get("tasks", [])
    if not task_entries:
        raise ValueError("E2E batch config must define at least one task")

    run_cfg = E2ERunConfig(
        output_root=run_section.get("output_root"),
        continue_on_failure=bool(run_section.get("continue_on_failure", True)),
        export_schema=str(run_section.get("export_schema", ADC_COMPATIBLE_SCHEMA)),
        build_export_artifacts=bool(run_section.get("build_export_artifacts", True)),
        build_preprocessed_artifacts=bool(run_section.get("build_preprocessed_artifacts", True)),
        preprocess_success_only=bool(run_section.get("preprocess_success_only", True)),
        generate_report=bool(run_section.get("generate_report", True)),
        resume=bool(run_section.get("resume", False)),
        cleanup_intermediates=bool(run_section.get("cleanup_intermediates", True)),
        require_all_targets_for_robot_dataset=bool(
            run_section.get("require_all_targets_for_robot_dataset", True)
        ),
    )
    if run_cfg.export_schema != ADC_COMPATIBLE_SCHEMA:
        raise ValueError(
            f"Only '{ADC_COMPATIBLE_SCHEMA}' export is currently supported by run_e2e_batch"
        )

    collection_cfg = E2ECollectionConfig(
        config_path=collection_section.get("config_path"),
        llm_model=str(collection_section.get("llm_model", "gpt-5-mini")),
        vlm_model=str(collection_section.get("vlm_model", "gpt-5")),
        use_vlm_judge=bool(collection_section.get("use_vlm_judge", True)),
        discard_failed_episodes=(
            bool(collection_section["discard_failed_episodes"])
            if "discard_failed_episodes" in collection_section
            else None
        ),
        keep_failed_raw_dataset=(
            bool(collection_section["keep_failed_raw_dataset"])
            if "keep_failed_raw_dataset" in collection_section
            else None
        ),
        gui=bool(collection_section.get("gui", False)),
        execution_timeout=(
            int(collection_section["execution_timeout"])
            if collection_section.get("execution_timeout") is not None
            else None
        ),
        default_max_attempts_multiplier=int(
            collection_section.get("default_max_attempts_multiplier", 5)
        ),
        task_retry_limit=int(collection_section.get("task_retry_limit", 0)),
        require_target=bool(collection_section.get("require_target", False)),
    )
    if collection_cfg.default_max_attempts_multiplier <= 0:
        raise ValueError("default_max_attempts_multiplier must be positive")
    if collection_cfg.task_retry_limit < 0:
        raise ValueError("task_retry_limit must be >= 0")

    hf_cfg = E2EHFConfig(
        upload=bool(hf_section.get("upload", False)),
        namespace=str(hf_section.get("namespace", "")).strip(),
        dataset_name=str(hf_section.get("dataset_name", "sim_dataset")).strip(),
        private=bool(hf_section.get("private", False)),
        token_env=str(hf_section.get("token_env", "HF_TOKEN")),
    )
    if not hf_cfg.dataset_name:
        raise ValueError("hf.dataset_name must be non-empty")
    if hf_cfg.upload and not hf_cfg.namespace:
        raise ValueError("hf.namespace is required when hf.upload=true")

    catalog = TaskCatalog()
    catalog_entries = catalog.discover()
    tasks: list[E2ETaskConfig] = []
    for task_entry in task_entries:
        if not isinstance(task_entry, dict):
            raise ValueError(f"Invalid task entry: {task_entry!r}")
        raw_ref = str(task_entry.get("yaml") or task_entry.get("task") or "").strip()
        if not raw_ref:
            raise ValueError("Each task entry must define either 'yaml' or 'task'")
        enabled = bool(task_entry.get("enabled", True))
        demos = int(task_entry.get("demos", 0))
        if demos <= 0:
            raise ValueError(f"Task '{raw_ref}' must set demos > 0")
        max_attempts_raw = task_entry.get("max_attempts")
        max_attempts = int(max_attempts_raw) if max_attempts_raw is not None else None
        if max_attempts is not None and max_attempts < demos:
            raise ValueError(f"Task '{raw_ref}' max_attempts must be >= demos")
        task_retry_limit_raw = task_entry.get("task_retry_limit")
        task_retry_limit = (
            int(task_retry_limit_raw) if task_retry_limit_raw is not None else None
        )
        if task_retry_limit is not None and task_retry_limit < 0:
            raise ValueError(f"Task '{raw_ref}' task_retry_limit must be >= 0")

        resolved_yaml = resolve_task_reference(raw_ref, catalog_entries)
        tasks.append(
            E2ETaskConfig(
                requested_ref=raw_ref,
                yaml_path=resolved_yaml,
                demos=demos,
                enabled=enabled,
                max_attempts=max_attempts,
                baseline_tag=(
                    str(task_entry.get("baseline_tag")).strip()
                    if task_entry.get("baseline_tag") is not None
                    else None
                ),
                task_retry_limit=task_retry_limit,
            )
        )

    return E2EBatchConfig(
        source_path=path,
        run=run_cfg,
        collection=collection_cfg,
        hf=hf_cfg,
        tasks=tuple(tasks),
    )


def run_e2e_batch(
    config: str | Path | E2EBatchConfig,
    *,
    resume: bool | None = None,
) -> dict[str, Any]:
    """Run the full collection/export/preprocess/LeRobot/HF batch from YAML config."""

    load_dotenv(PROJECT_ROOT / ".env")
    batch_cfg = config if isinstance(config, E2EBatchConfig) else load_e2e_batch_config(config)
    ensure_lerobot_available()
    if batch_cfg.hf.upload:
        ensure_huggingface_hub_available()
    _ensure_batch_environment(batch_cfg)

    batch_root = _resolve_batch_root(batch_cfg.run.output_root)
    task_runs_root = batch_root / "task_runs"
    task_reports_root = batch_root / "task_reports"
    successful_raw_root = batch_root / "successful_raw"
    merged_raw_root = batch_root / "merged_raw"
    exported_root = batch_root / "exported"
    preprocessed_root = batch_root / "preprocessed"
    lerobot_root = batch_root / "lerobot"
    publish_reports_root = batch_root / "publish_reports"
    reports_root = batch_root / "reports"
    for path in (
        batch_root,
        task_runs_root,
        task_reports_root,
        successful_raw_root,
        merged_raw_root,
        exported_root,
        preprocessed_root,
        lerobot_root,
        publish_reports_root,
        reports_root,
    ):
        path.mkdir(parents=True, exist_ok=True)

    resume_mode = resume if resume is not None else batch_cfg.run.resume

    snapshot_path = batch_root / "config_snapshot.yaml"
    if resume_mode:
        resume_snapshot = batch_root / f"config_snapshot_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
        _write_config_snapshot(batch_cfg, resume_snapshot)
    else:
        _write_config_snapshot(batch_cfg, snapshot_path)

    base_collection_cfg = load_pipeline_config(batch_cfg.collection.config_path)
    base_collection_cfg.llm_model = batch_cfg.collection.llm_model
    base_collection_cfg.vlm_model = batch_cfg.collection.vlm_model
    base_collection_cfg.use_vlm_judge = batch_cfg.collection.use_vlm_judge
    if batch_cfg.collection.discard_failed_episodes is not None:
        base_collection_cfg.discard_failed_episodes = batch_cfg.collection.discard_failed_episodes
    if batch_cfg.collection.keep_failed_raw_dataset is not None:
        base_collection_cfg.keep_failed_raw_dataset = batch_cfg.collection.keep_failed_raw_dataset
    base_collection_cfg.env_headless = not batch_cfg.collection.gui
    if batch_cfg.collection.execution_timeout is not None:
        base_collection_cfg.execution_timeout = batch_cfg.collection.execution_timeout

    failed_raw_cleanup = _cleanup_failed_task_raw_datasets(
        task_reports_root=task_reports_root,
        enabled=not base_collection_cfg.keep_failed_raw_dataset,
    )

    # --- Resume: skip only successful completed tasks; rerun all failed/incomplete tasks ---
    completed_task_results: dict[int, dict[str, Any]] = {}
    if resume_mode:
        if not batch_cfg.run.output_root:
            raise ValueError(
                "Resume mode requires an explicit 'run.output_root' in the batch config."
            )
        if not batch_root.exists():
            raise FileNotFoundError(
                f"Resume mode: batch root {batch_root} does not exist."
            )
        completed_task_results = _load_resume_task_results(
            task_reports_root, batch_cfg.tasks
        )
        logger.info(
            "Resume mode: found %d/%d successful completed tasks; rerunning the remaining %d task(s)",
            len(completed_task_results),
            len(batch_cfg.tasks),
            len(batch_cfg.tasks) - len(completed_task_results),
        )

    report: dict[str, Any] = {
        "batch_root": str(batch_root),
        "config_path": str(batch_cfg.source_path),
        "config_snapshot_path": str(snapshot_path),
        "task_runs": [],
        "robot_datasets": {},
        "robot_dataset_build_deferred": False,
        "robot_dataset_build_blockers": [],
        "summary": {},
        "batch_completed": False,
        "pass": False,
        "resumed": resume_mode,
        "resumed_task_count": len(completed_task_results),
        "failed_raw_cleanup": failed_raw_cleanup,
    }

    successful_task_records: list[dict[str, Any]] = []
    for task_idx, task_cfg in enumerate(batch_cfg.tasks):
        task_slug = _slugify_task(task_cfg.yaml_path)
        task_json_report_path = task_reports_root / f"{task_idx:03d}_{task_slug}.json"

        # Resume: preserve successful completed tasks and rerun failed/incomplete ones.
        if resume_mode and task_idx in completed_task_results:
            prev_result = completed_task_results[task_idx]
            prev_result["resumed_from_previous"] = True
            _normalize_resume_task_result(prev_result)
            report["task_runs"].append(prev_result)
            if prev_result.get("successful_raw_dataset"):
                successful_task_records.append(prev_result)
            logger.info(
                "Resume: skipping completed task %d/%d (%s) — successful_episodes=%d",
                task_idx + 1,
                len(batch_cfg.tasks),
                task_slug,
                prev_result.get("successful_episodes", 0),
            )
            continue

        if not task_cfg.enabled:
            task_result = _build_skipped_task_result(task_cfg)
            task_result["task_report_path"] = str(task_json_report_path)
            write_json_report(task_result, task_json_report_path)
            report["task_runs"].append(task_result)
            logger.info(
                "Skipping disabled task %d/%d (%s)",
                task_idx + 1,
                len(batch_cfg.tasks),
                task_slug,
            )
            continue

        try:
            task_result = _run_single_task_with_retries(
                task_cfg=task_cfg,
                base_config=base_collection_cfg,
                collection_cfg=batch_cfg.collection,
                task_runs_root=task_runs_root,
                successful_raw_root=successful_raw_root,
            )
        except Exception as exc:
            task_result = _build_failed_task_result(task_cfg, exc)
            logger.error("E2E batch task failed for %s: %s", task_cfg.yaml_path, exc)
            task_result["task_report_path"] = str(task_json_report_path)
            write_json_report(task_result, task_json_report_path)
            report["task_runs"].append(task_result)
            _cleanup_gpu_between_tasks()
            if not batch_cfg.run.continue_on_failure:
                raise
            continue

        task_result["task_report_path"] = str(task_json_report_path)
        write_json_report(task_result, task_json_report_path)
        report["task_runs"].append(task_result)
        if task_result.get("successful_raw_dataset"):
            successful_task_records.append(task_result)
        _cleanup_gpu_between_tasks()

    robot_dataset_build_blockers: list[dict[str, Any]] = []
    if batch_cfg.run.require_all_targets_for_robot_dataset:
        for task_result in report["task_runs"]:
            if task_result.get("skipped") or not task_result.get("enabled", True):
                continue
            if task_result.get("target_met", False):
                continue
            robot_dataset_build_blockers.append(
                {
                    "task_name": task_result.get("task_name"),
                    "requested_demos": int(task_result.get("requested_demos", 0) or 0),
                    "successful_dataset_episodes": int(
                        task_result.get("successful_dataset_episodes", 0) or 0
                    ),
                    "remaining_requested_demos": int(
                        task_result.get("remaining_requested_demos", 0) or 0
                    ),
                    "terminal_status": task_result.get("terminal_status"),
                }
            )

    if robot_dataset_build_blockers:
        report["robot_dataset_build_deferred"] = True
        report["robot_dataset_build_blockers"] = robot_dataset_build_blockers
        logger.info(
            "Deferring robot-level dataset build until all enabled tasks reach target demos. Remaining blockers: %s",
            ", ".join(
                f"{entry['task_name']} ({entry['successful_dataset_episodes']}/{entry['requested_demos']})"
                for entry in robot_dataset_build_blockers
            ),
        )
    else:
        single_robot_run = len({record["robot"] for record in successful_task_records}) <= 1
        grouped_records = _group_successful_task_records(successful_task_records)
        for robot_name, robot_records in sorted(grouped_records.items()):
            robot_report = _build_robot_dataset(
                robot_name=robot_name,
                task_records=robot_records,
                batch_cfg=batch_cfg,
                single_robot_run=single_robot_run,
                merged_raw_root=merged_raw_root,
                exported_root=exported_root,
                preprocessed_root=preprocessed_root,
                lerobot_root=lerobot_root,
                publish_reports_root=publish_reports_root,
            )
            report["robot_datasets"][robot_name] = robot_report

    # Cleanup intermediate directories to save disk space
    if batch_cfg.run.cleanup_intermediates and report["robot_datasets"]:
        _cleanup_intermediate_dirs(
            successful_raw_root=successful_raw_root,
            merged_raw_root=merged_raw_root,
            exported_root=exported_root,
        )

    report["batch_completed"] = len(report["task_runs"]) == len(batch_cfg.tasks)
    report["summary"] = _build_batch_summary(report, batch_cfg)
    report["pass"] = (
        bool(report["batch_completed"])
        and bool(report["robot_datasets"])
        and all(robot.get("pass", False) for robot in report["robot_datasets"].values())
        and (not batch_cfg.hf.upload or bool(report["robot_datasets"]))
    )

    markdown_report_path = reports_root / "run_summary.md"
    json_report_path = reports_root / "run_summary.json"
    vlm_audit_markdown_path = reports_root / "vlm_audit.md"
    vlm_audit_json_path = reports_root / "vlm_audit.json"
    report["markdown_report_path"] = str(markdown_report_path)
    report["json_report_path"] = str(json_report_path)
    report["vlm_audit_markdown_path"] = str(vlm_audit_markdown_path)
    report["vlm_audit_json_path"] = str(vlm_audit_json_path)
    report["vlm_audit"] = _build_vlm_audit(report)
    if batch_cfg.run.generate_report:
        markdown = _render_markdown_report(report, batch_cfg)
        markdown_report_path.write_text(markdown)
        vlm_audit_markdown = _render_vlm_audit_markdown(report["vlm_audit"])
        vlm_audit_markdown_path.write_text(vlm_audit_markdown)
    write_json_report(report["vlm_audit"], vlm_audit_json_path)
    write_json_report(report, json_report_path)
    return report


def resolve_task_reference(reference: str, catalog_entries: list[Any] | None = None) -> Path:
    """Resolve either a YAML path or a task name/stem into a task YAML path."""

    candidate = Path(reference).expanduser()
    if candidate.exists():
        return candidate.resolve()

    project_candidate = (PROJECT_ROOT / reference).resolve()
    if project_candidate.exists():
        return project_candidate

    entries = catalog_entries or TaskCatalog().discover()
    normalized_ref = reference.strip().casefold()
    exact_matches: list[Path] = []
    fuzzy_matches: list[Path] = []
    for entry in entries:
        yaml_path = Path(entry.yaml_path).resolve()
        relative_path = str(yaml_path.relative_to(PROJECT_ROOT)).casefold()
        stem = yaml_path.stem.casefold()
        task_name = entry.task_name.casefold()
        if normalized_ref in {relative_path, stem, task_name}:
            exact_matches.append(yaml_path)
        elif normalized_ref and (
            normalized_ref in relative_path
            or normalized_ref in stem
            or normalized_ref in task_name
        ):
            fuzzy_matches.append(yaml_path)

    matches = exact_matches or fuzzy_matches
    if not matches:
        raise FileNotFoundError(f"Could not resolve task reference '{reference}'")
    if len(matches) > 1:
        rel_matches = sorted(str(match.relative_to(PROJECT_ROOT)) for match in matches)
        raise ValueError(
            f"Task reference '{reference}' is ambiguous. Candidates: {rel_matches}"
        )
    return matches[0]


def _run_single_task_with_retries(
    *,
    task_cfg: E2ETaskConfig,
    base_config: DataCollectionConfig,
    collection_cfg: E2ECollectionConfig,
    task_runs_root: Path,
    successful_raw_root: Path,
) -> dict[str, Any]:
    retry_limit = (
        task_cfg.task_retry_limit
        if task_cfg.task_retry_limit is not None
        else collection_cfg.task_retry_limit
    )
    attempt_history: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None
    first_successful_pipeline_attempt: int | None = None
    max_pipeline_attempts = retry_limit + 1
    # require_target: keep re-running until target is met (with hard cap)
    require_target = collection_cfg.require_target
    require_target_max_rounds = 10  # hard cap to prevent infinite loops
    round_count = 0
    cumulative_successful = 0

    for pipeline_attempt in range(1, max_pipeline_attempts + 1):
        try:
            attempt_result = _run_single_task(
                task_cfg=task_cfg,
                base_config=base_config,
                collection_cfg=collection_cfg,
                task_runs_root=task_runs_root,
                successful_raw_root=successful_raw_root,
                pipeline_attempt=pipeline_attempt,
            )
        except Exception as exc:
            attempt_result = _build_failed_task_result(
                task_cfg,
                exc,
                pipeline_attempt=pipeline_attempt,
            )

        attempt_history.append(_build_pipeline_attempt_entry(attempt_result))
        final_result = attempt_result
        cumulative_successful += int(attempt_result.get("successful_episodes", 0))

        if attempt_result.get("pipeline_completed") and first_successful_pipeline_attempt is None:
            first_successful_pipeline_attempt = pipeline_attempt

        if pipeline_attempt >= max_pipeline_attempts or not _should_retry_task_pipeline(attempt_result):
            # Check require_target: if target not met, extend attempts
            if (
                require_target
                and attempt_result.get("pipeline_completed")
                and not attempt_result.get("target_met")
                and cumulative_successful < task_cfg.demos
                and round_count < require_target_max_rounds
            ):
                remaining = task_cfg.demos - cumulative_successful
                round_count += 1
                logger.info(
                    "require_target: %s has %d/%d successful episodes, need %d more (round %d/%d)",
                    task_cfg.yaml_path.stem,
                    cumulative_successful,
                    task_cfg.demos,
                    remaining,
                    round_count,
                    require_target_max_rounds,
                )
                # Continue — don't break
                max_pipeline_attempts += 1
                continue
            break

        logger.warning(
            "Retrying task pipeline for %s after pipeline attempt %d/%d (terminal_status=%s, failure_category=%s)",
            task_cfg.yaml_path,
            pipeline_attempt,
            max_pipeline_attempts,
            attempt_result.get("terminal_status"),
            attempt_result.get("failure_category"),
        )

    if final_result is None:
        raise RuntimeError(f"Failed to execute task pipeline for {task_cfg.yaml_path}")

    final_result["pipeline_retry_limit"] = retry_limit
    final_result["pipeline_attempts_used"] = len(attempt_history)
    final_result["pipeline_retried"] = len(attempt_history) > 1
    final_result["successful_pipeline_attempt"] = first_successful_pipeline_attempt
    final_result["pipeline_attempt_history"] = attempt_history
    final_result["require_target_rounds_used"] = round_count
    final_result["cumulative_successful"] = cumulative_successful
    return final_result


def _run_single_task(
    *,
    task_cfg: E2ETaskConfig,
    base_config: DataCollectionConfig,
    collection_cfg: E2ECollectionConfig,
    task_runs_root: Path,
    successful_raw_root: Path,
    pipeline_attempt: int = 1,
) -> dict[str, Any]:
    carryover = _load_partial_task_progress(
        task_cfg=task_cfg,
        task_runs_root=task_runs_root,
    )
    task_collection_cfg = copy.deepcopy(base_config)
    task_collection_cfg.output_dir = str(task_runs_root.resolve())
    prior_successful = int(carryover.get("successful_episodes", 0))
    prior_attempted = int(carryover.get("attempted_episodes", 0))
    remaining_demos = max(0, task_cfg.demos - prior_successful)
    task_collection_cfg.max_episodes = max(1, remaining_demos)
    task_collection_cfg.target_successful_episodes = max(1, remaining_demos)
    resolved_max_attempts = (
        task_cfg.max_attempts
        if task_cfg.max_attempts is not None
        else task_cfg.demos * collection_cfg.default_max_attempts_multiplier
    )
    remaining_attempts = max(1, resolved_max_attempts - prior_attempted)
    task_collection_cfg.max_total_attempts = remaining_attempts
    if collection_cfg.execution_timeout is not None:
        task_collection_cfg.execution_timeout = collection_cfg.execution_timeout

    if prior_successful > 0:
        logger.info(
            "Resume carryover for %s: reusing %d successful episode(s) from prior partial run(s); collecting %d more with up to %d remaining attempts",
            task_cfg.yaml_path.stem,
            prior_successful,
            remaining_demos,
            remaining_attempts,
        )

    pipeline = DataCollectionPipeline(
        yaml_path=str(task_cfg.yaml_path),
        config=task_collection_cfg,
        auto_convert_to_lerobot=False,
    )
    pipeline_result = pipeline.run()
    pipeline_result = _apply_partial_task_progress(
        pipeline_result=pipeline_result,
        carryover=carryover,
        requested_demos=task_cfg.demos,
    )

    successful_raw_dataset = None
    filtered_raw_episode_count = 0
    failed_raw_dataset_removed = False
    failed_raw_dataset_cleanup_reason = None
    candidate_raw_paths = [
        Path(path).expanduser().resolve()
        for path in carryover.get("raw_dataset_paths", [])
        if path
    ]
    raw_dataset_dir = pipeline_result.get("raw_dataset")
    if raw_dataset_dir:
        raw_path = Path(raw_dataset_dir).expanduser().resolve()
        try:
            raw_meta = load_raw_dataset_metadata(raw_path)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unusable raw dataset for %s: %s", task_cfg.yaml_path, exc)
            if raw_path.exists() and not task_collection_cfg.keep_failed_raw_dataset:
                import shutil

                shutil.rmtree(raw_path)
                failed_raw_dataset_removed = True
                failed_raw_dataset_cleanup_reason = "invalid_raw_dataset_removed"
            pipeline_result["raw_dataset"] = None
        else:
            raw_successful = int(raw_meta.get("successful_episodes", 0))
            if raw_successful > 0:
                candidate_raw_paths.append(raw_path)
            elif raw_path.exists() and not task_collection_cfg.keep_failed_raw_dataset:
                import shutil

                shutil.rmtree(raw_path)
                failed_raw_dataset_removed = True
                failed_raw_dataset_cleanup_reason = "zero_success_raw_dataset_removed"
                pipeline_result["raw_dataset"] = None

    successful_raw_dataset, filtered_raw_episode_count = _materialize_successful_raw_dataset(
        task_cfg=task_cfg,
        raw_dataset_paths=candidate_raw_paths,
        successful_raw_root=successful_raw_root,
    )

    classification = classify_task_yaml(task_cfg.yaml_path)
    task_report = _build_task_run_report(
        task_cfg=task_cfg,
        classification=classification,
        pipeline_result=pipeline_result,
        successful_raw_dataset=successful_raw_dataset,
        filtered_raw_episode_count=filtered_raw_episode_count,
        resolved_max_attempts=resolved_max_attempts,
        pipeline_attempt=pipeline_attempt,
        failed_raw_dataset_removed=failed_raw_dataset_removed,
        failed_raw_dataset_cleanup_reason=failed_raw_dataset_cleanup_reason,
        keep_failed_raw_dataset=task_collection_cfg.keep_failed_raw_dataset,
    )
    task_report["carryover_successful_episodes"] = prior_successful
    task_report["carryover_attempted_episodes"] = prior_attempted
    task_report["remaining_requested_demos"] = remaining_demos
    return task_report


def _load_partial_task_progress(
    *,
    task_cfg: E2ETaskConfig,
    task_runs_root: Path,
) -> dict[str, Any]:
    """Load partial successful progress from prior interrupted/failed runs for a task."""

    task_name = _load_task_name_from_yaml(task_cfg.yaml_path)
    run_dirs = sorted(
        [
            path
            for path in task_runs_root.glob(f"{task_name}_*")
            if path.is_dir()
        ]
    )

    partial_runs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        raw_dataset_dir = run_dir / "raw_dataset"
        metadata = None
        if raw_dataset_dir.exists():
            try:
                metadata = load_raw_dataset_metadata(raw_dataset_dir)
            except (FileNotFoundError, json.JSONDecodeError):
                metadata = None

        collection_results = None
        collection_results_path = run_dir / "collection_results.json"
        if collection_results_path.exists():
            try:
                with open(collection_results_path) as f:
                    collection_results = json.load(f)
            except (OSError, json.JSONDecodeError):
                collection_results = None

        successful = 0
        attempted = 0
        geometry_successful = 0
        vlm_successful = 0
        overall_successful = 0
        episodes: list[dict[str, Any]] = []

        if collection_results:
            successful = int(collection_results.get("successful_episodes", 0))
            attempted = int(collection_results.get("total_episodes", 0))
            geometry_successful = int(collection_results.get("geometry_successful_episodes", successful))
            vlm_successful = int(collection_results.get("vlm_successful_episodes", 0))
            overall_successful = int(collection_results.get("overall_successful_episodes", 0))
            episodes = list(collection_results.get("episodes") or [])
        elif metadata:
            successful = int(metadata.get("successful_episodes", 0))
            attempted = int(metadata.get("total_episodes", 0))
            geometry_successful = successful

        if successful <= 0 or metadata is None:
            continue

        partial_runs.append(
            {
                "run_dir": str(run_dir.resolve()),
                "raw_dataset_dir": str(raw_dataset_dir.resolve()),
                "successful_episodes": successful,
                "attempted_episodes": attempted,
                "geometry_successful_episodes": geometry_successful,
                "vlm_successful_episodes": vlm_successful,
                "overall_successful_episodes": overall_successful,
                "episodes": episodes,
                "front_video_generated": bool(metadata.get("front_video_generated", False)),
                "front_video_path": (
                    str((run_dir / metadata["front_video_path"]).resolve())
                    if metadata.get("front_video_path")
                    else None
                ),
                "front_video_camera_name": metadata.get("front_video_camera_name"),
                "front_video_episode": metadata.get("front_video_episode"),
                "front_video_success_type": metadata.get("front_video_success_type"),
            }
        )

    if not partial_runs:
        return {
            "successful_episodes": 0,
            "attempted_episodes": 0,
            "geometry_successful_episodes": 0,
            "vlm_successful_episodes": 0,
            "overall_successful_episodes": 0,
            "episodes": [],
            "raw_dataset_paths": [],
            "front_video_generated": False,
            "front_video_path": None,
            "front_video_camera_name": None,
            "front_video_episode": None,
            "front_video_success_type": None,
            "source_runs": [],
        }

    preferred_video = next(
        (
            run
            for run in partial_runs
            if run.get("front_video_generated") and run.get("front_video_path")
        ),
        partial_runs[0],
    )

    return {
        "successful_episodes": sum(int(run["successful_episodes"]) for run in partial_runs),
        "attempted_episodes": sum(int(run["attempted_episodes"]) for run in partial_runs),
        "geometry_successful_episodes": sum(int(run["geometry_successful_episodes"]) for run in partial_runs),
        "vlm_successful_episodes": sum(int(run["vlm_successful_episodes"]) for run in partial_runs),
        "overall_successful_episodes": sum(int(run["overall_successful_episodes"]) for run in partial_runs),
        "episodes": [episode for run in partial_runs for episode in run.get("episodes", [])],
        "raw_dataset_paths": [run["raw_dataset_dir"] for run in partial_runs],
        "front_video_generated": bool(preferred_video.get("front_video_generated")),
        "front_video_path": preferred_video.get("front_video_path"),
        "front_video_camera_name": preferred_video.get("front_video_camera_name"),
        "front_video_episode": preferred_video.get("front_video_episode"),
        "front_video_success_type": preferred_video.get("front_video_success_type"),
        "source_runs": [run["run_dir"] for run in partial_runs],
    }


def _apply_partial_task_progress(
    *,
    pipeline_result: dict[str, Any],
    carryover: dict[str, Any],
    requested_demos: int,
) -> dict[str, Any]:
    """Aggregate prior partial progress into the current pipeline result."""

    prior_successful = int(carryover.get("successful_episodes", 0))
    if prior_successful <= 0:
        return pipeline_result

    aggregated = dict(pipeline_result)
    results = dict((aggregated.get("results") or {}))
    current_episodes = list(results.get("episodes") or [])
    prior_episodes = list(carryover.get("episodes") or [])
    if prior_episodes:
        results["episodes"] = prior_episodes + current_episodes

    for key in (
        "total_episodes",
        "successful_episodes",
        "geometry_successful_episodes",
        "vlm_successful_episodes",
        "overall_successful_episodes",
    ):
        results[key] = int(results.get(key, 0)) + int(carryover.get(key, 0))

    results["target_met"] = int(results.get("successful_episodes", 0)) >= int(requested_demos)
    results["geometry_target_met"] = (
        int(results.get("geometry_successful_episodes", 0)) >= int(requested_demos)
    )
    results["vlm_target_met"] = (
        int(results.get("vlm_successful_episodes", 0)) >= int(requested_demos)
    )
    results["overall_target_met"] = (
        int(results.get("overall_successful_episodes", 0)) >= int(requested_demos)
    )

    if not aggregated.get("front_video_path") and carryover.get("front_video_path"):
        aggregated["front_video_generated"] = bool(carryover.get("front_video_generated"))
        aggregated["front_video_path"] = carryover.get("front_video_path")
        aggregated["front_video_camera_name"] = carryover.get("front_video_camera_name")
        aggregated["front_video_episode"] = carryover.get("front_video_episode")
        aggregated["front_video_success_type"] = carryover.get("front_video_success_type")

    aggregated["results"] = results
    aggregated["total_episodes"] = int(results.get("total_episodes", 0))
    aggregated["successful_episodes"] = int(results.get("successful_episodes", 0))
    aggregated["geometry_successful_episodes"] = int(results.get("geometry_successful_episodes", 0))
    aggregated["vlm_successful_episodes"] = int(results.get("vlm_successful_episodes", 0))
    aggregated["overall_successful_episodes"] = int(results.get("overall_successful_episodes", 0))
    aggregated["target_met"] = bool(results.get("target_met", False))
    return aggregated


def _materialize_successful_raw_dataset(
    *,
    task_cfg: E2ETaskConfig,
    raw_dataset_paths: list[Path],
    successful_raw_root: Path,
) -> tuple[Path | None, int]:
    """Build a combined success-only raw dataset for all known successful task runs."""

    usable_raw_paths: list[Path] = []
    for raw_path in raw_dataset_paths:
        try:
            metadata = load_raw_dataset_metadata(raw_path)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if int(metadata.get("successful_episodes", 0)) > 0:
            usable_raw_paths.append(raw_path)

    if not usable_raw_paths:
        return None, 0

    task_success_root = successful_raw_root / _slugify_task(task_cfg.yaml_path)
    if task_success_root.exists():
        shutil.rmtree(task_success_root)
    task_success_root.mkdir(parents=True, exist_ok=True)

    filtered_parts: list[Path] = []
    source_records: list[dict[str, Any]] = []
    for raw_path in usable_raw_paths:
        part_dir = task_success_root / "parts" / raw_path.parent.name / "raw_dataset"
        filtered = filter_raw_dataset_episodes(
            raw_path,
            part_dir,
            success_only=True,
        )
        filtered_meta = load_raw_dataset_metadata(filtered)
        if int(filtered_meta.get("total_episodes", 0)) <= 0:
            continue
        filtered_parts.append(filtered)
        source_records.append(
            {
                "task": _load_task_name_from_yaml(task_cfg.yaml_path),
                "yaml_path": str(task_cfg.yaml_path),
                "output_dir": str(raw_path.parent),
            }
        )

    if not filtered_parts:
        return None, 0

    merged_dir = task_success_root / "raw_dataset"
    merge_raw_datasets(
        [str(path) for path in filtered_parts],
        merged_dir,
        source_records=source_records,
    )
    merged_meta = load_raw_dataset_metadata(merged_dir)
    return merged_dir, int(merged_meta.get("total_episodes", 0))


def _load_task_name_from_yaml(yaml_path: Path) -> str:
    """Read the runtime task name from a task YAML document."""

    try:
        with open(yaml_path) as f:
            task_doc = yaml.safe_load(f) or {}
    except OSError:
        return yaml_path.stem
    return str((task_doc.get("task") or {}).get("name") or yaml_path.stem)


def _build_failed_task_result(
    task_cfg: E2ETaskConfig,
    exc: Exception,
    *,
    pipeline_attempt: int = 1,
) -> dict[str, Any]:
    classification = classify_task_yaml(task_cfg.yaml_path)
    return {
        "task": task_cfg.yaml_path.stem,
        "task_name": task_cfg.yaml_path.stem,
        "yaml_path": str(task_cfg.yaml_path),
        "requested_ref": task_cfg.requested_ref,
        "robot": _detect_robot_from_task_path(task_cfg.yaml_path),
        "classification": asdict(classification),
        "requested_demos": task_cfg.demos,
        "max_attempts": task_cfg.max_attempts,
        "baseline_tag": task_cfg.baseline_tag,
        "pipeline_attempt": pipeline_attempt,
        "pipeline_attempts_used": pipeline_attempt,
        "pipeline_retry_limit": (
            task_cfg.task_retry_limit if task_cfg.task_retry_limit is not None else 0
        ),
        "attempted_episodes": 0,
        "successful_episodes": 0,
        "geometry_successful_episodes": 0,
        "vlm_successful_episodes": 0,
        "overall_successful_episodes": 0,
        "successful_dataset_episodes": 0,
        "failed_episode_attempts": 0,
        "success_rate": 0.0,
        "success_policy": "geometry_or_vlm",
        "pipeline_completed": False,
        "pass": False,
        "terminal_status": "pipeline_failed",
        "target_met": False,
        "pipeline_error": str(exc),
        "pipeline_error_only": True,
        "successful_raw_dataset": None,
        "raw_dataset": None,
        "domain_randomization": {
            "successful_episode_count": 0,
            "scene_files_used": 0,
            "objects": {},
            "goals": {},
            "note": "pipeline failed before collection artifacts were available",
        },
        "representative_cap_code_path": None,
        "representative_cap_code": None,
        "discard_failed_episodes": True,
        "keep_failed_raw_dataset": False,
        "failed_raw_dataset_removed": False,
        "failed_raw_dataset_cleanup_reason": None,
        "enabled": bool(task_cfg.enabled),
        "skipped": False,
    }


def _classify_episode_failures(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify episode failures into categories for reporting."""
    counts: dict[str, int] = {
        "codegen_failed": 0,
        "execution_failed": 0,
        "vlm_false": 0,
        "vlm_uncertain": 0,
        "geometry_failed": 0,
        "success_kept": 0,
    }
    for ep in episodes:
        if ep.get("discarded"):
            if not ep.get("codegen_success", True):
                counts["codegen_failed"] += 1
            elif not ep.get("execution_success", True):
                counts["execution_failed"] += 1
            elif ep.get("judge_prediction") == "FALSE":
                counts["vlm_false"] += 1
            elif ep.get("judge_prediction") == "UNCERTAIN":
                counts["vlm_uncertain"] += 1
            else:
                counts["geometry_failed"] += 1
        else:
            counts["success_kept"] += 1
    counts["total_discarded"] = sum(v for k, v in counts.items() if k != "success_kept")
    return counts


def _aggregate_failure_breakdown(task_runs: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate failure breakdown across all tasks."""
    totals: dict[str, int] = {}
    for task in task_runs:
        fb = task.get("failure_breakdown") or {}
        for k, v in fb.items():
            totals[k] = totals.get(k, 0) + int(v)
    return totals


def _build_task_run_report(
    *,
    task_cfg: E2ETaskConfig,
    classification: TaskClassification,
    pipeline_result: dict[str, Any],
    successful_raw_dataset: Path | None,
    filtered_raw_episode_count: int,
    resolved_max_attempts: int,
    pipeline_attempt: int,
    failed_raw_dataset_removed: bool,
    failed_raw_dataset_cleanup_reason: str | None,
    keep_failed_raw_dataset: bool,
) -> dict[str, Any]:
    episodes = list((pipeline_result.get("results") or {}).get("episodes", []))
    representative_episode = _select_representative_episode(episodes)
    representative_code_path = (
        Path(representative_episode["generated_code_path"]).resolve()
        if representative_episode and representative_episode.get("generated_code_path")
        else None
    )
    representative_code = None
    if representative_code_path and representative_code_path.exists():
        representative_code = representative_code_path.read_text()
    env_dir_raw = pipeline_result.get("env_dir")
    representative_env_cfg_path = None
    representative_env_cfg = None
    if env_dir_raw:
        env_cfg_candidate = Path(env_dir_raw).expanduser().resolve() / "env_cfg.py"
        if env_cfg_candidate.exists():
            representative_env_cfg_path = env_cfg_candidate
            representative_env_cfg = env_cfg_candidate.read_text()

    attempted = int(pipeline_result.get("total_episodes", 0))
    successful = int(pipeline_result.get("successful_episodes", 0))
    geometry_successful = int(pipeline_result.get("geometry_successful_episodes", 0))
    vlm_successful = int(pipeline_result.get("vlm_successful_episodes", 0))
    overall_successful = int(pipeline_result.get("overall_successful_episodes", 0))
    failed_attempts = max(0, attempted - successful)
    success_rate = 0.0 if attempted <= 0 else successful / attempted
    collection_results = pipeline_result.get("results", {}) or {}
    failure_category = collection_results.get("failure_category")
    pipeline_error = collection_results.get("pipeline_error")
    pipeline_completed = bool(pipeline_result.get("pipeline_completed", False))

    # Episode-level failure breakdown
    failure_breakdown = _classify_episode_failures(episodes)

    return {
        "task": pipeline_result.get("task", task_cfg.yaml_path.stem),
        "task_name": pipeline_result.get("task", task_cfg.yaml_path.stem),
        "yaml_path": str(task_cfg.yaml_path),
        "requested_ref": task_cfg.requested_ref,
        "robot": str(pipeline_result.get("robot") or _detect_robot_from_task_path(task_cfg.yaml_path)),
        "classification": asdict(classification),
        "requested_demos": task_cfg.demos,
        "max_attempts": resolved_max_attempts,
        "baseline_tag": task_cfg.baseline_tag,
        "pipeline_attempt": pipeline_attempt,
        "attempted_episodes": attempted,
        "successful_episodes": successful,
        "geometry_successful_episodes": geometry_successful,
        "vlm_successful_episodes": vlm_successful,
        "overall_successful_episodes": overall_successful,
        "successful_dataset_episodes": filtered_raw_episode_count,
        "failed_episode_attempts": failed_attempts,
        "success_rate": success_rate,
        "pipeline_completed": pipeline_completed,
        "target_met": bool(pipeline_result.get("target_met", False)),
        "pass": pipeline_completed,
        "terminal_status": _determine_terminal_status(
            pipeline_completed=pipeline_completed,
            failure_category=failure_category,
        ),
        "success_policy": "geometry_or_vlm",
        "output_dir": pipeline_result.get("output_dir"),
        "raw_dataset": pipeline_result.get("raw_dataset"),
        "successful_raw_dataset": str(successful_raw_dataset) if successful_raw_dataset else None,
        "pipeline_error": pipeline_error,
        "failure_category": failure_category,
        "failure_breakdown": failure_breakdown,
        "domain_randomization": _summarize_domain_randomization(episodes),
        "front_video_generated": bool(pipeline_result.get("front_video_generated", False)),
        "front_video_path": pipeline_result.get("front_video_path"),
        "front_video_camera_name": pipeline_result.get("front_video_camera_name"),
        "front_video_episode": pipeline_result.get("front_video_episode"),
        "front_video_success_type": pipeline_result.get("front_video_success_type"),
        "front_video_error": pipeline_result.get("front_video_error"),
        "representative_env_cfg_path": (
            str(representative_env_cfg_path) if representative_env_cfg_path else None
        ),
        "representative_env_cfg": representative_env_cfg,
        "representative_cap_code_path": str(representative_code_path) if representative_code_path else None,
        "representative_cap_code": representative_code,
        "representative_episode": representative_episode,
        "collection_results": collection_results,
        "discard_failed_episodes": bool(pipeline_result.get("discard_failed_episodes", True)),
        "keep_failed_raw_dataset": keep_failed_raw_dataset,
        "failed_raw_dataset_removed": failed_raw_dataset_removed,
        "failed_raw_dataset_cleanup_reason": failed_raw_dataset_cleanup_reason,
        "enabled": bool(task_cfg.enabled),
        "skipped": False,
    }


def _build_skipped_task_result(task_cfg: E2ETaskConfig) -> dict[str, Any]:
    classification = classify_task_yaml(task_cfg.yaml_path)
    task_name = _load_task_name_from_yaml(task_cfg.yaml_path)
    return {
        "task": task_cfg.yaml_path.stem,
        "task_name": task_name,
        "yaml_path": str(task_cfg.yaml_path),
        "requested_ref": task_cfg.requested_ref,
        "robot": _detect_robot_from_task_path(task_cfg.yaml_path),
        "classification": asdict(classification),
        "requested_demos": 0,
        "max_attempts": 0,
        "baseline_tag": task_cfg.baseline_tag,
        "pipeline_attempt": 0,
        "pipeline_attempts_used": 0,
        "pipeline_retry_limit": 0,
        "attempted_episodes": 0,
        "successful_episodes": 0,
        "geometry_successful_episodes": 0,
        "vlm_successful_episodes": 0,
        "overall_successful_episodes": 0,
        "successful_dataset_episodes": 0,
        "failed_episode_attempts": 0,
        "success_rate": 0.0,
        "success_policy": "geometry_or_vlm",
        "pipeline_completed": True,
        "pass": True,
        "terminal_status": "task_disabled",
        "target_met": True,
        "pipeline_error": "Task disabled in batch config",
        "pipeline_error_only": False,
        "successful_raw_dataset": None,
        "raw_dataset": None,
        "output_dir": None,
        "domain_randomization": {},
        "failure_category": "task_disabled",
        "failure_breakdown": {},
        "front_video_generated": False,
        "front_video_path": None,
        "front_video_camera_name": None,
        "front_video_episode": None,
        "front_video_success_type": None,
        "front_video_error": None,
        "representative_env_cfg_path": None,
        "representative_env_cfg": None,
        "representative_cap_code_path": None,
        "representative_cap_code": None,
        "representative_episode": None,
        "collection_results": {},
        "discard_failed_episodes": True,
        "keep_failed_raw_dataset": False,
        "failed_raw_dataset_removed": False,
        "failed_raw_dataset_cleanup_reason": None,
        "enabled": False,
        "skipped": True,
    }


def _build_pipeline_attempt_entry(task_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "pipeline_attempt": int(task_report.get("pipeline_attempt", 1)),
        "terminal_status": task_report.get("terminal_status"),
        "pipeline_completed": bool(task_report.get("pipeline_completed", False)),
        "attempted_episodes": int(task_report.get("attempted_episodes", 0)),
        "successful_episodes": int(task_report.get("successful_episodes", 0)),
        "target_met": bool(task_report.get("target_met", False)),
        "failure_category": task_report.get("failure_category"),
        "pipeline_error": task_report.get("pipeline_error"),
        "output_dir": task_report.get("output_dir"),
        "raw_dataset": task_report.get("raw_dataset"),
    }


def _should_retry_task_pipeline(task_report: dict[str, Any]) -> bool:
    if bool(task_report.get("pipeline_completed", False)):
        return False
    terminal_status = str(task_report.get("terminal_status") or "")
    if terminal_status in {"pipeline_failed", "timed_out"}:
        return True
    return int(task_report.get("attempted_episodes", 0)) <= 0


def _build_robot_dataset(
    *,
    robot_name: str,
    task_records: list[dict[str, Any]],
    batch_cfg: E2EBatchConfig,
    single_robot_run: bool,
    merged_raw_root: Path,
    exported_root: Path,
    preprocessed_root: Path,
    lerobot_root: Path,
    publish_reports_root: Path,
) -> dict[str, Any]:
    local_repo_id = _build_local_repo_id(batch_cfg.hf.dataset_name, robot_name, single_robot_run)
    _reset_robot_dataset_outputs(
        robot_name=robot_name,
        merged_raw_root=merged_raw_root,
        exported_root=exported_root,
        preprocessed_root=preprocessed_root,
        lerobot_root=lerobot_root,
        local_repo_id=local_repo_id,
    )

    merged_raw_dir = merged_raw_root / robot_name / "raw_dataset"
    merge_raw_datasets(
        [record["successful_raw_dataset"] for record in task_records if record.get("successful_raw_dataset")],
        merged_raw_dir,
        source_records=[
            {
                "task": record.get("task_name"),
                "yaml_path": record.get("yaml_path"),
                "output_dir": record.get("output_dir"),
            }
            for record in task_records
            if record.get("successful_raw_dataset")
        ],
    )

    export_dir = None
    if batch_cfg.run.build_export_artifacts:
        export_dir = export_dataset(
            source_path=merged_raw_dir,
            source_type="sim_raw",
            schema=batch_cfg.run.export_schema,
            output_dir=exported_root / robot_name,
            robot_name=robot_name,
            link_images=True,
        )

    preprocess_dir = None
    if batch_cfg.run.build_preprocessed_artifacts and export_dir is not None:
        preprocess_dir = preprocess_exported_dataset(
            export_dir=export_dir,
            output_dir=preprocessed_root / robot_name,
            success_only=batch_cfg.run.preprocess_success_only,
        )

    lerobot_dataset_root = convert_raw_dataset_to_lerobot(
        merged_raw_dir,
        repo_id=local_repo_id,
        output_root=lerobot_root,
    )
    validation_report = check_lerobot_dataset(lerobot_dataset_root, repo_id=local_repo_id)

    if batch_cfg.run.cleanup_intermediates and bool(validation_report.get("pass", False)):
        _cleanup_robot_build_dirs(
            merged_raw_root / robot_name,
            exported_root / robot_name,
            preprocessed_root / robot_name,
        )

    publish_report = None
    publish_report_path = None
    hf_repo_id = _build_hf_repo_id(
        batch_cfg.hf.namespace,
        batch_cfg.hf.dataset_name,
        robot_name,
        single_robot_run,
    )
    if batch_cfg.hf.upload:
        publish_report = publish_lerobot_dataset(
            lerobot_dataset_root,
            repo_id=hf_repo_id,
            private=batch_cfg.hf.private,
            token_env=batch_cfg.hf.token_env,
            local_repo_id=local_repo_id,
        )
        publish_report_path = publish_reports_root / f"{robot_name}.json"
        write_json_report(publish_report, publish_report_path)

    return {
        "robot": robot_name,
        "task_count": len(task_records),
        "tasks": [record.get("task_name") for record in task_records],
        "merged_raw_dataset": str(merged_raw_dir),
        "export_dir": str(export_dir) if export_dir is not None else None,
        "preprocess_dir": str(preprocess_dir) if preprocess_dir is not None else None,
        "local_repo_id": local_repo_id,
        "lerobot_dataset_root": str(lerobot_dataset_root),
        "validation_report": validation_report,
        "hf_repo_id": hf_repo_id,
        "publish_report": publish_report,
        "publish_report_path": str(publish_report_path) if publish_report_path else None,
        "pass": bool(validation_report.get("pass", False))
        and (publish_report is None or bool(publish_report.get("pass", False))),
    }


def _cleanup_robot_build_dirs(*paths: Path) -> None:
    """Remove derived robot-level build directories as soon as LeRobot validation passes."""

    for path in paths:
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
            logger.info("Cleaned derived robot build dir after LeRobot conversion: %s", path)
        except Exception as exc:
            logger.warning("Failed to clean derived robot build dir %s: %s", path, exc)


def _build_batch_summary(report: dict[str, Any], batch_cfg: E2EBatchConfig) -> dict[str, Any]:
    task_runs = list(report.get("task_runs", []))
    skipped_tasks = [task for task in task_runs if task.get("skipped")]
    active_task_runs = [task for task in task_runs if not task.get("skipped")]
    attempted_tasks = len(active_task_runs)
    configured_task_count = len(batch_cfg.tasks)
    successful_tasks = sum(
        1 for task in active_task_runs if task.get("successful_episodes", 0) > 0
    )
    merged_task_count = sum(
        1 for task in active_task_runs if task.get("successful_raw_dataset")
    )
    target_met_task_count = sum(1 for task in active_task_runs if task.get("target_met"))
    pipeline_failed_task_count = sum(
        1 for task in active_task_runs if not task.get("pipeline_completed", False)
    )
    requested_demos = sum(int(task.get("requested_demos", 0)) for task in active_task_runs)
    collected_demos = sum(
        int(task.get("successful_dataset_episodes", 0)) for task in active_task_runs
    )
    total_attempts = sum(int(task.get("attempted_episodes", 0)) for task in active_task_runs)
    total_successes = sum(int(task.get("successful_episodes", 0)) for task in active_task_runs)
    total_failed_attempts = sum(
        int(task.get("failed_episode_attempts", 0)) for task in active_task_runs
    )
    aggregate_success_rate = 0.0 if total_attempts <= 0 else total_successes / total_attempts
    hf_repo_ids = {
        robot: robot_report.get("hf_repo_id")
        for robot, robot_report in sorted(report.get("robot_datasets", {}).items())
    }
    hf_repo_urls = {
        robot: (robot_report.get("publish_report") or {}).get("repo_url")
        for robot, robot_report in sorted(report.get("robot_datasets", {}).items())
        if (robot_report.get("publish_report") or {}).get("repo_url")
    }
    return {
        "summary_guide": {
            "configured_task_count": "Config에 정의된 전체 task 수",
            "skipped_task_count": "Config에 남아 있지만 disabled 되어 실행에서 제외된 task 수",
            "terminal_task_count": "성공/실패와 무관하게 terminal state까지 도달한 task 수",
            "task_with_success_count": "성공 episode가 하나 이상 있는 task 수",
            "target_met_task_count": "요청한 성공 demo 수를 채운 task 수",
            "pipeline_failed_task_count": "collection pipeline이 중간에 실패한 task 수",
            "successful_demo_total": "최종 dataset에 포함된 성공 episode 수",
            "total_failed_attempts": "전체 시도 episode 중 실패로 끝난 episode 수",
            "aggregate_success_rate": "전체 시도 episode 대비 성공 episode 비율 (geometry OR vlm)",
            "pipeline_retry_limit": "task pipeline이 조기 실패했을 때 허용하는 추가 재시도 횟수",
            "pipeline_attempts_used": "각 task에서 실제로 사용한 pipeline 실행 횟수",
            "hf_repo_ids": "최종 업로드된 Hugging Face dataset repo id",
        },
        "batch_completed": bool(report.get("batch_completed", False)),
        "configured_task_count": configured_task_count,
        "skipped_task_count": len(skipped_tasks),
        "terminal_task_count": attempted_tasks,
        "task_count": attempted_tasks,
        "task_with_success_count": successful_tasks,
        "target_met_task_count": target_met_task_count,
        "pipeline_failed_task_count": pipeline_failed_task_count,
        "merged_task_count": merged_task_count,
        "robot_dataset_count": len(report.get("robot_datasets", {})),
        "requested_demos": requested_demos,
        "successful_demo_total": collected_demos,
        "collected_successful_demos": collected_demos,
        "total_episode_attempts": total_attempts,
        "total_successful_attempts": total_successes,
        "total_failed_attempts": total_failed_attempts,
        "aggregate_success_rate": aggregate_success_rate,
        "aggregate_failure_breakdown": _aggregate_failure_breakdown(active_task_runs),
        "failed_raw_dataset_cleanup_count": int((report.get("failed_raw_cleanup") or {}).get("removed_count", 0)),
        "llm_model": batch_cfg.collection.llm_model,
        "vlm_model": batch_cfg.collection.vlm_model,
        "hf_upload": batch_cfg.hf.upload,
        "hf_private": batch_cfg.hf.private,
        "hf_repo_ids": hf_repo_ids,
        "hf_repo_urls": hf_repo_urls,
    }


def _render_markdown_report(report: dict[str, Any], batch_cfg: E2EBatchConfig) -> str:
    summary = report.get("summary", {})
    lines: list[str] = [
        "# E2E Batch Run Report",
        "",
        f"- 실행 시각: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Config: `{report.get('config_path')}`",
        f"- LLM 모델: `{batch_cfg.collection.llm_model}`",
        f"- VLM 모델: `{batch_cfg.collection.vlm_model}`",
        f"- HF 업로드: `{batch_cfg.hf.upload}`",
        f"- Dataset 이름: `{batch_cfg.hf.dataset_name}`",
        f"- Private: `{batch_cfg.hf.private}`",
        "",
        "## Summary",
        "",
        f"- Batch 완료: **{summary.get('batch_completed', False)}**",
        f"- Configured task 수: **{summary.get('configured_task_count', 0)}**",
        f"- Skipped task 수: **{summary.get('skipped_task_count', 0)}**",
        f"- Terminal task 수: **{summary.get('terminal_task_count', 0)}**",
        f"- Pipeline 실패 task 수: **{summary.get('pipeline_failed_task_count', 0)}**",
        f"- Target 달성 task 수: **{summary.get('target_met_task_count', 0)}**",
        f"- Task 수: **{summary.get('task_count', 0)}**",
        f"- 성공 task 수: **{summary.get('task_with_success_count', 0)}**",
        f"- 요청 demo 수: **{summary.get('requested_demos', 0)}**",
        f"- 수집 성공 demo 수: **{summary.get('collected_successful_demos', 0)}**",
        f"- 전체 시도 수: **{summary.get('total_episode_attempts', 0)}**",
        f"- 전체 성공 수: **{summary.get('total_successful_attempts', 0)}**",
        f"- 전체 실패 수: **{summary.get('total_failed_attempts', 0)}**",
        f"- 전체 성공률: **{summary.get('aggregate_success_rate', 0.0):.2%}**",
        f"- 실패 raw 정리 수: **{summary.get('failed_raw_dataset_cleanup_count', 0)}**",
        "",
        "## Summary 설명",
        "",
        "- `Configured task 수`: config에 정의된 전체 task 수",
        "- `Skipped task 수`: disabled 처리되어 실행에서 제외된 task 수",
        "- `Terminal task 수`: 성공/실패와 무관하게 실행이 종료된 task 수",
        "- `성공 task 수`: 성공 episode가 1개 이상 있는 task 수",
        "- `Target 달성 task 수`: 요청한 성공 demo 수를 채운 task 수",
        "- `수집 성공 demo 수`: 최종 dataset에 포함된 성공 episode 수",
        "- `전체 실패 수`: 전체 시도 episode 중 실패로 끝난 episode 수",
        "- `전체 성공률`: 전체 시도 episode 대비 성공 episode 비율",
        "- `성공 판정 기준`: `geometry OR vlm`",
        "- `실패 raw 정리 수`: 실행 시작 시 stale failed raw dataset을 자동 정리한 개수",
        "- `Pipeline retry limit`: task pipeline이 조기 실패했을 때 추가로 다시 띄우는 최대 횟수",
        "- `Pipeline attempts used`: 해당 task에서 실제로 사용한 pipeline 실행 횟수",
        "",
        "## Robot Datasets",
        "",
    ]

    robot_datasets = report.get("robot_datasets", {})
    if not robot_datasets:
        lines.append("- 생성된 robot별 dataset 없음")
    else:
        for robot_name, robot_report in sorted(robot_datasets.items()):
            lines.extend(
                [
                    f"### {robot_name}",
                    "",
                    f"- merged raw: `{robot_report.get('merged_raw_dataset')}`",
                    f"- export: `{robot_report.get('export_dir')}`",
                    f"- preprocess: `{robot_report.get('preprocess_dir')}`",
                    f"- LeRobot: `{robot_report.get('lerobot_dataset_root')}`",
                    f"- local repo id: `{robot_report.get('local_repo_id')}`",
                    f"- HF repo id: `{robot_report.get('hf_repo_id')}`",
                ]
            )
            publish_report = robot_report.get("publish_report") or {}
            if publish_report.get("repo_url"):
                lines.append(f"- HF URL: {publish_report['repo_url']}")
            lines.append("")

    lines.extend(
        [
            "## 대표 성공 영상",
            "",
        ]
    )
    successful_videos = [
        task_report for task_report in report.get("task_runs", [])
        if task_report.get("front_video_path") and not task_report.get("skipped")
    ]
    if not successful_videos:
        lines.append("- 대표 성공 영상이 생성된 task 없음")
    else:
        for task_report in successful_videos:
            lines.extend(
                [
                    f"- `{task_report.get('task_name')}`: `{task_report.get('front_video_path')}`",
                    f"  - camera: `{task_report.get('front_video_camera_name')}` / episode: `{task_report.get('front_video_episode')}` / success_type: `{task_report.get('front_video_success_type')}`",
                ]
            )
    lines.append("")

    lines.extend(
        [
            "## Tasks",
            "",
            "| 분류 | Task | Robot | 요청 demo | 성공 demo | 전체 시도 | 성공률 |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for task_report in report.get("task_runs", []):
        if task_report.get("skipped"):
            continue
        classification = task_report.get("classification", {})
        success_rate = float(task_report.get("success_rate", 0.0))
        lines.append(
            "| {classification} | {task} | {robot} | {requested} | {successful} | {attempted} | {rate:.2%} |".format(
                classification=classification.get("label", "unknown"),
                task=task_report.get("task_name", "unknown"),
                robot=task_report.get("robot", "unknown"),
                requested=int(task_report.get("requested_demos", 0)),
                successful=int(task_report.get("successful_dataset_episodes", 0)),
                attempted=int(task_report.get("attempted_episodes", 0)),
                rate=success_rate,
            )
        )

    for task_report in report.get("task_runs", []):
        classification = task_report.get("classification", {})
        lines.extend(
            [
                "",
                f"### {task_report.get('task_name', 'unknown')}",
                "",
                f"- 분류: `{classification.get('label', 'unknown')}`",
                f"- YAML: `{task_report.get('yaml_path')}`",
                f"- Robot: `{task_report.get('robot')}`",
                f"- baseline tag: `{task_report.get('baseline_tag')}`",
                f"- 요청 demo: `{task_report.get('requested_demos')}`",
                f"- 성공 demo: `{task_report.get('successful_dataset_episodes')}`",
                f"- 전체 시도: `{task_report.get('attempted_episodes')}`",
                f"- 실패 시도: `{task_report.get('failed_episode_attempts', 0)}`",
                f"- task 수집 성공률: `{float(task_report.get('success_rate', 0.0)):.2%}`",
                f"- pipeline retry limit: `{task_report.get('pipeline_retry_limit', 0)}`",
                f"- pipeline attempts used: `{task_report.get('pipeline_attempts_used', 1)}`",
                f"- successful pipeline attempt: `{task_report.get('successful_pipeline_attempt')}`",
                f"- pipeline completed: `{task_report.get('pipeline_completed')}`",
                f"- target met: `{task_report.get('target_met')}`",
                f"- terminal status: `{task_report.get('terminal_status')}`",
                f"- 성공 판정 기준: `{task_report.get('success_policy', 'geometry_or_vlm')}`",
                f"- geometry 성공 수: `{task_report.get('geometry_successful_episodes', 0)}`",
                f"- vlm 성공 수: `{task_report.get('vlm_successful_episodes', 0)}`",
                f"- strict agreement 수: `{task_report.get('overall_successful_episodes', 0)}`",
                f"- raw dataset: `{task_report.get('raw_dataset')}`",
                f"- success-only raw dataset: `{task_report.get('successful_raw_dataset')}`",
                f"- discard failed episodes: `{task_report.get('discard_failed_episodes')}`",
                f"- keep failed raw dataset: `{task_report.get('keep_failed_raw_dataset')}`",
                f"- failed raw removed: `{task_report.get('failed_raw_dataset_removed')}`",
                f"- failed raw cleanup reason: `{task_report.get('failed_raw_dataset_cleanup_reason')}`",
            ]
        )
        fb = task_report.get("failure_breakdown") or {}
        if fb.get("total_discarded", 0) > 0:
            lines.extend(
                [
                    "",
                    "#### 에피소드 실패 분류",
                    "",
                    "| 분류 | 건수 | 설명 |",
                    "|------|------|------|",
                    f"| codegen_failed | {fb.get('codegen_failed', 0)} | LLM 코드 생성 실패 (CaP plan 생성 불가) |",
                    f"| execution_failed | {fb.get('execution_failed', 0)} | CaP 실행 중 오류 (IK 실패, 충돌, 타임아웃 등) |",
                    f"| vlm_false | {fb.get('vlm_false', 0)} | VLM judge가 FALSE 판정 (물리적으로 태스크 미달성) |",
                    f"| vlm_uncertain | {fb.get('vlm_uncertain', 0)} | VLM judge가 UNCERTAIN 판정 |",
                    f"| geometry_failed | {fb.get('geometry_failed', 0)} | Geometry 검증 실패 |",
                    f"| **total_discarded** | **{fb.get('total_discarded', 0)}** | **폐기된 에피소드 총계** |",
                    f"| success_kept | {fb.get('success_kept', 0)} | 성공 에피소드 (보존) |",
                ]
            )
        lines.extend(
            [
                "",
                "#### 대표 아티팩트",
                "",
            ]
        )
        if task_report.get("front_video_path"):
            lines.extend(
                [
                    f"- 대표 성공 영상: `{task_report.get('front_video_path')}`",
                    f"- 대표 성공 영상 카메라: `{task_report.get('front_video_camera_name')}`",
                    f"- 대표 성공 영상 episode: `{task_report.get('front_video_episode')}`",
                    f"- 대표 성공 영상 기준: `{task_report.get('front_video_success_type')}`",
                ]
            )
        else:
            lines.append("- 대표 성공 영상: 없음")

        representative_code = task_report.get("representative_cap_code")
        if representative_code:
            lines.extend(
                [
                    f"- 대표 CaP 코드 경로: `{task_report.get('representative_cap_code_path')}`",
                    "",
                    "<details>",
                    "<summary>대표 CaP 코드</summary>",
                    "",
                    "```python",
                    representative_code.rstrip(),
                    "```",
                    "",
                    "</details>",
                ]
            )
        else:
            lines.append("- 대표 CaP 코드: 없음")

        lines.extend(
            [
                "",
                "#### Domain Randomization Summary",
                "",
            ]
        )
        if task_report.get("failure_category"):
            lines.append(f"- failure category: `{task_report.get('failure_category')}`")
        if task_report.get("pipeline_error"):
            lines.append(f"- pipeline error: `{task_report.get('pipeline_error')}`")
        attempt_history = task_report.get("pipeline_attempt_history") or []
        if attempt_history:
            lines.extend(
                [
                    "- pipeline attempt history:",
                ]
            )
            for attempt in attempt_history:
                lines.append(
                    "  - attempt {attempt}: status=`{status}`, completed=`{completed}`, episodes=`{episodes}`, success=`{success}`, failure_category=`{failure}`".format(
                        attempt=attempt.get("pipeline_attempt"),
                        status=attempt.get("terminal_status"),
                        completed=attempt.get("pipeline_completed"),
                        episodes=attempt.get("attempted_episodes"),
                        success=attempt.get("successful_episodes"),
                        failure=attempt.get("failure_category"),
                    )
                )
        lines.extend(_render_domain_randomization_markdown(task_report.get("domain_randomization", {})))

        representative_env_cfg = task_report.get("representative_env_cfg")
        if representative_env_cfg:
            lines.extend(
                [
                    "",
                    f"- 대표 env cfg 경로: `{task_report.get('representative_env_cfg_path')}`",
                    "",
                    "<details>",
                    "<summary>대표 Env CFG 코드</summary>",
                    "",
                    "```python",
                    representative_env_cfg.rstrip(),
                    "```",
                    "",
                    "</details>",
                ]
            )
        else:
            lines.extend(["", "- 대표 env cfg 코드: 없음"])

    return "\n".join(lines).rstrip() + "\n"


def _render_vlm_audit_markdown(audit: dict[str, Any]) -> str:
    summary = audit.get("summary", {})
    lines = [
        "# VLM Audit Report",
        "",
        f"- Task 수: `{summary.get('task_count', 0)}`",
        f"- Audit case 수: `{summary.get('case_count', 0)}`",
        f"- Judge artifact 보유 task 수: `{summary.get('task_with_judge_artifacts', 0)}`",
    ]
    for task_entry in audit.get("tasks", []):
        lines.extend(
            [
                "",
                f"## {task_entry.get('task_name', 'unknown')}",
                "",
                f"- YAML: `{task_entry.get('yaml_path')}`",
                f"- baseline tag: `{task_entry.get('baseline_tag')}`",
                f"- classification: `{task_entry.get('classification_label')}`",
            ]
        )
        cases = task_entry.get("cases", [])
        if not cases:
            lines.append("- judge artifact available case 없음")
            continue
        for case in cases:
            lines.extend(
                [
                    "",
                    f"### {case.get('case_type')}",
                    "",
                    f"- episode: `{case.get('episode')}`",
                    f"- success: `{case.get('success')}`",
                    f"- geometry/vlm/overall: `{case.get('geometry_success')}` / `{case.get('vlm_success')}` / `{case.get('overall_success')}`",
                    f"- success_basis: `{case.get('success_basis')}`",
                    f"- judge_prediction: `{case.get('judge_prediction')}`",
                    f"- judge_reasoning: `{case.get('judge_reasoning')}`",
                    f"- judge_prompt_path: `{case.get('judge_prompt_path')}`",
                    f"- judge_prompt_context_path: `{case.get('judge_prompt_context_path')}`",
                    f"- judge_raw_response_path: `{case.get('judge_raw_response_path')}`",
                    f"- initial_judge_images: `{case.get('initial_judge_images')}`",
                    f"- final_judge_images: `{case.get('final_judge_images')}`",
                    f"- generated_code_path: `{case.get('generated_code_path')}`",
                    f"- scene_positions_path: `{case.get('scene_positions_path')}`",
                ]
            )
            if case.get("judge_prompt"):
                lines.extend(
                    [
                        "",
                        "<details>",
                        "<summary>Judge Prompt</summary>",
                        "",
                        "```text",
                        str(case.get("judge_prompt", "")).rstrip(),
                        "```",
                        "",
                        "</details>",
                    ]
                )
            if case.get("judge_prompt_context") is not None:
                lines.extend(
                    [
                        "",
                        "<details>",
                        "<summary>Judge Prompt Context</summary>",
                        "",
                        "```json",
                        json.dumps(case.get("judge_prompt_context"), indent=2, ensure_ascii=False),
                        "```",
                        "",
                        "</details>",
                    ]
                )
            if case.get("judge_raw_response"):
                lines.extend(
                    [
                        "",
                        "<details>",
                        "<summary>Judge Raw Response</summary>",
                        "",
                        "```text",
                        str(case.get("judge_raw_response", "")).rstrip(),
                        "```",
                        "",
                        "</details>",
                    ]
                )
    return "\n".join(lines).rstrip() + "\n"


def _render_domain_randomization_markdown(summary: dict[str, Any]) -> list[str]:
    if not summary or not summary.get("successful_episode_count"):
        note = summary.get("note", "successful episode artifact가 없어 집계 불가")
        return [f"- {note}"]

    lines = [
        f"- 성공 episode 수: `{summary.get('successful_episode_count', 0)}`",
        f"- 사용한 scene_positions 수: `{summary.get('scene_files_used', 0)}`",
        "- Object 위치 범위:",
    ]
    objects = summary.get("objects", {})
    if not objects:
        lines.append("  - 없음")
    else:
        for name, ranges in sorted(objects.items()):
            lines.append(
                "  - {name}: x=[{xmin:.4f}, {xmax:.4f}], y=[{ymin:.4f}, {ymax:.4f}], z=[{zmin:.4f}, {zmax:.4f}]".format(
                    name=name,
                    xmin=ranges["x"][0],
                    xmax=ranges["x"][1],
                    ymin=ranges["y"][0],
                    ymax=ranges["y"][1],
                    zmin=ranges["z"][0],
                    zmax=ranges["z"][1],
                )
            )
    lines.append("- Goal 위치 범위:")
    goals = summary.get("goals", {})
    if not goals:
        lines.append("  - 없음")
    else:
        for name, ranges in sorted(goals.items()):
            lines.append(
                "  - {name}: x=[{xmin:.4f}, {xmax:.4f}], y=[{ymin:.4f}, {ymax:.4f}], z=[{zmin:.4f}, {zmax:.4f}]".format(
                    name=name,
                    xmin=ranges["x"][0],
                    xmax=ranges["x"][1],
                    ymin=ranges["y"][0],
                    ymax=ranges["y"][1],
                    zmin=ranges["z"][0],
                    zmax=ranges["z"][1],
                )
            )
    return lines


def _summarize_domain_randomization(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    successful_episodes = [episode for episode in episodes if episode.get("success")]
    if not successful_episodes:
        return {
            "successful_episode_count": 0,
            "scene_files_used": 0,
            "objects": {},
            "goals": {},
            "note": "no successful episode artifacts",
        }

    object_samples: dict[str, list[list[float]]] = defaultdict(list)
    goal_samples: dict[str, list[list[float]]] = defaultdict(list)
    scene_files_used = 0

    for episode in successful_episodes:
        scene_path_raw = episode.get("scene_positions_path")
        if not scene_path_raw:
            continue
        scene_path = Path(scene_path_raw)
        if not scene_path.exists():
            continue
        with open(scene_path) as f:
            scene_positions = json.load(f)
        scene_files_used += 1

        for object_name, entry in scene_positions.items():
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("task_role", ""))
            if role in OBJECT_TASK_ROLES and _is_xyz(entry.get("position")):
                object_samples[object_name].append(list(entry["position"]))
            if role in GOAL_TASK_ROLES and _is_xyz(entry.get("position")):
                goal_samples[f"{object_name}.position"].append(list(entry["position"]))
            for key in GOAL_POSITION_KEYS:
                if _is_xyz(entry.get(key)):
                    goal_samples[f"{object_name}.{key}"].append(list(entry[key]))

    return {
        "successful_episode_count": len(successful_episodes),
        "scene_files_used": scene_files_used,
        "objects": {
            name: _range_summary(samples) for name, samples in sorted(object_samples.items())
        },
        "goals": {
            name: _range_summary(samples) for name, samples in sorted(goal_samples.items())
        },
    }


def _range_summary(samples: list[list[float]]) -> dict[str, list[float]]:
    xs = [float(sample[0]) for sample in samples]
    ys = [float(sample[1]) for sample in samples]
    zs = [float(sample[2]) for sample in samples]
    return {
        "count": len(samples),
        "x": [min(xs), max(xs)],
        "y": [min(ys), max(ys)],
        "z": [min(zs), max(zs)],
    }


def _select_representative_episode(episodes: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not episodes:
        return None

    def _episode_order(item: dict[str, Any]) -> tuple[int, str]:
        raw = item.get("episode")
        return (int(raw), "") if raw is not None else (10**9, "")

    overall_successful = sorted(
        [episode for episode in episodes if episode.get("overall_success")],
        key=_episode_order,
    )
    if overall_successful:
        return overall_successful[0]
    successful = sorted(
        [episode for episode in episodes if episode.get("success")],
        key=_episode_order,
    )
    if successful:
        return successful[0]
    return None


def _determine_terminal_status(*, pipeline_completed: bool, failure_category: str | None) -> str:
    if pipeline_completed:
        return "pipeline_completed"
    if failure_category:
        return "timed_out" if "timeout" in str(failure_category).lower() else "pipeline_failed"
    return "pipeline_failed"


def _build_vlm_audit(report: dict[str, Any]) -> dict[str, Any]:
    task_entries: list[dict[str, Any]] = []
    case_count = 0
    task_with_judge_artifacts = 0
    for task_report in report.get("task_runs", []):
        task_cases = _select_vlm_audit_cases(task_report)
        if task_cases:
            task_with_judge_artifacts += 1
            case_count += len(task_cases)
        task_entries.append(
            {
                "task_name": task_report.get("task_name"),
                "yaml_path": task_report.get("yaml_path"),
                "baseline_tag": task_report.get("baseline_tag"),
                "classification_label": (task_report.get("classification") or {}).get("label"),
                "cases": task_cases,
            }
        )
    return {
        "summary": {
            "task_count": len(task_entries),
            "case_count": case_count,
            "task_with_judge_artifacts": task_with_judge_artifacts,
        },
        "tasks": task_entries,
    }


def _select_vlm_audit_cases(task_report: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = list((task_report.get("collection_results") or {}).get("episodes", []))
    if not episodes:
        return []

    def _episode_order(item: dict[str, Any]) -> tuple[int, str]:
        raw = item.get("episode")
        return (int(raw), "") if raw is not None else (10**9, "")

    judge_candidates = sorted(
        [
            episode
            for episode in episodes
            if episode.get("judge_available")
            and (
                episode.get("judge_prompt_path")
                or episode.get("judge_prediction")
                or episode.get("judge_reasoning")
            )
        ],
        key=_episode_order,
    )
    if not judge_candidates:
        return []

    success_episode = next(
        (episode for episode in judge_candidates if episode.get("overall_success")),
        None,
    )
    if success_episode is None:
        success_episode = next((episode for episode in judge_candidates if episode.get("success")), None)

    negative_episode = next(
        (
            episode
            for episode in judge_candidates
            if episode.get("geometry_success") != episode.get("vlm_success")
        ),
        None,
    )
    if negative_episode is None:
        negative_episode = next((episode for episode in judge_candidates if not episode.get("success")), None)

    selected_cases: list[dict[str, Any]] = []
    seen_episodes: set[int] = set()
    for case_type, episode in (
        ("representative_success", success_episode),
        ("representative_negative_or_mismatch", negative_episode),
    ):
        if not episode:
            continue
        episode_idx = int(episode.get("episode", -1))
        if episode_idx in seen_episodes:
            continue
        seen_episodes.add(episode_idx)
        selected_cases.append(
            {
                "case_type": case_type,
                "episode": episode.get("episode"),
                "success": bool(episode.get("success", False)),
                "geometry_success": bool(episode.get("geometry_success", False)),
                "vlm_success": bool(episode.get("vlm_success", False)),
                "overall_success": bool(episode.get("overall_success", False)),
                "success_basis": episode.get("success_basis"),
                "judge_prediction": episode.get("judge_prediction"),
                "judge_reasoning": episode.get("judge_reasoning"),
                "judge_prompt_path": episode.get("judge_prompt_path"),
                "judge_prompt": _read_artifact_text(episode.get("judge_prompt_path")),
                "judge_prompt_context_path": episode.get("judge_prompt_context_path"),
                "judge_prompt_context": _read_artifact_json(
                    episode.get("judge_prompt_context_path")
                ),
                "judge_raw_response_path": episode.get("judge_raw_response_path"),
                "judge_raw_response": _read_artifact_text(
                    episode.get("judge_raw_response_path")
                ),
                "initial_judge_images": episode.get("initial_judge_images"),
                "final_judge_images": episode.get("final_judge_images"),
                "generated_code_path": episode.get("generated_code_path"),
                "scene_positions_path": episode.get("scene_positions_path"),
            }
        )

    return selected_cases


def _group_successful_task_records(task_records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in task_records:
        grouped[str(record["robot"])].append(record)
    return dict(grouped)


def _build_local_repo_id(dataset_name: str, robot_name: str, single_robot_run: bool) -> str:
    suffix = "" if single_robot_run else f"-{robot_name}"
    return f"local/{dataset_name}{suffix}"


def _build_hf_repo_id(
    namespace: str,
    dataset_name: str,
    robot_name: str,
    single_robot_run: bool,
) -> str:
    suffix = "" if single_robot_run else f"-{robot_name}"
    return f"{namespace}/{dataset_name}{suffix}" if namespace else f"{dataset_name}{suffix}"


def _reset_robot_dataset_outputs(
    *,
    robot_name: str,
    merged_raw_root: Path,
    exported_root: Path,
    preprocessed_root: Path,
    lerobot_root: Path,
    local_repo_id: str,
) -> None:
    """Remove stale derived robot-level outputs before rebuilding merged datasets.

    These directories are fully derived from `successful_raw_dataset` task outputs, so
    clearing them makes resume idempotent after partial merge/export/LeRobot failures.
    """

    derived_paths = [
        merged_raw_root / robot_name,
        exported_root / robot_name,
        preprocessed_root / robot_name,
        lerobot_root / Path(local_repo_id),
    ]
    for path in derived_paths:
        if not path.exists():
            continue
        shutil.rmtree(path)
        logger.info("Removed stale derived dataset output before rebuild: %s", path)


def _cleanup_intermediate_dirs(
    *,
    successful_raw_root: Path,
    merged_raw_root: Path,
    exported_root: Path,
) -> None:
    """Remove intermediate directories after LeRobot conversion is complete.

    Keeps: task_runs/ (original raw), lerobot/ (final output), preprocessed/, reports/
    Removes: successful_raw/, merged_raw/, exported/ (redundant copies)
    """
    import shutil

    for path in (successful_raw_root, merged_raw_root, exported_root):
        if path.exists():
            try:
                size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
                shutil.rmtree(path)
                logger.info("Cleaned up intermediate dir %s (freed ~%.0f MB)", path.name, size_mb)
            except Exception as exc:
                logger.warning("Failed to clean up %s: %s", path, exc)


def _cleanup_failed_task_raw_datasets(
    *,
    task_reports_root: Path,
    enabled: bool,
) -> dict[str, Any]:
    """Remove stale failed raw_dataset directories referenced by failed task reports."""

    summary = {
        "enabled": bool(enabled),
        "removed_count": 0,
        "removed_paths": [],
    }
    if not enabled or not task_reports_root.exists():
        return summary

    pattern = re.compile(r"metadata\.json not found in raw dataset: (?P<path>.+)$")
    removed_paths: list[str] = []
    for report_path in sorted(task_reports_root.glob("*.json")):
        try:
            with open(report_path) as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("pass") or int(report.get("successful_dataset_episodes", 0)) > 0:
            continue

        candidates: list[Path] = []
        raw_dataset = report.get("raw_dataset")
        if raw_dataset:
            candidates.append(Path(raw_dataset).expanduser().resolve())
        output_dir = report.get("output_dir")
        if output_dir:
            candidates.append(Path(output_dir).expanduser().resolve() / "raw_dataset")
        pipeline_error = str(report.get("pipeline_error") or "")
        match = pattern.search(pipeline_error)
        if match:
            candidates.append(Path(match.group("path")).expanduser().resolve())

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if not candidate.exists():
                continue
            import shutil

            shutil.rmtree(candidate)
            removed_paths.append(str(candidate))
            logger.info("Removed stale failed raw dataset: %s", candidate)

    summary["removed_count"] = len(removed_paths)
    summary["removed_paths"] = removed_paths
    return summary


def _cleanup_gpu_between_tasks() -> None:
    """Best-effort GPU memory cleanup between tasks."""
    gc.collect()

    # Run torch.cuda.empty_cache() in a subprocess
    try:
        subprocess.run(
            [
                "python3",
                "-c",
                "import torch; torch.cuda.empty_cache(); torch.cuda.ipc_collect()",
            ],
            timeout=30,
            capture_output=True,
        )
    except Exception:
        pass

    # Check GPU memory via nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    used_mb = int(parts[0])
                    logger.info(
                        "GPU memory: %d MB used, %s MB free", used_mb, parts[1]
                    )
                    if used_mb > 2000:
                        logger.warning(
                            "GPU memory still high (%d MB) — attempting orphan cleanup",
                            used_mb,
                        )
                        _kill_orphan_gpu_processes()
    except Exception:
        pass

    # Brief pause for GPU driver to reclaim memory
    time.sleep(3)


def _kill_orphan_gpu_processes() -> None:
    """Kill GPU-using processes owned by the current user."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            timeout=10,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return
        my_uid = os.getuid()
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                pid = int(line)
                # Only kill processes owned by us
                stat_path = Path(f"/proc/{pid}/status")
                if stat_path.exists():
                    for sline in stat_path.read_text().split("\n"):
                        if sline.startswith("Uid:"):
                            uid = int(sline.split()[1])
                            if uid == my_uid:
                                logger.info("Killing orphan GPU process PID %d", pid)
                                os.kill(pid, signal.SIGKILL)
                            break
            except (ValueError, ProcessLookupError, PermissionError):
                continue
    except Exception:
        pass


def _load_resume_task_results(
    task_reports_root: Path,
    tasks: tuple[E2ETaskConfig, ...],
) -> dict[int, dict[str, Any]]:
    """Load successful completed task reports that can be safely reused on resume."""

    completed: dict[int, dict[str, Any]] = {}
    for task_idx, task_cfg in enumerate(tasks):
        task_slug = _slugify_task(task_cfg.yaml_path)
        report_path = task_reports_root / f"{task_idx:03d}_{task_slug}.json"
        if not report_path.exists():
            continue
        try:
            with open(report_path) as f:
                report = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Resume: could not read %s, will re-run that task", report_path)
            continue
        if bool(report.get("skipped", False)):
            completed[task_idx] = report
            continue
        if not (bool(report.get("pass")) and bool(report.get("target_met"))):
            continue
        _normalize_resume_task_result(report)
        raw_dataset_path = report.get("successful_raw_dataset") or report.get("raw_dataset")
        if raw_dataset_path and Path(raw_dataset_path).exists():
            completed[task_idx] = report
        else:
            logger.warning(
                "Resume: task %d (%s) is marked complete but dataset is missing at %s, will re-run",
                task_idx + 1,
                task_slug,
                raw_dataset_path,
            )
    return completed


def _normalize_resume_task_result(task_result: dict[str, Any]) -> dict[str, Any]:
    """Repair resume task paths after cleanup so downstream merge can reuse originals."""

    successful_raw_dataset = task_result.get("successful_raw_dataset")
    raw_dataset = task_result.get("raw_dataset")
    if successful_raw_dataset and Path(successful_raw_dataset).exists():
        return task_result
    if raw_dataset and Path(raw_dataset).exists():
        task_result["successful_raw_dataset"] = raw_dataset
    return task_result


def _resolve_batch_root(output_root: str | None) -> Path:
    if output_root:
        path = Path(output_root).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (PROJECT_ROOT / "outputs" / "e2e_batches" / timestamp).resolve()


def _ensure_batch_environment(batch_cfg: E2EBatchConfig) -> None:
    missing: list[str] = []
    for env_var in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"):
        if not os.environ.get(env_var):
            missing.append(env_var)

    if batch_cfg.hf.upload and not os.environ.get(batch_cfg.hf.token_env):
        missing.append(batch_cfg.hf.token_env)

    if missing:
        missing_list = ", ".join(sorted(set(missing)))
        raise EnvironmentError(
            "run_e2e_batch is missing required environment variables: "
            f"{missing_list}"
        )


def _write_config_snapshot(batch_cfg: E2EBatchConfig, destination: Path) -> None:
    snapshot = _serialize_for_yaml(batch_cfg)
    with open(destination, "w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False, allow_unicode=True)


def _serialize_for_yaml(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serialize_for_yaml(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize_for_yaml(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_yaml(item) for item in value]
    return value


def _slugify_task(task_path: Path) -> str:
    try:
        rel = task_path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = task_path
    return str(rel).replace("/", "_").replace(".yaml", "")


def _detect_robot_from_task_path(task_path: Path) -> str:
    try:
        parts = task_path.relative_to(PROJECT_ROOT).parts
        if parts and parts[0] == "tasks" and len(parts) >= 3:
            return parts[1]
    except ValueError:
        pass
    return "unknown"


def _is_xyz(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(coord, (int, float)) for coord in value)
    )


def _read_artifact_text(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    return path.read_text()


def _read_artifact_json(path_str: str | None) -> Any:
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
