"""Tests for the config-driven E2E batch orchestrator."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from src.data_collection.config import load_robot_config
from src.data_collection.e2e_orchestrator import load_e2e_batch_config, run_e2e_batch
from src.data_collection.raw_dataset_merge import filter_raw_dataset_episodes, load_raw_dataset_metadata
from src.data_collection.sim_recorder import SimRecorder


def _record_episode(recorder: SimRecorder, robot_name: str, *, success: bool) -> None:
    robot_cfg = load_robot_config(robot_name)
    zeros = np.zeros(robot_cfg.total_dofs, dtype=np.float32)
    recorder.start_episode(f"{robot_name} task")
    recorder.record_step(
        state=zeros,
        action=zeros,
        skill_label="noop",
        skill_type="noop",
        goal_joint=zeros,
        goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
        tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
        tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
        gripper_state=np.zeros(1, dtype=np.float32),
    )
    recorder.end_episode(success=success)


def _create_mixed_success_raw_dataset(base_dir: str | Path, robot_name: str) -> Path:
    robot_cfg = load_robot_config(robot_name)
    recorder = SimRecorder(robot_cfg, str(base_dir), dataset_name="raw_dataset", fps=20)
    _record_episode(recorder, robot_name, success=False)
    _record_episode(recorder, robot_name, success=True)
    return Path(recorder.finalize())


class E2EBatchTests(unittest.TestCase):
    def test_run_e2e_batch_fails_fast_when_required_env_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "batch.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "run": {
                            "output_root": str(Path(tmpdir) / "outputs"),
                        },
                        "collection": {
                            "llm_model": "gpt-5",
                            "vlm_model": "gpt-5",
                            "use_vlm_judge": True,
                        },
                        "hf": {
                            "upload": True,
                            "namespace": "vpraise00",
                            "dataset_name": "env-check",
                            "private": True,
                            "token_env": "HF_TOKEN",
                        },
                        "tasks": [
                            {"yaml": "tasks/franka/stack/franka_stack.yaml", "demos": 1},
                        ],
                    },
                    sort_keys=False,
                    allow_unicode=True,
                )
            )

            with patch.dict(os.environ, {}, clear=True):
                with patch(
                    "src.data_collection.e2e_orchestrator.load_dotenv",
                    return_value=None,
                ), patch(
                    "src.data_collection.e2e_orchestrator.ensure_lerobot_available",
                    return_value=None,
                ), patch(
                    "src.data_collection.e2e_orchestrator.ensure_huggingface_hub_available",
                    return_value=None,
                ):
                    with self.assertRaises(EnvironmentError) as ctx:
                        run_e2e_batch(config_path)

            message = str(ctx.exception)
            self.assertIn("AZURE_OPENAI_API_KEY", message)
            self.assertIn("AZURE_OPENAI_BASE_URL", message)
            self.assertIn("HF_TOKEN", message)

    def test_filter_raw_dataset_episodes_keeps_only_successes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dataset = _create_mixed_success_raw_dataset(Path(tmpdir) / "source", "franka")
            filtered = filter_raw_dataset_episodes(
                raw_dataset,
                Path(tmpdir) / "filtered" / "raw_dataset",
                success_only=True,
            )

            metadata = load_raw_dataset_metadata(filtered)
            self.assertEqual(metadata["total_episodes"], 1)
            self.assertEqual(metadata["successful_episodes"], 1)
            self.assertEqual(metadata["episodes"][0]["index"], 0)
            self.assertEqual(metadata["episodes"][0]["source_episode_index"], 1)
            self.assertTrue((filtered / "episodes" / "episode_000000" / "states.npy").exists())

    def test_load_e2e_batch_config_resolves_task_name_and_yaml_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "batch.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "hf": {"dataset_name": "unit-test"},
                        "collection": {
                            "vlm_model": "gpt-5",
                            "execution_timeout": 3600,
                            "task_retry_limit": 2,
                        },
                        "tasks": [
                            {
                                "task": "FrankaStack",
                                "demos": 1,
                                "baseline_tag": "known_working",
                                "task_retry_limit": 3,
                            },
                            {"yaml": "tasks/franka/assembly/franka_assembling_kits.yaml", "demos": 2},
                        ],
                    },
                    sort_keys=False,
                    allow_unicode=True,
                )
            )

            config = load_e2e_batch_config(config_path)

            self.assertEqual(len(config.tasks), 2)
            self.assertTrue(str(config.tasks[0].yaml_path).endswith("tasks/franka/stack/franka_stack.yaml"))
            self.assertEqual(config.tasks[0].baseline_tag, "known_working")
            self.assertEqual(config.tasks[0].task_retry_limit, 3)
            self.assertEqual(config.collection.vlm_model, "gpt-5")
            self.assertEqual(config.collection.execution_timeout, 3600)
            self.assertEqual(config.collection.task_retry_limit, 2)
            self.assertTrue(
                str(config.tasks[1].yaml_path).endswith(
                    "tasks/franka/assembly/franka_assembling_kits.yaml"
                )
            )

    def test_run_e2e_batch_retries_pipeline_failures_before_task_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            franka_raw = _create_mixed_success_raw_dataset(tmp_path / "franka_retry_run", "franka")
            success_code = tmp_path / "franka_retry_success.py"
            success_code.write_text("print('retry success')\n")
            front_video = tmp_path / "franka_retry_front_success.mp4"
            front_video.write_text("video")
            env_dir = tmp_path / "retry_env"
            env_dir.mkdir(parents=True, exist_ok=True)
            (env_dir / "env_cfg.py").write_text("class FrankaStackEnvCfg:\n    pass\n")
            scene = tmp_path / "franka_retry_scene.json"
            scene.write_text(
                json.dumps(
                    {
                        "cube": {
                            "position": [0.45, -0.10, 0.05],
                            "task_role": "goal_subject",
                        },
                        "target_marker": {
                            "position": [0.55, 0.12, 0.02],
                            "task_role": "placement_target",
                            "target_position": [0.55, 0.12, 0.02],
                        },
                    }
                )
            )

            config_path = tmp_path / "batch_retry.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "run": {
                            "output_root": str(tmp_path / "outputs"),
                            "continue_on_failure": True,
                            "export_schema": "adc_compatible",
                            "preprocess_success_only": True,
                            "generate_report": True,
                        },
                        "collection": {
                            "llm_model": "gpt-5",
                            "vlm_model": "gpt-5",
                            "use_vlm_judge": True,
                            "task_retry_limit": 2,
                        },
                        "hf": {
                            "upload": False,
                            "dataset_name": "retry-unit",
                            "private": True,
                        },
                        "tasks": [
                            {
                                "yaml": "tasks/franka/stack/franka_stack.yaml",
                                "demos": 1,
                                "task_retry_limit": 2,
                                "baseline_tag": "retry_check",
                            }
                        ],
                    },
                    sort_keys=False,
                    allow_unicode=True,
                )
            )

            results_queue = [
                {
                    "success": False,
                    "pipeline_completed": False,
                    "target_met": False,
                    "successful_episodes": 0,
                    "geometry_successful_episodes": 0,
                    "vlm_successful_episodes": 0,
                    "overall_successful_episodes": 0,
                    "total_episodes": 0,
                    "output_dir": str(tmp_path / "franka_retry_output_attempt1"),
                    "env_dir": str(env_dir),
                    "raw_dataset": None,
                    "robot": "franka",
                    "task": "FrankaStack",
                    "results": {
                        "episodes": [],
                        "failure_category": "startup_hang",
                        "pipeline_error": "boot failed",
                    },
                },
                {
                    "success": True,
                    "pipeline_completed": True,
                    "target_met": True,
                    "successful_episodes": 1,
                    "geometry_successful_episodes": 1,
                    "vlm_successful_episodes": 1,
                    "overall_successful_episodes": 1,
                    "total_episodes": 1,
                    "output_dir": str(tmp_path / "franka_retry_output_attempt2"),
                    "env_dir": str(env_dir),
                    "raw_dataset": str(franka_raw),
                    "front_video_generated": True,
                    "front_video_path": str(front_video),
                    "front_video_camera_name": "front",
                    "front_video_episode": 0,
                    "front_video_success_type": "overall",
                    "robot": "franka",
                    "task": "FrankaStack",
                    "results": {
                        "episodes": [
                            {
                                "episode": 0,
                                "success": True,
                                "geometry_success": True,
                                "vlm_success": True,
                                "overall_success": True,
                                "success_basis": "overall",
                                "generated_code_path": str(success_code),
                                "scene_positions_path": str(scene),
                            }
                        ]
                    },
                },
            ]

            class _FakeRetryPipeline:
                def __init__(self, yaml_path, config=None, env_dir=None, auto_convert_to_lerobot=True):
                    self.yaml_path = yaml_path

                def run(self):
                    result = dict(results_queue.pop(0))
                    result["yaml_path"] = self.yaml_path
                    return result

            def _fake_export_dataset(*, source_path, source_type, schema, output_dir, robot_name=None, link_images=True):
                export_dir = Path(output_dir) / f"{robot_name}_export"
                export_dir.mkdir(parents=True, exist_ok=True)
                (export_dir / "manifest.json").write_text(json.dumps({"schema": schema}))
                return export_dir

            def _fake_preprocess_exported_dataset(*, export_dir, output_dir, success_only=False, **kwargs):
                preprocess_dir = Path(output_dir) / "preprocessed"
                preprocess_dir.mkdir(parents=True, exist_ok=True)
                (preprocess_dir / "manifest.json").write_text(json.dumps({"success_only": success_only}))
                return preprocess_dir

            def _fake_convert(raw_dataset_dir, repo_id="local/sim_dataset", output_root=None):
                dataset_root = Path(output_root) / Path(repo_id)
                dataset_root.mkdir(parents=True, exist_ok=True)
                return dataset_root

            with patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_API_KEY": "test-key",
                    "AZURE_OPENAI_BASE_URL": "https://example.invalid",
                },
                clear=False,
            ):
                with (
                    patch("src.data_collection.e2e_orchestrator.load_dotenv", return_value=None),
                    patch("src.data_collection.e2e_orchestrator.ensure_lerobot_available"),
                    patch("src.data_collection.e2e_orchestrator.DataCollectionPipeline", _FakeRetryPipeline),
                    patch("src.data_collection.e2e_orchestrator.export_dataset", side_effect=_fake_export_dataset),
                    patch(
                        "src.data_collection.e2e_orchestrator.preprocess_exported_dataset",
                        side_effect=_fake_preprocess_exported_dataset,
                    ),
                    patch(
                        "src.data_collection.e2e_orchestrator.convert_raw_dataset_to_lerobot",
                        side_effect=_fake_convert,
                    ),
                    patch(
                        "src.data_collection.e2e_orchestrator.check_lerobot_dataset",
                        side_effect=lambda dataset_root, repo_id=None: {
                            "pass": True,
                            "dataset_root": str(dataset_root),
                            "repo_id": repo_id,
                        },
                    ),
                ):
                    report = run_e2e_batch(config_path)

            task_report = report["task_runs"][0]
            self.assertTrue(report["pass"])
            self.assertEqual(task_report["pipeline_retry_limit"], 2)
            self.assertEqual(task_report["pipeline_attempts_used"], 2)
            self.assertTrue(task_report["pipeline_retried"])
            self.assertEqual(task_report["successful_pipeline_attempt"], 2)
            self.assertEqual(len(task_report["pipeline_attempt_history"]), 2)
            self.assertEqual(
                task_report["pipeline_attempt_history"][0]["failure_category"],
                "startup_hang",
            )
            self.assertEqual(
                task_report["pipeline_attempt_history"][0]["terminal_status"],
                "pipeline_failed",
            )
            self.assertEqual(task_report["successful_dataset_episodes"], 1)
            markdown_report = Path(report["markdown_report_path"]).read_text()
            self.assertIn("pipeline attempts used", markdown_report)
            self.assertIn("attempt 1", markdown_report)

    def test_run_e2e_batch_builds_reports_and_robot_split_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            franka_raw = _create_mixed_success_raw_dataset(tmp_path / "franka_run", "franka")
            openarm_raw = _create_mixed_success_raw_dataset(tmp_path / "openarm_run", "openarm")

            franka_success_code = tmp_path / "franka_success.py"
            franka_success_code.write_text("print('franka success')\n")
            franka_video = tmp_path / "franka_front_success.mp4"
            franka_video.write_text("video")
            franka_env_dir = tmp_path / "franka_env"
            franka_env_dir.mkdir(parents=True, exist_ok=True)
            (franka_env_dir / "env_cfg.py").write_text("class FrankaStackEnvCfg:\n    pass\n")
            franka_scene = tmp_path / "franka_scene.json"
            franka_judge_prompt = tmp_path / "franka_judge_prompt.txt"
            franka_judge_prompt.write_text("judge prompt for franka")
            franka_judge_context = tmp_path / "franka_judge_context.json"
            franka_judge_context.write_text(json.dumps({"task": "FrankaStack"}))
            franka_judge_raw = tmp_path / "franka_judge_raw.txt"
            franka_judge_raw.write_text("PREDICTION: TRUE\nREASONING: stacked")
            franka_scene.write_text(
                json.dumps(
                    {
                        "cube": {
                            "position": [0.45, -0.12, 0.05],
                            "task_role": "goal_subject",
                        },
                        "target_marker": {
                            "position": [0.55, 0.10, 0.02],
                            "task_role": "placement_target",
                            "target_position": [0.55, 0.10, 0.02],
                        },
                    }
                )
            )

            openarm_success_code = tmp_path / "openarm_success.py"
            openarm_success_code.write_text("print('openarm success')\n")
            openarm_video = tmp_path / "openarm_front_success.mp4"
            openarm_video.write_text("video")
            openarm_env_dir = tmp_path / "openarm_env"
            openarm_env_dir.mkdir(parents=True, exist_ok=True)
            (openarm_env_dir / "env_cfg.py").write_text("class OpenArmPlugChargerEnvCfg:\n    pass\n")
            openarm_scene = tmp_path / "openarm_scene.json"
            openarm_judge_prompt = tmp_path / "openarm_judge_prompt.txt"
            openarm_judge_prompt.write_text("judge prompt for openarm")
            openarm_judge_context = tmp_path / "openarm_judge_context.json"
            openarm_judge_context.write_text(json.dumps({"task": "OpenArmPlugCharger"}))
            openarm_judge_raw = tmp_path / "openarm_judge_raw.txt"
            openarm_judge_raw.write_text("PREDICTION: TRUE\nREASONING: inserted")
            openarm_scene.write_text(
                json.dumps(
                    {
                        "charger_base": {
                            "position": [0.33, -0.08, 0.02],
                            "task_role": "goal_subject",
                        },
                        "receptacle": {
                            "position": [0.41, 0.12, 0.04],
                            "task_role": "insertion_target",
                            "entry_position": [0.41, 0.12, 0.05],
                        },
                    }
                )
            )

            task_a = Path("tasks/franka/stack/franka_stack.yaml").resolve()
            task_b = Path("tasks/openarm/assembly/openarm_plug_charger.yaml").resolve()
            config_path = tmp_path / "batch.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "run": {
                            "output_root": str(tmp_path / "outputs"),
                            "continue_on_failure": True,
                            "export_schema": "adc_compatible",
                            "preprocess_success_only": True,
                            "generate_report": True,
                        },
                        "collection": {
                            "llm_model": "gpt-5-mini",
                            "vlm_model": "gpt-5",
                            "use_vlm_judge": True,
                            "gui": False,
                            "execution_timeout": 3600,
                            "default_max_attempts_multiplier": 5,
                        },
                        "hf": {
                            "upload": False,
                            "dataset_name": "e2e-unit",
                            "private": True,
                        },
                        "tasks": [
                            {"yaml": str(task_a), "demos": 1, "baseline_tag": "known_working"},
                            {"yaml": str(task_b), "demos": 1, "baseline_tag": "known_working"},
                        ],
                    },
                    sort_keys=False,
                    allow_unicode=True,
                )
            )

            results_queue = [
                {
                    "success": True,
                    "pipeline_completed": True,
                    "target_met": True,
                    "successful_episodes": 1,
                    "geometry_successful_episodes": 0,
                    "vlm_successful_episodes": 1,
                    "overall_successful_episodes": 0,
                    "total_episodes": 2,
                    "output_dir": str(tmp_path / "franka_output"),
                    "env_dir": str(franka_env_dir),
                    "raw_dataset": str(franka_raw),
                    "front_video_generated": True,
                    "front_video_path": str(franka_video),
                    "front_video_camera_name": "front",
                    "front_video_episode": 1,
                    "front_video_success_type": "vlm_only",
                    "robot": "franka",
                    "task": "FrankaStack",
                    "results": {
                        "episodes": [
                            {"episode": 0, "success": False, "overall_success": False},
                            {
                                "episode": 1,
                                "success": True,
                                "geometry_success": False,
                                "vlm_success": True,
                                "overall_success": False,
                                "success_basis": "vlm_only",
                                "generated_code_path": str(franka_success_code),
                                "scene_positions_path": str(franka_scene),
                                "judge_available": True,
                                "judge_prediction": "TRUE",
                                "judge_reasoning": "stacked",
                                "judge_prompt_path": str(franka_judge_prompt),
                                "judge_prompt_context_path": str(franka_judge_context),
                                "judge_raw_response_path": str(franka_judge_raw),
                                "initial_judge_images": ["franka_initial_front.png"],
                                "final_judge_images": ["franka_final_front.png"],
                            },
                        ]
                    },
                },
                {
                    "success": True,
                    "pipeline_completed": True,
                    "target_met": True,
                    "successful_episodes": 1,
                    "geometry_successful_episodes": 1,
                    "vlm_successful_episodes": 1,
                    "overall_successful_episodes": 1,
                    "total_episodes": 2,
                    "output_dir": str(tmp_path / "openarm_output"),
                    "env_dir": str(openarm_env_dir),
                    "raw_dataset": str(openarm_raw),
                    "front_video_generated": True,
                    "front_video_path": str(openarm_video),
                    "front_video_camera_name": "front",
                    "front_video_episode": 1,
                    "front_video_success_type": "overall",
                    "robot": "openarm",
                    "task": "OpenArmPlugCharger",
                    "results": {
                        "episodes": [
                            {"episode": 0, "success": False, "overall_success": False},
                            {
                                "episode": 1,
                                "success": True,
                                "geometry_success": True,
                                "vlm_success": True,
                                "overall_success": True,
                                "success_basis": "overall",
                                "generated_code_path": str(openarm_success_code),
                                "scene_positions_path": str(openarm_scene),
                                "judge_available": True,
                                "judge_prediction": "TRUE",
                                "judge_reasoning": "inserted",
                                "judge_prompt_path": str(openarm_judge_prompt),
                                "judge_prompt_context_path": str(openarm_judge_context),
                                "judge_raw_response_path": str(openarm_judge_raw),
                                "initial_judge_images": ["openarm_initial_front.png"],
                                "final_judge_images": ["openarm_final_front.png"],
                            },
                        ]
                    },
                },
            ]

            class _FakePipeline:
                def __init__(self, yaml_path, config=None, env_dir=None, auto_convert_to_lerobot=True):
                    self.yaml_path = yaml_path

                def run(self):
                    result = dict(results_queue.pop(0))
                    result["yaml_path"] = self.yaml_path
                    return result

            def _fake_export_dataset(*, source_path, source_type, schema, output_dir, robot_name=None, link_images=True):
                export_dir = Path(output_dir) / f"{robot_name}_export"
                export_dir.mkdir(parents=True, exist_ok=True)
                (export_dir / "manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": schema,
                            "robot_name": robot_name,
                            "episodes": [],
                            "camera_names": ["top_cam", "wrist_cam", "front_cam"],
                        }
                    )
                )
                return export_dir

            def _fake_preprocess_exported_dataset(*, export_dir, output_dir, success_only=False, **kwargs):
                preprocess_dir = Path(output_dir) / "preprocessed"
                preprocess_dir.mkdir(parents=True, exist_ok=True)
                (preprocess_dir / "manifest.json").write_text(
                    json.dumps({"success_only": success_only})
                )
                return preprocess_dir

            def _fake_convert(raw_dataset_dir, repo_id="local/sim_dataset", output_root=None):
                dataset_root = Path(output_root) / Path(repo_id)
                dataset_root.mkdir(parents=True, exist_ok=True)
                return dataset_root

            with patch.dict(
                os.environ,
                {
                    "AZURE_OPENAI_API_KEY": "test-key",
                    "AZURE_OPENAI_BASE_URL": "https://example.invalid",
                },
                clear=False,
            ):
                with (
                    patch("src.data_collection.e2e_orchestrator.load_dotenv", return_value=None),
                    patch("src.data_collection.e2e_orchestrator.ensure_lerobot_available"),
                    patch("src.data_collection.e2e_orchestrator.DataCollectionPipeline", _FakePipeline),
                    patch("src.data_collection.e2e_orchestrator.export_dataset", side_effect=_fake_export_dataset),
                    patch(
                        "src.data_collection.e2e_orchestrator.preprocess_exported_dataset",
                        side_effect=_fake_preprocess_exported_dataset,
                    ),
                    patch(
                        "src.data_collection.e2e_orchestrator.convert_raw_dataset_to_lerobot",
                        side_effect=_fake_convert,
                    ),
                    patch(
                        "src.data_collection.e2e_orchestrator.check_lerobot_dataset",
                        side_effect=lambda dataset_root, repo_id=None: {
                            "pass": True,
                            "dataset_root": str(dataset_root),
                            "repo_id": repo_id,
                        },
                    ),
                ):
                    report = run_e2e_batch(config_path)

            self.assertTrue(report["pass"])
            self.assertTrue(report["batch_completed"])
            self.assertEqual(report["summary"]["task_count"], 2)
            self.assertEqual(report["summary"]["configured_task_count"], 2)
            self.assertEqual(set(report["robot_datasets"].keys()), {"franka", "openarm"})
            self.assertEqual(
                report["robot_datasets"]["franka"]["local_repo_id"],
                "local/e2e-unit-franka",
            )
            self.assertEqual(
                report["robot_datasets"]["openarm"]["local_repo_id"],
                "local/e2e-unit-openarm",
            )

            franka_task = next(task for task in report["task_runs"] if task["robot"] == "franka")
            self.assertEqual(franka_task["classification"]["label"], "배치 (Placement)")
            self.assertEqual(franka_task["baseline_tag"], "known_working")
            self.assertEqual(franka_task["successful_dataset_episodes"], 1)
            self.assertEqual(franka_task["geometry_successful_episodes"], 0)
            self.assertEqual(franka_task["vlm_successful_episodes"], 1)
            self.assertEqual(franka_task["success_policy"], "geometry_or_vlm")
            self.assertIn("FrankaStackEnvCfg", franka_task["representative_env_cfg"])
            self.assertIn("print('franka success')", franka_task["representative_cap_code"])
            self.assertEqual(franka_task["front_video_path"], str(franka_video))
            self.assertIn("cube", franka_task["domain_randomization"]["objects"])
            self.assertIn("target_marker.position", franka_task["domain_randomization"]["goals"])
            self.assertEqual(franka_task["terminal_status"], "pipeline_completed")

            merged_meta = load_raw_dataset_metadata(
                report["robot_datasets"]["franka"]["merged_raw_dataset"]
            )
            self.assertEqual(merged_meta["total_episodes"], 1)
            markdown_report = Path(report["markdown_report_path"]).read_text()
            vlm_audit_markdown = Path(report["vlm_audit_markdown_path"]).read_text()
            vlm_audit_json = json.loads(Path(report["vlm_audit_json_path"]).read_text())
            self.assertIn("FrankaStack", markdown_report)
            self.assertIn("Summary 설명", markdown_report)
            self.assertIn("대표 성공 영상", markdown_report)
            self.assertIn("대표 Env CFG 코드", markdown_report)
            self.assertIn("대표 CaP 코드", markdown_report)
            self.assertIn("baseline tag", markdown_report)
            self.assertIn("representative_success", vlm_audit_markdown)
            self.assertIn("Judge Prompt", vlm_audit_markdown)
            self.assertEqual(vlm_audit_json["summary"]["task_count"], 2)
            self.assertGreaterEqual(vlm_audit_json["summary"]["case_count"], 2)


if __name__ == "__main__":
    unittest.main()
