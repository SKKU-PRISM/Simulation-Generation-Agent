"""Tests for multi-task data collection orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.agent.data_collection.config import DataCollectionConfig, load_robot_config
from src.agent.data_collection.multitask_orchestrator import run_multitask_to_hf
from src.agent.data_collection.pipeline import DataCollectionPipeline
from src.agent.data_collection.raw_dataset_merge import merge_raw_datasets
from src.agent.data_collection.sim_recorder import SimRecorder


def _create_raw_dataset(base_dir: str, robot_name: str, dataset_name: str) -> Path:
    robot_cfg = load_robot_config(robot_name)
    recorder = SimRecorder(robot_cfg, base_dir, dataset_name=dataset_name, fps=20)
    recorder.start_episode(f"{robot_name} task")
    zeros = np.zeros(robot_cfg.total_dofs, dtype=np.float32)
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
    recorder.end_episode(success=True)
    return Path(recorder.finalize())


class MultiTaskPipelineTests(unittest.TestCase):
    def test_merge_raw_datasets_renumbers_episodes_and_tracks_sources(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_a = _create_raw_dataset(Path(tmpdir) / "run_a", "franka", "raw_dataset")
            raw_b = _create_raw_dataset(Path(tmpdir) / "run_b", "franka", "raw_dataset")

            merged = merge_raw_datasets(
                [raw_a, raw_b],
                Path(tmpdir) / "merged" / "raw_dataset",
                source_records=[
                    {"task": "TaskA", "yaml_path": "tasks/franka/a.yaml", "output_dir": "outputs/a"},
                    {"task": "TaskB", "yaml_path": "tasks/franka/b.yaml", "output_dir": "outputs/b"},
                ],
            )

            metadata = json.loads((merged / "metadata.json").read_text())
            self.assertEqual(metadata["total_episodes"], 2)
            self.assertEqual(metadata["successful_episodes"], 2)
            self.assertEqual(metadata["episodes"][0]["index"], 0)
            self.assertEqual(metadata["episodes"][1]["index"], 1)
            self.assertEqual(metadata["episodes"][0]["source_task"], "TaskA")
            self.assertEqual(metadata["episodes"][1]["source_task"], "TaskB")
            self.assertEqual(metadata["episodes"][0]["source_episode_index"], 0)
            self.assertTrue((merged / "episodes" / "episode_000000" / "states.npy").exists())
            self.assertTrue((merged / "episodes" / "episode_000001" / "states.npy").exists())

    def test_merge_raw_datasets_rejects_mismatched_robot_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_franka = _create_raw_dataset(Path(tmpdir) / "run_franka", "franka", "raw_dataset")
            raw_openarm = _create_raw_dataset(Path(tmpdir) / "run_openarm", "openarm", "raw_dataset")

            with self.assertRaises(ValueError):
                merge_raw_datasets(
                    [raw_franka, raw_openarm],
                    Path(tmpdir) / "merged" / "raw_dataset",
                )

    def test_pipeline_skips_auto_convert_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = DataCollectionConfig(output_dir=tmpdir)
            task_yaml = Path("tasks/franka/stack/franka_stack.yaml").resolve()
            env_dir = Path(tmpdir) / "env"
            env_dir.mkdir()

            pipeline = DataCollectionPipeline(
                yaml_path=str(task_yaml),
                config=cfg,
                env_dir=str(env_dir),
                auto_convert_to_lerobot=False,
            )

            def _fake_generate_collection_runner():
                runner_path = pipeline.output_dir / "collect_data.py"
                runner_path.parent.mkdir(parents=True, exist_ok=True)
                runner_path.write_text("print('stub')")
                return runner_path

            def _fake_execute_subprocess(_runner_path):
                raw_dir = pipeline.output_dir / "raw_dataset"
                raw_dir.mkdir(parents=True, exist_ok=True)
                (raw_dir / "metadata.json").write_text(json.dumps({"total_episodes": 0}))
                (pipeline.output_dir / "collection_results.json").write_text(
                    json.dumps({"pipeline_completed": True, "total_episodes": 0})
                )
                return True, "ok"

            with (
                patch.object(pipeline, "_generate_collection_runner", side_effect=_fake_generate_collection_runner),
                patch.object(pipeline, "_execute_subprocess", side_effect=_fake_execute_subprocess),
                patch.object(pipeline, "_convert_to_lerobot") as convert_mock,
            ):
                result = pipeline.run()

            self.assertTrue(result["pipeline_completed"])
            self.assertIsNone(result["lerobot_dataset"])
            convert_mock.assert_not_called()

    def test_run_multitask_to_hf_groups_by_robot_and_keeps_collecting_after_task_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_franka = _create_raw_dataset(Path(tmpdir) / "franka_run", "franka", "raw_dataset")
            raw_openarm = _create_raw_dataset(Path(tmpdir) / "openarm_run", "openarm", "raw_dataset")

            task_paths = []
            for idx in range(3):
                task_path = Path(tmpdir) / f"task_{idx}.yaml"
                task_path.write_text("task:\n  name: StubTask\nassets: []\n")
                task_paths.append(task_path)

            init_flags: list[bool] = []
            run_results = [
                {
                    "success": True,
                    "pipeline_completed": True,
                    "raw_dataset": str(raw_franka),
                    "robot": "franka",
                    "task": "FrankaTask",
                    "total_episodes": 1,
                },
                {
                    "success": False,
                    "pipeline_completed": False,
                    "raw_dataset": None,
                    "robot": "franka",
                    "task": "FailedTask",
                    "total_episodes": 0,
                },
                {
                    "success": True,
                    "pipeline_completed": True,
                    "raw_dataset": str(raw_openarm),
                    "robot": "openarm",
                    "task": "OpenArmTask",
                    "total_episodes": 1,
                },
            ]

            class _FakePipeline:
                def __init__(self, yaml_path, config=None, env_dir=None, auto_convert_to_lerobot=True):
                    self.yaml_path = yaml_path
                    init_flags.append(auto_convert_to_lerobot)

                def run(self):
                    return dict(run_results.pop(0), yaml_path=self.yaml_path)

            convert_calls: list[tuple[str, str, str]] = []

            def _fake_convert(raw_dataset_dir, repo_id="local/sim_dataset", output_root=None):
                dataset_root = Path(output_root) / Path(repo_id)
                dataset_root.mkdir(parents=True, exist_ok=True)
                convert_calls.append((str(raw_dataset_dir), repo_id, str(output_root)))
                return dataset_root

            with (
                patch("src.data_collection.multitask_orchestrator.ensure_lerobot_available"),
                patch("src.data_collection.multitask_orchestrator.DataCollectionPipeline", _FakePipeline),
                patch("src.data_collection.multitask_orchestrator.convert_raw_dataset_to_lerobot", side_effect=_fake_convert),
                patch(
                    "src.data_collection.multitask_orchestrator.check_lerobot_dataset",
                    side_effect=lambda dataset_root, repo_id=None: {
                        "pass": True,
                        "repo_id": repo_id,
                        "dataset_root": str(dataset_root),
                    },
                ),
            ):
                report = run_multitask_to_hf(
                    [str(path) for path in task_paths],
                    output_root=Path(tmpdir) / "batch",
                    local_repo_prefix="local/multitask",
                    upload_to_hf=False,
                    continue_on_failure=True,
                )

            self.assertEqual(init_flags, [False, False, False])
            self.assertTrue(report["pass"])
            self.assertEqual(set(report["robot_datasets"].keys()), {"franka", "openarm"})
            self.assertEqual(report["summary"]["merge_eligible_task_count"], 2)
            self.assertEqual(report["robot_datasets"]["franka"]["local_repo_id"], "local/multitask-franka")
            self.assertEqual(report["robot_datasets"]["openarm"]["local_repo_id"], "local/multitask-openarm")
            self.assertEqual(len(convert_calls), 2)

    def test_run_multitask_to_hf_uploads_robot_specific_repo_ids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_franka = _create_raw_dataset(Path(tmpdir) / "franka_run", "franka", "raw_dataset")
            task_path = Path(tmpdir) / "task.yaml"
            task_path.write_text("task:\n  name: StubTask\nassets: []\n")

            class _FakePipeline:
                def __init__(self, yaml_path, config=None, env_dir=None, auto_convert_to_lerobot=True):
                    self.yaml_path = yaml_path

                def run(self):
                    return {
                        "success": True,
                        "pipeline_completed": True,
                        "raw_dataset": str(raw_franka),
                        "robot": "franka",
                        "task": "FrankaTask",
                        "total_episodes": 1,
                        "yaml_path": self.yaml_path,
                    }

            def _fake_convert(raw_dataset_dir, repo_id="local/sim_dataset", output_root=None):
                dataset_root = Path(output_root) / Path(repo_id)
                dataset_root.mkdir(parents=True, exist_ok=True)
                return dataset_root

            with (
                patch("src.data_collection.multitask_orchestrator.ensure_lerobot_available"),
                patch("src.data_collection.multitask_orchestrator.ensure_huggingface_hub_available"),
                patch("src.data_collection.multitask_orchestrator.DataCollectionPipeline", _FakePipeline),
                patch("src.data_collection.multitask_orchestrator.convert_raw_dataset_to_lerobot", side_effect=_fake_convert),
                patch(
                    "src.data_collection.multitask_orchestrator.check_lerobot_dataset",
                    return_value={"pass": True, "repo_id": "local/multitask-franka"},
                ),
                patch(
                    "src.data_collection.multitask_orchestrator.publish_lerobot_dataset",
                    return_value={"pass": True, "repo_id": "org/mybatch-franka"},
                ) as publish_mock,
            ):
                report = run_multitask_to_hf(
                    [str(task_path)],
                    output_root=Path(tmpdir) / "batch",
                    local_repo_prefix="local/multitask",
                    hf_repo_prefix="org/mybatch",
                    upload_to_hf=True,
                    hf_private=True,
                )

            publish_mock.assert_called_once()
            self.assertEqual(
                publish_mock.call_args.kwargs["repo_id"],
                "org/mybatch-franka",
            )
            self.assertTrue(report["pass"])
            self.assertEqual(
                report["robot_datasets"]["franka"]["publish_report"]["repo_id"],
                "org/mybatch-franka",
            )


if __name__ == "__main__":
    unittest.main()
