"""Regression tests for evaluator/path/pipeline fixes."""

import json
import os
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.common.task_docs import load_task_document, resolve_task_document
from src.data_collection.config import load_pipeline_config, load_robot_config
from src.data_collection.dataset_export import (
    ADC_COMPATIBLE_SCHEMA,
    CANONICAL_TRAINING_SCHEMA,
    _resolve_adc_local_path,
    build_joint_normalization_spec,
    compose_adc_legacy_world_xyzrpy,
    compute_adc_tcp_world_xyzrpy,
    export_sim_raw_dataset,
    normalize_gripper_array,
    normalize_joint_matrix,
)
from src.data_collection.dataset_preprocess import preprocess_exported_dataset
from src.data_collection.lerobot_tools import (
    check_lerobot_dataset,
    convert_raw_dataset_to_lerobot,
    publish_lerobot_dataset,
)
from src.data_collection.multiview_grounding import MultiViewTargetGrounder
from src.data_collection.cap_generator import translate_scene_state
from src.data_collection.sim_recorder import SimRecorder, _coerce_goal_joint
from src.data_collection.sim_detector import SimDetector
from src.data_collection.sim_judge import build_task_aware_judge_prompt
from src.data_collection.sim_robot_interface import SimRobotInterface
from src.data_collection.sim_skills import SimSkills, create_ik_solver
from src.isaac_lab.assembling_kits_template import build_assembling_kits_template
from src.isaac_lab.agent import IsaacLabAgent
from src.isaac_lab.evaluator.mdp_correctness import MDPCorrectnessChecker
from src.isaac_lab.evaluator.parser import EnvCfgParser
from src.isaac_lab.evaluator.runtime_validity import RuntimeValidityChecker
from src.isaac_lab.evaluator.scene_fidelity import SceneFidelityChecker
from src.isaac_sim.scene_builder import SceneBuilder


class _DummyParser:
    def find_env_cfg_class(self):
        return "EnvCfg"

    def rewards_is_none(self):
        return False


class _RewardsNoneParser:
    def rewards_is_none(self):
        return True

    def extract_reward_terms(self):
        return []

    def extract_observation_terms(self):
        return []

    def extract_action_config(self):
        return {}

    def extract_termination_terms(self):
        return []

    def extract_event_terms(self):
        return []


class _FakeRobotInterface:
    def __init__(self, targets: np.ndarray, finger_targets: np.ndarray):
        self._targets = np.asarray(targets, dtype=np.float32)
        self._finger_targets = np.asarray(finger_targets, dtype=np.float32)

    def read_target_positions(self) -> np.ndarray:
        return self._targets.copy()

    def expand_gripper_target_positions(self, target) -> np.ndarray:
        return self._finger_targets.copy()

    def read_gripper_state(self) -> np.ndarray:
        if self._finger_targets.size == 0:
            return np.zeros(1, dtype=np.float32)
        return np.array([float(self._finger_targets[0])], dtype=np.float32)


class _FakeADCForwardKinematics:
    def __init__(self, position, rotation):
        self._position = np.asarray(position, dtype=np.float32)
        self._rotation = np.asarray(rotation, dtype=np.float32)

    def forward_kinematics(self, joint_positions):
        return self._position.copy(), self._rotation.copy()


class _FakeGroundingCameras:
    def __init__(self, depth_maps: dict[str, np.ndarray], transform_world: np.ndarray | None = None):
        self._depth_maps = depth_maps
        self._transform_world = np.asarray(transform_world if transform_world is not None else np.eye(4), dtype=np.float64)

    def capture_rgb_depth(self, camera_name: str):
        depth = self._depth_maps.get(camera_name)
        if depth is None:
            return None, None
        rgb = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
        return rgb, depth

    def get_intrinsics(self, camera_name: str):
        if camera_name not in self._depth_maps:
            raise KeyError(camera_name)
        depth = self._depth_maps[camera_name]
        width = depth.shape[1]
        height = depth.shape[0]
        return {
            "fx": 100.0,
            "fy": 100.0,
            "cx": (width - 1.0) * 0.5,
            "cy": (height - 1.0) * 0.5,
            "width": float(width),
            "height": float(height),
        }

    def get_camera_transform_world(self, camera_name: str) -> np.ndarray:
        if camera_name not in self._depth_maps:
            raise KeyError(camera_name)
        return self._transform_world.copy()


class _FakeGroundingRobot:
    def world_pose_to_robot_pose(self, position_world: np.ndarray, quat_world_wxyz: np.ndarray):
        return np.asarray(position_world, dtype=np.float64), np.asarray(quat_world_wxyz, dtype=np.float64)


def _install_fake_lerobot_dataset(sample: dict, features: dict | None = None):
    features = features or {key: {} for key in sample.keys()}

    class _FakeLeRobotDataset:
        def __init__(self, repo_id, root):
            self.repo_id = repo_id
            self.root = Path(root)
            self.features = features

        @classmethod
        def create(cls, repo_id, fps, features, root):
            dataset = cls(repo_id=repo_id, root=root)
            dataset._created = {
                "repo_id": repo_id,
                "fps": fps,
                "features": features,
                "root": root,
            }
            dataset._frames = []
            return dataset

        def add_frame(self, frame):
            self._frames.append(frame)

        def save_episode(self, task=""):
            self._task = task

        def consolidate(self):
            self._consolidated = True

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return sample

    lerobot_pkg = types.ModuleType("lerobot")
    common_pkg = types.ModuleType("lerobot.common")
    datasets_pkg = types.ModuleType("lerobot.common.datasets")
    dataset_mod = types.ModuleType("lerobot.common.datasets.lerobot_dataset")
    dataset_mod.LeRobotDataset = _FakeLeRobotDataset

    return _FakeLeRobotDataset, {
        "lerobot": lerobot_pkg,
        "lerobot.common": common_pkg,
        "lerobot.common.datasets": datasets_pkg,
        "lerobot.common.datasets.lerobot_dataset": dataset_mod,
    }


def _compile_goal_conditions_for_prompt(task_doc: dict, translated_positions: dict[str, dict]) -> list[dict]:
    goal = task_doc.get("goal", {})
    success_criteria = goal.get("success_criteria", {})
    conditions: list[dict] = []
    if isinstance(success_criteria, dict):
        for cond in success_criteria.get("conditions", []):
            if isinstance(cond, dict):
                conditions.append(dict(cond))
        stacking_order = success_criteria.get("stacking_order")
        if isinstance(stacking_order, list):
            for idx in range(1, len(stacking_order)):
                conditions.append(
                    {
                        "type": "on_top_of",
                        "object": stacking_order[idx],
                        "target": stacking_order[idx - 1],
                    }
                )
    for cond in goal.get("conditions", []):
        if isinstance(cond, dict):
            conditions.append(dict(cond))
    if conditions:
        if isinstance(success_criteria, dict) and success_criteria.get("inside_tray"):
            tray_center = success_criteria.get("tray_center")
            tray_half_extent = success_criteria.get("tray_half_extent")
            ordered_objects = [name for name in translated_positions if name.startswith("cube_")]
            if (
                isinstance(tray_center, (list, tuple))
                and len(tray_center) == 2
                and isinstance(tray_half_extent, (int, float))
            ):
                conditions.append(
                    {
                        "type": "inside_tray",
                        "objects": ordered_objects,
                        "tray_center": list(tray_center),
                        "tray_half_extent": float(tray_half_extent),
                    }
                )
        return conditions

    task_description = " ".join(
        part
        for part in (
            task_doc.get("task", {}).get("description", ""),
            goal.get("description", ""),
        )
        if isinstance(part, str)
    ).lower()
    if "stack" in task_description:
        ordered_objects = [name for name in translated_positions if name.startswith("cube_")]
        ordered_objects.sort()
        for idx in range(1, len(ordered_objects)):
            conditions.append(
                {
                    "type": "on_top_of",
                    "object": ordered_objects[idx],
                    "target": ordered_objects[idx - 1],
                }
            )
        if isinstance(success_criteria, dict) and success_criteria.get("inside_tray"):
            tray_center = success_criteria.get("tray_center")
            tray_half_extent = success_criteria.get("tray_half_extent")
            if (
                isinstance(tray_center, (list, tuple))
                and len(tray_center) == 2
                and isinstance(tray_half_extent, (int, float))
            ):
                conditions.append(
                    {
                        "type": "inside_tray",
                        "objects": ordered_objects,
                        "tray_center": list(tray_center),
                        "tray_half_extent": float(tray_half_extent),
                    }
                )
    return conditions


def _select_relevant_objects_for_prompt(
    translated_positions: dict[str, dict],
    conditions: list[dict],
) -> dict[str, dict]:
    referenced: set[str] = set()
    for condition in conditions:
        subject = condition.get("subject", condition.get("object"))
        target = condition.get("target")
        objects = condition.get("objects") or []
        if isinstance(subject, str):
            referenced.add(subject)
        if isinstance(target, str):
            referenced.add(target)
        if isinstance(objects, list):
            referenced.update(str(obj) for obj in objects if isinstance(obj, str))

    selected: dict[str, dict] = {}
    for name, entry in translated_positions.items():
        role = str(entry.get("task_role", ""))
        name_lower = name.lower()
        if name_lower.endswith("_anchor") or "anchor" in name_lower or name_lower == "command_pose":
            continue
        if name in referenced or role in {"target_marker", "support_surface", "goal_target"}:
            selected[name] = entry
    return selected


def _build_prompt_inputs_for_task(task_doc: dict) -> tuple[dict[str, dict], list[dict], dict[str, dict]]:
    scene_state: dict[str, dict] = {}
    for asset in task_doc.get("assets", []):
        name = asset.get("name")
        initial_state = asset.get("initial_state", {})
        position = initial_state.get("position", asset.get("position"))
        if not isinstance(name, str) or not isinstance(position, (list, tuple)) or len(position) < 3:
            continue
        quaternion = initial_state.get(
            "quaternion",
            asset.get("quaternion", asset.get("rotation", [1.0, 0.0, 0.0, 0.0])),
        )
        scene_state[name] = {
            "position": list(position[:3]),
            "quaternion": list(quaternion[:4]) if isinstance(quaternion, (list, tuple)) and len(quaternion) >= 4 else [1.0, 0.0, 0.0, 0.0],
        }
    translated = translate_scene_state(task_doc, scene_state)
    conditions = _compile_goal_conditions_for_prompt(task_doc, translated)
    relevant = _select_relevant_objects_for_prompt(translated, conditions)
    return translated, conditions, relevant


class RegressionTests(unittest.TestCase):
    def test_sim_detector_uses_prim_path_basename_alias_for_scene_mapping(self):
        detector = SimDetector.__new__(SimDetector)
        entity = detector._resolve_entity_key(
            "cube",
            {"prim_path": "/World/Object"},
            {"robot", "object"},
        )
        self.assertEqual(entity, "object")

    def test_sim_detector_does_not_singleton_fallback_visual_target_marker(self):
        detector = SimDetector.__new__(SimDetector)
        entity = detector._resolve_entity_key(
            "target_marker",
            {
                "prim_path": "/World/TargetMarker",
                "type": "rigid",
                "physics": {"rigid_body": False, "collision": False},
            },
            {"cube"},
            allow_singleton_fallback=not detector._should_use_stage_pose_mapping(
                "target_marker",
                {
                    "prim_path": "/World/TargetMarker",
                    "type": "rigid",
                    "physics": {"rigid_body": False, "collision": False},
                },
            ),
        )
        self.assertIsNone(entity)

    def test_sim_detector_get_all_objects_includes_stage_pose_targets(self):
        detector = SimDetector.__new__(SimDetector)
        detector._asset_meta = {
            "robot": {"type": "articulation"},
            "cube": {"type": "rigid"},
            "target_marker": {
                "type": "rigid",
                "physics": {"rigid_body": False, "collision": False},
            },
        }
        detector._robot_name = "robot"
        detector._entity_map = {"cube": "cube"}

        def _fake_get_object_pose(name: str):
            poses = {
                "cube": (np.array([0.4, -0.2, 0.055]), np.array([1.0, 0.0, 0.0, 0.0])),
                "target_marker": (np.array([0.5, 0.15, 0.02]), np.array([1.0, 0.0, 0.0, 0.0])),
            }
            return poses[name]

        detector.get_object_pose = _fake_get_object_pose

        objects = SimDetector.get_all_objects(detector)
        self.assertIn("cube", objects)
        self.assertIn("target_marker", objects)
        self.assertEqual(objects["target_marker"]["position"], [0.5, 0.15, 0.02])

    def test_sim_detector_prefers_stage_pose_for_assembling_kits_shape_assets(self):
        detector = SimDetector.__new__(SimDetector)
        detector._asset_meta = {"shape_12": {"asset_path": "assets/assembling_kits/shape_12.usd"}}
        self.assertTrue(detector._should_prefer_stage_pose("shape_12"))

    def test_parser_extracts_scene_entities_from_post_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

class SceneCfg(InteractiveSceneCfg):
    pass

class EnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        self.scene.cube_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 0.2, 0.3], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(usd_path="foo/bar.usd", scale=(1.0, 1.0, 1.0)),
        )
""".strip()
            )

            parser = EnvCfgParser(env_cfg_path)
            entities = parser.extract_scene_entities()

            self.assertIn("cube_1", entities)
            self.assertEqual(entities["cube_1"].get("prim_path"), "{ENV_REGEX_NS}/Cube_1")
            self.assertEqual(entities["cube_1"].get("usd_path"), "foo/bar.usd")
            self.assertEqual(entities["cube_1"].get("pos"), [0.1, 0.2, 0.3])

    def test_runtime_checker_handles_relative_output_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            fake_isaaclab = tmpdir_path / "IsaacLab"
            fake_isaaclab.mkdir(parents=True, exist_ok=True)
            launcher = fake_isaaclab / "isaaclab.sh"
            launcher.write_text("#!/usr/bin/env bash\n")
            launcher.chmod(0o755)

            rel_output_dir = Path("outputs") / "run_eval"
            cfg = {
                "isaaclab_path": str(fake_isaaclab),
                "conda_env": "env_isaaclab",
                "timeout": 5,
                "headless": True,
                "num_envs": 2,
                "eval_steps": 3,
            }

            seen = {}

            def fake_run(cmd, capture_output, text, timeout, cwd, env):
                seen["cmd"] = cmd
                seen["cwd"] = cwd
                results_file = Path(cwd) / "eval_results.json"
                results_file.parent.mkdir(parents=True, exist_ok=True)
                results_file.write_text(json.dumps({"env_created": True}))
                return None

            old_cwd = os.getcwd()
            os.chdir(tmpdir_path)
            try:
                checker = RuntimeValidityChecker(rel_output_dir, _DummyParser(), {"assets": []}, cfg)
                with patch("src.isaac_lab.evaluator.runtime_validity.subprocess.run", side_effect=fake_run):
                    ok = checker.execute()
            finally:
                os.chdir(old_cwd)

            self.assertTrue(ok)
            self.assertEqual(Path(seen["cwd"]).resolve(), (tmpdir_path / rel_output_dir).resolve())
            self.assertIn("eval_runner.py", seen["cmd"][-1])
            self.assertNotIn("outputs/run_eval/eval_runner.py", seen["cmd"][-1])

    def test_scene_builder_resolves_local_assets_from_repo_root(self):
        resolved = SceneBuilder._resolve_asset_path("assets/robots/so101/so101.usd")
        resolved_path = Path(resolved)
        self.assertTrue(resolved_path.is_absolute())
        self.assertTrue(resolved_path.as_posix().endswith("/assets/robots/so101/so101.usd"))
        self.assertNotIn("/src/assets/", resolved_path.as_posix())

    def test_task_doc_resolves_franka_assembly_assets_and_slot_target(self):
        doc = load_task_document("tasks/franka/assembly/franka_assembling_kits.yaml")
        assets = {asset["name"]: asset for asset in doc["assets"]}
        goal = doc["goal"]["conditions"][0]

        self.assertTrue(assets["kit_tray"]["asset_path"].endswith("/assets/assembling_kits/kit_303.usd"))
        self.assertEqual(goal["subject"], "shape_12")
        self.assertEqual(goal["target"], "kit_tray")
        self.assertEqual(goal["resolved_target_object_id"], 12)
        self.assertIn("target_position", goal)
        self.assertIn("target_rotation", goal)
        self.assertAlmostEqual(assets["shape_12"]["position"][2], 0.02, places=4)
        self.assertNotIn("randomize", assets["shape_12"])

    def test_franka_assembling_kits_template_uses_static_scene_assets_without_reset_dr(self):
        doc = load_task_document("tasks/franka/assembly/franka_assembling_kits.yaml")
        code = build_assembling_kits_template(doc, "franka")
        env_cfg = code["env_cfg.py"]

        self.assertIn("shape_09 = AssetBaseCfg(", env_cfg)
        self.assertIn("shape_12 = RigidObjectCfg(", env_cfg)
        self.assertNotIn("shape_09 = RigidObjectCfg(", env_cfg)
        self.assertIn("func=spawn_usd_with_child_physics,", env_cfg)
        self.assertIn("mass_props=sim_utils.MassPropertiesCfg(mass=0.02)", env_cfg)
        self.assertIn("rigid_props=sim_utils.RigidBodyPropertiesCfg(", env_cfg)
        self.assertIn("/assets/assembling_kits/kit_303.usd", env_cfg)
        self.assertIn("def post_env_setup(env):", env_cfg)
        self.assertIn("TriangleMeshPropertiesCfg()", env_cfg)
        self.assertNotIn('randomize_shape_12 = EventTerm(', env_cfg)

    def test_non_franka_assembling_kits_template_still_generates_reset_dr(self):
        doc = load_task_document("tasks/openarm/assembly/openarm_assembling_kits.yaml")
        code = build_assembling_kits_template(doc, "openarm")
        env_cfg = code["env_cfg.py"]

        self.assertIn('randomize_shape_12 = EventTerm(', env_cfg)
        self.assertIn("func=randomize_static_asset_pose", env_cfg)
        self.assertIn('"prim_path": \'/World/Shape12\'', env_cfg)
        self.assertIn('"yaw_range": (-3.14159, 3.14159)', env_cfg)

    def test_franka_assembly_tasks_define_movable_object_randomization(self):
        lift_doc = load_task_document("tasks/franka/assembly/franka_lift_peg_upright.yaml")
        lift_assets = {asset["name"]: asset for asset in lift_doc["assets"]}
        self.assertEqual(lift_assets["peg_red"]["randomize"]["position"]["x"], [-0.05, 0.05])
        self.assertEqual(lift_assets["peg_red"]["randomize"]["position"]["y"], [-0.10, 0.10])

        peg_doc = load_task_document("tasks/franka/assembly/franka_peg_insertion_side.yaml")
        peg_assets = {asset["name"]: asset for asset in peg_doc["assets"]}
        self.assertNotIn("randomize", peg_assets["peg_head"])

        charger_doc = load_task_document("tasks/franka/assembly/franka_plug_charger.yaml")
        charger_assets = {asset["name"]: asset for asset in charger_doc["assets"]}
        self.assertNotIn("randomize", charger_assets["charger_base"])

    def test_franka_factory_peg_insert_randomizes_hole_not_staging_peg(self):
        doc = load_task_document("tasks/franka/peg_insert/franka_peg_insert.yaml")
        assets = {asset["name"]: asset for asset in doc["assets"]}

        self.assertEqual(assets["hole"]["randomize"]["position"]["x"], [-0.05, 0.05])
        self.assertEqual(assets["hole"]["randomize"]["position"]["y"], [-0.05, 0.05])
        self.assertNotIn("randomize", assets["peg"])

    def test_agent_detects_assembling_kits_template_route(self):
        doc = load_task_document("tasks/franka/assembly/franka_assembling_kits.yaml")
        self.assertTrue(
            IsaacLabAgent._is_assembling_kits_task(
                "tasks/franka/assembly/franka_assembling_kits.yaml",
                doc,
            )
        )

    def test_task_doc_resolves_repo_local_so101_asset_to_absolute_path(self):
        doc = load_task_document("tasks/so101/stack/so101_stack.yaml")
        robot = next(asset for asset in doc["assets"] if asset.get("type") == "articulation")

        self.assertTrue(robot["asset_path"].startswith("/"))
        self.assertTrue(robot["asset_path"].endswith("/assets/robots/so101/so101.usd"))

    def test_task_doc_loads_ur10e_assembly_in_canonical_form(self):
        doc = load_task_document("tasks/ur10e/assembly/ur10e_plug_charger.yaml")
        robot = next(asset for asset in doc["assets"] if asset.get("type") == "articulation")

        self.assertEqual(robot["robot_type"], "ur10e")
        self.assertIn("/ur10e/ur10e.usd", robot["asset_path"])
        self.assertEqual(robot["variant_sets"]["Gripper"], "Robotiq_2f_85")
        self.assertEqual(robot["ee_frame"]["body"], "wrist_3_link")
        self.assertTrue(doc["task"]["name"].startswith("UR10e"))

    def test_task_doc_still_canonicalizes_external_legacy_ur10_assembly(self):
        doc = {
            "task": {"name": "UR10PlugCharger", "description": "UR10 robot with suction gripper picks up a charger"},
            "assets": [
                {
                    "name": "robot",
                    "type": "articulation",
                    "asset_path": "{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10/ur10.usd",
                    "variant_sets": {"Gripper": "Long_Suction"},
                    "ee_frame": {"body": "ee_link", "offset_position": [0.22, 0.0, 0.0]},
                }
            ],
        }
        resolved = resolve_task_document(doc, "tasks/external/assembly/legacy_ur10.yaml")
        robot = resolved["assets"][0]

        self.assertEqual(robot["robot_type"], "ur10e")
        self.assertIn("/ur10e/ur10e.usd", robot["asset_path"])
        self.assertEqual(robot["variant_sets"]["Gripper"], "Robotiq_2f_85")
        self.assertEqual(robot["ee_frame"]["body"], "wrist_3_link")
        self.assertEqual(robot["ee_frame"]["offset_position"], [0.0, 0.0, 0.0])
        self.assertTrue(resolved["task"]["name"].startswith("UR10e"))

    def test_reward_structure_allows_none_rewards_when_enabled(self):
        checker = MDPCorrectnessChecker(
            yaml_doc={"assets": []},
            parser=_RewardsNoneParser(),
            config={"allow_none_rewards": True},
        )
        result = checker.check_reward_structure()
        self.assertEqual(result["score"], 3)
        self.assertEqual(result["status"], "WARN")

    def test_robot_detection_normalizes_ur10_aliases_to_ur10e(self):
        agent = IsaacLabAgent.__new__(IsaacLabAgent)

        task_doc = {
            "task": {"name": "UR10PickPlaceBox"},
            "assets": [
                {
                    "type": "articulation",
                    "robot_type": "ur10",
                    "asset_path": "{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd",
                }
            ],
        }
        detected = agent._detect_robot(task_doc, "tasks/ur10e/pick_place/ur10e_pick_place_box.yaml")
        self.assertEqual(detected, "ur10e")

    def test_scene_fidelity_recognizes_ur10e_robot_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG

@configclass
class SceneCfg(InteractiveSceneCfg):
    pass

@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=2, env_spacing=2.5)

    def __post_init__(self):
        self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
""".strip()
            )

            parser = EnvCfgParser(env_cfg_path)
            checker = SceneFidelityChecker(
                yaml_doc={"assets": [{"name": "robot", "type": "articulation", "robot_type": "ur10e"}]},
                parser=parser,
                config={},
            )
            result = checker.check_robot_config()
            self.assertEqual(result["score"], 4)
            self.assertEqual(result["status"], "PASS")

    def test_parser_extracts_primitive_scale_and_color(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.assets import RigidObjectCfg
import isaaclab.sim as sim_utils

@configclass
class SceneCfg(InteractiveSceneCfg):
    peg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Peg",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.1, 0.2, 0.3]),
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.05, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.69, 0.055, 0.055)),
        ),
    )
""".strip()
            )
            parser = EnvCfgParser(env_cfg_path)
            entities = parser.extract_scene_entities()

            self.assertEqual(entities["peg"]["scale"], [0.12, 0.05, 0.05])
            self.assertEqual(entities["peg"]["color"], [0.69, 0.055, 0.055])

    def test_load_robot_config_maps_ur10_alias_to_ur10e_profile(self):
        cfg = load_robot_config("ur10")
        self.assertEqual(cfg.name, "ur10e")
        self.assertTrue(cfg.has_gripper_joints)
        self.assertEqual(len(cfg.finger_joint_names), 6)

    def test_load_robot_config_exposes_openarm_virtual_tcp_offset(self):
        cfg = load_robot_config("openarm")
        self.assertEqual(cfg.ee_frame_body, "openarm_hand")
        self.assertEqual(cfg.ee_frame_tcp, "")
        self.assertEqual(cfg.ee_frame_offset_position, [0.0, 0.0, 0.08])
        self.assertEqual(cfg.ik_ee_frame, "openarm_hand")
        self.assertEqual(cfg.ik_backend, "differential_ik")
        self.assertTrue(Path(cfg.urdf_path).is_absolute())
        self.assertTrue(Path(cfg.urdf_path).exists())

    def test_load_robot_config_exposes_so101_tcp_offset(self):
        cfg = load_robot_config("so101")
        self.assertEqual(cfg.ee_frame_body, "gripper_frame_link")
        self.assertEqual(cfg.ee_frame_offset_position, [0.04, 0.0, 0.0])
        self.assertEqual(cfg.grasp_lateral_bias, 0.0)

    def test_create_ik_solver_prefers_openarm_differential_ik_with_live_robot(self):
        cfg = load_robot_config("openarm")
        robot_interface = object()
        with patch("src.data_collection.sim_skills.DifferentialIKSolver", return_value="diff-ik") as mock_solver:
            solver = create_ik_solver(cfg, robot_interface=robot_interface)
        self.assertEqual(solver, "diff-ik")
        mock_solver.assert_called_once_with(robot_interface)

    def test_create_ik_solver_falls_back_to_pinocchio_for_openarm_without_live_robot(self):
        cfg = load_robot_config("openarm")
        with patch("src.data_collection.sim_skills.PinocchioIKSolver", return_value="pin-ik") as mock_solver:
            solver = create_ik_solver(cfg)
        self.assertEqual(solver, "pin-ik")
        mock_solver.assert_called_once_with(
            urdf_path=cfg.urdf_path,
            ee_frame=cfg.ik_ee_frame or cfg.ee_frame_tcp or cfg.ee_frame_body,
            joint_names=cfg.arm_joint_names,
            tcp_offset=[0.0, 0.0, 0.08],
        )

    def test_create_ik_solver_passes_so101_tcp_offset_to_pinocchio(self):
        cfg = load_robot_config("so101")
        with patch("src.data_collection.sim_skills.PinocchioIKSolver", return_value="pin-ik") as mock_solver:
            solver = create_ik_solver(cfg)
        self.assertEqual(solver, "pin-ik")
        mock_solver.assert_called_once_with(
            urdf_path=cfg.urdf_path,
            ee_frame=cfg.ik_ee_frame or cfg.ee_frame_tcp or cfg.ee_frame_body,
            joint_names=cfg.arm_joint_names,
            tcp_offset=[0.04, 0.0, 0.0],
        )

    def test_openarm_world_target_to_ik_target_applies_virtual_tcp_offset(self):
        cfg = load_robot_config("openarm")
        skills = SimSkills.__new__(SimSkills)
        skills.cfg = cfg
        skills.PALM_DOWN_ROTATION = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )

        class _DummyRobot:
            @staticmethod
            def read_ee_pose():
                return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        skills.robot = _DummyRobot()
        target_tcp_world = np.array([0.35, 0.05, 0.12], dtype=np.float64)
        target_rotation = np.asarray(skills.PALM_DOWN_ROTATION, dtype=np.float64)

        ik_target_world = skills._world_target_to_ik_target(
            target_tcp_world,
            target_rotation_world=target_rotation,
        )

        expected = target_tcp_world - target_rotation @ np.array([0.0, 0.0, 0.08], dtype=np.float64)
        np.testing.assert_allclose(ik_target_world, expected, atol=1e-8)

    def test_sim_skills_normalizes_wrapped_ik_targets_to_joint_limits(self):
        cfg = load_robot_config("so101")
        skills = SimSkills.__new__(SimSkills)
        skills.cfg = cfg

        current = [0.0, -1.0, 1.0, 0.0, 0.0]
        wrapped = [-13.75, -1.0, 1.0, 0.0, 0.0]
        normalized = skills._normalize_arm_joint_targets(wrapped, current)

        self.assertGreaterEqual(normalized[0], cfg.joint_limits["shoulder_pan"][0])
        self.assertLessEqual(normalized[0], cfg.joint_limits["shoulder_pan"][1])
        self.assertLess(abs(normalized[0] - current[0]), 2.0)

    def test_openarm_generator_fix_injects_tcp_frame_and_disables_overlap_prone_randomization(self):
        task_doc = load_task_document("tasks/openarm/stack/openarm_stack.yaml")
        agent = IsaacLabAgent.__new__(IsaacLabAgent)

        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg

class SceneCfg(InteractiveSceneCfg):
    robot = MISSING

class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg()
    class Events:
        randomize_cube_1 = object()
        randomize_cube_2 = object()
        randomize_cube_3 = object()
    class Observations:
        ee_pose = 'params={"body_name": "openarm_hand"}'
    events = Events()
    def __post_init__(self):
        super().__post_init__()
""".strip()
            )
            fixes = agent._apply_robot_specific_fixes(Path(tmpdir), "openarm", task_doc)
            patched = env_cfg_path.read_text()

        self.assertTrue(fixes)
        self.assertIn("OPENARM_UNI_CFG.replace", patched)
        self.assertIn("openarm_ee_tcp", patched)
        self.assertIn("randomize_cube_1", patched)
        self.assertIn('SceneEntityCfg("robot", body_names=["openarm_hand"])', patched)
        self.assertGreater(
            patched.index("OPENARM_UNI_CFG.replace"),
            patched.rindex("class EnvCfg"),
        )

    def test_franka_generator_fix_normalizes_scene_entity_body_name(self):
        task_doc = load_task_document("tasks/franka/assembly/franka_peg_insertion_side.yaml")
        agent = IsaacLabAgent.__new__(IsaacLabAgent)

        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from isaaclab.managers import SceneEntityCfg

class EnvCfg:
    class Observations:
        ee_pose = 'params={"asset_cfg": SceneEntityCfg("robot", body_name="panda_hand")}'
""".strip()
            )
            fixes = agent._apply_robot_specific_fixes(Path(tmpdir), "franka", task_doc)
            patched = env_cfg_path.read_text()

        self.assertTrue(fixes)
        self.assertIn('SceneEntityCfg("robot", body_names=["panda_hand"])', patched)

    def test_franka_generator_fix_restores_articulation_scale_from_yaml(self):
        task_doc = load_task_document("tasks/franka/cabinet/franka_cabinet_store_cube.yaml")
        agent = IsaacLabAgent.__new__(IsaacLabAgent)

        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from isaaclab.sim import UsdFileCfg
from isaaclab.assets import ArticulationCfg

drawer = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Drawer",
    spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0.15, 0.12), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={"drawer_top_joint": 0.0}),
)
""".strip()
            )
            fixes = agent._apply_robot_specific_fixes(Path(tmpdir), "franka", task_doc)
            patched = env_cfg_path.read_text()

        self.assertTrue(fixes)
        self.assertIn("restored articulation spawn scale", " ".join(fixes))
        self.assertIn("scale=(0.3, 0.3, 0.3)", patched)

    def test_so101_generator_fix_injects_explicit_primitive_mass_and_collision(self):
        task_doc = load_task_document("tasks/so101/stack/so101_stack.yaml")
        agent = IsaacLabAgent.__new__(IsaacLabAgent)

        with tempfile.TemporaryDirectory() as tmpdir:
            env_cfg_path = Path(tmpdir) / "env_cfg.py"
            env_cfg_path.write_text(
                """
from dataclasses import MISSING
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg

class SceneCfg(InteractiveSceneCfg):
    robot = MISSING

class EnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg()
    class Observations:
        ee_pose = 'params={"body_name": "gripper_frame_link", "asset_cfg": SceneEntityCfg("robot")}'
    def __post_init__(self):
        super().__post_init__()
""".strip()
            )
            fixes = agent._apply_robot_specific_fixes(Path(tmpdir), "so101", task_doc)
            patched = env_cfg_path.read_text()

        self.assertTrue(fixes)
        self.assertIn("UsdFileCfg(", patched)
        self.assertIn("MassPropertiesCfg(mass=0.02)", patched)
        self.assertIn("CollisionPropertiesCfg()", patched)
        self.assertIn('self.scene.table = AssetBaseCfg(', patched)
        self.assertIn('self.scene.cube_1 = RigidObjectCfg(', patched)
        self.assertIn('SceneEntityCfg("robot", body_names=["gripper_frame_link"])', patched)
        self.assertIn('open_command_expr={"gripper": 1.745}', patched)
        self.assertIn('close_command_expr={"gripper": -0.175}', patched)

    def test_sim_recorder_records_tcp_observations(self):
        cfg = load_robot_config("franka")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="test_dataset", fps=20)
            recorder.start_episode("test task", episode_idx=0)
            recorder.record_step(
                state=np.zeros(cfg.total_dofs, dtype=np.float32),
                action=np.ones(cfg.total_dofs, dtype=np.float32),
                skill_label="move",
                skill_type="move",
                skill_progress=0.5,
                goal_joint=np.arange(cfg.total_dofs, dtype=np.float32),
                goal_world_xyzrpy=np.array([0.4, 0.1, 0.2, 0.0, 0.1, 0.2], dtype=np.float32),
                goal_robot_xyzrpy=np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3], dtype=np.float32),
                goal_gripper=0.04,
                tcp_world_xyzrpy=np.array([0.5, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float32),
                tcp_robot_xyzrpy=np.array([0.2, 0.1, 0.15, 0.0, 0.0, 0.0], dtype=np.float32),
                gripper_state=np.array([0.02], dtype=np.float32),
            )
            recorder.end_episode(success=True)

            ep_dir = Path(tmpdir) / "test_dataset" / "episodes" / "episode_000000"
            tcp_world = np.load(ep_dir / "tcp_world_xyzrpy.npy")
            tcp_robot = np.load(ep_dir / "tcp_robot_xyzrpy.npy")
            gripper_state = np.load(ep_dir / "gripper_state.npy")
            goal_robot_dense = np.load(ep_dir / "goal_robot_xyzrpy.npy")
            with open(ep_dir / "skills.json") as f:
                skills = json.load(f)

        self.assertEqual(tcp_world.shape, (1, 6))
        self.assertEqual(tcp_robot.shape, (1, 6))
        self.assertEqual(gripper_state.shape, (1, 1))
        self.assertEqual(goal_robot_dense.shape, (1, 6))
        self.assertEqual(len(skills[0]["goal_joint"]), cfg.total_dofs)
        np.testing.assert_allclose(gripper_state, np.array([[0.02]], dtype=np.float32))
        np.testing.assert_allclose(
            goal_robot_dense,
            np.array([[0.1, -0.2, 0.3, -0.1, 0.2, -0.3]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            skills[0]["goal_world_xyzrpy"],
            [0.4, 0.1, 0.2, 0.0, 0.1, 0.2],
        )
        np.testing.assert_allclose(
            skills[0]["goal_robot_xyzrpy"],
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
        )

    def test_sim_recorder_generates_single_front_success_video(self):
        cfg = load_robot_config("franka")
        front = np.full((8, 8, 3), 127, dtype=np.uint8)
        wrist = np.full((8, 8, 3), 255, dtype=np.uint8)

        def _fake_encode(self, frames_dir, out_path):
            del frames_dir
            out_path.write_bytes(b"fake-mp4")

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(SimRecorder, "_encode_mp4_from_frames", autospec=True, side_effect=_fake_encode):
                recorder = SimRecorder(cfg, tmpdir, dataset_name="test_dataset", fps=20)
                recorder.start_episode("test task", episode_idx=0)
                recorder.record_step(
                    state=np.zeros(cfg.total_dofs, dtype=np.float32),
                    action=np.zeros(cfg.total_dofs, dtype=np.float32),
                    images={
                        "front_cam": front,
                        "wrist_cam": wrist,
                    },
                    skill_label="move",
                    skill_type="move",
                    skill_progress=0.5,
                    goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                    goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_gripper=0.0,
                    tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    gripper_state=np.zeros(1, dtype=np.float32),
                )
                recorder.end_episode(success=True)
                raw_dir = Path(recorder.finalize())

            wrist_dir = raw_dir / "episodes" / "episode_000000" / "images" / "wrist_cam"
            front_dir = raw_dir / "episodes" / "episode_000000" / "images" / "front_cam"
            videos_dir = Path(tmpdir) / "videos"

            self.assertTrue(wrist_dir.exists())
            self.assertTrue(front_dir.exists())
            self.assertTrue((videos_dir / "front_success.mp4").exists())
            self.assertEqual(recorder.front_video_episode, 0)
            self.assertTrue(recorder.front_video_generated)
            self.assertIsNone(recorder.front_video_error)
            self.assertEqual(recorder.front_video_camera_name, "front")

            metadata = json.loads((raw_dir / "metadata.json").read_text())
            self.assertEqual(metadata["front_video_path"], "videos/front_success.mp4")
            self.assertEqual(metadata["front_video_camera_name"], "front")
            self.assertEqual(metadata["front_video_episode"], 0)
            self.assertIn("front_cam", metadata["camera_names"])
            self.assertIn("wrist_cam", metadata["camera_names"])

    def test_sim_recorder_replaces_geometry_video_with_overall_video(self):
        cfg = load_robot_config("franka")
        front_a = np.full((8, 8, 3), 90, dtype=np.uint8)
        front_b = np.full((8, 8, 3), 180, dtype=np.uint8)

        def _fake_encode(self, frames_dir, out_path):
            frame_count = len(list(Path(frames_dir).glob("*.png")))
            out_path.write_bytes(f"frames={frame_count}".encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(SimRecorder, "_encode_mp4_from_frames", autospec=True, side_effect=_fake_encode):
                recorder = SimRecorder(
                    cfg,
                    tmpdir,
                    dataset_name="test_dataset",
                    fps=20,
                    front_video_max_priority=2,
                )

                recorder.start_episode("geometry-only task", episode_idx=0)
                recorder.record_step(
                    state=np.zeros(cfg.total_dofs, dtype=np.float32),
                    action=np.zeros(cfg.total_dofs, dtype=np.float32),
                    images={"front_cam": front_a},
                    skill_label="move",
                    skill_type="move",
                    skill_progress=0.0,
                    goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                    goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_gripper=0.0,
                    tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    gripper_state=np.zeros(1, dtype=np.float32),
                )
                recorder.end_episode(
                    success=True,
                    front_video_priority=1,
                    front_video_success_type="geometry_only",
                )

                recorder.start_episode("overall task", episode_idx=1)
                recorder.record_step(
                    state=np.zeros(cfg.total_dofs, dtype=np.float32),
                    action=np.zeros(cfg.total_dofs, dtype=np.float32),
                    images={"front_cam": front_b},
                    skill_label="move",
                    skill_type="move",
                    skill_progress=0.0,
                    goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                    goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_gripper=0.0,
                    tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    gripper_state=np.zeros(1, dtype=np.float32),
                )
                recorder.end_episode(
                    success=True,
                    front_video_priority=2,
                    front_video_success_type="overall",
                )
                raw_dir = Path(recorder.finalize())

            self.assertTrue(recorder.front_video_generated)
            self.assertEqual(recorder.front_video_episode, 1)
            self.assertEqual(recorder.front_video_success_type, "overall")
            metadata = json.loads((raw_dir / "metadata.json").read_text())
            self.assertEqual(metadata["front_video_success_type"], "overall")

    def test_sim_recorder_can_select_top_camera_for_success_video(self):
        cfg = load_robot_config("franka")
        front = np.full((8, 8, 3), 64, dtype=np.uint8)
        top = np.full((8, 8, 3), 192, dtype=np.uint8)

        def _fake_encode(self, frames_dir, out_path):
            frame_count = len(list(Path(frames_dir).glob("*.png")))
            out_path.write_bytes(f"frames={frame_count}".encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(SimRecorder, "_encode_mp4_from_frames", autospec=True, side_effect=_fake_encode):
                recorder = SimRecorder(
                    cfg,
                    tmpdir,
                    dataset_name="test_dataset",
                    fps=20,
                    front_video_camera_name="top",
                )
                recorder.start_episode("cabinet task", episode_idx=0)
                recorder.record_step(
                    state=np.zeros(cfg.total_dofs, dtype=np.float32),
                    action=np.zeros(cfg.total_dofs, dtype=np.float32),
                    images={"front_cam": front, "top_cam": top},
                    skill_label="move",
                    skill_type="move",
                    skill_progress=0.0,
                    goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                    goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_gripper=0.0,
                    tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    gripper_state=np.zeros(1, dtype=np.float32),
                )
                recorder.end_episode(success=True)
                raw_dir = Path(recorder.finalize())

            self.assertTrue(recorder.front_video_generated)
            self.assertEqual(recorder.front_video_camera_name, "top")
            metadata = json.loads((raw_dir / "metadata.json").read_text())
            self.assertEqual(metadata["front_video_camera_name"], "top")
            meta_path = Path(tmpdir) / "videos" / "front_success_meta.json"
            self.assertTrue(meta_path.exists())
            self.assertEqual(json.loads(meta_path.read_text())["camera_name"], "top")

    def test_sim_recorder_can_exclude_front_from_dataset_images(self):
        cfg = load_robot_config("franka")
        front = np.full((8, 8, 3), 127, dtype=np.uint8)
        wrist = np.full((8, 8, 3), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(
                cfg,
                tmpdir,
                dataset_name="test_dataset",
                fps=20,
                dataset_cameras=["top", "wrist"],
            )
            recorder.start_episode("test task", episode_idx=0)
            recorder.record_step(
                state=np.zeros(cfg.total_dofs, dtype=np.float32),
                action=np.zeros(cfg.total_dofs, dtype=np.float32),
                images={"front_cam": front, "wrist_cam": wrist},
                skill_label="move",
                skill_type="move",
                skill_progress=0.0,
                goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_gripper=0.0,
                tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                gripper_state=np.zeros(1, dtype=np.float32),
            )
            recorder.end_episode(success=False)
            raw_dir = Path(recorder.finalize())

            self.assertFalse((raw_dir / "episodes" / "episode_000000" / "images" / "front_cam").exists())
            self.assertTrue((raw_dir / "episodes" / "episode_000000" / "images" / "wrist_cam").exists())

    def test_sim_recorder_skips_front_video_for_failed_episode(self):
        cfg = load_robot_config("franka")
        front = np.full((8, 8, 3), 127, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="test_dataset", fps=20)
            recorder.start_episode("test task", episode_idx=0)
            recorder.record_step(
                state=np.zeros(cfg.total_dofs, dtype=np.float32),
                action=np.zeros(cfg.total_dofs, dtype=np.float32),
                images={"front_cam": front},
                skill_label="move",
                skill_type="move",
                skill_progress=0.0,
                goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_gripper=0.0,
                tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                gripper_state=np.zeros(1, dtype=np.float32),
            )
            recorder.end_episode(success=False)
            recorder.finalize()

            self.assertFalse((Path(tmpdir) / "videos").exists())
            self.assertFalse(recorder.front_video_generated)
            self.assertIsNone(recorder.front_video_path)

    def test_load_pipeline_config_includes_camera_selection_lists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "data_collection.yaml"
            config_path.write_text(
                """
cameras:
  dataset_cameras: ["top", "wrist", "front"]
  judge_cameras: ["front"]
""".strip()
            )
            cfg = load_pipeline_config(str(config_path))

        self.assertEqual(cfg.dataset_cameras, ["top", "wrist", "front"])
        self.assertEqual(cfg.judge_cameras, ["front"])

    def test_sim_robot_interface_read_gripper_state_uses_primary_finger_joint(self):
        iface = SimRobotInterface.__new__(SimRobotInterface)
        iface.cfg = load_robot_config("openarm")
        iface.env_idx = 0
        iface._finger_indices = [7, 8]

        class _DummyData:
            joint_pos = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.031, 0.029]], dtype=np.float32)

        class _DummyArticulation:
            data = _DummyData()

        iface._articulation = _DummyArticulation()

        gripper_state = iface.read_gripper_state()

        np.testing.assert_allclose(gripper_state, np.array([0.031], dtype=np.float32))

    def test_patch_skill_metadata_range_updates_dense_goal_robot_pose(self):
        cfg = load_robot_config("franka")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="test_dataset", fps=20)
            recorder.start_episode("test task", episode_idx=0)
            recorder.record_step(
                state=np.zeros(cfg.total_dofs, dtype=np.float32),
                action=np.zeros(cfg.total_dofs, dtype=np.float32),
                skill_label="move",
                skill_type="move",
                skill_progress=0.0,
                goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                goal_world_xyzrpy=None,
                goal_robot_xyzrpy=None,
                goal_gripper=0.0,
                tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
            )
            recorder.patch_skill_metadata_range(
                0,
                goal_robot_xyzrpy=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32),
            )
            recorder.end_episode(success=True)
            ep_dir = Path(tmpdir) / "test_dataset" / "episodes" / "episode_000000"
            goal_robot = np.load(ep_dir / "goal_robot_xyzrpy.npy")

        np.testing.assert_allclose(
            goal_robot,
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32),
        )

    def test_sim_skills_compose_full_goal_joint_expands_gripper_targets(self):
        cfg = load_robot_config("franka")
        skills = SimSkills.__new__(SimSkills)
        skills.cfg = cfg
        skills.robot = _FakeRobotInterface(
            targets=np.zeros(cfg.total_dofs, dtype=np.float32),
            finger_targets=np.array([0.04, 0.04], dtype=np.float32),
        )

        goal_joint = skills._compose_full_goal_joint(
            np.arange(cfg.arm_dofs, dtype=np.float32),
            gripper_target=cfg.gripper_open_position,
        )

        self.assertEqual(goal_joint.shape, (cfg.total_dofs,))
        np.testing.assert_allclose(goal_joint[: cfg.arm_dofs], np.arange(cfg.arm_dofs, dtype=np.float32))
        np.testing.assert_allclose(goal_joint[cfg.arm_dofs :], np.array([0.04, 0.04], dtype=np.float32))

    def test_robot_pose_conversion_uses_articulation_root_frame(self):
        iface = SimRobotInterface.__new__(SimRobotInterface)
        root_yaw = np.deg2rad(90.0)
        root_quat = np.array([np.cos(root_yaw / 2.0), 0.0, 0.0, np.sin(root_yaw / 2.0)])
        iface.read_root_pose_world = lambda: (np.array([1.0, 2.0, 0.5]), root_quat)

        pos_robot, quat_robot = iface.world_pose_to_robot_pose(
            np.array([1.0, 3.0, 0.5]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        pos_world, quat_world = iface.robot_pose_to_world_pose(pos_robot, quat_robot)

        np.testing.assert_allclose(pos_robot, np.array([1.0, 0.0, 0.0]), atol=1e-6)
        np.testing.assert_allclose(pos_world, np.array([1.0, 3.0, 0.5]), atol=1e-6)
        np.testing.assert_allclose(quat_world, np.array([1.0, 0.0, 0.0, 0.0]), atol=1e-6)

    def test_coerce_goal_joint_pads_arm_only_vectors_to_total_dofs(self):
        padded = np.asarray(_coerce_goal_joint([0.1, 0.2, 0.3], total_dofs=5))
        np.testing.assert_allclose(padded, np.array([0.1, 0.2, 0.3, 0.0, 0.0], dtype=np.float32))

    def test_build_joint_normalization_spec_uses_pi_fallback_for_openarm_arm_limits(self):
        cfg = load_robot_config("openarm")
        spec = build_joint_normalization_spec(cfg)

        self.assertEqual(spec.entries[0].name, "openarm_joint1")
        self.assertEqual(spec.entries[0].source, "fallback_pi")
        self.assertAlmostEqual(spec.entries[0].lower, -np.pi)
        self.assertAlmostEqual(spec.entries[0].upper, np.pi)
        self.assertEqual(spec.gripper_source, "gripper_open_close")

    def test_build_joint_normalization_spec_keeps_open_as_plus_100_for_ur10e_gripper(self):
        cfg = load_robot_config("ur10e")
        spec = build_joint_normalization_spec(cfg)

        first_finger = spec.entries[cfg.arm_dofs]
        self.assertEqual(first_finger.name, "finger_joint")
        self.assertAlmostEqual(first_finger.lower, 0.65)
        self.assertAlmostEqual(first_finger.upper, 0.0)

    def test_export_sim_raw_dataset_canonical_training_normalizes_joint_data(self):
        cfg = load_robot_config("franka")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="raw_dataset", fps=20)
            recorder.start_episode("test task", episode_idx=0)
            state = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0.02, 0.02], dtype=np.float32)
            action = state.copy()
            goal_joint = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0, 0.04, 0.04], dtype=np.float32)
            goal_world = np.array([0.4, 0.1, 0.2, 0.0, 0.1, 0.2], dtype=np.float32)
            goal_robot = np.array([0.1, -0.2, 0.3, -0.1, 0.2, -0.3], dtype=np.float32)
            tcp_world = np.array([0.5, 0.0, 0.25, 0.0, 0.0, 0.0], dtype=np.float32)
            tcp_robot = np.array([0.2, 0.1, 0.15, 0.0, 0.0, 0.0], dtype=np.float32)
            gripper_state = np.array([0.02], dtype=np.float32)
            recorder.record_step(
                state=state,
                action=action,
                skill_label="move",
                skill_type="move",
                skill_progress=0.5,
                goal_joint=goal_joint,
                goal_world_xyzrpy=goal_world,
                goal_robot_xyzrpy=goal_robot,
                goal_gripper=0.04,
                tcp_world_xyzrpy=tcp_world,
                tcp_robot_xyzrpy=tcp_robot,
                gripper_state=gripper_state,
            )
            recorder.end_episode(success=True)
            raw_dir = Path(recorder.finalize())

            export_dir = export_sim_raw_dataset(
                raw_dataset_dir=raw_dir,
                schema=CANONICAL_TRAINING_SCHEMA,
                output_dir=Path(tmpdir) / "exports",
                link_images=False,
            )

            ep_dir = export_dir / "episodes" / "episode_000000"
            exported_state = np.load(ep_dir / "observation.state.npy")
            exported_action = np.load(ep_dir / "action.npy")
            exported_goal_joint = np.load(ep_dir / "skill.goal_position.joint.npy")
            exported_goal_gripper = np.load(ep_dir / "skill.goal_position.gripper.npy")
            exported_gripper_state = np.load(ep_dir / "observation.gripper_state.npy")
            exported_tcp_world = np.load(ep_dir / "observation.tcp.world_xyzrpy.npy")
            exported_goal_tcp_robot = np.load(ep_dir / "skill.goal_position.tcp.robot_xyzrpy.npy")
            with open(export_dir / "manifest.json") as f:
                manifest = json.load(f)

        spec = build_joint_normalization_spec(cfg)
        np.testing.assert_allclose(exported_state, normalize_joint_matrix(state, spec)[None, :])
        np.testing.assert_allclose(exported_action, normalize_joint_matrix(action, spec)[None, :])
        np.testing.assert_allclose(exported_goal_joint, normalize_joint_matrix(goal_joint, spec)[None, :])
        np.testing.assert_allclose(exported_goal_gripper, normalize_gripper_array(np.array([[0.04]], dtype=np.float32), spec))
        np.testing.assert_allclose(exported_gripper_state, normalize_gripper_array(np.array([[0.02]], dtype=np.float32), spec))
        np.testing.assert_allclose(exported_tcp_world, tcp_world[None, :])
        np.testing.assert_allclose(exported_goal_tcp_robot, goal_robot[None, :])
        self.assertEqual(manifest["schema"], CANONICAL_TRAINING_SCHEMA)
        self.assertEqual(manifest["robot_name"], "franka")

    def test_export_sim_raw_dataset_adc_compatible_uses_legacy_world_orientation(self):
        cfg = load_robot_config("franka")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="raw_dataset", fps=20)
            recorder.start_episode("test task", episode_idx=0)
            recorder.record_step(
                state=np.zeros(cfg.total_dofs, dtype=np.float32),
                action=np.zeros(cfg.total_dofs, dtype=np.float32),
                skill_label="move",
                skill_type="move",
                skill_progress=0.5,
                goal_joint=np.zeros(cfg.total_dofs, dtype=np.float32),
                goal_world_xyzrpy=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype=np.float32),
                goal_robot_xyzrpy=np.array([4.0, 5.0, 6.0, 0.4, 0.5, 0.6], dtype=np.float32),
                goal_gripper=0.04,
                tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                tcp_robot_xyzrpy=np.array([0.1, 0.2, 0.3, 0.0, 0.1, 0.2], dtype=np.float32),
                gripper_state=np.array([0.02], dtype=np.float32),
            )
            recorder.end_episode(success=True)
            raw_dir = Path(recorder.finalize())

            export_dir = export_sim_raw_dataset(
                raw_dataset_dir=raw_dir,
                schema=ADC_COMPATIBLE_SCHEMA,
                output_dir=Path(tmpdir) / "exports",
                link_images=False,
            )

            legacy_world = np.load(
                export_dir
                / "episodes"
                / "episode_000000"
                / "skill.goal_position.world_xyzrpy.npy"
            )
            tcp_robot = np.load(
                export_dir
                / "episodes"
                / "episode_000000"
                / "observation.tcp.robot_xyzrpy.npy"
            )
            gripper_state = np.load(
                export_dir
                / "episodes"
                / "episode_000000"
                / "observation.gripper_state.npy"
            )
            manifest = json.loads((export_dir / "manifest.json").read_text())
            ep_dir = export_dir / "episodes" / "episode_000000"
            spec = build_joint_normalization_spec(cfg)

        np.testing.assert_allclose(legacy_world, np.array([[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]], dtype=np.float32))
        np.testing.assert_allclose(tcp_robot, np.array([[0.1, 0.2, 0.3, 0.0, 0.1, 0.2]], dtype=np.float32))
        np.testing.assert_allclose(gripper_state, normalize_gripper_array(np.array([[0.02]], dtype=np.float32), spec))
        self.assertEqual(manifest["schema"], ADC_COMPATIBLE_SCHEMA)
        self.assertFalse((ep_dir / "observation.tcp.world_xyzrpy.npy").exists())
        self.assertFalse((ep_dir / "skill.goal_position.tcp.world_xyzrpy.npy").exists())
        self.assertFalse((ep_dir / "skill.goal_position.tcp.robot_xyzrpy.npy").exists())

    def test_export_manifest_records_full_dof_policy_and_joint_layout(self):
        cfg = load_robot_config("openarm")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="raw_dataset", fps=20)
            recorder.start_episode("openarm stack", episode_idx=0)
            zeros = np.zeros(cfg.total_dofs, dtype=np.float32)
            recorder.record_step(
                state=zeros,
                action=zeros,
                skill_label="stack",
                skill_type="stack",
                skill_progress=0.0,
                goal_joint=zeros,
                goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_gripper=0.0,
                tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
            )
            recorder.end_episode(success=False)
            raw_dir = Path(recorder.finalize())

            export_dir = export_sim_raw_dataset(
                raw_dataset_dir=raw_dir,
                schema=ADC_COMPATIBLE_SCHEMA,
                output_dir=Path(tmpdir) / "exports",
                link_images=False,
            )
            manifest = json.loads((export_dir / "manifest.json").read_text())

        self.assertEqual(manifest["joint_shape_policy"], "full_controllable_dofs")
        self.assertEqual(manifest["arm_joint_names"], cfg.arm_joint_names)
        self.assertEqual(manifest["finger_joint_names"], cfg.finger_joint_names)
        self.assertEqual(manifest["gripper_type"], cfg.gripper_type)
        self.assertEqual(manifest["gripper_state_dim"], 1)
        self.assertIn("skill.goal_position.world_xyzrpy", manifest["field_semantics"])
        self.assertIn("observation.tcp.robot_xyzrpy", manifest["field_semantics"])
        self.assertIn("observation.gripper_state", manifest["field_semantics"])
        self.assertEqual(manifest["episodes"][0]["task_description"], "openarm stack")

    def test_export_dataset_cli_defaults_to_adc_compatible(self):
        script = Path(__file__).resolve().parent.parent / "scripts" / "export_dataset.py"
        result = subprocess.run(
            ["python3", str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--schema", result.stdout)
        self.assertIn("default: adc_compatible", result.stdout)

    def test_preprocess_exported_dataset_creates_per_episode_split_manifests(self):
        cfg = load_robot_config("franka")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="raw_dataset", fps=20)
            zeros = np.zeros(cfg.total_dofs, dtype=np.float32)
            ones = np.ones(cfg.total_dofs, dtype=np.float32) * 0.01
            for ep_idx, success in enumerate([True, False]):
                recorder.start_episode(f"task {ep_idx}", episode_idx=ep_idx)
                recorder.record_step(
                    state=zeros,
                    action=ones,
                    skill_label="pick",
                    skill_type="pick",
                    skill_progress=0.25,
                    goal_joint=ones,
                    goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                    goal_gripper=0.04,
                    tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                    tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                )
                recorder.end_episode(success=success)
            raw_dir = Path(recorder.finalize())

            export_dir = export_sim_raw_dataset(
                raw_dataset_dir=raw_dir,
                schema=ADC_COMPATIBLE_SCHEMA,
                output_dir=Path(tmpdir) / "exports",
                link_images=False,
            )
            preprocess_dir = preprocess_exported_dataset(
                export_dir=export_dir,
                output_dir=Path(tmpdir) / "preprocessed",
                train_ratio=0.5,
                seed=7,
            )

            preprocess_manifest = json.loads((preprocess_dir / "manifest.json").read_text())
            samples = [json.loads(line) for line in (preprocess_dir / "samples.jsonl").read_text().splitlines()]
            train_samples = [json.loads(line) for line in (preprocess_dir / "train.jsonl").read_text().splitlines()]
            val_samples = [json.loads(line) for line in (preprocess_dir / "val.jsonl").read_text().splitlines()]

        self.assertEqual(preprocess_manifest["schema"], ADC_COMPATIBLE_SCHEMA)
        self.assertEqual(preprocess_manifest["joint_shape_policy"], "full_controllable_dofs")
        self.assertEqual(preprocess_manifest["num_episodes"], 2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(len(train_samples), 1)
        self.assertEqual(len(val_samples), 1)
        self.assertIn("paths", samples[0])
        self.assertIn("gripper_state", samples[0]["paths"])
        self.assertIn("tcp_robot_xyzrpy", samples[0]["paths"])
        self.assertIn("goal_joint", samples[0]["paths"])
        self.assertEqual(samples[0]["robot_name"], "franka")

    def test_preprocess_exported_dataset_rejects_non_adc_compatible_schema(self):
        cfg = load_robot_config("franka")
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = SimRecorder(cfg, tmpdir, dataset_name="raw_dataset", fps=20)
            zeros = np.zeros(cfg.total_dofs, dtype=np.float32)
            recorder.start_episode("task", episode_idx=0)
            recorder.record_step(
                state=zeros,
                action=zeros,
                skill_label="move",
                skill_type="move",
                skill_progress=0.0,
                goal_joint=zeros,
                goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
                goal_gripper=0.04,
                tcp_world_xyzrpy=np.zeros(6, dtype=np.float32),
                tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
            )
            recorder.end_episode(success=True)
            raw_dir = Path(recorder.finalize())

            export_dir = export_sim_raw_dataset(
                raw_dataset_dir=raw_dir,
                schema=CANONICAL_TRAINING_SCHEMA,
                output_dir=Path(tmpdir) / "exports",
                link_images=False,
            )

            with self.assertRaises(ValueError):
                preprocess_exported_dataset(
                    export_dir=export_dir,
                    output_dir=Path(tmpdir) / "preprocessed",
                )

    def test_compose_adc_legacy_world_xyzrpy_uses_world_position_and_robot_orientation(self):
        world = np.array([[0.1, 0.2, 0.3, 9.0, 9.0, 9.0]], dtype=np.float32)
        robot = np.array([[1.1, 1.2, 1.3, 0.4, 0.5, 0.6]], dtype=np.float32)
        legacy = compose_adc_legacy_world_xyzrpy(world, robot)
        np.testing.assert_allclose(legacy, np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32))

    def test_compute_adc_tcp_world_xyzrpy_applies_frame_rotation(self):
        yaw_90 = np.pi / 2.0
        rot_z_90 = np.array(
            [
                [np.cos(yaw_90), -np.sin(yaw_90), 0.0],
                [np.sin(yaw_90), np.cos(yaw_90), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        ctx = type("ADCContext", (), {})()
        ctx.arm_dofs = 5
        ctx.calibration_limits = type(
            "Calib",
            (),
            {"normalized_to_radians": staticmethod(lambda values: np.asarray(values, dtype=np.float32))},
        )()
        ctx.kinematics = _FakeADCForwardKinematics([0.1, 0.0, 0.2], np.eye(3, dtype=np.float32))
        ctx.frame_name = "world"
        ctx.frame_transformer = type(
            "Frames",
            (),
            {
                "has_frame": staticmethod(lambda name: True),
                "frames": {
                    "world": {
                        "T_frame_from_base": np.array(
                            [
                                [0.0, -1.0, 0.0, 1.0],
                                [1.0, 0.0, 0.0, 2.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        )
                    }
                },
            },
        )()

        tcp_world = compute_adc_tcp_world_xyzrpy(ctx, np.zeros(6, dtype=np.float32))

        np.testing.assert_allclose(tcp_world[:3], np.array([1.0, 2.1, 0.2], dtype=np.float32), atol=1e-6)
        np.testing.assert_allclose(tcp_world[3:], np.array([0.0, 0.0, yaw_90], dtype=np.float32), atol=1e-6)

    def test_resolve_adc_local_path_maps_legacy_absolute_paths_by_anchor(self):
        adc_root = Path("/tmp/adc")
        resolved = _resolve_adc_local_path(
            "/tmp/legacy/lerobot_CaP_distillation/assets/urdf/so101_robot3.urdf",
            adc_root,
        )
        self.assertEqual(resolved, adc_root / "assets" / "urdf" / "so101_robot3.urdf")

        resolved = _resolve_adc_local_path(
            "/some/other/root/AutoDataCollector/robot_configs/robot/so101_robot3.yaml",
            adc_root,
        )
        self.assertEqual(resolved, adc_root / "robot_configs" / "robot" / "so101_robot3.yaml")

    def test_convert_raw_dataset_to_lerobot_delegates_to_converter(self):
        with patch(
            "src.data_collection.lerobot_tools.convert_to_lerobot",
            return_value="/tmp/lerobot/local/sim_dataset",
        ) as convert_mock:
            dataset_root = convert_raw_dataset_to_lerobot(
                "/tmp/raw_dataset",
                repo_id="local/sim_dataset",
                output_root="/tmp/lerobot",
            )

        convert_mock.assert_called_once_with(
            raw_dataset_dir="/tmp/raw_dataset",
            repo_id="local/sim_dataset",
            output_root="/tmp/lerobot",
        )
        self.assertEqual(dataset_root, Path("/tmp/lerobot/local/sim_dataset"))

    def test_check_lerobot_dataset_loads_local_dataset_and_validates_required_fields(self):
        sample = {
            "observation.state": np.zeros(9, dtype=np.float32),
            "observation.gripper_state": np.zeros(1, dtype=np.float32),
            "observation.tcp.robot_xyzrpy": np.zeros(6, dtype=np.float32),
            "action": np.zeros(9, dtype=np.float32),
            "skill.goal_position.robot_xyzrpy": np.zeros(6, dtype=np.float32),
        }
        features = {key: {"dtype": "float32"} for key in sample.keys()}
        _fake_cls, fake_modules = _install_fake_lerobot_dataset(sample, features)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "local" / "sim_dataset"
            (dataset_root / "meta").mkdir(parents=True)
            (dataset_root / "data").mkdir()

            with patch.dict("sys.modules", fake_modules, clear=False):
                report = check_lerobot_dataset(
                    dataset_root,
                    repo_id="local/sim_dataset",
                )

        self.assertTrue(report["pass"])
        self.assertEqual(report["repo_id"], "local/sim_dataset")
        self.assertIn("observation.gripper_state", report["feature_keys"])
        self.assertIn("skill.goal_position.robot_xyzrpy", report["sample_keys"])
        self.assertEqual(report["num_frames"], 1)

    def test_publish_lerobot_dataset_uses_hf_api_large_folder_upload(self):
        api_mock = MagicMock()
        api_mock.create_repo.return_value = "https://huggingface.co/datasets/org/test-dataset"

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "local" / "sim_dataset"
            dataset_root.mkdir(parents=True)

            with (
                patch(
                    "src.data_collection.lerobot_tools.check_lerobot_dataset",
                    return_value={"pass": True, "repo_id": "local/sim_dataset"},
                ),
                patch("huggingface_hub.HfApi", return_value=api_mock) as api_cls,
            ):
                report = publish_lerobot_dataset(
                    dataset_root,
                    repo_id="org/test-dataset",
                    private=True,
                    token="hf_test_token",
                    local_repo_id="local/sim_dataset",
                )

        api_cls.assert_called_once_with(token="hf_test_token")
        api_mock.create_repo.assert_called_once_with(
            repo_id="org/test-dataset",
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        api_mock.upload_large_folder.assert_called_once_with(
            repo_id="org/test-dataset",
            repo_type="dataset",
            folder_path=dataset_root.resolve(),
            private=True,
            print_report=False,
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["repo_id"], "org/test-dataset")
        self.assertEqual(report["validation_report"]["repo_id"], "local/sim_dataset")

    def test_tracked_robot_asset_configs_do_not_embed_user_specific_paths(self):
        for asset_cfg in (
            Path("assets/robots/so101/config.yaml"),
            Path("assets/robots/ur10e_robotiq_2f85/config.yaml"),
        ):
            text = asset_cfg.read_text()
            self.assertIsNone(re.search(r"/home/[^/]+", text))
            self.assertNotIn(str(Path.cwd().resolve()), text)

    def test_multiview_grounder_fuses_depth_candidates_into_robot_pose(self):
        depth_front = np.full((64, 64), np.nan, dtype=np.float32)
        depth_top = np.full((64, 64), np.nan, dtype=np.float32)
        depth_wrist = np.full((64, 64), np.nan, dtype=np.float32)
        center = 32
        depth_front[center, center] = 1.00
        depth_top[center, center] = 0.98
        depth_wrist[center, center] = 1.02

        cameras = _FakeGroundingCameras(
            {
                "front": depth_front,
                "top": depth_top,
                "wrist": depth_wrist,
            }
        )
        robot = _FakeGroundingRobot()
        grounder = MultiViewTargetGrounder(cameras=cameras, robot_interface=robot)
        grounder.begin_episode(0)

        result = grounder.ground_target(
            "drawer_handle_top",
            {
                "position": [0.005, -0.005, -1.0],
                "target_rotation_world": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            },
            stage="handle_open",
        )

        self.assertIsNotNone(result)
        self.assertEqual(set(result["contributing_views"]), {"front", "top", "wrist"})
        position_robot = np.asarray(result["position_robot"], dtype=np.float64)
        self.assertEqual(position_robot.shape, (3,))
        self.assertAlmostEqual(position_robot[2], -1.0, places=2)
        self.assertEqual(len(grounder.events), 1)
        self.assertEqual(grounder.events[0]["status"], "ok")

    def test_task_aware_prompt_renders_pick_place_success_checklist(self):
        task_doc = load_task_document("tasks/franka/pick_place/franka_pick_place_box.yaml")
        translated = {
            "box": {"position": [0.40, -0.15, 0.18], "task_role": "goal_subject", "asset_label": "sugar box"},
            "target_marker": {"position": [0.50, 0.15, 0.02], "task_role": "target_marker", "asset_label": "green target marker"},
            "table_surface": {"position": [0.50, 0.00, 0.00], "task_role": "support_surface", "asset_label": "table surface"},
            "command_pose": {"position": [0.50, 0.00, 0.37], "task_role": "placement_target", "asset_label": "command pose"},
        }
        conditions = _compile_goal_conditions_for_prompt(task_doc, translated)
        relevant = _select_relevant_objects_for_prompt(translated, conditions)

        prompt, context = build_task_aware_judge_prompt(
            task_description=task_doc["task"]["description"],
            goal_description=task_doc["goal"]["description"],
            goal_conditions=conditions,
            relevant_objects=relevant,
            object_positions=translated,
            executed_code="skills.execute_pick_and_place_on_target('box', 'target_marker')",
            image_resolution=(640, 480),
        )

        self.assertIn("Structured Success Criteria:", prompt)
        self.assertIn("box (sugar box) is near target_marker (green target marker)", prompt)
        self.assertIn("box (sugar box) is resting on table and not dropped or floating.", prompt)
        self.assertIn("Relevant Visible Objects and Targets:", prompt)
        self.assertNotIn("command_pose", prompt)
        self.assertEqual(context["goal_description"], task_doc["goal"]["description"])

    def test_task_aware_prompt_renders_stack_tray_and_filters_helper_targets(self):
        task_doc = load_task_document("tasks/franka/stack/franka_stack_tray.yaml")
        translated = {
            "cube_1": {"position": [0.46, -0.06, 0.02], "task_role": "goal_subject", "asset_label": "blue cube"},
            "cube_2": {"position": [0.52, -0.01, 0.02], "task_role": "goal_subject", "asset_label": "red cube"},
            "cube_3": {"position": [0.58, 0.04, 0.02], "task_role": "goal_subject", "asset_label": "green cube"},
            "tray_base": {"position": [0.50, 0.00, 0.00], "task_role": "goal_target", "asset_label": "tray"},
            "tray_anchor": {"position": [0.50, 0.00, 0.00], "task_role": "placement_target", "asset_label": "tray anchor"},
        }
        conditions = _compile_goal_conditions_for_prompt(task_doc, translated)
        relevant = _select_relevant_objects_for_prompt(translated, conditions)

        prompt, _ = build_task_aware_judge_prompt(
            task_description=task_doc["task"]["description"],
            goal_description=task_doc["goal"]["description"],
            goal_conditions=conditions,
            relevant_objects=relevant,
            object_positions=translated,
            executed_code="skills.execute_pick_and_stack_on_object('cube_2', 'cube_1')",
            image_resolution=(640, 480),
        )

        self.assertIn("cube_2 (red cube) is visibly stacked on top of cube_1 (blue cube).", prompt)
        self.assertIn("cube_3 (green cube) is visibly stacked on top of cube_2 (red cube).", prompt)
        self.assertIn("all remain inside the tray boundary", prompt)
        self.assertIn("tray_base (tray)", prompt)
        self.assertNotIn("tray_anchor", prompt)

    def test_task_aware_prompt_renders_lift_command_pose_without_object_listing(self):
        task_doc = load_task_document("tasks/franka/lift/franka_lift.yaml")
        translated = {
            "cube": {"position": [0.50, 0.00, 0.11], "task_role": "goal_subject", "asset_label": "DexCube"},
            "command_pose": {"position": [0.50, 0.00, 0.37], "task_role": "placement_target", "asset_label": "command pose"},
        }
        conditions = _compile_goal_conditions_for_prompt(task_doc, translated)
        relevant = _select_relevant_objects_for_prompt(translated, conditions)

        prompt, _ = build_task_aware_judge_prompt(
            task_description=task_doc["task"]["description"],
            goal_description=task_doc["goal"]["description"],
            goal_conditions=conditions,
            relevant_objects=relevant,
            object_positions=translated,
            executed_code="skills.execute_pick_and_lift_to_pose('cube', 'command_pose')",
            image_resolution=(640, 480),
        )

        self.assertIn("cube (DexCube) is lifted at least 0.04 m", prompt)
        self.assertIn("the commanded target pose approximately at (0.50, 0.00, 0.37)", prompt)
        self.assertNotIn("Relevant Visible Objects and Targets:\n- command_pose", prompt)

    def test_task_aware_prompt_renders_multi_object_checklist_for_cabinet_blocks(self):
        task_doc = load_task_document("tasks/franka/cabinet/franka_cabinet_blocks.yaml")
        translated = {
            "blue_block": {"position": [0.53, 0.20, 0.04], "task_role": "goal_subject", "asset_label": "blue block"},
            "red_block": {"position": [0.55, 0.18, 0.04], "task_role": "goal_subject", "asset_label": "red block"},
            "green_block": {"position": [0.57, 0.16, 0.04], "task_role": "goal_subject", "asset_label": "green block"},
            "target_zone": {"position": [0.65, 0.00, 0.01], "task_role": "goal_target", "asset_label": "target zone"},
        }
        conditions = _compile_goal_conditions_for_prompt(task_doc, translated)
        relevant = _select_relevant_objects_for_prompt(translated, conditions)

        prompt, _ = build_task_aware_judge_prompt(
            task_description=task_doc["task"]["description"],
            goal_description=task_doc["goal"]["description"],
            goal_conditions=conditions,
            relevant_objects=relevant,
            object_positions=translated,
            executed_code="skills.execute_pick_and_place_on_target('blue_block', 'target_zone')",
            image_resolution=(640, 480),
        )

        self.assertIn("blue_block (blue block) is near target_zone (target zone)", prompt)
        self.assertIn("red_block (red block) is near target_zone (target zone)", prompt)
        self.assertIn("green_block (green block) is near target_zone (target zone)", prompt)

    def test_task_aware_prompt_renders_color_sort_and_line_arrange_markers(self):
        for task_path, subject_prefix in (
            ("tasks/franka/sort/franka_color_sort.yaml", "blue_block"),
            ("tasks/franka/sort/franka_line_arrange.yaml", "blue_block"),
        ):
            task_doc = load_task_document(task_path)
            translated = {
                "blue_block": {"position": [0.45, -0.05, 0.02], "task_role": "goal_subject", "asset_label": "blue block"},
                "red_block": {"position": [0.50, 0.00, 0.02], "task_role": "goal_subject", "asset_label": "red block"},
                "green_block": {"position": [0.55, 0.05, 0.02], "task_role": "goal_subject", "asset_label": "green block"},
                "blue_zone": {"position": [0.60, -0.06, 0.01], "task_role": "target_marker", "asset_label": "blue zone"},
                "red_zone": {"position": [0.60, 0.00, 0.01], "task_role": "target_marker", "asset_label": "red zone"},
                "green_zone": {"position": [0.60, 0.06, 0.01], "task_role": "target_marker", "asset_label": "green zone"},
                "pos_marker_1": {"position": [0.55, -0.06, 0.01], "task_role": "target_marker", "asset_label": "position marker 1"},
                "pos_marker_2": {"position": [0.55, 0.00, 0.01], "task_role": "target_marker", "asset_label": "position marker 2"},
                "pos_marker_3": {"position": [0.55, 0.06, 0.01], "task_role": "target_marker", "asset_label": "position marker 3"},
            }
            conditions = _compile_goal_conditions_for_prompt(task_doc, translated)
            relevant = _select_relevant_objects_for_prompt(translated, conditions)
            prompt, _ = build_task_aware_judge_prompt(
                task_description=task_doc["task"]["description"],
                goal_description=task_doc["goal"]["description"],
                goal_conditions=conditions,
                relevant_objects=relevant,
                object_positions=translated,
                executed_code="pass",
                image_resolution=(640, 480),
            )
            self.assertIn(subject_prefix, prompt)
            self.assertIn("Structured Success Criteria:", prompt)

    def test_task_aware_prompt_smoke_covers_requested_franka_task_set(self):
        task_paths = [
            "tasks/franka/lift/franka_lift.yaml",
            "tasks/franka/lift/franka_lift_sugar_box.yaml",
            "tasks/franka/pick_place/franka_pick_place.yaml",
            "tasks/franka/pick_place/franka_pick_place_bottle.yaml",
            "tasks/franka/pick_place/franka_pick_place_box.yaml",
            "tasks/franka/pick_place/franka_pick_place_can.yaml",
            "tasks/franka/pick_place/franka_pick_place_drawer.yaml",
            "tasks/franka/pick_place/franka_pick_place_mug.yaml",
            "tasks/franka/pick_place/franka_pick_place_tuna.yaml",
            "tasks/franka/stack/franka_stack.yaml",
            "tasks/franka/cabinet/franka_cabinet_blocks.yaml",
            "tasks/franka/pick_place/franka_multi_pick_place.yaml",
            "tasks/franka/sort/franka_color_sort.yaml",
            "tasks/franka/sort/franka_line_arrange.yaml",
            "tasks/franka/stack/franka_stack_tray.yaml",
        ]

        for task_path in task_paths:
            with self.subTest(task_path=task_path):
                task_doc = load_task_document(task_path)
                translated, conditions, relevant = _build_prompt_inputs_for_task(task_doc)
                prompt, context = build_task_aware_judge_prompt(
                    task_description=task_doc.get("task", {}).get("description", ""),
                    goal_description=task_doc.get("goal", {}).get("description", ""),
                    goal_conditions=conditions,
                    relevant_objects=relevant,
                    object_positions=translated,
                    executed_code="pass",
                    image_resolution=(640, 480),
                )
                self.assertIn("Structured Success Criteria:", prompt)
                self.assertTrue(context["goal_conditions"])
                self.assertNotIn("No objects detected", prompt)
                self.assertNotIn("_anchor", prompt)


if __name__ == "__main__":
    unittest.main()
