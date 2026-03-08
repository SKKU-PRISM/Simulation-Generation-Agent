"""Regression tests for evaluator/path/pipeline fixes."""

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.common.task_docs import load_task_document, resolve_task_document
from src.data_collection.config import load_robot_config
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
from src.data_collection.sim_recorder import SimRecorder, _coerce_goal_joint
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


class _FakeADCForwardKinematics:
    def __init__(self, position, rotation):
        self._position = np.asarray(position, dtype=np.float32)
        self._rotation = np.asarray(rotation, dtype=np.float32)

    def forward_kinematics(self, joint_positions):
        return self._position.copy(), self._rotation.copy()


class RegressionTests(unittest.TestCase):
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
        self.assertIn("orientation", assets["shape_12"]["randomize"])

    def test_assembling_kits_template_uses_static_scene_assets(self):
        doc = load_task_document("tasks/franka/assembly/franka_assembling_kits.yaml")
        code = build_assembling_kits_template(doc, "franka")
        env_cfg = code["env_cfg.py"]

        self.assertIn("shape_09 = AssetBaseCfg(", env_cfg)
        self.assertIn("shape_12 = AssetBaseCfg(", env_cfg)
        self.assertNotIn("shape_09 = RigidObjectCfg(", env_cfg)
        self.assertIn("/assets/assembling_kits/kit_303.usd", env_cfg)

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

    def test_load_robot_config_exposes_openarm_tcp_frame(self):
        cfg = load_robot_config("openarm")
        self.assertEqual(cfg.ee_frame_body, "openarm_hand")
        self.assertEqual(cfg.ee_frame_tcp, "openarm_ee_tcp")
        self.assertEqual(cfg.ik_ee_frame, "openarm_hand_tcp")
        self.assertTrue(Path(cfg.urdf_path).is_absolute())
        self.assertTrue(Path(cfg.urdf_path).exists())

    def test_load_robot_config_exposes_so101_tcp_offset(self):
        cfg = load_robot_config("so101")
        self.assertEqual(cfg.ee_frame_body, "gripper_frame_link")
        self.assertEqual(cfg.ee_frame_offset_position, [0.04, 0.0, 0.0])
        self.assertEqual(cfg.grasp_lateral_bias, 0.0)

    def test_create_ik_solver_prefers_pinocchio_when_openarm_urdf_exists(self):
        cfg = load_robot_config("openarm")
        with patch("src.data_collection.sim_skills.PinocchioIKSolver", return_value="pin-ik") as mock_solver:
            solver = create_ik_solver(cfg)
        self.assertEqual(solver, "pin-ik")
        mock_solver.assert_called_once_with(
            urdf_path=cfg.urdf_path,
            ee_frame=cfg.ik_ee_frame or cfg.ee_frame_tcp or cfg.ee_frame_body,
            joint_names=cfg.arm_joint_names,
            tcp_offset=cfg.ee_frame_offset_position or None,
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
            )
            recorder.end_episode(success=True)

            ep_dir = Path(tmpdir) / "test_dataset" / "episodes" / "episode_000000"
            tcp_world = np.load(ep_dir / "tcp_world_xyzrpy.npy")
            tcp_robot = np.load(ep_dir / "tcp_robot_xyzrpy.npy")
            with open(ep_dir / "skills.json") as f:
                skills = json.load(f)

        self.assertEqual(tcp_world.shape, (1, 6))
        self.assertEqual(tcp_robot.shape, (1, 6))
        self.assertEqual(len(skills[0]["goal_joint"]), cfg.total_dofs)
        np.testing.assert_allclose(
            skills[0]["goal_world_xyzrpy"],
            [0.4, 0.1, 0.2, 0.0, 0.1, 0.2],
        )
        np.testing.assert_allclose(
            skills[0]["goal_robot_xyzrpy"],
            [0.1, -0.2, 0.3, -0.1, 0.2, -0.3],
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
            exported_tcp_world = np.load(ep_dir / "observation.tcp.world_xyzrpy.npy")
            exported_goal_tcp_robot = np.load(ep_dir / "skill.goal_position.tcp.robot_xyzrpy.npy")
            with open(export_dir / "manifest.json") as f:
                manifest = json.load(f)

        spec = build_joint_normalization_spec(cfg)
        np.testing.assert_allclose(exported_state, normalize_joint_matrix(state, spec)[None, :])
        np.testing.assert_allclose(exported_action, normalize_joint_matrix(action, spec)[None, :])
        np.testing.assert_allclose(exported_goal_joint, normalize_joint_matrix(goal_joint, spec)[None, :])
        np.testing.assert_allclose(exported_goal_gripper, normalize_gripper_array(np.array([[0.04]], dtype=np.float32), spec))
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
                tcp_robot_xyzrpy=np.zeros(6, dtype=np.float32),
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
            manifest = json.loads((export_dir / "manifest.json").read_text())
            ep_dir = export_dir / "episodes" / "episode_000000"

        np.testing.assert_allclose(legacy_world, np.array([[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]], dtype=np.float32))
        self.assertEqual(manifest["schema"], ADC_COMPATIBLE_SCHEMA)
        self.assertFalse((ep_dir / "observation.tcp.world_xyzrpy.npy").exists())
        self.assertFalse((ep_dir / "observation.tcp.robot_xyzrpy.npy").exists())
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
        self.assertIn("skill.goal_position.world_xyzrpy", manifest["field_semantics"])
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

    def test_tracked_robot_asset_configs_do_not_embed_user_specific_paths(self):
        for asset_cfg in (
            Path("assets/robots/so101/config.yaml"),
            Path("assets/robots/ur10e_robotiq_2f85/config.yaml"),
        ):
            text = asset_cfg.read_text()
            self.assertIsNone(re.search(r"/home/[^/]+", text))
            self.assertNotIn(str(Path.cwd().resolve()), text)


if __name__ == "__main__":
    unittest.main()
