"""Regression tests for evaluator/path/pipeline fixes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data_collection.config import load_robot_config
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
        self.assertIn("/Simulation-Generation-Agent/assets/robots/so101/so101.usd", resolved)
        self.assertNotIn("/Simulation-Generation-Agent/src/assets/", resolved)

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

    def test_load_robot_config_maps_ur10_alias_to_ur10e_profile(self):
        cfg = load_robot_config("ur10")
        self.assertEqual(cfg.name, "ur10e")
        self.assertTrue(cfg.has_gripper_joints)
        self.assertEqual(len(cfg.finger_joint_names), 6)


if __name__ == "__main__":
    unittest.main()
