"""Regression tests for evaluator/path/pipeline fixes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.isaac_lab.evaluator.mdp_correctness import MDPCorrectnessChecker
from src.isaac_lab.evaluator.parser import EnvCfgParser
from src.isaac_lab.evaluator.runtime_validity import RuntimeValidityChecker
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


if __name__ == "__main__":
    unittest.main()
