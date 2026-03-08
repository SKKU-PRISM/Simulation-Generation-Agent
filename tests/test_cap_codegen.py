import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.data_collection.cap_generator import (
    SimCaPGenerator,
    is_supported_tabletop_task,
    select_cap_profile,
    translate_scene_state,
)
from src.data_collection.cap_runtime import CaPRuntimeContext
from src.data_collection.cap_runtime.skills.base import PolicyFallbackBlockedError
from src.data_collection.cap_runtime.skills.skills_franka import FrankaSkills
from src.data_collection.config import DataCollectionConfig, load_robot_config
from src.data_collection.pipeline import DataCollectionPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyLLM:
    def generate(self, system_prompt: str, user_prompt: str, **_) -> str:
        del system_prompt
        assert "execute_task" in user_prompt
        return (
            "from skills.skills_franka import FrankaSkills\n\n"
            "def execute_task():\n"
            "    skills = FrankaSkills(frame=\"world\")\n"
            "    skills.connect()\n"
            "    try:\n"
            "        skills.move_to_ready(skill_description=\"move to ready\")\n"
            "    finally:\n"
            "        skills.disconnect()\n\n"
            "if __name__ == \"__main__\":\n"
            "    execute_task()\n"
        )


class CapGeneratorTests(unittest.TestCase):
    def test_select_cap_profile_is_robot_specific(self):
        franka = select_cap_profile(load_robot_config("franka"))
        openarm = select_cap_profile(load_robot_config("openarm"))
        self.assertEqual(franka.class_name, "FrankaSkills")
        self.assertEqual(franka.module_name, "skills.skills_franka")
        self.assertEqual(openarm.class_name, "OpenArmSkills")
        self.assertEqual(openarm.module_name, "skills.skills_openarm")

    def test_translate_scene_state_converts_center_z_to_top_surface(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.45, 0.02, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.52, -0.04, 0.0203], "quaternion": [1, 0, 0, 0]},
            "table": {"position": [0.5, 0.0, 0.0], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)
        self.assertAlmostEqual(translated["cube_1"]["position"][2], 0.0406, places=4)
        self.assertAlmostEqual(translated["cube_2"]["position"][2], 0.0406, places=4)
        self.assertNotIn("table", translated)

    def test_unsupported_task_is_rejected(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertFalse(supported)
        self.assertIn("unsupported", reason)

    def test_on_surface_pick_place_task_is_supported(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)

    def test_openarm_stack_is_not_rejected_as_articulation(self):
        task_path = REPO_ROOT / "tasks" / "openarm" / "stack" / "openarm_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)

    def test_translate_scene_state_keeps_static_goal_targets(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_gears.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "gear_small": {"position": [0.40, -0.10, 0.05], "quaternion": [1, 0, 0, 0]},
            "gear_medium": {"position": [0.50, 0.05, 0.05], "quaternion": [1, 0, 0, 0]},
            "m16_nut": {"position": [0.45, -0.02, 0.05], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)
        self.assertIn("tray", translated)
        self.assertEqual(translated["tray"]["position"][:2], [0.55, 0.18])
        self.assertAlmostEqual(translated["tray"]["position"][2], 0.0, places=4)

    def test_sim_cap_generator_returns_code_and_positions(self):
        robot_cfg = load_robot_config("franka")
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        generator = SimCaPGenerator(_DummyLLM(), robot_cfg)
        scene_state = {
            "cube_1": {"position": [0.45, 0.02, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.52, -0.04, 0.0203], "quaternion": [1, 0, 0, 0]},
        }
        result = generator.generate_code(task_doc, scene_state)
        self.assertIn("FrankaSkills", result.generated_code)
        self.assertIn("cube_1", result.translated_positions)


class PipelineRunnerTests(unittest.TestCase):
    def test_generated_runner_uses_cap_generator_not_skill_planner(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack.yaml"
        cfg = DataCollectionConfig(use_vlm_judge=False)
        pipeline = DataCollectionPipeline(
            yaml_path=str(task_path),
            config=cfg,
            env_dir=str(REPO_ROOT),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline.output_dir = Path(tmpdir)
            runner_path = pipeline._generate_collection_runner()
            runner_text = runner_path.read_text(encoding="utf-8")

        self.assertIn("SimCaPGenerator", runner_text)
        self.assertIn("CaPRuntimeContext", runner_text)
        self.assertIn("geometry_success", runner_text)
        self.assertIn("overall_success", runner_text)
        self.assertNotIn("SkillPlanner", runner_text)
        self.assertNotIn("execute_skill_sequence", runner_text)
        self.assertNotIn("_execute_task_skills_fallback", runner_text)
        self.assertNotIn("using fallback", runner_text)


class _FakeDetector:
    def __init__(self, positions: dict[str, list[float]]):
        self._positions = positions

    def get_object_position(self, name: str) -> np.ndarray:
        return np.asarray(self._positions[name], dtype=np.float64)


class _FakeSimSkills:
    def __init__(self):
        self.calls = []

    def move_to_ready(self, **kwargs):
        self.calls.append(("move_to_ready", kwargs))
        return True

    def move_to_position(self, position, **kwargs):
        self.calls.append(("move_to_position", np.asarray(position).tolist(), kwargs))
        return True

    def gripper_open(self, **kwargs):
        self.calls.append(("gripper_open", kwargs))
        return True

    def gripper_close(self, **kwargs):
        self.calls.append(("gripper_close", kwargs))
        return True

    def execute_pick(self, object_name, approach_offset=0.10, skill_description=None):
        self.calls.append(("execute_pick", object_name, approach_offset, skill_description))
        return True

    def execute_place(
        self,
        target_position,
        approach_offset=0.05,
        drop_offset=0.005,
        _placed_object=None,
        skill_description=None,
    ):
        self.calls.append(
            (
                "execute_place",
                np.asarray(target_position, dtype=np.float64).tolist(),
                approach_offset,
                drop_offset,
                _placed_object,
                skill_description,
            )
        )
        return True


class RuntimeWrapperTests(unittest.TestCase):
    def test_franka_runtime_wrapper_uses_simskills_pick_and_place(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector(
            {
                "cube_1": [0.50, 0.00, 0.0203],
                "cube_2": [0.42, -0.02, 0.0203],
            }
        )
        translated_positions = {
            "cube_1": {"position": [0.50, 0.00, 0.0406], "estimated_half_height": 0.0203},
            "cube_2": {"position": [0.42, -0.02, 0.0406], "estimated_half_height": 0.0203},
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )
        try:
            wrapper = FrankaSkills(frame="world")
            self.assertTrue(
                wrapper.execute_pick_object(
                    [0.42, -0.02, 0.0406],
                    object_name="cube_2",
                    skill_description="pick cube_2",
                )
            )
            self.assertTrue(
                wrapper.execute_place_object(
                    [0.50, 0.00, 0.0406],
                    target_name="cube_1",
                    is_table=False,
                    skill_description="place cube_2 on cube_1",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        self.assertEqual(fake_skills.calls[0][0], "execute_pick")
        self.assertEqual(fake_skills.calls[0][1], "cube_2")
        self.assertEqual(fake_skills.calls[1][0], "move_to_ready")
        self.assertEqual(fake_skills.calls[2][0], "execute_place")
        self.assertEqual(fake_skills.calls[2][4], "cube_2")
        self.assertAlmostEqual(fake_skills.calls[2][1][2], 0.0609, places=4)

    def test_runtime_wrapper_rejects_missing_object_name(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube_1": [0.50, 0.00, 0.0203]})
        translated_positions = {
            "cube_1": {"position": [0.50, 0.00, 0.0406], "estimated_half_height": 0.0203},
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )
        try:
            wrapper = FrankaSkills(frame="world")
            with self.assertRaises(PolicyFallbackBlockedError):
                wrapper.execute_pick_object(
                    [0.50, 0.00, 0.0406],
                    object_name=None,
                    skill_description="pick unnamed object",
                )
        finally:
            CaPRuntimeContext.clear()

    def test_runtime_wrapper_rejects_missing_stack_target_name(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube_1": [0.50, 0.00, 0.0203]})
        translated_positions = {
            "cube_1": {"position": [0.50, 0.00, 0.0406], "estimated_half_height": 0.0203},
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )
        try:
            wrapper = FrankaSkills(frame="world")
            wrapper._held_object_name = "cube_1"
            wrapper._held_half_height = 0.0203
            with self.assertRaises(PolicyFallbackBlockedError):
                wrapper.execute_place_object(
                    [0.50, 0.00, 0.0406],
                    target_name=None,
                    is_table=False,
                    skill_description="place without named target",
                )
        finally:
            CaPRuntimeContext.clear()


if __name__ == "__main__":
    unittest.main()
