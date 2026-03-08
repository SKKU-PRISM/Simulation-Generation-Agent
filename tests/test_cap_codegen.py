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
from src.data_collection.cap_runtime.skills.skills_openarm import OpenArmSkills
from src.data_collection.config import DataCollectionConfig, load_robot_config
from src.data_collection.pipeline import DataCollectionPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


class _DummyLLM:
    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str, **_) -> str:
        del system_prompt
        assert "execute_task" in user_prompt
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
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
        self.assertNotIn("grasp_face", franka.api_doc)
        self.assertNotIn("grasp_face", openarm.api_doc)
        self.assertNotIn("grasp_yaw_deg", openarm.api_doc)

    def test_code_example_matches_profile_api(self):
        franka_generator = SimCaPGenerator(_DummyLLM(), load_robot_config("franka"))
        openarm_generator = SimCaPGenerator(_DummyLLM(), load_robot_config("openarm"))

        franka_example = franka_generator._build_code_example()
        openarm_example = openarm_generator._build_code_example()

        self.assertNotIn("grasp_face", franka_example)
        self.assertNotIn("grasp_yaw_deg", franka_example)
        self.assertNotIn("grasp_face", openarm_example)
        self.assertNotIn("grasp_yaw_deg", openarm_example)

    def test_openarm_generator_rejects_policy_level_grasp_overrides(self):
        generator = SimCaPGenerator(_DummyLLM(), load_robot_config("openarm"))
        for forbidden_arg in ("approach_angle_deg=35.0", 'grasp_face="top"', "grasp_yaw_deg=0.0"):
            code = (
                "from skills.skills_openarm import OpenArmSkills\n\n"
                "def execute_task():\n"
                "    skills = OpenArmSkills(frame=\"world\")\n"
                "    skills.connect()\n"
                "    try:\n"
                "        skills.move_to_ready(skill_description=\"ready\")\n"
                "        skills.execute_pick_object(\n"
                "            positions[\"cube_1\"][\"position\"],\n"
                "            object_name=\"cube_1\",\n"
                f"            {forbidden_arg},\n"
                "            skill_description=\"pick cube_1\",\n"
                "        )\n"
                "    finally:\n"
                "        skills.disconnect()\n\n"
                "if __name__ == \"__main__\":\n"
                "    execute_task()\n"
            )

            with self.assertRaises(ValueError):
                generator._validate_generated_code(code)

    def test_translate_scene_state_adds_crowding_affordances(self):
        task_path = REPO_ROOT / "tasks" / "openarm" / "stack" / "openarm_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.30, 0.00, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.35, 0.00, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_3": {"position": [0.40, 0.00, 0.0203], "quaternion": [1, 0, 0, 0]},
        }

        translated = translate_scene_state(task_doc, scene_state)
        affordances = translated["cube_2"]["affordances"]

        self.assertEqual(affordances["preferred_grasp"], "top_down")
        self.assertTrue(affordances["stable_top_grasp_feasible"])
        self.assertTrue(affordances["blocked_side_grasp"])
        self.assertIn("cube_1", affordances["crowded_neighbors"])
        self.assertIn("cube_3", affordances["crowded_neighbors"])

    def test_openarm_prompt_includes_affordance_summary(self):
        generator = SimCaPGenerator(_DummyLLM(), load_robot_config("openarm"))
        task_path = REPO_ROOT / "tasks" / "openarm" / "stack" / "openarm_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.30, 0.00, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.35, 0.00, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_3": {"position": [0.40, 0.00, 0.0203], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)

        prompt = generator._build_user_prompt(task_doc, translated)
        self.assertIn("### Affordance Summary", prompt)
        self.assertIn("preferred_grasp=top_down", prompt)
        self.assertIn("blocked_side_grasp=true", prompt)

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

    def test_sim_cap_generator_retries_invalid_response_until_valid_code(self):
        robot_cfg = load_robot_config("franka")
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        llm = _DummyLLM(
            responses=[
                "I'm sorry, but I cannot assist with that request.",
                (
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
                ),
            ]
        )
        generator = SimCaPGenerator(llm, robot_cfg, retry_max=2)
        scene_state = {
            "cube_1": {"position": [0.45, 0.02, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.52, -0.04, 0.0203], "quaternion": [1, 0, 0, 0]},
        }

        result = generator.generate_code(task_doc, scene_state)
        self.assertIn("FrankaSkills", result.generated_code)
        self.assertEqual(llm.calls, 2)


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
        self.assertIn('getattr(super(), "__post_init__", None)', runner_text)
        self.assertIn('if "ImplicitActuatorCfg(" in line', runner_text)
        self.assertIn('line.replace("joint_names=", "joint_names_expr=")', runner_text)
        self.assertIn("for orientation_key in (", runner_text)
        self.assertIn('"orientation_range"', runner_text)
        self.assertIn('"orientation"', runner_text)
        self.assertIn('"rotation_range"', runner_text)
        self.assertIn('"rotation"', runner_text)
        self.assertIn("dropped unsupported {removed_orientation_keys}", runner_text)
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
    PALM_DOWN_ROTATION = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, ee_position=None):
        self.calls = []
        if ee_position is None:
            ee_position = [0.40, 0.00, 0.10]
        self.robot = _FakeRobot(ee_position)

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

    def execute_pick(
        self,
        object_name,
        approach_offset=0.10,
        approach_angle_deg=None,
        target_rotation_world=None,
        approach_direction_world=None,
        required_tool_axis_world=None,
        required_tool_axis_tolerance_deg=25.0,
        allow_position_only_fallback=True,
        skill_description=None,
    ):
        self.calls.append(
            (
                "execute_pick",
                object_name,
                approach_offset,
                approach_angle_deg,
                None
                if target_rotation_world is None
                else np.asarray(target_rotation_world, dtype=np.float64).tolist(),
                None
                if approach_direction_world is None
                else np.asarray(approach_direction_world, dtype=np.float64).tolist(),
                None
                if required_tool_axis_world is None
                else np.asarray(required_tool_axis_world, dtype=np.float64).tolist(),
                required_tool_axis_tolerance_deg,
                allow_position_only_fallback,
                skill_description,
            )
        )
        return True

    def execute_place(
        self,
        target_position,
        approach_offset=0.05,
        drop_offset=0.005,
        _placed_object=None,
        _held_xy_offset_world=None,
        approach_angle_deg=None,
        target_rotation_world=None,
        approach_direction_world=None,
        required_tool_axis_world=None,
        required_tool_axis_tolerance_deg=25.0,
        allow_position_only_fallback=True,
        skill_description=None,
    ):
        self.calls.append(
            (
                "execute_place",
                np.asarray(target_position, dtype=np.float64).tolist(),
                approach_offset,
                drop_offset,
                _placed_object,
                None
                if _held_xy_offset_world is None
                else np.asarray(_held_xy_offset_world, dtype=np.float64).tolist(),
                approach_angle_deg,
                None
                if target_rotation_world is None
                else np.asarray(target_rotation_world, dtype=np.float64).tolist(),
                None
                if approach_direction_world is None
                else np.asarray(approach_direction_world, dtype=np.float64).tolist(),
                None
                if required_tool_axis_world is None
                else np.asarray(required_tool_axis_world, dtype=np.float64).tolist(),
                required_tool_axis_tolerance_deg,
                allow_position_only_fallback,
                skill_description,
            )
        )
        return True


class _FakeRobot:
    def __init__(self, ee_position):
        self._ee_position = np.asarray(ee_position, dtype=np.float64)

    def read_ee_pose(self):
        return self._ee_position.copy(), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


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

    def test_openarm_runtime_places_without_ready_pose_transit(self):
        robot_cfg = load_robot_config("openarm")
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
            wrapper = OpenArmSkills(frame="world")
            wrapper._held_object_name = "cube_2"
            wrapper._held_half_height = 0.0203
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

        self.assertEqual([call[0] for call in fake_skills.calls], ["execute_place"])
        self.assertAlmostEqual(fake_skills.calls[0][5][0], 0.02, places=6)
        self.assertAlmostEqual(fake_skills.calls[0][5][1], -0.02, places=6)
        self.assertIsNone(fake_skills.calls[0][7])
        self.assertIsNone(fake_skills.calls[0][8])
        self.assertIsNone(fake_skills.calls[0][9])

    def test_openarm_runtime_tracks_held_xy_offset_for_place(self):
        robot_cfg = load_robot_config("openarm")
        fake_skills = _FakeSimSkills(ee_position=[0.39, -0.03, 0.10])
        detector = _FakeDetector(
            {
                "cube_1": [0.50, 0.00, 0.0203],
                "cube_2": [0.41, -0.01, 0.0609],
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
            wrapper = OpenArmSkills(frame="world")
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
        self.assertEqual(fake_skills.calls[1][0], "execute_place")
        self.assertIsNotNone(fake_skills.calls[0][4])
        self.assertEqual(np.round(fake_skills.calls[0][5], 6).tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(np.round(fake_skills.calls[0][6], 6).tolist(), [0.0, 0.0, -1.0])
        self.assertFalse(fake_skills.calls[0][8])
        self.assertAlmostEqual(fake_skills.calls[1][5][0], 0.02, places=6)
        self.assertAlmostEqual(fake_skills.calls[1][5][1], 0.02, places=6)
        self.assertIsNotNone(fake_skills.calls[1][7])
        self.assertEqual(np.round(fake_skills.calls[1][8], 6).tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(np.round(fake_skills.calls[1][9], 6).tolist(), [0.0, 0.0, -1.0])
        self.assertFalse(fake_skills.calls[1][11])

    def test_openarm_hidden_grasp_override_forwards_manual_spec(self):
        robot_cfg = load_robot_config("openarm")
        fake_skills = _FakeSimSkills(ee_position=[0.39, -0.03, 0.10])
        detector = _FakeDetector(
            {
                "cube_1": [0.50, 0.00, 0.0203],
                "cube_2": [0.41, -0.01, 0.0609],
            }
        )
        translated_positions = {
            "cube_1": {
                "position": [0.50, 0.00, 0.0406],
                "estimated_half_height": 0.0203,
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
            "cube_2": {
                "position": [0.42, -0.02, 0.0406],
                "estimated_half_height": 0.0203,
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )
        try:
            wrapper = OpenArmSkills(frame="world")
            self.assertTrue(
                wrapper.execute_pick_object(
                    [0.42, -0.02, 0.0406],
                    object_name="cube_2",
                    grasp_face="front",
                    grasp_yaw_deg=45.0,
                    skill_description="pick cube_2 from the front face",
                )
            )
            self.assertTrue(
                wrapper.execute_place_object(
                    [0.50, 0.00, 0.0406],
                    target_name="cube_1",
                    is_table=False,
                    grasp_face=None,
                    grasp_yaw_deg=None,
                    skill_description="place cube_2 while preserving the carried grasp frame",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        pick_call = fake_skills.calls[0]
        place_call = fake_skills.calls[1]
        self.assertEqual(pick_call[0], "execute_pick")
        self.assertIsNone(pick_call[3])
        self.assertFalse(pick_call[8])
        self.assertEqual(np.round(pick_call[5], 6).tolist(), [1.0, 0.0, 0.0])
        pick_rotation = np.asarray(pick_call[4], dtype=np.float64)
        self.assertEqual(np.round(pick_rotation[:, 2], 6).tolist(), [-1.0, 0.0, 0.0])
        self.assertEqual(np.round(pick_call[6], 6).tolist(), [-1.0, 0.0, 0.0])

        self.assertEqual(place_call[0], "execute_place")
        self.assertIsNone(place_call[6])
        self.assertFalse(place_call[11])
        self.assertEqual(place_call[8], pick_call[5])
        self.assertEqual(place_call[7], pick_call[4])
        self.assertEqual(place_call[9], pick_call[6])

    def test_openarm_runtime_rejects_invalid_grasp_face(self):
        robot_cfg = load_robot_config("openarm")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube_2": [0.42, -0.02, 0.0203]})
        translated_positions = {
            "cube_2": {
                "position": [0.42, -0.02, 0.0406],
                "estimated_half_height": 0.0203,
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )
        try:
            wrapper = OpenArmSkills(frame="world")
            with self.assertRaises(PolicyFallbackBlockedError):
                wrapper.execute_pick_object(
                    [0.42, -0.02, 0.0406],
                    object_name="cube_2",
                    grasp_face="diagonal",
                    skill_description="pick cube_2 with an invalid grasp face",
                )
        finally:
            CaPRuntimeContext.clear()


if __name__ == "__main__":
    unittest.main()
