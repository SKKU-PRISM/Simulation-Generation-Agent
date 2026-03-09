import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.data_collection.cap_generator import (
    SimCaPGenerator,
    assess_task_capability,
    _infer_ordered_objects,
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
from src.data_collection.pipeline import build_task_texture_audit, classify_texture_asset_path, should_preserve_textured_usd


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
    def test_classify_texture_asset_path_flags_proxy_prone_assets(self):
        mug = classify_texture_asset_path("{ISAAC_NUCLEUS_DIR}/Props/Mugs/SM_Mug_A2.usd")
        tuna = classify_texture_asset_path("{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd")

        self.assertEqual(mug["loader_policy"], "preserve_visual_usd_with_injected_physics")
        self.assertEqual(tuna["loader_policy"], "preserve_visual_usd_with_injected_physics")
        self.assertEqual(mug["physics_strategy"], "child_mesh_injection")
        self.assertEqual(mug["physics_target_prim_basename"], "SM_Mug_A2")
        self.assertEqual(tuna["physics_target_prim_basename"], "_07_tuna_fish_can")
        self.assertTrue(should_preserve_textured_usd(mug["asset_path"]))
        self.assertTrue(should_preserve_textured_usd(tuna["asset_path"]))

    def test_build_task_texture_audit_captures_direct_usd_audit_targets(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_can.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        records = build_task_texture_audit(task_doc)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["asset_name"], "can")
        self.assertEqual(records[0]["loader_policy"], "audit_axis_aligned_variant")
        self.assertEqual(records[0]["asset_family"], "tomato_soup_can")

    def test_select_cap_profile_is_robot_specific(self):
        franka = select_cap_profile(load_robot_config("franka"))
        openarm = select_cap_profile(load_robot_config("openarm"))
        self.assertEqual(franka.class_name, "FrankaSkills")
        self.assertEqual(franka.module_name, "skills.skills_franka")
        self.assertEqual(openarm.class_name, "OpenArmSkills")
        self.assertEqual(openarm.module_name, "skills.skills_openarm")
        self.assertIn("execute_pick_and_place_on_target", franka.api_doc)
        self.assertIn("execute_pick_and_stack_on_object", franka.api_doc)
        self.assertIn("execute_pick_and_lift_to_pose", franka.api_doc)
        self.assertIn("execute_pick_and_place_on_support", franka.api_doc)
        self.assertIn("execute_pull_handle_open", franka.api_doc)
        self.assertIn("execute_pick_and_place_upright", franka.api_doc)
        self.assertIn("execute_pick_and_insert_into_target", franka.api_doc)
        self.assertIn("execute_pick_and_fit_into_slot", franka.api_doc)
        self.assertNotIn("grasp_face", franka.api_doc)
        self.assertNotIn("grasp_face", openarm.api_doc)
        self.assertNotIn("grasp_yaw_deg", openarm.api_doc)

    def test_code_example_matches_profile_api(self):
        franka_generator = SimCaPGenerator(_DummyLLM(), load_robot_config("franka"))
        openarm_generator = SimCaPGenerator(_DummyLLM(), load_robot_config("openarm"))

        franka_example = franka_generator._build_code_example()
        openarm_example = openarm_generator._build_code_example()

        self.assertIn("execute_pick_and_place_on_target", franka_example)
        self.assertIn("execute_pick_and_place_on_target", openarm_example)
        self.assertNotIn("grasp_face", franka_example)
        self.assertNotIn("grasp_yaw_deg", franka_example)
        self.assertNotIn("grasp_face", openarm_example)
        self.assertNotIn("grasp_yaw_deg", openarm_example)

    def test_code_example_uses_new_family_skills(self):
        generator = SimCaPGenerator(_DummyLLM(), load_robot_config("franka"))
        self.assertIn("execute_pull_handle_open", generator._build_code_example("articulated_pull"))
        self.assertIn("execute_pick_and_insert_into_target", generator._build_code_example("axial_insertion"))
        self.assertIn("execute_pick_and_fit_into_slot", generator._build_code_example("slot_fit"))
        self.assertIn("execute_pick_and_place_upright", generator._build_code_example("upright_placement"))

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

    def test_translate_scene_state_infers_color_and_task_role_metadata(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack_tray.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.45, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.55, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_3": {"position": [0.60, -0.25, 0.0203], "quaternion": [1, 0, 0, 0]},
            "tray_base": {"position": [0.50, 0.00, 0.0], "quaternion": [1, 0, 0, 0]},
        }

        translated = translate_scene_state(task_doc, scene_state)

        self.assertEqual(translated["cube_1"]["color_name"], "blue")
        self.assertEqual(translated["cube_2"]["color_name"], "red")
        self.assertEqual(translated["cube_3"]["color_name"], "green")
        self.assertEqual(translated["tray_base"]["task_role"], "placement_target")
        self.assertIn("blue block", translated["cube_1"]["aliases"])

    def test_translate_scene_state_adds_tray_anchor_for_inside_tray_tasks(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack_tray.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.45, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.55, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_3": {"position": [0.60, -0.25, 0.0203], "quaternion": [1, 0, 0, 0]},
        }

        translated = translate_scene_state(task_doc, scene_state)

        self.assertIn("tray_anchor", translated)
        self.assertEqual(translated["tray_anchor"]["task_role"], "placement_target")
        self.assertEqual(translated["tray_anchor"]["position"][:2], [0.5, 0.0])
        self.assertIn("tray center", translated["tray_anchor"]["aliases"])

    def test_translate_scene_state_adds_command_pose_for_lift_tasks(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "lift" / "franka_lift.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube": {"position": [0.5, 0.0, 0.055], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)

        self.assertIn("command_pose", translated)
        self.assertEqual(translated["command_pose"]["task_role"], "placement_target")
        self.assertEqual(translated["command_pose"]["aliases"], ["command pose", "target pose"])
        self.assertAlmostEqual(translated["command_pose"]["position"][0], 0.5, places=4)
        self.assertAlmostEqual(translated["command_pose"]["position"][1], 0.0, places=4)
        self.assertAlmostEqual(translated["command_pose"]["position"][2], 0.375, places=4)

    def test_translate_scene_state_adds_drawer_support_and_handle_targets(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(task_doc, {})
        self.assertIn("drawer_handle_top", translated)
        self.assertEqual(translated["drawer_handle_top"]["task_role"], "handle_target")
        self.assertIn("pull_axis_world", translated["drawer_handle_top"])

    def test_translate_scene_state_adds_support_surface_anchor_for_drawer_task(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_drawer.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube": {"position": [0.4, -0.15, 0.055], "quaternion": [1, 0, 0, 0]},
            "target_marker": {"position": [0.5, 0.15, 0.245], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)
        self.assertIn("drawer_top_surface", translated)
        self.assertEqual(translated["drawer_top_surface"]["support_alignment_target"], "target_marker")

    def test_translate_scene_state_adds_insertion_and_slot_targets(self):
        peg_path = REPO_ROOT / "tasks" / "franka" / "peg_insert" / "franka_peg_insert.yaml"
        with open(peg_path) as f:
            peg_task = yaml.safe_load(f)
        peg_translated = translate_scene_state(
            peg_task,
            {"peg": {"position": [0.0, 0.4, 0.1], "quaternion": [1, 0, 0, 0]}},
        )
        self.assertIn("entry_position", peg_translated["hole"])
        self.assertEqual(peg_translated["hole"]["insertion_axis_world"], [0.0, 0.0, -1.0])

        slot_path = REPO_ROOT / "tasks" / "franka" / "assembly" / "franka_assembling_kits.yaml"
        with open(slot_path) as f:
            slot_task = yaml.safe_load(f)
        slot_translated = translate_scene_state(
            slot_task,
            {"shape_12": {"position": [0.45, -0.2, 0.0], "quaternion": [1, 0, 0, 0]}},
        )
        self.assertIn("shape_to_place", slot_translated)
        self.assertIn("matching_cutout", slot_translated)
        self.assertIn("slot_position", slot_translated["matching_cutout"])

    def test_ordered_object_inference_avoids_partial_color_matches(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack_tray.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.45, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.55, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_3": {"position": [0.60, -0.25, 0.0203], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)

        ordered = _infer_ordered_objects(task_doc["task"]["description"], translated)
        self.assertEqual(ordered[:3], ["cube_1", "cube_2", "cube_3"])

    def test_prompt_includes_structured_task_brief_and_goal_roles(self):
        generator = SimCaPGenerator(_DummyLLM(), load_robot_config("franka"))
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack_tray.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "cube_1": {"position": [0.45, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_2": {"position": [0.55, -0.20, 0.0203], "quaternion": [1, 0, 0, 0]},
            "cube_3": {"position": [0.60, -0.25, 0.0203], "quaternion": [1, 0, 0, 0]},
            "tray_base": {"position": [0.50, 0.00, 0.0], "quaternion": [1, 0, 0, 0]},
        }
        translated = translate_scene_state(task_doc, scene_state)

        prompt = generator._build_user_prompt(task_doc, translated)
        self.assertIn("### Structured Task Brief", prompt)
        self.assertIn("### Goal Condition Summary", prompt)
        self.assertIn("### Task-Role Summary", prompt)
        self.assertIn("Treat the YAML goal mappings and task-role metadata below as authoritative.", prompt)
        self.assertIn("cube_1: task_role=movable_object", prompt)
        self.assertIn("color_name=blue", prompt)
        self.assertIn("Move the first listed object into `tray_anchor`", prompt)
        self.assertIn("execute_pick_and_place_in_container", prompt)
        self.assertIn("execute_pick_and_stack_on_object", prompt)

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

    def test_on_surface_pick_place_task_is_supported(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)

    def test_lift_task_is_supported(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "lift" / "franka_lift.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "lift_hold")

    def test_pick_place_drawer_is_supported_as_transfer_task(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_drawer.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "support_surface_transfer")

    def test_cabinet_task_is_supported_as_articulated_pull(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "articulated_pull")

    def test_insertion_and_upright_families_are_supported(self):
        checks = [
            ("tasks/franka/peg_insert/franka_peg_insert.yaml", "axial_insertion"),
            ("tasks/franka/assembly/franka_peg_insertion_side.yaml", "axial_insertion"),
            ("tasks/franka/assembly/franka_plug_charger.yaml", "axial_insertion"),
            ("tasks/franka/assembly/franka_assembling_kits.yaml", "slot_fit"),
            ("tasks/franka/assembly/franka_lift_peg_upright.yaml", "upright_placement"),
        ]
        for rel_path, expected_family in checks:
            with self.subTest(task=rel_path):
                task_path = REPO_ROOT / rel_path
                with open(task_path) as f:
                    task_doc = yaml.safe_load(f)
                supported, reason = is_supported_tabletop_task(task_doc)
                self.assertTrue(supported, reason)
                capability = assess_task_capability(task_doc)
                self.assertEqual(capability.family, expected_family)

    def test_cabinet_blocks_is_supported_as_simple_transfer(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet_blocks.yaml"
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

    def test_translate_scene_state_clamps_obviously_invalid_negative_z_to_yaml_pose(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "sort" / "franka_shape_sort.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "small_cube_1": {"position": [0.40, 0.02, -1.23], "quaternion": [1, 0, 0, 0]},
        }

        translated = translate_scene_state(task_doc, scene_state)
        self.assertGreater(translated["small_cube_1"]["position"][2], 0.0)

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
            texture_audit = json.loads((Path(tmpdir) / "texture_asset_audit.json").read_text(encoding="utf-8"))

        self.assertIn("SimCaPGenerator", runner_text)
        self.assertIn("classify_texture_asset_path", runner_text)
        self.assertIn("should_preserve_textured_usd", runner_text)
        self.assertIn("spawn_textured_usd_with_child_physics", runner_text)
        self.assertIn("preserved textured USD visual with child physics injection", runner_text)
        self.assertIn("physics_target_prim_basename", runner_text)
        self.assertNotIn("replaced rootless USD with primitive proxy", runner_text)
        self.assertIn("CaPRuntimeContext", runner_text)
        self.assertIn("geometry_success", runner_text)
        self.assertIn("overall_success", runner_text)
        self.assertIn("front_video_generated", runner_text)
        self.assertIn("front_video_path", runner_text)
        self.assertIn("front_video_episode", runner_text)
        self.assertIn("startup_diagnostics.json", runner_text)
        self.assertIn("phase_trace.json", runner_text)
        self.assertIn("_record_phase(", runner_text)
        self.assertIn('_early_record_phase("env_cfg_import", "ok")', runner_text)
        self.assertIn('"env_cfg_import_failed"', runner_text)
        self.assertIn('getattr(super(), "__post_init__", None)', runner_text)
        self.assertIn('annotated bare MISSING fields', runner_text)
        self.assertIn('if "ImplicitActuatorCfg(" in line', runner_text)
        self.assertIn('line.replace("joint_names=", "joint_names_expr=")', runner_text)
        self.assertIn("dropped unsupported InitialStateCfg scale arg", runner_text)
        self.assertIn("rewrote ArticulationCfg joints -> init_state.joint_pos", runner_text)
        self.assertIn("for orientation_key in (", runner_text)
        self.assertIn('"orientation_range"', runner_text)
        self.assertIn('"orientation"', runner_text)
        self.assertIn('"rotation_range"', runner_text)
        self.assertIn('"rotation"', runner_text)
        self.assertIn('disabled xform root observation', runner_text)
        self.assertIn("scene.ee_frame injected for franka", runner_text)
        self.assertIn("dropped unsupported {removed_orientation_keys}", runner_text)
        self.assertIn("dropped unsupported ['min_separation']", runner_text)
        self.assertIn("scene.replicate_physics=False", runner_text)
        self.assertIn("removed helper config", runner_text)
        self.assertIn("promoted init_state to ArticulationCfg.InitialStateCfg", runner_text)
        self.assertIn("disabled missing remote camera asset", runner_text)
        self.assertIn('for name in ("top_cam", "top")', runner_text)
        self.assertNotIn('args.rendering_mode = "performance"', runner_text)
        self.assertNotIn("SkillPlanner", runner_text)
        self.assertNotIn("execute_skill_sequence", runner_text)
        self.assertNotIn("_execute_task_skills_fallback", runner_text)
        self.assertNotIn("using fallback", runner_text)
        self.assertEqual(texture_audit, [])


class _FakeDetector:
    def __init__(self, positions: dict[str, list[float]]):
        self._positions = positions

    def get_object_position(self, name: str) -> np.ndarray:
        return np.asarray(self._positions[name], dtype=np.float64)

    def get_object_pose(self, name: str):
        return self.get_object_position(name), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


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

    def move_to_pose(self, position, rotation, duration=None, allow_position_only_fallback=True):
        self.calls.append(
            (
                "move_to_pose",
                np.asarray(position).tolist(),
                np.asarray(rotation).tolist(),
                duration,
                allow_position_only_fallback,
            )
        )
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
        self.assertAlmostEqual(fake_skills.calls[2][5][0], 0.02, places=6)
        self.assertAlmostEqual(fake_skills.calls[2][5][1], -0.02, places=6)
        self.assertAlmostEqual(fake_skills.calls[2][1][2], 0.0609, places=4)

    def test_runtime_wrapper_uses_tighter_offsets_for_precise_table_targets(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube_1": [0.50, 0.00, 0.0203]})
        translated_positions = {
            "cube_1": {"position": [0.50, 0.00, 0.0406], "estimated_half_height": 0.0203},
            "pos_marker_1": {"position": [0.55, -0.06, 0.004], "estimated_half_height": 0.002},
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
            wrapper._held_xy_offset_world = np.array([0.005, -0.003], dtype=np.float64)
            self.assertTrue(
                wrapper.execute_place_object(
                    [0.55, -0.06, 0.004],
                    target_name="pos_marker_1",
                    is_table=True,
                    skill_description="place cube_1 on pos_marker_1",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        self.assertEqual(fake_skills.calls[1][0], "execute_place")
        self.assertAlmostEqual(fake_skills.calls[1][2], 0.035, places=4)
        self.assertAlmostEqual(fake_skills.calls[1][3], 0.002, places=4)

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

    def test_franka_affordance_methods_delegate_to_pick_and_place_paths(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector(
            {
                "cube_1": [0.50, 0.00, 0.0203],
                "cube_2": [0.42, -0.02, 0.0203],
                "target_marker": [0.55, 0.03, 0.004],
                "command_pose": [0.50, 0.00, 0.375],
            }
        )
        translated_positions = {
            "cube_1": {"position": [0.50, 0.00, 0.0406], "estimated_half_height": 0.0203},
            "cube_2": {"position": [0.42, -0.02, 0.0406], "estimated_half_height": 0.0203},
            "target_marker": {
                "position": [0.55, 0.03, 0.004],
                "estimated_half_height": 0.0,
                "task_role": "placement_target",
            },
            "command_pose": {
                "position": [0.50, 0.00, 0.375],
                "estimated_half_height": 0.0,
                "task_role": "placement_target",
            },
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
                wrapper.execute_pick_and_place_on_target(
                    object_name="cube_2",
                    target_name="target_marker",
                    skill_description="pick cube_2 and place it on target_marker",
                )
            )
            self.assertTrue(
                wrapper.execute_pick_and_lift_to_pose(
                    object_name="cube_1",
                    target_name="command_pose",
                    skill_description="pick cube_1 and hold it at command_pose",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        call_names = [call[0] for call in fake_skills.calls]
        self.assertIn("execute_pick", call_names)
        self.assertIn("execute_place", call_names)
        self.assertIn("move_to_position", call_names)

    def test_container_affordance_uses_tray_anchor_when_available(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube_1": [0.50, 0.00, 0.0203]})
        translated_positions = {
            "cube_1": {"position": [0.50, 0.00, 0.0406], "estimated_half_height": 0.0203},
            "tray_base": {
                "position": [0.50, 0.00, 0.0],
                "estimated_half_height": 0.0,
                "task_role": "placement_target",
            },
            "tray_anchor": {
                "position": [0.52, 0.01, 0.0],
                "estimated_half_height": 0.0,
                "task_role": "placement_target",
            },
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
            self.assertTrue(
                wrapper.execute_place_in_container(
                    container_name="tray_base",
                    skill_description="place cube_1 into the tray",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        place_call = next(call for call in fake_skills.calls if call[0] == "execute_place")
        self.assertAlmostEqual(place_call[1][0], 0.52, places=4)
        self.assertAlmostEqual(place_call[1][1], 0.01, places=4)

    def test_support_surface_affordance_places_using_support_anchor(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube": [0.40, -0.15, 0.055]})
        translated_positions = {
            "cube": {"position": [0.40, -0.15, 0.110], "estimated_half_height": 0.055},
            "target_marker": {"position": [0.50, 0.15, 0.245], "estimated_half_height": 0.0},
            "drawer_top_surface": {
                "position": [0.48, 0.14, 0.245],
                "estimated_half_height": 0.0,
                "task_role": "placement_target",
                "support_alignment_target": "target_marker",
            },
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
                wrapper.execute_pick_and_place_on_support(
                    object_name="cube",
                    support_name="drawer_top_surface",
                    skill_description="place cube on drawer",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        place_call = next(call for call in fake_skills.calls if call[0] == "execute_place")
        self.assertAlmostEqual(place_call[1][0], 0.50, places=4)
        self.assertAlmostEqual(place_call[1][1], 0.15, places=4)

    def test_handle_pull_affordance_uses_handle_pose_and_pull_axis(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({})
        translated_positions = {
            "drawer_handle_top": {
                "position": [0.495, 0.0, 0.41],
                "estimated_half_height": 0.0,
                "task_role": "handle_target",
                "pull_axis_world": [-1.0, 0.0, 0.0],
                "pull_distance": 0.30,
                "target_rotation_world": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            }
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )
        try:
            wrapper = FrankaSkills(frame="world")
            self.assertTrue(wrapper.execute_pull_handle_open("drawer_handle_top", skill_description="open drawer"))
        finally:
            CaPRuntimeContext.clear()

        call_names = [call[0] for call in fake_skills.calls]
        self.assertIn("move_to_pose", call_names)
        self.assertIn("gripper_close", call_names)
        self.assertIn("gripper_open", call_names)

    def test_insertion_affordance_uses_semantic_alias_and_pose_moves(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"peg_tail": [0.345, -0.15, 0.02], "peg_head": [0.395, -0.15, 0.02]})
        translated_positions = {
            "peg": {
                "position": [0.37, -0.15, 0.04],
                "estimated_half_height": 0.02,
                "grasp_object_name": "peg_tail",
            },
            "box_wall_back": {
                "position": [0.62, 0.15, 0.06],
                "estimated_half_height": 0.0,
                "entry_position": [0.49, 0.15, 0.06],
                "target_position": [0.62, 0.15, 0.06],
                "insertion_axis_world": [1.0, 0.0, 0.0],
            },
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
                wrapper.execute_pick_and_insert_into_target(
                    object_name="peg",
                    target_name="box_wall_back",
                    skill_description="insert peg into box",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        pick_call = next(call for call in fake_skills.calls if call[0] == "execute_pick")
        self.assertEqual(pick_call[1], "peg_tail")
        move_count = sum(1 for call in fake_skills.calls if call[0] in {"move_to_pose", "move_to_position"})
        self.assertGreaterEqual(move_count, 3)

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
