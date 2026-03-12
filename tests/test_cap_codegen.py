import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.data_collection.cap_generator import (
    SimCaPGenerator,
    assess_task_capability,
    build_task_skill_preflight,
    extract_skill_calls_from_code,
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
from src.data_collection.pipeline import (
    apply_task_top_camera_override,
    build_task_texture_audit,
    classify_texture_asset_path,
    should_preserve_textured_usd,
)


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

    def test_extract_skill_calls_from_code_returns_unique_ordered_calls(self):
        code = (
            "def execute_task():\n"
            "    skills.move_to_ready(skill_description='ready')\n"
            "    skills.execute_pick_and_place_on_target('cube', 'marker', skill_description='place cube')\n"
            "    skills.execute_pick_and_place_on_target('cube', 'marker', skill_description='place cube again')\n"
            "    skills.move_to_ready(skill_description='done')\n"
        )

        self.assertEqual(
            extract_skill_calls_from_code(code),
            ("move_to_ready", "execute_pick_and_place_on_target"),
        )

    def test_build_task_skill_preflight_tracks_primary_skill_and_targets(self):
        cases = [
            (
                "tasks/franka/lift/franka_lift.yaml",
                "lift_hold",
                "execute_pick_and_lift_to_pose",
                {"command_pose"},
            ),
            (
                "tasks/franka/cabinet/franka_cabinet.yaml",
                "articulated_pull",
                "execute_pull_handle_open",
                {"drawer_handle_top"},
            ),
            (
                "tasks/franka/assembly/franka_assembling_kits.yaml",
                "slot_fit",
                "execute_pick_and_fit_into_slot",
                {"matching_cutout"},
            ),
        ]
        for rel_path, expected_family, expected_skill, expected_targets in cases:
            with self.subTest(task=rel_path):
                with open(REPO_ROOT / rel_path) as f:
                    task_doc = yaml.safe_load(f)
                preflight = build_task_skill_preflight(task_doc)
                self.assertEqual(preflight.family, expected_family)
                self.assertEqual(preflight.primary_skill, expected_skill)
                self.assertTrue(expected_targets.issubset(set(preflight.required_targets)))

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
        self.assertIn("target_rotation_world", peg_translated["hole"])

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
        self.assertIn("### Preferred Skill Usage", prompt)
        self.assertIn("Treat the YAML goal mappings and task-role metadata below as authoritative.", prompt)
        self.assertIn("cube_1: task_role=movable_object", prompt)
        self.assertIn("color_name=blue", prompt)
        self.assertIn("Move the first listed object into `tray_anchor`", prompt)
        self.assertIn("Primary skill: `execute_pick_and_place_in_container`", prompt)
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

    def test_stack_tasks_use_stack_families_even_without_explicit_conditions(self):
        checks = [
            ("tasks/franka/stack/franka_stack.yaml", "stack"),
            ("tasks/franka/stack/franka_stack_tray.yaml", "container_stack"),
        ]
        for rel_path, expected_family in checks:
            with self.subTest(task=rel_path):
                with open(REPO_ROOT / rel_path) as f:
                    task_doc = yaml.safe_load(f)
                capability = assess_task_capability(task_doc)
                self.assertEqual(capability.family, expected_family)

    def test_pick_place_drawer_is_supported_as_transfer_task(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_drawer.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "support_surface_transfer")

    def test_tray_collection_task_is_supported_as_container_transfer(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_gears.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "container_transfer")

    def test_translate_scene_state_adds_generic_container_anchor(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_gears.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(
            task_doc,
            {
                "gear_small": {"position": [0.40, -0.10, 0.05], "quaternion": [1, 0, 0, 0]},
                "gear_medium": {"position": [0.50, 0.00, 0.05], "quaternion": [1, 0, 0, 0]},
                "m16_nut": {"position": [0.45, -0.02, 0.05], "quaternion": [1, 0, 0, 0]},
            },
        )

        self.assertIn("tray_anchor", translated)
        self.assertEqual(translated["tray_anchor"]["support_asset_name"], "tray")

    def test_translate_scene_state_uses_primitive_scale_for_rigid_marker_height(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_drawer.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(
            task_doc,
            {
                "cube": {"position": [0.40, -0.15, 0.021], "quaternion": [1, 0, 0, 0]},
                "target_marker": {"position": [0.50, 0.15, 0.245], "quaternion": [1, 0, 0, 0]},
            },
        )

        self.assertAlmostEqual(translated["target_marker"]["estimated_half_height"], 0.003, places=4)
        self.assertAlmostEqual(translated["target_marker"]["position"][2], 0.248, places=3)
        self.assertAlmostEqual(translated["drawer_top_surface"]["support_top_z"], 0.248, places=3)

    def test_translate_scene_state_adds_fixed_joint_composite_metadata(self):
        task_doc = {
            "task": {"name": "CompositePeg"},
            "assets": [
                {"name": "peg_red", "type": "rigid", "source": "primitive", "primitive": "cube", "scale": [0.12, 0.05, 0.05]},
                {"name": "peg_blue", "type": "rigid", "source": "primitive", "primitive": "cube", "scale": [0.12, 0.05, 0.05]},
            ],
            "constraints": [
                {
                    "name": "PegFixedJoint",
                    "type": "fixed_joint",
                    "parent": "/World/PegRed",
                    "child": "/World/PegBlue",
                }
            ],
            "goal": {
                "conditions": [
                    {"subject": "peg_red", "relation": "upright", "target": "table"},
                ]
            },
        }

        translated = translate_scene_state(
            task_doc,
            {
                "peg_red": {"position": [0.39, 0.0, 0.05], "quaternion": [1, 0, 0, 0]},
                "peg_blue": {"position": [0.51, 0.0, 0.05], "quaternion": [1, 0, 0, 0]},
            },
        )

        self.assertEqual(translated["peg_red"]["composite_pick_position"], [0.45, 0.0, 0.07500000000000001])
        self.assertEqual(translated["peg_red"]["pick_position"], [0.45, 0.0, 0.07500000000000001])
        self.assertEqual(translated["peg_red"]["composite_object_names"], ["peg_red", "peg_blue"])
        self.assertIn("peg", translated["peg_red"]["aliases"])
        self.assertAlmostEqual(translated["peg_red"]["estimated_half_extents"][0], 0.06, places=4)

    def test_translate_scene_state_adds_gear_rim_pick_hint(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "pick_place" / "franka_pick_place_gears.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(
            task_doc,
            {
                "gear_small": {"position": [0.40, -0.10, 0.05], "quaternion": [1, 0, 0, 0]},
            },
        )

        self.assertAlmostEqual(translated["gear_small"]["estimated_half_height"], 0.008, places=4)
        self.assertEqual(translated["gear_small"]["preferred_grasp_region"], "center")
        self.assertLess(translated["gear_small"]["pick_position"][2], translated["gear_small"]["position"][2])
        self.assertEqual(len(translated["gear_small"]["pick_position_candidates"]), 1)
        self.assertNotIn("grasp_offset_override", translated["gear_small"])
        self.assertAlmostEqual(translated["gear_small"]["gripper_close_duration_override"], 1.0, places=4)
        self.assertAlmostEqual(translated["gear_small"]["pick_breakout_lift_override"], 0.05, places=4)
        self.assertEqual(translated["gear_small"]["pick_yaw_candidates_deg"][0], 0.0)

    def test_translate_scene_state_uses_factory_peg_proxy_dimensions(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "peg_insert" / "franka_peg_insert.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(
            task_doc,
            {
                "peg": {"position": [0.0, 0.4, 0.2], "quaternion": [1, 0, 0, 0]},
                "hole": {"position": [0.6, 0.0, 0.05], "quaternion": [1, 0, 0, 0]},
            },
        )

        self.assertAlmostEqual(translated["peg"]["estimated_half_height"], 0.004, places=4)
        self.assertEqual(translated["peg"]["estimated_half_extents"], [0.025, 0.004, 0.004])
        self.assertAlmostEqual(translated["hole"]["estimated_half_height"], 0.025, places=4)

    def test_translate_scene_state_adds_assembling_shape_grasp_overrides(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "assembly" / "franka_assembling_kits.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(
            task_doc,
            {
                "shape_12": {"position": [0.45, -0.20, 0.0], "quaternion": [1, 0, 0, 0]},
            },
        )

        self.assertAlmostEqual(translated["shape_12"]["estimated_half_height"], 0.01, places=4)
        self.assertAlmostEqual(translated["shape_12"]["approach_offset_override"], 0.060, places=4)
        self.assertEqual(translated["shape_to_place"]["preferred_grasp_region"], "outer_rim")
        self.assertEqual(translated["shape_to_place"]["source_object_name"], "shape_12")
        self.assertEqual(translated["shape_12"]["pick_required_tool_axis_world"], [0.0, 0.0, -1.0])
        self.assertEqual(len(translated["shape_12"]["grasp_points_local"]), 5)
        self.assertGreaterEqual(len(translated["shape_12"]["pick_position_candidates"]), 5)
        self.assertNotAlmostEqual(translated["shape_12"]["pick_position_candidates"][0][0], translated["shape_12"]["position"][0], places=3)
        self.assertLess(translated["shape_12"]["pick_position_candidates"][0][2], translated["shape_12"]["position"][2])
        self.assertAlmostEqual(translated["shape_to_place"]["gripper_close_duration_override"], 1.0, places=4)
        self.assertAlmostEqual(translated["shape_to_place"]["pick_breakout_lift_override"], 0.05, places=4)
        self.assertAlmostEqual(translated["shape_to_place"]["place_approach_offset_override"], 0.03, places=4)
        self.assertAlmostEqual(translated["shape_to_place"]["place_drop_offset_override"], 0.0, places=4)
        self.assertGreaterEqual(len(translated["shape_12"]["pick_yaw_candidates_deg"]), 4)
        self.assertTrue(translated["shape_to_place"]["slot_fit_debug"])
        self.assertAlmostEqual(translated["matching_cutout"]["slot_yaw_tolerance_deg"], 25.0, places=4)
        self.assertAlmostEqual(translated["matching_cutout"]["slot_z_tolerance"], 0.012, places=4)

    def test_cabinet_task_is_supported_as_articulated_pull(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "articulated_pull")

    def test_articulated_pull_example_uses_handle_skill(self):
        robot_cfg = load_robot_config("franka")
        generator = SimCaPGenerator(_DummyLLM(), robot_cfg)
        example = generator._build_code_example("articulated_pull")
        self.assertIn("execute_pull_handle_open", example)
        self.assertNotIn("execute_set_handle_open_fraction", example)

    def test_articulated_container_transfer_example_uses_open_place_close_sequence(self):
        robot_cfg = load_robot_config("franka")
        generator = SimCaPGenerator(_DummyLLM(), robot_cfg)
        example = generator._build_code_example("articulated_container_transfer")
        self.assertIn("execute_pull_handle_open", example)
        self.assertIn("execute_pick_and_place_in_container", example)
        self.assertIn("execute_push_handle_closed", example)
        self.assertIn('container_name="drawer_container"', example)

    def test_cabinet_translation_exposes_handle_joint_metadata(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(task_doc, {})
        self.assertIn("drawer_handle_top", translated)
        self.assertEqual(translated["drawer_handle_top"]["articulation_name"], "cabinet")
        self.assertEqual(translated["drawer_handle_top"]["joint_name"], "drawer_top_joint")
        self.assertAlmostEqual(translated["drawer_handle_top"]["target_joint_position"], 0.3, places=3)
        self.assertIn("closed_position", translated["drawer_handle_top"])
        self.assertIn("slide_axis_world", translated["drawer_handle_top"])
        self.assertIn("open_target_joint_position", translated["drawer_handle_top"])
        self.assertIn("close_target_joint_position", translated["drawer_handle_top"])

    def test_cabinet_store_cube_task_is_supported_as_articulated_container_transfer(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet_store_cube.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        supported, reason = is_supported_tabletop_task(task_doc)
        self.assertTrue(supported, reason)
        capability = assess_task_capability(task_doc)
        self.assertEqual(capability.family, "articulated_container_transfer")

    def test_translate_scene_state_adds_drawer_container_target_for_store_task(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet_store_cube.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(task_doc, {})
        self.assertIn("drawer_handle_top", translated)
        self.assertIn("drawer_container", translated)
        self.assertAlmostEqual(translated["drawer_handle_top"]["position"][0], 0.45, places=4)
        self.assertAlmostEqual(translated["drawer_handle_top"]["position"][1], 0.15, places=4)
        self.assertAlmostEqual(translated["drawer_handle_top"]["position"][2], 0.22, places=4)
        self.assertEqual(translated["drawer_handle_top"]["closed_position"], translated["drawer_handle_top"]["position"])
        self.assertEqual(translated["drawer_container"]["task_role"], "placement_target")
        self.assertEqual(translated["drawer_container"]["joint_name"], "drawer_top_joint")
        self.assertEqual(translated["drawer_container"]["articulation_name"], "drawer")
        self.assertEqual(translated["drawer_container"]["container_half_extents"], [0.05, 0.045, 0.045])
        self.assertAlmostEqual(translated["drawer_container"]["open_target_joint_position"], 0.12, places=3)
        self.assertIn("entry_position", translated["drawer_container"])
        self.assertIn("target_position", translated["drawer_container"])
        self.assertIn("insertion_axis_world", translated["drawer_container"])
        self.assertIn("target_rotation_world", translated["drawer_container"])
        self.assertLess(translated["drawer_container"]["entry_position"][0], translated["drawer_container"]["target_position"][0])

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

    def test_slot_fit_translation_includes_dynamic_shape_alias(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "assembly" / "franka_assembling_kits.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        translated = translate_scene_state(task_doc, {})
        self.assertIn("shape_12", translated)
        self.assertIn("shape_to_place", translated)
        self.assertIn("matching_cutout", translated)

    def test_side_insertion_translation_includes_composite_peg_alias(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "assembly" / "franka_peg_insertion_side.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        scene_state = {
            "peg_head": {"position": [0.395, -0.15, 0.04], "quaternion": [1.0, 0.0, 0.0, 0.0]},
            "peg_tail": {"position": [0.345, -0.15, 0.04], "quaternion": [1.0, 0.0, 0.0, 0.0]},
            "box_wall_back": {"position": [0.62, 0.15, 0.12], "quaternion": [1.0, 0.0, 0.0, 0.0]},
        }
        translated = translate_scene_state(task_doc, scene_state)
        self.assertIn("peg", translated)
        self.assertEqual(translated["peg"]["grasp_object_name"], "peg_tail")
        self.assertEqual(translated["peg"]["insertion_subject_name"], "peg_head")
        self.assertEqual(translated["peg"]["pick_position"], translated["peg_tail"]["position"])
        self.assertAlmostEqual(translated["peg"]["insertion_subject_offset_local"][0], 0.05, places=6)
        self.assertAlmostEqual(translated["peg"]["insertion_subject_offset_local"][1], 0.0, places=6)
        self.assertAlmostEqual(translated["peg"]["insertion_subject_offset_local"][2], 0.0, places=6)
        self.assertAlmostEqual(translated["box_wall_back"]["target_position"][2], 0.06, places=6)

    def test_axial_insertion_validation_rejects_tip_subpart_when_alias_exists(self):
        generator = SimCaPGenerator(_DummyLLM(), load_robot_config("franka"))
        translated_positions = {
            "peg": {
                "position": [0.37, -0.15, 0.04],
                "grasp_object_name": "peg_tail",
                "insertion_subject_name": "peg_head",
            },
            "peg_head": {"position": [0.395, -0.15, 0.02]},
            "box_wall_back": {"position": [0.62, 0.15, 0.06]},
        }
        bad_code = """
from skills.skills_franka import FrankaSkills

def execute_task():
    skills = FrankaSkills(frame="world")
    skills.connect()
    try:
        skills.move_to_ready(skill_description="ready")
        skills.execute_pick_and_insert_into_target(
            object_name="peg_head",
            target_name="box_wall_back",
            skill_description="insert peg",
        )
        skills.move_to_ready(skill_description="done")
    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
"""
        with self.assertRaisesRegex(ValueError, "semantic alias 'peg'"):
            generator._validate_task_specific_code(
                bad_code,
                translated_positions=translated_positions,
                family="axial_insertion",
            )

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
    def test_apply_task_top_camera_override_updates_franka_top_camera(self):
        robot_cfg = load_robot_config("franka")
        task_path = REPO_ROOT / "tasks" / "franka" / "stack" / "franka_stack.yaml"
        with open(task_path) as f:
            task_doc = yaml.safe_load(f)

        updated = apply_task_top_camera_override(robot_cfg, task_doc)

        self.assertEqual(updated.cameras["top"].position, [0.25, 0.0, 1.05])
        self.assertEqual(updated.cameras["top"].target, [0.25, 0.0, 0.05])
        self.assertEqual(updated.cameras["top"].up_vector, [1.0, 0.0, 0.0])

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
        self.assertIn("goal_description=goal.get(\"description\", \"\")", runner_text)
        self.assertIn("goal_conditions=vlm_goal_conditions", runner_text)
        self.assertIn("relevant_objects=vlm_relevant_objects", runner_text)
        self.assertIn("_select_vlm_relevant_objects(", runner_text)
        self.assertIn("judge_prompt.txt", runner_text)
        self.assertIn("judge_prompt_context.json", runner_text)
        self.assertIn("front_video_generated", runner_text)
        self.assertIn("front_video_path", runner_text)
        self.assertIn("front_video_camera_name", runner_text)
        self.assertIn("front_video_episode", runner_text)
        self.assertIn('"front_video_fps"', runner_text)
        self.assertIn("_disable_debug_visualization", runner_text)
        self.assertIn("Disabled debug visualizers:", runner_text)
        self.assertIn("apply_task_top_camera_override", runner_text)
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
        self.assertIn("rewrote SceneEntityCfg body_name -> body_names", runner_text)
        self.assertIn("rewrote CuboidCfg mass -> mass_props", runner_text)
        self.assertIn("for orientation_key in (", runner_text)
        self.assertIn('"orientation_range"', runner_text)
        self.assertIn('"orientation"', runner_text)
        self.assertIn('"rotation_range"', runner_text)
        self.assertIn('"rotation"', runner_text)
        self.assertIn('disabled xform root observation', runner_text)

    def test_generated_runner_uses_top_success_video_for_franka_cabinet(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "cabinet" / "franka_cabinet.yaml"
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
            config_json = json.loads((Path(tmpdir) / "pipeline_config.json").read_text(encoding="utf-8"))

        self.assertEqual(config_json["front_video_camera_name"], "top")
        self.assertIn('front_video_camera_name=cfg.get("front_video_camera_name", "front")', runner_text)

    def test_generated_runner_stabilizes_factory_peg_assets_with_proxy(self):
        task_path = REPO_ROOT / "tasks" / "franka" / "peg_insert" / "franka_peg_insert.yaml"
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

        self.assertIn("factory/factory_peg_8mm.usd", runner_text)
        self.assertIn("replaced unstable factory USD with stable {proxy_spec.get('shape')} proxy", runner_text)
        self.assertIn('"disable_gravity": True', runner_text)


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
        grasp_offset=None,
        object_position_override=None,
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
                None
                if object_position_override is None
                else np.asarray(object_position_override, dtype=np.float64).tolist(),
                grasp_offset,
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


class _FakeTargetGrounder:
    def __init__(self, grounded_position):
        self._grounded_position = np.asarray(grounded_position, dtype=np.float64)
        self.calls = []

    def ground_target(self, target_name, target_entry, *, stage, camera_names):
        self.calls.append(
            {
                "target_name": target_name,
                "stage": stage,
                "camera_names": tuple(camera_names),
                "prior_position": list(target_entry.get("position", [])),
            }
        )
        return {
            "target_name": target_name,
            "position_world": self._grounded_position.tolist(),
            "position_robot": self._grounded_position.tolist(),
            "orientation_robot_wxyz": [1.0, 0.0, 0.0, 0.0],
        }


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

    def test_upright_affordance_uses_object_vertical_extent_for_release_height(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"peg_red": [0.39, 0.0, 0.025]})
        translated_positions = {
            "peg_red": {
                "position": [0.39, 0.0, 0.05],
                "estimated_half_height": 0.025,
                "estimated_half_extents": [0.12, 0.025, 0.025],
                "upright_axis_local": [1.0, 0.0, 0.0],
                "upright_target_rotation_world": [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                "upright_grasp_offset_local": [0.036, 0.0, 0.0],
                "upright_grasp_offset_candidates_local": [[0.036, 0.0, 0.0], [-0.036, 0.0, 0.0]],
            },
            "table_surface": {
                "position": [0.5, 0.0, 0.0],
                "estimated_half_height": 0.0,
                "support_top_z": 0.0,
                "task_role": "support_surface",
                "aliases": ["table", "table surface", "table top"],
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
                wrapper.execute_pick_and_place_upright(
                    object_name="peg_red",
                    support_name="table",
                    skill_description="stand peg upright",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        pick_call = next(call for call in fake_skills.calls if call[0] == "execute_pick")
        self.assertAlmostEqual(pick_call[10][0], 0.426, places=4)
        move_calls = [call for call in fake_skills.calls if call[0] == "move_to_pose"]
        self.assertGreaterEqual(len(move_calls), 2)
        release_call = move_calls[1]
        self.assertAlmostEqual(release_call[1][2], 0.156, places=4)

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

    def test_handle_pull_uses_visual_grounding_when_available(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({})
        target_grounder = _FakeTargetGrounder([0.44, 0.02, 0.39])
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
            target_grounder=target_grounder,
        )
        try:
            wrapper = FrankaSkills(frame="world")
            self.assertTrue(wrapper.execute_pull_handle_open("drawer_handle_top", skill_description="open drawer"))
        finally:
            CaPRuntimeContext.clear()

        self.assertEqual(len(target_grounder.calls), 1)
        pose_calls = [call for call in fake_skills.calls if call[0] == "move_to_pose"]
        self.assertGreaterEqual(len(pose_calls), 2)
        self.assertAlmostEqual(pose_calls[0][1][0], 0.38, places=4)
        self.assertAlmostEqual(pose_calls[0][1][1], 0.02, places=4)
        self.assertAlmostEqual(pose_calls[0][1][2], 0.39, places=4)

    def test_dynamic_handle_pull_prefers_joint_geometry_over_visual_grounding(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({})
        target_grounder = _FakeTargetGrounder([0.44, 0.02, 0.39])
        translated_positions = {
            "drawer_handle_top": {
                "position": [0.495, 0.0, 0.41],
                "estimated_half_height": 0.0,
                "task_role": "handle_target",
                "articulation_name": "drawer",
                "joint_name": "drawer_top_joint",
                "closed_position": [0.495, 0.0, 0.41],
                "slide_axis_world": [-1.0, 0.0, 0.0],
                "pull_axis_world": [-1.0, 0.0, 0.0],
                "open_target_joint_position": 0.12,
                "close_target_joint_position": 0.0,
                "target_rotation_world": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            }
        }

        CaPRuntimeContext.configure(
            sim_skills=fake_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
            target_grounder=target_grounder,
        )
        try:
            wrapper = FrankaSkills(frame="world")
            values = [0.12, 0.12, 0.12, 0.12]
            wrapper._read_articulation_joint_position = lambda articulation_name, joint_name: values.pop(0) if values else 0.12
            wrapper._drive_articulation_joint = lambda **kwargs: True
            self.assertTrue(wrapper.execute_pull_handle_open("drawer_handle_top", skill_description="open drawer"))
        finally:
            CaPRuntimeContext.clear()

        self.assertEqual(len(target_grounder.calls), 0)
        position_calls = [call for call in fake_skills.calls if call[0] == "move_to_position"]
        self.assertGreaterEqual(len(position_calls), 2)
        self.assertAlmostEqual(position_calls[0][1][0], 0.315, places=4)
        self.assertAlmostEqual(position_calls[0][1][1], 0.0, places=4)
        self.assertAlmostEqual(position_calls[0][1][2], 0.41, places=4)

    def test_set_handle_open_fraction_closes_via_push_skill(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({})
        translated_positions = {
            "drawer_handle_top": {
                "position": [0.495, 0.0, 0.41],
                "estimated_half_height": 0.0,
                "task_role": "handle_target",
                "articulation_name": "drawer",
                "joint_name": "drawer_top_joint",
                "target_joint_position": 0.12,
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
            pushed = {}

            def _fake_push_handle_closed(**kwargs):
                pushed.update(kwargs)
                return True

            wrapper.execute_push_handle_closed = _fake_push_handle_closed
            self.assertTrue(
                wrapper.execute_set_handle_open_fraction(
                    "drawer_handle_top",
                    0.0,
                    skill_description="close the drawer",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        self.assertEqual(pushed["handle_name"], "drawer_handle_top")
        self.assertEqual(pushed["skill_description"], "close the drawer")

    def test_refresh_related_articulated_targets_updates_drawer_handle_and_container(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({})
        translated_positions = {
            "drawer_handle_top": {
                "position": [0.495, 0.0, 0.41],
                "estimated_half_height": 0.0,
                "task_role": "handle_target",
                "articulation_name": "drawer",
                "joint_name": "drawer_top_joint",
                "closed_position": [0.495, 0.0, 0.41],
                "slide_axis_world": [-1.0, 0.0, 0.0],
                "pull_axis_world": [-1.0, 0.0, 0.0],
                "target_joint_position": 0.12,
                "open_target_joint_position": 0.12,
                "close_target_joint_position": 0.0,
            },
            "drawer_container": {
                "position": [0.5, 0.15, 0.175],
                "task_role": "placement_target",
                "articulation_name": "drawer",
                "joint_name": "drawer_top_joint",
                "close_target_joint_position": 0.0,
                "closed_floor_position": [0.5, 0.15, 0.148],
                "closed_center_position": [0.5, 0.15, 0.175],
                "slide_axis_world": [-1.0, 0.0, 0.0],
                "container_half_extents": [0.05, 0.045, 0.045],
                "entry_clearance": 0.08,
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
            wrapper._read_articulation_joint_position = lambda articulation_name, joint_name: 0.12
            wrapper._refresh_related_articulated_targets("drawer")
        finally:
            CaPRuntimeContext.clear()

        handle_entry = translated_positions["drawer_handle_top"]
        self.assertAlmostEqual(handle_entry["position"][0], 0.375, places=4)
        self.assertAlmostEqual(handle_entry["current_joint_position"], 0.12, places=4)
        drawer_entry = translated_positions["drawer_container"]
        self.assertAlmostEqual(drawer_entry["target_position"][0], 0.38, places=4)
        self.assertAlmostEqual(drawer_entry["position"][0], 0.38, places=4)
        self.assertAlmostEqual(drawer_entry["entry_position"][0], 0.30, places=4)
        self.assertAlmostEqual(drawer_entry["support_top_z"], 0.148, places=4)

    def test_execute_push_handle_closed_uses_handle_pose_and_push_axis(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({})
        translated_positions = {
            "drawer_handle_top": {
                "position": [0.375, 0.0, 0.41],
                "estimated_half_height": 0.0,
                "task_role": "handle_target",
                "articulation_name": "drawer",
                "joint_name": "drawer_top_joint",
                "closed_position": [0.495, 0.0, 0.41],
                "slide_axis_world": [-1.0, 0.0, 0.0],
                "pull_axis_world": [-1.0, 0.0, 0.0],
                "open_target_joint_position": 0.12,
                "close_target_joint_position": 0.0,
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
            values = [0.12, 0.12, 0.12, 0.0]
            wrapper._read_articulation_joint_position = lambda articulation_name, joint_name: values.pop(0) if values else 0.0
            wrapper._drive_articulation_joint = lambda **kwargs: True
            self.assertTrue(wrapper.execute_push_handle_closed("drawer_handle_top", skill_description="close drawer"))
        finally:
            CaPRuntimeContext.clear()

        position_calls = [call for call in fake_skills.calls if call[0] == "move_to_position"]
        self.assertGreaterEqual(len(position_calls), 3)
        self.assertAlmostEqual(position_calls[0][1][0], 0.315, places=4)
        self.assertAlmostEqual(position_calls[1][1][0], 0.405, places=4)
        self.assertAlmostEqual(position_calls[2][1][0], 0.375, places=4)

    def test_execute_place_in_container_uses_top_down_place_for_dynamic_drawer_targets(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"cube": [0.40, 0.00, 0.02]})
        translated_positions = {
            "cube": {
                "position": [0.40, 0.00, 0.04],
                "estimated_half_height": 0.02,
            },
            "drawer_container": {
                "position": [0.38, 0.15, 0.175],
                "task_role": "placement_target",
                "articulation_name": "drawer",
                "joint_name": "drawer_top_joint",
                "close_target_joint_position": 0.0,
                "closed_floor_position": [0.5, 0.15, 0.148],
                "closed_center_position": [0.5, 0.15, 0.175],
                "slide_axis_world": [-1.0, 0.0, 0.0],
                "container_half_extents": [0.05, 0.045, 0.045],
                "entry_position": [0.30, 0.15, 0.175],
                "target_position": [0.38, 0.15, 0.175],
                "insertion_axis_world": [1.0, 0.0, 0.0],
                "target_rotation_world": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            },
            "drawer_container_anchor": {
                "position": [0.5, 0.15, 0.122],
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
            wrapper._held_object_name = "cube"
            wrapper._held_half_height = 0.02
            wrapper._held_xy_offset_world = np.zeros(2, dtype=np.float64)
            wrapper._read_articulation_joint_position = lambda articulation_name, joint_name: 0.12
            self.assertTrue(
                wrapper.execute_place_in_container(
                    "drawer_container",
                    skill_description="store cube in drawer",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        place_calls = [call for call in fake_skills.calls if call[0] == "execute_place"]
        self.assertEqual(len(place_calls), 1)
        self.assertAlmostEqual(place_calls[0][1][0], 0.38, places=4)
        self.assertAlmostEqual(place_calls[0][1][1], 0.15, places=4)
        self.assertAlmostEqual(place_calls[0][1][2], 0.168, places=4)
        self.assertEqual(place_calls[0][4], "cube")
        self.assertIsNotNone(place_calls[0][7])

    def test_insertion_affordance_uses_semantic_alias_and_pose_moves(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"peg_tail": [0.345, -0.15, 0.02], "peg_head": [0.395, -0.15, 0.02]})
        translated_positions = {
            "peg": {
                "position": [0.37, -0.15, 0.04],
                "estimated_half_height": 0.02,
                "grasp_object_name": "peg_tail",
                "insertion_subject_name": "peg_head",
                "insertion_subject_offset_local": [0.05, 0.0, 0.0],
                "pick_position": [0.345, -0.15, 0.02],
            },
            "box_wall_back": {
                "position": [0.62, 0.15, 0.06],
                "estimated_half_height": 0.0,
                "entry_position": [0.49, 0.15, 0.06],
                "target_position": [0.62, 0.15, 0.06],
                "insertion_axis_world": [1.0, 0.0, 0.0],
                "target_rotation_world": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
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
        pose_calls = [call for call in fake_skills.calls if call[0] == "move_to_pose"]
        self.assertGreaterEqual(len(pose_calls), 3)
        pose_x = [call[1][0] for call in pose_calls[:3]]
        self.assertLess(pose_x[0], pose_x[1])
        self.assertLess(pose_x[1], pose_x[2])
        self.assertLess(pose_x[1], 0.49)
        self.assertLess(pose_x[2], 0.62)

    def test_slot_fit_affordance_uses_precise_place_execution(self):
        robot_cfg = load_robot_config("franka")
        fake_skills = _FakeSimSkills()
        detector = _FakeDetector({"shape_12": [0.45, -0.2, 0.01]})
        translated_positions = {
            "shape_to_place": {
                "position": [0.45, -0.2, 0.01],
                "grasp_object_name": "shape_12",
                "estimated_half_height": 0.01,
                "place_approach_offset_override": 0.03,
                "place_drop_offset_override": 0.0,
            },
            "matching_cutout": {
                "position": [0.28, 0.05, 0.0],
                "slot_position": [0.28, 0.05, 0.0],
                "slot_yaw_deg": 15.0,
                "estimated_half_height": 0.0,
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
                wrapper.execute_pick_and_fit_into_slot(
                    object_name="shape_to_place",
                    target_name="matching_cutout",
                    skill_description="fit shape into slot",
                )
            )
        finally:
            CaPRuntimeContext.clear()

        place_call = next(call for call in fake_skills.calls if call[0] == "execute_place")
        self.assertAlmostEqual(place_call[1][0], 0.28, places=4)
        self.assertAlmostEqual(place_call[1][1], 0.05, places=4)
        self.assertAlmostEqual(place_call[2], 0.03, places=4)
        self.assertAlmostEqual(place_call[3], 0.0, places=4)
        self.assertEqual(place_call[9], [0.0, 0.0, -1.0])

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
