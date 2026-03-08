"""
Data collection pipeline orchestrator.

Coordinates: environment setup → episode loop (detect → plan → execute → record → judge)
→ dataset finalize.

Execution model: Generates a `collect_data.py` script that runs inside the
IsaacLab conda subprocess (env_isaaclab), similar to the existing IsaacLab agent.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.common.robot_names import normalize_robot_name

from .config import (
    PROJECT_ROOT,
    DataCollectionConfig,
    RobotSimConfig,
    get_default_isaaclab_path,
    load_pipeline_config,
    load_robot_config,
)

logger = logging.getLogger(__name__)

ISAACLAB_PATH = str(get_default_isaaclab_path())
CONDA_ENV = "env_isaaclab"


class DataCollectionPipeline:
    """
    Full data collection pipeline.

    Flow:
    1. Load task YAML → detect robot type
    2. Generate/load IsaacLab environment (via IsaacLabAgent or existing output)
    3. Generate collect_data.py (IsaacLab subprocess script)
    4. Execute in conda subprocess
    5. Post-process: convert raw data → LeRobot v3.0

    Args:
        yaml_path: Path to task YAML document
        config: Pipeline config (auto-loaded if None)
        env_dir: Path to pre-generated env code (optional)
    """

    def __init__(
        self,
        yaml_path: str,
        config: Optional[DataCollectionConfig] = None,
        env_dir: Optional[str] = None,
    ):
        self.yaml_path = Path(yaml_path).resolve()
        self.config = config or load_pipeline_config()
        self.env_dir = Path(env_dir) if env_dir else None

        # Load task document
        with open(self.yaml_path) as f:
            self.task_doc = yaml.safe_load(f)

        # Detect robot
        self.robot_name = self._detect_robot()
        self.robot_cfg = load_robot_config(self.robot_name)

        # Output directory
        task_name = self.task_doc.get("task", {}).get("name", "unknown")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            PROJECT_ROOT / self.config.output_dir / f"{task_name}_{timestamp}"
        )

        logger.info(
            f"Pipeline initialized: task={task_name}, robot={self.robot_name}, "
            f"output={self.output_dir}"
        )

    def run(self) -> dict:
        """
        Execute the full data collection pipeline.

        Returns:
            Report dict with episode summaries and dataset path.
        """
        # Step 1: Ensure IsaacLab environment code exists
        if self.env_dir is None:
            self.env_dir = self._generate_environment()

        if self.env_dir is None or not self.env_dir.exists():
            raise RuntimeError(
                f"IsaacLab environment code not found at {self.env_dir}. "
                f"Generate first with: python scripts/run_isaac_lab.py {self.yaml_path}"
            )

        # Step 2: Generate collection runner script
        self.output_dir.mkdir(parents=True, exist_ok=True)
        runner_path = self._generate_collection_runner()

        # Step 3: Execute in IsaacLab subprocess
        success, output = self._execute_subprocess(runner_path)

        # Step 4: Load results
        results_path = self.output_dir / "collection_results.json"
        results = {}
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)

        # Step 5: Post-process (convert to LeRobot if raw data exists)
        raw_data_dir = self.output_dir / "raw_dataset"
        if raw_data_dir.exists():
            self._convert_to_lerobot(raw_data_dir)

        pipeline_completed = bool(
            success or results.get("pipeline_completed", False)
        )
        target_met = bool(results.get("target_met", False))
        successful_episodes = int(results.get("successful_episodes", 0))
        geometry_successful_episodes = int(
            results.get("geometry_successful_episodes", successful_episodes)
        )
        vlm_successful_episodes = int(results.get("vlm_successful_episodes", 0))
        overall_successful_episodes = int(results.get("overall_successful_episodes", 0))
        total_episodes = int(results.get("total_episodes", 0))

        return {
            "success": pipeline_completed,
            "pipeline_completed": pipeline_completed,
            "target_met": target_met,
            "successful_episodes": successful_episodes,
            "geometry_successful_episodes": geometry_successful_episodes,
            "vlm_successful_episodes": vlm_successful_episodes,
            "overall_successful_episodes": overall_successful_episodes,
            "total_episodes": total_episodes,
            "output_dir": str(self.output_dir),
            "raw_dataset": str(raw_data_dir) if raw_data_dir.exists() else None,
            "results": results,
            "robot": self.robot_name,
            "task": self.task_doc.get("task", {}).get("name", "unknown"),
        }

    def _detect_robot(self) -> str:
        """Detect robot name from task YAML."""
        # Check assets for articulation type
        for asset in self.task_doc.get("assets", []):
            if asset.get("type") == "articulation":
                robot_type = normalize_robot_name(asset.get("robot_type", ""))
                if robot_type:
                    return robot_type
                robot_type = normalize_robot_name(asset.get("asset_path", ""))
                if robot_type:
                    return robot_type

        # Fallback: check task name or metadata
        task_name = self.task_doc.get("task", {}).get("name", "").lower()
        robot = normalize_robot_name(task_name)
        if robot:
            return robot

        # Check file path
        yaml_str = str(self.yaml_path).lower()
        robot = normalize_robot_name(yaml_str)
        if robot:
            return robot

        logger.warning("Could not detect robot type, defaulting to 'franka'")
        return "franka"

    def _generate_environment(self) -> Optional[Path]:
        """Generate IsaacLab environment using IsaacLabAgent."""
        try:
            sys.path.insert(0, str(PROJECT_ROOT))
            from src.isaac_lab.agent import IsaacLabAgent

            agent = IsaacLabAgent()
            result = agent.run(str(self.yaml_path), dry_run=True)

            if result.get("success"):
                return Path(result["output_dir"])
            else:
                logger.error(f"Environment generation failed: {result}")
                return None
        except Exception as e:
            logger.error(f"Failed to generate environment: {e}")
            return None

    def _generate_collection_runner(self) -> Path:
        """
        Generate collect_data.py that runs inside IsaacLab subprocess.

        This script:
        1. Initializes AppLauncher (before physics imports)
        2. Imports env_cfg.py
        3. Creates ManagerBasedRLEnv
        4. Runs episode loop with skills + recording
        """
        runner_path = self.output_dir / "collect_data.py"

        # Compute success loop parameters
        target_success = self.config.target_successful_episodes
        if target_success <= 0:
            target_success = self.config.max_episodes  # disabled: run all
        max_attempts = self.config.max_total_attempts
        if max_attempts <= 0:
            max_attempts = target_success * 5

        # Serialize config to JSON for the subprocess
        config_data = {
            "yaml_path": str(self.yaml_path),
            "env_dir": str(self.env_dir),
            "output_dir": str(self.output_dir),
            "robot_name": self.robot_name,
            "max_episodes": self.config.max_episodes,
            "max_steps": self.config.max_steps_per_episode,
            "recording_fps": self.config.effective_fps,
            "headless": self.config.env_headless,
            "num_envs": self.config.env_num_envs,
            "use_vlm_judge": self.config.use_vlm_judge,
            "llm_model": self.config.llm_model,
            "skill_retry_max": self.config.skill_retry_max,
            "target_successful_episodes": target_success,
            "max_total_attempts": max_attempts,
        }
        config_json_path = self.output_dir / "pipeline_config.json"
        with open(config_json_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Generate the runner script
        # NOTE: AppLauncher MUST be initialized before any physics imports
        headless_val = self.config.env_headless
        script = textwrap.dedent(f'''\
            """Auto-generated data collection runner for IsaacLab."""
            import argparse
            import json
            import sys
            import os
            import re
            import traceback

            # === AppLauncher MUST be first (before any physics imports) ===
            from isaaclab.app import AppLauncher

            parser = argparse.ArgumentParser()
            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.headless = {headless_val}
            args.num_envs = {self.config.env_num_envs}
            args.enable_cameras = True  # Always enable cameras (EGL in headless, X11 in GUI)
            args.rendering_mode = "performance"  # Rasterization only (no RTX path tracing) — fast shader compile
            app_launcher = AppLauncher(args)
            simulation_app = app_launcher.app

            # === Now safe to import physics modules ===
            import numpy as np
            import torch

            # Add project paths
            sys.path.insert(0, "{PROJECT_ROOT}")
            sys.path.insert(0, "{Path(self.env_dir).resolve()}")
            sys.path.insert(0, "{(PROJECT_ROOT / 'src' / 'data_collection' / 'cap_runtime').resolve()}")

            from isaaclab.envs import ManagerBasedRLEnv

            def _patch_generated_env_cfg_source():
                """Patch common import-time issues in generated env_cfg.py files."""
                env_cfg_path = os.path.join("{Path(self.env_dir).resolve()}", "env_cfg.py")
                if not os.path.exists(env_cfg_path):
                    return

                with open(env_cfg_path, "r", encoding="utf-8") as f:
                    env_cfg_text = f.read()

                def _rewrite_implicit_actuator_joint_names(content):
                    lines = content.splitlines()
                    patched_lines = []
                    in_implicit_actuator = False
                    paren_depth = 0
                    replaced = False

                    for line in lines:
                        if "ImplicitActuatorCfg(" in line:
                            in_implicit_actuator = True
                            paren_depth = line.count("(") - line.count(")")
                        elif in_implicit_actuator:
                            paren_depth += line.count("(") - line.count(")")

                        if in_implicit_actuator and "joint_names=" in line:
                            line = line.replace("joint_names=", "joint_names_expr=")
                            replaced = True

                        patched_lines.append(line)

                        if in_implicit_actuator and paren_depth <= 0:
                            in_implicit_actuator = False
                            paren_depth = 0

                    return "\\n".join(patched_lines), replaced

                patched_text = re.sub(
                    r'^(?P<indent>\\s*)super\\(\\).__post_init__\\(\\)\\s*$',
                    (
                        r'\\g<indent>_base_post_init = getattr(super(), "__post_init__", None)\\n'
                        r'\\g<indent>if callable(_base_post_init):\\n'
                        r'\\g<indent>    _base_post_init()'
                    ),
                    env_cfg_text,
                    flags=re.MULTILINE,
                )
                patched_text, patched_implicit_joint_names = _rewrite_implicit_actuator_joint_names(patched_text)

                if patched_text != env_cfg_text:
                    with open(env_cfg_path, "w", encoding="utf-8") as f:
                        f.write(patched_text)
                    patch_notes = []
                    if "super().__post_init__()" in env_cfg_text:
                        patch_notes.append("guarded super().__post_init__()")
                    if patched_implicit_joint_names:
                        patch_notes.append("rewrote ImplicitActuatorCfg joint_names -> joint_names_expr")
                    print(f"Patched generated env_cfg.py: {{patch_notes}}")

            def _install_generated_mdp_stubs():
                \"\"\"Install no-op stubs for missing custom MDP helpers in generated envs.

                Data collection disables task success terminations after env creation, so
                for missing task-specific `mdp.*` helpers we only need import-time shims
                that keep `env_cfg.py` loadable.
                \"\"\"
                env_cfg_path = os.path.join("{Path(self.env_dir).resolve()}", "env_cfg.py")
                mdp_dir = os.path.join("{Path(self.env_dir).resolve()}", "mdp")
                if not os.path.exists(env_cfg_path) or not os.path.isdir(mdp_dir):
                    return

                import mdp as generated_mdp

                with open(env_cfg_path, "r", encoding="utf-8") as f:
                    env_cfg_text = f.read()

                def _num_envs(env):
                    return int(getattr(env, "num_envs", getattr(getattr(env, "scene", None), "num_envs", 1)))

                def _device(env):
                    return getattr(env, "device", "cpu")

                def _bool_stub(env, *_, **__):
                    return torch.zeros(_num_envs(env), device=_device(env), dtype=torch.bool)

                def _float_stub(env, *_, **__):
                    return torch.zeros(_num_envs(env), device=_device(env), dtype=torch.float32)

                def _event_stub(*_, **__):
                    return None

                done_terms = set(re.findall(r"DoneTerm\\(\\s*func=mdp\\.([A-Za-z_][A-Za-z0-9_]*)", env_cfg_text))
                reward_terms = set(re.findall(r"RewTerm\\(\\s*func=mdp\\.([A-Za-z_][A-Za-z0-9_]*)", env_cfg_text))
                event_terms = set(re.findall(r"EventTerm\\(\\s*func=mdp\\.([A-Za-z_][A-Za-z0-9_]*)", env_cfg_text))

                missing_done = sorted(name for name in done_terms if not hasattr(generated_mdp, name))
                missing_reward = sorted(name for name in reward_terms if not hasattr(generated_mdp, name))
                missing_event = sorted(name for name in event_terms if not hasattr(generated_mdp, name))

                for name in missing_done:
                    setattr(generated_mdp, name, _bool_stub)
                for name in missing_reward:
                    setattr(generated_mdp, name, _float_stub)
                for name in missing_event:
                    setattr(generated_mdp, name, _event_stub)

                if missing_done or missing_reward or missing_event:
                    print(
                        "Installed generated mdp stubs: "
                        f"done={{missing_done}}, reward={{missing_reward}}, event={{missing_event}}"
                    )

            _patch_generated_env_cfg_source()
            _install_generated_mdp_stubs()

            # Import generated environment config
            from env_cfg import *  # noqa: F403

            # Import data collection modules
            from src.data_collection.config import load_robot_config
            from src.data_collection.sim_robot_interface import SimRobotInterface
            from src.data_collection.sim_detector import SimDetector
            from src.data_collection.sim_camera import SimCamera, MultiCameraManager, inject_cameras_into_scene, SceneCameraManager
            from src.data_collection.sim_recorder import SimRecorder, RecorderStorageError
            from src.data_collection.sim_skills import SimSkills, create_ik_solver
            from src.data_collection.cap_artifacts import make_episode_run_dir, save_code_generation_artifacts, save_execution_context, save_judge_images
            from src.data_collection.cap_generator import SimCaPGenerator, is_supported_tabletop_task, select_cap_profile
            from src.data_collection.cap_runtime import CaPRuntimeContext
            from src.data_collection.cap_runtime.skills.base import PolicyFallbackBlockedError
            from src.data_collection.sim_judge import SimJudge

            MARKER_FILE = os.path.join("{Path(self.output_dir).resolve()}", "COLLECTION_COMPLETE_MARKER")
            USE_VLM_JUDGE = {self.config.use_vlm_judge}

            def _find_env_cfg_class():
                \"\"\"Find the EnvCfg class from module globals (imported via env_cfg).\"\"\"
                for name, obj in globals().items():
                    if (isinstance(obj, type)
                            and name.endswith("EnvCfg")
                            and name != "ManagerBasedRLEnvCfg"
                            and issubclass(obj, ManagerBasedRLEnvCfg)):
                        return obj
                return None

            def _sanitize_generated_env_cfg(env_cfg, task_doc=None):
                \"\"\"Patch common schema mismatches from generated env_cfg files.

                Some generated observation terms still pass `body_name=...` directly
                into IsaacLab observation helpers such as `mdp.body_pose_w()`.
                Current IsaacLab expects the body selection to live in
                `asset_cfg.body_names`, not as a standalone keyword argument.

                Some generated scene configs also map static primitive assets
                (for example tray walls) onto nested rigid prim paths such as
                `{{ENV_REGEX_NS}}/Tray/Base`. IsaacLab's rigid-object spawners can
                choke on those intermediate prims, so flatten static primitive
                assets onto `{{ENV_REGEX_NS}}/<asset_name>` and, when needed,
                replace `RigidObjectCfg` with `AssetBaseCfg`.
                \"\"\"
                patched_terms = []
                observations = getattr(env_cfg, "observations", None)
                if observations is not None:
                    for group_name, group_cfg in vars(observations).items():
                        if group_name.startswith("_") or group_cfg is None:
                            continue
                        for term_name, term_cfg in vars(group_cfg).items():
                            if term_name.startswith("_") or term_cfg is None:
                                continue
                            params = getattr(term_cfg, "params", None)
                            if not isinstance(params, dict):
                                continue

                            func_name = getattr(getattr(term_cfg, "func", None), "__name__", "")
                            if (
                                "robot_cfg" in params
                                and "asset_cfg" not in params
                                and func_name in {{"body_pose_w", "body_pos_w", "body_quat_w", "body_state_w"}}
                            ):
                                params["asset_cfg"] = params.pop("robot_cfg")
                                patched_terms.append(f"{{group_name}}/{{term_name}} (robot_cfg->asset_cfg)")

                            if "body_name" not in params:
                                continue

                            body_name = params.pop("body_name")
                            asset_cfg = params.get("asset_cfg")
                            if asset_cfg is not None and hasattr(asset_cfg, "body_names"):
                                asset_cfg.body_names = body_name
                                patched_terms.append(f"{{group_name}}/{{term_name}} (body_name->body_names)")
                            else:
                                # Put the parameter back if we cannot patch it safely.
                                params["body_name"] = body_name

                patched_scene_assets = []
                scene = getattr(env_cfg, "scene", None)
                if scene is not None and isinstance(task_doc, dict):
                    for asset in task_doc.get("assets", []):
                        if asset.get("source") != "primitive":
                            continue
                        if asset.get("physics", {{}}).get("rigid_body", False):
                            continue

                        asset_name = asset.get("name")
                        if not asset_name or not hasattr(scene, asset_name):
                            continue

                        scene_cfg = getattr(scene, asset_name)
                        old_prim_path = str(getattr(scene_cfg, "prim_path", ""))
                        flat_prim_path = "{{ENV_REGEX_NS}}/" + asset_name
                        init_state = getattr(scene_cfg, "init_state", None)
                        pos = list(getattr(init_state, "pos", [0.0, 0.0, 0.0]))
                        rot = list(getattr(init_state, "rot", [1.0, 0.0, 0.0, 0.0]))

                        if scene_cfg.__class__.__name__ == "RigidObjectCfg":
                            setattr(
                                scene,
                                asset_name,
                                AssetBaseCfg(
                                    prim_path=flat_prim_path,
                                    init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
                                    spawn=getattr(scene_cfg, "spawn", None),
                                ),
                            )
                            patched_scene_assets.append(
                                f"{{asset_name}}: RigidObjectCfg -> AssetBaseCfg @ {{flat_prim_path}}"
                            )
                            continue

                        if old_prim_path != flat_prim_path and old_prim_path.startswith("{{ENV_REGEX_NS}}/"):
                            scene_cfg.prim_path = flat_prim_path
                            patched_scene_assets.append(
                                f"{{asset_name}}: prim_path {{old_prim_path}} -> {{flat_prim_path}}"
                            )

                patched_actuators = []
                robot_scene_cfg = getattr(scene, "robot", None) if scene is not None else None
                actuator_cfgs = getattr(robot_scene_cfg, "actuators", None)
                if isinstance(actuator_cfgs, dict):
                    normalized_actuators = {{}}
                    for actuator_name, actuator_cfg in actuator_cfgs.items():
                        if not isinstance(actuator_cfg, dict):
                            normalized_actuators[actuator_name] = actuator_cfg
                            continue

                        joint_names = actuator_cfg.get("joint_names_expr", actuator_cfg.get("joint_names", []))
                        if isinstance(joint_names, str):
                            joint_names = [joint_names]

                        effort_limit = actuator_cfg.get("effort_limit")
                        effort_limit_per_joint = actuator_cfg.get("effort_limit_per_joint", {{}})
                        if effort_limit is None and isinstance(effort_limit_per_joint, dict) and effort_limit_per_joint:
                            effort_limit = max(float(value) for value in effort_limit_per_joint.values())
                        if effort_limit is None:
                            effort_limit = 1e9

                        normalized_actuators[actuator_name] = ImplicitActuatorCfg(
                            joint_names_expr=list(joint_names),
                            effort_limit=float(effort_limit),
                            stiffness=float(actuator_cfg.get("stiffness", 80.0)),
                            damping=float(actuator_cfg.get("damping", 4.0)),
                        )
                        patched_actuators.append(actuator_name)

                    if patched_actuators:
                        robot_scene_cfg.actuators = normalized_actuators

                patched_event_params = []
                disabled_event_terms = []
                events_cfg = getattr(env_cfg, "events", None)
                if events_cfg is not None:
                    for event_name, event_cfg in vars(events_cfg).items():
                        if event_name.startswith("_") or event_cfg is None:
                            continue
                        params = getattr(event_cfg, "params", None)
                        if not isinstance(params, dict):
                            continue
                        removed_orientation_keys = []
                        for orientation_key in (
                            "orientation_range",
                            "orientation",
                            "rotation_range",
                            "rotation",
                        ):
                            if orientation_key in params:
                                params.pop(orientation_key, None)
                                removed_orientation_keys.append(orientation_key)
                        if removed_orientation_keys:
                            patched_event_params.append(
                                f"{{event_name}} (dropped unsupported {{removed_orientation_keys}})"
                            )
                        if "asset_cfgs" in params:
                            setattr(events_cfg, event_name, None)
                            disabled_event_terms.append(
                                f"{{event_name}} (unsupported asset_cfgs/min_separation reset term)"
                            )

                if patched_terms:
                    print(
                        "Patched observation body selectors: "
                        f"{{patched_terms}}"
                    )
                if patched_scene_assets:
                    print(
                        "Patched static primitive scene assets: "
                        f"{{patched_scene_assets}}"
                    )
                if patched_actuators:
                    print(
                        "Normalized raw actuator dicts: "
                        f"{{patched_actuators}}"
                    )
                if patched_event_params:
                    print(
                        "Patched event params: "
                        f"{{patched_event_params}}"
                    )
                if disabled_event_terms:
                    print(
                        "Disabled unsupported event terms: "
                        f"{{disabled_event_terms}}"
                    )
                return patched_terms, patched_scene_assets, patched_actuators, patched_event_params, disabled_event_terms

            def _capture_judge_images(cameras):
                \"\"\"Capture wrist + front images for VLM judge (multi-view assessment).

                Returns dict {{"wrist": array, "front": array}} or None.
                Wrist view: close-up gripper/object interaction.
                Front view: full scene overview for task completion.
                \"\"\"
                if cameras is None:
                    return None
                try:
                    all_imgs = cameras.capture_all()
                    judge_imgs = {{}}
                    for name in ("wrist_cam", "wrist"):
                        if name in all_imgs and all_imgs[name] is not None:
                            judge_imgs["wrist"] = all_imgs[name]
                            break
                    for name in ("front_cam", "front"):
                        if name in all_imgs and all_imgs[name] is not None:
                            judge_imgs["front"] = all_imgs[name]
                            break
                    return judge_imgs if judge_imgs else None
                except Exception:
                    return None

            def _verify_goal_conditions(detector, goal, xy_threshold=0.05, height_diff=0.05, z_threshold=0.01):
                \"\"\"Verify task success by re-querying object positions against goal conditions.

                Uses geometric checks on the actual scene graph positions.
                Returns (success: bool, details: str).
                \"\"\"
                try:
                    # Use thresholds from success_criteria if available
                    sc = goal.get("success_criteria", {{}})
                    if "xy_threshold" in sc:
                        xy_threshold = sc["xy_threshold"]
                    if "height_diff" in sc:
                        height_diff = sc["height_diff"]
                    if "height_threshold" in sc:
                        z_threshold = sc["height_threshold"]

                    conditions = []
                    if "conditions" in sc:
                        conditions = sc["conditions"]
                    elif "stacking_order" in sc:
                        order = sc["stacking_order"]
                        for i in range(1, len(order)):
                            conditions.append({{
                                "type": "on_top_of",
                                "object": order[i],
                                "target": order[i - 1],
                            }})
                    elif "conditions" in goal:
                        conditions = goal["conditions"]

                    # Infer stacking from goal description if no explicit conditions
                    if not conditions:
                        desc = goal.get("description", "").lower()
                        if "stack" in desc:
                            # Direct stacking verification: check that all manipulable
                            # objects are vertically aligned with proper height differences.
                            all_objs = detector.get_all_objects()
                            stackable = sorted([
                                name for name in all_objs
                                if "cube" in name.lower() or "block" in name.lower() or "box" in name.lower()
                            ])
                            if len(stackable) >= 2:
                                positions = {{}}
                                for name in stackable:
                                    try:
                                        positions[name] = detector.get_object_position(name)
                                    except Exception:
                                        pass
                                # Sort by Z to find actual stacking order
                                sorted_by_z = sorted(positions.items(), key=lambda x: x[1][2])
                                # Check: all adjacent pairs must be XY-aligned and Z-spaced
                                all_stacked = True
                                stack_details = []
                                for i in range(1, len(sorted_by_z)):
                                    top_name, top_pos = sorted_by_z[i]
                                    bot_name, bot_pos = sorted_by_z[i-1]
                                    xy_err = np.linalg.norm(top_pos[:2] - bot_pos[:2])
                                    z_diff = top_pos[2] - bot_pos[2]
                                    ok = xy_err < xy_threshold and abs(z_diff - height_diff) < z_threshold
                                    if not ok:
                                        all_stacked = False
                                    stack_details.append(
                                        f"{{top_name}} on {{bot_name}}: xy_err={{xy_err:.3f}}m z_diff={{z_diff:.3f}}m {{'OK' if ok else 'FAIL'}}"
                                    )
                                detail_str = "; ".join(stack_details)
                                n_ok = sum(1 for d in stack_details if "OK" in d)
                                n_total = len(stack_details)
                                return all_stacked, f"{{n_ok}}/{{n_total}} stack pairs: {{detail_str}}"

                    if not conditions:
                        return False, "no verifiable conditions"

                    passed = 0
                    total = 0
                    details = []
                    for cond in conditions:
                        cond_type = cond.get("type", cond.get("relation", ""))
                        if cond_type in ("on_top_of", "stacked"):
                            total += 1
                            top_obj = cond.get(
                                "object",
                                cond.get("top", cond.get("subject")),
                            )
                            bottom_obj = cond.get("target", cond.get("bottom"))
                            cond_tol = cond.get("tolerance", xy_threshold)
                            if not top_obj or not bottom_obj:
                                details.append(f"skip: missing object names")
                                continue
                            try:
                                top_pos = detector.get_object_position(top_obj)
                                bot_pos = detector.get_object_position(bottom_obj)
                            except Exception as e:
                                details.append(f"{{top_obj}} on {{bottom_obj}}: detection failed ({{e}})")
                                continue
                            xy_err = np.linalg.norm(top_pos[:2] - bot_pos[:2])
                            z_diff = top_pos[2] - bot_pos[2]
                            ok = (
                                xy_err < cond_tol
                                and abs(z_diff - height_diff)
                                < max(z_threshold, cond_tol)
                            )
                            if ok:
                                passed += 1
                            details.append(
                                f"{{top_obj}} on {{bottom_obj}}: xy_err={{xy_err:.3f}}m z_diff={{z_diff:.3f}}m {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type == "at_position":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            target = cond.get("position", cond.get("target"))
                            tol = cond.get("tolerance", xy_threshold)
                            if not obj or target is None:
                                details.append("skip: missing at_position fields")
                                continue
                            try:
                                obj_pos = detector.get_object_position(obj)
                            except Exception as e:
                                details.append(f"{{obj}} at_position: detection failed ({{e}})")
                                continue
                            target_pos = None
                            if isinstance(target, (list, tuple)) and len(target) == 3:
                                target_pos = np.array(target, dtype=float)
                            elif isinstance(target, str):
                                try:
                                    target_pos = detector.get_object_position(target)
                                except Exception as e:
                                    details.append(f"{{obj}} at_position target '{{target}}' missing ({{e}})")
                                    continue
                            if target_pos is None:
                                details.append(f"{{obj}} at_position: unsupported target={{target}}")
                                continue
                            xy_err = np.linalg.norm(obj_pos[:2] - target_pos[:2])
                            ok = xy_err < tol
                            if ok:
                                passed += 1
                            details.append(
                                f"{{obj}} at_position: xy_err={{xy_err:.3f}}m {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type in ("lifted", "above", "height_above"):
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            min_height = cond.get("height", cond.get("value", 0.1))
                            try:
                                pos = detector.get_object_position(obj)
                                ok = pos[2] > min_height
                                if ok:
                                    passed += 1
                                details.append(f"{{obj}} lifted: z={{pos[2]:.3f}}m {{'OK' if ok else 'FAIL'}}")
                            except Exception as e:
                                details.append(f"{{obj}} lifted: detection failed ({{e}})")

                    if total == 0:
                        return False, "no checkable conditions"
                    success = passed == total
                    return success, f"{{passed}}/{{total}} conditions met: {{'; '.join(details)}}"
                except Exception as e:
                    return False, f"verification error: {{e}}"

            def main():
                # Load config
                with open("{config_json_path}") as f:
                    cfg = json.load(f)

                robot_cfg = load_robot_config(cfg["robot_name"])

                # Load task YAML
                import yaml
                with open(cfg["yaml_path"]) as f:
                    task_doc = yaml.safe_load(f)

                # Create environment with cameras injected into scene config
                env_cfg_class = _find_env_cfg_class()
                if env_cfg_class is None:
                    print("ERROR: Could not find EnvCfg class in env_cfg.py")
                    sys.exit(1)

                env_cfg = env_cfg_class()
                env_cfg.scene.num_envs = cfg["num_envs"]
                _sanitize_generated_env_cfg(env_cfg, task_doc)

                # --- Patch episode length to prevent auto-reset mid-collection ---
                # LLM-generated env_cfg uses episode_length_s=30s (typical for RL).
                # Data collection skills need much longer episodes (5+ minutes).
                # Set to 600s to avoid premature resets during manipulation sequences.
                env_cfg.episode_length_s = 600.0
                print("Patched episode_length_s: 600.0s (prevents auto-reset during collection)")

                # --- Disable non-timeout terminations to prevent mid-episode auto-reset ---
                # The env's success/failure terminations (e.g. cubes_stacked, object_dropped)
                # trigger auto-reset which destroys the scene mid-episode.
                # Data collection handles success verification separately (goal check / VLM).
                if hasattr(env_cfg, 'terminations'):
                    disabled = []
                    for term_name in list(vars(env_cfg.terminations)):
                        if term_name.startswith('_'):
                            continue
                        term = getattr(env_cfg.terminations, term_name, None)
                        if term is None:
                            continue
                        # Keep time_out (it's our episode length guard), disable everything else
                        if hasattr(term, 'time_out') and term.time_out:
                            continue
                        # Disable by setting func to None
                        setattr(env_cfg.terminations, term_name, None)
                        disabled.append(term_name)
                    if disabled:
                        print(f"Disabled terminations (prevent mid-episode reset): {{disabled}}")

                # --- Patch action space for absolute joint position control ---
                # IK solver outputs absolute joint angles (radians).
                # LLM-generated env_cfg may use scale<1.0 + use_default_offset=True,
                # which transforms: target = action * scale + default_pose (WRONG).
                # We need pass-through: target = action (scale=1.0, offset=0).
                if hasattr(env_cfg, 'actions') and hasattr(env_cfg.actions, 'arm_action'):
                    env_cfg.actions.arm_action.scale = 1.0
                    env_cfg.actions.arm_action.use_default_offset = False
                    print("Patched arm_action: scale=1.0, use_default_offset=False (absolute joint control)")

                # --- Patch initial joint state from robot profile ready_pose ---
                # LLM-generated env_cfg may use stale joint values (e.g. stiff standing pose).
                # Override with the robot profile's task-ready pose for proper workspace alignment.
                ready_pose = {repr({k: v for k, v in self.robot_cfg.ready_pose.items()})}
                if ready_pose and hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'init_state'):
                    env_cfg.scene.robot.init_state.joint_pos = ready_pose
                    print(f"Patched initial joint_pos from robot profile ready_pose: {{ready_pose}}")

                # --- Patch actuator PD gains + disable gravity (HIGH_PD config) ---
                # For IK-based absolute control, need HIGH stiffness/damping.
                # Use max(original, target) to never downgrade — UR10e shoulder
                # defaults to 1320 Nm/rad which we must preserve, while its
                # wrist defaults to 216 which needs upgrading to resist gripper
                # contact forces during grasping.
                if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'actuators'):
                    def _cfg_get(cfg_obj, key, default=None):
                        if isinstance(cfg_obj, dict):
                            return cfg_obj.get(key, default)
                        return getattr(cfg_obj, key, default)

                    def _cfg_set(cfg_obj, key, value):
                        if isinstance(cfg_obj, dict):
                            cfg_obj[key] = value
                        else:
                            setattr(cfg_obj, key, value)

                    for act_name, actuator in env_cfg.scene.robot.actuators.items():
                        if 'hand' not in act_name and 'finger' not in act_name and 'gripper' not in act_name:
                            orig_s = _cfg_get(actuator, 'stiffness', 80.0)
                            orig_d = _cfg_get(actuator, 'damping', 4.0)
                            orig_s = float(orig_s) if isinstance(orig_s, (int, float)) else 80.0
                            orig_d = float(orig_d) if isinstance(orig_d, (int, float)) else 4.0
                            # Proportional scaling: preserve per-joint ratios.
                            # Wrist joints (orig=216) stay at 1080 — higher values
                            # cause numerical instability due to low wrist link inertia
                            # at 100Hz physics. Joint coupling is handled via Cartesian
                            # waypoint subdivision in sim_skills instead.
                            STIFFNESS_MULT = 5.0
                            MAX_STIFFNESS = 5000.0
                            MAX_DAMPING = 400.0
                            new_s = min(orig_s * STIFFNESS_MULT, MAX_STIFFNESS)
                            new_d = min(orig_d * STIFFNESS_MULT, MAX_DAMPING)
                            if robot_cfg.name == "so101":
                                new_s = max(new_s, 300.0)
                                new_d = max(new_d, 30.0)
                            if robot_cfg.name == "ur10e" and act_name == "wrist":
                                # The Robotiq grasp loads wrist_2 heavily during post-close lift.
                                # Give the shared UR wrist actuator more authority than the
                                # generic x5 scaling so the wrist can hold its fixed-joint pose.
                                new_s = max(new_s, 2000.0)
                                new_d = max(new_d, 200.0)
                            _cfg_set(actuator, 'stiffness', new_s)
                            _cfg_set(actuator, 'damping', new_d)
                            # Remove effort limit clipping — let PD solver produce
                            # whatever torque it needs for unconditional stability.
                            _cfg_set(actuator, 'effort_limit', 1e9)
                            _cfg_set(actuator, 'effort_limit_sim', 1e9)
                            print(f"  {{act_name}}: stiffness={{orig_s}}->{{new_s}}, damping={{orig_d}}->{{new_d}}, effort_limit=1e9")
                    print("Patched arm actuators: proportional PD scaling (×5, cap 5000), effort_limit=1e9")
                    # --- Patch gripper drive stiffness for reliable grasping ---
                    # Some grippers (e.g. Robotiq 2F-85: stiffness=11.25 Nm/rad) use
                    # very low stiffness that causes slow/incomplete closure at data-collection
                    # timescales. Override with robot profile grasp_stiffness when set.
                    grasp_stiffness = {self.robot_cfg.gripper_grasp_stiffness!r}
                    if grasp_stiffness > 0:
                        for act_name, actuator in env_cfg.scene.robot.actuators.items():
                            act_name_lower = act_name.lower()
                            if (
                                'gripper_drive' in act_name_lower
                                or 'gripper_finger' in act_name_lower
                                or 'gripper' in act_name_lower
                                or 'hand' in act_name_lower
                                or 'finger' in act_name_lower
                            ):
                                old_s = _cfg_get(actuator, 'stiffness', None)
                                _cfg_set(actuator, 'stiffness', grasp_stiffness)
                                _cfg_set(actuator, 'damping', grasp_stiffness * 0.05)
                                _cfg_set(actuator, 'effort_limit_sim', max(100.0, grasp_stiffness * 0.2))
                                print(f"Patched {{act_name}}: stiffness {{old_s}} -> {{grasp_stiffness}} (grasp_stiffness override)")
                            elif 'gripper_passive' in act_name_lower or 'passive' in act_name_lower:
                                # Passive mimic joints (stiffness=0, damping=0) are completely
                                # unconstrained, causing reaction forces on wrist joints during
                                # gripper closure. Give them enough stiffness to track targets.
                                old_s = _cfg_get(actuator, 'stiffness', None)
                                _cfg_set(actuator, 'stiffness', grasp_stiffness * 0.5)
                                _cfg_set(actuator, 'damping', grasp_stiffness * 0.025)
                                print(f"Patched {{act_name}}: stiffness {{old_s}} -> {{grasp_stiffness * 0.5}} (passive joint stabilization)")
                if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'spawn'):
                    if hasattr(env_cfg.scene.robot.spawn, 'rigid_props') and env_cfg.scene.robot.spawn.rigid_props is not None:
                        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
                        print("Patched robot: disable_gravity=True (matching HIGH_PD_CFG)")
                    # Moderately increase solver iterations for tighter constraint
                    # satisfaction without GPU OOM (64 caused OOM on 8GB GPUs).
                    if hasattr(env_cfg.scene.robot.spawn, 'articulation_props') and env_cfg.scene.robot.spawn.articulation_props is not None:
                        env_cfg.scene.robot.spawn.articulation_props.solver_position_iteration_count = 32
                        env_cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 2
                        print("Patched solver iterations: position=32, velocity=2")
                        # Keep self-collisions DISABLED (UR10e default).
                        # Finger collision shapes still collide with external objects (cubes)
                        # but NOT with other robot links (prevents wrist instability).
                        env_cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
                        print("Patched articulation: enabled_self_collisions=False (finger-cube only)")
                    # Enable collision on all robot body meshes and set contact offset
                    # for reliable contact detection between finger pads and objects.
                    import isaaclab.sim as sim_utils_patch
                    if not hasattr(env_cfg.scene.robot.spawn, 'collision_props') or env_cfg.scene.robot.spawn.collision_props is None:
                        env_cfg.scene.robot.spawn.collision_props = sim_utils_patch.CollisionPropertiesCfg(
                            collision_enabled=True,
                            contact_offset=0.005,
                            rest_offset=0.0,
                        )
                        print("Patched robot spawn: collision_props(collision_enabled=True, contact_offset=0.005)")

                # --- Inject cameras into scene config BEFORE env construction ---
                # This ensures cameras get the PLAY event and initialize properly.
                # Top view: near-top-down, robot at bottom of frame, workspace in center.
                # Wrist view: body-mounted on end-effector for close-up manipulation view.
                camera_attr_names = inject_cameras_into_scene(env_cfg, robot_cfg, camera_names=["top", "wrist", "front"])
                if camera_attr_names:
                    print(f"Injected cameras into scene: {{camera_attr_names}}")

                using_local_ur10e_asset = False
                if robot_cfg.name == "ur10e":
                    from src.data_collection.ur10e_patches import (
                        apply_local_ur10e_asset_override,
                        should_use_local_ur10e_asset,
                    )

                    if should_use_local_ur10e_asset():
                        using_local_ur10e_asset = apply_local_ur10e_asset_override(env_cfg, verbose=True)
                    else:
                        print("Using default Isaac Sim UR10e asset (local override disabled)")

                env = ManagerBasedRLEnv(cfg=env_cfg)

                # Add the minimal UR10e/Robotiq pad-collision patch only in the
                # UR10e path.  Keep this logic centralized so debug harnesses and
                # the main pipeline use the same geometry source-of-truth.
                try:
                    if robot_cfg.name == "ur10e" and not using_local_ur10e_asset:
                        from src.data_collection.ur10e_patches import ensure_ur10e_pad_collisions

                        added_paths = ensure_ur10e_pad_collisions(env.sim.stage, env_idx=0, verbose=True)
                        print(f"Ensured {{len(added_paths)}} UR10e finger-pad collision shapes")
                    elif robot_cfg.name == "so101":
                        from src.data_collection.so101_patches import ensure_so101_claw_collisions

                        added_paths = ensure_so101_claw_collisions(env.sim.stage, env_idx=0, verbose=True)
                        print(f"Ensured {{len(added_paths)}} SO-101 claw collision proxies")
                except Exception as coll_err:
                    print(f"WARNING: Failed to add runtime collision shapes: {{coll_err}}")
                    traceback.print_exc()

                # Initialize components
                robot_interface = SimRobotInterface(env, robot_cfg, env_idx=0)
                detector = SimDetector(env, task_doc, env_idx=0)

                # Read robot base position for IK frame conversion
                robot_base_pos = env.scene["robot"].data.root_pos_w[0].cpu().numpy()
                print(f"Robot base position (world frame): {{robot_base_pos}}")

                # Initialize multi-camera system from scene-integrated cameras
                cameras = None
                camera = None  # legacy fallback
                if camera_attr_names:
                    try:
                        cameras = SceneCameraManager(env, camera_attr_names)
                        print(f"SceneCameraManager initialized: {{cameras.camera_names}}")
                    except Exception as cam_err:
                        print(f"WARNING: SceneCameraManager init failed ({{cam_err}}).")
                        cameras = None

                recorder = SimRecorder(
                    robot_cfg=robot_cfg,
                    output_dir=cfg["output_dir"],
                    dataset_name="raw_dataset",
                    fps=cfg["recording_fps"],
                )

                # Create IK solver (Pinocchio when URDF exists, otherwise DifferentialIK fallback)
                ik_solver = create_ik_solver(robot_cfg, robot_interface=robot_interface)

                # Create skills (with multi-camera for image recording)
                skills = SimSkills(
                    robot_interface=robot_interface,
                    robot_cfg=robot_cfg,
                    detector=detector,
                    recorder=recorder,
                    camera=camera,
                    cameras=cameras,
                    ik_solver=ik_solver,
                    base_offset=robot_base_pos,
                )

                cap_profile = select_cap_profile(robot_cfg)
                cap_generator = None
                try:
                    from src.common.llm_client import AzureOpenAIClient

                    llm_client = AzureOpenAIClient(model=cfg["llm_model"])
                    cap_generator = SimCaPGenerator(
                        llm_client,
                        robot_cfg,
                        retry_max=cfg.get("skill_retry_max", 1),
                    )
                    print(
                        f"SimCaPGenerator initialized: model={{cfg['llm_model']}}, "
                        f"skill_class={{cap_profile.class_name}}"
                    )
                except Exception as e:
                    print(f"SimCaPGenerator init failed: {{e}}")

                # Initialize VLM judge if enabled
                sim_judge = None
                judge_init_error = ""
                if USE_VLM_JUDGE:
                    sim_judge = SimJudge(model="gpt-5")
                    if not sim_judge.is_available:
                        judge_init_error = sim_judge.availability_reason
                        print(f"SimJudge unavailable: {{judge_init_error}}")
                        sim_judge = None
                    else:
                        print("SimJudge initialized (VLM-based episode judging enabled)")

                results = {{
                    "episodes": [],
                    "total_episodes": 0,
                    "successful_episodes": 0,
                    "geometry_successful_episodes": 0,
                    "vlm_successful_episodes": 0,
                    "overall_successful_episodes": 0,
                    "pipeline_completed": False,
                    "target_met": False,
                    "geometry_target_met": False,
                    "vlm_target_met": False,
                    "overall_target_met": False,
                    "raw_dataset": None,
                    "robot_skill_class": cap_profile.class_name,
                    "judge_enabled": USE_VLM_JUDGE,
                    "judge_available": sim_judge is not None,
                    "judge_unavailable_reason": judge_init_error or None,
                }}

                # Success-based episode loop
                target_success = cfg.get("target_successful_episodes", cfg["max_episodes"])
                max_attempts = cfg.get("max_total_attempts", target_success * 5)
                episode_idx = 0
                successful_count = 0

                try:
                    supported_task, unsupported_reason = is_supported_tabletop_task(task_doc)
                    if not supported_task:
                        raise RuntimeError(
                            f"Task unsupported by current CaP tabletop profile: {{unsupported_reason}}"
                        )
                    if cap_generator is None:
                        raise RuntimeError("CaP generator unavailable; check Azure OpenAI configuration")
                    if USE_VLM_JUDGE and sim_judge is None:
                        raise RuntimeError(f"VLM judge unavailable: {{judge_init_error}}")

                    while successful_count < target_success and episode_idx < max_attempts:
                        print(f"\\n=== Episode {{episode_idx + 1}} (success: {{successful_count}}/{{target_success}}, max: {{max_attempts}}) ===")

                        # Reset environment
                        obs, info = env.reset()
                        robot_interface.reset_termination_flags()

                        # Render warmup: hold initial position for a few steps so camera
                        # buffers are populated (first frames after reset can be black).
                        # CRITICAL: Must send initial joint positions as action, NOT zeros!
                        # With scale=1.0 + use_default_offset=False, zero action = target 0.0
                        # which drives the robot away from its initial pose.
                        _warmup_action = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
                        _num_arm_joints = min(
                            len(robot_cfg.arm_joint_names),
                            env.scene["robot"].data.joint_pos.shape[1],
                            env.action_space.shape[-1],
                        )
                        _warmup_action[0, :_num_arm_joints] = env.scene["robot"].data.joint_pos[0, :_num_arm_joints].clone()
                        if env.action_space.shape[-1] > _num_arm_joints:
                            _warmup_action[0, _num_arm_joints] = 1.0  # gripper open
                        for _ in range(10):
                            env.step(_warmup_action)

                        # Start recording
                        task_desc = task_doc.get("task", {{}}).get("description", "")
                        recorder.start_episode(task_desc, episode_idx)

                        # Get scene state
                        scene_state = detector.get_all_objects()
                        print(f"Scene objects: {{list(scene_state.keys())}}")

                        # Capture initial images for VLM judge (wrist + front)
                        initial_images = None
                        if sim_judge is not None:
                            initial_images = _capture_judge_images(cameras)

                        # Debug: save initial images from ALL cameras (first episode)
                        if episode_idx == 0 and cameras is not None:
                            try:
                                from PIL import Image
                                all_dbg = cameras.capture_all()
                                for _cname, _cimg in all_dbg.items():
                                    if _cimg is not None:
                                        _dp = os.path.join(cfg["output_dir"], f"debug_initial_{{_cname}}.png")
                                        Image.fromarray(_cimg).save(_dp)
                                        print(f"Debug image saved: {{_dp}} (shape={{_cimg.shape}})")
                            except Exception as _de:
                                print(f"Debug image capture failed: {{_de}}")

                        run_dir = make_episode_run_dir(cfg["output_dir"], episode_idx)
                        initial_image_paths = save_judge_images(run_dir, "initial", initial_images)
                        final_image_paths = []
                        generated_code = ""
                        translated_positions = {{}}
                        artifact_paths = {{}}
                        execution_context_path = None
                        codegen_success = False
                        codegen_error = ""
                        execution_error = ""
                        blocked_fallback_reason = ""
                        geometry_success = False
                        vlm_success = False
                        overall_success = False
                        judge_available = False

                        # Execute ADC-style generated code
                        try:
                            generation = cap_generator.generate_code(
                                task_doc=task_doc,
                                scene_state=scene_state,
                                task_description=task_desc,
                            )
                            translated_positions = generation.translated_positions
                            generated_code = generation.generated_code
                            artifact_paths = save_code_generation_artifacts(
                                run_dir=run_dir,
                                generated_code=generated_code,
                                raw_response=generation.raw_response,
                                translated_positions=translated_positions,
                            )
                            codegen_success = True
                            print(
                                f"Generated CaP code: {{artifact_paths.get('generated_code', '')}} "
                                f"({{len(generated_code)}} chars)"
                            )

                            CaPRuntimeContext.configure(
                                sim_skills=skills,
                                detector=detector,
                                robot_cfg=robot_cfg,
                                translated_positions=translated_positions,
                            )
                            exec_globals = {{
                                "__name__": "__main__",
                                "positions": translated_positions,
                            }}
                            exec(generated_code, exec_globals)

                            # Capture env termination + goal state BEFORE move_to_ready
                            # (move_to_ready can knock objects; env auto-resets on success)
                            env_terminated_during_skills = robot_interface.env_terminated
                            env_truncated_during_skills = robot_interface.env_truncated

                            # Verify goal conditions geometrically BEFORE moving away
                            goal = task_doc.get("goal", {{}})
                            pre_retreat_goal_ok = False
                            pre_retreat_goal_details = ""
                            if not env_terminated_during_skills:
                                pre_retreat_goal_ok, pre_retreat_goal_details = _verify_goal_conditions(detector, goal)

                            # Capture final images BEFORE retreat (for VLM judge)
                            final_images = None
                            if sim_judge is not None:
                                final_images = _capture_judge_images(cameras)
                                final_image_paths = save_judge_images(run_dir, "final", final_images)

                            skills.move_to_ready(
                                duration=1.5,
                                safe_retreat=True,
                                skill_description="return to the ready pose after executing the generated policy",
                            )
                            execution_ok = True
                        except Exception as e:
                            if codegen_success:
                                execution_error = str(e)
                            else:
                                codegen_error = str(e)
                            if isinstance(e, PolicyFallbackBlockedError):
                                blocked_fallback_reason = str(e)
                            if isinstance(e, RecorderStorageError):
                                print(f"Recorder storage error: {{e}}")
                            else:
                                print(f"CaP execution error: {{e}}")
                            traceback.print_exc()
                            execution_ok = False
                            env_terminated_during_skills = robot_interface.env_terminated
                            env_truncated_during_skills = robot_interface.env_truncated
                            pre_retreat_goal_ok = False
                            pre_retreat_goal_details = "execution error"
                            final_images = None
                        finally:
                            CaPRuntimeContext.clear()
                            if codegen_success:
                                execution_context_path = save_execution_context(
                                    run_dir=run_dir,
                                    instruction=task_desc,
                                    object_positions=translated_positions,
                                    generated_code=generated_code,
                                    execution_success=execution_ok,
                                    metadata={{
                                        "robot_name": robot_cfg.name,
                                        "skill_class": cap_profile.class_name,
                                        "episode": episode_idx,
                                    }},
                                )

                        verdict = None

                        # Determine geometric success first; VLM is tracked separately.
                        if execution_ok:
                            env_success = env_terminated_during_skills and not env_truncated_during_skills
                            if env_success:
                                print("Success: environment termination condition met during skills")
                                geometry_success = True
                            elif pre_retreat_goal_ok:
                                print(f"Success (goal verified): {{pre_retreat_goal_details}}")
                                geometry_success = True
                            else:
                                if pre_retreat_goal_details:
                                    print(f"Goal verification: {{pre_retreat_goal_details}}")
                                print("Episode FAILED: goal conditions not met")
                                geometry_success = False

                        if sim_judge is not None:
                            judge_available = (
                                execution_ok
                                and final_images is not None
                                and initial_images is not None
                            )
                        if judge_available:
                            verdict = sim_judge.judge_episode(
                                task_description=task_desc,
                                initial_images=initial_images,
                                final_images=final_images,
                                object_positions=translated_positions or scene_state,
                                executed_code=generated_code,
                            )
                            vlm_success = bool(verdict["success"])
                            print(f"VLM Judge: {{verdict['prediction']}} — {{verdict['reasoning'][:80]}}")
                        elif sim_judge is not None:
                            print("VLM Judge unavailable for episode: missing images or execution failed")

                        overall_success = geometry_success and (vlm_success if sim_judge is not None else True)
                        success = geometry_success

                        # End episode (discard failures when targeting success count)
                        discard_failed = (cfg.get("target_successful_episodes", 0) > 0
                                         and cfg.get("target_successful_episodes", 0) < cfg.get("max_total_attempts", 999)
                                         and not success)
                        recorder.end_episode(success=success, discard=discard_failed)

                        episode_result = {{
                            "episode": episode_idx,
                            "success": success,
                            "geometry_success": geometry_success,
                            "vlm_success": vlm_success,
                            "overall_success": overall_success,
                            "steps": recorder.current_episode_steps,
                            "discarded": discard_failed,
                            "robot_skill_class": cap_profile.class_name,
                            "codegen_success": codegen_success,
                            "execution_success": execution_ok,
                            "judge_enabled": sim_judge is not None,
                            "judge_available": judge_available,
                            "generated_code_path": artifact_paths.get("generated_code"),
                            "raw_response_path": artifact_paths.get("raw_response"),
                            "scene_positions_path": artifact_paths.get("scene_positions"),
                            "execution_context_path": execution_context_path,
                            "initial_judge_images": initial_image_paths,
                            "final_judge_images": final_image_paths,
                        }}
                        if blocked_fallback_reason:
                            episode_result["blocked_fallback_reason"] = blocked_fallback_reason
                        if codegen_error:
                            episode_result["codegen_error"] = codegen_error
                        if execution_error:
                            episode_result["execution_error"] = execution_error
                        if verdict is not None:
                            episode_result["judge_prediction"] = verdict.get("prediction", "N/A")
                            episode_result["judge_reasoning"] = verdict.get("reasoning", "")
                        results["episodes"].append(episode_result)
                        results["total_episodes"] += 1
                        if success:
                            results["successful_episodes"] += 1
                            results["geometry_successful_episodes"] += 1
                            successful_count += 1
                        if vlm_success:
                            results["vlm_successful_episodes"] += 1
                        if overall_success:
                            results["overall_successful_episodes"] += 1

                        print(
                            f"Episode {{episode_idx + 1}} complete: "
                            f"geometry={{'SUCCESS' if geometry_success else 'FAILED'}} "
                            f"vlm={{'SUCCESS' if vlm_success else 'FAILED'}} "
                            f"overall={{'SUCCESS' if overall_success else 'FAILED'}}"
                            f"{{'(discarded)' if discard_failed else ''}}"
                        )
                        episode_idx += 1

                    print(
                        f"\\nCompleted: geometry={{results['geometry_successful_episodes']}}/{{target_success}} "
                        f"overall={{results['overall_successful_episodes']}}/{{results['total_episodes']}} "
                        f"in {{episode_idx}} attempts"
                    )

                except KeyboardInterrupt:
                    print("\\nCollection interrupted by user")
                    results["pipeline_error"] = "interrupted"
                except Exception as e:
                    print(f"Pipeline error: {{e}}")
                    results["pipeline_error"] = str(e)
                    traceback.print_exc()
                finally:
                    if recorder.is_recording:
                        try:
                            recorder.end_episode(success=False, discard=False)
                        except Exception as end_err:
                            print(f"WARNING: Failed to close active episode: {{end_err}}")
                            traceback.print_exc()
                            results.setdefault("cleanup_errors", []).append(
                                f"end_episode failed: {{end_err}}"
                            )

                    dataset_path = None
                    try:
                        dataset_path = recorder.finalize()
                    except Exception as finalize_err:
                        print(f"WARNING: Dataset finalize failed: {{finalize_err}}")
                        traceback.print_exc()
                        results.setdefault("cleanup_errors", []).append(
                            f"finalize failed: {{finalize_err}}"
                        )
                        results.setdefault(
                            "pipeline_error",
                            f"finalize failed: {{finalize_err}}",
                        )

                    results["dataset_path"] = dataset_path
                    results["raw_dataset"] = dataset_path
                    results["target_met"] = (
                        results["successful_episodes"] >= target_success
                    )
                    results["geometry_target_met"] = (
                        results["geometry_successful_episodes"] >= target_success
                    )
                    if USE_VLM_JUDGE:
                        results["vlm_target_met"] = (
                            results["vlm_successful_episodes"] >= target_success
                        )
                    results["overall_target_met"] = (
                        results["overall_successful_episodes"] >= target_success
                    )
                    results["pipeline_completed"] = dataset_path is not None
                    results["success"] = results["pipeline_completed"]

                    # Save results
                    results_path = os.path.join(cfg["output_dir"], "collection_results.json")
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

                    if results["pipeline_completed"]:
                        with open(MARKER_FILE, "w") as f:
                            f.write("COMPLETE")

                    print(f"\\nResults saved to {{results_path}}")
                    print(f"Dataset at {{dataset_path}}")

                    # Close simulation
                    env.close()
                    simulation_app.close()


            if __name__ == "__main__":
                main()
        ''')

        with open(runner_path, "w") as f:
            f.write(script)

        logger.info(f"Generated collection runner: {runner_path}")
        return runner_path

    def _execute_subprocess(self, runner_path: Path) -> tuple[bool, str]:
        """Execute the collection runner in IsaacLab conda subprocess."""
        isaaclab_sh = Path(ISAACLAB_PATH) / "isaaclab.sh"

        if not isaaclab_sh.exists():
            # Fallback to conda run
            cmd = (
                f"conda run -n {CONDA_ENV} --no-capture-output "
                f"python {runner_path}"
            )
        else:
            isaaclab_cmd = f"{isaaclab_sh} -p {runner_path}"
            if self.config.env_headless:
                isaaclab_cmd += " --headless"
            # Wrap with conda run to ensure correct Python environment
            cmd = f"conda run -n {CONDA_ENV} --no-capture-output bash -c \"{isaaclab_cmd}\""

        logger.info(f"Executing: {cmd}")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if self.config.env_headless:
            # Headless rendering: remove DISPLAY (forces EGL instead of GLX)
            # and force NVIDIA-only Vulkan ICD to avoid non-NVIDIA driver issues.
            # NOTE: First-run shader compilation can take 5-10 minutes.
            env.pop("DISPLAY", None)
            env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(self.env_dir),
            )

            output_lines = []
            start_time = time.time()

            for line in proc.stdout:
                line = line.rstrip()
                print(line)
                output_lines.append(line)

                # Check timeout
                if time.time() - start_time > self.config.execution_timeout:
                    proc.kill()
                    logger.error("Execution timeout reached")
                    break

            proc.wait()
            output = "\n".join(output_lines)

            # Check completion marker / results file
            marker_path = self.output_dir / "COLLECTION_COMPLETE_MARKER"
            legacy_marker_path = self.output_dir / "COLLECTION_SUCCESS_MARKER"
            success = marker_path.exists() or legacy_marker_path.exists()

            if not success:
                results_path = self.output_dir / "collection_results.json"
                if results_path.exists():
                    with open(results_path) as f:
                        results = json.load(f)
                    success = bool(results.get("pipeline_completed", False))

            if success:
                logger.info("Data collection completed successfully")
            else:
                logger.warning(
                    f"Data collection may have failed (exit code: {proc.returncode})"
                )

            return success, output

        except Exception as e:
            logger.error(f"Subprocess execution failed: {e}")
            return False, str(e)

    def _convert_to_lerobot(self, raw_data_dir: Path):
        """Convert raw dataset to LeRobot v3.0 format (post-processing)."""
        try:
            from .sim_recorder import convert_to_lerobot

            lerobot_path = convert_to_lerobot(
                raw_dataset_dir=str(raw_data_dir),
                repo_id=self.config.dataset_repo_id,
            )
            logger.info(f"LeRobot dataset created at: {lerobot_path}")
        except ImportError:
            logger.warning(
                "LeRobot conversion skipped (lerobot package not available). "
                "Raw data saved at: {raw_data_dir}"
            )
        except Exception as e:
            logger.error(f"LeRobot conversion failed: {e}")


def run_batch(
    task_dir: str,
    config: Optional[DataCollectionConfig] = None,
    episodes_per_task: int = 10,
) -> list[dict]:
    """
    Run data collection for all task YAMLs in a directory.

    Args:
        task_dir: Directory containing task YAML files
        config: Pipeline config
        episodes_per_task: Override max_episodes per task
    """
    task_dir = Path(task_dir)
    yamls = sorted(task_dir.rglob("*.yaml"))

    if not yamls:
        logger.warning(f"No YAML files found in {task_dir}")
        return []

    results = []
    for yaml_path in yamls:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {yaml_path}")
        logger.info(f"{'='*60}")

        cfg = config or load_pipeline_config()
        cfg.max_episodes = episodes_per_task

        try:
            pipeline = DataCollectionPipeline(
                yaml_path=str(yaml_path),
                config=cfg,
            )
            result = pipeline.run()
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for {yaml_path}: {e}")
            results.append({"success": False, "error": str(e), "yaml": str(yaml_path)})

    return results
