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

from .config import (
    PROJECT_ROOT,
    DataCollectionConfig,
    RobotSimConfig,
    load_pipeline_config,
    load_robot_config,
)

logger = logging.getLogger(__name__)

ISAACLAB_PATH = os.environ.get(
    "ISAACLAB_PATH", "/home/vpraise/workspace/IsaacLab"
)
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

        return {
            "success": success,
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
                robot_type = asset.get("robot_type", "").lower()
                if robot_type in ("franka", "openarm", "ur10", "so101"):
                    return robot_type

        # Fallback: check task name or metadata
        task_name = self.task_doc.get("task", {}).get("name", "").lower()
        for robot in ("franka", "openarm", "ur10", "so101"):
            if robot in task_name:
                return robot

        # Check file path
        yaml_str = str(self.yaml_path).lower()
        for robot in ("franka", "openarm", "ur10", "so101"):
            if robot in yaml_str:
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

            from isaaclab.envs import ManagerBasedRLEnv

            # Import generated environment config
            from env_cfg import *  # noqa: F403

            # Import data collection modules
            from src.data_collection.config import load_robot_config
            from src.data_collection.sim_robot_interface import SimRobotInterface
            from src.data_collection.sim_detector import SimDetector
            from src.data_collection.sim_camera import SimCamera, MultiCameraManager, inject_cameras_into_scene, SceneCameraManager
            from src.data_collection.sim_recorder import SimRecorder
            from src.data_collection.sim_skills import SimSkills, create_ik_solver
            from src.data_collection.skill_planner import SkillPlanner
            from src.data_collection.sim_judge import SimJudge

            MARKER_FILE = os.path.join("{Path(self.output_dir).resolve()}", "COLLECTION_SUCCESS_MARKER")
            USE_LLM_PLANNER = {self.config.skill_retry_max > 0}
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

            def _execute_task_skills_fallback(skills, detector, goal, scene_state):
                \"\"\"Fallback: execute task-specific skills based on hardcoded goal parsing.

                For stacking: re-detects base object position before each place
                (previous picks may have displaced objects).
                \"\"\"
                conditions = []

                if "success_criteria" in goal:
                    sc = goal["success_criteria"]
                    if "conditions" in sc:
                        conditions = sc["conditions"]
                    elif "stacking_order" in sc:
                        order = sc["stacking_order"]
                        base_obj = order[0] if order else None
                        if base_obj and base_obj in scene_state:
                            height = 0.05
                            # Stack bottom-up: place each object on top of the previous
                            for i, obj in enumerate(order[1:], 1):
                                # Re-detect base position right before place
                                # (previous operations may have shifted objects)
                                try:
                                    prev_obj = order[i - 1]
                                    base_pos = detector.get_object_position(prev_obj)
                                    print(f"Re-detected '{{prev_obj}}' at {{base_pos}} for stacking")
                                except Exception:
                                    base_pos = scene_state[base_obj]["position"]
                                target = base_pos.copy()
                                target[2] += height
                                skills.execute_pick_and_place(
                                    obj, target, place_on_object=prev_obj,
                                )
                            return
                elif "conditions" in goal:
                    conditions = goal["conditions"]

                for cond in conditions:
                    cond_type = cond.get("type", "")
                    if cond_type in ("on_top_of", "stacked"):
                        obj = cond.get("object", cond.get("top"))
                        target_obj = cond.get("target", cond.get("bottom"))
                        if obj and target_obj and target_obj in scene_state:
                            target_pos = scene_state[target_obj]["position"].copy()
                            target_pos[2] += 0.05
                            skills.execute_pick_and_place(
                                obj, target_pos, place_on_object=target_obj,
                            )
                    elif cond_type == "at_position":
                        obj = cond.get("object", "")
                        pos = cond.get("position", [])
                        if obj and len(pos) == 3:
                            skills.execute_pick_and_place(obj, np.array(pos))
                    elif cond_type in ("lifted", "above"):
                        obj = cond.get("object", "")
                        height = cond.get("height", 0.1)
                        if obj and obj in scene_state:
                            target = scene_state[obj]["position"].copy()
                            target[2] += height
                            skills.execute_pick(obj)

            def _capture_top_image(cameras):
                \"\"\"Capture a single image for VLM judge (prefer top view for best scene overview).\"\"\"
                if cameras is not None:
                    try:
                        images = cameras.capture_all()
                        for name in ("top_cam", "top", "front_cam", "front"):
                            if name in images and images[name] is not None:
                                return images[name]
                        # Return first available
                        for img in images.values():
                            if img is not None:
                                return img
                    except Exception:
                        pass
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
                        cond_type = cond.get("type", "")
                        if cond_type in ("on_top_of", "stacked"):
                            total += 1
                            top_obj = cond.get("object", cond.get("top"))
                            bottom_obj = cond.get("target", cond.get("bottom"))
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
                            ok = xy_err < xy_threshold and abs(z_diff - height_diff) < z_threshold
                            if ok:
                                passed += 1
                            details.append(
                                f"{{top_obj}} on {{bottom_obj}}: xy_err={{xy_err:.3f}}m z_diff={{z_diff:.3f}}m {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type in ("lifted", "above"):
                            total += 1
                            obj = cond.get("object", "")
                            min_height = cond.get("height", 0.1)
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

                # --- Patch actuator PD gains + disable gravity (HIGH_PD config) ---
                # Default stiffness=80/damping=4 is tuned for RL (small action deltas).
                # For IK-based absolute control, need HIGH_PD: stiffness=400, damping=80,
                # with gravity disabled (matching FRANKA_PANDA_HIGH_PD_CFG).
                if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'actuators'):
                    for act_name, actuator in env_cfg.scene.robot.actuators.items():
                        if 'hand' not in act_name and 'finger' not in act_name and 'gripper' not in act_name:
                            actuator.stiffness = 400.0
                            actuator.damping = 80.0
                            # Override low effort limits (e.g. Franka forearm: 12 Nm)
                            # that prevent HIGH_PD torques from converging.
                            # For ImplicitActuatorCfg, both effort_limit and effort_limit_sim
                            # must be set to the same value (or only effort_limit_sim).
                            actuator.effort_limit = 200.0
                            actuator.effort_limit_sim = 200.0
                    print("Patched actuators: stiffness=400, damping=80, effort_limit=200 (HIGH_PD)")
                if hasattr(env_cfg.scene, 'robot') and hasattr(env_cfg.scene.robot, 'spawn'):
                    if hasattr(env_cfg.scene.robot.spawn, 'rigid_props') and env_cfg.scene.robot.spawn.rigid_props is not None:
                        env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True
                        print("Patched robot: disable_gravity=True (matching HIGH_PD_CFG)")

                # --- Inject cameras into scene config BEFORE env construction ---
                # This ensures cameras get the PLAY event and initialize properly.
                # Top view: near-top-down, robot at bottom of frame, workspace in center.
                # Wrist view: body-mounted on end-effector for close-up manipulation view.
                camera_attr_names = inject_cameras_into_scene(env_cfg, robot_cfg, camera_names=["top", "wrist"])
                if camera_attr_names:
                    print(f"Injected cameras into scene: {{camera_attr_names}}")

                env = ManagerBasedRLEnv(cfg=env_cfg)

                # Initialize components
                robot_interface = SimRobotInterface(env, robot_cfg, env_idx=0)
                detector = SimDetector(env, task_doc, env_idx=0)

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

                # Create IK solver (Pinocchio-only, URDF from robot profile)
                ik_solver = create_ik_solver(robot_cfg)

                # Create skills (with multi-camera for image recording)
                skills = SimSkills(
                    robot_interface=robot_interface,
                    robot_cfg=robot_cfg,
                    detector=detector,
                    recorder=recorder,
                    camera=camera,
                    cameras=cameras,
                    ik_solver=ik_solver,
                )

                # Initialize LLM-based skill planner if enabled
                skill_planner = None
                if USE_LLM_PLANNER:
                    try:
                        from src.common.llm_client import AzureOpenAIClient
                        llm_client = AzureOpenAIClient()
                        skill_planner = SkillPlanner(llm_client, robot_cfg)
                        print("SkillPlanner initialized (LLM-based skill planning enabled)")
                    except Exception as e:
                        print(f"SkillPlanner init failed ({{e}}), using fallback skill logic")

                # Initialize VLM judge if enabled
                sim_judge = None
                if USE_VLM_JUDGE:
                    try:
                        sim_judge = SimJudge(model="gpt-5")
                        if sim_judge.is_available:
                            print("SimJudge initialized (VLM-based episode judging enabled)")
                        else:
                            print("SimJudge: VLM not available (missing API keys), skipping")
                            sim_judge = None
                    except Exception as e:
                        print(f"SimJudge init failed ({{e}}), skipping VLM judging")

                results = {{
                    "episodes": [],
                    "total_episodes": 0,
                    "successful_episodes": 0,
                }}

                # Success-based episode loop
                target_success = cfg.get("target_successful_episodes", cfg["max_episodes"])
                max_attempts = cfg.get("max_total_attempts", target_success * 5)
                episode_idx = 0
                successful_count = 0

                try:
                    while successful_count < target_success and episode_idx < max_attempts:
                        print(f"\\n=== Episode {{episode_idx + 1}} (success: {{successful_count}}/{{target_success}}, max: {{max_attempts}}) ===")

                        # Reset environment
                        obs, info = env.reset()
                        robot_interface.reset_termination_flags()

                        # Render warmup: step the environment a few times so camera
                        # buffers are populated (first frames after reset can be black).
                        for _ in range(10):
                            env.step(torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device))

                        # Start recording
                        task_desc = task_doc.get("task", {{}}).get("description", "")
                        recorder.start_episode(task_desc, episode_idx)

                        # Get scene state
                        scene_state = detector.get_all_objects()
                        print(f"Scene objects: {{list(scene_state.keys())}}")

                        # Capture initial image for VLM judge
                        initial_image = None
                        if sim_judge is not None:
                            initial_image = _capture_top_image(cameras)

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

                        # Track executed skills description for judge
                        executed_skills_desc = ""

                        # Execute skills
                        try:
                            skills.move_to_ready(duration=1.0)

                            if skill_planner is not None:
                                # LLM-based dynamic skill planning
                                skill_sequence = skill_planner.plan_skills(
                                    task_doc, scene_state
                                )
                                print(f"Planned {{len(skill_sequence)}} skills via LLM")
                                all_ok, errors = skill_planner.execute_skill_sequence(
                                    skills, skill_sequence
                                )
                                executed_skills_desc = str(skill_sequence)
                                if errors:
                                    print(f"Skill verification errors: {{errors}}")
                                else:
                                    print("All skills passed verification")
                            else:
                                # Fallback: hardcoded goal-based skills
                                goal = task_doc.get("goal", {{}})
                                _execute_task_skills_fallback(
                                    skills, detector, goal, scene_state
                                )
                                executed_skills_desc = "fallback goal-based execution"

                            # Capture env termination + goal state BEFORE move_to_ready
                            # (move_to_ready can knock objects; env auto-resets on success)
                            env_terminated_during_skills = robot_interface.env_terminated
                            env_truncated_during_skills = robot_interface.env_truncated

                            # Verify goal conditions geometrically BEFORE moving away
                            goal = task_doc.get("goal", {{}})
                            pre_retreat_goal_ok = False
                            pre_retreat_goal_details = ""
                            if not env_terminated_during_skills:
                                # Only check if env hasn't auto-reset yet
                                pre_retreat_goal_ok, pre_retreat_goal_details = _verify_goal_conditions(detector, goal)

                            # Capture final image BEFORE retreat (for VLM judge)
                            final_image = None
                            if sim_judge is not None:
                                final_image = _capture_top_image(cameras)

                            skills.move_to_ready(duration=1.5, safe_retreat=True)
                            execution_ok = True
                        except Exception as e:
                            print(f"Skill execution error: {{e}}")
                            traceback.print_exc()
                            execution_ok = False
                            env_terminated_during_skills = robot_interface.env_terminated
                            env_truncated_during_skills = robot_interface.env_truncated
                            pre_retreat_goal_ok = False
                            pre_retreat_goal_details = "execution error"
                            final_image = None

                        # Determine episode success
                        # Priority: VLM judge > goal verification > env termination
                        if sim_judge is not None and execution_ok and final_image is not None and initial_image is not None:
                            verdict = sim_judge.judge_episode(
                                task_description=task_desc,
                                initial_image=initial_image,
                                final_image=final_image,
                                object_positions=scene_state,
                                executed_skills=executed_skills_desc,
                            )
                            success = verdict["success"]
                            print(f"VLM Judge: {{verdict['prediction']}} — {{verdict['reasoning'][:80]}}")
                        elif execution_ok:
                            # Check: env termination (success condition fired during skills)
                            env_success = env_terminated_during_skills and not env_truncated_during_skills
                            if env_success:
                                print("Success: environment termination condition met during skills")
                                success = True
                            elif pre_retreat_goal_ok:
                                print(f"Success (goal verified): {{pre_retreat_goal_details}}")
                                success = True
                            else:
                                if pre_retreat_goal_details:
                                    print(f"Goal verification: {{pre_retreat_goal_details}}")
                                print("Episode FAILED: goal conditions not met")
                                success = False
                        else:
                            success = False

                        # End episode (discard failures when targeting success count)
                        discard_failed = (cfg.get("target_successful_episodes", 0) > 0
                                         and cfg.get("target_successful_episodes", 0) < cfg.get("max_total_attempts", 999)
                                         and not success)
                        recorder.end_episode(success=success, discard=discard_failed)

                        episode_result = {{
                            "episode": episode_idx,
                            "success": success,
                            "steps": recorder.current_episode_steps,
                            "discarded": discard_failed,
                        }}
                        if sim_judge is not None and execution_ok and initial_image is not None:
                            episode_result["judge_prediction"] = verdict.get("prediction", "N/A")
                            episode_result["judge_reasoning"] = verdict.get("reasoning", "")
                        results["episodes"].append(episode_result)
                        results["total_episodes"] += 1
                        if success:
                            results["successful_episodes"] += 1
                            successful_count += 1

                        print(f"Episode {{episode_idx + 1}} complete: {{'SUCCESS' if success else 'FAILED'}}{{'(discarded)' if discard_failed else ''}}")
                        episode_idx += 1

                    print(f"\\nCompleted: {{successful_count}}/{{target_success}} successful episodes in {{episode_idx}} attempts")

                except KeyboardInterrupt:
                    print("\\nCollection interrupted by user")
                except Exception as e:
                    print(f"Pipeline error: {{e}}")
                    traceback.print_exc()
                finally:
                    # Finalize dataset
                    dataset_path = recorder.finalize()
                    results["dataset_path"] = dataset_path

                    # Save results
                    results_path = os.path.join(cfg["output_dir"], "collection_results.json")
                    with open(results_path, "w") as f:
                        json.dump(results, f, indent=2)

                    # Write success marker
                    with open(MARKER_FILE, "w") as f:
                        f.write("SUCCESS")

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

            # Check success marker
            marker_path = self.output_dir / "COLLECTION_SUCCESS_MARKER"
            success = marker_path.exists()

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
