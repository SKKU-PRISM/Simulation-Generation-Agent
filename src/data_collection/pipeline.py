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
import signal
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
    CameraConfig,
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

TEXTURE_PRESERVE_DIRECT_USD = {
    "sm_mug_a2.usd": {
        "asset_family": "mug",
        "loader_policy": "preserve_visual_usd_with_injected_physics",
        "physics_strategy": "child_mesh_injection",
        "physics_target_prim_basename": "SM_Mug_A2",
        "default_mass": 0.18,
        "notes": "Keep the original textured USD and inject rigid/collision properties instead of proxying.",
    },
    "007_tuna_fish_can.usd": {
        "asset_family": "tuna_can",
        "loader_policy": "preserve_visual_usd_with_injected_physics",
        "physics_strategy": "child_mesh_injection",
        "physics_target_prim_basename": "_07_tuna_fish_can",
        "default_mass": 0.12,
        "notes": "Keep the original YCB textured USD and add physics at spawn time instead of using a cylinder proxy.",
    },
}

TEXTURE_AUDIT_DIRECT_USD = {
    "004_sugar_box.usd": {
        "asset_family": "sugar_box",
        "loader_policy": "audit_axis_aligned_variant",
        "notes": "Compare Axis_Aligned_Physics visual fidelity against the canonical Axis_Aligned asset.",
    },
    "005_tomato_soup_can.usd": {
        "asset_family": "tomato_soup_can",
        "loader_policy": "audit_axis_aligned_variant",
        "notes": "Audit material bindings and asset-root resolution for the YCB can visual asset.",
    },
    "006_mustard_bottle.usd": {
        "asset_family": "mustard_bottle",
        "loader_policy": "audit_axis_aligned_variant",
        "notes": "Audit material bindings and asset-root resolution for the YCB bottle visual asset.",
    },
    "025_mug.usd": {
        "asset_family": "ycb_mug",
        "loader_policy": "audit_axis_aligned_variant",
        "notes": "Audit YCB mug material bindings when used directly from Axis_Aligned assets.",
    },
}


def classify_texture_asset_path(asset_path: str) -> dict[str, object] | None:
    """Return texture-sensitive asset handling metadata for known Isaac Sim props."""
    asset_path = str(asset_path or "").strip()
    if not asset_path:
        return None
    lowered = asset_path.lower()

    for suffix, meta in TEXTURE_PRESERVE_DIRECT_USD.items():
        if suffix in lowered:
            return {
                "asset_path": asset_path,
                "path_suffix": suffix,
                **meta,
            }

    for suffix, meta in TEXTURE_AUDIT_DIRECT_USD.items():
        if suffix in lowered:
            return {
                "asset_path": asset_path,
                "path_suffix": suffix,
                **meta,
            }

    return None


def should_preserve_textured_usd(asset_path: str) -> bool:
    record = classify_texture_asset_path(asset_path)
    return bool(record and record.get("loader_policy") == "preserve_visual_usd_with_injected_physics")


def build_task_texture_audit(task_doc: dict) -> list[dict[str, object]]:
    """Build a per-task audit record for texture-sensitive Isaac Sim assets."""
    task_name = str((task_doc or {}).get("task", {}).get("name", "unknown"))
    records: list[dict[str, object]] = []
    for asset in (task_doc or {}).get("assets", []):
        if not isinstance(asset, dict):
            continue
        record = classify_texture_asset_path(asset.get("asset_path", ""))
        if record is None:
            continue
        records.append(
            {
                "task_name": task_name,
                "asset_name": asset.get("name", ""),
                "asset_type": asset.get("type", ""),
                "source": asset.get("source", ""),
                **record,
            }
        )
    return records


def apply_task_top_camera_override(
    robot_cfg: RobotSimConfig,
    task_doc: dict | None,
) -> RobotSimConfig:
    """Apply a task YAML top-view camera override to Franka data collection.

    Franka task YAML documents already carry a representative ``camera`` block.
    Data collection historically ignored it and always used the robot-profile
    top camera, which made the recorded top view too high and inconsistent with
    the task scene. For Franka, reuse the task camera as the top-view source.
    """
    if robot_cfg.name != "franka" or not isinstance(task_doc, dict):
        return robot_cfg

    task_camera = task_doc.get("camera", {}) or {}
    position = task_camera.get("position")
    target = task_camera.get("target")

    if not (
        isinstance(position, list)
        and len(position) == 3
        and isinstance(target, list)
        and len(target) == 3
    ):
        return robot_cfg

    top_cfg = robot_cfg.cameras.get("top")
    if top_cfg is None:
        top_cfg = CameraConfig(
            name="top",
            cam_type="fixed",
            up_vector=[1.0, 0.0, 0.0],
            resolution=(640, 480),
        )
        robot_cfg.cameras["top"] = top_cfg

    top_cfg.cam_type = "fixed"
    top_cfg.position = [float(v) for v in position]
    top_cfg.target = [float(v) for v in target]
    if not top_cfg.up_vector:
        top_cfg.up_vector = [1.0, 0.0, 0.0]

    logger.info(
        "Applied Franka task top-camera override: position=%s target=%s",
        top_cfg.position,
        top_cfg.target,
    )
    return robot_cfg


def select_success_video_camera_name(
    robot_name: str,
    task_doc: dict | None,
) -> str:
    """Select the representative success-video camera for a task.

    Most tasks use the front view. Franka cabinet videos are easier to inspect
    from the normalized top view because the articulated handle and top surface
    stay visible during the whole sequence.
    """
    if robot_name != "franka" or not isinstance(task_doc, dict):
        return "front"
    task_name = str(task_doc.get("task", {}).get("name", "")).strip().lower()
    if task_name == "frankacabinet":
        return "top"
    return "front"


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
        auto_convert_to_lerobot: bool = True,
    ):
        self.yaml_path = Path(yaml_path).resolve()
        self.config = config or load_pipeline_config()
        self.env_dir = Path(env_dir) if env_dir else None
        self.auto_convert_to_lerobot = bool(auto_convert_to_lerobot)
        self._last_execution_timed_out = False

        # Load task document
        with open(self.yaml_path) as f:
            self.task_doc = yaml.safe_load(f)

        # Detect robot
        self.robot_name = self._detect_robot()
        self.robot_cfg = load_robot_config(self.robot_name)
        self.robot_cfg = apply_task_top_camera_override(self.robot_cfg, self.task_doc)

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
        if self._last_execution_timed_out:
            results.setdefault("failure_category", "timed_out")
            results.setdefault("failure_phase", "execution_timeout")
            results.setdefault("pipeline_error", "execution timeout")

        # Step 5: Post-process (convert to LeRobot if raw data exists)
        raw_data_dir = self.output_dir / "raw_dataset"
        lerobot_dataset = None
        if raw_data_dir.exists() and self.auto_convert_to_lerobot:
            lerobot_dataset = self._convert_to_lerobot(raw_data_dir)

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
            "env_dir": str(self.env_dir) if self.env_dir is not None else None,
            "raw_dataset": str(raw_data_dir) if raw_data_dir.exists() else None,
            "lerobot_dataset": str(lerobot_dataset) if lerobot_dataset else None,
            "front_video_generated": bool(results.get("front_video_generated", False)),
            "front_video_path": results.get("front_video_path"),
            "front_video_camera_name": results.get("front_video_camera_name"),
            "front_video_episode": results.get("front_video_episode"),
            "front_video_success_type": results.get("front_video_success_type"),
            "results": results,
            "robot": self.robot_name,
            "task": self.task_doc.get("task", {}).get("name", "unknown"),
            "failure_category": (
                results.get("failure_category")
                or ("timed_out" if self._last_execution_timed_out else None)
            ),
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
            "front_video_fps": self.config.effective_front_video_fps,
            "front_video_camera_name": select_success_video_camera_name(self.robot_name, self.task_doc),
            "headless": self.config.env_headless,
            "num_envs": self.config.env_num_envs,
            "use_vlm_judge": self.config.use_vlm_judge,
            "llm_model": self.config.llm_model,
            "vlm_model": self.config.vlm_model,
            "skill_retry_max": self.config.skill_retry_max,
            "target_successful_episodes": target_success,
            "max_total_attempts": max_attempts,
            "dataset_cameras": list(self.config.dataset_cameras),
            "judge_cameras": list(self.config.judge_cameras),
            "discard_failed_episodes": bool(self.config.discard_failed_episodes),
            "keep_failed_raw_dataset": bool(self.config.keep_failed_raw_dataset),
            "ik_debug": bool(self.config.ik_debug),
        }
        config_json_path = self.output_dir / "pipeline_config.json"
        with open(config_json_path, "w") as f:
            json.dump(config_data, f, indent=2)

        texture_audit_path = self.output_dir / "texture_asset_audit.json"
        with open(texture_audit_path, "w") as f:
            json.dump(build_task_texture_audit(self.task_doc), f, indent=2)

        # Generate the runner script
        # NOTE: AppLauncher MUST be initialized before any physics imports
        headless_val = self.config.env_headless
        script = textwrap.dedent(f'''\
            """Auto-generated data collection runner for IsaacLab."""
            import argparse
            from dataclasses import MISSING
            import json
            import sys
            import os
            import re
            import shutil
            import time
            import traceback
            from datetime import datetime

            # === AppLauncher MUST be first (before any physics imports) ===
            from isaaclab.app import AppLauncher

            parser = argparse.ArgumentParser()
            AppLauncher.add_app_launcher_args(parser)
            args = parser.parse_args([])
            args.headless = {headless_val}
            args.num_envs = {self.config.env_num_envs}
            args.enable_cameras = True  # Always enable cameras (EGL in headless, X11 in GUI)
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
            from isaaclab.sensors import FrameTransformerCfg
            from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
            from src.data_collection.pipeline import apply_task_top_camera_override

            MARKER_FILE = os.path.join("{Path(self.output_dir).resolve()}", "COLLECTION_COMPLETE_MARKER")
            USE_VLM_JUDGE = {self.config.use_vlm_judge}
            STARTUP_DIAGNOSTICS_PATH = os.path.join("{Path(self.output_dir).resolve()}", "startup_diagnostics.json")
            PHASE_TRACE_PATH = os.path.join("{Path(self.output_dir).resolve()}", "phase_trace.json")
            HEARTBEAT_PATH = os.path.join("{Path(self.output_dir).resolve()}", "last_heartbeat.json")
            STDERR_TAIL_PATH = os.path.join("{Path(self.output_dir).resolve()}", "stderr_tail.txt")

            def _early_write_json(path, payload):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)

            def _early_write_text(path, text):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)

            def _early_record_phase(phase, status="ok", **details):
                payload = []
                if os.path.exists(PHASE_TRACE_PATH):
                    try:
                        with open(PHASE_TRACE_PATH, "r", encoding="utf-8") as f:
                            loaded = json.load(f)
                        if isinstance(loaded, list):
                            payload = loaded
                    except Exception:
                        payload = []
                event = {{
                    "timestamp": datetime.now().isoformat(),
                    "phase": phase,
                    "status": status,
                }}
                if details:
                    event["details"] = details
                payload.append(event)
                _early_write_json(PHASE_TRACE_PATH, payload)
                _early_write_json(HEARTBEAT_PATH, event)
                return event

            _early_record_phase("script_boot", "ok", env_dir="{Path(self.env_dir).resolve()}")

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

                def _rewrite_initial_state_scale_arg(content):
                    lines = content.splitlines()
                    patched_lines = []
                    in_initial_state = False
                    paren_depth = 0
                    replaced = False

                    for line in lines:
                        if "InitialStateCfg(" in line:
                            in_initial_state = True
                            paren_depth = line.count("(") - line.count(")")
                        elif in_initial_state:
                            paren_depth += line.count("(") - line.count(")")

                        if in_initial_state and "scale=" in line:
                            new_line = re.sub(
                                r",\s*scale\s*=\s*(\[[^\]]*\]|\([^\)]*\))",
                                "",
                                line,
                            )
                            if new_line != line:
                                line = new_line
                                replaced = True

                        patched_lines.append(line)

                        if in_initial_state and paren_depth <= 0:
                            in_initial_state = False
                            paren_depth = 0

                    return "\\n".join(patched_lines), replaced

                def _rewrite_articulation_joints_arg(content):
                    lines = content.splitlines()
                    patched_lines = []
                    replaced = False

                    articulation_pattern = re.compile(
                        r'^(?P<indent>\s*)(?P<target>self\.scene\.[A-Za-z_][A-Za-z0-9_]*)\s*=\s*ArticulationCfg\('
                    )

                    line_idx = 0
                    while line_idx < len(lines):
                        line = lines[line_idx]
                        match = articulation_pattern.match(line)
                        if not match:
                            patched_lines.append(line)
                            line_idx += 1
                            continue

                        articulation_target = match.group("target")
                        articulation_indent = match.group("indent")
                        block_lines = [line]
                        paren_depth = line.count("(") - line.count(")")
                        line_idx += 1
                        while line_idx < len(lines):
                            block_line = lines[line_idx]
                            block_lines.append(block_line)
                            paren_depth += block_line.count("(") - block_line.count(")")
                            line_idx += 1
                            if paren_depth <= 0:
                                break

                        rewritten_block = []
                        articulation_joints_lines = []
                        block_idx = 0
                        while block_idx < len(block_lines):
                            block_line = block_lines[block_idx]
                            block_line = re.sub(
                                r'init_state\s*=\s*AssetBaseCfg\.InitialStateCfg\(',
                                'init_state=ArticulationCfg.InitialStateCfg(',
                                block_line,
                            )
                            joints_match = re.match(
                                r'^(?P<indent>\s*)joints\s*=\s*(?P<rhs>.+)$',
                                block_line,
                            )
                            if joints_match:
                                rhs = joints_match.group("rhs")
                                articulation_joints_lines = [rhs]
                                joints_brace_depth = rhs.count("{") - rhs.count("}")
                                block_idx += 1
                                while block_idx < len(block_lines) and joints_brace_depth > 0:
                                    joint_line = block_lines[block_idx]
                                    articulation_joints_lines.append(joint_line)
                                    joints_brace_depth += joint_line.count("{") - joint_line.count("}")
                                    block_idx += 1
                                replaced = True
                                continue

                            rewritten_block.append(block_line)
                            block_idx += 1

                        patched_lines.extend(rewritten_block)
                        if articulation_joints_lines:
                            normalized = articulation_joints_lines[:]
                            for idx in range(len(normalized) - 1, -1, -1):
                                stripped = normalized[idx].rstrip()
                                if not stripped:
                                    continue
                                if stripped.endswith(","):
                                    normalized[idx] = stripped[:-1]
                                else:
                                    normalized[idx] = stripped
                                break
                            first_line = normalized[0].lstrip()
                            patched_lines.append(
                                f"{{articulation_indent}}{{articulation_target}}.init_state.joint_pos = {{first_line}}"
                            )
                            patched_lines.extend(normalized[1:])

                    return "\\n".join(patched_lines), replaced

                def _rewrite_scene_entity_body_name_arg(content):
                    pattern = re.compile(
                        r'SceneEntityCfg\\(\\s*(?P<entity>"[^"]+"|\\\'[^\\\']+\\\')\\s*,\\s*body_name\\s*=\\s*(?P<name>"[^"]+"|\\\'[^\\\']+\\\')(?P<suffix>\\s*,[^\\)]*)?\\)'
                    )
                    replaced = False

                    def _replace(match):
                        nonlocal replaced
                        replaced = True
                        entity = match.group("entity")
                        name = match.group("name")
                        suffix = match.group("suffix") or ""
                        return "SceneEntityCfg(" + entity + ", body_names=[" + name + "]" + suffix + ")"

                    return pattern.sub(_replace, content), replaced

                def _rewrite_cuboid_mass_arg(content):
                    lines = content.splitlines()
                    patched_lines = []
                    in_cuboid = False
                    cuboid_depth = 0
                    replaced = False
                    for line in lines:
                        stripped = line.strip()
                        if "sim_utils.CuboidCfg(" in line:
                            in_cuboid = True
                            cuboid_depth = line.count("(") - line.count(")")
                            patched_lines.append(line)
                            continue
                        if in_cuboid and re.match(r"mass\s*=\s*[^,]+,?\s*$", stripped):
                            indent = re.match(r"^(\s*)", line).group(1)
                            mass_expr = stripped.split("=", 1)[1].rstrip(",").strip()
                            patched_lines.append(
                                f"{{indent}}mass_props=sim_utils.MassPropertiesCfg(mass={{mass_expr}}),"
                            )
                            replaced = True
                        else:
                            patched_lines.append(line)
                        if in_cuboid:
                            cuboid_depth += line.count("(") - line.count(")")
                            if cuboid_depth <= 0:
                                in_cuboid = False
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
                patched_text = re.sub(
                    r'^(?P<indent>\\s*)(?P<name>[A-Za-z_][A-Za-z0-9_]*)\\s*=\\s*MISSING\\s*$',
                    r'\\g<indent>\\g<name>: object = MISSING',
                    patched_text,
                    flags=re.MULTILINE,
                )
                patched_text, patched_cuboid_mass = _rewrite_cuboid_mass_arg(patched_text)
                patched_text, patched_implicit_joint_names = _rewrite_implicit_actuator_joint_names(patched_text)
                patched_text, patched_initial_state_scale = _rewrite_initial_state_scale_arg(patched_text)
                patched_text, patched_articulation_joints = _rewrite_articulation_joints_arg(patched_text)
                patched_text, patched_scene_entity_body_name = _rewrite_scene_entity_body_name_arg(patched_text)

                if patched_text != env_cfg_text:
                    with open(env_cfg_path, "w", encoding="utf-8") as f:
                        f.write(patched_text)
                    patch_notes = []
                    if "super().__post_init__()" in env_cfg_text:
                        patch_notes.append("guarded super().__post_init__()")
                    if re.search(r'^\\s*[A-Za-z_][A-Za-z0-9_]*\\s*=\\s*MISSING\\s*$', env_cfg_text, flags=re.MULTILINE):
                        patch_notes.append("annotated bare MISSING fields")
                    if patched_cuboid_mass:
                        patch_notes.append("rewrote CuboidCfg mass -> mass_props")
                    if patched_implicit_joint_names:
                        patch_notes.append("rewrote ImplicitActuatorCfg joint_names -> joint_names_expr")
                    if patched_initial_state_scale:
                        patch_notes.append("dropped unsupported InitialStateCfg scale arg")
                    if patched_articulation_joints:
                        patch_notes.append("rewrote ArticulationCfg joints -> init_state.joint_pos")
                    if patched_scene_entity_body_name:
                        patch_notes.append("rewrote SceneEntityCfg body_name -> body_names")
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
            try:
                from env_cfg import *  # noqa: F403
                _early_record_phase("env_cfg_import", "ok")
            except Exception as exc:
                tb_text = traceback.format_exc()
                _early_record_phase("env_cfg_import", "error", error=str(exc), category="env_cfg_import_failed")
                _early_write_json(
                    STARTUP_DIAGNOSTICS_PATH,
                    {{
                        "failure_phase": "env_cfg_import",
                        "failure_category": "env_cfg_import_failed",
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": tb_text,
                        "last_phase": "env_cfg_import",
                        "updated_at": datetime.now().isoformat(),
                    }},
                )
                _early_write_text(STDERR_TAIL_PATH, tb_text)
                raise

            # Import data collection modules
            from src.data_collection.config import load_robot_config
            from src.data_collection.pipeline import classify_texture_asset_path, should_preserve_textured_usd
            from src.data_collection.sim_robot_interface import SimRobotInterface
            from src.data_collection.sim_detector import SimDetector
            from src.data_collection.sim_camera import SimCamera, MultiCameraManager, inject_cameras_into_scene, SceneCameraManager
            from src.data_collection.sim_recorder import SimRecorder, RecorderStorageError
            from src.data_collection.sim_skills import SimSkills, create_ik_solver
            from src.data_collection.multiview_grounding import MultiViewTargetGrounder
            from src.data_collection.textured_usd_spawn import spawn_textured_usd_with_child_physics
            from src.data_collection.cap_artifacts import (
                make_episode_run_dir,
                save_code_generation_artifacts,
                save_execution_context,
                save_judge_images,
                write_json_artifact,
                write_text_artifact,
            )
            from src.data_collection.cap_generator import SimCaPGenerator, is_supported_tabletop_task, select_cap_profile
            from src.data_collection.cap_runtime import CaPRuntimeContext
            from src.data_collection.cap_runtime.skills.base import PolicyFallbackBlockedError
            from src.data_collection.sim_judge import SimJudge

            def _load_existing_json(path, default):
                if not os.path.exists(path):
                    return default
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    return payload
                except Exception:
                    return default

            _PHASE_TRACE = _load_existing_json(PHASE_TRACE_PATH, [])
            _STARTUP_DIAGNOSTICS = _load_existing_json(STARTUP_DIAGNOSTICS_PATH, {{}})
            _LAST_PHASE = _PHASE_TRACE[-1]["phase"] if _PHASE_TRACE else "script_boot"

            def _write_phase_trace():
                write_json_artifact(PHASE_TRACE_PATH, _PHASE_TRACE)

            def _record_phase(phase, status="ok", **details):
                global _LAST_PHASE
                event = {{
                    "timestamp": datetime.now().isoformat(),
                    "phase": phase,
                    "status": status,
                }}
                if details:
                    event["details"] = details
                _PHASE_TRACE.append(event)
                _LAST_PHASE = phase
                _write_phase_trace()
                write_json_artifact(HEARTBEAT_PATH, event)
                return event

            def _update_startup_diagnostics(**updates):
                _STARTUP_DIAGNOSTICS.update(updates)
                _STARTUP_DIAGNOSTICS["last_phase"] = _LAST_PHASE
                _STARTUP_DIAGNOSTICS["updated_at"] = datetime.now().isoformat()
                write_json_artifact(STARTUP_DIAGNOSTICS_PATH, _STARTUP_DIAGNOSTICS)

            def _write_exception_artifacts(exc, failure_phase, failure_category):
                tb_text = traceback.format_exc()
                payload = {{
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "failure_phase": failure_phase,
                    "failure_category": failure_category,
                    "traceback": tb_text,
                }}
                _update_startup_diagnostics(**payload)
                write_text_artifact(STDERR_TAIL_PATH, tb_text)
                _record_phase(failure_phase, "error", error=str(exc), category=failure_category)
                return payload

            def _classify_pipeline_failure(exc):
                exc_text = f"{{type(exc).__name__}}: {{exc}}"
                lowered = exc_text.lower()
                if "unsupported by current cap tabletop profile" in lowered:
                    return "unsupported_profile"
                if "xformprimview" in lowered and "has no attribute 'data'" in lowered:
                    return "env_observation_crash"
                if "vlm judge unavailable" in lowered:
                    return "judge_unavailable"
                if "cap generator unavailable" in lowered or "model did not return executable cap code" in lowered:
                    return "codegen_failure"
                return "pipeline_error"

            def _find_env_cfg_class():
                \"\"\"Find the EnvCfg class from module globals (imported via env_cfg).\"\"\"
                for name, obj in globals().items():
                    if (isinstance(obj, type)
                            and name.endswith("EnvCfg")
                            and name != "ManagerBasedRLEnvCfg"
                            and issubclass(obj, ManagerBasedRLEnvCfg)):
                        return obj
                return None

            def _disable_debug_visualization(obj, path="env_cfg", seen=None, patched=None):
                \"\"\"Recursively disable IsaacLab debug visuals in generated configs.\"\"\"
                if seen is None:
                    seen = set()
                if patched is None:
                    patched = []
                if obj is None:
                    return patched
                obj_id = id(obj)
                if obj_id in seen:
                    return patched
                seen.add(obj_id)

                if isinstance(obj, dict):
                    for key, value in obj.items():
                        _disable_debug_visualization(value, f"{{path}}.{{key}}", seen, patched)
                    return patched
                if isinstance(obj, (list, tuple, set)):
                    for idx, value in enumerate(obj):
                        _disable_debug_visualization(value, f"{{path}}[{{idx}}]", seen, patched)
                    return patched
                if isinstance(obj, (str, bytes, int, float, bool)):
                    return patched

                if hasattr(obj, "debug_vis"):
                    try:
                        if getattr(obj, "debug_vis") is not False:
                            setattr(obj, "debug_vis", False)
                            patched.append(path)
                    except Exception:
                        pass

                try:
                    items = list(vars(obj).items())
                except Exception:
                    items = []
                for attr_name, attr_value in items:
                    if attr_name.startswith("_"):
                        continue
                    _disable_debug_visualization(attr_value, f"{{path}}.{{attr_name}}", seen, patched)
                return patched

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
                insertion_target_names = set()
                if isinstance(task_doc, dict):
                    insertion_target_names = {{
                        target
                        for target in (
                            cond.get("target")
                            for cond in (task_doc.get("goal", {{}}).get("conditions", []) or [])
                            if str(cond.get("type", cond.get("relation", "")) or "").lower()
                            in {{"inserted_into", "height_below"}}
                        )
                        if isinstance(target, str)
                    }}

                def _uses_xform_scene_entity(asset_name):
                    if not isinstance(task_doc, dict):
                        return False
                    if asset_name in insertion_target_names:
                        return True
                    for asset in task_doc.get("assets", []):
                        if asset.get("name") != asset_name:
                            continue
                        physics = asset.get("physics", {{}}) or {{}}
                        if asset.get("type") == "static":
                            return True
                        if asset.get("type") in {{"rigid", "primitive"}} and not bool(physics.get("rigid_body", False)):
                            return True
                    return False

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
                            asset_cfg = params.get("asset_cfg") or params.get("robot_cfg")
                            asset_name = getattr(asset_cfg, "name", None)
                            if (
                                isinstance(asset_name, str)
                                and func_name in {{"root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "root_state_w"}}
                                and _uses_xform_scene_entity(asset_name)
                            ):
                                setattr(group_cfg, term_name, None)
                                patched_terms.append(
                                    f"{{group_name}}/{{term_name}} (disabled xform root observation for {{asset_name}})"
                                )
                                continue

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
                            if asset_cfg is None:
                                asset_cfg = SceneEntityCfg("robot")
                                params["asset_cfg"] = asset_cfg
                                patched_terms.append(
                                    f"{{group_name}}/{{term_name}} (default asset_cfg=robot)"
                                )
                            if asset_cfg is not None and hasattr(asset_cfg, "body_names"):
                                asset_cfg.body_names = body_name
                                patched_terms.append(f"{{group_name}}/{{term_name}} (body_name->body_names)")
                            else:
                                # Put the parameter back if we cannot patch it safely.
                                params["body_name"] = body_name

                patched_scene_assets = []
                scene = getattr(env_cfg, "scene", None)
                if scene is not None and isinstance(task_doc, dict):
                    for attr_name, attr_value in list(vars(scene).items()):
                        if attr_name.startswith("_") or attr_value is None:
                            continue
                        attr_type_name = attr_value.__class__.__name__
                        if attr_name.endswith(
                            (
                                "_rigid_props",
                                "_collision_props",
                                "_mass_props",
                                "_articulation_props",
                            )
                        ) or attr_type_name in {
                            "RigidBodyPropertiesCfg",
                            "CollisionPropertiesCfg",
                            "MassPropertiesCfg",
                            "ArticulationRootPropertiesCfg",
                        }:
                            setattr(scene, attr_name, None)
                            patched_scene_assets.append(
                                f"{{attr_name}}: removed helper config ({{attr_type_name}}) from scene"
                            )

                    if hasattr(scene, "replicate_physics") and getattr(scene, "replicate_physics", True):
                        scene.replicate_physics = False
                        patched_scene_assets.append("scene.replicate_physics=False")

                    robot_asset = next(
                        (
                            asset
                            for asset in task_doc.get("assets", [])
                            if asset.get("type") == "articulation" and asset.get("robot_type")
                        ),
                        None,
                    )
                    robot_type = str((robot_asset or {{}}).get("robot_type", "")).lower()
                    insertion_target_names = set(insertion_target_names)
                    insertion_subject_names = {{
                        subject
                        for subject in (
                            cond.get("subject", cond.get("object"))
                            for cond in (task_doc.get("goal", {{}}).get("conditions", []) or [])
                            if str(cond.get("type", cond.get("relation", "")) or "").lower()
                            in {{"inserted_into", "height_below"}}
                        )
                        if isinstance(subject, str)
                    }}
                    unstable_factory_proxy_assets = {{
                        "gear_small": {{
                            "shape": "cylinder",
                            "radius": 0.030,
                            "height": 0.016,
                            "mass": 0.05,
                            "color": (0.55, 0.57, 0.60),
                        }},
                        "gear_medium": {{
                            "shape": "cylinder",
                            "radius": 0.040,
                            "height": 0.020,
                            "mass": 0.07,
                            "color": (0.42, 0.44, 0.47),
                        }},
                        "m16_nut": {{
                            "shape": "cylinder",
                            "radius": 0.014,
                            "height": 0.010,
                            "mass": 0.03,
                            "color": (0.32, 0.62, 0.34),
                        }},
                    }}
                    unstable_factory_asset_tokens = {{
                        "factory/factory_peg_8mm.usd": {{
                            "shape": "cuboid",
                            "size": (0.050, 0.008, 0.008),
                            "mass": 0.019,
                            "color": (0.77, 0.67, 0.22),
                            "disable_gravity": True,
                        }},
                    }}

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

                    for asset in task_doc.get("assets", []):
                        asset_name = asset.get("name")
                        if not asset_name or not hasattr(scene, asset_name):
                            continue
                        scene_cfg = getattr(scene, asset_name)
                        if scene_cfg is None:
                            continue

                        spawn_cfg = getattr(scene_cfg, "spawn", None)
                        usd_path = str(getattr(spawn_cfg, "usd_path", "") or "")
                        asset_path_lower = str(asset.get("asset_path", "")).lower()
                        init_state = getattr(scene_cfg, "init_state", None)
                        pos = list(getattr(init_state, "pos", asset.get("position", [0.0, 0.0, 0.0])))
                        rot = list(getattr(init_state, "rot", asset.get("rotation", [1.0, 0.0, 0.0, 0.0])))

                        if "standard_camera.usd" in usd_path:
                            setattr(scene, asset_name, None)
                            patched_scene_assets.append(
                                f"{{asset_name}}: disabled missing remote camera asset"
                            )
                            continue

                        if (
                            asset_name in insertion_target_names
                            and scene_cfg.__class__.__name__ == "RigidObjectCfg"
                        ):
                            if spawn_cfg is not None:
                                if hasattr(spawn_cfg, "rigid_props"):
                                    spawn_cfg.rigid_props = None
                                if hasattr(spawn_cfg, "mass_props"):
                                    spawn_cfg.mass_props = None
                                articulation_props = getattr(spawn_cfg, "articulation_props", None)
                                if articulation_props is None:
                                    try:
                                        articulation_props = ArticulationRootPropertiesCfg(
                                            articulation_enabled=False
                                        )
                                        spawn_cfg.articulation_props = articulation_props
                                    except Exception:
                                        articulation_props = None
                                if articulation_props is not None and hasattr(articulation_props, "articulation_enabled"):
                                    articulation_props.articulation_enabled = False
                            setattr(
                                scene,
                                asset_name,
                                AssetBaseCfg(
                                    prim_path=str(
                                        getattr(
                                            scene_cfg,
                                            "prim_path",
                                            "{{ENV_REGEX_NS}}/" + str(asset_name),
                                        )
                                    ),
                                    init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
                                    spawn=spawn_cfg,
                                ),
                            )
                            patched_scene_assets.append(
                                f"{{asset_name}}: demoted insertion target rigid object to static AssetBaseCfg"
                            )
                            continue

                        if scene_cfg.__class__.__name__ != "RigidObjectCfg":
                            continue

                        rigid_props = sim_utils.RigidBodyPropertiesCfg(
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                            max_angular_velocity=1000.0,
                            max_linear_velocity=1000.0,
                            max_depenetration_velocity=5.0,
                            disable_gravity=False,
                        )
                        collision_props = sim_utils.CollisionPropertiesCfg(
                            collision_enabled=True,
                            contact_offset=0.005,
                            rest_offset=0.0,
                        )
                        if asset_name in insertion_subject_names and spawn_cfg is not None:
                            articulation_props = getattr(spawn_cfg, "articulation_props", None)
                            if articulation_props is None:
                                try:
                                    articulation_props = ArticulationRootPropertiesCfg(
                                        articulation_enabled=False
                                    )
                                    spawn_cfg.articulation_props = articulation_props
                                except Exception:
                                    articulation_props = None
                            if articulation_props is not None and hasattr(articulation_props, "articulation_enabled"):
                                articulation_props.articulation_enabled = False
                                patched_scene_assets.append(
                                    f"{{asset_name}}: disabled articulation root on insertion subject"
                                )

                        proxy_spec = unstable_factory_proxy_assets.get(asset_name)
                        if proxy_spec is None:
                            for asset_token, token_proxy_spec in unstable_factory_asset_tokens.items():
                                if asset_token in asset_path_lower:
                                    proxy_spec = token_proxy_spec
                                    break
                        if proxy_spec is not None:
                            proxy_rigid_props = sim_utils.RigidBodyPropertiesCfg(
                                solver_position_iteration_count=16,
                                solver_velocity_iteration_count=1,
                                max_angular_velocity=1000.0,
                                max_linear_velocity=1000.0,
                                max_depenetration_velocity=5.0,
                                disable_gravity=bool(proxy_spec.get("disable_gravity", False)),
                            )
                            visual_material = sim_utils.PreviewSurfaceCfg(
                                diffuse_color=tuple(proxy_spec.get("color", (0.5, 0.5, 0.5)))
                            )
                            common_kwargs = dict(
                                rigid_props=proxy_rigid_props,
                                collision_props=collision_props,
                                mass_props=sim_utils.MassPropertiesCfg(mass=float(proxy_spec.get("mass", 0.05))),
                                visual_material=visual_material,
                            )
                            if proxy_spec.get("shape") == "cylinder":
                                proxy_spawn = sim_utils.CylinderCfg(
                                    radius=float(proxy_spec["radius"]),
                                    height=float(proxy_spec["height"]),
                                    **common_kwargs,
                                )
                            else:
                                proxy_spawn = sim_utils.CuboidCfg(
                                    size=tuple(proxy_spec.get("size", (0.03, 0.03, 0.02))),
                                    **common_kwargs,
                                )
                            setattr(
                                scene,
                                asset_name,
                                RigidObjectCfg(
                                    prim_path=str(
                                        getattr(
                                            scene_cfg,
                                            "prim_path",
                                            "{{ENV_REGEX_NS}}/" + str(asset_name),
                                        )
                                    ),
                                    init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
                                    spawn=proxy_spawn,
                                ),
                            )
                            patched_scene_assets.append(
                                f"{{asset_name}}: replaced unstable factory USD with stable {{proxy_spec.get('shape')}} proxy"
                            )
                            continue

                        if spawn_cfg is not None:
                            injected_spawn_defaults = []
                            if getattr(spawn_cfg, "rigid_props", None) is None:
                                spawn_cfg.rigid_props = rigid_props
                                injected_spawn_defaults.append("rigid_props")
                            if getattr(spawn_cfg, "collision_props", None) is None:
                                spawn_cfg.collision_props = collision_props
                                injected_spawn_defaults.append("collision_props")
                            if getattr(spawn_cfg, "mass_props", None) is None:
                                default_mass = 0.05
                                size = getattr(spawn_cfg, "size", None)
                                radius = getattr(spawn_cfg, "radius", None)
                                height = getattr(spawn_cfg, "height", None)
                                try:
                                    if isinstance(size, (list, tuple)) and len(size) == 3:
                                        volume = abs(float(size[0]) * float(size[1]) * float(size[2]))
                                        default_mass = max(min(volume * 200.0, 0.2), 0.01)
                                    elif radius is not None and height is not None:
                                        volume = math.pi * float(radius) * float(radius) * float(height)
                                        default_mass = max(min(volume * 250.0, 0.2), 0.01)
                                except Exception:
                                    default_mass = 0.05
                                spawn_cfg.mass_props = sim_utils.MassPropertiesCfg(mass=float(default_mass))
                                injected_spawn_defaults.append(f"mass_props={{default_mass:.4f}}")
                            if injected_spawn_defaults:
                                patched_scene_assets.append(
                                    f"{{asset_name}}: injected default spawn physics -> {{injected_spawn_defaults}}"
                                )

                        texture_record = classify_texture_asset_path(asset.get("asset_path", ""))
                        if should_preserve_textured_usd(asset.get("asset_path", "")) and hasattr(spawn_cfg, "usd_path"):
                            spawn_cfg.func = spawn_textured_usd_with_child_physics
                            spawn_cfg.rigid_props = rigid_props
                            spawn_cfg.collision_props = collision_props
                            mass_value = texture_record.get("default_mass") if isinstance(texture_record, dict) else None
                            if mass_value is not None:
                                spawn_cfg.mass_props = sim_utils.MassPropertiesCfg(mass=float(mass_value))
                            physics_target = (
                                texture_record.get("physics_target_prim_basename")
                                if isinstance(texture_record, dict)
                                else None
                            )
                            if physics_target:
                                setattr(spawn_cfg, "physics_target_prim_basename", physics_target)
                            patched_scene_assets.append(
                                f"{{asset_name}}: preserved textured USD visual with child physics injection"
                            )
                        elif texture_record is not None:
                            patched_scene_assets.append(
                                f"{{asset_name}}: texture audit candidate -> {{texture_record.get('loader_policy')}}"
                            )

                        if (
                            asset.get("type") == "articulation"
                            and scene_cfg.__class__.__name__ == "ArticulationCfg"
                        ):
                            init_state = getattr(scene_cfg, "init_state", None)
                            if init_state is not None and not hasattr(init_state, "lin_vel"):
                                joint_pos = getattr(init_state, "joint_pos", {{}})
                                if joint_pos is None:
                                    joint_pos = {{}}
                                scene_cfg.init_state = ArticulationCfg.InitialStateCfg(
                                    pos=list(getattr(init_state, "pos", [0.0, 0.0, 0.0])),
                                    rot=list(getattr(init_state, "rot", [1.0, 0.0, 0.0, 0.0])),
                                    lin_vel=[0.0, 0.0, 0.0],
                                    ang_vel=[0.0, 0.0, 0.0],
                                    joint_pos=joint_pos,
                                )
                                patched_scene_assets.append(
                                    f"{{asset_name}}: promoted init_state to ArticulationCfg.InitialStateCfg"
                                )

                    scene_ee_frame = getattr(scene, "ee_frame", None)
                    if scene_ee_frame is None or scene_ee_frame is MISSING:
                        if robot_type == "franka":
                            scene.ee_frame = FrameTransformerCfg(
                                prim_path="{{ENV_REGEX_NS}}/Robot/panda_link0",
                                debug_vis=False,
                                target_frames=[
                                    FrameTransformerCfg.FrameCfg(
                                        prim_path="{{ENV_REGEX_NS}}/Robot/panda_hand",
                                        name="end_effector",
                                        offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                                    ),
                                ],
                            )
                            patched_scene_assets.append("scene.ee_frame injected for franka")
                            scene_ee_frame = scene.ee_frame

                    env_ee_frame = getattr(env_cfg, "ee_frame", None)
                    if (
                        (env_ee_frame is None or env_ee_frame is MISSING)
                        and scene_ee_frame is not None
                        and scene_ee_frame is not MISSING
                    ):
                        env_cfg.ee_frame = scene_ee_frame
                        patched_scene_assets.append("env_cfg.ee_frame synced from scene.ee_frame")

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
                        if "min_separation" in params:
                            params.pop("min_separation", None)
                            patched_event_params.append(
                                f"{{event_name}} (dropped unsupported ['min_separation'])"
                            )
                        asset_cfg = params.get("asset_cfg")
                        asset_name = getattr(asset_cfg, "name", None)
                        if isinstance(asset_name, str) and _uses_xform_scene_entity(asset_name):
                            setattr(events_cfg, event_name, None)
                            disabled_event_terms.append(
                                f"{{event_name}} (static/xform reset term for {{asset_name}})"
                            )
                            continue
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
                patched_debug_vis = _disable_debug_visualization(env_cfg)
                if patched_debug_vis:
                    print(
                        "Disabled debug visualizers: "
                        f"{{patched_debug_vis[:20]}}"
                    )
                return (
                    patched_terms,
                    patched_scene_assets,
                    patched_actuators,
                    patched_event_params,
                    disabled_event_terms,
                    patched_debug_vis,
                )

            def _camera_aliases(camera_name):
                name = str(camera_name)
                if name.endswith("_cam"):
                    return (name, name[:-4])
                return (f"{{name}}_cam", name)

            def _capture_judge_images(cameras, requested_names):
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
                    for requested_name in requested_names:
                        logical_name = requested_name[:-4] if str(requested_name).endswith("_cam") else str(requested_name)
                        for name in _camera_aliases(requested_name):
                            if name in all_imgs and all_imgs[name] is not None:
                                judge_imgs[logical_name] = all_imgs[name]
                                break
                    return judge_imgs if judge_imgs else None
                except Exception:
                    return None

            def _estimate_task_asset_top_z(asset):
                asset_position = asset.get("position")
                if not isinstance(asset_position, (list, tuple)) or len(asset_position) != 3:
                    return None
                if asset.get("type") == "static":
                    return float(asset_position[2])
                scale = asset.get("scale")
                if isinstance(scale, (list, tuple)) and len(scale) == 3:
                    return float(asset_position[2]) + max(float(scale[2]) / 2.0, 0.0)
                return float(asset_position[2])

            def _resolve_target_position(detector, target, translated_positions=None, task_doc=None):
                if isinstance(target, (list, tuple)) and len(target) == 3:
                    return np.array(target, dtype=float)

                if isinstance(target, str):
                    if isinstance(translated_positions, dict):
                        translated = translated_positions.get(target)
                        if isinstance(translated, dict):
                            position = translated.get("position")
                            if isinstance(position, (list, tuple)) and len(position) == 3:
                                return np.array(position, dtype=float)
                        synthetic_support = translated_positions.get(f"{{target}}_top_surface")
                        if isinstance(synthetic_support, dict):
                            position = synthetic_support.get("position")
                            if isinstance(position, (list, tuple)) and len(position) == 3:
                                return np.array(position, dtype=float)
                        for info in translated_positions.values():
                            if not isinstance(info, dict):
                                continue
                            if info.get("support_asset_name") != target:
                                continue
                            position = info.get("position")
                            if isinstance(position, (list, tuple)) and len(position) == 3:
                                return np.array(position, dtype=float)

                    if isinstance(task_doc, dict):
                        for asset in task_doc.get("assets", []):
                            if asset.get("name") != target:
                                continue
                            asset_position = asset.get("position")
                            if not isinstance(asset_position, (list, tuple)) or len(asset_position) != 3:
                                break
                            top_z = _estimate_task_asset_top_z(asset)
                            if top_z is None:
                                break
                            return np.array(
                                [
                                    float(asset_position[0]),
                                    float(asset_position[1]),
                                    float(top_z),
                                ],
                                dtype=float,
                            )

                    try:
                        return np.array(detector.get_object_position(target), dtype=float)
                    except Exception:
                        return None

                return None

            def _resolve_target_metadata(translated_positions, target, field):
                if not isinstance(target, str) or not isinstance(translated_positions, dict):
                    return None
                translated = translated_positions.get(target)
                if not isinstance(translated, dict):
                    return None
                value = translated.get(field)
                if isinstance(value, (list, tuple)):
                    return np.array(value, dtype=float)
                return value

            def _resolve_runtime_object_name(name, translated_positions=None):
                if not isinstance(name, str) or not isinstance(translated_positions, dict):
                    return name
                translated = translated_positions.get(name)
                if not isinstance(translated, dict):
                    return name
                runtime_name = translated.get("grasp_object_name") or translated.get("source_object_name")
                if isinstance(runtime_name, str) and runtime_name:
                    return runtime_name
                return name

            def _resolve_object_position(detector, name, translated_positions=None):
                last_error = None
                for candidate in dict.fromkeys([
                    _resolve_runtime_object_name(name, translated_positions),
                    name,
                ]):
                    if not isinstance(candidate, str):
                        continue
                    try:
                        return np.asarray(detector.get_object_position(candidate), dtype=float)
                    except Exception as exc:
                        last_error = exc
                if isinstance(translated_positions, dict):
                    translated = translated_positions.get(name)
                    if isinstance(translated, dict):
                        position = translated.get("position")
                        if isinstance(position, (list, tuple)) and len(position) == 3:
                            return np.asarray(position, dtype=float)
                if last_error is not None:
                    raise last_error
                raise KeyError(f"Object '{{name}}' position unavailable")

            def _resolve_object_pose(detector, name, translated_positions=None):
                last_error = None
                for candidate in dict.fromkeys([
                    _resolve_runtime_object_name(name, translated_positions),
                    name,
                ]):
                    if not isinstance(candidate, str):
                        continue
                    try:
                        pos, quat = detector.get_object_pose(candidate)
                        return np.asarray(pos, dtype=float), np.asarray(quat, dtype=float)
                    except Exception as exc:
                        last_error = exc
                if isinstance(translated_positions, dict):
                    translated = translated_positions.get(name)
                    if isinstance(translated, dict):
                        position = translated.get("position")
                        quat = translated.get("quaternion")
                        if (
                            isinstance(position, (list, tuple))
                            and len(position) == 3
                            and isinstance(quat, (list, tuple))
                            and len(quat) == 4
                        ):
                            return np.asarray(position, dtype=float), np.asarray(quat, dtype=float)
                if last_error is not None:
                    raise last_error
                raise KeyError(f"Object '{{name}}' pose unavailable")

            def _read_named_joint_position(env, task_doc, joint_name):
                if env is None or not isinstance(joint_name, str) or not isinstance(task_doc, dict):
                    return None

                articulation_name = None
                for asset in task_doc.get("assets", []):
                    if asset.get("type") != "articulation":
                        continue
                    joints = asset.get("joints", {{}})
                    if isinstance(joints, dict) and joint_name in joints:
                        articulation_name = asset.get("name")
                        break
                if not articulation_name:
                    return None

                try:
                    articulation = env.scene[articulation_name]
                except Exception:
                    articulation = getattr(env.scene, articulation_name, None)
                if articulation is None or not hasattr(articulation, "data"):
                    return None

                joint_names = list(getattr(articulation, "joint_names", []) or [])
                joint_idx = None
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                elif hasattr(articulation, "find_joints"):
                    try:
                        result = articulation.find_joints(joint_name)
                        if isinstance(result, tuple):
                            joint_indices = result[0]
                        else:
                            joint_indices = result
                        if len(joint_indices):
                            joint_idx = int(joint_indices[0])
                    except Exception:
                        joint_idx = None
                if joint_idx is None:
                    return None

                try:
                    joint_pos = articulation.data.joint_pos[0, joint_idx].cpu().numpy()
                    return float(joint_pos)
                except Exception:
                    return None

            def _read_articulation_joint_position(env, articulation_name, joint_name):
                if env is None or not isinstance(articulation_name, str) or not articulation_name:
                    return None
                if not isinstance(joint_name, str) or not joint_name:
                    return None

                try:
                    articulation = env.scene[articulation_name]
                except Exception:
                    articulation = getattr(env.scene, articulation_name, None)
                if articulation is None or not hasattr(articulation, "data"):
                    return None

                joint_names = list(getattr(articulation, "joint_names", []) or [])
                joint_idx = None
                if joint_name in joint_names:
                    joint_idx = joint_names.index(joint_name)
                elif hasattr(articulation, "find_joints"):
                    try:
                        result = articulation.find_joints(joint_name)
                        if isinstance(result, tuple):
                            joint_indices = result[0]
                        else:
                            joint_indices = result
                        if len(joint_indices):
                            joint_idx = int(joint_indices[0])
                    except Exception:
                        joint_idx = None
                if joint_idx is None:
                    return None

                try:
                    joint_pos = articulation.data.joint_pos[0, joint_idx].cpu().numpy()
                    return float(joint_pos)
                except Exception:
                    return None

            def _infer_ordered_objects_for_goal(task_description, translated_positions):
                def _candidate_index(description, candidate):
                    pattern = rf"(?<![a-z0-9_]){{re.escape(candidate)}}(?![a-z0-9_])"
                    match = re.search(pattern, description)
                    return match.start() if match else -1

                description = str(task_description or "").lower()
                ordered = []
                seen = set()
                for name, info in (translated_positions or {{}}).items():
                    task_role = str(info.get("task_role", ""))
                    if task_role in {{"placement_target", "target_marker", "support_surface", "goal_target"}}:
                        continue
                    candidates = [name.lower()]
                    for key in ("color_name", "asset_label"):
                        value = info.get(key)
                        if isinstance(value, str):
                            candidates.append(value.lower())
                    aliases = info.get("aliases") or []
                    for alias in aliases:
                        if isinstance(alias, str):
                            candidates.append(alias.lower())
                    indices = [_candidate_index(description, candidate) for candidate in candidates if candidate]
                    indices = [idx for idx in indices if idx >= 0]
                    if indices and name not in seen:
                        ordered.append((min(indices), name))
                        seen.add(name)
                ordered.sort()
                return [name for _, name in ordered]

            def _compile_goal_conditions(goal, task_description="", translated_positions=None):
                sc = goal.get("success_criteria", {{}})
                conditions = []
                if "conditions" in sc:
                    conditions.extend(sc["conditions"])
                elif "stacking_order" in sc:
                    order = sc["stacking_order"]
                    for i in range(1, len(order)):
                        conditions.append({{
                            "type": "on_top_of",
                            "object": order[i],
                            "target": order[i - 1],
                        }})
                elif "conditions" in goal:
                    conditions.extend(goal["conditions"])

                if conditions:
                    return conditions

                desc = " ".join(
                    part for part in (task_description, goal.get("description", "")) if isinstance(part, str)
                ).lower()
                if "stack" not in desc:
                    return conditions

                ordered_objects = _infer_ordered_objects_for_goal(desc, translated_positions or {{}})
                if len(ordered_objects) >= 2:
                    for i in range(1, len(ordered_objects)):
                        conditions.append({{
                            "type": "on_top_of",
                            "object": ordered_objects[i],
                            "target": ordered_objects[i - 1],
                        }})

                if conditions and isinstance(sc, dict) and sc.get("inside_tray"):
                    tray_center = sc.get("tray_center")
                    tray_half_extent = sc.get("tray_half_extent")
                    if isinstance(tray_center, (list, tuple)) and len(tray_center) == 2 and tray_half_extent is not None:
                        conditions.append({{
                            "type": "inside_tray",
                            "objects": ordered_objects,
                            "tray_center": list(tray_center),
                            "tray_half_extent": float(tray_half_extent),
                        }})

                return conditions

            def _is_vlm_helper_target(name, info):
                name_lower = str(name or "").lower()
                if not name_lower:
                    return False
                if name_lower.endswith("_anchor") or "anchor" in name_lower:
                    return True
                if name_lower in {{"command_pose", "command_target", "target_pose"}}:
                    return True
                task_role = str((info or {{}}).get("task_role", ""))
                if task_role in {{"handle_target"}}:
                    return True
                if name_lower.startswith("drawer_container"):
                    return True
                asset_label = str((info or {{}}).get("asset_label", "")).lower()
                aliases = [
                    str(alias).lower()
                    for alias in ((info or {{}}).get("aliases") or [])
                    if isinstance(alias, str)
                ]
                synthetic_tokens = ("anchor", "command pose", "drawer container")
                return any(token in asset_label for token in synthetic_tokens) or any(
                    any(token in alias for token in synthetic_tokens) for alias in aliases
                )

            def _select_vlm_relevant_objects(goal_conditions, translated_positions, scene_state):
                translated_positions = translated_positions or {{}}
                scene_state = scene_state or {{}}
                referenced = set()
                for condition in goal_conditions or []:
                    if not isinstance(condition, dict):
                        continue
                    subject = condition.get("subject", condition.get("object"))
                    target = condition.get("target")
                    objects = condition.get("objects") or []
                    if isinstance(subject, str):
                        referenced.add(subject)
                    if isinstance(target, str):
                        referenced.add(target)
                    if isinstance(objects, list):
                        for name in objects:
                            if isinstance(name, str):
                                referenced.add(name)

                selected = {{}}
                combined_names = list(dict.fromkeys(list(translated_positions.keys()) + list(scene_state.keys())))
                for name in combined_names:
                    base = scene_state.get(name, {{}})
                    translated = translated_positions.get(name, {{}})
                    merged = dict(base) if isinstance(base, dict) else {{}}
                    if isinstance(translated, dict):
                        merged.update(translated)
                    if not merged:
                        continue
                    task_role = str(merged.get("task_role", ""))
                    include = name in referenced
                    if task_role in {{"target_marker", "support_surface", "goal_target"}}:
                        include = True
                    if not include:
                        continue
                    if _is_vlm_helper_target(name, merged):
                        continue
                    selected[name] = merged
                return selected

            def _verify_goal_conditions(
                detector,
                goal,
                env=None,
                translated_positions=None,
                task_doc=None,
                task_description="",
                xy_threshold=0.05,
                height_diff=0.05,
                z_threshold=0.01,
            ):
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

                    conditions = _compile_goal_conditions(
                        goal,
                        task_description=task_description,
                        translated_positions=translated_positions,
                    )

                    # Infer stacking from goal description if no explicit conditions
                    if not conditions:
                        desc = " ".join(
                            part for part in (task_description, goal.get("description", "")) if isinstance(part, str)
                        ).lower()
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
                                top_pos = _resolve_object_position(
                                    detector,
                                    top_obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{top_obj}} on {{bottom_obj}}: detection failed ({{e}})")
                                continue
                            try:
                                bot_pos = _resolve_object_position(
                                    detector,
                                    bottom_obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                bot_pos = _resolve_target_position(
                                    detector,
                                    bottom_obj,
                                    translated_positions=translated_positions,
                                    task_doc=task_doc,
                                )
                                if bot_pos is None:
                                    details.append(f"{{top_obj}} on {{bottom_obj}}: target missing ({{e}})")
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
                                obj_pos = _resolve_object_position(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} at_position: detection failed ({{e}})")
                                continue
                            target_pos = _resolve_target_position(
                                detector,
                                target,
                                translated_positions=translated_positions,
                                task_doc=task_doc,
                            )
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
                        elif cond_type == "on_surface":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            target = cond.get("target")
                            if not obj or target is None:
                                details.append("skip: missing on_surface fields")
                                continue
                            try:
                                obj_pos = _resolve_object_position(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} on_surface: detection failed ({{e}})")
                                continue
                            target_pos = _resolve_target_position(
                                detector,
                                target,
                                translated_positions=translated_positions,
                                task_doc=task_doc,
                            )
                            if target_pos is None:
                                details.append(f"{{obj}} on_surface target '{{target}}' missing")
                                continue
                            support_margin = max(z_threshold, 0.01)
                            ok = obj_pos[2] >= (target_pos[2] - support_margin)
                            if ok:
                                passed += 1
                            details.append(
                                f"{{obj}} on_surface {{target}}: obj_z={{obj_pos[2]:.3f}}m support_z={{target_pos[2]:.3f}}m {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type == "inside_tray":
                            total += 1
                            objects = cond.get("objects", [])
                            tray_center = cond.get("tray_center")
                            tray_half_extent = float(cond.get("tray_half_extent", 0.0))
                            if (
                                not objects
                                or not isinstance(tray_center, (list, tuple))
                                or len(tray_center) != 2
                                or tray_half_extent <= 0.0
                            ):
                                details.append("skip: invalid inside_tray fields")
                                continue
                            tray_center_xy = np.array(tray_center[:2], dtype=float)
                            inside = True
                            tray_details = []
                            for obj in objects:
                                try:
                                    obj_pos = _resolve_object_position(
                                        detector,
                                        obj,
                                        translated_positions=translated_positions,
                                    )
                                except Exception as e:
                                    inside = False
                                    tray_details.append(f"{{obj}} missing ({{e}})")
                                    continue
                                delta = np.abs(obj_pos[:2] - tray_center_xy)
                                obj_ok = bool(np.all(delta <= (tray_half_extent + xy_threshold)))
                                inside = inside and obj_ok
                                tray_details.append(
                                    f"{{obj}} delta=({{delta[0]:.3f}}, {{delta[1]:.3f}}) {{'OK' if obj_ok else 'FAIL'}}"
                                )
                            if inside:
                                passed += 1
                            details.append("inside_tray: " + "; ".join(tray_details))
                        elif cond_type == "inside_drawer":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            target = cond.get("target")
                            tolerance = float(cond.get("tolerance", xy_threshold) or xy_threshold)
                            try:
                                obj_pos = _resolve_object_position(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} inside_drawer {{target}}: detection failed ({{e}})")
                                continue
                            closed_center = _resolve_target_metadata(translated_positions, target, "closed_center_position")
                            half_extents = _resolve_target_metadata(translated_positions, target, "container_half_extents")
                            slide_axis = _resolve_target_metadata(translated_positions, target, "slide_axis_world")
                            articulation_name = _resolve_target_metadata(translated_positions, target, "articulation_name")
                            joint_name = _resolve_target_metadata(translated_positions, target, "joint_name")
                            close_target = _resolve_target_metadata(translated_positions, target, "close_target_joint_position")
                            if (
                                closed_center is None
                                or half_extents is None
                                or slide_axis is None
                                or articulation_name is None
                                or joint_name is None
                            ):
                                details.append(f"{{obj}} inside_drawer {{target}}: target metadata unavailable")
                                continue
                            joint_pos = _read_articulation_joint_position(env, str(articulation_name), str(joint_name))
                            if joint_pos is None:
                                details.append(f"{{obj}} inside_drawer {{target}}: joint state unavailable")
                                continue
                            close_value = float(close_target or 0.0)
                            center = np.asarray(closed_center, dtype=float) + np.asarray(slide_axis, dtype=float) * (joint_pos - close_value)
                            delta = np.abs(np.asarray(obj_pos, dtype=float) - center)
                            bounds = np.asarray(half_extents, dtype=float) + float(tolerance)
                            ok = bool(np.all(delta <= bounds))
                            if ok:
                                passed += 1
                            details.append(
                                f"{{obj}} inside_drawer {{target}}: delta=({{delta[0]:.3f}}, {{delta[1]:.3f}}, {{delta[2]:.3f}}) "
                                f"bounds=({{bounds[0]:.3f}}, {{bounds[1]:.3f}}, {{bounds[2]:.3f}}) {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type in ("lifted", "above", "height_above"):
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            min_height = cond.get("height", cond.get("value"))
                            target = cond.get("target")
                            try:
                                pos = _resolve_object_position(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                                if isinstance(target, str):
                                    target_pos = _resolve_target_position(
                                        detector,
                                        target,
                                        translated_positions=translated_positions,
                                        task_doc=task_doc,
                                    )
                                    baseline_z = float(target_pos[2]) if target_pos is not None else 0.0
                                    if cond_type == "height_above":
                                        threshold = float(min_height if min_height is not None else z_threshold)
                                        ok = (pos[2] - baseline_z) > threshold
                                        details.append(
                                            f"{{obj}} {{cond_type}} {{target}}: delta_z={{pos[2] - baseline_z:.3f}}m {{'OK' if ok else 'FAIL'}}"
                                        )
                                    else:
                                        support_margin = max(z_threshold, 0.01)
                                        ok = pos[2] >= (baseline_z - support_margin)
                                        details.append(
                                            f"{{obj}} {{cond_type}} {{target}}: obj_z={{pos[2]:.3f}}m support_z={{baseline_z:.3f}}m {{'OK' if ok else 'FAIL'}}"
                                        )
                                else:
                                    threshold = float(min_height if min_height is not None else 0.1)
                                    ok = pos[2] > threshold
                                    details.append(f"{{obj}} lifted: z={{pos[2]:.3f}}m {{'OK' if ok else 'FAIL'}}")
                                if ok:
                                    passed += 1
                            except Exception as e:
                                details.append(f"{{obj}} lifted: detection failed ({{e}})")
                        elif cond_type == "position_above":
                            total += 1
                            joint_name = cond.get("subject", cond.get("object", ""))
                            threshold = float(cond.get("value", cond.get("height", 0.0)) or 0.0)
                            tolerance = float(cond.get("tolerance", max(z_threshold, 0.02)) or max(z_threshold, 0.02))
                            joint_pos = _read_named_joint_position(env, task_doc, joint_name)
                            if joint_pos is None:
                                details.append(f"{{joint_name}} position_above: joint state unavailable")
                                continue
                            ok = joint_pos >= (threshold - tolerance)
                            if ok:
                                passed += 1
                            details.append(
                                f"{{joint_name}} position_above {{threshold:.3f}} (tol={{tolerance:.3f}}): joint={{joint_pos:.3f}} {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type == "position_below":
                            total += 1
                            joint_name = cond.get("subject", cond.get("object", ""))
                            threshold = float(cond.get("value", cond.get("height", 0.0)) or 0.0)
                            tolerance = float(cond.get("tolerance", max(z_threshold, 0.02)) or max(z_threshold, 0.02))
                            joint_pos = _read_named_joint_position(env, task_doc, joint_name)
                            if joint_pos is None:
                                details.append(f"{{joint_name}} position_below: joint state unavailable")
                                continue
                            ok = joint_pos <= (threshold + tolerance)
                            if ok:
                                passed += 1
                            details.append(
                                f"{{joint_name}} position_below {{threshold:.3f}} (tol={{tolerance:.3f}}): joint={{joint_pos:.3f}} {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type == "inserted_into":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            target = cond.get("target")
                            tol = float(cond.get("tolerance", xy_threshold))
                            try:
                                obj_pos, _ = _resolve_object_pose(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} inserted_into {{target}}: detection failed ({{e}})")
                                continue
                            target_pos = _resolve_target_metadata(translated_positions, target, "target_position")
                            if target_pos is None:
                                target_pos = _resolve_target_position(
                                    detector,
                                    target,
                                    translated_positions=translated_positions,
                                    task_doc=task_doc,
                                )
                            entry_pos = _resolve_target_metadata(translated_positions, target, "entry_position")
                            insertion_axis = _resolve_target_metadata(translated_positions, target, "insertion_axis_world")
                            if insertion_axis is not None:
                                insertion_axis = np.asarray(insertion_axis, dtype=float)
                                axis_norm = np.linalg.norm(insertion_axis)
                                if axis_norm > 1e-6:
                                    insertion_axis = insertion_axis / axis_norm
                            if target_pos is None:
                                details.append(f"{{obj}} inserted_into {{target}}: target metadata unavailable")
                                continue
                            delta = obj_pos - np.asarray(target_pos, dtype=float)
                            radial_err = float(np.linalg.norm(delta[:2]))
                            depth = None
                            if insertion_axis is not None and entry_pos is not None:
                                entry_pos = np.asarray(entry_pos, dtype=float)
                                depth = float(np.dot(obj_pos - entry_pos, insertion_axis))
                                radial_vec = (obj_pos - np.asarray(target_pos, dtype=float)) - insertion_axis * float(np.dot(obj_pos - np.asarray(target_pos, dtype=float), insertion_axis))
                                radial_err = float(np.linalg.norm(radial_vec))
                            required_depth = _resolve_target_metadata(translated_positions, target, "required_depth")
                            if required_depth is None:
                                required_depth = 0.0
                            ok = radial_err < tol and (depth is None or depth >= float(required_depth))
                            if ok:
                                passed += 1
                            if depth is None:
                                details.append(
                                    f"{{obj}} inserted_into {{target}}: radial_err={{radial_err:.3f}}m {{'OK' if ok else 'FAIL'}}"
                                )
                            else:
                                details.append(
                                    f"{{obj}} inserted_into {{target}}: radial_err={{radial_err:.3f}}m depth={{depth:.3f}}m {{'OK' if ok else 'FAIL'}}"
                                )
                        elif cond_type == "height_below":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            target = cond.get("target")
                            required_depth = float(cond.get("value", cond.get("height", 0.0)) or 0.0)
                            try:
                                obj_pos = _resolve_object_position(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} height_below {{target}}: detection failed ({{e}})")
                                continue
                            entry_pos = _resolve_target_metadata(translated_positions, target, "entry_position")
                            insertion_axis = _resolve_target_metadata(translated_positions, target, "insertion_axis_world")
                            if entry_pos is None or insertion_axis is None:
                                target_pos = _resolve_target_position(
                                    detector,
                                    target,
                                    translated_positions=translated_positions,
                                    task_doc=task_doc,
                                )
                                if target_pos is None:
                                    details.append(f"{{obj}} height_below {{target}}: target metadata unavailable")
                                    continue
                                depth = float(np.asarray(target_pos, dtype=float)[2] - obj_pos[2])
                            else:
                                insertion_axis = np.asarray(insertion_axis, dtype=float)
                                axis_norm = np.linalg.norm(insertion_axis)
                                if axis_norm > 1e-6:
                                    insertion_axis = insertion_axis / axis_norm
                                depth = float(np.dot(obj_pos - np.asarray(entry_pos, dtype=float), insertion_axis))
                            ok = depth >= required_depth
                            if ok:
                                passed += 1
                            details.append(
                                f"{{obj}} height_below {{target}}: depth={{depth:.3f}}m {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type == "upright":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            tolerance = float(cond.get("tolerance", 0.1) or 0.1)
                            try:
                                _, obj_quat = _resolve_object_pose(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} upright: pose lookup failed ({{e}})")
                                continue
                            info = translated_positions.get(obj, {{}}) if isinstance(translated_positions, dict) else {{}}
                            axis_local = np.asarray(info.get("upright_axis_local", [1.0, 0.0, 0.0]), dtype=float)
                            w, x, y, z = obj_quat
                            rot = np.array([
                                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                            ], dtype=float)
                            axis_world = rot @ axis_local
                            axis_world = axis_world / max(np.linalg.norm(axis_world), 1e-9)
                            vertical = np.array([0.0, 0.0, 1.0], dtype=float)
                            angle = float(np.arccos(np.clip(abs(np.dot(axis_world, vertical)), -1.0, 1.0)))
                            ok = angle <= tolerance
                            if ok:
                                passed += 1
                            details.append(
                                f"{{obj}} upright: angle={{angle:.3f}}rad {{'OK' if ok else 'FAIL'}}"
                            )
                        elif cond_type == "in_slot":
                            total += 1
                            obj = cond.get("object", cond.get("subject", ""))
                            target = cond.get("target")
                            tol = float(cond.get("tolerance", xy_threshold))
                            try:
                                obj_pos, obj_quat = _resolve_object_pose(
                                    detector,
                                    obj,
                                    translated_positions=translated_positions,
                                )
                            except Exception as e:
                                details.append(f"{{obj}} in_slot {{target}}: detection failed ({{e}})")
                                continue
                            slot_pos = _resolve_target_metadata(translated_positions, target, "slot_position")
                            if slot_pos is None:
                                slot_pos = _resolve_target_position(
                                    detector,
                                    target,
                                    translated_positions=translated_positions,
                                    task_doc=task_doc,
                                )
                            if slot_pos is None:
                                details.append(f"{{obj}} in_slot {{target}}: slot metadata unavailable")
                                continue
                            slot_pos = np.asarray(slot_pos, dtype=float)
                            xy_err = float(np.linalg.norm(obj_pos[:2] - slot_pos[:2]))
                            z_err = float(abs(obj_pos[2] - slot_pos[2]))
                            z_tol = float(
                                _resolve_target_metadata(translated_positions, target, "slot_z_tolerance")
                                or max(z_threshold, tol)
                            )
                            yaw_target = _resolve_target_metadata(translated_positions, target, "slot_yaw_deg")
                            yaw_tol = _resolve_target_metadata(translated_positions, target, "slot_yaw_tolerance_deg")
                            yaw_err_deg = None
                            if yaw_target is not None:
                                w, x, y, z = obj_quat
                                yaw_deg = float(
                                    np.degrees(
                                        np.arctan2(
                                            2.0 * (w * z + x * y),
                                            1.0 - 2.0 * (y * y + z * z),
                                        )
                                    )
                                )
                                yaw_err_deg = abs(((yaw_deg - float(yaw_target) + 180.0) % 360.0) - 180.0)
                            ok = xy_err < tol and z_err < z_tol and (
                                yaw_err_deg is None or yaw_tol is None or yaw_err_deg <= float(yaw_tol)
                            )
                            if ok:
                                passed += 1
                            if yaw_err_deg is None or yaw_tol is None:
                                details.append(
                                    f"{{obj}} in_slot {{target}}: xy_err={{xy_err:.3f}}m z_err={{z_err:.3f}}m {{'OK' if ok else 'FAIL'}}"
                                )
                            else:
                                details.append(
                                    f"{{obj}} in_slot {{target}}: xy_err={{xy_err:.3f}}m z_err={{z_err:.3f}}m "
                                    f"yaw_err={{yaw_err_deg:.1f}}deg {{'OK' if ok else 'FAIL'}}"
                                )

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
                dataset_camera_names = list(dict.fromkeys(cfg.get("dataset_cameras", ["top", "wrist", "front"])))
                judge_camera_names = list(dict.fromkeys(cfg.get("judge_cameras", ["wrist", "front"])))
                scene_camera_names = list(dict.fromkeys(dataset_camera_names + judge_camera_names))
                _record_phase("config_loaded", "ok", config_path="{config_json_path}", output_dir=cfg["output_dir"])

                # Load task YAML
                import yaml
                with open(cfg["yaml_path"]) as f:
                    task_doc = yaml.safe_load(f)
                task_name = task_doc.get("task", {{}}).get("name", "unknown")
                task_desc = task_doc.get("task", {{}}).get("description", "")
                robot_cfg = load_robot_config(cfg["robot_name"])
                robot_cfg = apply_task_top_camera_override(robot_cfg, task_doc)
                _record_phase("robot_config_loaded", "ok", robot_name=robot_cfg.name)
                cap_profile = select_cap_profile(robot_cfg)
                recorder = None
                env = None
                results = {{
                    "episodes": [],
                    "total_episodes": 0,
                    "successful_episodes": 0,
                    "geometry_successful_episodes": 0,
                    "vlm_successful_episodes": 0,
                    "overall_successful_episodes": 0,
                    "success_policy": "geometry_or_vlm",
                    "pipeline_completed": False,
                    "target_met": False,
                    "geometry_target_met": False,
                    "vlm_target_met": False,
                    "overall_target_met": False,
                    "raw_dataset": None,
                    "robot_skill_class": cap_profile.class_name,
                    "judge_enabled": USE_VLM_JUDGE,
                    "judge_available": False,
                    "judge_unavailable_reason": None,
                    "front_video_generated": False,
                    "front_video_path": None,
                    "front_video_camera_name": cfg.get("front_video_camera_name", "front"),
                    "front_video_episode": None,
                    "front_video_success_type": None,
                    "front_video_error": None,
                    "failure_phase": None,
                    "failure_category": None,
                    "phase_trace_path": PHASE_TRACE_PATH,
                    "startup_diagnostics_path": STARTUP_DIAGNOSTICS_PATH,
                    "heartbeat_path": HEARTBEAT_PATH,
                    "stderr_tail_path": STDERR_TAIL_PATH,
                }}

                def _write_results():
                    results_path = os.path.join(cfg["output_dir"], "collection_results.json")
                    with open(results_path, "w") as _f:
                        json.dump(results, _f, indent=2)
                    return results_path

                _update_startup_diagnostics(
                    task_name=task_name,
                    task_description=task_desc,
                    robot_name=robot_cfg.name,
                    env_dir=cfg.get("env_dir"),
                    output_dir=cfg["output_dir"],
                    use_vlm_judge=USE_VLM_JUDGE,
                )
                _record_phase("task_loaded", "ok", task_name=task_name)

                # Create environment with cameras injected into scene config
                env_cfg_class = _find_env_cfg_class()
                if env_cfg_class is None:
                    msg = "Could not find EnvCfg class in env_cfg.py"
                    print(f"ERROR: {{msg}}")
                    results["pipeline_error"] = msg
                    results["failure_phase"] = "env_cfg_lookup"
                    results["failure_category"] = "env_cfg_lookup_failed"
                    _record_phase("env_cfg_lookup", "error", error=msg)
                    _update_startup_diagnostics(pipeline_error=msg, failure_phase="env_cfg_lookup", failure_category="env_cfg_lookup_failed")
                    _write_results()
                    simulation_app.close()
                    return

                _record_phase("env_cfg_lookup", "ok", class_name=env_cfg_class.__name__)
                try:
                    env_cfg = env_cfg_class()
                    _record_phase("env_cfg_instantiated", "ok", class_name=env_cfg_class.__name__)
                    env_cfg.scene.num_envs = cfg["num_envs"]
                    _sanitize_generated_env_cfg(env_cfg, task_doc)
                    _record_phase("env_cfg_sanitized", "ok")
                except Exception as exc:
                    payload = _write_exception_artifacts(exc, "env_cfg_prepare", _classify_pipeline_failure(exc))
                    results["pipeline_error"] = payload["message"]
                    results["failure_phase"] = payload["failure_phase"]
                    results["failure_category"] = payload["failure_category"]
                    _write_results()
                    simulation_app.close()
                    return

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
                camera_attr_names = inject_cameras_into_scene(env_cfg, robot_cfg, camera_names=scene_camera_names)
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

                _record_phase("env_constructor", "start")
                try:
                    env = ManagerBasedRLEnv(cfg=env_cfg)
                    _record_phase("env_constructor", "ok")
                except Exception as exc:
                    payload = _write_exception_artifacts(exc, "env_constructor", _classify_pipeline_failure(exc))
                    results["pipeline_error"] = payload["message"]
                    results["failure_phase"] = payload["failure_phase"]
                    results["failure_category"] = payload["failure_category"]
                    _write_results()
                    simulation_app.close()
                    return

                _record_phase("env_post_setup", "start")
                try:
                    _post_env_setup = globals().get("post_env_setup")
                    if callable(_post_env_setup):
                        _post_env_setup(env)
                        print("Applied env_cfg post_env_setup hook")
                    _record_phase("env_post_setup", "ok")
                except Exception as exc:
                    payload = _write_exception_artifacts(exc, "env_post_setup", _classify_pipeline_failure(exc))
                    results["pipeline_error"] = payload["message"]
                    results["failure_phase"] = payload["failure_phase"]
                    results["failure_category"] = payload["failure_category"]
                    _write_results()
                    simulation_app.close()
                    return

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

                _record_phase("components_init", "start")
                try:
                    # Initialize components
                    robot_interface = SimRobotInterface(env, robot_cfg, env_idx=0)
                    detector = SimDetector(env, task_doc, env_idx=0)

                    # Read robot base position for IK frame conversion
                    robot_base_pos = env.scene["robot"].data.root_pos_w[0].cpu().numpy()
                    print(f"Robot base position (world frame): {{robot_base_pos}}")

                    # Initialize multi-camera system from scene-integrated cameras
                    cameras = None
                    target_grounder = None
                    camera = None  # legacy fallback
                    if camera_attr_names:
                        try:
                            cameras = SceneCameraManager(env, camera_attr_names, robot_cfg.cameras)
                            print(f"SceneCameraManager initialized: {{cameras.camera_names}}")
                        except Exception as cam_err:
                            print(f"WARNING: SceneCameraManager init failed ({{cam_err}}).")
                            cameras = None

                        recorder = SimRecorder(
                            robot_cfg=robot_cfg,
                            output_dir=cfg["output_dir"],
                            dataset_name="raw_dataset",
                            fps=cfg["recording_fps"],
                            front_video_fps=cfg.get("front_video_fps", cfg["recording_fps"]),
                            front_video_max_priority=3 if USE_VLM_JUDGE else 1,
                            front_video_camera_name=cfg.get("front_video_camera_name", "front"),
                            dataset_cameras=dataset_camera_names,
                    )

                    # Create IK solver (Pinocchio when URDF exists, otherwise DifferentialIK fallback)
                    ik_solver = create_ik_solver(robot_cfg, robot_interface=robot_interface)

                    if cameras is not None:
                        try:
                            target_grounder = MultiViewTargetGrounder(
                                cameras=cameras,
                                robot_interface=robot_interface,
                                robot_cfg=robot_cfg,
                            )
                            print("MultiViewTargetGrounder initialized")
                        except Exception as grounder_err:
                            print(f"WARNING: MultiViewTargetGrounder init failed ({{grounder_err}}).")
                            target_grounder = None

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
                        ik_debug=bool(cfg.get("ik_debug", False)),
                    )
                    _record_phase("components_init", "ok")
                except Exception as exc:
                    payload = _write_exception_artifacts(exc, "components_init", _classify_pipeline_failure(exc))
                    results["pipeline_error"] = payload["message"]
                    results["failure_phase"] = payload["failure_phase"]
                    results["failure_category"] = payload["failure_category"]
                    _write_results()
                    try:
                        env.close()
                    except Exception:
                        pass
                    simulation_app.close()
                    return

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
                    _record_phase("cap_generator_init", "ok", model=cfg["llm_model"])
                except Exception as e:
                    print(f"SimCaPGenerator init failed: {{e}}")
                    _record_phase("cap_generator_init", "error", error=str(e))

                # Initialize VLM judge if enabled
                sim_judge = None
                judge_init_error = ""
                if USE_VLM_JUDGE:
                    sim_judge = SimJudge(model=cfg.get("vlm_model", "gpt-5"))
                    if not sim_judge.is_available:
                        judge_init_error = sim_judge.availability_reason
                        print(f"SimJudge unavailable: {{judge_init_error}}")
                        sim_judge = None
                        _record_phase("vlm_judge_init", "error", error=judge_init_error)
                    else:
                        print("SimJudge initialized (VLM-based episode judging enabled)")
                        _record_phase("vlm_judge_init", "ok", model=cfg.get("vlm_model", "gpt-5"))

                results["judge_available"] = sim_judge is not None
                results["judge_unavailable_reason"] = judge_init_error or None

                # Success-based episode loop
                target_success = cfg.get("target_successful_episodes", cfg["max_episodes"])
                max_attempts = cfg.get("max_total_attempts", target_success * 5)
                episode_idx = 0
                successful_count = 0

                try:
                    _record_phase("policy_support_check", "start")
                    supported_task, unsupported_reason = is_supported_tabletop_task(task_doc)
                    if not supported_task:
                        raise RuntimeError(
                            f"Task unsupported by current CaP tabletop profile: {{unsupported_reason}}"
                        )
                    if cap_generator is None:
                        raise RuntimeError("CaP generator unavailable; check Azure OpenAI configuration")
                    if USE_VLM_JUDGE and sim_judge is None:
                        raise RuntimeError(f"VLM judge unavailable: {{judge_init_error}}")
                    _update_startup_diagnostics(supported_task=True, supported_reason=unsupported_reason)
                    _record_phase("policy_support_check", "ok", reason=unsupported_reason)

                    while successful_count < target_success and episode_idx < max_attempts:
                        print(f"\\n=== Episode {{episode_idx + 1}} (success: {{successful_count}}/{{target_success}}, max: {{max_attempts}}) ===")
                        _record_phase("episode_start", "start", episode=episode_idx)

                        # Reset environment
                        _record_phase("episode_reset", "start", episode=episode_idx)
                        obs, info = env.reset()
                        robot_interface.reset_termination_flags()
                        _record_phase("episode_reset", "ok", episode=episode_idx)

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
                        _record_phase("episode_warmup", "ok", episode=episode_idx)

                        # Start recording
                        recorder.start_episode(task_desc, episode_idx)
                        _record_phase("recorder_start_episode", "ok", episode=episode_idx)

                        # Get scene state
                        _record_phase("scene_detect", "start", episode=episode_idx)
                        scene_state = detector.get_all_objects()
                        print(f"Scene objects: {{list(scene_state.keys())}}")
                        _update_startup_diagnostics(last_scene_objects=sorted(scene_state.keys()))
                        _record_phase("scene_detect", "ok", episode=episode_idx, object_count=len(scene_state))

                        # Capture initial images for VLM judge (wrist + front)
                        initial_images = None
                        if sim_judge is not None:
                            _record_phase("initial_judge_capture", "start", episode=episode_idx)
                            initial_images = _capture_judge_images(cameras, judge_camera_names)
                            _record_phase(
                                "initial_judge_capture",
                                "ok",
                                episode=episode_idx,
                                image_views=sorted(initial_images.keys()) if initial_images else [],
                            )

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
                        write_json_artifact(
                            os.path.join(run_dir, "scene_detect_objects.json"),
                            {{
                                "object_count": len(scene_state),
                                "objects": scene_state,
                            }},
                        )
                        initial_image_paths = save_judge_images(run_dir, "initial", initial_images)
                        final_image_paths = []
                        generated_code = ""
                        translated_positions = {{}}
                        artifact_paths = {{}}
                        execution_context_path = None
                        handle_grounding_path = None
                        codegen_success = False
                        codegen_error = ""
                        execution_error = ""
                        blocked_fallback_reason = ""
                        geometry_success = False
                        vlm_success = False
                        overall_success = False
                        judge_available = False
                        judge_prompt_path = None
                        judge_prompt_context_path = None
                        judge_raw_response_path = None

                        # Execute ADC-style generated code
                        try:
                            _record_phase("codegen_request", "start", episode=episode_idx)
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
                            if (
                                isinstance(translated_positions, dict)
                                and "shape_to_place" in translated_positions
                                and "matching_cutout" in translated_positions
                            ):
                                write_json_artifact(
                                    os.path.join(run_dir, "slot_fit_debug.json"),
                                    {{
                                        "scene_object_names": sorted(scene_state.keys()),
                                        "shape_to_place": translated_positions.get("shape_to_place"),
                                        "matching_cutout": translated_positions.get("matching_cutout"),
                                    }},
                                )
                            codegen_success = True
                            _record_phase(
                                "codegen_request",
                                "ok",
                                episode=episode_idx,
                                generated_code_chars=len(generated_code),
                            )
                            print(
                                f"Generated CaP code: {{artifact_paths.get('generated_code', '')}} "
                                f"({{len(generated_code)}} chars)"
                            )

                            if target_grounder is not None:
                                target_grounder.begin_episode(episode_idx)

                            CaPRuntimeContext.configure(
                                sim_skills=skills,
                                detector=detector,
                                robot_cfg=robot_cfg,
                                translated_positions=translated_positions,
                                cameras=cameras,
                                target_grounder=target_grounder,
                            )
                            exec_globals = {{
                                "__name__": "__main__",
                                "positions": translated_positions,
                            }}
                            _record_phase("cap_execute", "start", episode=episode_idx)
                            exec(generated_code, exec_globals)
                            _record_phase("cap_execute", "ok", episode=episode_idx)

                            # Capture env termination + goal state BEFORE move_to_ready
                            # (move_to_ready can knock objects; env auto-resets on success)
                            env_terminated_during_skills = robot_interface.env_terminated
                            env_truncated_during_skills = robot_interface.env_truncated

                            # Verify goal conditions geometrically BEFORE moving away
                            goal = task_doc.get("goal", {{}})
                            pre_retreat_goal_ok = False
                            pre_retreat_goal_details = ""
                            if not env_terminated_during_skills:
                                pre_retreat_goal_ok, pre_retreat_goal_details = _verify_goal_conditions(
                                    detector,
                                    goal,
                                    env=env,
                                    translated_positions=translated_positions,
                                    task_doc=task_doc,
                                    task_description=task_desc,
                                )
                            _record_phase(
                                "geometry_verify",
                                "ok",
                                episode=episode_idx,
                                goal_ok=bool(pre_retreat_goal_ok),
                                details=pre_retreat_goal_details,
                            )

                            # Capture final images BEFORE retreat (for VLM judge)
                            final_images = None
                            if sim_judge is not None:
                                _record_phase("final_judge_capture", "start", episode=episode_idx)
                                final_images = _capture_judge_images(cameras, judge_camera_names)
                                final_image_paths = save_judge_images(run_dir, "final", final_images)
                                _record_phase(
                                    "final_judge_capture",
                                    "ok",
                                    episode=episode_idx,
                                    image_views=sorted(final_images.keys()) if final_images else [],
                                )

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
                            _record_phase(
                                "episode_runtime_error",
                                "error",
                                episode=episode_idx,
                                error=str(e),
                                stage="execution" if codegen_success else "codegen",
                            )
                            traceback.print_exc()
                            execution_ok = False
                            env_terminated_during_skills = robot_interface.env_terminated
                            env_truncated_during_skills = robot_interface.env_truncated
                            pre_retreat_goal_ok = False
                            pre_retreat_goal_details = "execution error"
                            final_images = None
                        finally:
                            if (
                                codegen_success
                                and target_grounder is not None
                                and getattr(target_grounder, "events", None)
                            ):
                                grounding_events = target_grounder.events
                                if grounding_events:
                                    handle_grounding_path = os.path.join(run_dir, "handle_grounding.json")
                                    write_json_artifact(handle_grounding_path, grounding_events)
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
                            vlm_goal_conditions = _compile_goal_conditions(
                                goal,
                                task_description=task_desc,
                                translated_positions=translated_positions,
                            )
                            vlm_relevant_objects = _select_vlm_relevant_objects(
                                vlm_goal_conditions,
                                translated_positions,
                                scene_state,
                            )
                            _record_phase("vlm_judge_episode", "start", episode=episode_idx)
                            verdict = sim_judge.judge_episode(
                                task_description=task_desc,
                                goal_description=goal.get("description", ""),
                                goal_conditions=vlm_goal_conditions,
                                relevant_objects=vlm_relevant_objects,
                                initial_images=initial_images,
                                final_images=final_images,
                                object_positions=translated_positions or scene_state,
                                executed_code=generated_code,
                            )
                            if verdict.get("user_prompt"):
                                judge_prompt_path = os.path.join(run_dir, "judge_prompt.txt")
                                write_text_artifact(judge_prompt_path, verdict["user_prompt"])
                            if verdict.get("prompt_context"):
                                judge_prompt_context_path = os.path.join(
                                    run_dir, "judge_prompt_context.json"
                                )
                                write_json_artifact(
                                    judge_prompt_context_path,
                                    verdict["prompt_context"],
                                )
                            if verdict.get("raw_response"):
                                judge_raw_response_path = os.path.join(
                                    run_dir, "judge_raw_response.txt"
                                )
                                write_text_artifact(
                                    judge_raw_response_path,
                                    verdict["raw_response"],
                                )
                            vlm_success = bool(verdict["success"])
                            _record_phase(
                                "vlm_judge_episode",
                                "ok",
                                episode=episode_idx,
                                prediction=verdict.get("prediction", "N/A"),
                            )
                            print(f"VLM Judge: {{verdict['prediction']}} — {{verdict['reasoning'][:80]}}")
                        elif sim_judge is not None:
                            print("VLM Judge unavailable for episode: missing images or execution failed")
                            _record_phase("vlm_judge_episode", "error", episode=episode_idx, error="missing_images_or_execution_failed")

                        overall_success = geometry_success and (vlm_success if sim_judge is not None else True)
                        success = geometry_success or vlm_success
                        success_basis = None
                        if overall_success:
                            success_basis = "overall"
                        elif geometry_success:
                            success_basis = "geometry_only"
                        elif vlm_success:
                            success_basis = "vlm_only"

                        # End episode (optionally discard every failed episode immediately)
                        discard_failed = bool(cfg.get("discard_failed_episodes", True)) and not success
                        front_video_before_priority = recorder.front_video_priority
                        front_video_priority = 3 if overall_success else (2 if geometry_success else (1 if vlm_success else 0))
                        front_video_success_type = success_basis
                        recorder.end_episode(
                            success=success,
                            discard=discard_failed,
                            front_video_priority=front_video_priority,
                            front_video_success_type=front_video_success_type,
                        )
                        front_video_selected = (
                            recorder.front_video_priority > front_video_before_priority
                            and recorder.front_video_generated
                            and recorder.front_video_episode == episode_idx
                        )

                        episode_result = {{
                            "episode": episode_idx,
                            "success": success,
                            "geometry_success": geometry_success,
                            "vlm_success": vlm_success,
                            "overall_success": overall_success,
                            "success_basis": success_basis,
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
                            "judge_prompt_path": judge_prompt_path,
                            "judge_prompt_context_path": judge_prompt_context_path,
                            "judge_raw_response_path": judge_raw_response_path,
                        }}
                        if front_video_selected:
                            episode_result["front_video_selected"] = True
                        if handle_grounding_path:
                            episode_result["handle_grounding_path"] = handle_grounding_path
                        if recorder.front_video_success_type:
                            episode_result["front_video_success_type"] = recorder.front_video_success_type
                        if blocked_fallback_reason:
                            episode_result["blocked_fallback_reason"] = blocked_fallback_reason
                        if codegen_error:
                            episode_result["codegen_error"] = codegen_error
                        if execution_error:
                            episode_result["execution_error"] = execution_error
                        if recorder.front_video_error:
                            episode_result["front_video_error"] = recorder.front_video_error
                        if verdict is not None:
                            episode_result["judge_prediction"] = verdict.get("prediction", "N/A")
                            episode_result["judge_reasoning"] = verdict.get("reasoning", "")
                        results["episodes"].append(episode_result)
                        results["total_episodes"] += 1
                        if success:
                            results["successful_episodes"] += 1
                            successful_count += 1
                        if geometry_success:
                            results["geometry_successful_episodes"] += 1
                        if vlm_success:
                            results["vlm_successful_episodes"] += 1
                        if overall_success:
                            results["overall_successful_episodes"] += 1

                        _write_results()
                        print(
                            f"Episode {{episode_idx + 1}} complete: "
                            f"success={{'SUCCESS' if success else 'FAILED'}} "
                            f"geometry={{'SUCCESS' if geometry_success else 'FAILED'}} "
                            f"vlm={{'SUCCESS' if vlm_success else 'FAILED'}} "
                            f"overall={{'SUCCESS' if overall_success else 'FAILED'}}"
                            f"{{'(discarded)' if discard_failed else ''}}"
                        )
                        _record_phase(
                            "episode_complete",
                            "ok",
                            episode=episode_idx,
                            success=bool(success),
                            geometry_success=bool(geometry_success),
                            vlm_success=bool(vlm_success),
                            overall_success=bool(overall_success),
                            success_basis=success_basis,
                        )
                        episode_idx += 1

                    print(
                        f"\\nCompleted: success={{results['successful_episodes']}}/{{target_success}} "
                        f"geometry={{results['geometry_successful_episodes']}}/{{target_success}} "
                        f"vlm={{results['vlm_successful_episodes']}}/{{target_success}} "
                        f"overall={{results['overall_successful_episodes']}}/{{results['total_episodes']}} "
                        f"in {{episode_idx}} attempts"
                    )

                except KeyboardInterrupt:
                    print("\\nCollection interrupted by user")
                    results["pipeline_error"] = "interrupted"
                    results["failure_phase"] = _LAST_PHASE
                    results["failure_category"] = "interrupted"
                    _record_phase(_LAST_PHASE, "error", error="interrupted", category="interrupted")
                    _update_startup_diagnostics(
                        pipeline_error="interrupted",
                        failure_phase=_LAST_PHASE,
                        failure_category="interrupted",
                    )
                except Exception as e:
                    print(f"Pipeline error: {{e}}")
                    results["pipeline_error"] = str(e)
                    failure_category = _classify_pipeline_failure(e)
                    results["failure_phase"] = _LAST_PHASE
                    results["failure_category"] = failure_category
                    _write_exception_artifacts(e, _LAST_PHASE, failure_category)
                    traceback.print_exc()
                finally:
                    keep_failed_raw_dataset = bool(cfg.get("keep_failed_raw_dataset", False))
                    discard_failed_episodes = bool(cfg.get("discard_failed_episodes", True))
                    if recorder is not None and recorder.is_recording:
                        try:
                            recorder.end_episode(
                                success=False,
                                discard=discard_failed_episodes,
                            )
                        except Exception as end_err:
                            print(f"WARNING: Failed to close active episode: {{end_err}}")
                            traceback.print_exc()
                            results.setdefault("cleanup_errors", []).append(
                                f"end_episode failed: {{end_err}}"
                            )
                    dataset_path = None
                    if recorder is not None:
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
                    results["discard_failed_episodes"] = discard_failed_episodes
                    results["front_video_generated"] = bool(recorder and recorder.front_video_generated)
                    results["front_video_path"] = (
                        str(recorder.front_video_path) if recorder and recorder.front_video_path is not None else None
                    )
                    results["front_video_camera_name"] = (
                        recorder.front_video_camera_name if recorder else cfg.get("front_video_camera_name", "front")
                    )
                    results["front_video_episode"] = (
                        int(recorder.front_video_episode) if recorder and recorder.front_video_episode is not None else None
                    )
                    results["front_video_success_type"] = (
                        recorder.front_video_success_type if recorder else None
                    )
                    results["front_video_error"] = (
                        recorder.front_video_error if recorder else None
                    )
                    results["phase_trace_path"] = PHASE_TRACE_PATH
                    results["startup_diagnostics_path"] = STARTUP_DIAGNOSTICS_PATH
                    results["heartbeat_path"] = HEARTBEAT_PATH
                    results["stderr_tail_path"] = STDERR_TAIL_PATH
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
                    if (
                        dataset_path
                        and not keep_failed_raw_dataset
                        and int(results.get("successful_episodes", 0)) <= 0
                    ):
                        try:
                            shutil.rmtree(dataset_path, ignore_errors=False)
                            dataset_path = None
                            results["dataset_path"] = None
                            results["raw_dataset"] = None
                            results["failed_raw_dataset_removed"] = True
                            results["failed_raw_dataset_cleanup_reason"] = "zero_success_dataset_removed"
                        except Exception as cleanup_err:
                            print(f"WARNING: Failed to remove failed raw dataset: {{cleanup_err}}")
                            traceback.print_exc()
                            results.setdefault("cleanup_errors", []).append(
                                f"failed raw cleanup failed: {{cleanup_err}}"
                            )
                    results["pipeline_completed"] = not bool(
                        results.get("pipeline_error") or results.get("failure_category")
                    )
                    results["success"] = results["pipeline_completed"]
                    _update_startup_diagnostics(
                        pipeline_completed=bool(results["pipeline_completed"]),
                        failure_phase=results.get("failure_phase"),
                        failure_category=results.get("failure_category"),
                        dataset_path=dataset_path,
                    )
                    _record_phase(
                        "pipeline_finalize",
                        "ok" if results["pipeline_completed"] else "error",
                        dataset_path=dataset_path,
                        pipeline_completed=bool(results["pipeline_completed"]),
                    )

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
                    if env is not None:
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
        # ``conda run`` + ``isaaclab.sh`` can inherit ``TERM=dumb`` in non-interactive
        # shells, which breaks shell startup helpers inside IsaacLab. Force a sane
        # terminal type so batch collection behaves the same as a normal user shell.
        if not env.get("TERM") or env.get("TERM") == "dumb":
            env["TERM"] = "xterm-256color"
        if self.config.env_headless:
            # Headless rendering: remove DISPLAY (forces EGL instead of GLX)
            # and force NVIDIA-only Vulkan ICD to avoid non-NVIDIA driver issues.
            # NOTE: First-run shader compilation can take 5-10 minutes.
            env.pop("DISPLAY", None)
            env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

        try:
            self._last_execution_timed_out = False
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(self.env_dir),
                preexec_fn=os.setsid,
            )

            output_lines = []
            start_time = time.time()

            for line in proc.stdout:
                line = line.rstrip()
                print(line)
                output_lines.append(line)

                # Check timeout
                if time.time() - start_time > self.config.execution_timeout:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    self._last_execution_timed_out = True
                    logger.error("Execution timeout reached")
                    break

            proc.wait()
            # Safety: kill any remaining processes in the group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
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

    def _convert_to_lerobot(self, raw_data_dir: Path) -> Path | None:
        """Convert raw dataset to LeRobot v3.0 format (post-processing)."""
        try:
            from .sim_recorder import convert_to_lerobot

            lerobot_path = convert_to_lerobot(
                raw_dataset_dir=str(raw_data_dir),
                repo_id=self.config.dataset_repo_id,
            )
            logger.info(f"LeRobot dataset created at: {lerobot_path}")
            return Path(lerobot_path).expanduser().resolve()
        except ImportError:
            logger.warning(
                "LeRobot conversion skipped (lerobot package not available). "
                "Raw data saved at: {raw_data_dir}"
            )
        except Exception as e:
            logger.error(f"LeRobot conversion failed: {e}")
        return None


def run_batch(
    task_dir: str,
    config: Optional[DataCollectionConfig] = None,
    episodes_per_task: int = 10,
    auto_convert_to_lerobot: bool = True,
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
                auto_convert_to_lerobot=auto_convert_to_lerobot,
            )
            result = pipeline.run()
            results.append(result)
        except Exception as e:
            logger.error(f"Failed for {yaml_path}: {e}")
            results.append({"success": False, "error": str(e), "yaml": str(yaml_path)})

    return results
