#!/usr/bin/env python3
"""Run a deterministic motion sanity check against a generated IsaacLab env.

This script evaluates whether the current motion backend can:
1. Move the robot to its ready pose
2. Reach a pre-grasp approach pose
3. Pick an object
4. Optionally place it at the task target

It writes a compact JSON report under ``outputs/motion_eval`` by default.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.app import AppLauncher


def _default_output_root() -> Path:
    return PROJECT_ROOT / "outputs" / "motion_eval"


parser = argparse.ArgumentParser(description="Check motion-library tracking in a generated IsaacLab env.")
parser.add_argument("--env-dir", required=True, help="Path to generated IsaacLab env directory")
parser.add_argument("--task-yaml", required=True, help="Task YAML used for detector/goal lookup")
parser.add_argument("--pick-object", default=None, help="Override pick object name")
parser.add_argument("--output-dir", default=str(_default_output_root()), help="Root directory for JSON reports")
parser.add_argument(
    "--so101-calibrate",
    action="store_true",
    help="Sweep SO-101 grasp depth/lateral bias candidates and save debug images.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True
if not hasattr(args, "num_envs") or args.num_envs is None:
    args.num_envs = 1
args.enable_cameras = bool(args.so101_calibrate)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

ENV_DIR = Path(args.env_dir).resolve()
TASK_YAML = Path(args.task_yaml).resolve()
sys.path.insert(0, str(ENV_DIR))

generated_env_cfg = importlib.import_module("env_cfg")

from src.agent.common.robot_names import normalize_robot_name
from src.agent.common.task_docs import load_task_document
from src.agent.data_collection.config import load_robot_config
from src.agent.data_collection.sim_camera import SceneCameraManager, inject_cameras_into_scene
from src.agent.data_collection.sim_detector import SimDetector
from src.agent.data_collection.sim_robot_interface import SimRobotInterface
from src.agent.data_collection.sim_skills import SimSkills, create_ik_solver


SO101_DEPTH_SWEEP = [0.032, 0.036, 0.040, 0.044]
SO101_LATERAL_BIAS_SWEEP = [-0.008, -0.004, 0.0, 0.004, 0.008]


def _detect_robot_name(task_yaml: Path, task_doc: dict[str, Any]) -> str:
    parts = task_yaml.parts
    if "tasks" in parts:
        idx = parts.index("tasks")
        if idx + 1 < len(parts):
            robot_name = parts[idx + 1]
            normalized = normalize_robot_name(robot_name, default=robot_name)
            if normalized:
                return normalized

    for asset in task_doc.get("assets", []):
        robot_type = asset.get("robot_type")
        if robot_type:
            normalized = normalize_robot_name(robot_type, default=robot_type)
            if normalized:
                return normalized
    raise RuntimeError(f"Could not determine robot for task '{task_yaml}'")


def _apply_runtime_patches(env_cfg, robot_cfg) -> None:
    env_cfg.episode_length_s = 600.0
    if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
        env_cfg.scene.num_envs = int(args.num_envs)

    if hasattr(env_cfg, "terminations"):
        for term_name in list(vars(env_cfg.terminations)):
            if term_name.startswith("_"):
                continue
            term = getattr(env_cfg.terminations, term_name, None)
            if term is None:
                continue
            if hasattr(term, "time_out") and term.time_out:
                continue
            setattr(env_cfg.terminations, term_name, None)

    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "arm_action"):
        env_cfg.actions.arm_action.scale = 1.0
        env_cfg.actions.arm_action.use_default_offset = False

    if (
        robot_cfg.ready_pose
        and hasattr(env_cfg.scene, "robot")
        and hasattr(env_cfg.scene.robot, "init_state")
    ):
        env_cfg.scene.robot.init_state.joint_pos = dict(robot_cfg.ready_pose)

    if hasattr(env_cfg.scene, "robot") and hasattr(env_cfg.scene.robot, "actuators"):
        for act_name, actuator in env_cfg.scene.robot.actuators.items():
            if "hand" in act_name or "finger" in act_name or "gripper" in act_name:
                continue
            orig_s = getattr(actuator, "stiffness", 80.0)
            orig_d = getattr(actuator, "damping", 4.0)
            orig_s = float(orig_s) if isinstance(orig_s, (int, float)) else 80.0
            orig_d = float(orig_d) if isinstance(orig_d, (int, float)) else 4.0
            new_s = min(orig_s * 5.0, 5000.0)
            new_d = min(orig_d * 5.0, 400.0)
            if robot_cfg.name == "so101":
                new_s = max(new_s, 300.0)
                new_d = max(new_d, 30.0)
            actuator.stiffness = new_s
            actuator.damping = new_d
            actuator.effort_limit = 1e9
            actuator.effort_limit_sim = 1e9

        grasp_stiffness = robot_cfg.gripper_grasp_stiffness
        if grasp_stiffness > 0:
            for act_name, actuator in env_cfg.scene.robot.actuators.items():
                act_name_lower = act_name.lower()
                if (
                    "gripper_drive" in act_name_lower
                    or "gripper_finger" in act_name_lower
                    or "gripper" in act_name_lower
                    or "hand" in act_name_lower
                    or "finger" in act_name_lower
                ):
                    actuator.stiffness = grasp_stiffness
                    actuator.damping = grasp_stiffness * 0.05
                    actuator.effort_limit_sim = max(100.0, grasp_stiffness * 0.2)
                elif "gripper_passive" in act_name_lower or "passive" in act_name_lower:
                    actuator.stiffness = grasp_stiffness * 0.5
                    actuator.damping = grasp_stiffness * 0.025

    if hasattr(env_cfg.scene, "robot") and hasattr(env_cfg.scene.robot, "spawn"):
        spawn = env_cfg.scene.robot.spawn
        if hasattr(spawn, "rigid_props") and spawn.rigid_props is not None:
            spawn.rigid_props.disable_gravity = True
        if hasattr(spawn, "articulation_props") and spawn.articulation_props is not None:
            spawn.articulation_props.solver_position_iteration_count = 32
            spawn.articulation_props.solver_velocity_iteration_count = 2
            spawn.articulation_props.enabled_self_collisions = False


def _warmup_env(env: ManagerBasedRLEnv, robot_cfg) -> None:
    action_dim = env.action_space.shape[-1]
    warmup_action = torch.zeros(env.num_envs, action_dim, device=env.device)
    robot = env.scene["robot"]
    n_arm = min(robot_cfg.arm_dofs, robot.data.joint_pos.shape[1], action_dim)
    warmup_action[0, :n_arm] = robot.data.joint_pos[0, :n_arm].clone()
    if action_dim > n_arm:
        warmup_action[0, n_arm] = 1.0
    for _ in range(10):
        env.step(warmup_action)


def _infer_pick_object(task_doc: dict[str, Any], category: str) -> str:
    rigid_names = [
        asset.get("name", "")
        for asset in task_doc.get("assets", [])
        if asset.get("type") == "rigid"
    ]
    if category == "stack":
        cubes = sorted(name for name in rigid_names if name.startswith("cube_"))
        if len(cubes) >= 2:
            return cubes[1]
    for name in rigid_names:
        if any(token in name for token in ("target", "marker", "tray", "zone", "cabinet", "drawer")):
            continue
        return name
    raise RuntimeError("Could not infer a pick object from task YAML")


def _stack_place_target(task_doc: dict[str, Any], detector: SimDetector) -> np.ndarray:
    stackable = sorted(
        asset.get("name", "")
        for asset in task_doc.get("assets", [])
        if asset.get("type") == "rigid" and asset.get("name", "").startswith("cube_")
    )
    if len(stackable) < 2:
        raise RuntimeError("Stack task needs at least two stackable cubes")
    base_name = stackable[0]
    base_pos = detector.get_object_position(base_name)
    height_diff = float(task_doc.get("goal", {}).get("success_criteria", {}).get("height_diff", 0.05))
    target = base_pos.copy()
    target[2] += height_diff
    return target


def _pick_place_target(task_doc: dict[str, Any], detector: SimDetector) -> np.ndarray:
    conditions = task_doc.get("goal", {}).get("conditions", [])
    target_name = None
    for cond in conditions:
        relation = cond.get("relation")
        if relation == "at_position":
            target_name = cond.get("target")
            break
    if target_name is None:
        for asset in task_doc.get("assets", []):
            name = asset.get("name", "")
            if "target" in name or "marker" in name:
                target_name = name
                break
    if target_name is None:
        raise RuntimeError("Could not infer place target from task YAML")
    return detector.get_object_position(target_name)


def _build_report_dir(output_root: Path, task_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{task_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _load_generated_env_cfg() -> ManagerBasedRLEnvCfg:
    candidates = []
    for name, value in vars(generated_env_cfg).items():
        if name == "ManagerBasedRLEnvCfg":
            continue
        if not name.endswith("EnvCfg"):
            continue
        if not isinstance(value, type):
            continue
        if not issubclass(value, ManagerBasedRLEnvCfg):
            continue
        if value.__module__ != generated_env_cfg.__name__:
            continue
        candidates.append(value)
    if not candidates:
        raise RuntimeError(f"No generated ManagerBasedRLEnvCfg subclass found in {ENV_DIR / 'env_cfg.py'}")
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous env cfg classes in {ENV_DIR / 'env_cfg.py'}: {[cls.__name__ for cls in candidates]}"
        )
    return candidates[0]()


def _make_runtime_bundle(env: ManagerBasedRLEnv, robot_cfg, task_doc: dict[str, Any]):
    robot_interface = SimRobotInterface(env, robot_cfg, env_idx=0)
    detector = SimDetector(env, task_doc, env_idx=0)
    robot_base_pos = env.scene["robot"].data.root_pos_w[0].cpu().numpy()
    ik_solver = create_ik_solver(robot_cfg, robot_interface=robot_interface)
    skills = SimSkills(
        robot_interface=robot_interface,
        robot_cfg=robot_cfg,
        detector=detector,
        ik_solver=ik_solver,
        base_offset=robot_base_pos,
    )
    return robot_interface, detector, skills


def _reset_env_for_trial(env: ManagerBasedRLEnv, robot_cfg, seed: int = 0) -> None:
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    _warmup_env(env, robot_cfg)


def _save_captured_images(camera_manager: SceneCameraManager | None, output_dir: Path, label: str) -> dict[str, str]:
    if not camera_manager:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for camera_name, image in camera_manager.capture_all().items():
        if image is None:
            continue
        path = output_dir / f"{label}_{camera_name}.png"
        Image.fromarray(image).save(path)
        saved[camera_name] = str(path)
    return saved


def _get_primary_finger_angle(robot_interface: SimRobotInterface) -> float | None:
    if not robot_interface._finger_indices:
        return None
    finger_state = robot_interface.articulation.data.joint_pos[
        robot_interface.env_idx, robot_interface._finger_indices
    ].cpu().numpy()
    if len(finger_state) == 0:
        return None
    return float(finger_state[0])


def _is_empty_close(robot_cfg, finger_angle: float | None, z_delta: float) -> bool:
    if finger_angle is None:
        return False
    open_cfg = robot_cfg.gripper_open_position
    close_cfg = robot_cfg.gripper_close_position
    open_pos = float(open_cfg[0] if isinstance(open_cfg, (list, tuple)) else open_cfg)
    close_pos = float(close_cfg[0] if isinstance(close_cfg, (list, tuple)) else close_cfg)
    near_close = abs(finger_angle - close_pos) < max(0.1, abs(open_pos - close_pos) * 0.15)
    return bool(near_close and z_delta < 0.01)


def _run_single_pick_trial(
    env: ManagerBasedRLEnv,
    base_robot_cfg,
    task_doc: dict[str, Any],
    pick_object: str,
    *,
    camera_manager: SceneCameraManager | None,
    report_dir: Path,
    depth: float | None = None,
    lateral_bias: float | None = None,
) -> dict[str, Any]:
    robot_cfg = deepcopy(base_robot_cfg)
    if depth is not None:
        robot_cfg.ee_finger_offset = float(depth)
        if robot_cfg.ee_frame_offset_position:
            offset = list(robot_cfg.ee_frame_offset_position)
            offset[0] = float(depth)
            robot_cfg.ee_frame_offset_position = offset
    if lateral_bias is not None:
        robot_cfg.grasp_lateral_bias = float(lateral_bias)

    _reset_env_for_trial(env, robot_cfg, seed=0)
    robot_interface, detector, skills = _make_runtime_bundle(env, robot_cfg, task_doc)

    obj_pos = detector.get_object_position(pick_object)
    pre_pos = obj_pos.copy()
    pre_z = float(pre_pos[2])
    ready_ok = skills.move_to_ready(duration=1.0)
    grasp_pos, approach_pos = skills._compute_pick_targets(obj_pos, approach_offset=0.10)
    approach_ok = skills._move_palm_down(approach_pos, duration=1.0)
    image_paths = {"approach": _save_captured_images(camera_manager, report_dir, "approach")}

    if skills.cfg.arm_dofs == 6:
        skills._move_cartesian_steps(approach_pos, grasp_pos, max_step=0.015)
    else:
        skills._move_palm_down(grasp_pos)

    ee_pos, _ = robot_interface.read_ee_pose()
    grasp_error = float(np.linalg.norm(ee_pos - grasp_pos))
    if grasp_error > 0.015:
        skills._move_palm_down(grasp_pos)

    skills.gripper_close(duration=robot_cfg.gripper_close_duration)
    finger_angle = _get_primary_finger_angle(robot_interface)
    image_paths["post_close"] = _save_captured_images(camera_manager, report_dir, "post_close")

    if skills.cfg.arm_dofs == 6:
        ee_pos, _ = robot_interface.read_ee_pose()
        skills._move_cartesian_steps(
            ee_pos,
            approach_pos,
            max_step=0.015,
            fixed_joints=[3, 4, 5],
        )
    elif skills.cfg.arm_dofs <= 5 and skills._orientation_lock_joints:
        ee_pos, _ = robot_interface.read_ee_pose()
        skills._move_cartesian_steps(
            ee_pos,
            approach_pos,
            max_step=0.01,
            fixed_joints=list(skills._orientation_lock_joints),
        )
    else:
        skills._move_palm_down(approach_pos)

    lifted, z_delta = skills._verify_object_lifted(pick_object, pre_z)
    image_paths["post_lift"] = _save_captured_images(camera_manager, report_dir, "post_lift")
    post_pos = detector.get_object_position(pick_object)
    _, ee_error = skills._verify_ee_position(approach_pos)
    xy_shift = float(np.linalg.norm(post_pos[:2] - pre_pos[:2]))

    return {
        "depth": float(robot_cfg.ee_finger_offset),
        "lateral_bias": float(robot_cfg.grasp_lateral_bias),
        "ready_ok": bool(ready_ok),
        "approach_ok": bool(approach_ok),
        "pick_ok": bool(lifted),
        "ee_error_m": float(ee_error),
        "grasp_error_m": float(grasp_error),
        "lift_z_delta_m": float(z_delta),
        "cube_xy_shift_m": xy_shift,
        "finger_angle": finger_angle,
        "empty_close": _is_empty_close(robot_cfg, finger_angle, z_delta),
        "images": image_paths,
        "terminated": bool(robot_interface.env_terminated),
        "truncated": bool(robot_interface.env_truncated),
    }


def _score_calibration_trial(trial: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(trial["lift_z_delta_m"]),
        -float(trial["cube_xy_shift_m"]),
        -float(trial["ee_error_m"]),
    )


def _run_so101_calibration(
    env: ManagerBasedRLEnv,
    robot_cfg,
    task_doc: dict[str, Any],
    task_name: str,
    pick_object: str,
    report_dir: Path,
    camera_manager: SceneCameraManager | None,
) -> dict[str, Any]:
    trials: list[dict[str, Any]] = []
    for depth in SO101_DEPTH_SWEEP:
        for bias in SO101_LATERAL_BIAS_SWEEP:
            trial_dir = report_dir / f"trial_depth_{depth:.3f}_bias_{bias:+.3f}"
            trial = _run_single_pick_trial(
                env,
                robot_cfg,
                task_doc,
                pick_object,
                camera_manager=camera_manager,
                report_dir=trial_dir,
                depth=depth,
                lateral_bias=bias,
            )
            trial["score"] = _score_calibration_trial(trial)
            trials.append(trial)
            print(
                json.dumps(
                    {
                        "depth": depth,
                        "bias": bias,
                        "lift_z_delta_m": trial["lift_z_delta_m"],
                        "ee_error_m": trial["ee_error_m"],
                        "cube_xy_shift_m": trial["cube_xy_shift_m"],
                        "empty_close": trial["empty_close"],
                    }
                )
            )

    valid_trials = [trial for trial in trials if not trial["empty_close"]]
    ranked = sorted(valid_trials or trials, key=_score_calibration_trial, reverse=True)
    best_trial = ranked[0] if ranked else None

    report = {
        "task": task_name,
        "robot": robot_cfg.name,
        "env_dir": str(ENV_DIR),
        "task_yaml": str(TASK_YAML),
        "mode": "so101_calibration",
        "pick_object": pick_object,
        "depth_candidates": SO101_DEPTH_SWEEP,
        "lateral_bias_candidates": SO101_LATERAL_BIAS_SWEEP,
        "best_trial": best_trial,
        "trials": trials,
    }
    report_path = report_dir / "motion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")
    return report


def _run_standard_motion_check(
    env: ManagerBasedRLEnv,
    robot_cfg,
    task_doc: dict[str, Any],
    task_name: str,
    category: str,
    pick_object: str,
    report_dir: Path,
) -> dict[str, Any]:
    _reset_env_for_trial(env, robot_cfg, seed=0)
    robot_interface, detector, skills = _make_runtime_bundle(env, robot_cfg, task_doc)

    obj_pos = detector.get_object_position(pick_object)
    pre_z = float(obj_pos[2])
    grasp_offset = 0.0 if robot_cfg.ee_frame_offset_position else robot_cfg.ee_finger_offset
    approach_pos = obj_pos.copy()
    approach_pos[2] += grasp_offset + 0.10

    ready_ok = skills.move_to_ready(duration=1.0)
    approach_ok = skills._move_palm_down(approach_pos, duration=1.0)
    pick_ok = skills.execute_pick(pick_object)
    post_pick_pos = detector.get_object_position(pick_object)
    lift_delta = float(post_pick_pos[2] - pre_z)

    place_ok = None
    last_target = approach_pos
    if pick_ok:
        if category == "stack":
            place_target = _stack_place_target(task_doc, detector)
            place_ok = skills.execute_place(place_target, _placed_object=pick_object)
            last_target = place_target
        elif category == "pick_place":
            place_target = _pick_place_target(task_doc, detector)
            place_ok = skills.execute_place(place_target, _placed_object=pick_object)
            last_target = place_target

    _, ee_error = skills._verify_ee_position(last_target)
    overall_pass = bool(ready_ok and approach_ok and pick_ok and (place_ok is not False))

    report = {
        "task": task_name,
        "robot": robot_cfg.name,
        "env_dir": str(ENV_DIR),
        "task_yaml": str(TASK_YAML),
        "ik_backend": getattr(skills.ik, "backend_name", "unknown"),
        "ready_ok": bool(ready_ok),
        "approach_ok": bool(approach_ok),
        "pick_ok": bool(pick_ok),
        "place_ok": None if place_ok is None else bool(place_ok),
        "ee_error_m": float(ee_error),
        "lift_z_delta_m": float(lift_delta),
        "terminated": bool(robot_interface.env_terminated),
        "truncated": bool(robot_interface.env_truncated),
        "overall_pass": overall_pass,
    }

    report_path = report_dir / "motion_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {report_path}")
    return report


def main():
    task_doc = load_task_document(TASK_YAML)
    robot_name = _detect_robot_name(TASK_YAML, task_doc)
    robot_cfg = load_robot_config(robot_name)
    category = TASK_YAML.parent.name
    task_name = task_doc.get("task", {}).get("name", TASK_YAML.stem)
    pick_object = args.pick_object or _infer_pick_object(task_doc, category)

    env_cfg = _load_generated_env_cfg()
    _apply_runtime_patches(env_cfg, robot_cfg)

    camera_attr_names: list[str] = []
    if args.so101_calibrate:
        camera_attr_names = inject_cameras_into_scene(env_cfg, robot_cfg, camera_names=["wrist", "top"])

    env = ManagerBasedRLEnv(cfg=env_cfg)
    try:
        env.reset()
        _warmup_env(env, robot_cfg)
        report_dir = _build_report_dir(Path(args.output_dir), task_name)
        camera_manager = SceneCameraManager(env, camera_attr_names) if camera_attr_names else None

        if robot_cfg.name == "so101":
            from src.agent.data_collection.so101_patches import ensure_so101_claw_collisions

            added_paths = ensure_so101_claw_collisions(env.sim.stage, env_idx=0, verbose=True)
            print(f"Ensured {len(added_paths)} SO-101 claw collision proxies")

        if args.so101_calibrate:
            if robot_name != "so101":
                raise RuntimeError("--so101-calibrate is only valid for SO-101 tasks")
            _run_so101_calibration(
                env,
                robot_cfg,
                task_doc,
                task_name,
                pick_object,
                report_dir,
                camera_manager,
            )
        else:
            _run_standard_motion_check(
                env,
                robot_cfg,
                task_doc,
                task_name,
                category,
                pick_object,
                report_dir,
            )
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
