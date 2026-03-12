"""Deterministic IsaacLab template for AssemblingKits scene tasks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_assembling_kits_template(task_doc: dict, robot: str) -> dict[str, str]:
    """Build a checked IsaacLab template for the AssemblingKits family."""
    task_name = task_doc.get("task", {}).get("name", "AssemblingKits")
    env_class = f"{task_name}EnvCfg"
    scene_class = f"{task_name}SceneCfg"

    assets = {asset["name"]: asset for asset in task_doc.get("assets", []) if asset.get("name")}
    robot_asset = next(asset for asset in task_doc.get("assets", []) if asset.get("type") == "articulation")
    camera_cfg = task_doc.get("camera", {})
    active_shape_name = _find_active_shape_name(task_doc)

    scene_blocks = [_build_asset_block(asset) for asset in assets.values() if asset.get("name") != robot_asset.get("name")]
    scene_assets = "\n\n".join(scene_blocks)
    robot_imports, robot_setup = _build_robot_setup(robot, robot_asset)
    randomization_event = ""
    if active_shape_name and active_shape_name in assets:
        randomization_event = _build_reset_event_block(assets[active_shape_name])
    post_env_setup = _build_post_env_setup_block(assets.values())

    env_cfg = f'''# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from copy import deepcopy
import math
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import schemas as sim_schemas
from isaaclab.sim.spawners.from_files import from_files as from_files_impl
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.utils import clone
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from pxr import Gf, Usd, UsdGeom, UsdShade
{robot_imports}
import mdp as mdp


def _quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def randomize_static_asset_pose(env, env_ids, prim_path, base_pos, base_quat, position_range, yaw_range=None):
    del env, env_ids
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Could not find prim to randomize: {{prim_path}}")

    xformable = UsdGeom.Xformable(prim)
    ordered_ops = {{op.GetOpName(): op for op in xformable.GetOrderedXformOps()}}
    translate_op = ordered_ops.get("xformOp:translate") or xformable.AddTranslateOp()
    orient_op = ordered_ops.get("xformOp:orient") or xformable.AddOrientOp()

    dx = random.uniform(*position_range.get("x", (0.0, 0.0)))
    dy = random.uniform(*position_range.get("y", (0.0, 0.0)))
    dz = random.uniform(*position_range.get("z", (0.0, 0.0)))
    pos = (base_pos[0] + dx, base_pos[1] + dy, base_pos[2] + dz)

    quat = tuple(base_quat)
    if yaw_range is not None:
        yaw = random.uniform(*yaw_range)
        half_yaw = 0.5 * yaw
        delta_quat = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        quat = _quat_multiply(quat, delta_quat)

    translate_op.Set(Gf.Vec3f(*[float(value) for value in pos]))
    orient_op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))


def _apply_triangle_mesh_collision(root_prim_path):
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Could not find triangle-collision root prim: {{root_prim_path}}")

    mesh_cfg = sim_schemas.TriangleMeshPropertiesCfg()
    applied = 0
    for prim in root_prim.GetChildren():
        if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
            sim_schemas.define_mesh_collision_properties(prim.GetPath().pathString, mesh_cfg, stage=stage)
            applied += 1

    if applied == 0:
        for prim in Usd.PrimRange(root_prim):
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Mesh):
                sim_schemas.define_mesh_collision_properties(prim.GetPath().pathString, mesh_cfg, stage=stage)
                applied += 1

    if applied == 0:
        raise RuntimeError(f"Could not find mesh children under {{root_prim_path}} for triangle collision setup")
    return applied


def _resolve_single_mesh_child(root_prim_path):
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    if not root_prim or not root_prim.IsValid():
        raise RuntimeError(f"Could not find rigid-shape root prim: {{root_prim_path}}")

    mesh_prims = []
    mesh_with_material = []
    for prim in Usd.PrimRange(root_prim):
        if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Mesh):
            continue
        mesh_prims.append(prim)
        binding = UsdShade.MaterialBindingAPI(prim).GetDirectBinding()
        if str(binding.GetMaterialPath()):
            mesh_with_material.append(prim)

    if len(mesh_with_material) == 1:
        return mesh_with_material[0]
    if len(mesh_prims) == 1:
        return mesh_prims[0]

    candidate_paths = [prim.GetPath().pathString for prim in mesh_prims]
    raise RuntimeError(
        f"Could not uniquely resolve rigid-shape mesh under {{root_prim_path}}. Candidates: {{candidate_paths}}"
    )


@clone
def spawn_usd_with_child_physics(prim_path, cfg, translation=None, orientation=None, **kwargs):
    cfg_copy = deepcopy(cfg)
    rigid_props = getattr(cfg, "rigid_props", None)
    collision_props = getattr(cfg, "collision_props", None)
    mass_props = getattr(cfg, "mass_props", None)

    cfg_copy.rigid_props = None
    cfg_copy.collision_props = None
    cfg_copy.mass_props = None

    prim = from_files_impl._spawn_from_usd_file(
        prim_path,
        cfg_copy.usd_path,
        cfg_copy,
        translation,
        orientation,
        **kwargs,
    )

    stage = get_current_stage()
    target_prim = _resolve_single_mesh_child(prim_path)
    target_path = target_prim.GetPath().pathString
    if rigid_props is not None:
        sim_schemas.define_rigid_body_properties(target_path, rigid_props, stage=stage)
    if collision_props is not None:
        sim_schemas.define_collision_properties(target_path, collision_props, stage=stage)
    if mass_props is not None:
        sim_schemas.define_mass_properties(target_path, mass_props, stage=stage)
    sim_schemas.define_mesh_collision_properties(
        target_path,
        sim_schemas.ConvexHullPropertiesCfg(),
        stage=stage,
    )
    print(f"Applied child rigid-body physics: root={{prim_path}}, target={{target_path}}")
    return prim


def post_env_setup(env):
{_indent(post_env_setup, 4) if post_env_setup else "    return None"}


@configclass
class {scene_class}(InteractiveSceneCfg):
    """Scene-only AssemblingKits layout."""

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos={_tuple(task_doc.get("scene", {}).get("ground", {}).get("position", [0.0, 0.0, -1.05]))}),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color={_tuple(task_doc.get("scene", {}).get("lighting", {}).get("color", [0.75, 0.75, 0.75]))},
            intensity={float(task_doc.get("scene", {}).get("lighting", {}).get("intensity", 3000.0))},
        ),
    )

    robot: ArticulationCfg = MISSING

{_indent(scene_assets, 4)}


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
{_indent(randomization_event, 4) if randomization_event else ""}


@configclass
class RewardsCfg:
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class {env_class}(ManagerBasedRLEnvCfg):
    scene: {scene_class} = {scene_class}(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.decimation = {int(task_doc.get("simulation", {}).get("decimation", 2))}
        self.episode_length_s = {float(task_doc.get("simulation", {}).get("episode_length", 10.0))}
        self.sim.dt = {float(task_doc.get("simulation", {}).get("timestep", 0.01))}
        self.sim.render_interval = {int(task_doc.get("simulation", {}).get("render_interval", 2))}
        self.sim.physx.bounce_threshold_velocity = {float(task_doc.get("simulation", {}).get("physx", {}).get("bounce_threshold", 0.01))}
        self.sim.physx.friction_correlation_distance = {float(task_doc.get("simulation", {}).get("physx", {}).get("friction_correlation_distance", 0.00625))}
        try:
            self.sim.physx.gpu_max_rigid_contact_count = {int(task_doc.get("simulation", {}).get("physx", {}).get("gpu_max_rigid_contact_count", 8388608))}
        except Exception:
            pass

{_indent(robot_setup, 8)}
'''

    run_env = f'''"""Generated IsaacLab environment runner."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from env_cfg import {env_class}


def main():
    env_cfg = {env_class}()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    post_setup = globals().get("post_env_setup")
    if callable(post_setup):
        post_setup(env)

    env.reset()
    for _ in range(10):
        actions = torch.zeros_like(env.action_manager.action)
        env.step(actions)

    env.close()
    print("SUCCESS: Environment validated", flush=True)
    marker = os.environ.get("ISAACLAB_SUCCESS_MARKER")
    if marker:
        Path(marker).write_text("SUCCESS")
    simulation_app.close()


if __name__ == "__main__":
    main()
'''

    capture_scene = f'''"""Capture a reference scene image for AssemblingKits."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, {str(PROJECT_ROOT)!r})

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from PIL import Image
from types import SimpleNamespace
from isaaclab.envs import ManagerBasedRLEnv

from env_cfg import {env_class}
from src.data_collection.config import CameraConfig
from src.data_collection.sim_camera import SceneCameraManager, inject_cameras_into_scene


def main():
    env_cfg = {env_class}()
    env_cfg.scene.num_envs = 1
    camera_cfg = CameraConfig(
        name="scene",
        cam_type="fixed",
        position={repr(list(camera_cfg.get("position", [0.3, 0.3, 0.5])))},
        target={repr(list(camera_cfg.get("target", [0.45, -0.01, 0.01])))},
        resolution={_tuple(camera_cfg.get("resolution", [1280, 720]))},
    )
    camera_attr_names = inject_cameras_into_scene(
        env_cfg,
        SimpleNamespace(cameras={{"scene": camera_cfg}}),
        camera_names=["scene"],
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    post_setup = globals().get("post_env_setup")
    if callable(post_setup):
        post_setup(env)
    env.reset()
    cameras = SceneCameraManager(env, camera_attr_names)

    for _ in range(4):
        actions = torch.zeros_like(env.action_manager.action)
        env.step(actions)

    image = cameras.capture_all().get("scene")
    if image is None:
        raise RuntimeError("Failed to capture AssemblingKits scene image")

    out_path = Path(__file__).resolve().parent / "scene_capture.png"
    Image.fromarray(image).save(out_path)
    print(f"SCENE_CAPTURE: {{out_path}}", flush=True)

    marker = os.environ.get("ISAACLAB_SUCCESS_MARKER")
    if marker:
        Path(marker).write_text("SCENE_CAPTURE")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
'''

    return {
        "env_cfg.py": env_cfg.strip() + "\n",
        "run_env.py": run_env.strip() + "\n",
        "capture_scene.py": capture_scene.strip() + "\n",
    }


def _build_asset_block(asset: dict[str, Any]) -> str:
    if asset.get("physics", {}).get("rigid_body"):
        return _build_rigid_asset_block(asset)
    return _build_static_asset_block(asset)


def _build_static_asset_block(asset: dict[str, Any]) -> str:
    spawn_lines = [
        f"usd_path={_python_path_literal(asset['asset_path'])},",
        f"scale={_tuple(asset.get('scale', [1.0, 1.0, 1.0]))},",
        f'semantic_tags=[("class", "{asset["name"]}")],',
    ]
    color = asset.get("color")
    if color:
        spawn_lines.append(f"visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color={_tuple(color)}),")
    if asset.get("physics", {}).get("collision"):
        spawn_lines.append(
            "collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),"
        )

    note_lines = []
    if asset.get("physics", {}).get("collision_mesh") == "triangle":
        note_lines.append("    # YAML requests triangle-mesh collision for tray cutouts.")

    return f'''{asset["name"]} = AssetBaseCfg(
    prim_path={asset.get("prim_path", f"/World/{asset['name']}")!r},
    init_state=AssetBaseCfg.InitialStateCfg(
        pos={_tuple(asset.get("position", [0.0, 0.0, 0.0]))},
        rot={_tuple(asset.get("rotation", [1.0, 0.0, 0.0, 0.0]))},
    ),
{chr(10).join(note_lines)}
    spawn=UsdFileCfg(
{_indent(chr(10).join(spawn_lines), 8)}
    ),
)'''


def _build_reset_event_block(asset: dict[str, Any]) -> str:
    randomize = asset.get("randomize", {})
    position = randomize.get("position", {}) if isinstance(randomize, dict) else {}
    orientation = randomize.get("orientation", {}) if isinstance(randomize, dict) else {}
    pose_range: dict[str, tuple[float, float]] = {}

    for axis in ("x", "y", "z"):
        value = position.get(axis)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            pose_range[axis] = (float(value[0]), float(value[1]))
        elif isinstance(value, (int, float)):
            pose_range[axis] = (float(value), float(value))
    if "yaw" in orientation:
        yaw = orientation["yaw"]
        if isinstance(yaw, (list, tuple)) and len(yaw) == 2:
            pose_range["yaw"] = (float(yaw[0]), float(yaw[1]))
        elif isinstance(yaw, (int, float)):
            pose_range["yaw"] = (float(yaw), float(yaw))

    if not pose_range:
        return ""

    return "\n".join(
        [
            f"randomize_{asset['name']} = EventTerm(",
            "    func=randomize_static_asset_pose,",
            '    mode="reset",',
            "    params={",
            f"        \"prim_path\": {asset.get('prim_path', f'/World/{asset['name']}')!r},",
            f"        \"base_pos\": {_tuple(asset.get('position', [0.0, 0.0, 0.0]))},",
            f"        \"base_quat\": {_tuple(asset.get('rotation', [1.0, 0.0, 0.0, 0.0]))},",
            f"        \"position_range\": { {axis: pose_range[axis] for axis in ('x', 'y', 'z') if axis in pose_range}!r},",
            f"        \"yaw_range\": {pose_range.get('yaw')!r},",
            "    },",
            ")",
        ]
    )


def _build_rigid_asset_block(asset: dict[str, Any]) -> str:
    physics = asset.get("physics", {})
    spawn_lines = [
        f"usd_path={_python_path_literal(asset['asset_path'])},",
        f"scale={_tuple(asset.get('scale', [1.0, 1.0, 1.0]))},",
        f'semantic_tags=[("class", "{asset["name"]}")],',
    ]
    color = asset.get("color")
    if color:
        spawn_lines.append(f"visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color={_tuple(color)}),")
    if physics.get("collision", True):
        spawn_lines.append("collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),")

    rigid_props_args = [
        "disable_gravity=False",
        f"solver_position_iteration_count={int(physics.get('solver_position_iterations', 32))}",
        f"solver_velocity_iteration_count={int(physics.get('solver_velocity_iterations', 1))}",
        f"max_depenetration_velocity={float(physics.get('max_depenetration_velocity', 5.0))}",
    ]
    if physics.get("max_linear_velocity") is not None:
        rigid_props_args.append(f"max_linear_velocity={float(physics['max_linear_velocity'])}")
    if physics.get("max_angular_velocity") is not None:
        rigid_props_args.append(f"max_angular_velocity={float(physics['max_angular_velocity'])}")
    spawn_lines.append(
        "rigid_props=sim_utils.RigidBodyPropertiesCfg("
        + ", ".join(rigid_props_args)
        + "),"
    )

    mass = physics.get("mass")
    if mass is not None:
        spawn_lines.append(f"mass_props=sim_utils.MassPropertiesCfg(mass={float(mass)}),")

    return f'''{asset["name"]} = RigidObjectCfg(
    prim_path={asset.get("prim_path", f"/World/{asset['name']}")!r},
    init_state=RigidObjectCfg.InitialStateCfg(
        pos={_tuple(asset.get("position", [0.0, 0.0, 0.0]))},
        rot={_tuple(asset.get("rotation", [1.0, 0.0, 0.0, 0.0]))},
    ),
    spawn=UsdFileCfg(
        func=spawn_usd_with_child_physics,
{_indent(chr(10).join(spawn_lines), 8)}
    ),
)'''


def _build_post_env_setup_block(assets: Any) -> str:
    triangle_roots: list[str] = []
    for asset in assets:
        if asset.get("physics", {}).get("collision_mesh") == "triangle":
            triangle_roots.append(asset.get("prim_path", f"/World/{asset['name']}"))
    if not triangle_roots:
        return ""

    lines = [
        "applied = 0",
        f"for _root in {triangle_roots!r}:",
        "    applied += _apply_triangle_mesh_collision(_root)",
        'print(f\"Applied triangle-mesh collision to {applied} mesh prims\")',
        "return applied",
    ]
    return "\n".join(lines)


def _find_active_shape_name(task_doc: dict[str, Any]) -> str | None:
    episode = task_doc.get("episode", {})
    resolved_name = episode.get("resolved_shape_to_place")
    if isinstance(resolved_name, str):
        return resolved_name

    for asset in task_doc.get("assets", []):
        if asset.get("type") == "rigid" and str(asset.get("name", "")).startswith("shape_"):
            return asset["name"]
    return None

def _build_robot_setup(robot: str, robot_asset: dict) -> tuple[str, str]:
    initial_joint_values = dict(robot_asset.get("initial_joints", {}))
    if robot == "franka" and "panda_finger_joint" in initial_joint_values:
        finger_value = initial_joint_values.pop("panda_finger_joint")
        initial_joint_values["panda_finger_joint1"] = finger_value
        initial_joint_values["panda_finger_joint2"] = finger_value
    initial_joints = repr(initial_joint_values)
    if robot == "franka":
        return (
            "from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG",
            "\n".join(
                [
                    'self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
                    f"self.scene.robot.init_state.joint_pos = {initial_joints}",
                    "self.actions.arm_action = mdp.JointPositionActionCfg(",
                    '    asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True',
                    ")",
                    "self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(",
                    '    asset_name="robot",',
                    '    joint_names=["panda_finger.*"],',
                    '    open_command_expr={"panda_finger_.*": 0.04},',
                    '    close_command_expr={"panda_finger_.*": 0.0},',
                    ")",
                ]
            ),
        )

    if robot == "openarm":
        return (
            "from isaaclab.actuators import ImplicitActuatorCfg\nfrom isaaclab_assets.robots.openarm import OPENARM_UNI_CFG",
            "\n".join(
                [
                    'self.scene.robot = OPENARM_UNI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
                    f"self.scene.robot.init_state.joint_pos = {initial_joints}",
                    "self.scene.robot.actuators = {",
                    '    "openarm_arm": ImplicitActuatorCfg(',
                    '        joint_names_expr=["openarm_joint[1-7]"],',
                    '        velocity_limit_sim={"openarm_joint[1-2]": 2.175, "openarm_joint[3-4]": 2.175, "openarm_joint[5-7]": 2.61},',
                    '        effort_limit_sim={"openarm_joint[1-2]": 40.0, "openarm_joint[3-4]": 27.0, "openarm_joint[5-7]": 7.0},',
                    "        stiffness=80.0,",
                    "        damping=4.0,",
                    "    ),",
                    '    "openarm_gripper": ImplicitActuatorCfg(',
                    '        joint_names_expr=["openarm_finger_joint.*"],',
                    "        velocity_limit_sim=0.2,",
                    "        effort_limit_sim=333.33,",
                    "        stiffness=2000.0,",
                    "        damping=100.0,",
                    "    ),",
                    "}",
                    "self.actions.arm_action = mdp.JointPositionActionCfg(",
                    '    asset_name="robot", joint_names=["openarm_joint.*"], scale=0.25, use_default_offset=True',
                    ")",
                    "gripper_joints = ['openarm_finger_joint1', 'openarm_finger_joint2']",
                    "self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(",
                    '    asset_name="robot",',
                    "    joint_names=gripper_joints,",
                    '    open_command_expr={j: 0.044 for j in gripper_joints},',
                    '    close_command_expr={j: 0.0 for j in gripper_joints},',
                    ")",
                ]
            ),
        )

    if robot == "so101":
        return (
            "from isaaclab.actuators import ImplicitActuatorCfg",
            "\n".join(
                [
                    "self.scene.robot = ArticulationCfg(",
                    '    prim_path="{ENV_REGEX_NS}/Robot",',
                    f"    init_state=ArticulationCfg.InitialStateCfg(pos={_tuple(robot_asset.get('position', [0.0, 0.0, 0.0]))}, rot={_tuple(robot_asset.get('rotation', [1.0, 0.0, 0.0, 0.0]))}, joint_pos={initial_joints}),",
                    "    spawn=UsdFileCfg(",
                    f"        usd_path={_python_path_literal(robot_asset['asset_path'])},",
                    "        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),",
                    "        articulation_props=sim_utils.ArticulationRootPropertiesCfg(",
                    "            enabled_self_collisions=False,",
                    "            solver_position_iteration_count=8,",
                    "            solver_velocity_iteration_count=0,",
                    "        ),",
                    "    ),",
                    "    actuators={",
                    '        "arm": ImplicitActuatorCfg(',
                    '            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],',
                    "            effort_limit_sim=10.0,",
                    "            stiffness=17.8,",
                    "            damping=0.6,",
                    "        ),",
                    '        "gripper": ImplicitActuatorCfg(',
                    '            joint_names_expr=["gripper"],',
                    "            effort_limit_sim=10.0,",
                    "            stiffness=200.0,",
                    "            damping=10.0,",
                    "        ),",
                    "    },",
                    ")",
                    "self.actions.arm_action = mdp.JointPositionActionCfg(",
                    '    asset_name="robot", joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"], scale=0.25, use_default_offset=True',
                    ")",
                    "self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(",
                    '    asset_name="robot",',
                    '    joint_names=["gripper"],',
                    '    open_command_expr={"gripper": 1.0},',
                    '    close_command_expr={"gripper": 0.0},',
                    ")",
                ]
            ),
        )

    if robot == "ur10e":
        return (
            "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG",
            "\n".join(
                [
                    'self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")',
                    f"self.scene.robot.init_state.joint_pos = {initial_joints}",
                    "self.actions.arm_action = mdp.JointPositionActionCfg(",
                    '    asset_name="robot", joint_names=["shoulder_.*", "elbow_joint", "wrist_.*"], scale=0.25, use_default_offset=True',
                    ")",
                    "self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(",
                    '    asset_name="robot",',
                    '    joint_names=["finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint", "right_inner_finger_joint", "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"],',
                    '    open_command_expr={"finger_joint": 0.0, "right_outer_knuckle_joint": 0.0, "left_inner_finger_joint": 0.0, "right_inner_finger_joint": 0.0, "left_inner_finger_knuckle_joint": 0.0, "right_inner_finger_knuckle_joint": 0.0},',
                    '    close_command_expr={"finger_joint": 0.65, "right_outer_knuckle_joint": 0.65, "left_inner_finger_joint": -0.65, "right_inner_finger_joint": 0.65, "left_inner_finger_knuckle_joint": -0.65, "right_inner_finger_knuckle_joint": -0.65},',
                    ")",
                ]
            ),
        )

    raise ValueError(f"Unsupported AssemblingKits robot: {robot}")


def _tuple(values) -> str:
    return repr(tuple(values))


def _python_path_literal(path: str) -> str:
    if path.startswith("{ISAAC_NUCLEUS_DIR}"):
        suffix = path[len("{ISAAC_NUCLEUS_DIR}"):]
        return f'f"{{ISAAC_NUCLEUS_DIR}}{suffix}"'
    if path.startswith("{ISAACLAB_NUCLEUS_DIR}"):
        suffix = path[len("{ISAACLAB_NUCLEUS_DIR}"):]
        return f'f"{{ISAACLAB_NUCLEUS_DIR}}{suffix}"'
    return repr(path)


def _indent(text: str, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else "" for line in text.splitlines())
