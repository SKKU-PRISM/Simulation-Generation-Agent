"""Deterministic IsaacLab template for AssemblingKits scene tasks."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_assembling_kits_template(task_doc: dict, robot: str) -> dict[str, str]:
    """Build a checked IsaacLab template for the AssemblingKits family."""
    task_name = task_doc.get("task", {}).get("name", "AssemblingKits")
    env_class = f"{task_name}EnvCfg"
    scene_class = f"{task_name}SceneCfg"

    assets = {asset["name"]: asset for asset in task_doc.get("assets", []) if asset.get("name")}
    robot_asset = next(asset for asset in task_doc.get("assets", []) if asset.get("type") == "articulation")
    camera_cfg = task_doc.get("camera", {})

    static_asset_names = ["table", "kit_tray", "shape_09", "shape_11", "shape_12", "shape_15", "shape_17"]
    scene_assets = "\n\n".join(_build_asset_block(assets[name]) for name in static_asset_names if name in assets)
    robot_imports, robot_setup = _build_robot_setup(robot, robot_asset)

    env_cfg = f'''# Copyright (c) 2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
{robot_imports}
import mdp as mdp


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


def _build_asset_block(asset: dict) -> str:
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
