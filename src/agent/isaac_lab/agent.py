"""IsaacLab Environment Generation Agent.

Converts YAML task documents into IsaacLab ManagerBasedRLEnv Python code
and validates by executing with IsaacLab.

Usage:
    python scripts/isaaclab_agent.py tasks/franka/stack/franka_stack.yaml
    python scripts/isaaclab_agent.py tasks/franka/lift/franka_lift.yaml --dry-run
    python scripts/isaaclab_agent.py --batch tasks/franka/
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from src.agent.common.llm_client import AzureOpenAIClient
from src.agent.common.robot_names import contains_robot_name, normalize_robot_name
from src.agent.common.task_docs import dump_task_document, load_task_document
from src.agent.isaac_lab.assembling_kits_template import build_assembling_kits_template
from src.agent.isaac_lab.scene_capture import build_capture_script

console = Console()


# ---------------------------------------------------------------------------
# Reference map: YAML task category -> IsaacLab source files (code pattern examples)
# ---------------------------------------------------------------------------

REFERENCE_MAP = {
    "stack": {
        "base": "stack/stack_env_cfg.py",
        "robot": {
            "default": ["stack/config/franka/stack_joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "so101": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": [
                "stack/config/ur10_gripper/stack_joint_pos_env_cfg.py",
                "deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py",
            ],
        },
        "mdp": ["stack/mdp/terminations.py", "stack/mdp/observations.py"],
    },
    "lift": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["lift/mdp/rewards.py", "lift/mdp/terminations.py"],
    },
    "reach": {
        "base": "reach/reach_env_cfg.py",
        "robot": {
            "default": ["reach/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["reach/config/openarm/unimanual/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/reach/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": [],
    },
    "cabinet": {
        "base": "cabinet/cabinet_env_cfg.py",
        "robot": {
            "default": ["cabinet/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["cabinet/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": [],
    },
    # Categories without direct IsaacLab equivalent — use closest match
    "pick_place": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "so101": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": [
                "deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py",
                "deploy/reach/config/ur_10e/joint_pos_env_cfg.py",
            ],
        },
        "mdp": ["lift/mdp/rewards.py", "pick_place/mdp/observations.py", "pick_place/mdp/terminations.py"],
    },
    "sort": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "so101": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["lift/mdp/rewards.py"],
    },
    "peg_insert": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["lift/mdp/rewards.py"],
    },
    "assembly": {
        "base": "deploy/gear_assembly/gear_assembly_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "so101": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["deploy/mdp/terminations.py", "deploy/mdp/events.py"],
    },
}


class IsaacLabAgent:
    """Agent that converts YAML task documents to IsaacLab environment code."""

    def __init__(self, config_path: str | None = None):
        config_path = config_path or str(PROJECT_ROOT / "configs" / "isaaclab_agent_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.isaaclab_path = Path(
            self.config["isaaclab"].get("path")
            or os.environ.get("ISAACLAB_PATH")
            or str((PROJECT_ROOT.parent / "IsaacLab").resolve())
        )
        self.reference_base = self.isaaclab_path / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "manipulation"
        self.output_dir = PROJECT_ROOT / self.config["agent"]["output_dir"]
        self.max_retries = self.config["agent"]["max_retries"]
        self.execution_timeout = self.config["agent"]["execution_timeout"]
        self.headless = self.config["agent"]["headless"]
        self.num_envs = self.config["agent"]["validation_num_envs"]
        self.save_intermediate = self.config["agent"]["save_intermediate"]

        # Load prompt templates
        self.system_prompt = (PROJECT_ROOT / "prompts" / "isaaclab_generation.md").read_text()
        self.error_fix_template = (PROJECT_ROOT / "prompts" / "isaaclab_error_fix.md").read_text()

        # Initialize LLM client
        self.llm = AzureOpenAIClient()

    # ----- YAML Loading -----

    def load_task_yaml(self, yaml_path: str) -> dict:
        """Load and return parsed YAML task document."""
        return load_task_document(yaml_path)

    def _detect_category(self, yaml_path: str) -> str:
        """Detect task category from path (tasks/{robot}/{category}/...)."""
        parts = Path(yaml_path).parts
        # Look for known categories
        for part in parts:
            if part in REFERENCE_MAP:
                return part
        # Fallback: guess from filename
        name = Path(yaml_path).stem.lower()
        for cat in REFERENCE_MAP:
            if cat in name:
                return cat
        return "lift"  # default fallback

    def _detect_robot(self, task_doc: dict, yaml_path: str | None = None) -> str:
        """Detect robot type from task document."""
        for asset in task_doc.get("assets", []):
            if asset.get("type") == "articulation":
                robot = contains_robot_name(
                    [
                        str(asset.get("robot_type", "")),
                        str(asset.get("name", "")),
                        str(asset.get("asset_path", "")),
                    ]
                )
                if robot:
                    return robot

        task_meta = task_doc.get("task", {})
        robot = contains_robot_name([str(task_meta.get("name", "")), str(task_meta.get("description", ""))])
        if robot:
            return robot

        if yaml_path:
            robot = normalize_robot_name(str(Path(yaml_path)))
            if robot:
                return robot

        return "franka"

    @staticmethod
    def _is_assembling_kits_task(yaml_path: str, task_doc: dict) -> bool:
        """Return True when the task belongs to the AssemblingKits family."""
        stem = Path(yaml_path).stem.lower()
        task_name = str(task_doc.get("task", {}).get("name", "")).lower()
        return "assembling_kits" in stem or "assemblingkits" in task_name

    @staticmethod
    def _is_factory_task(task_doc: dict) -> bool:
        """Return True for Factory tasks (PegInsert, GearMesh, NutThread).

        Detected by the ``initial_grasp`` field in the YAML or by name.
        """
        if task_doc.get("task", {}).get("initial_grasp"):
            return True
        name = task_doc.get("task", {}).get("name", "").lower().replace("_", "").replace(" ", "")
        return any(k in name for k in ("peginsert", "gearmesh", "nutthread"))

    _FACTORY_CFG_MAP = {
        "peginsert": ("FactoryTaskPegInsertCfg", "peg_insert"),
        "gearmesh": ("FactoryTaskGearMeshCfg", "gear_mesh"),
        "nutthread": ("FactoryTaskNutThreadCfg", "nut_thread"),
    }

    # Factory asset definitions for ManagerBasedRLEnv env_cfg.py templates
    _FACTORY_ASSETS = {
        "peginsert": {
            "task_class": "FrankaPegInsert",
            "held_name": "peg",
            "held_usd": "factory_peg_8mm.usd",
            "held_mass": 0.019,
            "held_init_pos": (0.0, 0.2, 0.15),
            "fixed_name": "hole",
            "fixed_usd": "factory_hole_8mm.usd",
            "fixed_mass": 0.05,
            "fixed_init_pos": (0.5, 0.0, 0.025),
            "extra_objects": [],
        },
        "gearmesh": {
            "task_class": "FrankaGearMesh",
            "held_name": "gear",
            "held_usd": "factory_gear_medium.usd",
            "held_mass": 0.012,
            "held_init_pos": (0.0, 0.2, 0.15),
            "fixed_name": "gear_base",
            "fixed_usd": "factory_gear_base.usd",
            "fixed_mass": 0.05,
            "fixed_init_pos": (0.5, 0.0, 0.05),
            "extra_objects": [
                ("small_gear", "factory_gear_small.usd", 0.019, (0.55, 0.1, 0.05)),
                ("large_gear", "factory_gear_large.usd", 0.019, (0.55, -0.1, 0.05)),
            ],
        },
        "nutthread": {
            "task_class": "FrankaNutThread",
            "held_name": "nut",
            "held_usd": "factory_nut_m16.usd",
            "held_mass": 0.03,
            "held_init_pos": (0.0, 0.2, 0.15),
            "fixed_name": "bolt",
            "fixed_usd": "factory_bolt_m16.usd",
            "fixed_mass": 0.05,
            "fixed_init_pos": (0.5, 0.0, 0.05),
            "extra_objects": [],
        },
    }

    def _build_factory_template(self, task_doc: dict) -> dict:
        """Build ManagerBasedRLEnv env_cfg.py + run_env.py for Factory tasks.

        This generates code compatible with the data-collection pipeline
        (Step 3), which requires env_cfg.py with ManagerBasedRLEnvCfg.
        """
        raw_name = task_doc["task"]["name"].lower().replace("_", "").replace(" ", "")

        assets = None
        for key, info in self._FACTORY_ASSETS.items():
            if key in raw_name:
                assets = info
                break
        if assets is None:
            raise ValueError(f"Unknown factory task: {task_doc['task']['name']}")

        task_class = assets["task_class"]
        env_cfg_class = f"{task_class}EnvCfg"

        def _rigid_object_block(name: str, usd: str, mass: float, pos: tuple,
                                disable_gravity: bool = False, kinematic: bool = False) -> str:
            pos_str = f"[{pos[0]}, {pos[1]}, {pos[2]}]"
            return f'''
    {name} = RigidObjectCfg(
        prim_path="{{ENV_REGEX_NS}}/{name.capitalize()}",
        init_state=RigidObjectCfg.InitialStateCfg(pos={pos_str}, rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{{ISAACLAB_NUCLEUS_DIR}}/Factory/{usd}",
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity={disable_gravity},
                max_depenetration_velocity=5.0,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass={mass}),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
        ),
    )'''

        # Build scene objects
        held_block = _rigid_object_block(
            assets["held_name"], assets["held_usd"], assets["held_mass"],
            assets["held_init_pos"], disable_gravity=True,
        )
        fixed_block = _rigid_object_block(
            assets["fixed_name"], assets["fixed_usd"], assets["fixed_mass"],
            assets["fixed_init_pos"], disable_gravity=False,
        )
        extra_blocks = ""
        extra_names = []
        for (ename, eusd, emass, epos) in assets.get("extra_objects", []):
            extra_blocks += _rigid_object_block(ename, eusd, emass, epos)
            extra_names.append(ename)

        # Scene entity names for observations
        all_object_names = [assets["held_name"], assets["fixed_name"]] + extra_names
        obs_terms = "\n".join(
            f'        {n}_pos = ObsTerm(func=mdp.root_pos_w, params={{"asset_cfg": SceneEntityCfg("{n}")}})\n'
            f'        {n}_quat = ObsTerm(func=mdp.root_quat_w, params={{"asset_cfg": SceneEntityCfg("{n}")}})'
            for n in all_object_names
        )
        reset_terms = "\n".join(
            f'    reset_{n} = EventTerm(\n'
            f'        func=mdp.reset_root_state_uniform,\n'
            f'        mode="reset",\n'
            f'        params={{\n'
            f'            "asset_cfg": SceneEntityCfg("{n}"),\n'
            f'            "pose_range": {{"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.0, 0.0)}},\n'
            f'            "velocity_range": {{}},\n'
            f'        }},\n'
            f'    )'
            for n in all_object_names
        )

        env_cfg_code = f'''\
# Auto-generated ManagerBasedRLEnv config for Factory task: {task_class}
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg

import mdp as mdp

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class {task_class}SceneCfg(InteractiveSceneCfg):
    """Factory task scene: robot + table + factory objects."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
{held_block}
{fixed_block}
{extra_blocks}

    table = AssetBaseCfg(
        prim_path="{{ENV_REGEX_NS}}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{{ISAAC_NUCLEUS_DIR}}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


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
{obs_terms}

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
{reset_terms}


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class {env_cfg_class}(ManagerBasedRLEnvCfg):
    scene: {task_class}SceneCfg = {task_class}SceneCfg(num_envs=1, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
        ),
    )

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 30.0
        self.sim.render_interval = self.decimation

        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{{ENV_REGEX_NS}}/Robot",
        )
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {{
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }}

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{{ENV_REGEX_NS}}/Robot/panda_link0",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{{ENV_REGEX_NS}}/Robot/panda_hand",
                    name="ee_frame",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
                ),
            ],
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={{"panda_finger_.*": 0.04}},
            close_command_expr={{"panda_finger_.*": 0.0}},
        )
'''

        # run_env.py — standard ManagerBasedRLEnv runner (same pattern as all other tasks)
        run_env_code = f'''\
"""Factory ManagerBasedRLEnv runner (auto-generated)."""

import argparse
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Factory task runner")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaaclab.envs import ManagerBasedRLEnv
from env_cfg import {env_cfg_class}

def main():
    env_cfg = {env_cfg_class}()
    env_cfg.scene.num_envs = args.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    for _ in range(200):
        action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
        env.step(action)

    print("SUCCESS")
    marker = os.environ.get("ISAACLAB_SUCCESS_MARKER")
    if marker:
        Path(marker).write_text("ok")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
'''

        # mdp/__init__.py — re-export isaaclab.envs.mdp (standard pattern)
        mdp_init = '''\
from isaaclab.envs.mdp import *  # noqa: F401,F403
'''

        return {
            "env_cfg.py": env_cfg_code,
            "run_env.py": run_env_code,
            "mdp/__init__.py": mdp_init,
        }

    # ----- Reference Code Loading -----

    def _robot_reference_paths(self, category: str, robot: str) -> list[str]:
        """Return ordered robot-specific reference paths."""
        ref_info = REFERENCE_MAP.get(category, REFERENCE_MAP["lift"])
        robot_refs = ref_info.get("robot", {})
        if isinstance(robot_refs, str):
            return [robot_refs]
        return robot_refs.get(robot, robot_refs.get("default", []))

    def select_reference(self, category: str, robot: str) -> str:
        """Read IsaacLab reference source files for the given task category.

        These are used as CODE PATTERN EXAMPLES in the LLM prompt.
        The actual task content comes from the YAML document.
        """
        ref_info = REFERENCE_MAP.get(category, REFERENCE_MAP["lift"])
        parts = []

        # Base config
        base_path = self.reference_base / ref_info["base"]
        if base_path.exists():
            parts.append(f"### Reference: {ref_info['base']}\n```python\n{base_path.read_text()}\n```")

        # Robot-specific config
        for robot_ref in self._robot_reference_paths(category, robot):
            robot_path = self.reference_base / robot_ref
            if robot_path.exists():
                content = robot_path.read_text()
                if len(content) > 6000:
                    content = content[:6000] + "\n# ... (truncated)"
                parts.append(f"### Reference: {robot_ref}\n```python\n{content}\n```")

        # MDP files (rewards, terminations)
        for mdp_file in ref_info.get("mdp", []):
            mdp_path = self.reference_base / mdp_file
            if mdp_path.exists():
                content = mdp_path.read_text()
                # Truncate if too long (save tokens for gpt-5-mini)
                if len(content) > 3000:
                    content = content[:3000] + "\n# ... (truncated)"
                parts.append(f"### Reference: {mdp_file}\n```python\n{content}\n```")

        return "\n\n".join(parts)

    def _build_robot_guidance(self, robot: str) -> str:
        """Return robot-specific implementation rules to inject into the prompt."""
        if robot == "ur10e":
            return """## Robot-Specific Rules (UR10e + Robotiq 2F-85)

- This task MUST use `UR10e_ROBOTIQ_2F_85_CFG`, not `FRANKA_PANDA_CFG`, `UR10_CFG`, or suction gripper configs.
- Add this import when `robot_type` is `ur10e`:
  `from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG`
- Robot setup pattern:
  ```python
  self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
  ```
- UR10e arm action should target the 6 arm joints only:
  `["shoulder_.*", "elbow_joint", "wrist_.*"]`
- Use `wrist_3_link` as the end-effector body for observations / frame transformers.
- Never use `panda_hand`, `panda_link0`, `ee_link`, or `panda_finger.*` in UR10e code.
- Prefer a binary gripper action with the Robotiq joints:
  ```python
  self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
      asset_name="robot",
      joint_names=[
          "finger_joint",
          "right_outer_knuckle_joint",
          "left_inner_finger_joint",
          "right_inner_finger_joint",
          "left_inner_finger_knuckle_joint",
          "right_inner_finger_knuckle_joint",
      ],
      open_command_expr={
          "finger_joint": 0.0,
          "right_outer_knuckle_joint": 0.0,
          "left_inner_finger_joint": 0.0,
          "right_inner_finger_joint": 0.0,
          "left_inner_finger_knuckle_joint": 0.0,
          "right_inner_finger_knuckle_joint": 0.0,
      },
      close_command_expr={
          "finger_joint": 0.65,
          "right_outer_knuckle_joint": 0.65,
          "left_inner_finger_joint": -0.65,
          "right_inner_finger_joint": 0.65,
          "left_inner_finger_knuckle_joint": -0.65,
          "right_inner_finger_knuckle_joint": -0.65,
      },
  )
  ```
- For UR10e observations, use `SceneEntityCfg("robot", body_names=["wrist_3_link"])`.
"""

        if robot == "openarm":
            return """## Robot-Specific Rules (OpenArm)

- This task MUST import and use `OPENARM_UNI_CFG` from `isaaclab_assets.robots.openarm`.
- Do not rely on the preset alone. After assigning `self.scene.robot = OPENARM_UNI_CFG.replace(...)`, explicitly set:
  - `self.scene.robot.init_state.joint_pos` from the YAML `initial_joints`
  - `self.scene.robot.actuators` from the YAML actuator values
- The end-effector frame must use the OpenArm TCP, not an object prim:
  - source prim: `{ENV_REGEX_NS}/Robot/openarm_link0`
  - target prim: `{ENV_REGEX_NS}/Robot/openarm_ee_tcp`
- Never create a `FrameTransformerCfg` whose `prim_path` or target frame points to `/Cube_*` or another rigid object.
- Keep the gripper joints as `openarm_finger_joint1` and `openarm_finger_joint2`.
- For OpenArm stack/sort/pick-place tasks, preserve the YAML rigid-object spawn ranges and do not invent Franka-specific joints or body names.
"""

        if robot == "so101":
            return """## Robot-Specific Rules (SO-101)

- SO-101 is a repo-local USD robot. Use the YAML `asset_path` exactly as provided after normalization; do not rewrite it relative to `outputs/`.
- Do not use `FRANKA_PANDA_CFG`, `OPENARM_UNI_CFG`, or any IsaacLab preset for SO-101.
- Build `self.scene.robot` as an explicit `ArticulationCfg` with:
  - `spawn=UsdFileCfg(usd_path=<normalized so101 usd path>)`
  - `init_state.joint_pos` from the YAML `initial_joints`
  - `actuators` matching the YAML arm/gripper actuator specs
- Use `gripper_frame_link` as the SO-101 end-effector body for observations and frame transformers.
- For stack tasks with `min_separation`, do not reset cubes independently if that can violate the separation constraint. Keep object resets deterministic or implement a single non-overlapping sampler.
- Never use Franka joints/body names such as `panda_joint.*` or `panda_hand` in SO-101 code.
"""

        return ""

    @staticmethod
    def _build_assembly_guidance(task_doc: dict) -> str:
        """Return assembly-specific instructions for grounded code generation."""
        conditions = task_doc.get("goal", {}).get("conditions", [])
        has_constraints = bool(task_doc.get("constraints"))
        has_tri_mesh = any(
            asset.get("physics", {}).get("collision_mesh") == "triangle"
            for asset in task_doc.get("assets", [])
        )

        guidance = [
            "## Assembly-Specific Rules",
            "",
            "- Preserve the YAML `scale`, `color`, `primitive`, and `asset_path` values exactly.",
            "- If the YAML contains `target_position` / `target_rotation`, use those resolved values directly in custom termination logic.",
            "- Generate custom `mdp/terminations.py` for assembly goals such as `upright`, `inserted_into`, and `in_slot`.",
            "- For `inserted_into` / `in_slot`, compare the subject pose against the resolved target pose, not just a generic object distance.",
            "- For `upright`, implement a custom orientation-based check against world +Z using the YAML tolerance.",
            "- Keep local asset paths like `assets/assembling_kits/*.usd` as repo-local paths in the generated code.",
        ]

        if has_constraints:
            guidance.append("- The YAML `constraints` define rigid bodies that must behave as a single assembled object; preserve that semantics in the generated environment.")
        if has_tri_mesh:
            guidance.append("- `physics.collision_mesh: triangle` is required for cutout-hole geometry; preserve that instead of replacing it with a convex proxy.")
        if any(cond.get("relation") == "in_slot" for cond in conditions):
            guidance.append("- For assembling-kits tasks, the resolved `subject` is the active misplaced shape and the slot target pose is already attached to the goal condition.")

        return "\n".join(guidance) + "\n"

    # ----- Prompt Building -----

    def build_prompt(self, yaml_path: str, task_doc: dict, reference_code: str) -> str:
        """Build the user prompt with YAML content and reference code."""
        yaml_content = dump_task_document(task_doc)
        category = self._detect_category(yaml_path)
        robot = self._detect_robot(task_doc, yaml_path)
        task_name = task_doc.get("task", {}).get("name", "UnknownTask")
        robot_guidance = self._build_robot_guidance(robot)
        assembly_guidance = self._build_assembly_guidance(task_doc) if category == "assembly" else ""

        prompt = f"""## Task

Convert the following YAML task document into IsaacLab ManagerBasedRLEnv Python code.

**Task name**: {task_name}
**Robot**: {robot}
**Category**: {category}
**Config class name**: `{task_name}EnvCfg`

## YAML Task Document (THIS IS THE INPUT — use these values)

```yaml
{yaml_content}
```

## IsaacLab Reference Code (USE AS CODE PATTERN EXAMPLES ONLY)

The following IsaacLab source files show the correct Python code structure.
Use this structure/pattern, but fill in values from the YAML above.

{reference_code}

{robot_guidance}
{assembly_guidance}

## Instructions

1. Generate `env_cfg.py` containing the full environment configuration:
   - Scene class with robot, table, ground plane, lighting, and all objects from YAML
   - Actions, observations, events, rewards, and terminations config classes
   - Main `{task_name}EnvCfg(ManagerBasedRLEnvCfg)` class

2. Generate `run_env.py` for validation (follow the exact AppLauncher pattern from the system prompt)

3. Map ALL YAML values to IsaacLab equivalents:
   - Robot: use the correct pre-defined config for the detected robot (`FRANKA_PANDA_CFG` for franka, `UR10e_ROBOTIQ_2F_85_CFG` for ur10e)
   - Objects: map asset_path, position, rotation, scale, physics properties
   - Simulation: map timestep, decimation, episode_length, physx settings
   - Randomization: convert to EventTermCfg
   - Goals: convert to termination conditions
   - If any rigid object uses `randomize.position.min_separation`, preserve that constraint; do not emit independent resets that can overlap objects

4. **CRITICAL — Pre-submission checklist (all items REQUIRED):**
   - [ ] `env_cfg.py`: ALL YAML objects present in SceneCfg with correct positions
   - [ ] `env_cfg.py`: ObservationsCfg has `root_pos_w` + `root_quat_w` for EACH rigid object
   - [ ] `mdp/rewards.py`: At least one `object_ee_distance` reward per manipulation target
   - [ ] `mdp/terminations.py`: One custom termination per YAML goal condition
   - [ ] `mdp/__init__.py`: Imports from both `isaaclab.envs.mdp`, `.rewards`, and `.terminations`
   - [ ] `env_cfg.py` RewardsCfg: References custom reward functions with `weight` parameter
   - [ ] `env_cfg.py` TerminationsCfg: References custom termination with `time_out=False`
   - [ ] Never submit with only `action_rate_l2` + `time_out`
"""
        return prompt

    # ----- Code Parsing -----

    def parse_generated_code(self, response: str) -> dict[str, str]:
        """Parse LLM response into {filename: code} dict.

        Expects format:
            ```python:filename.py
            ...code...
            ```
        """
        files = {}
        # Match ```python:filename.py ... ``` blocks
        pattern = r"```python:([^\n]+)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        for filename, code in matches:
            filename = filename.strip()
            files[filename] = code.strip()

        # Fallback: if no filename annotations, try to detect by content
        if not files:
            # Try plain ```python blocks
            plain_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
            for i, block in enumerate(plain_blocks):
                block = block.strip()
                if "AppLauncher" in block:
                    files["run_env.py"] = block
                elif "ManagerBasedRLEnvCfg" in block or "@configclass" in block:
                    files["env_cfg.py"] = block
                elif "def " in block and i > 1:
                    # Likely a custom MDP function
                    if "reward" in block.lower():
                        files["mdp/rewards.py"] = block
                    else:
                        files["mdp/terminations.py"] = block

        return files

    # ----- File Writing -----

    def write_output(self, code_dict: dict[str, str], output_dir: Path) -> Path:
        """Write generated code files to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename, code in code_dict.items():
            filepath = output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(code)
            try:
                label = filepath.resolve().relative_to(PROJECT_ROOT)
            except ValueError:
                label = filepath
            console.print(f"  [green]Wrote[/green] {label}")

        # ALWAYS ensure mdp/ package exists (env_cfg.py uses `import mdp as mdp`)
        mdp_dir = output_dir / "mdp"
        mdp_dir.mkdir(parents=True, exist_ok=True)

        # Regenerate __init__.py to match actual custom files
        init_file = mdp_dir / "__init__.py"
        exports = []
        if (mdp_dir / "rewards.py").exists():
            exports.append("from .rewards import *  # noqa: F401, F403")
        if (mdp_dir / "terminations.py").exists():
            exports.append("from .terminations import *  # noqa: F401, F403")
        if (mdp_dir / "observations.py").exists():
            exports.append("from .observations import *  # noqa: F401, F403")
        init_file.write_text(
            '"""MDP functions — re-exports isaaclab.envs.mdp + custom functions."""\n\n'
            "from isaaclab.envs.mdp import *  # noqa: F401, F403\n\n"
            + "\n".join(exports)
            + "\n"
        )

        # Auto-patch env_cfg.py: ensure it uses local mdp/ package
        self._patch_mdp_import(output_dir)

        # Clear __pycache__ to avoid stale bytecode
        pycache = mdp_dir / "__pycache__"
        if pycache.exists():
            import shutil
            shutil.rmtree(pycache)

        return output_dir

    def _patch_mdp_import(self, output_dir: Path):
        """Ensure env_cfg.py imports local mdp/ package instead of isaaclab.envs.mdp."""
        env_cfg_path = output_dir / "env_cfg.py"
        if not env_cfg_path.exists():
            return

        content = env_cfg_path.read_text()
        original = content

        # Remove ALL variants of isaaclab.envs.mdp imports
        for pattern in [
            "import isaaclab.envs.mdp as mdp\n",
            "import isaaclab.envs.mdp as mdp  # noqa: F401, F403\n",
            "from isaaclab.envs.mdp import *  # noqa: F401, F403\n",
            "from isaaclab.envs.mdp import *\n",
        ]:
            content = content.replace(pattern, "")

        # Remove `from mdp import *` (wrong pattern — need `import mdp as mdp`)
        content = content.replace("from mdp import *  # noqa: F401, F403\n", "")
        content = content.replace("from mdp import *\n", "")
        # Also handle the "corrected import" comment variant
        for line in content.split("\n"):
            if line.strip().startswith("from mdp import *"):
                content = content.replace(line + "\n", "")

        # Ensure exactly one `import mdp as mdp` line exists
        if "import mdp as mdp" not in content:
            # Insert after the last isaaclab import
            lines = content.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("from isaaclab") or stripped.startswith("import isaaclab"):
                    insert_idx = i + 1
            lines.insert(insert_idx, "import mdp as mdp  # local mdp/ package")
            content = "\n".join(lines)

        if content != original:
            env_cfg_path.write_text(content)
            console.print("  [yellow]Auto-patched[/yellow] env_cfg.py → import mdp as mdp (local)")

    # ----- Post-Generation Validation -----

    @staticmethod
    def _ensure_import_line(content: str, import_line: str) -> str:
        """Insert an import line near related imports if it is missing."""
        if import_line in content:
            return content

        lines = content.split("\n")
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from isaaclab_assets") or line.startswith("from isaaclab.actuators"):
                insert_idx = i
                break
            if line.startswith("from isaaclab") or line.startswith("import isaaclab"):
                insert_idx = i + 1
        lines.insert(insert_idx, import_line)
        return "\n".join(lines)

    @staticmethod
    def _inject_post_init_snippet(content: str, snippet: str) -> str:
        """Append a snippet at the end of the generated __post_init__ method."""
        marker = "    def __post_init__(self):"
        start = content.find(marker)
        if start == -1 or snippet.strip() in content:
            return content

        lines = content.split("\n")
        method_matches = [i for i, line in enumerate(lines) if line.startswith(marker)]
        method_idx = method_matches[-1] if method_matches else None
        if method_idx is None:
            return content

        insert_idx = len(lines)
        for i in range(method_idx + 1, len(lines)):
            line = lines[i]
            stripped = line.lstrip()
            if not stripped:
                continue
            indent = len(line) - len(stripped)
            if indent <= 4 and not stripped.startswith("#"):
                insert_idx = i
                break

        snippet_lines = snippet.rstrip().split("\n")
        lines[insert_idx:insert_idx] = [""] + snippet_lines + [""]
        return "\n".join(lines)

    @staticmethod
    def _inject_after_super_post_init(content: str, snippet: str) -> str:
        """Insert a snippet immediately after the generated super().__post_init__() call."""
        if snippet.strip() in content:
            return content

        lines = content.split("\n")
        method_matches = [i for i, line in enumerate(lines) if line.startswith("    def __post_init__(self):")]
        if method_matches:
            method_idx = method_matches[-1]
            for i in range(method_idx + 1, len(lines)):
                if "super().__post_init__()" in lines[i]:
                    snippet_lines = snippet.rstrip().split("\n")
                    lines[i + 1:i + 1] = [""] + snippet_lines + [""]
                    return "\n".join(lines)
        return IsaacLabAgent._inject_post_init_snippet(content, snippet)

    @staticmethod
    def _task_has_min_separation(task_doc: dict) -> bool:
        for asset in task_doc.get("assets", []):
            if asset.get("type") != "rigid":
                continue
            randomize = asset.get("randomize", {})
            if randomize.get("min_separation") is not None:
                return True
        return False

    @staticmethod
    def _patch_articulation_spawn_scales(content: str, task_doc: dict) -> tuple[str, bool]:
        """Ensure articulated USD assets preserve YAML scale in generated ArticulationCfg blocks."""
        changed = False

        for asset in task_doc.get("assets", []):
            if asset.get("type") != "articulation":
                continue

            scale = asset.get("scale")
            if not isinstance(scale, (list, tuple)) or len(scale) != 3:
                continue
            if all(abs(float(value) - 1.0) < 1e-6 for value in scale):
                continue

            asset_name = str(asset.get("name", "")).strip()
            if not asset_name:
                continue

            scale_literal = f"scale=({float(scale[0])}, {float(scale[1])}, {float(scale[2])})"
            pattern = re.compile(
                rf'(?P<prefix>{re.escape(asset_name)}\s*=\s*ArticulationCfg\([\s\S]*?spawn=UsdFileCfg\()'
                rf'(?P<args>[\s\S]*?)'
                rf'(?P<suffix>\)\s*,\s*init_state=)',
                re.MULTILINE,
            )

            def repl(match: re.Match[str]) -> str:
                args = match.group("args")
                if "scale=" in args:
                    return match.group(0)

                stripped_args = args.rstrip()
                trailing = args[len(stripped_args):]
                if "\n" in stripped_args:
                    stripped_args += f",\n                {scale_literal}"
                else:
                    stripped_args += f", {scale_literal}"
                return match.group("prefix") + stripped_args + trailing + match.group("suffix")

            updated, count = pattern.subn(repl, content, count=1)
            if count > 0 and updated != content:
                content = updated
                changed = True

        return content, changed

    @staticmethod
    def _build_disable_randomization_snippet(task_doc: dict) -> str:
        if not IsaacLabAgent._task_has_min_separation(task_doc):
            return ""

        event_names = []
        for asset in task_doc.get("assets", []):
            if asset.get("type") != "rigid":
                continue
            name = asset.get("name")
            if name:
                event_names.append(f"randomize_{name.lower()}")
                event_names.append(f"reset_{name.lower()}")
                event_names.append(f"{name.lower()}_position")

        if not event_names:
            return ""

        return "\n".join([
            "        # Auto-patch: keep object reset deterministic for stack/sort smoke validation.",
            "        # Independent uniform resets ignore YAML min_separation and can destabilize the scene.",
            f"        for _event_name in {event_names!r}:",
            "            if hasattr(self.events, _event_name):",
            "                setattr(self.events, _event_name, None)",
        ])

    @staticmethod
    def _build_primitive_scene_override_snippet(task_doc: dict) -> str:
        """Rebuild primitive assets with explicit physics so smoke tests use stable objects."""
        lines: list[str] = []

        for asset in task_doc.get("assets", []):
            if asset.get("source") != "primitive" or asset.get("primitive") != "cube":
                continue

            name = asset.get("name")
            if not name:
                continue

            physics = asset.get("physics", {})
            prim_name = Path(str(asset.get("prim_path", f"/World/{name}"))).name
            env_prim_path = f"{{ENV_REGEX_NS}}/{prim_name}"
            position = asset.get("position", [0.0, 0.0, 0.0])
            rotation = asset.get("rotation", [1.0, 0.0, 0.0, 0.0])
            size = tuple(float(v) for v in asset.get("scale", [1.0, 1.0, 1.0]))
            color = tuple(float(v) for v in asset.get("color", [0.7, 0.7, 0.7]))

            if physics.get("rigid_body", False):
                mass = float(physics.get("mass", 0.05))
                solver_pos = int(physics.get("solver_position_iterations", 8))
                solver_vel = int(physics.get("solver_velocity_iterations", 1))
                max_ang = float(physics.get("max_angular_velocity", 1000.0))
                max_lin = float(physics.get("max_linear_velocity", 1000.0))
                max_dep = float(physics.get("max_depenetration_velocity", 5.0))
                lines.extend(
                    [
                        f"        self.scene.{name} = RigidObjectCfg(",
                        f"            prim_path={env_prim_path!r},",
                        f"            init_state=RigidObjectCfg.InitialStateCfg(pos={position!r}, rot={rotation!r}),",
                        "            spawn=sim_utils.CuboidCfg(",
                        f"                size={size!r},",
                        "                rigid_props=sim_utils.RigidBodyPropertiesCfg(",
                        "                    disable_gravity=False,",
                        f"                    solver_position_iteration_count={solver_pos},",
                        f"                    solver_velocity_iteration_count={solver_vel},",
                        f"                    max_angular_velocity={max_ang},",
                        f"                    max_linear_velocity={max_lin},",
                        f"                    max_depenetration_velocity={max_dep},",
                        "                ),",
                        f"                mass_props=sim_utils.MassPropertiesCfg(mass={mass}),",
                        "                collision_props=sim_utils.CollisionPropertiesCfg(),",
                        f"                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color={color!r}),",
                        "            ),",
                        "        )",
                    ]
                )
            else:
                lines.extend(
                    [
                        f"        self.scene.{name} = AssetBaseCfg(",
                        f"            prim_path={env_prim_path!r},",
                        f"            init_state=AssetBaseCfg.InitialStateCfg(pos={position!r}, rot={rotation!r}),",
                        "            spawn=sim_utils.CuboidCfg(",
                        f"                size={size!r},",
                        "                collision_props=sim_utils.CollisionPropertiesCfg(),",
                        f"                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color={color!r}),",
                        "            ),",
                        "        )",
                    ]
                )

        if not lines:
            return ""

        return "\n".join(
            [
                "        # Auto-patch: rebuild primitive assets with explicit collision and mass properties.",
                "        # The generated defaults can leave tiny cubes with invalid mass/inertia in smoke tests.",
                *lines,
            ]
        )

    @staticmethod
    def _build_openarm_override_snippet(task_doc: dict) -> str:
        robot_asset = next(
            (asset for asset in task_doc.get("assets", []) if asset.get("type") == "articulation"),
            None,
        )
        if not robot_asset:
            return ""

        initial_joints = robot_asset.get("initial_joints", {})
        simulation = task_doc.get("simulation", {})
        physx = simulation.get("physx", {})
        disable_randomization = IsaacLabAgent._build_disable_randomization_snippet(task_doc)
        return "\n".join([
            "        # Auto-patch: normalize OpenArm robot config to the official asset + valid EE frame.",
            f"        self.decimation = {int(simulation.get('decimation', 5))}",
            f"        self.episode_length_s = {float(simulation.get('episode_length', 30.0))}",
            f"        self.sim.dt = {float(simulation.get('timestep', 0.01))}",
            f"        self.sim.render_interval = {int(simulation.get('render_interval', max(1, simulation.get('decimation', 5))))}",
            f"        self.sim.physx.bounce_threshold_velocity = {float(physx.get('bounce_threshold', 0.01))}",
            f"        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = {int(physx.get('gpu_found_lost_aggregate_pairs_capacity', 1024 * 1024 * 4))}",
            f"        self.sim.physx.gpu_total_aggregate_pairs_capacity = {int(physx.get('gpu_total_aggregate_pairs_capacity', 16 * 1024))}",
            f"        self.sim.physx.friction_correlation_distance = {float(physx.get('friction_correlation_distance', 0.00625))}",
            "        self.scene.robot = OPENARM_UNI_CFG.replace(prim_path=\"{ENV_REGEX_NS}/Robot\")",
            f"        self.scene.robot.init_state.joint_pos = {repr(initial_joints)}",
            "        self.scene.robot.actuators = {",
            "            \"openarm_arm\": ImplicitActuatorCfg(",
            "                joint_names_expr=[\"openarm_joint[1-7]\"],",
            "                velocity_limit_sim={\"openarm_joint[1-2]\": 2.175, \"openarm_joint[3-4]\": 2.175, \"openarm_joint[5-7]\": 2.61},",
            "                effort_limit_sim={\"openarm_joint[1-2]\": 40.0, \"openarm_joint[3-4]\": 27.0, \"openarm_joint[5-7]\": 7.0},",
            "                stiffness=80.0,",
            "                damping=4.0,",
            "            ),",
            "            \"openarm_gripper\": ImplicitActuatorCfg(",
            "                joint_names_expr=[\"openarm_finger_joint.*\"],",
            "                velocity_limit_sim=0.2,",
            "                effort_limit_sim=333.33,",
            "                stiffness=2000.0,",
            "                damping=100.0,",
            "            ),",
            "        }",
            "        try:",
            "            self.scene.robot.spawn.semantic_tags = [(\"class\", \"robot\")]",
            "        except Exception:",
            "            pass",
            "        self.scene.ee_frame = FrameTransformerCfg(",
            "            prim_path=\"{ENV_REGEX_NS}/Robot/openarm_link0\",",
            "            debug_vis=False,",
            "            target_frames=[",
            "                FrameTransformerCfg.FrameCfg(",
            "                    prim_path=\"{ENV_REGEX_NS}/Robot/openarm_ee_tcp\",",
            "                    name=\"end_effector\",",
            "                ),",
            "            ],",
            "        )",
            "        self.actions.arm_action = mdp.JointPositionActionCfg(",
            "            asset_name=\"robot\",",
            "            joint_names=[\"openarm_joint.*\"],",
            "            scale=0.5,",
            "            use_default_offset=True,",
            "        )",
            "        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(",
            "            asset_name=\"robot\",",
            "            joint_names=[\"openarm_finger_joint.*\"],",
            "            open_command_expr={\"openarm_finger_joint.*\": 0.044},",
            "            close_command_expr={\"openarm_finger_joint.*\": 0.0},",
            "        )",
            disable_randomization,
            "        return",
        ])

    @staticmethod
    def _build_so101_override_snippet(task_doc: dict) -> str:
        robot_asset = next(
            (asset for asset in task_doc.get("assets", []) if asset.get("type") == "articulation"),
            None,
        )
        if not robot_asset:
            return ""

        asset_path = str(robot_asset.get("asset_path", ""))
        initial_joints = robot_asset.get("initial_joints", {})
        position = robot_asset.get("position", [0.0, 0.0, 0.0])
        rotation = robot_asset.get("rotation", [1.0, 0.0, 0.0, 0.0])
        simulation = task_doc.get("simulation", {})
        physx = simulation.get("physx", {})
        disable_randomization = IsaacLabAgent._build_disable_randomization_snippet(task_doc)
        return "\n".join([
            "        # Auto-patch: normalize SO-101 robot config to explicit ArticulationCfg.",
            f"        self.decimation = {int(simulation.get('decimation', 5))}",
            f"        self.episode_length_s = {float(simulation.get('episode_length', 30.0))}",
            f"        self.sim.dt = {float(simulation.get('timestep', 0.01))}",
            f"        self.sim.render_interval = {int(simulation.get('render_interval', max(1, simulation.get('decimation', 5))))}",
            f"        self.sim.physx.bounce_threshold_velocity = {float(physx.get('bounce_threshold', 0.01))}",
            f"        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = {int(physx.get('gpu_found_lost_aggregate_pairs_capacity', 1024 * 1024 * 4))}",
            f"        self.sim.physx.gpu_total_aggregate_pairs_capacity = {int(physx.get('gpu_total_aggregate_pairs_capacity', 16 * 1024))}",
            f"        self.sim.physx.friction_correlation_distance = {float(physx.get('friction_correlation_distance', 0.00625))}",
            "        self.scene.robot = ArticulationCfg(",
            "            prim_path=\"{ENV_REGEX_NS}/Robot\",",
            f"            init_state=ArticulationCfg.InitialStateCfg(pos={position!r}, rot={rotation!r}, joint_pos={initial_joints!r}),",
            "            spawn=UsdFileCfg(",
            f"                usd_path={asset_path!r},",
            "                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),",
            "                articulation_props=sim_utils.ArticulationRootPropertiesCfg(",
            "                    enabled_self_collisions=False,",
            "                    solver_position_iteration_count=8,",
            "                    solver_velocity_iteration_count=0,",
            "                ),",
            "            ),",
            "            actuators={",
            "                \"arm\": ImplicitActuatorCfg(",
            "                    joint_names_expr=[\"shoulder_pan\", \"shoulder_lift\", \"elbow_flex\", \"wrist_flex\", \"wrist_roll\"],",
            "                    effort_limit_sim=10.0,",
            "                    stiffness=17.8,",
            "                    damping=0.6,",
            "                ),",
            "                \"gripper\": ImplicitActuatorCfg(",
            "                    joint_names_expr=[\"gripper\"],",
            "                    effort_limit_sim=10.0,",
            "                    stiffness=200.0,",
            "                    damping=10.0,",
            "                ),",
            "            },",
            "        )",
            "        try:",
            "            self.scene.robot.spawn.semantic_tags = [(\"class\", \"robot\")]",
            "        except Exception:",
            "            pass",
            "        self.scene.ee_frame = FrameTransformerCfg(",
            "            prim_path=\"{ENV_REGEX_NS}/Robot/gripper_frame_link\",",
            "            debug_vis=False,",
            "            target_frames=[",
            "                FrameTransformerCfg.FrameCfg(",
            "                    prim_path=\"{ENV_REGEX_NS}/Robot/gripper_frame_link\",",
            "                    name=\"end_effector\",",
            "                ),",
            "            ],",
            "        )",
            "        self.actions.arm_action = mdp.JointPositionActionCfg(",
            "            asset_name=\"robot\",",
            "            joint_names=[\"shoulder_pan\", \"shoulder_lift\", \"elbow_flex\", \"wrist_flex\", \"wrist_roll\"],",
            "            scale=0.5,",
            "            use_default_offset=True,",
            "        )",
            "        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(",
            "            asset_name=\"robot\",",
            "            joint_names=[\"gripper\"],",
            "            open_command_expr={\"gripper\": 1.745},",
            "            close_command_expr={\"gripper\": -0.175},",
            "        )",
            IsaacLabAgent._build_primitive_scene_override_snippet(task_doc),
            disable_randomization,
            "        return",
        ])

    def _validate_generated_code(self, output_dir: Path, robot: str, task_doc: dict | None = None) -> list[str]:
        """Post-generation validation: auto-fix known mistakes before execution."""
        fixes = []

        # Fix: `in env.scene:` → `in env.scene.keys():`
        # InteractiveScene has no __contains__, so `in` operator causes KeyError
        for py_file in output_dir.rglob("*.py"):
            try:
                text = py_file.read_text()
            except Exception:
                continue
            original = text
            # Match patterns like `not in env.scene:` or `in env.scene:`
            # but not already `in env.scene.keys():`
            text = re.sub(
                r'\bnot in env\.scene\b(?!\.keys)',
                'not in env.scene.keys()',
                text,
            )
            text = re.sub(
                r'(?<!not )\bin env\.scene\b(?!\.keys|\.articulations|\.rigid_objects|\.sensors|\[)',
                'in env.scene.keys()',
                text,
            )
            if text != original:
                py_file.write_text(text)
                fixes.append(f"Patched {py_file.name}: env.scene → env.scene.keys()")

        fixes.extend(self._apply_robot_specific_fixes(output_dir, robot, task_doc or {}))
        return fixes

    def _apply_robot_specific_fixes(self, output_dir: Path, robot: str, task_doc: dict) -> list[str]:
        """Patch predictable robot-specific mistakes from the LLM output."""
        env_cfg_path = output_dir / "env_cfg.py"
        if not env_cfg_path.exists():
            return []

        original = env_cfg_path.read_text()
        content = original
        fixes = []

        content = re.sub(
            r'SceneEntityCfg\(\s*(?P<entity>"[^"]+"|\'[^\']+\')\s*,\s*body_name\s*=\s*(?P<name>"[^"]+"|\'[^\']+\')(?P<suffix>\s*,[^\)]*)?\)',
            lambda match: (
                "SceneEntityCfg("
                + match.group("entity")
                + ", body_names=["
                + match.group("name")
                + "]"
                + (match.group("suffix") or "")
                + ")"
            ),
            content,
        )
        if content != original:
            fixes.append("Patched env_cfg.py: normalized SceneEntityCfg body_name -> body_names")
            original = content

        if task_doc:
            content, scaled = self._patch_articulation_spawn_scales(content, task_doc)
            if scaled:
                fixes.append("Patched env_cfg.py: restored articulation spawn scale from YAML")
                original = content

        if robot == "ur10e":
            replacements = [
                (
                    "from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # if robot_type is franka",
                    "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG",
                ),
                (
                    "from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG",
                    "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG",
                ),
                ("FRANKA_PANDA_CFG", "UR10e_ROBOTIQ_2F_85_CFG"),
                ('["panda_joint.*"]', '["shoulder_.*", "elbow_joint", "wrist_.*"]'),
                (
                    '["panda_finger.*"]',
                    '["finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint", '
                    '"right_inner_finger_joint", "left_inner_finger_knuckle_joint", '
                    '"right_inner_finger_knuckle_joint"]',
                ),
                ('{"panda_finger_.*": 0.04}', '{"finger_joint": 0.0, "right_outer_knuckle_joint": 0.0, '
                 '"left_inner_finger_joint": 0.0, "right_inner_finger_joint": 0.0, '
                 '"left_inner_finger_knuckle_joint": 0.0, "right_inner_finger_knuckle_joint": 0.0}'),
                ('{"panda_finger_.*": 0.0}', '{"finger_joint": 0.65, "right_outer_knuckle_joint": 0.65, '
                 '"left_inner_finger_joint": -0.65, "right_inner_finger_joint": 0.65, '
                 '"left_inner_finger_knuckle_joint": -0.65, "right_inner_finger_knuckle_joint": -0.65}'),
                ('gripper_joint_names = ["panda_finger_.*"]',
                 'gripper_joint_names = ["finger_joint", "right_outer_knuckle_joint", '
                 '"left_inner_finger_joint", "right_inner_finger_joint", '
                 '"left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]'),
                ("gripper_open_val = 0.04", "gripper_open_val = 0.0"),
                ("gripper_threshold = 0.005", "gripper_threshold = 0.05"),
                ("panda_hand", "wrist_3_link"),
                ("panda_link0", "base_link"),
                ("ee_link", "wrist_3_link"),
            ]
            for source, target in replacements:
                content = content.replace(source, target)
            content = self._ensure_import_line(
                content, "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG"
            )
            if content != original:
                fixes.append("Patched env_cfg.py: normalized UR10e Robotiq 2F-85 config and body/joint names")

        elif robot == "openarm":
            content = re.sub(
                r'params=\{\s*"asset_cfg"\s*:\s*SceneEntityCfg\("robot"\)\s*,\s*"body_name"\s*:\s*"([^"]+)"\s*\}',
                r'params={"asset_cfg": SceneEntityCfg("robot", body_names=["\1"])}',
                content,
            )
            content = re.sub(
                r'params=\{\s*"body_name"\s*:\s*"([^"]+)"\s*,\s*"asset_cfg"\s*:\s*SceneEntityCfg\("robot"\)\s*\}',
                r'params={"asset_cfg": SceneEntityCfg("robot", body_names=["\1"])}',
                content,
            )
            content = re.sub(
                r'params=\{\s*"body_name"\s*:\s*"([^"]+)"\s*\}',
                r'params={"asset_cfg": SceneEntityCfg("robot", body_names=["\1"])}',
                content,
            )
            content = self._ensure_import_line(
                content, "from isaaclab_assets.robots.openarm import OPENARM_UNI_CFG"
            )
            content = self._ensure_import_line(
                content, "from isaaclab.actuators import ImplicitActuatorCfg"
            )
            if task_doc:
                content = self._inject_after_super_post_init(content, self._build_openarm_override_snippet(task_doc))
            if content != original:
                fixes.append("Patched env_cfg.py: forced OpenArm asset/actuators/tcp frame and deterministic object reset")

        elif robot == "so101":
            content = re.sub(
                r'params=\{\s*"asset_cfg"\s*:\s*SceneEntityCfg\("robot"\)\s*,\s*"body_name"\s*:\s*"([^"]+)"\s*\}',
                r'params={"asset_cfg": SceneEntityCfg("robot", body_names=["\1"])}',
                content,
            )
            content = re.sub(
                r'params=\{\s*"body_name"\s*:\s*"([^"]+)"\s*,\s*"asset_cfg"\s*:\s*SceneEntityCfg\("robot"\)\s*\}',
                r'params={"asset_cfg": SceneEntityCfg("robot", body_names=["\1"])}',
                content,
            )
            content = self._ensure_import_line(
                content, "from isaaclab.actuators import ImplicitActuatorCfg"
            )
            if task_doc:
                content = self._inject_after_super_post_init(content, self._build_so101_override_snippet(task_doc))
            if content != original:
                fixes.append("Patched env_cfg.py: forced SO-101 articulation config and deterministic object reset")

        if fixes:
            env_cfg_path.write_text(content)
        return fixes

    # ----- Execution -----

    def _execute_isaaclab_script(
        self,
        output_dir: Path,
        script_name: str,
        *,
        enable_cameras: bool = False,
        num_envs: int | None = None,
        headless: bool | None = None,
    ) -> tuple[bool, str]:
        """Execute a generated IsaacLab script via isaaclab.sh inside the conda env."""
        run_script = output_dir / script_name
        if not run_script.exists():
            return False, f"{script_name} not found in output directory"

        launcher = self.isaaclab_path / "isaaclab.sh"
        if not launcher.exists():
            return False, f"isaaclab.sh not found at {launcher}"

        conda_env = self.config["isaaclab"]["conda_env"]
        script_num_envs = self.num_envs if num_envs is None else num_envs
        script_headless = self.headless if headless is None else headless

        # Build the isaaclab.sh command
        isaaclab_cmd = f"{launcher} -p {run_script} --num_envs {script_num_envs}"
        if script_headless:
            isaaclab_cmd += " --headless"
        if enable_cameras:
            isaaclab_cmd += " --enable_cameras"

        # Execute directly via isaaclab.sh (handles its own Python env)
        cmd = ["bash", "-c", isaaclab_cmd]

        console.print(f"  [blue]Executing[/blue]: {isaaclab_cmd}")

        # Marker file for reliable SUCCESS detection (stdout can be lost when simulation_app.close() hangs)
        marker_file = output_dir / ".success_marker"
        if marker_file.exists():
            marker_file.unlink()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered stdout in subprocess
        env["ISAACLAB_SUCCESS_MARKER"] = str(marker_file)
        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"  # Required for headless camera rendering

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                cwd=str(output_dir),
                env=env,
            )
            output = result.stdout + "\n" + result.stderr
            # Check for SUCCESS in stdout/stderr OR via marker file
            success = "SUCCESS" in output or marker_file.exists()
            return success, output

        except subprocess.TimeoutExpired as e:
            # If timed out, check if SUCCESS was printed before the hang OR marker file exists
            partial = ""
            if e.stdout:
                partial += e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
            if e.stderr:
                partial += e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
            if "SUCCESS" in partial or marker_file.exists():
                return True, partial
            return False, f"Execution timed out after {self.execution_timeout}s\n{partial[-2000:]}"
        except Exception as e:
            return False, f"Execution error: {e}"

    def execute_isaaclab(self, output_dir: Path) -> tuple[bool, str]:
        """Execute generated run_env.py via isaaclab.sh inside the conda env."""
        return self._execute_isaaclab_script(output_dir, "run_env.py")

    @staticmethod
    def _detect_env_class(output_dir: Path) -> str | None:
        """Detect the EnvCfg class name from generated env_cfg.py."""
        env_cfg_path = output_dir / "env_cfg.py"
        if not env_cfg_path.exists():
            return None
        content = env_cfg_path.read_text()
        match = re.search(r"class\s+(\w+EnvCfg)\b", content)
        return match.group(1) if match else None

    def capture_scene(self, output_dir: Path) -> tuple[bool, str]:
        """Capture a scene image for template-based tasks."""
        return self._execute_isaaclab_script(
            output_dir,
            "capture_scene.py",
            enable_cameras=True,
            num_envs=1,
            headless=False,
        )

    # ----- Error Parsing -----

    def _extract_traceback(self, output: str) -> str:
        """Extract the Python traceback from IsaacLab output, stripping GPU info/warnings."""
        lines = output.split("\n")
        # Find "Traceback (most recent call last):" and take everything from there
        traceback_start = None
        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                traceback_start = i
                break

        if traceback_start is not None:
            traceback_text = "\n".join(lines[traceback_start:])
            # Truncate if still too long
            if len(traceback_text) > 3000:
                traceback_text = traceback_text[:3000] + "\n... (truncated)"
            return traceback_text

        # No traceback found — look for ERROR lines or return last 2000 chars
        error_lines = [l for l in lines if "Error" in l or "error" in l.lower() or "FAILED" in l]
        if error_lines:
            return "\n".join(error_lines[-20:])

        # Fallback: return last 2000 chars
        if len(output) > 2000:
            return "... (no traceback found, last 2000 chars)\n" + output[-2000:]
        return output

    # ----- Error Fix -----

    def fix_errors(self, code_dict: dict[str, str], error_msg: str, attempt: int) -> dict[str, str]:
        """Ask LLM to fix errors in generated code."""
        console.print(f"  [yellow]Attempt {attempt}[/yellow]: Asking LLM to fix errors...")

        # Build original code section
        original_parts = []
        for filename, code in code_dict.items():
            original_parts.append(f"```python:{filename}\n{code}\n```")
        original_code = "\n\n".join(original_parts)

        # Extract only the traceback from the error output (skip GPU info, warnings)
        error_msg = self._extract_traceback(error_msg)

        prompt = self.error_fix_template.replace("{ORIGINAL_CODE}", original_code)
        prompt = prompt.replace("{ERROR_TRACEBACK}", error_msg)

        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            temperature=0.1,
            step="isaaclab_error_fix",
        )

        fixed_files = self.parse_generated_code(response)

        # Merge: update only changed files, keep unchanged
        merged = dict(code_dict)
        merged.update(fixed_files)
        return merged

    def refine_from_verification(
        self, code_dict: dict[str, str], code_eval: dict, yaml_text: str
    ) -> dict[str, str]:
        """Refine generated code based on SceneVerifier 4-category feedback.

        Args:
            code_dict: Current code files (env_cfg.py, run_env.py, mdp/*.py).
            code_eval: SceneVerifier code_evaluation result with 4-category scores.
            yaml_text: Original YAML task specification.

        Returns:
            Updated code_dict with refined files.
        """
        # Build feedback from 4-category scores
        feedback_lines = ["=== SceneVerifier Code Evaluation Feedback ==="]
        categories = {
            "scene_fidelity": ("Scene Fidelity", 40),
            "mdp_correctness": ("MDP Correctness", 20),
            "task_alignment": ("Task Alignment", 15),
            "runtime_validity": ("Runtime Validity", 25),
        }
        for key, (label, max_score) in categories.items():
            entry = code_eval.get(key, {})
            score = entry.get("score", 0)
            details = entry.get("details", "")
            feedback_lines.append(f"  {label}: {score}/{max_score} — {details}")

        issues = code_eval.get("issues", [])
        if issues:
            feedback_lines.append("\n=== Specific Issues to Fix ===")
            for issue in issues:
                feedback_lines.append(f"  - {issue}")

        total = code_eval.get("total_score", 0)
        feedback_lines.append(f"\n  Total: {total}/100")
        feedback_lines.append("\nPlease fix the issues above to improve the score.")
        feedback_lines.append("Focus on the lowest-scoring categories first.")

        feedback = "\n".join(feedback_lines)
        console.print(f"  [yellow]Refinement:[/yellow] Sending verification feedback to LLM...")

        # Build code section
        original_parts = []
        for filename, code in code_dict.items():
            original_parts.append(f"```python:{filename}\n{code}\n```")
        original_code = "\n\n".join(original_parts)

        prompt = f"""The following IsaacLab environment code was generated from a YAML task specification,
but the quality evaluation found issues. Please fix the code to address the feedback below.

=== YAML Task Specification ===
{yaml_text[:2000]}

=== Current Generated Code ===
{original_code}

=== Quality Evaluation Feedback ===
{feedback}

Please output the COMPLETE fixed code for each file that needs changes.
Use the format: ```python:filename.py ... ```
Only include files that need modifications."""

        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            temperature=0.1,
            step="isaaclab_verify_refine",
        )

        fixed_files = self.parse_generated_code(response)
        merged = dict(code_dict)
        merged.update(fixed_files)
        return merged

    # ----- Main Pipeline -----

    def run(self, yaml_path: str, dry_run: bool = False,
            evaluate: bool = False) -> dict:
        """Run the full pipeline: YAML → code generation → execution → error fix loop.

        Args:
            yaml_path: Path to the YAML task document.
            dry_run: If True, generate code but don't execute.
            evaluate: If True, run quality evaluation after successful execution.

        Returns:
            dict with keys: success, output_dir, attempts, error, evaluation (optional)
        """
        yaml_path = str(Path(yaml_path).resolve())
        task_doc = self.load_task_yaml(yaml_path)
        task_name = task_doc.get("task", {}).get("name", "UnknownTask")
        category = self._detect_category(yaml_path)
        robot = self._detect_robot(task_doc, yaml_path)
        use_assembling_kits_template = category == "assembly" and self._is_assembling_kits_task(yaml_path, task_doc)
        use_factory_template = category == "assembly" and self._is_factory_task(task_doc)

        console.print(Panel(
            f"[bold]Task[/bold]: {task_name}\n"
            f"[bold]Robot[/bold]: {robot}\n"
            f"[bold]Category[/bold]: {category}\n"
            f"[bold]YAML[/bold]: {yaml_path}",
            title="IsaacLab Agent",
            border_style="blue",
        ))

        # Output directory: outputs/{timestamp}_{task_name}/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_slug = task_name.lower().replace(" ", "_")
        output_dir = self.output_dir / f"{timestamp}_{task_slug}"

        if use_factory_template:
            console.print("[bold]Step 1:[/bold] Using Factory environment template...")
            code_dict = self._build_factory_template(task_doc)
            console.print(f"  Generated {len(code_dict)} template file(s): {list(code_dict.keys())}")
        elif use_assembling_kits_template:
            console.print("[bold]Step 1:[/bold] Using deterministic AssemblingKits template...")
            code_dict = build_assembling_kits_template(task_doc, robot)
            console.print(f"  Generated {len(code_dict)} template file(s): {list(code_dict.keys())}")
        else:
            # Step 1: Load reference code (pattern examples)
            console.print("[bold]Step 1:[/bold] Loading reference code patterns...")
            reference_code = self.select_reference(category, robot)

            # Step 2: Build prompt and generate code
            console.print("[bold]Step 2:[/bold] Generating IsaacLab code via LLM...")
            user_prompt = self.build_prompt(yaml_path, task_doc, reference_code)
            response = self.llm.generate(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt,
                temperature=self.config["agent"]["temperature"],
                step="isaaclab_codegen",
            )

            code_dict = self.parse_generated_code(response)
            if not code_dict:
                console.print("[red]Error:[/red] LLM returned no parseable code blocks")
                return {"success": False, "output_dir": None, "attempts": 0, "error": "No code generated"}

            console.print(f"  Generated {len(code_dict)} file(s): {list(code_dict.keys())}")

        # Step 3: Write files
        console.print("[bold]Step 3:[/bold] Writing generated code...")
        self.write_output(code_dict, output_dir)

        if dry_run:
            console.print("[green]Dry run complete.[/green] Code saved to:", str(output_dir))
            return {"success": True, "output_dir": str(output_dir), "attempts": 0, "error": None}

        # Step 3.5: Validate and auto-fix known mistakes
        use_deterministic_template = use_factory_template or use_assembling_kits_template
        if not use_deterministic_template:
            fixes = self._validate_generated_code(output_dir, robot, task_doc)
            for fix in fixes:
                console.print(f"  [yellow]Auto-fix[/yellow]: {fix}")

        # Step 4: Execute and iterate
        max_attempts = 1 if use_deterministic_template else self.max_retries
        for attempt in range(1, max_attempts + 1):
            console.print(f"[bold]Step 4:[/bold] Execution attempt {attempt}/{max_attempts}...")
            success, output = self.execute_isaaclab(output_dir)

            if success:
                console.print(f"[green bold]SUCCESS[/green bold] on attempt {attempt}")
                result = {"success": True, "output_dir": str(output_dir), "attempts": attempt, "error": None}

                # Copy task YAML to output dir for traceability
                shutil.copy2(yaml_path, output_dir / "task.yaml")

                if evaluate:
                    console.print("[bold]Step 5:[/bold] Running quality evaluation...")
                    from src.agent.isaac_lab.evaluator import IsaacLabEvaluator
                    evaluator = IsaacLabEvaluator()
                    eval_result = evaluator.evaluate(
                        output_dir=str(output_dir), yaml_path=yaml_path,
                    )
                    result["evaluation"] = eval_result
                    result["eval_score"] = eval_result.get("total_score", 0)

                # Debug capture: 3-angle screenshots for all tasks
                console.print("[bold]Step 6:[/bold] Capturing debug screenshots (front/top/wrist)...")
                if use_deterministic_template:
                    # AssemblingKits / Factory: use existing single-camera capture
                    captured, capture_output = self.capture_scene(output_dir)
                    if captured and (output_dir / "scene_capture.png").exists():
                        result["scene_capture"] = str(output_dir / "scene_capture.png")
                    else:
                        result["scene_capture_error"] = capture_output[-500:] if capture_output else "Unknown capture error"
                else:
                    # General tasks: generate + run 3-angle capture script
                    env_class = self._detect_env_class(output_dir)
                    if env_class:
                        capture_code = build_capture_script(env_class, robot)
                        (output_dir / "capture_scene.py").write_text(capture_code)
                        captured, capture_output = self.capture_scene(output_dir)
                        debug_dir = output_dir / "debug"
                        if captured and debug_dir.exists() and any(debug_dir.glob("*.png")):
                            result["debug_captures"] = [str(p) for p in sorted(debug_dir.glob("*.png"))]
                            console.print(f"  Saved {len(result['debug_captures'])} screenshots to debug/")
                        else:
                            result["debug_capture_error"] = capture_output[-500:] if capture_output else "Capture failed"
                            console.print(f"  [yellow]Warning:[/yellow] Debug capture failed")
                    else:
                        console.print("  [yellow]Warning:[/yellow] Could not detect env class, skipping capture")

                # Step 7: Scene verification (code + VLM)
                try:
                    from src.agent.isaac_lab.scene_verifier import SceneVerifier
                    verifier = SceneVerifier()
                    if verifier.available:
                        console.print("[bold]Step 7:[/bold] Scene verification (code + VLM)...")
                        front_img = output_dir / "debug" / "front.png"
                        top_img = output_dir / "debug" / "top.png"
                        env_cfg_path = output_dir / "env_cfg.py"

                        if env_cfg_path.exists() and (front_img.exists() or top_img.exists()):
                            env_cfg_text = env_cfg_path.read_text()
                            yaml_text = Path(yaml_path).read_text()
                            verification = verifier.verify(front_img, top_img, env_cfg_text, yaml_text)
                            result["scene_verification"] = verification

                            # Display code evaluation (4-category)
                            code_eval = verification.get("code_evaluation", {})
                            total = code_eval.get("total_score", 0)
                            console.print(f"  Code evaluation: {total}/100")
                            for cat in ["scene_fidelity", "mdp_correctness", "task_alignment", "runtime_validity"]:
                                entry = code_eval.get(cat, {})
                                console.print(f"    {cat}: {entry.get('score', 0)}/{entry.get('max', 0)}")

                            # Display image evaluation
                            img_eval = verification.get("image_evaluation", {})
                            front_s = img_eval.get("front", {}).get("score", 0)
                            top_s = img_eval.get("top", {}).get("score", 0)
                            vlm_pass = img_eval.get("vlm_pass", False)
                            console.print(f"  Image evaluation: front={front_s}/100, top={top_s}/100, pass={vlm_pass}")

                            # Step 7.1: Verification-based refinement
                            static_score = (
                                code_eval.get("scene_fidelity", {}).get("score", 0)
                                + code_eval.get("mdp_correctness", {}).get("score", 0)
                                + code_eval.get("task_alignment", {}).get("score", 0)
                            )
                            verify_threshold = 25  # out of 75 (runtime excluded)
                            verify_max_refine = 5

                            if (
                                static_score < verify_threshold
                                and not use_deterministic_template
                                and not hasattr(self, "_verify_refine_count")
                            ):
                                self._verify_refine_count = 0

                            if (
                                static_score < verify_threshold
                                and not use_deterministic_template
                                and getattr(self, "_verify_refine_count", 0) < verify_max_refine
                            ):
                                self._verify_refine_count = getattr(self, "_verify_refine_count", 0) + 1
                                console.print(
                                    f"  [yellow]Score {static_score}/75 < {verify_threshold}.[/yellow] "
                                    f"Refinement round {self._verify_refine_count}/{verify_max_refine}..."
                                )
                                yaml_text_for_refine = Path(yaml_path).read_text()
                                code_dict = self.refine_from_verification(
                                    code_dict, code_eval, yaml_text_for_refine
                                )
                                self.write_output(code_dict, output_dir)
                                fixes = self._validate_generated_code(output_dir, robot, task_doc)
                                for fix in fixes:
                                    console.print(f"  [yellow]Auto-fix[/yellow]: {fix}")
                                # Re-enter execution loop (next attempt)
                                continue
                            else:
                                # Clean up refine counter
                                if hasattr(self, "_verify_refine_count"):
                                    del self._verify_refine_count

                            if verification["overall_pass"]:
                                console.print("  [green]Scene verification PASSED[/green]")
                            else:
                                console.print("  [red]Scene verification FAILED[/red]")
                                for issue in code_eval.get("issues", []):
                                    console.print(f"    [yellow]code:[/yellow] {issue}")
                        else:
                            console.print("  [yellow]Skipped:[/yellow] Missing env_cfg.py or debug images")
                    else:
                        console.print("  [yellow]Step 7:[/yellow] SceneVerifier not available (OPENAI_API_KEY?)")
                except Exception as e:
                    console.print(f"  [yellow]Step 7 error:[/yellow] {e}")
                    result["scene_verification_error"] = str(e)

                # Clean up refine counter on exit
                if hasattr(self, "_verify_refine_count"):
                    del self._verify_refine_count

                # Save result.json
                result["timestamp"] = timestamp
                result["task_name"] = task_name
                result["robot"] = robot
                result["category"] = category
                result["yaml_path"] = yaml_path
                with open(output_dir / "result.json", "w") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)

                return result

            console.print(f"  [red]Failed[/red] (attempt {attempt})")

            # Save intermediate output
            if self.save_intermediate:
                error_file = output_dir / f"error_attempt_{attempt}.txt"
                error_file.write_text(output)

            # Fix errors (except on last attempt)
            if attempt < max_attempts:
                code_dict = self.fix_errors(code_dict, output, attempt + 1)
                self.write_output(code_dict, output_dir)
                # Re-validate after error fix
                fixes = self._validate_generated_code(output_dir, robot, task_doc)
                for fix in fixes:
                    console.print(f"  [yellow]Auto-fix[/yellow]: {fix}")

        console.print(f"[red bold]FAILED[/red bold] after {max_attempts} attempts")
        return {
            "success": False,
            "output_dir": str(output_dir),
            "attempts": max_attempts,
            "error": output[-500:] if output else "Unknown error",
        }

    def run_batch(self, task_dir: str, dry_run: bool = False) -> list[dict]:
        """Run the agent on all YAML files in a directory (recursively)."""
        task_dir = Path(task_dir)
        yaml_files = sorted(task_dir.rglob("*.yaml"))
        yaml_files = [f for f in yaml_files if "template" not in str(f)]

        console.print(f"[bold]Batch mode:[/bold] Found {len(yaml_files)} task YAML files in {task_dir}")

        results = []
        for i, yaml_file in enumerate(yaml_files, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold][{i}/{len(yaml_files)}][/bold] {yaml_file.name}")
            console.print(f"{'='*60}")

            result = self.run(str(yaml_file), dry_run=dry_run)
            result["yaml_path"] = str(yaml_file)
            results.append(result)

        # Summary
        successes = sum(1 for r in results if r["success"])
        console.print(f"\n[bold]Batch Summary:[/bold] {successes}/{len(results)} succeeded")
        for r in results:
            status = "[green]OK[/green]" if r["success"] else "[red]FAIL[/red]"
            console.print(f"  {status} {Path(r['yaml_path']).name}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert YAML task documents to IsaacLab ManagerBasedRLEnv code"
    )
    parser.add_argument("yaml_path", nargs="?", help="Path to task YAML file")
    parser.add_argument("--batch", type=str, help="Directory of YAML files to process")
    parser.add_argument("--dry-run", action="store_true", help="Generate code without executing")
    parser.add_argument("--evaluate", action="store_true", help="Run quality evaluation after success")
    parser.add_argument("--eval-only", type=str, help="Evaluate existing output directory (skip generation)")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--config", type=str, help="Path to agent config YAML")
    args = parser.parse_args()

    # Eval-only mode: evaluate existing output without re-generating
    if args.eval_only:
        if not args.yaml_path:
            parser.error("--eval-only requires yaml_path")
        from src.agent.isaac_lab.evaluator import IsaacLabEvaluator
        evaluator = IsaacLabEvaluator()
        report = evaluator.evaluate(output_dir=args.eval_only, yaml_path=args.yaml_path)
        sys.exit(0 if report["total_score"] >= 70 else 1)

    if not args.yaml_path and not args.batch:
        parser.error("Either yaml_path or --batch is required")

    agent = IsaacLabAgent(config_path=args.config)

    if args.output_dir:
        agent.output_dir = Path(args.output_dir)

    if args.batch:
        agent.run_batch(args.batch, dry_run=args.dry_run)
    else:
        result = agent.run(args.yaml_path, dry_run=args.dry_run,
                           evaluate=args.evaluate)
        sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
