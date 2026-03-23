"""
Configuration module for data collection pipeline.

Loads robot profiles from configs/robot_profiles/{name}.yaml and
pipeline settings from configs/data_collection_config.yaml.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from src.common.robot_names import normalize_robot_name

logger = logging.getLogger(__name__)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROBOT_PROFILES_DIR = PROJECT_ROOT / "configs" / "robot_profiles"
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "data_collection_config.yaml"


def _default_external_repo_dir(repo_name: str) -> Path:
    """Return the conventional sibling checkout location for an external repo."""
    return (PROJECT_ROOT.parent / repo_name).resolve()


def get_default_isaaclab_path() -> Path:
    """Resolve the default IsaacLab checkout path without user-specific hardcoding."""
    raw = os.environ.get("ISAACLAB_PATH")
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_external_repo_dir("IsaacLab")


def get_default_isaac_sim_dir() -> Path:
    """Resolve the default Isaac Sim checkout path without user-specific hardcoding."""
    raw = os.environ.get("ISAAC_SIM_DIR")
    if raw:
        return Path(raw).expanduser().resolve()
    return _default_external_repo_dir("isaac-sim")


@dataclass
class CameraConfig:
    """Single camera configuration for data collection.

    Two modes:
      - ``fixed``: World-frame camera defined by ``position`` + ``target`` (look-at).
        Orientation computed automatically via ``_look_at_quaternion()``.
      - ``body_mounted``: Camera attached to a robot body, defined by
        ``offset_pos`` + ``offset_rot`` (wxyz quaternion) in the given ``convention``.
    """

    name: str
    cam_type: str  # "fixed" | "body_mounted"
    # --- fixed camera ---
    position: list[float] = field(default_factory=list)
    target: list[float] = field(default_factory=list)
    up_vector: list[float] = field(default_factory=list)
    # --- body_mounted camera ---
    parent_body: str = ""
    offset_pos: list[float] = field(default_factory=lambda: [0, 0, 0])
    offset_rot: list[float] = field(default_factory=lambda: [1, 0, 0, 0])  # wxyz
    convention: str = "ros"  # "ros" | "world" | "opengl"
    # --- common ---
    resolution: tuple[int, int] = (640, 480)


@dataclass
class RobotSimConfig:
    """Robot configuration loaded from configs/robot_profiles/{name}.yaml."""

    # Identity
    name: str
    full_name: str
    arm_dofs: int
    total_dofs: int

    # Joint names
    arm_joint_names: list[str] = field(default_factory=list)
    finger_joint_names: list[str] = field(default_factory=list)

    # Joint limits: {joint_name: (lower, upper)} in radians
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Velocity limits: {joint_name: max_vel} in rad/s
    velocity_limits: dict[str, float] = field(default_factory=dict)

    # Gripper
    gripper_type: str = "parallel_jaw"  # "parallel_jaw" | "claw" | "suction"
    gripper_open_position: float | list[float] = 0.04
    gripper_close_position: float | list[float] = 0.0
    ee_finger_offset: float = 0.0      # m — EE frame to finger tip along grasp axis
    grasp_lateral_bias: float = 0.0    # m — local tool-frame bias for asymmetric claw grasps
    gripper_close_duration: float = 1.5  # s — how long to hold close command (low-stiffness grippers need more)
    gripper_grasp_stiffness: float = 0.0  # Nm/rad — if >0, override gripper_drive stiffness for reliable grasping

    # End-effector
    ee_frame_body: str = "panda_hand"
    ee_frame_tcp: str = ""
    ee_frame_offset_position: list[float] = field(default_factory=list)
    ik_ee_frame: str = ""

    # Poses: {joint_name: radian_value}
    ready_pose: dict[str, float] = field(default_factory=dict)
    default_pose: dict[str, float] = field(default_factory=dict)

    # Kinematics
    max_reach: float = 0.855
    tabletop_reach: float = 0.75
    workspace_envelope: dict[str, list[float]] = field(default_factory=dict)
    ik_backend: str = "auto"  # "auto" | "pinocchio" | "differential_ik"

    # Asset
    usd_path: str = ""
    prim_path: str = "/World/Robot"
    urdf_path: str = ""  # for Pinocchio IK

    # Camera defaults (legacy single-camera, kept for backward compat)
    camera_position: list[float] = field(default_factory=lambda: [1.5, 1.2, 1.0])
    camera_target: list[float] = field(default_factory=lambda: [0.25, 0.0, 0.3])

    # Multi-camera configuration {name: CameraConfig}
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Stiffness/damping for joint control
    arm_stiffness: float = 80.0
    arm_damping: float = 4.0

    @property
    def all_joint_names(self) -> list[str]:
        """All joint names (arm + finger)."""
        return self.arm_joint_names + self.finger_joint_names

    @property
    def has_gripper_joints(self) -> bool:
        """Whether the robot has controllable gripper joints."""
        return len(self.finger_joint_names) > 0

    # Gripper mirrored (loaded from YAML finger.mirrored, default True for parallel_jaw)
    _gripper_mirrored: Optional[bool] = None

    @property
    def gripper_mirrored(self) -> bool:
        """Whether finger joints are mirrored (both set to same value)."""
        if self._gripper_mirrored is not None:
            return self._gripper_mirrored
        return len(self.finger_joint_names) == 2 and self.gripper_type == "parallel_jaw"


def _resolve_urdf_path(raw: str) -> str:
    """Resolve URDF path templates used by robot profiles."""
    if not raw:
        return raw

    if "{ISAAC_SIM_DIR}" in raw:
        raw = raw.replace("{ISAAC_SIM_DIR}", str(get_default_isaac_sim_dir()))

    if "{ADC_URDF_DIR}" in raw:
        try:
            from .adc_imports import get_adc_root

            adc_root = get_adc_root()
            return raw.replace("{ADC_URDF_DIR}", str(adc_root / "assets" / "urdf"))
        except ImportError:
            logger.info(
                f"ADC submodule not available — URDF path '{raw}' unresolved, "
                "Pinocchio IK will be skipped"
            )
            return ""

    if "{ISAACLAB_URDF_DIR}" in raw:
        isaaclab_path = get_default_isaaclab_path()
        urdf_dir = Path(isaaclab_path) / "source" / "isaaclab" / "isaaclab" / "controllers" / "config" / "data"
        return raw.replace("{ISAACLAB_URDF_DIR}", str(urdf_dir))

    if "{ISAAC_SIM_URDF_DIR}" in raw:
        isaac_sim_dir = get_default_isaac_sim_dir()
        urdf_dir = Path(isaac_sim_dir) / "exts" / "isaacsim.asset.importer.urdf" / "data" / "urdf" / "robots"
        return raw.replace("{ISAAC_SIM_URDF_DIR}", str(urdf_dir))

    if "{ISAAC_SIM_MOTION_GEN_URDF_DIR}" in raw:
        isaac_sim_dir = get_default_isaac_sim_dir()
        urdf_dir = (
            Path(isaac_sim_dir)
            / "exts"
            / "isaacsim.robot_motion.motion_generation"
            / "motion_policy_configs"
            / "universal_robots"
        )
        return raw.replace("{ISAAC_SIM_MOTION_GEN_URDF_DIR}", str(urdf_dir))

    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return str(resolved.resolve())


def load_robot_config(robot_name: str) -> RobotSimConfig:
    """
    Load robot configuration from configs/robot_profiles/{robot_name}.yaml.

    Args:
        robot_name: One of "franka", "openarm", "ur10e", "so101" (`ur10` alias supported)

    Returns:
        RobotSimConfig populated from YAML
    """
    canonical_name = normalize_robot_name(robot_name, default=robot_name) or robot_name
    yaml_path = ROBOT_PROFILES_DIR / f"{canonical_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Robot profile not found: {yaml_path}. "
            f"Available: {[p.stem for p in ROBOT_PROFILES_DIR.glob('*.yaml')]}"
        )

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    robot = data.get("robot", {})
    joints = data.get("joints", {})
    gripper = data.get("gripper", {})
    kinematics = data.get("kinematics", {})
    asset = data.get("asset", {})
    spawn = data.get("spawn_ranges", {})

    # Extract arm joint info
    arm_section = joints.get("arm", {})
    arm_joint_names = arm_section.get("names", [])
    arm_limits = arm_section.get("limits", {})
    arm_vel_limits = arm_section.get("velocity_limits", {})

    # Extract finger joint info
    finger_section = joints.get("finger", {})
    finger_joint_names = finger_section.get("names", [])
    finger_limits = finger_section.get("limits", {})
    gripper_mirrored = finger_section.get("mirrored")  # None if not specified

    # Merge all joint limits
    joint_limits = {}
    for name, limits in arm_limits.items():
        joint_limits[name] = tuple(limits)
    for name, limits in finger_limits.items():
        joint_limits[name] = tuple(limits)

    # Gripper positions
    gripper_type = gripper.get("type", "parallel_jaw")
    gripper_open = gripper.get("open_position")
    gripper_close = gripper.get("close_position")

    # For parallel_jaw with mirrored fingers, open/close is per-finger
    if gripper_open is None:
        gripper_open = 0.0
    if gripper_close is None:
        gripper_close = 0.0

    # Camera defaults from spawn_ranges (legacy single camera)
    camera_cfg = spawn.get("camera", {})

    # Multi-camera configuration
    cameras_section = data.get("cameras", {})
    cameras: dict[str, CameraConfig] = {}
    for cam_name, cam_data in cameras_section.items():
        cam_type = cam_data.get("type", "fixed")
        res = cam_data.get("resolution", [640, 480])
        cameras[cam_name] = CameraConfig(
            name=cam_name,
            cam_type=cam_type,
            position=cam_data.get("position", []),
            target=cam_data.get("target", []),
            up_vector=cam_data.get("up_vector", []),
            parent_body=cam_data.get("parent_body", ""),
            offset_pos=cam_data.get("offset_pos", [0, 0, 0]),
            offset_rot=cam_data.get("offset_rot", [1, 0, 0, 0]),
            convention=cam_data.get("convention", "ros"),
            resolution=tuple(res),
        )

    # Stiffness/damping (can be dict per-group or scalar)
    stiffness = arm_section.get("stiffness", 80.0)
    damping = arm_section.get("damping", 4.0)
    if isinstance(stiffness, dict):
        stiffness = list(stiffness.values())[0]  # take first group value
    if isinstance(damping, dict):
        damping = list(damping.values())[0]

    return RobotSimConfig(
        name=robot.get("name", canonical_name),
        full_name=robot.get("full_name", robot_name),
        arm_dofs=robot.get("arm_dofs", len(arm_joint_names)),
        total_dofs=robot.get("total_dofs", len(arm_joint_names) + len(finger_joint_names)),
        arm_joint_names=arm_joint_names,
        finger_joint_names=finger_joint_names,
        joint_limits=joint_limits,
        velocity_limits=arm_vel_limits,
        gripper_type=gripper_type,
        gripper_open_position=gripper_open,
        gripper_close_position=gripper_close,
        ee_finger_offset=gripper.get("ee_finger_offset", 0.0),
        grasp_lateral_bias=gripper.get("lateral_bias", 0.0),
        gripper_close_duration=gripper.get("close_duration", 1.5),
        gripper_grasp_stiffness=gripper.get("grasp_stiffness", 0.0),
        ee_frame_body=asset.get("ee_frame", {}).get("body", ""),
        ee_frame_tcp=asset.get("ee_frame", {}).get("tcp_frame", ""),
        ee_frame_offset_position=asset.get("ee_frame", {}).get("offset_position", []),
        ik_ee_frame=asset.get("ee_frame", {}).get("ik_frame", asset.get("ee_frame", {}).get("body", "")),
        ready_pose=asset.get("ready_pose", {}),
        default_pose=asset.get("default_pose", {}),
        max_reach=kinematics.get("max_reach", 1.0),
        tabletop_reach=kinematics.get("tabletop_reach", 0.75),
        workspace_envelope=kinematics.get("workspace_envelope", {}),
        ik_backend=str(kinematics.get("ik_backend", "auto")),
        usd_path=asset.get("usd_path", ""),
        prim_path=asset.get("prim_path", "/World/Robot"),
        urdf_path=_resolve_urdf_path(asset.get("urdf_path", "")),
        camera_position=camera_cfg.get("position", [1.5, 1.2, 1.0]),
        camera_target=camera_cfg.get("target", [0.25, 0.0, 0.3]),
        cameras=cameras,
        arm_stiffness=stiffness,
        arm_damping=damping,
        _gripper_mirrored=gripper_mirrored,
    )


@dataclass
class DataCollectionConfig:
    """Pipeline configuration for data collection."""

    # Simulation control
    sim_dt: float = 0.01            # physics timestep (100Hz)
    decimation: int = 5             # control decimation (-> 20Hz control)
    control_hz: float = 20.0        # derived: sim_dt * decimation

    # Recording
    recording_fps: int = 20         # match control_hz by default
    front_video_fps: int = 20       # representative front-success video FPS
    camera_resolution: tuple[int, int] = (640, 480)
    dataset_cameras: list[str] = field(default_factory=lambda: ["top", "wrist", "front"])
    judge_cameras: list[str] = field(default_factory=lambda: ["wrist", "front"])

    # Episodes
    max_episodes: int = 50
    max_steps_per_episode: int = 600  # 30s at 20Hz
    target_successful_episodes: int = 0   # 0 = disabled (run max_episodes)
    max_total_attempts: int = 0           # 0 = auto (target * 5)

    # Dataset
    dataset_repo_id: str = "local/sim_dataset"
    output_dir: str = "outputs/data_collection"
    discard_failed_episodes: bool = True
    keep_failed_raw_dataset: bool = False

    # LLM (CaP code generation)
    llm_model: str = "gpt-5-mini"

    # VLM (success judging)
    use_vlm_judge: bool = True
    vlm_backend: str = "auto"
    vlm_model: str = "gpt-5"
    success_threshold: float = 0.8

    # Legacy compatibility
    skill_retry_max: int = 3        # retained for config compatibility; no longer used by the CaP path

    # IsaacLab environment
    env_headless: bool = True
    env_num_envs: int = 1           # single env for data collection
    execution_timeout: int = 1800   # seconds (30 min; first-run shader compilation takes ~10 min)
    ik_debug: bool = False          # emit verbose IK / grasp diagnostics to stdout

    @property
    def effective_fps(self) -> int:
        """Effective recording FPS (capped by control rate)."""
        return min(self.recording_fps, int(self.control_hz))

    @property
    def effective_front_video_fps(self) -> int:
        """Effective front-success video FPS (cannot exceed recording FPS)."""
        return min(self.front_video_fps, self.effective_fps)


def load_pipeline_config(config_path: Optional[str] = None) -> DataCollectionConfig:
    """
    Load pipeline configuration from YAML.

    Args:
        config_path: Path to config YAML. Defaults to configs/data_collection_config.yaml.

    Returns:
        DataCollectionConfig populated from YAML with defaults.
    """
    path = Path(config_path) if config_path else PIPELINE_CONFIG_PATH
    if not path.exists():
        return DataCollectionConfig()

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    sim = data.get("simulation", {})
    recording = data.get("recording", {})
    episodes = data.get("episodes", {})
    dataset = data.get("dataset", {})
    llm = data.get("llm", {})
    vlm = data.get("vlm", {})
    skill = data.get("skill", {})
    ik = data.get("ik", {})
    cameras = data.get("cameras", {})
    env = data.get("environment", {})

    config = DataCollectionConfig(
        sim_dt=sim.get("dt", 0.01),
        decimation=sim.get("decimation", 5),
        control_hz=sim.get("control_hz", 20.0),
        recording_fps=recording.get("fps", 20),
        front_video_fps=recording.get("front_video_fps", recording.get("fps", 20)),
        camera_resolution=tuple(recording.get("resolution", [640, 480])),
        dataset_cameras=list(cameras.get("dataset_cameras", ["top", "wrist", "front"])),
        judge_cameras=list(cameras.get("judge_cameras", ["wrist", "front"])),
        max_episodes=episodes.get("max", 50),
        max_steps_per_episode=episodes.get("max_steps", 600),
        target_successful_episodes=episodes.get("target_successful", 0),
        max_total_attempts=episodes.get("max_total_attempts", 0),
        dataset_repo_id=dataset.get("repo_id", "local/sim_dataset"),
        output_dir=dataset.get("output_dir", "outputs/data_collection"),
        discard_failed_episodes=bool(dataset.get("discard_failed_episodes", True)),
        keep_failed_raw_dataset=bool(dataset.get("keep_failed_raw_dataset", False)),
        llm_model=llm.get("model", "gpt-5-mini"),
        use_vlm_judge=vlm.get("enabled", True),
        vlm_backend=vlm.get("backend", "auto"),
        vlm_model=vlm.get("model", "gpt-5"),
        success_threshold=vlm.get("success_threshold", 0.8),
        skill_retry_max=skill.get("retry_max", 3),
        env_headless=env.get("headless", True),
        env_num_envs=env.get("num_envs", 1),
        execution_timeout=env.get("timeout", 600),
        ik_debug=bool(ik.get("debug", False)),
    )
    return config
