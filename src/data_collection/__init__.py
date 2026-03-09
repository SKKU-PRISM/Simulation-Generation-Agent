"""
Data collection module for IsaacLab simulation environments.

Adapts AutoDataCollector's CaP (Code-as-Policies) pipeline for simulation:
- Scene Graph object detection (replaces Grounding DINO)
- Forward-only CaP code generation and execution artifacts
- Pinocchio IK/FK motion planning (reused from ADC)
- Embodiment-specific skill runtimes via IsaacLab Articulation API
- LeRobot v3.0 dataset recording

Supports: Franka, OpenARM, UR10e, SO-101
"""

from .cap_generator import SimCaPGenerator
from .config import DataCollectionConfig, RobotSimConfig, load_robot_config
from .dataset_export import (
    ADC_COMPATIBLE_SCHEMA,
    CANONICAL_TRAINING_SCHEMA,
    export_dataset,
)
from .dataset_preprocess import preprocess_exported_dataset
from .lerobot_tools import (
    check_lerobot_dataset,
    convert_raw_dataset_to_lerobot,
    publish_lerobot_dataset,
)
from .sim_camera import SimCamera
from .sim_detector import SimDetector
from .sim_recorder import SimRecorder, convert_to_lerobot
from .sim_robot_interface import SimRobotInterface
from .sim_skills import SimSkills
from .sim_judge import SimJudge
from .pipeline import DataCollectionPipeline
