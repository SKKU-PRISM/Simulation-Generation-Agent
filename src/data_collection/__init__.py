"""
Data collection module for IsaacLab simulation environments.

Adapts AutoDataCollector's CaP (Code-as-Policies) pipeline for simulation:
- Scene Graph object detection (replaces Grounding DINO)
- Pinocchio IK/FK motion planning (reused from ADC)
- Skill-based robot control via IsaacLab Articulation API
- LeRobot v3.0 dataset recording

Supports: Franka, OpenARM, UR10, SO-101
"""

from .config import DataCollectionConfig, RobotSimConfig, load_robot_config
from .sim_camera import SimCamera
from .sim_detector import SimDetector
from .sim_recorder import SimRecorder, convert_to_lerobot
from .sim_robot_interface import SimRobotInterface
from .sim_skills import SimSkills
from .skill_planner import SkillPlanner
from .sim_judge import SimJudge
from .pipeline import DataCollectionPipeline
