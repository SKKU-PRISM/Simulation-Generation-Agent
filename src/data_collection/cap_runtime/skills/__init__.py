"""Embodiment-specific skill wrappers for generated CaP code."""

from .skills_franka import FrankaSkills
from .skills_openarm import OpenArmSkills
from .skills_lerobot import LeRobotSkills
from .skills_ur10e import UR10eSkills

__all__ = ["FrankaSkills", "OpenArmSkills", "LeRobotSkills", "UR10eSkills"]
