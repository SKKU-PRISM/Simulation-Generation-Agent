"""
Import bridge for kinematics and judge utilities.

Previously loaded from the ADC submodule (external/AutoDataCollector/).
Now uses SGA's internal ``src/kinematics/`` and ``src/common/judge_prompts``
packages directly.

Usage::

    from src.agent.data_collection.adc_imports import get_kinematics_engine

    KinematicsEngine = get_kinematics_engine()
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------ #
# Public: availability check
# ------------------------------------------------------------------ #


def is_adc_available() -> bool:
    """Always True — kinematics are now internal to SGA."""
    return True


def get_adc_root() -> Path:
    """Return the SGA project root (replaces former ADC root)."""
    return _PROJECT_ROOT


# ------------------------------------------------------------------ #
# Kinematics imports (src/kinematics/)
# ------------------------------------------------------------------ #


def get_kinematics_engine():
    """Return ``KinematicsEngine`` class.

    Requires: ``pinocchio`` (pin) package installed.
    """
    from src.agent.kinematics.engine import KinematicsEngine
    return KinematicsEngine


def get_calibration_limits_loader():
    """Return ``load_calibration_limits`` helper."""
    from src.agent.kinematics.calibration_limits import load_calibration_limits
    return load_calibration_limits


def get_frame_transformer():
    """Return ``FrameTransformer`` class."""
    from src.agent.kinematics.transforms import FrameTransformer
    return FrameTransformer


def get_interpolation_module():
    """Return the interpolation module.

    Available functions:
    - ``smoothstep(t)``
    - ``smooth_linear_interpolation(start, end, num_points, smooth_type)``
    - ``s_curve_profile(t, accel_ratio)``
    - ``s_curve_interpolation(start, end, num_points, accel_ratio)``
    - ``linear_interpolation(start, end, num_points)``
    - ``cubic_interpolation(start, end, num_points)``

    Requires: ``scipy`` package installed.
    """
    from src.agent.kinematics import interpolation
    return interpolation


# ------------------------------------------------------------------ #
# Judge prompt imports (src/common/judge_prompts.py)
# ------------------------------------------------------------------ #


def get_judge_prompts():
    """Return judge prompt templates.

    Returns:
        Tuple of (JUDGE_SYSTEM_PROMPT: str, build_judge_prompt: callable)
    """
    from src.agent.common.judge_prompts import (
        JUDGE_SYSTEM_PROMPT,
        build_judge_prompt,
    )
    return JUDGE_SYSTEM_PROMPT, build_judge_prompt


def get_task_judge():
    """Not available — use ``SimJudge`` (sim_judge.py) instead.

    Raises NotImplementedError as TaskJudge was ADC-specific and is
    replaced by SGA's SimJudge which uses Azure OpenAI.
    """
    raise NotImplementedError(
        "TaskJudge is no longer available. Use SimJudge from sim_judge.py instead."
    )
