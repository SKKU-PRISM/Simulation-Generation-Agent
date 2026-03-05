"""
Centralized import bridge for AutoDataCollector (ADC) submodule.

ADC is a git submodule at external/AutoDataCollector/. This module provides
clean import functions with graceful fallback when ADC is not initialized.

Problem: ADC's ``lerobot_cap/__init__.py`` imports hardware modules
(FeetechController, DynamixelController) which require ``feetech-servo-sdk``.
Since ADC is read-only, we use ``importlib`` to load individual ``.py`` files
directly, bypassing the package ``__init__.py`` chain.

The ``judge/`` package has no hardware dependencies and can be imported
normally via ``sys.path``.

Usage::

    from src.data_collection.adc_imports import get_kinematics_engine

    KinematicsEngine = get_kinematics_engine()  # raises ImportError if unavailable
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root: two levels up from this file (src/data_collection/adc_imports.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ADC location: env override > submodule
_ADC_PATH_OVERRIDE = os.environ.get("ADC_PATH")
if _ADC_PATH_OVERRIDE:
    _ADC_ROOT = Path(_ADC_PATH_OVERRIDE)
else:
    _ADC_ROOT = _PROJECT_ROOT / "external" / "AutoDataCollector"

_ADC_SRC = _ADC_ROOT / "src"


# ------------------------------------------------------------------ #
# Public: availability check
# ------------------------------------------------------------------ #


def is_adc_available() -> bool:
    """Check if ADC submodule is initialized and accessible."""
    return (
        _ADC_SRC.exists()
        and (_ADC_SRC / "lerobot_cap" / "kinematics" / "engine.py").is_file()
    )


def get_adc_root() -> Path:
    """Return the ADC repository root (for assets/urdf/ etc)."""
    if not _ADC_ROOT.exists():
        raise ImportError(
            f"ADC not found at {_ADC_ROOT}. "
            "Run: git submodule update --init external/AutoDataCollector"
        )
    return _ADC_ROOT


# ------------------------------------------------------------------ #
# Internal: importlib file loader (bypasses __init__.py)
# ------------------------------------------------------------------ #


def _load_module_from_file(module_name: str, file_path: Path) -> Any:
    """Load a Python module directly from file path.

    This bypasses the package ``__init__.py`` chain, avoiding hardware
    dependency imports in ``lerobot_cap/__init__.py``.
    """
    if not file_path.is_file():
        raise ImportError(f"Module file not found: {file_path}")

    # Return cached module if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ #
# lerobot_cap imports (importlib — bypass hardware __init__.py)
# ------------------------------------------------------------------ #


def get_kinematics_engine():
    """Import and return ``KinematicsEngine`` class from ADC.

    Requires: ``pinocchio`` (pin) package installed.

    Raises:
        ImportError: If ADC submodule or pinocchio is not available.
    """
    engine_path = _ADC_SRC / "lerobot_cap" / "kinematics" / "engine.py"
    mod = _load_module_from_file("_adc_kinematics_engine", engine_path)
    return mod.KinematicsEngine


def get_interpolation_module():
    """Import and return ADC's interpolation module.

    Available functions on the returned module:
    - ``smoothstep(t)``
    - ``smooth_linear_interpolation(start, end, num_points, smooth_type)``
    - ``s_curve_profile(t, accel_ratio)``
    - ``s_curve_interpolation(start, end, num_points, accel_ratio)``
    - ``linear_interpolation(start, end, num_points)``
    - ``cubic_interpolation(start, end, num_points)``

    Requires: ``scipy`` package installed.

    Raises:
        ImportError: If ADC submodule or scipy is not available.
    """
    interp_path = _ADC_SRC / "lerobot_cap" / "planning" / "interpolation.py"
    return _load_module_from_file("_adc_interpolation", interp_path)


# ------------------------------------------------------------------ #
# judge imports (sys.path — no hardware dependencies)
# ------------------------------------------------------------------ #

_judge_path_registered = False


def _ensure_judge_path():
    """Add ADC root to sys.path for judge package import."""
    global _judge_path_registered
    if _judge_path_registered:
        return

    if not _ADC_ROOT.exists():
        raise ImportError(
            f"ADC not found at {_ADC_ROOT}. "
            "Run: git submodule update --init external/AutoDataCollector"
        )

    adc_root_str = str(_ADC_ROOT)
    if adc_root_str not in sys.path:
        sys.path.insert(0, adc_root_str)
    _judge_path_registered = True


def get_judge_prompts():
    """Import ADC's judge prompt templates.

    Returns:
        Tuple of (JUDGE_SYSTEM_PROMPT: str, build_judge_prompt: callable)
    """
    _ensure_judge_path()
    from judge.forward_execution.prompt import (  # type: ignore[import-untyped]
        JUDGE_SYSTEM_PROMPT,
        build_judge_prompt,
    )
    return JUDGE_SYSTEM_PROMPT, build_judge_prompt


def get_task_judge():
    """Import and return ``TaskJudge`` class from ADC's judge module.

    Note: For SGA usage, prefer ``SimJudge`` (sim_judge.py) which uses
    ADC prompts with SGA's ``AzureOpenAIClient`` instead of ADC's
    direct OpenAI calls.
    """
    _ensure_judge_path()
    from judge.forward_execution.judge import TaskJudge  # type: ignore[import-untyped]
    return TaskJudge
