"""Kinematics and trajectory utilities (migrated from AutoDataCollector).

Imports are lazy — ``pinocchio`` is only required when
``KinematicsEngine`` is actually instantiated.
"""

from src.agent.kinematics.calibration_limits import (
    CalibrationJointLimits,
    load_calibration_limits,
    compare_limits,
    print_limits_comparison,
)

__all__ = [
    "KinematicsEngine",
    "CalibrationJointLimits",
    "load_calibration_limits",
    "compare_limits",
    "print_limits_comparison",
]


def __getattr__(name: str):
    if name == "KinematicsEngine":
        from src.agent.kinematics.engine import KinematicsEngine
        return KinematicsEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
