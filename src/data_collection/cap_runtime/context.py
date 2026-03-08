"""Shared runtime context used by generated CaP code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RuntimeResources:
    """Objects exposed to the embodiment-specific CaP runtime wrappers."""

    sim_skills: Any
    detector: Any
    robot_cfg: Any
    translated_positions: dict[str, dict[str, Any]]


class CaPRuntimeContext:
    """Singleton-like holder for the active episode runtime resources."""

    _resources: RuntimeResources | None = None

    @classmethod
    def configure(
        cls,
        *,
        sim_skills: Any,
        detector: Any,
        robot_cfg: Any,
        translated_positions: dict[str, dict[str, Any]],
    ) -> None:
        cls._resources = RuntimeResources(
            sim_skills=sim_skills,
            detector=detector,
            robot_cfg=robot_cfg,
            translated_positions=translated_positions,
        )

    @classmethod
    def get(cls) -> RuntimeResources:
        if cls._resources is None:
            raise RuntimeError("CaPRuntimeContext is not configured for this episode")
        return cls._resources

    @classmethod
    def clear(cls) -> None:
        cls._resources = None
