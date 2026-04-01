"""Robot naming helpers shared across pipelines."""

from __future__ import annotations

from collections.abc import Iterable


ROBOT_ALIASES = {
    "franka": "franka",
    "panda": "franka",
    "openarm": "openarm",
    "so101": "so101",
    "so-101": "so101",
    "ur10": "ur10e",
    "ur10e": "ur10e",
    "ur_10": "ur10e",
    "ur_10e": "ur10e",
}


def normalize_robot_name(value: str | None, default: str | None = None) -> str | None:
    """Return the canonical robot identifier for a free-form string."""
    if not value:
        return default

    normalized = value.strip().lower().replace("-", "_")
    if normalized in ROBOT_ALIASES:
        return ROBOT_ALIASES[normalized]

    for token, canonical in ROBOT_ALIASES.items():
        if token in normalized:
            return canonical

    return default


def known_robot_names(include_aliases: bool = True) -> tuple[str, ...]:
    """Return canonical robot names, optionally including aliases."""
    canonical = tuple(dict.fromkeys(ROBOT_ALIASES.values()))
    if not include_aliases:
        return canonical
    ordered = list(canonical)
    for alias in ROBOT_ALIASES:
        if alias not in ordered:
            ordered.append(alias)
    return tuple(ordered)


def contains_robot_name(values: Iterable[str]) -> str | None:
    """Return the first canonical robot identifier found in the input strings."""
    for value in values:
        robot = normalize_robot_name(value)
        if robot:
            return robot
    return None
