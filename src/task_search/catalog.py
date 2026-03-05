"""Task catalog: discovers and parses all task YAML files for search indexing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TaskEntry:
    """Metadata extracted from a single task YAML file."""

    yaml_path: str
    task_name: str
    task_description: str
    goal_description: str
    robot: str
    category: str
    asset_names: list[str] = field(default_factory=list)
    search_text: str = ""


class TaskCatalog:
    """Discovers and parses all task YAML files under the tasks directory."""

    EXCLUDED_DIRS = {"templates"}

    def __init__(self, tasks_dir: Path | None = None):
        self._tasks_dir = tasks_dir or (PROJECT_ROOT / "tasks")

    def discover(self) -> list[TaskEntry]:
        """Walk tasks directory, parse each YAML, return list of TaskEntry."""
        entries: list[TaskEntry] = []

        if not self._tasks_dir.exists():
            logger.warning("Tasks directory not found: %s", self._tasks_dir)
            return entries

        yaml_files = sorted(self._tasks_dir.rglob("*.yaml"))

        for yf in yaml_files:
            # Skip templates and non-task files
            if any(part in self.EXCLUDED_DIRS for part in yf.parts):
                continue
            try:
                entry = self._parse_yaml(yf)
                entry.search_text = self._build_search_text(entry)
                entries.append(entry)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", yf, e)

        logger.info("Discovered %d task YAMLs in %s", len(entries), self._tasks_dir)
        return entries

    def _parse_yaml(self, yaml_path: Path) -> TaskEntry:
        """Parse a single YAML file into a TaskEntry."""
        with open(yaml_path) as f:
            doc = yaml.safe_load(f) or {}

        task_sec = doc.get("task", {})
        goal_sec = doc.get("goal", {})
        assets = doc.get("assets", [])

        # Extract asset names (skip the robot articulation)
        asset_names: list[str] = []
        for asset in assets:
            if asset.get("type") == "articulation":
                continue
            name = asset.get("name", "")
            if name:
                asset_names.append(name)

        return TaskEntry(
            yaml_path=str(yaml_path.resolve()),
            task_name=task_sec.get("name", yaml_path.stem),
            task_description=task_sec.get("description", ""),
            goal_description=goal_sec.get("description", ""),
            robot=self._detect_robot_from_path(yaml_path),
            category=self._detect_category_from_path(yaml_path),
            asset_names=asset_names,
        )

    def _build_search_text(self, entry: TaskEntry) -> str:
        """Construct the text that will be embedded for semantic search."""
        parts = [
            f"Task: {entry.task_name}.",
            entry.task_description,
            f"Robot: {entry.robot}. Category: {entry.category}.",
        ]
        if entry.goal_description:
            parts.append(f"Goal: {entry.goal_description}")
        if entry.asset_names:
            parts.append(f"Objects: {', '.join(entry.asset_names)}.")
        return " ".join(parts)

    @staticmethod
    def _detect_robot_from_path(yaml_path: Path) -> str:
        """Extract robot name from path: tasks/{robot}/{category}/..."""
        try:
            # Find the 'tasks' directory in the path
            parts = yaml_path.parts
            for i, part in enumerate(parts):
                if part == "tasks" and i + 1 < len(parts):
                    return parts[i + 1]
        except (IndexError, ValueError):
            pass
        return "unknown"

    @staticmethod
    def _detect_category_from_path(yaml_path: Path) -> str:
        """Extract category from path: tasks/{robot}/{category}/..."""
        try:
            parts = yaml_path.parts
            for i, part in enumerate(parts):
                if part == "tasks" and i + 2 < len(parts):
                    return parts[i + 2]
        except (IndexError, ValueError):
            pass
        return "unknown"
