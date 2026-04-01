"""Configuration for task search module."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class TaskSearchConfig:
    """Task search configuration."""

    # Embedding backend: "auto", "local", "azure", "tfidf"
    embedding_backend: str = "auto"
    local_model_name: str = "all-MiniLM-L6-v2"
    azure_embedding_model: str = "text-embedding-3-small"

    # Search defaults
    default_top_k: int = 5
    min_score: float = 0.3
    auto_select_threshold: float = 0.7

    # Paths
    cache_dir: str = ".cache/task_search"
    tasks_dir: str = "tasks"

    # Display
    show_scores: bool = True

    @property
    def abs_cache_dir(self) -> Path:
        p = Path(self.cache_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def abs_tasks_dir(self) -> Path:
        p = Path(self.tasks_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_search_config(config_path: Optional[str] = None) -> TaskSearchConfig:
    """Load task search config from YAML, falling back to defaults."""
    cfg = TaskSearchConfig()

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "task_search_config.yaml")

    path = Path(config_path)
    if not path.exists():
        logger.debug("Config not found at %s, using defaults", path)
        return cfg

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    embedding = data.get("embedding", {})
    if "backend" in embedding:
        cfg.embedding_backend = embedding["backend"]
    if "local_model" in embedding:
        cfg.local_model_name = embedding["local_model"]
    if "azure_model" in embedding:
        cfg.azure_embedding_model = embedding["azure_model"]

    search = data.get("search", {})
    if "default_top_k" in search:
        cfg.default_top_k = search["default_top_k"]
    if "min_score" in search:
        cfg.min_score = search["min_score"]
    if "auto_select_threshold" in search:
        cfg.auto_select_threshold = search["auto_select_threshold"]

    cache = data.get("cache", {})
    if "dir" in cache:
        cfg.cache_dir = cache["dir"]

    if "tasks_dir" in data:
        cfg.tasks_dir = data["tasks_dir"]

    return cfg
