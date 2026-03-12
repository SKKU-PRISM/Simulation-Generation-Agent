"""Utilities for merging compatible simulator raw datasets."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


REQUIRED_COMPAT_KEYS = (
    "robot_name",
    "total_dofs",
    "joint_names",
    "arm_joint_names",
    "finger_joint_names",
    "camera_names",
    "fps",
)


def load_raw_dataset_metadata(raw_dataset_dir: str | Path) -> dict[str, Any]:
    """Load and return ``metadata.json`` from a raw dataset directory."""

    raw_dir = Path(raw_dataset_dir).expanduser().resolve()
    meta_path = raw_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in raw dataset: {raw_dir}")
    with open(meta_path) as f:
        return json.load(f)


def filter_raw_dataset_episodes(
    raw_dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    success_only: bool = False,
    selected_episode_indices: list[int] | None = None,
) -> Path:
    """Copy a raw dataset into a new directory with a filtered episode subset."""

    raw_dir = Path(raw_dataset_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    episodes_out_dir = output_root / "episodes"

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Filtered raw dataset output already exists: {output_root}")

    metadata = load_raw_dataset_metadata(raw_dir)
    requested_indices = {int(idx) for idx in selected_episode_indices or []}
    source_episodes = list(metadata.get("episodes", []))
    filtered_source_episodes = [
        episode
        for episode in source_episodes
        if (not success_only or bool(episode.get("success", False)))
        and (not requested_indices or int(episode.get("index", -1)) in requested_indices)
    ]

    output_root.mkdir(parents=True, exist_ok=True)
    episodes_out_dir.mkdir(parents=True, exist_ok=True)

    filtered_episodes: list[dict[str, Any]] = []
    for new_index, episode in enumerate(filtered_source_episodes):
        old_index = int(episode["index"])
        src_ep_dir = raw_dir / "episodes" / f"episode_{old_index:06d}"
        if not src_ep_dir.exists():
            raise FileNotFoundError(f"Episode directory missing during filter: {src_ep_dir}")

        dst_ep_dir = episodes_out_dir / f"episode_{new_index:06d}"
        shutil.copytree(src_ep_dir, dst_ep_dir)

        filtered_episode = dict(episode)
        filtered_episode["index"] = new_index
        filtered_episode["source_episode_index"] = old_index
        filtered_episodes.append(filtered_episode)

    filtered_metadata = dict(metadata)
    filtered_metadata["episodes"] = filtered_episodes
    filtered_metadata["total_episodes"] = len(filtered_episodes)
    filtered_metadata["successful_episodes"] = sum(
        1 for episode in filtered_episodes if episode.get("success")
    )
    filtered_metadata["filtered_from_raw_dataset"] = str(raw_dir)
    filtered_metadata["filter_success_only"] = bool(success_only)
    filtered_metadata["filtered_episode_indices"] = [
        int(episode["source_episode_index"]) for episode in filtered_episodes
    ]
    filtered_metadata["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    filtered_metadata["front_video_generated"] = False
    filtered_metadata["front_video_path"] = None
    filtered_metadata["front_video_episode"] = None
    filtered_metadata["front_video_success_type"] = None

    with open(output_root / "metadata.json", "w") as f:
        json.dump(filtered_metadata, f, indent=2)

    return output_root


def merge_raw_datasets(
    raw_dataset_dirs: list[str | Path],
    output_dir: str | Path,
    *,
    source_records: list[dict[str, Any]] | None = None,
) -> Path:
    """Merge compatible raw datasets into one renumbered raw dataset directory."""

    if not raw_dataset_dirs:
        raise ValueError("merge_raw_datasets requires at least one raw dataset")

    raw_dirs = [Path(path).expanduser().resolve() for path in raw_dataset_dirs]
    output_root = Path(output_dir).expanduser().resolve()
    episodes_out_dir = output_root / "episodes"

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Merged raw dataset output already exists: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    episodes_out_dir.mkdir(parents=True, exist_ok=True)

    if source_records is not None and len(source_records) != len(raw_dirs):
        raise ValueError("source_records length must match raw_dataset_dirs length")

    metadata_list = [load_raw_dataset_metadata(path) for path in raw_dirs]
    base_meta = metadata_list[0]
    _validate_compatible_raw_datasets(metadata_list, raw_dirs)

    merged_episodes: list[dict[str, Any]] = []
    merged_index = 0

    for dataset_idx, (raw_dir, metadata) in enumerate(zip(raw_dirs, metadata_list)):
        source = (source_records or [{}] * len(raw_dirs))[dataset_idx]
        episodes = metadata.get("episodes", [])
        for episode in episodes:
            source_episode_index = int(episode["index"])
            src_ep_dir = raw_dir / "episodes" / f"episode_{source_episode_index:06d}"
            if not src_ep_dir.exists():
                raise FileNotFoundError(f"Episode directory missing during merge: {src_ep_dir}")

            dst_ep_dir = episodes_out_dir / f"episode_{merged_index:06d}"
            shutil.copytree(src_ep_dir, dst_ep_dir)

            merged_episode = dict(episode)
            merged_episode["index"] = merged_index
            merged_episode["source_task"] = source.get("task")
            merged_episode["source_yaml"] = _stringify_optional_path(source.get("yaml_path"))
            merged_episode["source_output_dir"] = _stringify_optional_path(source.get("output_dir"))
            merged_episode["source_raw_dataset"] = str(raw_dir)
            merged_episode["source_episode_index"] = source_episode_index
            merged_episodes.append(merged_episode)
            merged_index += 1

    merged_metadata = dict(base_meta)
    merged_metadata["episodes"] = merged_episodes
    merged_metadata["total_episodes"] = len(merged_episodes)
    merged_metadata["successful_episodes"] = sum(
        1 for episode in merged_episodes if episode.get("success")
    )
    merged_metadata["front_video_generated"] = False
    merged_metadata["front_video_path"] = None
    merged_metadata["front_video_episode"] = None
    merged_metadata["front_video_success_type"] = None
    merged_metadata["merged_from_raw_datasets"] = [str(path) for path in raw_dirs]
    merged_metadata["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    with open(output_root / "metadata.json", "w") as f:
        json.dump(merged_metadata, f, indent=2)

    return output_root


def _validate_compatible_raw_datasets(
    metadata_list: list[dict[str, Any]],
    raw_dirs: list[Path],
) -> None:
    base_meta = metadata_list[0]
    for key in REQUIRED_COMPAT_KEYS:
        if key not in base_meta:
            raise KeyError(f"Missing '{key}' in raw dataset metadata: {raw_dirs[0]}")

    for raw_dir, metadata in zip(raw_dirs[1:], metadata_list[1:]):
        for key in REQUIRED_COMPAT_KEYS:
            if key not in metadata:
                raise KeyError(f"Missing '{key}' in raw dataset metadata: {raw_dir}")
            if metadata[key] != base_meta[key]:
                raise ValueError(
                    f"Incompatible raw datasets for merge on '{key}': "
                    f"{raw_dirs[0]}={base_meta[key]!r}, {raw_dir}={metadata[key]!r}"
                )


def _stringify_optional_path(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(Path(value).expanduser().resolve())
