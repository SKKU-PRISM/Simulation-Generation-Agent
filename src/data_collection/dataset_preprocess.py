"""Preprocess exported datasets into per-robot training manifests."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from src.data_collection.config import load_robot_config
from src.data_collection.dataset_export import ADC_COMPATIBLE_SCHEMA


def preprocess_exported_dataset(
    *,
    export_dir: str | Path,
    output_dir: str | Path,
    camera_names: list[str] | None = None,
    success_only: bool = False,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> Path:
    """Build per-frame manifests from an exported dataset.

    The input must be an ``adc_compatible`` export. The output is a model-agnostic
    manifest layer for downstream training code.
    """
    export_root = Path(export_dir)
    manifest_path = export_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Export manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        export_manifest = json.load(f)

    schema = export_manifest.get("schema")
    if schema != ADC_COMPATIBLE_SCHEMA:
        raise ValueError(
            f"preprocess_exported_dataset expects '{ADC_COMPATIBLE_SCHEMA}', got '{schema}'"
        )
    robot_cfg = load_robot_config(export_manifest["robot_name"])

    if not 0.0 < train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be in (0, 1], got {train_ratio}")

    preprocess_root = Path(output_dir) / f"{export_root.name}_preprocessed"
    if preprocess_root.exists():
        shutil.rmtree(preprocess_root)
    preprocess_root.mkdir(parents=True, exist_ok=True)

    selected_cameras = list(camera_names or export_manifest.get("camera_names") or [])
    episode_infos = [dict(ep) for ep in export_manifest.get("episodes", [])]
    if success_only:
        episode_infos = [ep for ep in episode_infos if ep.get("success", False)]

    episode_ids = [int(ep["episode_index"]) for ep in episode_infos]
    rng = random.Random(seed)
    rng.shuffle(episode_ids)

    if len(episode_ids) <= 1:
        train_episode_ids = episode_ids
        val_episode_ids: list[int] = []
    else:
        train_count = int(len(episode_ids) * train_ratio)
        train_count = max(1, min(len(episode_ids) - 1, train_count))
        train_episode_ids = sorted(episode_ids[:train_count])
        val_episode_ids = sorted(episode_ids[train_count:])

    split_by_episode = {ep_idx: "train" for ep_idx in train_episode_ids}
    split_by_episode.update({ep_idx: "val" for ep_idx in val_episode_ids})

    all_samples: list[dict[str, Any]] = []
    train_samples: list[dict[str, Any]] = []
    val_samples: list[dict[str, Any]] = []

    episodes_dir = export_root / "episodes"
    for ep_info in sorted(episode_infos, key=lambda item: int(item["episode_index"])):
        ep_idx = int(ep_info["episode_index"])
        ep_dir = episodes_dir / f"episode_{ep_idx:06d}"
        if not ep_dir.exists():
            continue

        split = split_by_episode.get(ep_idx, "train")
        task_description = str(ep_info.get("task_description", ""))
        num_frames = _resolve_num_frames(ep_dir, fallback=int(ep_info.get("num_frames", 0)))
        skill_labels = _load_json_list(ep_dir / "skill.natural_language.json", num_frames)
        skill_types = _load_json_list(ep_dir / "skill.type.json", num_frames)
        available_cameras = _discover_episode_cameras(ep_dir)
        episode_camera_names = (
            selected_cameras if selected_cameras else available_cameras
        )

        for frame_idx in range(num_frames):
            sample = {
                "sample_id": f"episode_{ep_idx:06d}:{frame_idx:06d}",
                "robot_name": export_manifest["robot_name"],
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "split": split,
                "success": bool(ep_info.get("success", False)),
                "task_description": task_description,
                "skill_label": skill_labels[frame_idx],
                "skill_type": skill_types[frame_idx],
                "paths": {
                    "state": _rel(ep_dir / "observation.state.npy", export_root),
                    "gripper_state": _rel(
                        ep_dir / "observation.gripper_state.npy",
                        export_root,
                    ),
                    "tcp_robot_xyzrpy": _rel(
                        ep_dir / "observation.tcp.robot_xyzrpy.npy",
                        export_root,
                    ),
                    "action": _rel(ep_dir / "action.npy", export_root),
                    "goal_joint": _rel(ep_dir / "skill.goal_position.joint.npy", export_root),
                    "goal_world_xyzrpy": _rel(
                        ep_dir / "skill.goal_position.world_xyzrpy.npy",
                        export_root,
                    ),
                    "goal_robot_xyzrpy": _rel(
                        ep_dir / "skill.goal_position.robot_xyzrpy.npy",
                        export_root,
                    ),
                    "goal_gripper": _rel(
                        ep_dir / "skill.goal_position.gripper.npy",
                        export_root,
                    ),
                },
                "images": _build_image_path_map(
                    ep_dir=ep_dir,
                    export_root=export_root,
                    frame_idx=frame_idx,
                    camera_names=episode_camera_names,
                ),
            }
            all_samples.append(sample)
            if split == "train":
                train_samples.append(sample)
            else:
                val_samples.append(sample)

    resolved_camera_names = (
        selected_cameras
        if selected_cameras
        else _discover_all_cameras(export_root, episode_infos)
    )

    preprocess_manifest = {
        "source_export_root": str(export_root.resolve()),
        "schema": schema,
        "robot_name": export_manifest["robot_name"],
        "joint_names": export_manifest.get("joint_names", []),
        "arm_joint_names": export_manifest.get("arm_joint_names") or robot_cfg.arm_joint_names,
        "finger_joint_names": export_manifest.get("finger_joint_names") or robot_cfg.finger_joint_names,
        "gripper_type": export_manifest.get("gripper_type") or robot_cfg.gripper_type,
        "gripper_state_dim": int(export_manifest.get("gripper_state_dim", 1)),
        "joint_shape_policy": export_manifest.get("joint_shape_policy", "full_controllable_dofs"),
        "selected_camera_names": resolved_camera_names,
        "success_only": success_only,
        "train_ratio": float(train_ratio),
        "seed": int(seed),
        "num_episodes": len(episode_infos),
        "num_samples": len(all_samples),
        "num_train_samples": len(train_samples),
        "num_val_samples": len(val_samples),
        "train_episode_indices": train_episode_ids,
        "val_episode_indices": val_episode_ids,
    }

    _write_json(preprocess_root / "manifest.json", preprocess_manifest)
    _write_jsonl(preprocess_root / "samples.jsonl", all_samples)
    _write_jsonl(preprocess_root / "train.jsonl", train_samples)
    _write_jsonl(preprocess_root / "val.jsonl", val_samples)
    return preprocess_root


def _resolve_num_frames(ep_dir: Path, *, fallback: int) -> int:
    action_path = ep_dir / "action.npy"
    if action_path.exists():
        return int(np.load(action_path, mmap_mode="r").shape[0])
    return int(fallback)


def _load_json_list(path: Path, length: int) -> list[str]:
    if not path.exists():
        return [""] * length
    with open(path) as f:
        values = json.load(f)
    if not isinstance(values, list):
        return [""] * length
    items = [str(v) for v in values[:length]]
    if len(items) < length:
        items.extend([""] * (length - len(items)))
    return items


def _discover_episode_cameras(ep_dir: Path) -> list[str]:
    images_dir = ep_dir / "images"
    if not images_dir.exists():
        return []
    return sorted(p.name for p in images_dir.iterdir() if p.is_dir())


def _discover_all_cameras(export_root: Path, episode_infos: list[dict[str, Any]]) -> list[str]:
    camera_names: set[str] = set()
    for ep_info in episode_infos:
        ep_idx = int(ep_info["episode_index"])
        ep_dir = export_root / "episodes" / f"episode_{ep_idx:06d}"
        camera_names.update(_discover_episode_cameras(ep_dir))
    return sorted(camera_names)


def _build_image_path_map(
    *,
    ep_dir: Path,
    export_root: Path,
    frame_idx: int,
    camera_names: list[str],
) -> dict[str, str]:
    image_paths: dict[str, str] = {}
    for camera_name in camera_names:
        image_path = ep_dir / "images" / camera_name / f"{frame_idx:06d}.png"
        if image_path.exists():
            image_paths[camera_name] = _rel(image_path, export_root)
    return image_paths


def _rel(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _write_json(path: Path, value: Any) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
