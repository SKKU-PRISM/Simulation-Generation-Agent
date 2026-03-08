"""
Dataset export utilities for aligning simulator data with ADC/LeRobot semantics.

This module keeps the raw source datasets intact and writes derived exports:

- ``adc_compatible``: preserves the legacy ADC feature names and normalization.
- ``canonical_training``: clean training schema with explicit TCP current/goal poses.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.common.robot_names import normalize_robot_name
from src.data_collection.config import RobotSimConfig, load_robot_config
from src.data_collection.sim_recorder import _coerce_goal_joint

logger = logging.getLogger(__name__)

ADC_COMPATIBLE_SCHEMA = "adc_compatible"
CANONICAL_TRAINING_SCHEMA = "canonical_training"
SUPPORTED_EXPORT_SCHEMAS = (ADC_COMPATIBLE_SCHEMA, CANONICAL_TRAINING_SCHEMA)
SCHEMA_VERSION = "2026-03-08"


@dataclass(frozen=True)
class JointLimitEntry:
    name: str
    lower: float  # value mapped to -100
    upper: float  # value mapped to +100
    source: str


@dataclass(frozen=True)
class JointNormalizationSpec:
    joint_names: tuple[str, ...]
    entries: tuple[JointLimitEntry, ...]
    gripper_open: float
    gripper_close: float
    gripper_source: str


@dataclass
class _ADCExportContext:
    robot_name: str
    joint_names: tuple[str, ...]
    arm_dofs: int
    frame_name: str
    calibration_limits: Any
    kinematics: Any
    frame_transformer: Any


def export_dataset(
    *,
    source_path: str | Path,
    source_type: str,
    schema: str,
    output_dir: str | Path,
    robot_name: str | None = None,
    adc_robot_config: str | Path | None = None,
    link_images: bool = True,
) -> Path:
    """Export a raw dataset into a normalized training-compatible schema."""
    if schema not in SUPPORTED_EXPORT_SCHEMAS:
        raise ValueError(
            f"Unsupported schema '{schema}'. Expected one of {SUPPORTED_EXPORT_SCHEMAS}."
        )

    source = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if source_type == "sim_raw":
        return export_sim_raw_dataset(
            raw_dataset_dir=source,
            schema=schema,
            output_dir=output,
            robot_name=robot_name,
            link_images=link_images,
        )
    if source_type == "adc_raw":
        return export_adc_raw_dataset(
            adc_dataset_dir=source,
            schema=schema,
            output_dir=output,
            robot_name=robot_name,
            adc_robot_config=adc_robot_config,
            link_images=link_images,
        )
    raise ValueError("source_type must be 'sim_raw' or 'adc_raw'")


def export_sim_raw_dataset(
    *,
    raw_dataset_dir: str | Path,
    schema: str,
    output_dir: str | Path,
    robot_name: str | None = None,
    link_images: bool = True,
) -> Path:
    raw_dir = Path(raw_dataset_dir)
    with open(raw_dir / "metadata.json") as f:
        metadata = json.load(f)

    canonical_robot = normalize_robot_name(
        robot_name or metadata.get("robot_name"),
        default=metadata.get("robot_name"),
    )
    if not canonical_robot:
        raise ValueError("Could not resolve robot name for sim raw dataset export")

    robot_cfg = load_robot_config(canonical_robot)
    joint_names = tuple(metadata.get("joint_names") or robot_cfg.all_joint_names)
    norm_spec = build_joint_normalization_spec(robot_cfg, joint_names=joint_names)
    camera_names = list(metadata.get("camera_names") or [])

    export_root = Path(output_dir) / f"{raw_dir.parent.name}_{raw_dir.name}_{schema}"
    if export_root.exists():
        shutil.rmtree(export_root)
    (export_root / "episodes").mkdir(parents=True, exist_ok=True)

    episode_summaries: list[dict[str, Any]] = []
    for ep_info in metadata.get("episodes", []):
        ep_idx = int(ep_info["index"])
        ep_dir = raw_dir / "episodes" / f"episode_{ep_idx:06d}"
        if not ep_dir.exists():
            logger.warning("Skipping missing episode directory: %s", ep_dir)
            continue

        export_ep_dir = export_root / "episodes" / ep_dir.name
        export_ep_dir.mkdir(parents=True, exist_ok=True)

        states = np.load(ep_dir / "states.npy").astype(np.float32)
        actions = np.load(ep_dir / "actions.npy").astype(np.float32)
        tcp_world = _load_pose_array(ep_dir / "tcp_world_xyzrpy.npy", length=states.shape[0])
        tcp_robot = _load_pose_array(ep_dir / "tcp_robot_xyzrpy.npy", length=states.shape[0])

        with open(ep_dir / "skills.json") as f:
            skills = json.load(f)

        T = states.shape[0]
        progress = np.zeros((T, 1), dtype=np.float32)
        goal_joint_native = np.zeros((T, len(joint_names)), dtype=np.float32)
        goal_world = np.zeros((T, 6), dtype=np.float32)
        goal_robot = np.zeros((T, 6), dtype=np.float32)
        goal_gripper_native = np.zeros((T, 1), dtype=np.float32)
        labels: list[str] = []
        types: list[str] = []

        for t in range(T):
            skill = skills[t] if t < len(skills) else {}
            labels.append(str(skill.get("label", "")))
            types.append(str(skill.get("type", "")))
            progress[t, 0] = float(skill.get("progress", 0.0))
            goal_joint_native[t] = _coerce_goal_joint(
                skill.get("goal_joint"),
                total_dofs=len(joint_names),
            )
            goal_world[t] = _as_pose_array(skill.get("goal_world_xyzrpy"))
            goal_robot[t] = _as_pose_array(skill.get("goal_robot_xyzrpy"))
            goal_gripper_native[t, 0] = float(skill.get("goal_gripper", 0.0))

        np.save(export_ep_dir / "observation.state.npy", normalize_joint_matrix(states, norm_spec))
        np.save(export_ep_dir / "action.npy", normalize_joint_matrix(actions, norm_spec))
        np.save(
            export_ep_dir / "skill.goal_position.joint.npy",
            normalize_joint_matrix(goal_joint_native, norm_spec),
        )
        np.save(
            export_ep_dir / "skill.goal_position.gripper.npy",
            normalize_gripper_array(goal_gripper_native, norm_spec),
        )
        np.save(export_ep_dir / "skill.progress.npy", progress)
        _write_json(export_ep_dir / "skill.natural_language.json", labels)
        _write_json(export_ep_dir / "skill.type.json", types)

        if schema == ADC_COMPATIBLE_SCHEMA:
            legacy_world = compose_adc_legacy_world_xyzrpy(goal_world, goal_robot)
            np.save(export_ep_dir / "skill.goal_position.world_xyzrpy.npy", legacy_world)
            np.save(export_ep_dir / "skill.goal_position.robot_xyzrpy.npy", goal_robot)
        else:
            np.save(export_ep_dir / "observation.tcp.world_xyzrpy.npy", tcp_world)
            np.save(export_ep_dir / "observation.tcp.robot_xyzrpy.npy", tcp_robot)
            np.save(export_ep_dir / "skill.goal_position.tcp.world_xyzrpy.npy", goal_world)
            np.save(export_ep_dir / "skill.goal_position.tcp.robot_xyzrpy.npy", goal_robot)

        if link_images:
            _link_sim_images(ep_dir / "images", export_ep_dir / "images")

        episode_summaries.append(
            {
                "episode_index": ep_idx,
                "num_frames": int(T),
                "success": bool(ep_info.get("success", False)),
                "has_images": bool(ep_info.get("has_images", False)),
                "task_description": str(ep_info.get("task_description", "")),
            }
        )

    manifest = build_export_manifest(
        schema=schema,
        source_type="sim_raw",
        source_path=raw_dir,
        robot_cfg=robot_cfg,
        norm_spec=norm_spec,
        episodes=episode_summaries,
        image_mode="frame_dirs" if link_images else "none",
        camera_names=camera_names,
    )
    _write_json(export_root / "manifest.json", manifest)
    return export_root


def export_adc_raw_dataset(
    *,
    adc_dataset_dir: str | Path,
    schema: str,
    output_dir: str | Path,
    robot_name: str | None = None,
    adc_robot_config: str | Path | None = None,
    link_images: bool = True,
) -> Path:
    adc_root = Path(adc_dataset_dir)
    rows = load_adc_dataset_rows(adc_root)
    ctx = load_adc_export_context(
        robot_name=robot_name,
        adc_robot_config=adc_robot_config,
    )
    robot_cfg = load_robot_config(ctx.robot_name)
    norm_spec = build_joint_normalization_spec(
        robot_cfg,
        joint_names=ctx.joint_names,
    )

    export_root = Path(output_dir) / f"{adc_root.name}_{schema}"
    if export_root.exists():
        shutil.rmtree(export_root)
    (export_root / "episodes").mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row.get("episode_index", 0))].append(row)

    episode_summaries: list[dict[str, Any]] = []
    for ep_idx, ep_rows in sorted(grouped.items()):
        export_ep_dir = export_root / "episodes" / f"episode_{ep_idx:06d}"
        export_ep_dir.mkdir(parents=True, exist_ok=True)

        obs_state = np.stack(
            [_coerce_vector(row.get("observation.state"), len(ctx.joint_names)) for row in ep_rows]
        ).astype(np.float32)
        action = np.stack(
            [_coerce_vector(row.get("action"), len(ctx.joint_names)) for row in ep_rows]
        ).astype(np.float32)
        goal_joint = np.stack(
            [
                _coerce_vector(
                    row.get("skill.goal_position.joint"),
                    len(ctx.joint_names),
                )
                for row in ep_rows
            ]
        ).astype(np.float32)
        goal_gripper = np.asarray(
            [[_coerce_scalar(row.get("skill.goal_position.gripper"))] for row in ep_rows],
            dtype=np.float32,
        )
        progress = np.asarray(
            [[_coerce_scalar(row.get("skill.progress"))] for row in ep_rows],
            dtype=np.float32,
        )
        labels = [str(row.get("skill.natural_language", "")) for row in ep_rows]
        types = [str(row.get("skill.type", "")) for row in ep_rows]

        np.save(export_ep_dir / "observation.state.npy", obs_state)
        np.save(export_ep_dir / "action.npy", action)
        np.save(export_ep_dir / "skill.goal_position.joint.npy", goal_joint)
        np.save(export_ep_dir / "skill.goal_position.gripper.npy", goal_gripper)
        np.save(export_ep_dir / "skill.progress.npy", progress)
        _write_json(export_ep_dir / "skill.natural_language.json", labels)
        _write_json(export_ep_dir / "skill.type.json", types)

        if schema == ADC_COMPATIBLE_SCHEMA:
            goal_world = np.stack(
                [_coerce_pose_array_from_row(row.get("skill.goal_position.world_xyzrpy")) for row in ep_rows]
            ).astype(np.float32)
            goal_robot = np.stack(
                [_coerce_pose_array_from_row(row.get("skill.goal_position.robot_xyzrpy")) for row in ep_rows]
            ).astype(np.float32)
            np.save(export_ep_dir / "skill.goal_position.world_xyzrpy.npy", goal_world)
            np.save(export_ep_dir / "skill.goal_position.robot_xyzrpy.npy", goal_robot)
        else:
            tcp_world = np.stack(
                [compute_adc_tcp_world_xyzrpy(ctx, row["observation.state"]) for row in ep_rows]
            ).astype(np.float32)
            tcp_robot = np.stack(
                [compute_adc_tcp_robot_xyzrpy(ctx, row["observation.state"]) for row in ep_rows]
            ).astype(np.float32)
            goal_tcp_world = np.stack(
                [compute_adc_tcp_world_xyzrpy(ctx, row["skill.goal_position.joint"]) for row in ep_rows]
            ).astype(np.float32)
            goal_tcp_robot = np.stack(
                [compute_adc_tcp_robot_xyzrpy(ctx, row["skill.goal_position.joint"]) for row in ep_rows]
            ).astype(np.float32)

            np.save(export_ep_dir / "observation.tcp.world_xyzrpy.npy", tcp_world)
            np.save(export_ep_dir / "observation.tcp.robot_xyzrpy.npy", tcp_robot)
            np.save(export_ep_dir / "skill.goal_position.tcp.world_xyzrpy.npy", goal_tcp_world)
            np.save(export_ep_dir / "skill.goal_position.tcp.robot_xyzrpy.npy", goal_tcp_robot)

        episode_summaries.append(
            {
                "episode_index": ep_idx,
                "num_frames": len(ep_rows),
                "success": False,
                "has_images": bool((adc_root / "videos").exists()),
                "task_description": "",
            }
        )

    if link_images and (adc_root / "videos").exists():
        videos_link = export_root / "videos"
        videos_link.symlink_to((adc_root / "videos").resolve(), target_is_directory=True)

    manifest = build_export_manifest(
        schema=schema,
        source_type="adc_raw",
        source_path=adc_root,
        robot_cfg=robot_cfg,
        norm_spec=norm_spec,
        episodes=episode_summaries,
        image_mode="lerobot_videos" if link_images and (adc_root / "videos").exists() else "none",
        camera_names=["front"] if link_images and (adc_root / "videos").exists() else [],
    )
    _write_json(export_root / "manifest.json", manifest)
    return export_root


def build_joint_normalization_spec(
    robot_cfg: RobotSimConfig,
    *,
    joint_names: tuple[str, ...] | list[str] | None = None,
) -> JointNormalizationSpec:
    names = tuple(joint_names or robot_cfg.all_joint_names)
    entries: list[JointLimitEntry] = []
    finger_names = set(robot_cfg.finger_joint_names)

    open_positions = _expand_positions(robot_cfg.gripper_open_position, len(robot_cfg.finger_joint_names))
    close_positions = _expand_positions(robot_cfg.gripper_close_position, len(robot_cfg.finger_joint_names))
    finger_map = {
        name: (open_positions[i], close_positions[i])
        for i, name in enumerate(robot_cfg.finger_joint_names)
        if i < len(open_positions) and i < len(close_positions)
    }

    for name in names:
        if name in finger_map:
            open_pos, close_pos = finger_map[name]
            lower, upper = float(close_pos), float(open_pos)
            source = "gripper_open_close"
        elif name in robot_cfg.joint_limits:
            lower, upper = robot_cfg.joint_limits[name]
            source = "profile_joint_limits"
        elif name in finger_names:
            lower, upper = -1.0, 1.0
            source = "fallback_unit"
        else:
            lower, upper = -math.pi, math.pi
            source = "fallback_pi"
        if math.isclose(lower, upper, abs_tol=1e-8):
            lower -= 1.0
            upper += 1.0
            source = f"{source}_expanded"
        entries.append(JointLimitEntry(name=name, lower=float(lower), upper=float(upper), source=source))

    gripper_open = float(_expand_positions(robot_cfg.gripper_open_position, 1)[0])
    gripper_close = float(_expand_positions(robot_cfg.gripper_close_position, 1)[0])
    gripper_source = "gripper_open_close"
    if math.isclose(gripper_open, gripper_close, abs_tol=1e-8):
        gripper_close = gripper_open - 1.0
        gripper_source = "gripper_open_close_expanded"

    return JointNormalizationSpec(
        joint_names=names,
        entries=tuple(entries),
        gripper_open=gripper_open,
        gripper_close=gripper_close,
        gripper_source=gripper_source,
    )


def normalize_joint_matrix(values: np.ndarray, norm_spec: JointNormalizationSpec) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    squeeze = False
    if array.ndim == 1:
        array = array[None, :]
        squeeze = True
    if array.shape[-1] != len(norm_spec.entries):
        raise ValueError(
            f"Expected joint dimension {len(norm_spec.entries)}, got {array.shape[-1]}"
        )

    lower = np.array([entry.lower for entry in norm_spec.entries], dtype=np.float32)
    upper = np.array([entry.upper for entry in norm_spec.entries], dtype=np.float32)
    center = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0
    normalized = ((array - center) / half_range) * 100.0
    normalized = np.clip(normalized, -100.0, 100.0).astype(np.float32)
    return normalized[0] if squeeze else normalized


def normalize_gripper_array(values: np.ndarray, norm_spec: JointNormalizationSpec) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    squeeze = False
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[-1] != 1:
        raise ValueError(f"Expected gripper array shape (*, 1), got {array.shape}")

    lower = norm_spec.gripper_close
    upper = norm_spec.gripper_open
    center = (lower + upper) / 2.0
    half_range = (upper - lower) / 2.0
    normalized = ((array - center) / half_range) * 100.0
    normalized = np.clip(normalized, -100.0, 100.0).astype(np.float32)
    return normalized[0] if squeeze else normalized


def compose_adc_legacy_world_xyzrpy(
    goal_world_xyzrpy: np.ndarray,
    goal_robot_xyzrpy: np.ndarray,
) -> np.ndarray:
    world_input = np.asarray(goal_world_xyzrpy, dtype=np.float32)
    world = world_input
    robot = np.asarray(goal_robot_xyzrpy, dtype=np.float32)
    if world.ndim == 1:
        world = world[None, :]
    if robot.ndim == 1:
        robot = robot[None, :]
    if world.shape != robot.shape or world.shape[-1] != 6:
        raise ValueError(
            f"Expected matching pose arrays of shape (T, 6), got {world.shape} and {robot.shape}"
        )
    legacy = np.concatenate([world[:, :3], robot[:, 3:6]], axis=1).astype(np.float32)
    return legacy[0] if world_input.ndim == 1 else legacy


def build_export_manifest(
    *,
    schema: str,
    source_type: str,
    source_path: Path,
    robot_cfg: RobotSimConfig,
    norm_spec: JointNormalizationSpec,
    episodes: list[dict[str, Any]],
    image_mode: str,
    camera_names: list[str],
) -> dict[str, Any]:
    if schema == ADC_COMPATIBLE_SCHEMA:
        field_semantics = {
            "observation.state": "normalized full controllable joint state in ADC-compatible range [-100, 100]",
            "action": "normalized full controllable joint target in ADC-compatible range [-100, 100]",
            "skill.goal_position.joint": "normalized full controllable joint goal in ADC-compatible range [-100, 100]",
            "skill.goal_position.gripper": "normalized scalar gripper goal in ADC-compatible range [-100, 100]",
            "skill.goal_position.world_xyzrpy": "ADC legacy world target semantics (world xyz + robot-frame rpy)",
            "skill.goal_position.robot_xyzrpy": "robot-base TCP target pose [x,y,z,roll,pitch,yaw]",
        }
    else:
        field_semantics = {
            "observation.state": "normalized full controllable joint state in range [-100, 100]",
            "action": "normalized full controllable joint target in range [-100, 100]",
            "skill.goal_position.joint": "normalized full controllable joint goal in range [-100, 100]",
            "skill.goal_position.gripper": "normalized scalar gripper goal in range [-100, 100]",
            "observation.tcp.world_xyzrpy": "current TCP pose in world frame [x,y,z,roll,pitch,yaw]",
            "observation.tcp.robot_xyzrpy": "current TCP pose in robot-base frame [x,y,z,roll,pitch,yaw]",
            "skill.goal_position.tcp.world_xyzrpy": "target TCP pose in world frame [x,y,z,roll,pitch,yaw]",
            "skill.goal_position.tcp.robot_xyzrpy": "target TCP pose in robot-base frame [x,y,z,roll,pitch,yaw]",
        }

    return {
        "schema": schema,
        "schema_version": SCHEMA_VERSION,
        "source_type": source_type,
        "source_path": str(source_path.resolve()),
        "robot_name": robot_cfg.name,
        "robot_full_name": robot_cfg.full_name,
        "arm_dofs": robot_cfg.arm_dofs,
        "total_dofs": robot_cfg.total_dofs,
        "joint_names": list(norm_spec.joint_names),
        "arm_joint_names": list(robot_cfg.arm_joint_names),
        "finger_joint_names": list(robot_cfg.finger_joint_names),
        "gripper_type": robot_cfg.gripper_type,
        "joint_shape_policy": "full_controllable_dofs",
        "normalization_spec": {
            "range": [-100.0, 100.0],
            "joints": [
                {
                    "name": entry.name,
                    "minus_100": entry.lower,
                    "plus_100": entry.upper,
                    "source": entry.source,
                }
                for entry in norm_spec.entries
            ],
            "gripper": {
                "minus_100": norm_spec.gripper_close,
                "plus_100": norm_spec.gripper_open,
                "source": norm_spec.gripper_source,
            },
        },
        "world_frame_definition": "sim:/World for sim_raw, ADC calibrated world for adc_raw",
        "robot_base_definition": "articulation root for sim_raw, calibrated base_link for adc_raw",
        "tcp_definition": "ee_frame_tcp if present, else ee_frame_body + offset_position",
        "gripper_scalar_definition": (
            "Normalized scalar open/close target exported separately from the full-DOF joint vector"
        ),
        "camera_names": camera_names,
        "image_mode": image_mode,
        "field_semantics": field_semantics,
        "episodes": episodes,
    }


def load_adc_dataset_rows(adc_dataset_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(adc_dataset_dir)
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"ADC dataset data/ directory not found: {data_dir}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required to read ADC parquet datasets. "
            "Install it in the host environment before exporting adc_raw datasets."
        ) from exc

    rows: list[dict[str, Any]] = []
    for parquet_path in sorted(data_dir.glob("chunk-*/file-*.parquet")):
        table = pq.read_table(parquet_path)
        columns = table.to_pydict()
        num_rows = table.num_rows
        for idx in range(num_rows):
            row = {name: columns[name][idx] for name in columns}
            rows.append(row)
    return rows


def load_adc_export_context(
    *,
    robot_name: str | None = None,
    adc_robot_config: str | Path | None = None,
) -> _ADCExportContext:
    from src.data_collection.adc_imports import (
        get_adc_root,
        get_calibration_limits_loader,
        get_frame_transformer,
        get_kinematics_engine,
    )

    adc_root = get_adc_root()
    if adc_robot_config is None:
        canonical_robot = normalize_robot_name(robot_name, default="so101")
        if canonical_robot != "so101":
            raise ValueError(
                "adc_raw export currently requires an explicit --adc-robot-config for non-SO101 robots"
            )
        adc_robot_config = adc_root / "robot_configs" / "robot" / "so101_robot3.yaml"

    config_path = Path(adc_robot_config)
    with open(config_path) as f:
        adc_cfg = yaml.safe_load(f) or {}

    canonical_robot = normalize_robot_name(
        robot_name or adc_cfg.get("robot_type"),
        default=adc_cfg.get("robot_type"),
    )
    if not canonical_robot:
        raise ValueError(f"Could not resolve robot name from ADC robot config: {config_path}")

    joint_names = tuple(adc_cfg.get("joint_names") or [])
    if not joint_names:
        raise ValueError(f"ADC robot config has no joint_names: {config_path}")
    arm_joint_names = joint_names[:-1]

    kin_cls = get_kinematics_engine()
    load_calibration_limits = get_calibration_limits_loader()
    FrameTransformer = get_frame_transformer()

    kin_cfg = adc_cfg.get("kinematics", {})
    end_effector_frame = kin_cfg.get("end_effector_frame", "gripper_frame_link")
    urdf_path = _resolve_adc_local_path(kin_cfg.get("urdf_path", ""), adc_root)
    frames_file = _resolve_adc_local_path(adc_cfg.get("frames_file", ""), adc_root)
    calibration_file = _resolve_adc_local_path(adc_cfg.get("calibration_file", ""), adc_root)

    sim_robot_cfg = load_robot_config(canonical_robot)
    tcp_offset = list(sim_robot_cfg.ee_frame_offset_position or [])
    if not tcp_offset:
        tcp_offset = None

    kinematics = kin_cls(
        urdf_path=str(urdf_path),
        end_effector_frame=end_effector_frame,
        joint_names=list(arm_joint_names),
        tcp_offset=tcp_offset,
    )
    calibration_limits = load_calibration_limits(
        str(calibration_file),
        list(arm_joint_names),
    )
    frame_transformer = FrameTransformer(str(frames_file)) if frames_file else None

    return _ADCExportContext(
        robot_name=canonical_robot,
        joint_names=joint_names,
        arm_dofs=len(arm_joint_names),
        frame_name="world",
        calibration_limits=calibration_limits,
        kinematics=kinematics,
        frame_transformer=frame_transformer,
    )


def compute_adc_tcp_robot_xyzrpy(
    ctx: _ADCExportContext,
    normalized_goal_joint: np.ndarray | list[float],
) -> np.ndarray:
    arm_norm = np.asarray(normalized_goal_joint, dtype=np.float32).ravel()[: ctx.arm_dofs]
    arm_rad = ctx.calibration_limits.normalized_to_radians(arm_norm)
    pos_robot, rot_robot = ctx.kinematics.forward_kinematics(arm_rad)
    roll, pitch, yaw = rotation_matrix_to_rpy(rot_robot)
    return np.array([pos_robot[0], pos_robot[1], pos_robot[2], roll, pitch, yaw], dtype=np.float32)


def compute_adc_tcp_world_xyzrpy(
    ctx: _ADCExportContext,
    normalized_goal_joint: np.ndarray | list[float],
) -> np.ndarray:
    arm_norm = np.asarray(normalized_goal_joint, dtype=np.float32).ravel()[: ctx.arm_dofs]
    arm_rad = ctx.calibration_limits.normalized_to_radians(arm_norm)
    pos_robot, rot_robot = ctx.kinematics.forward_kinematics(arm_rad)

    if (
        ctx.frame_transformer is not None
        and ctx.frame_name != "base_link"
        and ctx.frame_transformer.has_frame(ctx.frame_name)
    ):
        T_frame_from_base = ctx.frame_transformer.frames[ctx.frame_name]["T_frame_from_base"]
        pos_world = (T_frame_from_base @ np.array([*pos_robot, 1.0], dtype=np.float64))[:3]
        rot_world = T_frame_from_base[:3, :3] @ rot_robot
    else:
        pos_world = pos_robot
        rot_world = rot_robot
    roll, pitch, yaw = rotation_matrix_to_rpy(rot_world)
    return np.array([pos_world[0], pos_world[1], pos_world[2], roll, pitch, yaw], dtype=np.float32)


def rotation_matrix_to_rpy(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _resolve_adc_local_path(raw_path: str, adc_root: Path) -> Path:
    if not raw_path:
        return Path()
    p = Path(raw_path)
    if p.exists():
        return p
    if p.is_absolute():
        parts = p.parts
        for repo_marker in ("AutoDataCollector", "lerobot_CaP_distillation"):
            if repo_marker in parts:
                marker_idx = parts.index(repo_marker)
                return adc_root / Path(*parts[marker_idx + 1 :])
        for anchor in ("assets", "robot_configs", "judge", "src", "videos"):
            if anchor in parts:
                anchor_idx = parts.index(anchor)
                return adc_root / Path(*parts[anchor_idx:])
    if raw_path.startswith("/"):
        return Path(raw_path)
    return adc_root / raw_path


def _expand_positions(
    value: float | list[float] | tuple[float, ...],
    target_len: int,
) -> list[float]:
    if target_len <= 0:
        return [float(value[0] if isinstance(value, (list, tuple)) else value)]
    if isinstance(value, (list, tuple)):
        if not value:
            return [0.0] * target_len
        if len(value) >= target_len:
            return [float(v) for v in value[:target_len]]
        return [float(value[0])] * target_len
    return [float(value)] * target_len


def _load_pose_array(path: Path, *, length: int) -> np.ndarray:
    if path.exists():
        return np.load(path).astype(np.float32)
    return np.zeros((length, 6), dtype=np.float32)


def _as_pose_array(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(6, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.shape != (6,):
        return np.zeros(6, dtype=np.float32)
    return arr


def _coerce_vector(value: Any, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] > length:
        return arr[:length]
    padded = np.zeros(length, dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def _coerce_pose_array_from_row(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(6, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.shape == (6,):
        return arr
    padded = np.zeros(6, dtype=np.float32)
    padded[: min(6, arr.shape[0])] = arr[:6]
    return padded


def _coerce_scalar(value: Any) -> float:
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def _write_json(path: Path, value: Any) -> None:
    with open(path, "w") as f:
        json.dump(value, f, indent=2)


def _link_sim_images(source_images_dir: Path, target_images_dir: Path) -> None:
    if not source_images_dir.exists():
        return
    target_images_dir.mkdir(parents=True, exist_ok=True)
    for cam_dir in source_images_dir.iterdir():
        if not cam_dir.is_dir():
            continue
        link = target_images_dir / cam_dir.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(cam_dir.resolve(), target_is_directory=True)
