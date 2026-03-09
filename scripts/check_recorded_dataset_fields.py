#!/usr/bin/env python3
"""Validate required dataset fields across raw/export/preprocess artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _first_episode_dir(root: Path) -> Path:
    episodes_dir = root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Episodes directory not found: {episodes_dir}")
    candidates = sorted(p for p in episodes_dir.iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No episode directories found under: {episodes_dir}")
    return candidates[0]


def _shape(path: Path) -> list[int]:
    return list(np.load(path, mmap_mode="r").shape)


def _check_exists(path: Path) -> dict[str, Any]:
    return {"path": str(path), "exists": path.exists()}


def _validate_raw(raw_dir: Path) -> dict[str, Any]:
    ep_dir = _first_episode_dir(raw_dir)
    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Raw metadata not found: {metadata_path}")

    metadata = json.loads(metadata_path.read_text())
    required_files = {
        "states": ep_dir / "states.npy",
        "actions": ep_dir / "actions.npy",
        "tcp_robot_xyzrpy": ep_dir / "tcp_robot_xyzrpy.npy",
        "gripper_state": ep_dir / "gripper_state.npy",
        "goal_robot_xyzrpy": ep_dir / "goal_robot_xyzrpy.npy",
        "skills": ep_dir / "skills.json",
    }
    for path in required_files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing raw dataset file: {path}")

    tcp_robot = np.load(required_files["tcp_robot_xyzrpy"])
    gripper_state = np.load(required_files["gripper_state"])
    goal_robot = np.load(required_files["goal_robot_xyzrpy"])
    skills = json.loads(required_files["skills"].read_text())

    checks = {
        "episode_dir": str(ep_dir),
        "required_files": {name: _check_exists(path) for name, path in required_files.items()},
        "metadata": {
            "robot_name": metadata.get("robot_name"),
            "gripper_state_dim": metadata.get("gripper_state_dim"),
            "required_step_fields": metadata.get("required_step_fields"),
            "joint_shape_policy": metadata.get("joint_shape_policy"),
        },
        "shapes": {
            "tcp_robot_xyzrpy": list(tcp_robot.shape),
            "gripper_state": list(gripper_state.shape),
            "goal_robot_xyzrpy": list(goal_robot.shape),
        },
        "value_checks": {
            "tcp_robot_nonzero": bool(np.any(np.abs(tcp_robot) > 1e-6)),
            "goal_robot_nonzero": bool(np.any(np.abs(goal_robot) > 1e-6)),
            "gripper_state_range": float(np.ptp(gripper_state)),
            "skills_goal_robot_all_present": bool(
                all(entry.get("goal_robot_xyzrpy") is not None for entry in skills)
            ),
        },
    }
    checks["pass"] = bool(
        checks["metadata"]["gripper_state_dim"] == 1
        and checks["metadata"]["joint_shape_policy"] == "full_controllable_dofs"
        and checks["value_checks"]["tcp_robot_nonzero"]
        and checks["value_checks"]["goal_robot_nonzero"]
        and checks["value_checks"]["gripper_state_range"] > 0.01
        and checks["value_checks"]["skills_goal_robot_all_present"]
        and checks["shapes"]["tcp_robot_xyzrpy"][1:] == [6]
        and checks["shapes"]["gripper_state"][1:] == [1]
        and checks["shapes"]["goal_robot_xyzrpy"][1:] == [6]
    )
    return checks


def _validate_export(export_dir: Path) -> dict[str, Any]:
    manifest_path = export_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Export manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    ep_dir = _first_episode_dir(export_dir)

    required_files = {
        "state": ep_dir / "observation.state.npy",
        "action": ep_dir / "action.npy",
        "gripper_state": ep_dir / "observation.gripper_state.npy",
        "tcp_robot_xyzrpy": ep_dir / "observation.tcp.robot_xyzrpy.npy",
        "goal_joint": ep_dir / "skill.goal_position.joint.npy",
        "goal_gripper": ep_dir / "skill.goal_position.gripper.npy",
        "goal_world_xyzrpy": ep_dir / "skill.goal_position.world_xyzrpy.npy",
        "goal_robot_xyzrpy": ep_dir / "skill.goal_position.robot_xyzrpy.npy",
    }
    for path in required_files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing export dataset file: {path}")

    gripper_state = np.load(required_files["gripper_state"])
    tcp_robot = np.load(required_files["tcp_robot_xyzrpy"])
    goal_robot = np.load(required_files["goal_robot_xyzrpy"])

    checks = {
        "episode_dir": str(ep_dir),
        "schema": manifest.get("schema"),
        "gripper_state_dim": manifest.get("gripper_state_dim"),
        "field_semantics_keys": sorted((manifest.get("field_semantics") or {}).keys()),
        "shapes": {
            "gripper_state": list(gripper_state.shape),
            "tcp_robot_xyzrpy": list(tcp_robot.shape),
            "goal_robot_xyzrpy": list(goal_robot.shape),
        },
        "value_checks": {
            "gripper_state_in_range": bool(
                np.nanmin(gripper_state) >= -100.0001 and np.nanmax(gripper_state) <= 100.0001
            ),
            "tcp_robot_nonzero": bool(np.any(np.abs(tcp_robot) > 1e-6)),
            "goal_robot_nonzero": bool(np.any(np.abs(goal_robot) > 1e-6)),
        },
    }
    checks["pass"] = bool(
        checks["schema"] == "adc_compatible"
        and checks["gripper_state_dim"] == 1
        and "observation.gripper_state" in checks["field_semantics_keys"]
        and "observation.tcp.robot_xyzrpy" in checks["field_semantics_keys"]
        and checks["value_checks"]["gripper_state_in_range"]
        and checks["value_checks"]["tcp_robot_nonzero"]
        and checks["value_checks"]["goal_robot_nonzero"]
        and checks["shapes"]["gripper_state"][1:] == [1]
        and checks["shapes"]["tcp_robot_xyzrpy"][1:] == [6]
        and checks["shapes"]["goal_robot_xyzrpy"][1:] == [6]
    )
    return checks


def _validate_preprocess(preprocess_dir: Path, export_dir: Path) -> dict[str, Any]:
    manifest_path = preprocess_dir / "manifest.json"
    samples_path = preprocess_dir / "samples.jsonl"
    train_path = preprocess_dir / "train.jsonl"
    val_path = preprocess_dir / "val.jsonl"
    for path in (manifest_path, samples_path, train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing preprocess artifact: {path}")

    manifest = json.loads(manifest_path.read_text())
    sample_line = samples_path.read_text().splitlines()
    if not sample_line:
        raise ValueError(f"No samples found in {samples_path}")
    sample = json.loads(sample_line[0])
    paths = sample.get("paths", {})
    required_keys = ["gripper_state", "tcp_robot_xyzrpy", "goal_robot_xyzrpy"]

    missing_keys = [key for key in required_keys if key not in paths]
    resolved_paths = {key: (export_dir / paths[key]) for key in required_keys if key in paths}
    existing_paths = {key: path.exists() for key, path in resolved_paths.items()}

    checks = {
        "schema": manifest.get("schema"),
        "gripper_state_dim": manifest.get("gripper_state_dim"),
        "sample_id": sample.get("sample_id"),
        "required_path_keys": required_keys,
        "missing_path_keys": missing_keys,
        "resolved_paths": {key: str(path) for key, path in resolved_paths.items()},
        "existing_paths": existing_paths,
    }
    checks["pass"] = bool(
        checks["schema"] == "adc_compatible"
        and checks["gripper_state_dim"] == 1
        and not missing_keys
        and all(existing_paths.values())
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate required dataset fields across raw/export/preprocess outputs"
    )
    parser.add_argument("--raw-dir", required=True, help="Path to raw_dataset directory")
    parser.add_argument("--export-dir", required=True, help="Path to exported dataset directory")
    parser.add_argument(
        "--preprocess-dir",
        required=True,
        help="Path to preprocessed dataset directory",
    )
    parser.add_argument(
        "--output-report",
        default=None,
        help="Optional path for the JSON validation report (default: <preprocess-dir>/field_check_report.json)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    export_dir = Path(args.export_dir)
    preprocess_dir = Path(args.preprocess_dir)
    report_path = Path(args.output_report) if args.output_report else preprocess_dir / "field_check_report.json"

    report = {
        "raw": _validate_raw(raw_dir),
        "export": _validate_export(export_dir),
        "preprocess": _validate_preprocess(preprocess_dir, export_dir),
    }
    report["pass"] = bool(report["raw"]["pass"] and report["export"]["pass"] and report["preprocess"]["pass"])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(report_path)
    print(json.dumps({"pass": report["pass"]}, indent=2))


if __name__ == "__main__":
    main()
