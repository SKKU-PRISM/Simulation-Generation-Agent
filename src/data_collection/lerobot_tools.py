"""Local LeRobot conversion, validation, and publication helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .sim_recorder import convert_to_lerobot

REQUIRED_LEROBOT_FEATURES = (
    "observation.state",
    "observation.gripper_state",
    "observation.tcp.robot_xyzrpy",
    "action",
    "skill.goal_position.robot_xyzrpy",
)


def ensure_lerobot_available() -> None:
    """Raise ImportError if the ``lerobot`` package is not importable."""

    _load_lerobot_dataset_class()


def ensure_huggingface_hub_available() -> None:
    """Raise ImportError if ``huggingface_hub`` is not importable."""

    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hugging Face uploads. "
            "Install it with: pip install huggingface_hub"
        ) from exc


def convert_raw_dataset_to_lerobot(
    raw_dataset_dir: str | os.PathLike[str],
    *,
    repo_id: str = "local/sim_dataset",
    output_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Convert a raw simulator dataset into a local LeRobot dataset."""

    dataset_path = convert_to_lerobot(
        raw_dataset_dir=str(raw_dataset_dir),
        repo_id=repo_id,
        output_root=str(output_root) if output_root is not None else None,
    )
    return Path(dataset_path).expanduser().resolve()


def check_lerobot_dataset(
    dataset_root: str | os.PathLike[str],
    *,
    repo_id: str | None = None,
) -> dict[str, Any]:
    """Validate a local LeRobot dataset by structure and by loading one sample."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LeRobot dataset path not found: {root}")

    report: dict[str, Any] = {
        "dataset_root": str(root),
        "repo_id": None,
        "root_dir": None,
        "required_features": list(REQUIRED_LEROBOT_FEATURES),
        "feature_keys": [],
        "num_frames": 0,
        "sample_keys": [],
        "missing_features": [],
        "structure": {
            "meta_exists": (root / "meta").exists(),
            "data_exists": (root / "data").exists(),
            "videos_exists": (root / "videos").exists(),
        },
        "pass": False,
    }

    LeRobotDataset = _load_lerobot_dataset_class()
    dataset, resolved_repo_id, resolved_root = _load_local_lerobot_dataset(
        LeRobotDataset,
        root,
        repo_id=repo_id,
    )

    report["repo_id"] = resolved_repo_id
    report["root_dir"] = str(resolved_root)

    feature_map = getattr(dataset, "features", {}) or {}
    if isinstance(feature_map, dict):
        report["feature_keys"] = sorted(feature_map.keys())

    report["missing_features"] = [
        key for key in REQUIRED_LEROBOT_FEATURES if key not in report["feature_keys"]
    ]

    report["num_frames"] = int(len(dataset))
    if report["num_frames"] <= 0:
        raise ValueError(f"LeRobot dataset contains no frames: {root}")

    sample = dataset[0]
    if not isinstance(sample, dict):
        raise TypeError("Expected LeRobotDataset sample to be a dict-like frame")
    report["sample_keys"] = sorted(sample.keys())

    missing_sample_keys = [
        key for key in REQUIRED_LEROBOT_FEATURES if key not in report["sample_keys"]
    ]
    if missing_sample_keys:
        report["missing_features"] = sorted(
            set(report["missing_features"]) | set(missing_sample_keys)
        )

    report["pass"] = (
        not report["missing_features"]
        and report["num_frames"] > 0
        and report["structure"]["meta_exists"]
    )
    return report


def publish_lerobot_dataset(
    dataset_root: str | os.PathLike[str],
    *,
    repo_id: str,
    private: bool = False,
    token_env: str = "HF_TOKEN",
    token: str | None = None,
    local_repo_id: str | None = None,
) -> dict[str, Any]:
    """Upload a local LeRobot dataset folder to a Hugging Face dataset repo."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LeRobot dataset path not found: {root}")

    validation_report = check_lerobot_dataset(root, repo_id=local_repo_id)
    if not validation_report.get("pass"):
        raise ValueError(f"Local LeRobot dataset validation failed: {validation_report}")

    from huggingface_hub import HfApi

    resolved_token = token or os.environ.get(token_env)
    api = HfApi(token=resolved_token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=root,
        private=private,
        print_report=False,
    )
    return {
        "dataset_root": str(root),
        "repo_id": repo_id,
        "repo_url": str(repo_url),
        "private": bool(private),
        "auth_source": token_env if resolved_token else "cached_or_default",
        "validation_report": validation_report,
        "pass": True,
    }


def _load_lerobot_dataset_class():
    import_errors: list[Exception] = []
    for import_path in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            module = __import__(import_path, fromlist=["LeRobotDataset"])
            return module.LeRobotDataset
        except ImportError as exc:
            import_errors.append(exc)

    raise ImportError(
        "lerobot package is required for local LeRobot validation. "
        "Install it with: pip install lerobot"
    ) from import_errors[-1]


def _load_local_lerobot_dataset(
    dataset_cls,
    dataset_root: Path,
    *,
    repo_id: str | None,
):
    errors: list[str] = []
    for candidate_repo_id, candidate_root in _iter_repo_candidates(dataset_root, repo_id):
        try:
            dataset = dataset_cls(repo_id=candidate_repo_id, root=candidate_root)
            return dataset, candidate_repo_id, Path(candidate_root).resolve()
        except Exception as exc:  # pragma: no cover - aggregated error path
            errors.append(
                f"repo_id={candidate_repo_id!r}, root={str(candidate_root)!r}: {exc}"
            )
    joined = "; ".join(errors) if errors else "no candidates tried"
    raise RuntimeError(f"Unable to load local LeRobot dataset {dataset_root}: {joined}")


def _iter_repo_candidates(dataset_root: Path, repo_id: str | None):
    dataset_root = dataset_root.resolve()
    seen: set[tuple[str, str]] = set()

    def add(candidate_repo_id: str, candidate_root: Path):
        key = (candidate_repo_id, str(candidate_root.resolve()))
        if key in seen:
            return
        seen.add(key)
        yield candidate_repo_id, candidate_root

    if repo_id:
        # LeRobot >=0.5 expects ``root`` to point at the dataset root itself.
        yield from add(repo_id, dataset_root)

        # Backward-compat fallback for older path layouts where ``root`` pointed
        # at the parent directory and the repo_id path was appended internally.
        segments = [part for part in repo_id.split("/") if part]
        if segments and len(dataset_root.parents) >= len(segments):
            candidate_root = dataset_root.parents[len(segments) - 1]
            yield from add(repo_id, candidate_root)
        return

    # Best-effort inference for the local path shapes produced by convert_to_lerobot.
    yield from add(dataset_root.name, dataset_root.parent)
    if dataset_root.parent != dataset_root and dataset_root.parent.name:
        yield from add(
            f"{dataset_root.parent.name}/{dataset_root.name}",
            dataset_root.parent.parent,
        )


def write_json_report(report: dict[str, Any], destination: str | os.PathLike[str]) -> Path:
    """Write a JSON report and return the resolved path."""

    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path
