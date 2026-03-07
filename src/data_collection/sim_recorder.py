"""
Two-stage simulation data recorder for LeRobot v3.0 format datasets.

Stage 1 (SimRecorder): Records raw numpy arrays + PNG images during IsaacLab
    simulation. No lerobot dependency — runs inside the IsaacLab conda env.

Stage 2 (convert_to_lerobot): Reads raw dataset directory and converts to
    LeRobot v3.0 format. Runs on host Python with the lerobot package.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.data_collection.config import RobotSimConfig

logger = logging.getLogger(__name__)


class RecorderStorageError(RuntimeError):
    """Raised when raw dataset files cannot be written to disk."""


class SimRecorder:
    """Stage 1: Records raw simulation data to a structured directory.

    Output structure::

        {output_dir}/{dataset_name}/
        ├── episodes/
        │   ├── episode_000000/
        │   │   ├── states.npy      # (T, N_dof) float32
        │   │   ├── actions.npy     # (T, N_dof) float32
        │   │   ├── images/         # multi-camera:
        │   │   │   ├── front/      #   000000.png, 000001.png, ...
        │   │   │   ├── wrist/
        │   │   │   └── top/
        │   │   └── skills.json     # per-frame skill metadata
        │   └── ...
        └── metadata.json           # robot info, dataset stats
    """

    def __init__(
        self,
        robot_cfg: RobotSimConfig,
        output_dir: str,
        dataset_name: str = "sim_dataset",
        fps: int = 20,
    ) -> None:
        self._robot_cfg = robot_cfg
        self._fps = fps
        self._dataset_dir = Path(output_dir) / dataset_name
        self._episodes_dir = self._dataset_dir / "episodes"
        self._episodes_dir.mkdir(parents=True, exist_ok=True)

        # Episode tracking
        self._episode_idx: int = 0
        self._recording: bool = False
        self._current_ep_dir: Optional[Path] = None

        # Per-episode buffers
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._skills: list[dict] = []
        self._image_count: int = 0
        self._has_images: bool = False
        self._camera_names: list[str] = []  # populated on first multi-camera record

        # Dataset-level stats
        self._completed_episodes: list[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def current_episode_steps(self) -> int:
        return len(self._states)

    @property
    def dataset_dir(self) -> Path:
        return self._dataset_dir

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(
        self, task_description: str, episode_idx: int | None = None
    ) -> None:
        """Begin recording a new episode.

        Args:
            task_description: Natural-language description of the task/episode.
            episode_idx: Optional explicit index. Auto-increments if None.
        """
        if self._recording:
            raise RuntimeError(
                "Cannot start a new episode while already recording. "
                "Call end_episode() first."
            )

        if episode_idx is not None:
            self._episode_idx = episode_idx

        ep_name = f"episode_{self._episode_idx:06d}"
        self._current_ep_dir = self._episodes_dir / ep_name
        self._current_ep_dir.mkdir(parents=True, exist_ok=True)
        (self._current_ep_dir / "images").mkdir(exist_ok=True)

        # Reset buffers
        self._states.clear()
        self._actions.clear()
        self._skills.clear()
        self._image_count = 0
        self._has_images = False
        self._recording = True
        self._current_task_description = task_description

        logger.info("Started episode %d: %s", self._episode_idx, task_description)

    def record_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        image: np.ndarray | None = None,
        images: dict[str, np.ndarray | None] | None = None,
        skill_label: str = "",
        skill_type: str = "",
        skill_progress: float = 0.0,
        goal_joint: np.ndarray | None = None,
        goal_world_xyzrpy: np.ndarray | None = None,
        goal_robot_xyzrpy: np.ndarray | None = None,
        goal_gripper: float = 0.0,
    ) -> None:
        """Record a single simulation step.

        Args:
            state: Current joint positions, shape (N_dof,) float32.
            action: Target joint positions, shape (N_dof,) float32.
            image: Single RGB camera image (legacy, treated as "front").
            images: Multi-camera dict ``{cam_name: rgb_array}``. Takes
                precedence over ``image`` if both are provided.
            skill_label: Natural-language skill description.
            skill_type: Skill category (move, pick, place, etc.).
            skill_progress: Skill completion progress 0.0~1.0.
            goal_joint: Target joint positions for the skill, (N_dof,) float32.
            goal_world_xyzrpy: Target EE world coordinates, (6,) float32.
            goal_robot_xyzrpy: Target EE robot base frame, (6,) float32.
            goal_gripper: Gripper target value.
        """
        if not self._recording:
            raise RuntimeError("Not recording. Call start_episode() first.")

        n_dof = self._robot_cfg.total_dofs
        state = np.asarray(state, dtype=np.float32).ravel()
        action = np.asarray(action, dtype=np.float32).ravel()
        if state.shape[0] != n_dof:
            raise ValueError(
                f"state has {state.shape[0]} dims, expected {n_dof}"
            )
        if action.shape[0] != n_dof:
            raise ValueError(
                f"action has {action.shape[0]} dims, expected {n_dof}"
            )

        self._states.append(state)
        self._actions.append(action)

        # Judge-only cameras: excluded from dataset recording (VLM judge only)
        _JUDGE_ONLY_CAMERAS = {"front_cam", "front"}

        # Save images (multi-camera or legacy single-camera)
        if images is not None:
            # Multi-camera path
            for cam_name, cam_img in images.items():
                if cam_name in _JUDGE_ONLY_CAMERAS:
                    continue  # VLM judge only, not in dataset
                if cam_img is not None:
                    self._save_image(cam_img, self._image_count, cam_name=cam_name)
                    self._has_images = True
                    if cam_name not in self._camera_names:
                        self._camera_names.append(cam_name)
        elif image is not None:
            # Legacy single-camera (save as "front")
            self._save_image(image, self._image_count, cam_name="front")
            self._has_images = True
            if "front" not in self._camera_names:
                self._camera_names.append("front")
        self._image_count += 1

        # Build skill metadata entry
        skill_entry: dict = {
            "label": skill_label,
            "type": skill_type,
            "progress": float(skill_progress),
            "goal_joint": goal_joint.tolist() if goal_joint is not None else None,
            "goal_world_xyzrpy": (
                goal_world_xyzrpy.tolist()
                if goal_world_xyzrpy is not None
                else None
            ),
            "goal_robot_xyzrpy": (
                goal_robot_xyzrpy.tolist()
                if goal_robot_xyzrpy is not None
                else None
            ),
            "goal_gripper": float(goal_gripper),
        }
        self._skills.append(skill_entry)

    def end_episode(self, success: bool = False, discard: bool = False) -> None:
        """Finish the current episode and flush data to disk.

        Args:
            success: Whether the episode completed the task successfully.
            discard: If True, delete the episode directory instead of saving.
        """
        if not self._recording:
            raise RuntimeError("Not recording. Call start_episode() first.")

        self._recording = False
        ep_dir = self._current_ep_dir
        steps = len(self._states)

        if discard or steps == 0:
            if ep_dir is not None and ep_dir.exists():
                shutil.rmtree(ep_dir)
            logger.info(
                "Discarded episode %d (%d steps)", self._episode_idx, steps
            )
            return

        # Flush numpy arrays
        np.save(ep_dir / "states.npy", np.stack(self._states))
        np.save(ep_dir / "actions.npy", np.stack(self._actions))

        # Flush skills metadata
        with open(ep_dir / "skills.json", "w") as f:
            json.dump(self._skills, f)

        # Track completed episode
        self._completed_episodes.append(
            {
                "index": self._episode_idx,
                "steps": steps,
                "success": success,
                "has_images": self._has_images,
                "task_description": self._current_task_description,
            }
        )
        logger.info(
            "Saved episode %d: %d steps, success=%s",
            self._episode_idx,
            steps,
            success,
        )
        self._episode_idx += 1

    # ------------------------------------------------------------------
    # Finalize
    # ------------------------------------------------------------------

    def finalize(self) -> str:
        """Write dataset metadata and return the dataset directory path.

        Must be called after all episodes are recorded.

        Returns:
            Absolute path to the raw dataset directory.
        """
        if self._recording:
            raise RuntimeError(
                "Cannot finalize while recording. Call end_episode() first."
            )

        metadata = {
            "robot_name": self._robot_cfg.name,
            "robot_full_name": self._robot_cfg.full_name,
            "total_dofs": self._robot_cfg.total_dofs,
            "arm_dofs": self._robot_cfg.arm_dofs,
            "joint_names": self._robot_cfg.all_joint_names,
            "fps": self._fps,
            "camera_names": self._camera_names,
            "total_episodes": len(self._completed_episodes),
            "successful_episodes": sum(
                1 for e in self._completed_episodes if e["success"]
            ),
            "episodes": self._completed_episodes,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        with open(self._dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        path = str(self._dataset_dir.resolve())
        logger.info(
            "Finalized dataset: %s (%d episodes)", path, len(self._completed_episodes)
        )
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_image(
        self, image: np.ndarray, frame_idx: int, cam_name: str = "front"
    ) -> None:
        """Save an RGB image as PNG to the current episode's images/{cam_name}/ dir."""
        img = np.asarray(image, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) uint8 image, got shape {img.shape}"
            )
        cam_dir = self._current_ep_dir / "images" / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        out_path = cam_dir / f"{frame_idx:06d}.png"
        try:
            from PIL import Image

            Image.fromarray(img).save(out_path)
        except OSError as exc:
            raise RecorderStorageError(
                f"Failed to save image to {out_path}: {exc}"
            ) from exc
        except ImportError:
            import cv2

            # cv2 expects BGR
            ok = cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RecorderStorageError(
                    f"Failed to save image to {out_path}: cv2.imwrite returned False"
                )


# ======================================================================
# Stage 2: LeRobot conversion
# ======================================================================


def convert_to_lerobot(
    raw_dataset_dir: str,
    repo_id: str = "local/sim_dataset",
    output_root: str | None = None,
) -> str:
    """Convert a raw dataset directory to LeRobot v3.0 format.

    Reads the structured raw directory produced by :class:`SimRecorder` and
    creates a LeRobot-compatible dataset with proper features spec.

    Args:
        raw_dataset_dir: Path to the raw dataset (contains metadata.json).
        repo_id: LeRobot dataset repository ID (e.g. "local/sim_dataset").
        output_root: Root directory for the LeRobot dataset. If None, places
            the dataset next to the raw directory.

    Returns:
        Absolute path to the created LeRobot dataset.
    """
    raw_dir = Path(raw_dataset_dir)
    meta_path = raw_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {raw_dir}")

    with open(meta_path) as f:
        metadata = json.load(f)

    total_dofs = metadata["total_dofs"]
    fps = metadata["fps"]
    episodes_info = metadata["episodes"]
    has_images = any(e.get("has_images", False) for e in episodes_info)
    camera_names = metadata.get("camera_names", ["front"] if has_images else [])

    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        raise ImportError(
            "lerobot package is required for conversion. "
            "Install it with: pip install lerobot"
        )

    # Define LeRobot v3.0 features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (total_dofs,),
            "names": [metadata["joint_names"]],
        },
        "action": {
            "dtype": "float32",
            "shape": (total_dofs,),
            "names": [metadata["joint_names"]],
        },
        "skill.natural_language": {"dtype": "string", "shape": (1,)},
        "skill.type": {"dtype": "string", "shape": (1,)},
        "skill.progress": {"dtype": "float32", "shape": (1,)},
        "skill.goal_position.joint": {
            "dtype": "float32",
            "shape": (total_dofs,),
        },
        "skill.goal_position.world_xyzrpy": {
            "dtype": "float32",
            "shape": (6,),
        },
        "skill.goal_position.robot_xyzrpy": {
            "dtype": "float32",
            "shape": (6,),
        },
        "skill.goal_position.gripper": {"dtype": "float32", "shape": (1,)},
    }
    for cam_name in camera_names:
        features[f"observation.images.{cam_name}"] = {
            "dtype": "image",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }

    if output_root is None:
        output_root = str(raw_dir.parent)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_root,
    )

    for ep_info in episodes_info:
        ep_idx = ep_info["index"]
        ep_dir = raw_dir / "episodes" / f"episode_{ep_idx:06d}"
        if not ep_dir.exists():
            logger.warning("Episode dir missing, skipping: %s", ep_dir)
            continue

        states = np.load(ep_dir / "states.npy")  # (T, N_dof)
        actions = np.load(ep_dir / "actions.npy")  # (T, N_dof)
        T = states.shape[0]

        with open(ep_dir / "skills.json") as f:
            skills = json.load(f)

        # Load images per camera
        images_dir = ep_dir / "images"
        cam_image_paths: dict[str, list[Path]] = {}
        for cam_name in camera_names:
            cam_dir = images_dir / cam_name
            if cam_dir.exists():
                cam_image_paths[cam_name] = sorted(cam_dir.glob("*.png"))
            else:
                cam_image_paths[cam_name] = []

        for t in range(T):
            skill = skills[t] if t < len(skills) else {}
            goal_joint = skill.get("goal_joint")
            goal_xyzrpy = skill.get("goal_world_xyzrpy")
            goal_robot_xyzrpy = skill.get("goal_robot_xyzrpy")

            frame: dict = {
                "observation.state": states[t],
                "action": actions[t],
                "skill.natural_language": skill.get("label", ""),
                "skill.type": skill.get("type", ""),
                "skill.progress": np.float32(skill.get("progress", 0.0)),
                "skill.goal_position.joint": (
                    np.array(goal_joint, dtype=np.float32)
                    if goal_joint is not None
                    else np.zeros(total_dofs, dtype=np.float32)
                ),
                "skill.goal_position.world_xyzrpy": (
                    np.array(goal_xyzrpy, dtype=np.float32)
                    if goal_xyzrpy is not None
                    else np.zeros(6, dtype=np.float32)
                ),
                "skill.goal_position.robot_xyzrpy": (
                    np.array(goal_robot_xyzrpy, dtype=np.float32)
                    if goal_robot_xyzrpy is not None
                    else np.zeros(6, dtype=np.float32)
                ),
                "skill.goal_position.gripper": np.float32(
                    skill.get("goal_gripper", 0.0)
                ),
            }

            # Add images from each camera
            for cam_name in camera_names:
                paths = cam_image_paths.get(cam_name, [])
                if t < len(paths):
                    from PIL import Image

                    frame[f"observation.images.{cam_name}"] = np.array(
                        Image.open(paths[t])
                    )

            dataset.add_frame(frame)

        dataset.save_episode(task=ep_info.get("task_description", ""))

    dataset.consolidate()
    dataset_path = str(Path(output_root).resolve() / repo_id)
    logger.info("LeRobot dataset created at: %s", dataset_path)
    return dataset_path
