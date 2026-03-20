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
        │   │   ├── tcp_world_xyzrpy.npy  # (T, 6) float32
        │   │   ├── tcp_robot_xyzrpy.npy  # (T, 6) float32
        │   │   ├── gripper_state.npy     # (T, 1) float32
        │   │   ├── goal_robot_xyzrpy.npy # (T, 6) float32
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
        front_video_fps: int = 10,
        front_video_max_priority: int = 1,
        front_video_camera_name: str = "front",
        dataset_cameras: list[str] | None = None,
    ) -> None:
        self._robot_cfg = robot_cfg
        self._fps = fps
        self._run_output_dir = Path(output_dir)
        self._dataset_dir = Path(output_dir) / dataset_name
        self._episodes_dir = self._dataset_dir / "episodes"
        self._episodes_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir = self._run_output_dir / "videos"
        self._front_video_tmp_root = self._run_output_dir / ".front_video_frames"
        self._front_video_target_fps = max(1, min(int(front_video_fps), int(fps)))
        self._front_video_stride = max(1, round(float(fps) / float(self._front_video_target_fps)))
        self._front_video_actual_fps = max(1, round(float(fps) / float(self._front_video_stride)))
        self._front_video_max_priority = max(0, int(front_video_max_priority))
        self._front_video_camera_name = str(front_video_camera_name or "front").strip() or "front"
        self._front_video_camera_aliases = self._aliases_for_camera_name(self._front_video_camera_name)
        self._dataset_cameras = list(dataset_cameras or ["top", "wrist", "front"])
        self._dataset_camera_aliases = self._build_camera_aliases(self._dataset_cameras)

        # Episode tracking
        self._episode_idx: int = 0
        self._recording: bool = False
        self._current_ep_dir: Optional[Path] = None
        self._current_front_video_frames_dir: Optional[Path] = None
        self._front_video_frame_count: int = 0

        # Per-episode buffers
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._tcp_world_xyzrpy: list[np.ndarray] = []
        self._tcp_robot_xyzrpy: list[np.ndarray] = []
        self._gripper_state: list[np.ndarray] = []
        self._goal_robot_xyzrpy_dense: list[np.ndarray] = []
        self._skills: list[dict] = []
        self._image_count: int = 0
        self._has_images: bool = False
        self._camera_names: list[str] = []  # populated on first multi-camera record

        # Dataset-level stats
        self._completed_episodes: list[dict] = []
        self._front_video_generated: bool = False
        self._front_video_priority: int = 0
        self._front_video_episode: Optional[int] = None
        self._front_video_path: Optional[Path] = None
        self._front_video_success_type: Optional[str] = None
        self._front_video_error: Optional[str] = None
        self._write_metadata()

    @staticmethod
    def _aliases_for_camera_name(name: str) -> set[str]:
        aliases = {str(name)}
        if name.endswith("_cam"):
            aliases.add(name[:-4])
        else:
            aliases.add(f"{name}_cam")
        return aliases

    @classmethod
    def _build_camera_aliases(cls, names: list[str]) -> set[str]:
        aliases: set[str] = set()
        for name in names:
            aliases.update(cls._aliases_for_camera_name(name))
        return aliases

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

    @property
    def front_video_generated(self) -> bool:
        return self._front_video_generated and self._front_video_path is not None

    @property
    def front_video_priority(self) -> int:
        return self._front_video_priority

    @property
    def front_video_path(self) -> Path | None:
        if self._front_video_path is None:
            return None
        return self._front_video_path if self._front_video_path.exists() else None

    @property
    def front_video_episode(self) -> int | None:
        return self._front_video_episode

    @property
    def front_video_success_type(self) -> str | None:
        return self._front_video_success_type

    @property
    def front_video_error(self) -> str | None:
        return self._front_video_error

    @property
    def front_video_camera_name(self) -> str:
        return self._front_video_camera_name

    @property
    def front_video_camera_capture_names(self) -> tuple[str, ...]:
        if self._front_video_camera_name.endswith("_cam"):
            return (self._front_video_camera_name, self._front_video_camera_name[:-4])
        return (f"{self._front_video_camera_name}_cam", self._front_video_camera_name)

    def should_capture_front_video_frame(self, step_idx: int) -> bool:
        """Whether the current step should capture a front video frame."""
        return (
            self._recording
            and self._front_video_priority < self._front_video_max_priority
            and self._current_front_video_frames_dir is not None
            and step_idx % self._front_video_stride == 0
        )

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
        self._tcp_world_xyzrpy.clear()
        self._tcp_robot_xyzrpy.clear()
        self._gripper_state.clear()
        self._goal_robot_xyzrpy_dense.clear()
        self._skills.clear()
        self._image_count = 0
        self._has_images = False
        self._front_video_frame_count = 0
        self._recording = True
        self._current_task_description = task_description
        if self._front_video_priority < self._front_video_max_priority:
            self._current_front_video_frames_dir = (
                self._front_video_tmp_root / f"episode_{self._episode_idx:06d}"
            )
            if self._current_front_video_frames_dir.exists():
                shutil.rmtree(self._current_front_video_frames_dir)
            self._current_front_video_frames_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._current_front_video_frames_dir = None

        logger.info("Started episode %d: %s", self._episode_idx, task_description)

    def record_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        image: np.ndarray | None = None,
        images: dict[str, np.ndarray | None] | None = None,
        front_video_image: np.ndarray | None = None,
        skill_label: str = "",
        skill_type: str = "",
        skill_progress: float = 0.0,
        goal_joint: np.ndarray | None = None,
        goal_world_xyzrpy: np.ndarray | None = None,
        goal_robot_xyzrpy: np.ndarray | None = None,
        goal_gripper: float = 0.0,
        tcp_world_xyzrpy: np.ndarray | None = None,
        tcp_robot_xyzrpy: np.ndarray | None = None,
        gripper_state: np.ndarray | None = None,
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
            tcp_world_xyzrpy: Current TCP pose in simulator world frame, (6,) float32.
            tcp_robot_xyzrpy: Current TCP pose in robot base frame, (6,) float32.
            gripper_state: Current gripper state scalar, (1,) float32.
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
        self._tcp_world_xyzrpy.append(
            self._coerce_pose_vector(tcp_world_xyzrpy, "tcp_world_xyzrpy")
        )
        self._tcp_robot_xyzrpy.append(
            self._coerce_pose_vector(tcp_robot_xyzrpy, "tcp_robot_xyzrpy")
        )
        self._gripper_state.append(
            self._coerce_gripper_state_vector(gripper_state)
        )
        self._goal_robot_xyzrpy_dense.append(
            self._coerce_pose_vector(goal_robot_xyzrpy, "goal_robot_xyzrpy")
        )

        front_frame = front_video_image

        # Save images (multi-camera or legacy single-camera)
        if images is not None:
            # Multi-camera path
            for cam_name, cam_img in images.items():
                if front_frame is None and cam_name in self._front_video_camera_aliases and cam_img is not None:
                    front_frame = cam_img
                if cam_name not in self._dataset_camera_aliases:
                    continue
                if cam_img is not None:
                    self._save_image(cam_img, self._image_count, cam_name=cam_name)
                    self._has_images = True
                    if cam_name not in self._camera_names:
                        self._camera_names.append(cam_name)
        elif image is not None:
            # Legacy single-camera (save as "front")
            if front_frame is None:
                front_frame = image
            if "front" in self._dataset_camera_aliases:
                self._save_image(image, self._image_count, cam_name="front")
                self._has_images = True
                if "front" not in self._camera_names:
                    self._camera_names.append("front")
        self._maybe_save_front_video_frame(front_frame, self._image_count)
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

    def end_episode(
        self,
        success: bool = False,
        discard: bool = False,
        front_video_priority: int | None = None,
        front_video_success_type: str | None = None,
    ) -> None:
        """Finish the current episode and flush data to disk.

        Args:
            success: Whether the episode completed the task successfully.
            discard: If True, delete the episode directory instead of saving.
            front_video_priority: Higher-priority successful episodes replace
                lower-priority representative videos.
            front_video_success_type: Human-readable label for the representative
                video selection (`overall`, `geometry_only`, `vlm_only`, etc.).
        """
        if not self._recording:
            raise RuntimeError("Not recording. Call start_episode() first.")

        self._recording = False
        ep_dir = self._current_ep_dir
        steps = len(self._states)

        if discard or steps == 0:
            self._cleanup_front_video_frames()
            if ep_dir is not None and ep_dir.exists():
                shutil.rmtree(ep_dir)
            self._write_metadata()
            logger.info(
                "Discarded episode %d (%d steps)", self._episode_idx, steps
            )
            return

        # Flush numpy arrays
        np.save(ep_dir / "states.npy", np.stack(self._states))
        np.save(ep_dir / "actions.npy", np.stack(self._actions))
        np.save(ep_dir / "tcp_world_xyzrpy.npy", np.stack(self._tcp_world_xyzrpy))
        np.save(ep_dir / "tcp_robot_xyzrpy.npy", np.stack(self._tcp_robot_xyzrpy))
        np.save(ep_dir / "gripper_state.npy", np.stack(self._gripper_state))
        np.save(ep_dir / "goal_robot_xyzrpy.npy", np.stack(self._goal_robot_xyzrpy_dense))

        # Flush skills metadata
        with open(ep_dir / "skills.json", "w") as f:
            json.dump(self._skills, f)

        if front_video_priority is None:
            front_video_priority = 1 if success else 0
        self._finalize_front_video(
            int(front_video_priority),
            front_video_success_type,
        )

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
        self._write_metadata()
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

    def _build_metadata(self) -> dict:
        """Build dataset metadata from the episodes recorded so far."""
        return {
            "robot_name": self._robot_cfg.name,
            "robot_full_name": self._robot_cfg.full_name,
            "total_dofs": self._robot_cfg.total_dofs,
            "arm_dofs": self._robot_cfg.arm_dofs,
            "joint_names": self._robot_cfg.all_joint_names,
            "arm_joint_names": self._robot_cfg.arm_joint_names,
            "finger_joint_names": self._robot_cfg.finger_joint_names,
            "gripper_type": self._robot_cfg.gripper_type,
            "fps": self._fps,
            "camera_names": self._camera_names,
            "joint_shape_policy": "full_controllable_dofs",
            "tcp_observations": [
                "observation.tcp.world_xyzrpy",
                "observation.tcp.robot_xyzrpy",
            ],
            "gripper_state_dim": 1,
            "gripper_state_definition": (
                "Primary gripper command coordinate stored as a scalar observation"
            ),
            "required_step_fields": [
                "observation.tcp.robot_xyzrpy",
                "observation.gripper_state",
                "skill.goal_position.robot_xyzrpy",
            ],
            "world_frame_definition": "sim:/World",
            "robot_base_definition": "articulation root",
            "tcp_definition": (
                "ee_frame_tcp if present, else ee_frame_body + offset_position"
            ),
            "total_episodes": len(self._completed_episodes),
            "successful_episodes": sum(
                1 for e in self._completed_episodes if e["success"]
            ),
            "episodes": self._completed_episodes,
            "front_video_generated": self.front_video_generated,
            "front_video_path": (
                str(self.front_video_path.relative_to(self._run_output_dir))
                if self.front_video_path is not None
                else None
            ),
            "front_video_camera_name": self._front_video_camera_name,
            "front_video_episode": self._front_video_episode,
            "front_video_success_type": self._front_video_success_type,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def _write_metadata(self) -> None:
        """Persist a checkpoint metadata.json for partial-progress recovery."""
        self._dataset_dir.mkdir(parents=True, exist_ok=True)
        with open(self._dataset_dir / "metadata.json", "w") as f:
            json.dump(self._build_metadata(), f, indent=2)

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

        metadata = self._build_metadata()

        with open(self._dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        path = str(self._dataset_dir.resolve())
        logger.info(
            "Finalized dataset: %s (%d episodes)", path, len(self._completed_episodes)
        )
        if self._front_video_tmp_root.exists():
            shutil.rmtree(self._front_video_tmp_root)
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

    def _maybe_save_front_video_frame(
        self,
        image: np.ndarray | None,
        frame_idx: int,
    ) -> None:
        """Persist a front-view frame for later mp4 generation."""
        if image is None or self._current_front_video_frames_dir is None:
            return
        if frame_idx % self._front_video_stride != 0:
            return
        img = np.asarray(image, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Expected (H, W, 3) uint8 image for front video, got shape {img.shape}"
            )
        self._current_front_video_frames_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._current_front_video_frames_dir / f"{self._front_video_frame_count:06d}.png"
        try:
            from PIL import Image

            Image.fromarray(img).save(out_path)
        except OSError as exc:
            raise RecorderStorageError(
                f"Failed to save front video frame to {out_path}: {exc}"
            ) from exc
        except ImportError:
            import cv2

            ok = cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not ok:
                raise RecorderStorageError(
                    f"Failed to save front video frame to {out_path}: cv2.imwrite returned False"
                )
        self._front_video_frame_count += 1

    def _cleanup_front_video_frames(self) -> None:
        if self._current_front_video_frames_dir is not None and self._current_front_video_frames_dir.exists():
            shutil.rmtree(self._current_front_video_frames_dir)
        self._current_front_video_frames_dir = None
        self._front_video_frame_count = 0

    def _finalize_front_video(
        self,
        success_priority: int,
        success_type: str | None,
    ) -> None:
        """Encode or replace the representative front-view mp4."""
        self._front_video_error = None
        frames_dir = self._current_front_video_frames_dir
        frame_count = self._front_video_frame_count
        if frames_dir is None:
            return
        try:
            if success_priority > self._front_video_priority and frame_count > 0:
                self._videos_dir.mkdir(parents=True, exist_ok=True)
                out_path = self._videos_dir / "front_success.mp4"
                meta_path = self._videos_dir / "front_success_meta.json"
                self._encode_mp4_from_frames(frames_dir, out_path)
                meta = {
                    "episode": self._episode_idx,
                    "fps": self._front_video_actual_fps,
                    "frame_count": frame_count,
                    "codec": "libx264",
                    "camera_name": self._front_video_camera_name,
                    "success_priority": int(success_priority),
                    "success_type": success_type,
                }
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
                self._front_video_generated = True
                self._front_video_priority = int(success_priority)
                self._front_video_episode = self._episode_idx
                self._front_video_path = out_path
                self._front_video_success_type = success_type
            elif success_priority <= 0 and self._videos_dir.exists() and not any(self._videos_dir.iterdir()):
                self._videos_dir.rmdir()
        except Exception as exc:
            self._front_video_error = str(exc)
            logger.warning("Failed to encode representative front video: %s", exc)
        finally:
            self._cleanup_front_video_frames()

    def _encode_mp4_from_frames(self, frames_dir: Path, out_path: Path) -> None:
        frame_paths = sorted(frames_dir.glob("*.png"))
        if not frame_paths:
            raise RecorderStorageError(
                f"No front video frames found in {frames_dir}"
            )

        try:
            import imageio.v2 as imageio
            from PIL import Image
        except ImportError as exc:
            raise RecorderStorageError(
                "imageio and Pillow are required to encode front success videos"
            ) from exc

        writer = imageio.get_writer(
            str(out_path),
            fps=self._front_video_actual_fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"],
        )
        try:
            for frame_path in frame_paths:
                frame = np.asarray(Image.open(frame_path), dtype=np.uint8)
                writer.append_data(frame)
        except Exception as exc:
            raise RecorderStorageError(
                f"Failed to encode front success video at {out_path}: {exc}"
            ) from exc
        finally:
            writer.close()

    @staticmethod
    def _coerce_pose_vector(
        pose: np.ndarray | None,
        field_name: str,
    ) -> np.ndarray:
        """Normalize a pose vector to shape (6,) float32."""
        if pose is None:
            return np.zeros(6, dtype=np.float32)
        pose_arr = np.asarray(pose, dtype=np.float32).ravel()
        if pose_arr.shape != (6,):
            raise ValueError(
                f"{field_name} has shape {pose_arr.shape}, expected (6,)"
            )
        return pose_arr

    @staticmethod
    def _coerce_gripper_state_vector(
        value: np.ndarray | None,
    ) -> np.ndarray:
        """Normalize gripper state to shape (1,) float32."""
        if value is None:
            return np.zeros(1, dtype=np.float32)
        arr = np.asarray(value, dtype=np.float32).ravel()
        if arr.shape != (1,):
            raise ValueError(f"gripper_state has shape {arr.shape}, expected (1,)")
        return arr

    def patch_skill_metadata_range(
        self,
        start_idx: int,
        end_idx: int | None = None,
        *,
        goal_joint: np.ndarray | None = None,
        goal_world_xyzrpy: np.ndarray | None = None,
        goal_robot_xyzrpy: np.ndarray | None = None,
        goal_gripper: float | None = None,
    ) -> None:
        """Patch already-recorded skill metadata entries in the current episode."""
        if end_idx is None:
            end_idx = len(self._skills)
        start = max(0, start_idx)
        end = max(start, min(end_idx, len(self._skills)))

        goal_joint_list = None
        if goal_joint is not None:
            goal_joint_list = np.asarray(goal_joint, dtype=np.float32).ravel().tolist()
        goal_world_list = None
        if goal_world_xyzrpy is not None:
            goal_world_list = self._coerce_pose_vector(
                goal_world_xyzrpy, "goal_world_xyzrpy"
            ).tolist()
        goal_robot_list = None
        if goal_robot_xyzrpy is not None:
            goal_robot_list = self._coerce_pose_vector(
                goal_robot_xyzrpy, "goal_robot_xyzrpy"
            ).tolist()

        for idx in range(start, end):
            entry = self._skills[idx]
            if goal_joint_list is not None:
                entry["goal_joint"] = goal_joint_list
            if goal_world_list is not None:
                entry["goal_world_xyzrpy"] = goal_world_list
            if goal_robot_list is not None:
                entry["goal_robot_xyzrpy"] = goal_robot_list
                if idx < len(self._goal_robot_xyzrpy_dense):
                    self._goal_robot_xyzrpy_dense[idx] = np.asarray(
                        goal_robot_list,
                        dtype=np.float32,
                    )
            if goal_gripper is not None:
                entry["goal_gripper"] = float(goal_gripper)


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

    import_errors: list[Exception] = []
    LeRobotDataset = None
    for import_path in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            module = __import__(import_path, fromlist=["LeRobotDataset"])
            LeRobotDataset = module.LeRobotDataset
            break
        except ImportError as exc:
            import_errors.append(exc)
    if LeRobotDataset is None:
        raise ImportError(
            "lerobot package is required for conversion. "
            "Install it with: pip install lerobot"
        ) from import_errors[-1]

    # Define LeRobot v3.0 features
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (total_dofs,),
            "names": [metadata["joint_names"]],
        },
        "observation.gripper_state": {
            "dtype": "float32",
            "shape": (1,),
        },
        "observation.tcp.world_xyzrpy": {
            "dtype": "float32",
            "shape": (6,),
        },
        "observation.tcp.robot_xyzrpy": {
            "dtype": "float32",
            "shape": (6,),
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
        dataset_root = raw_dir.parent / repo_id
    else:
        dataset_root = Path(output_root).expanduser().resolve() / repo_id

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_root,
    )

    for ep_info in episodes_info:
        ep_idx = ep_info["index"]
        ep_dir = raw_dir / "episodes" / f"episode_{ep_idx:06d}"
        if not ep_dir.exists():
            logger.warning("Episode dir missing, skipping: %s", ep_dir)
            continue

        states = np.load(ep_dir / "states.npy")  # (T, N_dof)
        actions = np.load(ep_dir / "actions.npy")  # (T, N_dof)
        tcp_world = (
            np.load(ep_dir / "tcp_world_xyzrpy.npy")
            if (ep_dir / "tcp_world_xyzrpy.npy").exists()
            else np.zeros((states.shape[0], 6), dtype=np.float32)
        )
        tcp_robot = (
            np.load(ep_dir / "tcp_robot_xyzrpy.npy")
            if (ep_dir / "tcp_robot_xyzrpy.npy").exists()
            else np.zeros((states.shape[0], 6), dtype=np.float32)
        )
        gripper_state = (
            np.load(ep_dir / "gripper_state.npy")
            if (ep_dir / "gripper_state.npy").exists()
            else np.zeros((states.shape[0], 1), dtype=np.float32)
        )
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
            goal_joint = _coerce_goal_joint(
                skill.get("goal_joint"),
                total_dofs=total_dofs,
            )
            goal_xyzrpy = skill.get("goal_world_xyzrpy")
            goal_robot_xyzrpy = skill.get("goal_robot_xyzrpy")

            frame: dict = {
                "observation.state": states[t],
                "observation.gripper_state": gripper_state[t],
                "observation.tcp.world_xyzrpy": tcp_world[t],
                "observation.tcp.robot_xyzrpy": tcp_robot[t],
                "action": actions[t],
                "task": ep_info.get("task_description", ""),
                "skill.natural_language": skill.get("label", ""),
                "skill.type": skill.get("type", ""),
                "skill.progress": np.array(
                    [skill.get("progress", 0.0)],
                    dtype=np.float32,
                ),
                "skill.goal_position.joint": goal_joint,
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
                "skill.goal_position.gripper": np.array(
                    [skill.get("goal_gripper", 0.0)],
                    dtype=np.float32,
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
                else:
                    frame[f"observation.images.{cam_name}"] = np.zeros(
                        (480, 640, 3),
                        dtype=np.uint8,
                    )

            dataset.add_frame(frame)

        dataset.save_episode()

    dataset.finalize()
    dataset_path = str(dataset_root.resolve())
    logger.info("LeRobot dataset created at: %s", dataset_path)
    return dataset_path


def _coerce_goal_joint(goal_joint: list[float] | None, total_dofs: int) -> np.ndarray:
    """Normalize goal_joint arrays from raw skills.json into full-DOF vectors."""
    if goal_joint is None:
        return np.zeros(total_dofs, dtype=np.float32)

    joint_arr = np.asarray(goal_joint, dtype=np.float32).ravel()
    if joint_arr.shape[0] == total_dofs:
        return joint_arr
    if joint_arr.shape[0] > total_dofs:
        return joint_arr[:total_dofs]

    padded = np.zeros(total_dofs, dtype=np.float32)
    padded[: joint_arr.shape[0]] = joint_arr
    return padded
