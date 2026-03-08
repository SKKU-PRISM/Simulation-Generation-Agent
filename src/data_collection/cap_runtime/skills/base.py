"""Embodiment runtime wrappers used by generated CaP code."""

from __future__ import annotations

import numpy as np

from src.data_collection.cap_runtime.context import CaPRuntimeContext


class PolicyFallbackBlockedError(RuntimeError):
    """Raised when generated CaP code relied on a forbidden policy fallback."""


class BaseTabletopCaPSkills:
    """Shared tabletop skill wrapper on top of ``SimSkills``."""

    PICK_APPROACH_OFFSET = 0.10
    PLACE_APPROACH_OFFSET = 0.05
    PLACE_DROP_OFFSET = 0.005
    GRASP_FACE_AXES = {
        "top": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        "bottom": np.array([0.0, 0.0, -1.0], dtype=np.float64),
        "front": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "back": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        "left": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "right": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    }

    def __init__(self, *_, frame: str = "world", **__):
        resources = CaPRuntimeContext.get()
        self._resources = resources
        self._skills = resources.sim_skills
        self._detector = resources.detector
        self._robot_cfg = resources.robot_cfg
        self._positions = resources.translated_positions
        self.frame = frame
        self._held_object_name: str | None = None
        self._held_half_height: float | None = None
        self._held_xy_offset_world: np.ndarray | None = None
        self._held_approach_angle_deg: float | None = None
        self._held_grasp_face: str | None = None
        self._held_grasp_yaw_deg: float | None = None
        self._held_target_rotation_world: np.ndarray | None = None
        self._held_approach_direction_world: np.ndarray | None = None

    def connect(self) -> bool:
        return True

    def disconnect(self) -> bool:
        return True

    def detect_objects(self, queries: list[str] | None = None, **_) -> dict[str, dict]:
        if not queries:
            return dict(self._positions)
        return {name: self._positions.get(name) for name in queries}

    def move_to_ready(self, skill_description: str | None = None) -> bool:
        return self._skills.move_to_ready(
            duration=1.0,
            skill_description=skill_description,
        )

    def move_to_initial_state(self, skill_description: str | None = None) -> bool:
        return self.move_to_ready(skill_description=skill_description)

    def move_to_free_state(self, skill_description: str | None = None) -> bool:
        return self._skills.move_to_ready(
            duration=1.2,
            safe_retreat=True,
            skill_description=skill_description,
        )

    def move_to_position(
        self,
        position,
        target_name: str | None = None,
        skill_description: str | None = None,
    ) -> bool:
        return self._skills.move_to_position(
            np.asarray(position, dtype=np.float64),
            target_name=target_name,
            skill_description=skill_description,
        )

    def gripper_open(
        self,
        skill_description: str | None = None,
        ratio: float = 1.0,
    ) -> bool:
        del ratio
        return self._skills.gripper_open(
            duration=0.3,
            skill_description=skill_description,
        )

    def gripper_close(self, skill_description: str | None = None) -> bool:
        return self._skills.gripper_close(
            duration=self._robot_cfg.gripper_close_duration,
            skill_description=skill_description,
        )

    def rotate_90degree(self, *_, **__) -> bool:
        raise RuntimeError("rotate_90degree is not supported in the IsaacLab CaP tabletop runtime")

    def execute_pick_object(
        self,
        object_position,
        object_name: str | None = None,
        approach_angle_deg: float | None = None,
        skill_description: str | None = None,
    ) -> bool:
        del object_position
        if not object_name:
            raise PolicyFallbackBlockedError(
                "execute_pick_object requires explicit object_name; runtime inference is disabled"
            )

        pick_ok = self._skills.execute_pick(
            object_name,
            approach_offset=self.PICK_APPROACH_OFFSET,
            approach_angle_deg=approach_angle_deg,
            skill_description=skill_description,
        )
        if pick_ok:
            self._held_object_name = object_name
            self._held_half_height = self._resolve_half_height(object_name)
            self._held_xy_offset_world = self._capture_held_object_xy_offset(object_name)
            self._held_approach_angle_deg = approach_angle_deg
        return pick_ok

    def execute_place_object(
        self,
        place_position,
        target_name: str | None = None,
        is_table: bool = True,
        approach_angle_deg: float | None = None,
        skill_description: str | None = None,
        **_,
    ) -> bool:
        held_name = self._held_object_name
        if held_name is None:
            return False
        if not is_table and not target_name:
            raise PolicyFallbackBlockedError(
                "execute_place_object with is_table=False requires explicit target_name"
            )

        target_center = self._resolve_place_center(place_position, target_name, is_table)
        self._skills.move_to_ready(
            duration=1.0,
            open_gripper=False,
            skill_description=(
                f"{skill_description} (carry the held object through the ready pose before placement)"
                if skill_description
                else "carry the held object through the ready pose before placement"
            ),
        )
        placed = self._skills.execute_place(
            target_center,
            approach_offset=self.PLACE_APPROACH_OFFSET,
            drop_offset=self.PLACE_DROP_OFFSET,
            _placed_object=held_name,
            approach_angle_deg=approach_angle_deg,
            skill_description=skill_description,
        )
        if placed:
            self._clear_held_state()
        return placed

    def _clear_held_state(self) -> None:
        """Clear bookkeeping for the currently held object."""

        self._held_object_name = None
        self._held_half_height = None
        self._held_xy_offset_world = None
        self._held_approach_angle_deg = None
        self._held_grasp_face = None
        self._held_grasp_yaw_deg = None
        self._held_target_rotation_world = None
        self._held_approach_direction_world = None

    def _current_object_position(self, object_name: str) -> np.ndarray:
        """Resolve the latest center position for a named object."""

        try:
            current_pos = self._detector.get_object_position(object_name)
        except Exception as exc:
            raise PolicyFallbackBlockedError(
                f"unable to read live pose for '{object_name}' during placement: {exc}"
            ) from exc
        return np.asarray(current_pos, dtype=np.float64)

    def _capture_held_object_xy_offset(self, object_name: str) -> np.ndarray | None:
        """Measure the held object's live XY offset from the EE frame."""

        if self._detector is None:
            return None
        try:
            object_pos = np.asarray(self._detector.get_object_position(object_name), dtype=np.float64)
            ee_pos, _ = self._skills.robot.read_ee_pose()
        except Exception:
            return self._held_xy_offset_world

        ee_pos = np.asarray(ee_pos, dtype=np.float64)
        xy_offset = object_pos[:2] - ee_pos[:2]
        if not np.all(np.isfinite(xy_offset)):
            return self._held_xy_offset_world
        return np.clip(xy_offset, -0.04, 0.04)

    def _resolve_half_height(self, object_name: str | None) -> float:
        """Resolve the best available half-height estimate for an object."""

        if not object_name:
            raise PolicyFallbackBlockedError(
                "half-height lookup requires explicit object_name"
            )

        info = self._positions.get(object_name)
        if info is None:
            raise PolicyFallbackBlockedError(
                f"object '{object_name}' missing from translated positions"
            )

        estimated = info.get("estimated_half_height")
        if estimated is not None:
            return max(float(estimated), 0.005)

        position = info.get("position")
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            return max(float(position[2]) / 2.0, 0.005)

        raise PolicyFallbackBlockedError(
            f"no half-height metadata available for '{object_name}'"
        )

    def _resolve_place_center(
        self,
        place_position,
        target_name: str | None,
        is_table: bool,
    ) -> np.ndarray:
        """Resolve the desired center position for the placed object."""

        target = np.asarray(place_position, dtype=np.float64).copy()
        held_half_height = self._held_half_height or 0.02

        if is_table:
            target[2] = float(target[2]) + held_half_height
            return target

        if not target_name:
            raise PolicyFallbackBlockedError(
                "object-on-object placement requires explicit target_name"
            )

        target_center = self._current_object_position(target_name)
        target_half_height = self._resolve_half_height(target_name)
        target_center[2] = float(target_center[2]) + target_half_height + held_half_height
        return target_center

    @staticmethod
    def _normalize_degrees(angle_deg: float) -> float:
        """Normalize degrees to [-180, 180)."""

        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

    @staticmethod
    def _normalize_vector(vector: np.ndarray) -> np.ndarray:
        """Return a normalized copy of a vector."""

        vec = np.asarray(vector, dtype=np.float64).reshape(-1)
        norm = np.linalg.norm(vec)
        if norm < 1e-9:
            raise PolicyFallbackBlockedError("zero-length grasp direction is invalid")
        return vec / norm

    @staticmethod
    def _quat_to_rotation_matrix(quat_wxyz) -> np.ndarray:
        """Convert wxyz quaternion to 3x3 rotation matrix."""

        quat = np.asarray(quat_wxyz if quat_wxyz is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        w, x, y, z = quat
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64)

    @staticmethod
    def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
        """Return a rotation matrix for an axis-angle rotation."""

        axis = BaseTabletopCaPSkills._normalize_vector(axis)
        x, y, z = axis
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        one_c = 1.0 - c
        return np.array([
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ], dtype=np.float64)

    @classmethod
    def _align_vectors(cls, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Return the rotation matrix aligning source to target."""

        src = cls._normalize_vector(source)
        dst = cls._normalize_vector(target)
        cross = np.cross(src, dst)
        dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        cross_norm = np.linalg.norm(cross)

        if cross_norm < 1e-9:
            if dot > 0.0:
                return np.eye(3, dtype=np.float64)
            fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(src, fallback)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = cls._normalize_vector(np.cross(src, fallback))
            return cls._axis_angle_rotation(axis, np.pi)

        skew = np.array([
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + skew + (skew @ skew) * ((1.0 - dot) / (cross_norm ** 2))

    def _object_info(self, object_name: str) -> dict:
        """Return translated object metadata."""

        info = self._positions.get(object_name)
        if info is None:
            raise PolicyFallbackBlockedError(
                f"object '{object_name}' missing from translated positions"
            )
        return info

    def _object_rotation(self, object_name: str) -> np.ndarray:
        """Return object rotation matrix from translated metadata."""

        return self._quat_to_rotation_matrix(self._object_info(object_name).get("quaternion"))

    def _resolve_grasp_face_axis_local(self, grasp_face: str) -> np.ndarray:
        """Map a grasp face string to an object-local unit axis."""

        axis = self.GRASP_FACE_AXES.get(str(grasp_face).lower())
        if axis is None:
            raise PolicyFallbackBlockedError(
                f"unsupported grasp_face '{grasp_face}'; expected one of {sorted(self.GRASP_FACE_AXES)}"
            )
        return axis.copy()
