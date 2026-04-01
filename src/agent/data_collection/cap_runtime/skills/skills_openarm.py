"""OpenArm CaP runtime skills for generated code."""

from __future__ import annotations

import numpy as np

from .base import BaseTabletopCaPSkills, PolicyFallbackBlockedError


class OpenArmSkills(BaseTabletopCaPSkills):
    """OpenArm-specific tabletop CaP skill wrapper."""

    DEFAULT_GRASP_FACE = "top"
    PLACE_DROP_OFFSET = 0.002
    GRASP_YAW_RETRY_OFFSETS_DEG = (45.0, -45.0, 90.0, -90.0)
    GRASP_LATERAL_BIAS_CANDIDATES_M = (0.0, 0.01, -0.01, 0.015, -0.015)
    TOP_DOWN_ALIGNMENT_TOLERANCE_DEG = 25.0

    @classmethod
    def _has_hidden_grasp_override(
        cls,
        grasp_face: str | None,
        grasp_yaw_deg: float | None,
        approach_angle_deg: float | None,
    ) -> bool:
        face = cls.DEFAULT_GRASP_FACE if grasp_face is None else str(grasp_face).lower()
        return (
            approach_angle_deg is not None
            or grasp_yaw_deg is not None
            or face != cls.DEFAULT_GRASP_FACE
        )

    @classmethod
    def _candidate_grasp_yaws(cls, requested_yaw_deg: float | None) -> list[float]:
        base_yaw = 0.0 if requested_yaw_deg is None else cls._normalize_degrees(requested_yaw_deg)
        candidates = [base_yaw]
        for delta in cls.GRASP_YAW_RETRY_OFFSETS_DEG:
            candidate = cls._normalize_degrees(base_yaw + delta)
            if any(abs(candidate - existing) < 1e-6 for existing in candidates):
                continue
            candidates.append(candidate)
        return candidates

    def _resolve_openarm_grasp_spec(
        self,
        object_name: str,
        *,
        grasp_face: str = "top",
        grasp_yaw_deg: float | None = None,
    ) -> dict[str, np.ndarray | float | str]:
        """Resolve an object-relative OpenArm grasp into a world-frame pose spec."""

        local_face_axis = self._resolve_grasp_face_axis_local(grasp_face)
        object_rotation = self._object_rotation(object_name)
        surface_normal_world = self._normalize_vector(object_rotation @ local_face_axis)
        approach_direction_world = surface_normal_world
        # The grasp face normal points outward from the object. The hand's local
        # +Z axis should point back into the object, so align it with the
        # opposite of the outward surface normal while keeping approach waypoints
        # offset along the outward normal.
        desired_hand_axis_world = -surface_normal_world
        base_rotation = np.asarray(self._skills.PALM_DOWN_ROTATION, dtype=np.float64)
        base_hand_axis_world = self._normalize_vector(base_rotation[:, 2])
        align_rotation = self._align_vectors(base_hand_axis_world, desired_hand_axis_world)
        target_rotation_world = align_rotation @ base_rotation
        grasp_yaw = 0.0 if grasp_yaw_deg is None else self._normalize_degrees(grasp_yaw_deg)
        if abs(grasp_yaw) > 1e-9:
            target_rotation_world = self._axis_angle_rotation(
                approach_direction_world,
                np.radians(grasp_yaw),
            ) @ target_rotation_world
        return {
            "grasp_face": str(grasp_face).lower(),
            "grasp_yaw_deg": grasp_yaw,
            "target_rotation_world": target_rotation_world,
            "approach_direction_world": approach_direction_world,
        }

    def _clear_held_spec(self) -> None:
        """Clear held-object bookkeeping after a completed place."""

        self._clear_held_state()

    def _get_skill_lateral_bias(self) -> float:
        cfg = getattr(self._skills, "cfg", None)
        return float(getattr(cfg, "grasp_lateral_bias", 0.0) or 0.0)

    def _set_skill_lateral_bias(self, value: float) -> None:
        cfg = getattr(self._skills, "cfg", None)
        if cfg is not None:
            cfg.grasp_lateral_bias = float(value)

    def _remember_default_pick(self, object_name: str) -> None:
        """Store held-object state for the default auto-grasp path."""

        self._held_object_name = object_name
        self._held_half_height = self._resolve_half_height(object_name)
        self._held_xy_offset_world = self._capture_held_object_xy_offset(object_name)
        self._held_approach_angle_deg = None
        self._held_grasp_face = None
        self._held_grasp_yaw_deg = None
        self._held_target_rotation_world = None
        self._held_approach_direction_world = None
        self._held_grasp_lateral_bias = None

    def _store_resolved_grasp_spec(
        self,
        object_name: str,
        grasp_spec: dict[str, np.ndarray | float | str],
    ) -> None:
        """Persist the grasp spec used for the currently held object."""

        self._held_object_name = object_name
        self._held_half_height = self._resolve_half_height(object_name)
        self._held_xy_offset_world = self._capture_held_object_xy_offset(object_name)
        self._held_grasp_face = str(grasp_spec["grasp_face"])
        self._held_grasp_yaw_deg = float(grasp_spec["grasp_yaw_deg"])
        self._held_target_rotation_world = np.asarray(
            grasp_spec["target_rotation_world"],
            dtype=np.float64,
        )
        self._held_approach_direction_world = np.asarray(
            grasp_spec["approach_direction_world"],
            dtype=np.float64,
        )
        self._held_approach_angle_deg = None
        self._held_grasp_lateral_bias = self._get_skill_lateral_bias()

    def _default_pick_grasp_spec(self, object_name: str) -> tuple[str, float | None]:
        """Choose the default hidden grasp spec from affordance metadata."""

        affordances = self._object_info(object_name).get("affordances", {})
        preferred_grasp = str(affordances.get("preferred_grasp", self.DEFAULT_GRASP_FACE)).lower()
        if preferred_grasp != "top_down":
            return self.DEFAULT_GRASP_FACE, 0.0
        return self.DEFAULT_GRASP_FACE, 0.0

    def _default_pick_with_topdown_spec(
        self,
        object_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Run the default hidden top-down pick with yaw retries."""

        face, base_yaw = self._default_pick_grasp_spec(object_name)
        original_bias = self._get_skill_lateral_bias()
        attempts = [
            (candidate_yaw, lateral_bias)
            for lateral_bias in self.GRASP_LATERAL_BIAS_CANDIDATES_M
            for candidate_yaw in self._candidate_grasp_yaws(base_yaw)
        ]
        for attempt_idx, (candidate_yaw, lateral_bias) in enumerate(attempts):
            if attempt_idx > 0:
                self._skills.move_to_ready(
                    duration=1.0,
                    skill_description=f"recover to ready pose before retrying top-down pick of {object_name}",
                )
                self._skills.gripper_open(
                    duration=0.3,
                    skill_description=f"re-open gripper before retrying top-down pick of {object_name}",
                )
            grasp_spec = self._resolve_openarm_grasp_spec(
                object_name,
                grasp_face=face,
                grasp_yaw_deg=candidate_yaw,
            )
            attempt_desc = (
                f"{skill_description} (retry top-down grasp with yaw={candidate_yaw:.1f}, lateral_bias={lateral_bias:+.3f}m)"
                if skill_description and attempt_idx > 0
                else skill_description
            )
            self._set_skill_lateral_bias(float(lateral_bias))
            try:
                pick_ok = self._skills.execute_pick(
                    object_name,
                    approach_offset=self.PICK_APPROACH_OFFSET,
                    target_rotation_world=grasp_spec["target_rotation_world"],
                    approach_direction_world=grasp_spec["approach_direction_world"],
                    required_tool_axis_world=np.asarray(
                        grasp_spec["target_rotation_world"],
                        dtype=np.float64,
                    )[:, 2],
                    required_tool_axis_tolerance_deg=self.TOP_DOWN_ALIGNMENT_TOLERANCE_DEG,
                    allow_position_only_fallback=False,
                    skill_description=attempt_desc,
                )
            finally:
                self._set_skill_lateral_bias(original_bias)
            if pick_ok:
                self._set_skill_lateral_bias(float(lateral_bias))
                self._store_resolved_grasp_spec(object_name, grasp_spec)
                return True
        return False

    def execute_pick_object(
        self,
        object_position,
        object_name: str | None = None,
        grasp_face: str = "top",
        grasp_yaw_deg: float | None = None,
        approach_angle_deg: float | None = None,
        skill_description: str | None = None,
    ) -> bool:
        del object_position
        if not object_name:
            raise PolicyFallbackBlockedError(
                "execute_pick_object requires explicit object_name; runtime inference is disabled"
            )

        if not self._has_hidden_grasp_override(grasp_face, grasp_yaw_deg, approach_angle_deg):
            return self._default_pick_with_topdown_spec(
                object_name,
                skill_description=skill_description,
            )

        # Legacy compatibility path for older generated code.
        if (
            approach_angle_deg is not None
            and grasp_yaw_deg is None
            and str(grasp_face).lower() == self.DEFAULT_GRASP_FACE
        ):
            pick_ok = self._skills.execute_pick(
                object_name,
                approach_offset=self.PICK_APPROACH_OFFSET,
                approach_angle_deg=approach_angle_deg,
                allow_position_only_fallback=False,
                skill_description=skill_description,
            )
            if pick_ok:
                self._remember_default_pick(object_name)
                self._held_approach_angle_deg = approach_angle_deg
            return pick_ok

        face = self.DEFAULT_GRASP_FACE if grasp_face is None else str(grasp_face).lower()
        for attempt_idx, candidate_yaw in enumerate(self._candidate_grasp_yaws(grasp_yaw_deg)):
            if attempt_idx > 0:
                self._skills.move_to_ready(
                    duration=1.0,
                    skill_description=f"recover to ready pose before retrying pick of {object_name}",
                )
                self._skills.gripper_open(
                    duration=0.3,
                    skill_description=f"re-open gripper before retrying pick of {object_name}",
                )
            grasp_spec = self._resolve_openarm_grasp_spec(
                object_name,
                grasp_face=face,
                grasp_yaw_deg=candidate_yaw,
            )
            attempt_desc = (
                f"{skill_description} (retry with grasp_yaw_deg={candidate_yaw:.1f})"
                if skill_description and attempt_idx > 0
                else skill_description
            )
            pick_ok = self._skills.execute_pick(
                object_name,
                approach_offset=self.PICK_APPROACH_OFFSET,
                target_rotation_world=grasp_spec["target_rotation_world"],
                approach_direction_world=grasp_spec["approach_direction_world"],
                required_tool_axis_world=np.asarray(
                    grasp_spec["target_rotation_world"],
                    dtype=np.float64,
                )[:, 2],
                required_tool_axis_tolerance_deg=self.TOP_DOWN_ALIGNMENT_TOLERANCE_DEG,
                allow_position_only_fallback=False,
                skill_description=attempt_desc,
            )
            if pick_ok:
                self._store_resolved_grasp_spec(object_name, grasp_spec)
                return True
        return False

    def execute_place_object(
        self,
        place_position,
        target_name: str | None = None,
        is_table: bool = True,
        grasp_face: str | None = None,
        grasp_yaw_deg: float | None = None,
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
        held_xy_offset = self._capture_held_object_xy_offset(held_name)
        if held_xy_offset is not None:
            self._held_xy_offset_world = held_xy_offset

        explicit_override = self._has_hidden_grasp_override(
            grasp_face,
            grasp_yaw_deg,
            approach_angle_deg,
        )

        if not explicit_override and self._held_target_rotation_world is None and self._held_approach_angle_deg is None:
            original_bias = self._get_skill_lateral_bias()
            if self._held_grasp_lateral_bias is not None:
                self._set_skill_lateral_bias(float(self._held_grasp_lateral_bias))
            try:
                placed = self._skills.execute_place(
                    target_center,
                    approach_offset=self.PLACE_APPROACH_OFFSET,
                    drop_offset=self.PLACE_DROP_OFFSET,
                    _placed_object=held_name,
                    _held_xy_offset_world=self._held_xy_offset_world,
                    allow_position_only_fallback=False,
                    skill_description=skill_description,
                )
            finally:
                self._set_skill_lateral_bias(original_bias)
            if placed:
                self._clear_held_spec()
            return placed

        # Legacy compatibility path for older generated code.
        effective_approach_angle = (
            self._held_approach_angle_deg
            if approach_angle_deg is None and grasp_face is None and grasp_yaw_deg is None
            else approach_angle_deg
        )
        if (
            effective_approach_angle is not None
            and grasp_face is None
            and grasp_yaw_deg is None
            and self._held_target_rotation_world is None
        ):
            placed = self._skills.execute_place(
                target_center,
                approach_offset=self.PLACE_APPROACH_OFFSET,
                drop_offset=self.PLACE_DROP_OFFSET,
                _placed_object=held_name,
                _held_xy_offset_world=self._held_xy_offset_world,
                approach_angle_deg=effective_approach_angle,
                allow_position_only_fallback=False,
                skill_description=skill_description,
            )
            if placed:
                self._clear_held_spec()
            return placed

        if grasp_face is None and grasp_yaw_deg is None and self._held_target_rotation_world is not None:
            target_rotation_world = self._held_target_rotation_world
            approach_direction_world = self._held_approach_direction_world
        else:
            face = self._held_grasp_face if grasp_face is None else str(grasp_face).lower()
            if face is None:
                face = self.DEFAULT_GRASP_FACE
            yaw = self._held_grasp_yaw_deg if grasp_yaw_deg is None else grasp_yaw_deg
            grasp_spec = self._resolve_openarm_grasp_spec(
                held_name,
                grasp_face=face,
                grasp_yaw_deg=yaw,
            )
            target_rotation_world = np.asarray(grasp_spec["target_rotation_world"], dtype=np.float64)
            approach_direction_world = np.asarray(grasp_spec["approach_direction_world"], dtype=np.float64)

        original_bias = self._get_skill_lateral_bias()
        if self._held_grasp_lateral_bias is not None:
            self._set_skill_lateral_bias(float(self._held_grasp_lateral_bias))
        try:
            placed = self._skills.execute_place(
                target_center,
                approach_offset=self.PLACE_APPROACH_OFFSET,
                drop_offset=self.PLACE_DROP_OFFSET,
                _placed_object=held_name,
                _held_xy_offset_world=self._held_xy_offset_world,
                target_rotation_world=target_rotation_world,
                approach_direction_world=approach_direction_world,
                required_tool_axis_world=np.asarray(target_rotation_world, dtype=np.float64)[:, 2],
                required_tool_axis_tolerance_deg=self.TOP_DOWN_ALIGNMENT_TOLERANCE_DEG,
                allow_position_only_fallback=False,
                skill_description=skill_description,
            )
        finally:
            self._set_skill_lateral_bias(original_bias)
        if placed:
            self._clear_held_spec()
        return placed
