"""Embodiment runtime wrappers used by generated CaP code."""

from __future__ import annotations

import numpy as np

from src.agent.data_collection.cap_runtime.context import CaPRuntimeContext


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
    _PRECISE_TABLE_TARGET_TOKENS = (
        "marker",
        "zone",
        "anchor",
        "command_pose",
        "tray",
        "bin",
        "container",
        "collection",
        "slot",
        "cutout",
    )

    def __init__(self, *_, frame: str = "world", **__):
        resources = CaPRuntimeContext.get()
        self._resources = resources
        self._skills = resources.sim_skills
        self._detector = resources.detector
        self._robot_cfg = resources.robot_cfg
        self._positions = resources.translated_positions
        self._cameras = resources.cameras
        self._target_grounder = resources.target_grounder
        self.frame = frame
        self._held_object_name: str | None = None
        self._held_half_height: float | None = None
        self._held_xy_offset_world: np.ndarray | None = None
        self._held_approach_angle_deg: float | None = None
        self._held_grasp_face: str | None = None
        self._held_grasp_yaw_deg: float | None = None
        self._held_target_rotation_world: np.ndarray | None = None
        self._held_approach_direction_world: np.ndarray | None = None
        self._held_grasp_lateral_bias: float | None = None

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
        *,
        target_rotation_world: np.ndarray | None = None,
        approach_direction_world: np.ndarray | None = None,
        required_tool_axis_world: np.ndarray | None = None,
        allow_position_only_fallback: bool = True,
    ) -> bool:
        if not object_name:
            raise PolicyFallbackBlockedError(
                "execute_pick_object requires explicit object_name; runtime inference is disabled"
            )
        runtime_object_name = self._resolve_runtime_object_name(object_name)
        object_entry = self._lookup_translated_entry(object_name) or {}
        runtime_object_entry = self._lookup_translated_entry(runtime_object_name) or {}
        if runtime_object_entry is object_entry:
            merged_object_entry = object_entry
        else:
            # Semantic aliases such as ``shape_to_place`` or ``peg`` should
            # inherit low-level grasp metadata from the concrete runtime object
            # while still allowing alias-specific fields to override.
            merged_object_entry = dict(runtime_object_entry)
            merged_object_entry.update(object_entry)
        object_entry = merged_object_entry
        pick_position = object_entry.get(
            "pick_position",
            object_entry.get("composite_pick_position", object_position),
        )
        pick_position_candidates = object_entry.get("pick_position_candidates")
        if isinstance(pick_position_candidates, (list, tuple)) and pick_position_candidates:
            candidate_positions = [
                np.asarray(candidate, dtype=np.float64)
                for candidate in pick_position_candidates
                if isinstance(candidate, (list, tuple)) and len(candidate) == 3
            ]
            if not candidate_positions:
                candidate_positions = [np.asarray(pick_position, dtype=np.float64)]
        else:
            candidate_positions = [np.asarray(pick_position, dtype=np.float64)]
        approach_offset = float(object_entry.get("approach_offset_override", self.PICK_APPROACH_OFFSET))
        grasp_offset = object_entry.get("grasp_offset_override")
        if grasp_offset is not None:
            grasp_offset = float(grasp_offset)
        grasp_surface_clearance = object_entry.get("grasp_surface_clearance_override")
        if grasp_surface_clearance is not None:
            if grasp_offset is None:
                grasp_offset = float(self._robot_cfg.ee_finger_offset)
            grasp_offset += float(grasp_surface_clearance)
        close_duration_override = object_entry.get("gripper_close_duration_override")
        if close_duration_override is not None:
            close_duration_override = float(close_duration_override)
        breakout_lift_override = object_entry.get("pick_breakout_lift_override")
        if breakout_lift_override is not None:
            breakout_lift_override = float(breakout_lift_override)
        default_rotation = target_rotation_world
        if default_rotation is None:
            default_rotation = self._metadata_rotation(object_entry, field="pick_target_rotation_world")
        default_direction = approach_direction_world
        if default_direction is None:
            default_direction = self._metadata_vector(object_entry, "pick_approach_direction_world")
        default_required_axis = required_tool_axis_world
        if default_required_axis is None:
            default_required_axis = self._metadata_vector(object_entry, "pick_required_tool_axis_world")

        yaw_candidates = object_entry.get("pick_yaw_candidates_deg")
        if not isinstance(yaw_candidates, (list, tuple)):
            if str(object_entry.get("preferred_grasp_region", "")).lower() == "outer_rim":
                yaw_candidates = [0.0, 90.0, -90.0, 45.0, -45.0]
            else:
                yaw_candidates = [0.0]

        pick_ok = False
        for candidate_position in candidate_positions:
            for yaw_deg in yaw_candidates:
                base_rotation = default_rotation
                if base_rotation is None and abs(float(yaw_deg)) > 1e-6:
                    base_rotation = np.asarray(self._skills.PALM_DOWN_ROTATION, dtype=np.float64)
                rotated_target = base_rotation
                if base_rotation is not None and abs(float(yaw_deg)) > 1e-6:
                    rotated_target = np.asarray(base_rotation, dtype=np.float64) @ self._axis_angle_rotation(
                        np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        np.radians(float(yaw_deg)),
                    )
                pick_kwargs = dict(
                    approach_offset=approach_offset,
                    grasp_offset=grasp_offset,
                    approach_angle_deg=approach_angle_deg,
                    object_position_override=np.asarray(candidate_position, dtype=np.float64),
                    target_rotation_world=rotated_target,
                    approach_direction_world=default_direction,
                    required_tool_axis_world=default_required_axis,
                    allow_position_only_fallback=allow_position_only_fallback,
                    skill_description=skill_description,
                )
                if close_duration_override is not None:
                    pick_kwargs["close_duration_override"] = close_duration_override
                if breakout_lift_override is not None:
                    pick_kwargs["breakout_lift_override"] = breakout_lift_override
                pick_ok = self._skills.execute_pick(
                    runtime_object_name,
                    **pick_kwargs,
                )
                if pick_ok:
                    break
            if pick_ok:
                break

        if pick_ok:
            self._held_object_name = runtime_object_name
            self._held_half_height = self._resolve_half_height(runtime_object_name)
            self._held_xy_offset_world = self._capture_held_object_xy_offset(runtime_object_name)
            self._held_approach_angle_deg = approach_angle_deg
        return pick_ok

    def execute_place_object(
        self,
        place_position,
        target_name: str | None = None,
        is_table: bool = True,
        approach_angle_deg: float | None = None,
        skill_description: str | None = None,
        *,
        target_rotation_world: np.ndarray | None = None,
        approach_direction_world: np.ndarray | None = None,
        required_tool_axis_world: np.ndarray | None = None,
        allow_position_only_fallback: bool = True,
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
        held_entry = self._lookup_translated_entry(held_name) or {}
        approach_offset = float(held_entry.get("place_approach_offset_override", self.PLACE_APPROACH_OFFSET))
        drop_offset = float(held_entry.get("place_drop_offset_override", self.PLACE_DROP_OFFSET))
        if is_table and self._is_precise_table_target(target_name):
            approach_offset = min(approach_offset, 0.035)
            drop_offset = min(drop_offset, 0.002)
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
            approach_offset=approach_offset,
            drop_offset=drop_offset,
            _placed_object=held_name,
            _held_xy_offset_world=self._held_xy_offset_world,
            approach_angle_deg=approach_angle_deg,
            target_rotation_world=target_rotation_world,
            approach_direction_world=approach_direction_world,
            required_tool_axis_world=required_tool_axis_world,
            allow_position_only_fallback=allow_position_only_fallback,
            skill_description=skill_description,
        )
        if placed:
            self._clear_held_state()
        return placed

    def execute_place_on_target(
        self,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Place the currently held object onto a named tabletop target."""

        target_entry = self._resolve_named_target_entry(target_name)
        return self.execute_place_object(
            target_entry["position"],
            target_name=target_name,
            is_table=True,
            skill_description=skill_description,
        )

    def execute_place_in_container(
        self,
        container_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Place the currently held object into a named tray/bin/container target."""

        container_entry = self._resolve_named_target_entry(container_name)
        if self._is_dynamic_articulated_container(container_entry):
            self._refresh_dynamic_target_entry(container_name, container_entry)
            target_position = np.asarray(
                container_entry.get("target_position", container_entry["position"]),
                dtype=np.float64,
            ).copy()
            target_position[2] = float(container_entry.get("support_top_z", target_position[2]))
            palm_down = getattr(self._skills, "PALM_DOWN_ROTATION", None)
            target_rotation = None if palm_down is None else np.asarray(palm_down, dtype=np.float64)
            return self.execute_place_object(
                target_position,
                target_name=container_name,
                is_table=True,
                skill_description=skill_description,
                target_rotation_world=target_rotation,
                required_tool_axis_world=np.array([0.0, 0.0, -1.0], dtype=np.float64),
                allow_position_only_fallback=False,
            )

        anchor_name = self._resolve_container_anchor_name(container_name)
        anchor_entry = self._resolve_named_target_entry(anchor_name)
        return self.execute_place_object(
            anchor_entry["position"],
            target_name=anchor_name,
            is_table=True,
            skill_description=skill_description,
        )

    def execute_stack_on_object(
        self,
        bottom_object_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Place the currently held object on top of another named object."""

        target_entry = self._resolve_named_target_entry(bottom_object_name)
        return self.execute_place_object(
            target_entry["position"],
            target_name=bottom_object_name,
            is_table=False,
            skill_description=skill_description,
        )

    def execute_pick_and_place_on_target(
        self,
        object_name: str,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object and place it on a named tabletop target."""

        object_entry = self._resolve_named_target_entry(object_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False
        return self.execute_place_on_target(
            target_name=target_name,
            skill_description=skill_description,
        )

    def execute_pick_and_place_in_container(
        self,
        object_name: str,
        container_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object and place it into a tray/bin/container target."""

        object_entry = self._resolve_named_target_entry(object_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False
        return self.execute_place_in_container(
            container_name=container_name,
            skill_description=skill_description,
        )

    def execute_pick_and_stack_on_object(
        self,
        object_name: str,
        bottom_object_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object and stack it on another named object."""

        object_entry = self._resolve_named_target_entry(object_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False
        return self.execute_stack_on_object(
            bottom_object_name=bottom_object_name,
            skill_description=skill_description,
        )

    def execute_lift_to_pose(
        self,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Move the currently held object to a named commanded pose while keeping hold."""

        if self._held_object_name is None:
            raise PolicyFallbackBlockedError(
                "execute_lift_to_pose requires an already held object"
            )
        target_entry = self._resolve_named_target_entry(target_name)
        moved = self._skills.move_to_position(
            np.asarray(target_entry["position"], dtype=np.float64),
            target_name=target_name,
            skill_description=skill_description,
        )
        if moved:
            self._held_xy_offset_world = self._capture_held_object_xy_offset(self._held_object_name)
        return bool(moved)

    def execute_pick_and_lift_to_pose(
        self,
        object_name: str,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object and lift/hold it at a named commanded pose."""

        object_entry = self._resolve_named_target_entry(object_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False
        return self.execute_lift_to_pose(
            target_name=target_name,
            skill_description=skill_description,
        )

    def execute_pick_and_place_on_support(
        self,
        object_name: str,
        support_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick an object and place it on a named elevated support surface."""

        object_entry = self._resolve_named_target_entry(object_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False

        support_entry = self._resolve_named_target_entry(support_name)
        target_position = np.asarray(support_entry["position"], dtype=np.float64).copy()
        alignment_target = support_entry.get("support_alignment_target")
        if isinstance(alignment_target, str) and alignment_target in self._positions:
            target_position[:2] = np.asarray(
                self._positions[alignment_target]["position"][:2],
                dtype=np.float64,
            )

        return self.execute_place_object(
            target_position,
            target_name=support_name,
            is_table=True,
            skill_description=skill_description,
        )

    def execute_pull_handle_open(
        self,
        handle_name: str,
        open_fraction: float | None = None,
        skill_description: str | None = None,
    ) -> bool:
        """Grasp a named handle and pull it along its configured open axis."""

        handle_entry = self._resolve_named_target_entry(handle_name)
        handle_pos = self._resolve_handle_contact_position(handle_name, handle_entry, stage="open")
        pull_axis = self._metadata_vector(handle_entry, "pull_axis_world")
        if pull_axis is None:
            raise PolicyFallbackBlockedError(
                f"handle target '{handle_name}' is missing pull_axis_world metadata"
            )
        pull_axis = self._normalize_vector(pull_axis)
        articulation_name = handle_entry.get("articulation_name")
        joint_name = handle_entry.get("joint_name")
        open_joint_position = handle_entry.get("open_target_joint_position", handle_entry.get("target_joint_position"))
        close_joint_position = handle_entry.get("close_target_joint_position", 0.0)
        has_articulated_target = (
            isinstance(articulation_name, str)
            and articulation_name
            and isinstance(joint_name, str)
            and joint_name
            and open_joint_position is not None
        )
        target_rotation = self._metadata_rotation(handle_entry)
        current_joint = None
        target_value = None
        if has_articulated_target:
            current_joint = self._read_articulation_joint_position(articulation_name, joint_name)
            open_value = float(open_joint_position)
            close_value = float(close_joint_position or 0.0)
            target_value = open_value
            if open_fraction is not None:
                requested = float(np.clip(open_fraction, 0.0, 1.0))
                target_value = close_value + (open_value - close_value) * requested
            reference_joint = current_joint if current_joint is not None else close_value
            pull_distance = max(float(target_value - reference_joint), 0.0)
        else:
            pull_distance = float(handle_entry.get("pull_distance", 0.25))
            if open_fraction is not None:
                pull_distance *= float(np.clip(open_fraction, 0.0, 1.0))

        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper before grasping {handle_name})"
                if skill_description
                else f"open the gripper before grasping {handle_name}"
            )
        )

        approach_pos = handle_pos + pull_axis * 0.06
        if has_articulated_target:
            approach_ok = bool(self._skills.move_to_position(approach_pos))
        else:
            approach_ok = self._move_tool_to_pose(
                approach_pos,
                target_rotation_world=target_rotation,
                allow_position_only_fallback=False,
            )
        if not approach_ok and not has_articulated_target:
            return False

        self.gripper_close(
            skill_description=(
                f"{skill_description} (close the gripper to grasp {handle_name})"
                if skill_description
                else f"close the gripper to grasp {handle_name}"
            )
        )

        preview_distance = min(pull_distance, 0.03) if has_articulated_target else pull_distance
        preview_target = handle_pos + pull_axis * preview_distance
        if has_articulated_target:
            pulled = bool(self._skills.move_to_position(preview_target))
        else:
            pulled = self._move_tool_to_pose(
                preview_target,
                target_rotation_world=target_rotation,
                allow_position_only_fallback=False,
            )
        if has_articulated_target and target_value is not None:
            joint_driven = self._drive_articulation_joint(
                articulation_name=articulation_name,
                joint_name=joint_name,
                target_position=float(target_value),
            )
            pulled = bool(joint_driven)
        elif pulled:
            pull_target = handle_pos + pull_axis * pull_distance
            pulled = self._move_tool_to_pose(
                pull_target,
                target_rotation_world=target_rotation,
                allow_position_only_fallback=False,
            )
        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper to release {handle_name})"
                if skill_description
                else f"open the gripper to release {handle_name}"
            )
        )
        if not pulled:
            return False
        retreat_pos = preview_target - pull_axis * 0.03
        if has_articulated_target:
            self._skills.move_to_position(retreat_pos)
        else:
            self._move_tool_to_pose(
                retreat_pos,
                target_rotation_world=target_rotation,
                allow_position_only_fallback=False,
            )
        if isinstance(articulation_name, str) and articulation_name:
            self._refresh_related_articulated_targets(articulation_name)
        if has_articulated_target and target_value is not None:
            actual_joint = self._read_articulation_joint_position(articulation_name, joint_name)
            if actual_joint is None:
                return False
            tolerance = max(0.02, 0.15 * abs(float(target_value - float(close_joint_position or 0.0))))
            return bool(actual_joint >= float(target_value) - tolerance)
        return True

    def execute_push_handle_closed(
        self,
        handle_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Grasp a named handle and push it along its closing axis until the drawer closes."""

        handle_entry = self._resolve_named_target_entry(handle_name)
        handle_pos = self._resolve_handle_contact_position(handle_name, handle_entry, stage="close")
        pull_axis = self._metadata_vector(handle_entry, "pull_axis_world")
        if pull_axis is None:
            raise PolicyFallbackBlockedError(
                f"handle target '{handle_name}' is missing pull_axis_world metadata"
            )
        pull_axis = self._normalize_vector(pull_axis)
        push_axis = -pull_axis
        articulation_name = handle_entry.get("articulation_name")
        joint_name = handle_entry.get("joint_name")
        close_target = float(handle_entry.get("close_target_joint_position", 0.0) or 0.0)
        if not isinstance(articulation_name, str) or not articulation_name:
            raise PolicyFallbackBlockedError(
                f"handle target '{handle_name}' is missing articulation_name metadata"
            )
        if not isinstance(joint_name, str) or not joint_name:
            raise PolicyFallbackBlockedError(
                f"handle target '{handle_name}' is missing joint_name metadata"
            )
        current_joint = self._read_articulation_joint_position(articulation_name, joint_name)
        if current_joint is None:
            raise PolicyFallbackBlockedError(
                f"unable to read articulation joint position for handle target '{handle_name}'"
            )
        push_distance = max(float(current_joint - close_target), 0.0)
        if push_distance <= 1e-4:
            self._refresh_related_articulated_targets(articulation_name)
            return True
        target_rotation = self._metadata_rotation(handle_entry)

        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper before grasping {handle_name})"
                if skill_description
                else f"open the gripper before grasping {handle_name}"
            )
        )

        approach_pos = handle_pos + pull_axis * 0.06
        approach_ok = bool(self._skills.move_to_position(approach_pos))
        if not approach_ok:
            return False

        self.gripper_close(
            skill_description=(
                f"{skill_description} (close the gripper to grasp {handle_name})"
                if skill_description
                else f"close the gripper to grasp {handle_name}"
            )
        )

        preview_distance = min(push_distance, 0.03)
        preview_target = handle_pos + push_axis * preview_distance
        self._skills.move_to_position(preview_target)
        pushed = self._drive_articulation_joint(
            articulation_name=articulation_name,
            joint_name=joint_name,
            target_position=close_target,
        )
        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper to release {handle_name})"
                if skill_description
                else f"open the gripper to release {handle_name}"
            )
        )
        if not pushed:
            return False
        self._skills.move_to_position(preview_target - push_axis * 0.03)
        self._refresh_related_articulated_targets(articulation_name)
        actual_joint = self._read_articulation_joint_position(articulation_name, joint_name)
        if actual_joint is None:
            return False
        return bool(actual_joint <= close_target + 0.02)

    def execute_set_handle_open_fraction(
        self,
        handle_name: str,
        open_fraction: float,
        skill_description: str | None = None,
    ) -> bool:
        """Set a named articulated handle to a normalized open fraction."""
        requested_fraction = float(np.clip(open_fraction, 0.0, 1.0))
        if requested_fraction <= 1e-6:
            success = self.execute_push_handle_closed(
                handle_name=handle_name,
                skill_description=skill_description,
            )
            return success
        success = self.execute_pull_handle_open(
            handle_name=handle_name,
            open_fraction=requested_fraction,
            skill_description=skill_description,
        )
        return success

    def execute_pick_and_place_upright(
        self,
        object_name: str,
        support_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick an object and place it upright on a named support surface."""

        object_entry = self._resolve_named_target_entry(object_name)
        support_entry = self._resolve_named_target_entry(support_name)
        current_position = np.asarray(object_entry["position"], dtype=np.float64)
        object_center_target = np.asarray(support_entry["position"], dtype=np.float64).copy()
        if support_name == "table_surface":
            object_center_target[:2] = current_position[:2]
        support_top_z = float(support_entry.get("support_top_z", object_center_target[2]))
        upright_vertical_extent = self._resolve_upright_vertical_extent(object_entry)
        object_center_target[2] = support_top_z + upright_vertical_extent
        release_depth = float(object_entry.get("upright_release_depth", 0.0) or 0.0)
        settle_steps = max(int(object_entry.get("upright_settle_steps", 8) or 0), 0)
        upright_rotation = self._metadata_rotation(object_entry, field="upright_target_rotation_world")
        if upright_rotation is None:
            palm_down = np.asarray(self._skills.PALM_DOWN_ROTATION, dtype=np.float64)
            upright_rotation = palm_down @ self._axis_angle_rotation(np.array([0.0, 1.0, 0.0]), -np.pi / 2.0)
        initial_object_rot = self._quat_to_rotation_matrix(object_entry.get("quaternion", [1.0, 0.0, 0.0, 0.0]))
        grasp_offset_candidates = self._resolve_upright_grasp_offset_candidates_local(object_entry)

        chosen_grasp_offset_local: np.ndarray | None = None
        for grasp_offset_local in grasp_offset_candidates:
            pick_position = current_position + initial_object_rot @ grasp_offset_local
            picked = self.execute_pick_object(
                pick_position,
                object_name=object_name,
                skill_description=skill_description,
            )
            if picked:
                chosen_grasp_offset_local = grasp_offset_local
                break
        if chosen_grasp_offset_local is None:
            return False

        self.move_to_ready(
            skill_description=(
                f"{skill_description} (carry the held object into the upright placement approach)"
                if skill_description
                else "carry the held object into the upright placement approach"
            )
        )

        grasp_offset_world = np.asarray(upright_rotation, dtype=np.float64) @ chosen_grasp_offset_local
        release_position = object_center_target + grasp_offset_world
        if release_depth > 0.0:
            release_position = release_position.copy()
            release_position[2] = max(release_position[2] - release_depth, support_top_z + 0.5 * upright_vertical_extent)
        approach_offset = float(object_entry.get("place_approach_offset_override", self.PLACE_APPROACH_OFFSET))
        approach_position = release_position.copy()
        approach_position[2] += approach_offset

        move_with_retry = getattr(self._skills, "_move_with_target_rotation_until_reached", None)
        if callable(move_with_retry):
            approach_ok = bool(
                move_with_retry(
                    approach_position,
                    target_rotation_world=upright_rotation,
                    allow_position_only_fallback=False,
                    retry_durations=[1.25],
                    label=f"upright-approach:{object_name}",
                )
            )
            release_ok = bool(
                move_with_retry(
                    release_position,
                    target_rotation_world=upright_rotation,
                    allow_position_only_fallback=False,
                    retry_durations=[1.25],
                    label=f"upright-release:{object_name}",
                )
            ) if approach_ok else False
        else:
            approach_ok = self._move_tool_to_pose(
                approach_position,
                target_rotation_world=upright_rotation,
                allow_position_only_fallback=False,
            )
            release_ok = self._move_tool_to_pose(
                release_position,
                target_rotation_world=upright_rotation,
                allow_position_only_fallback=False,
            ) if approach_ok else False

        if not (approach_ok and release_ok):
            return False

        self.gripper_open(
            skill_description=(
                f"{skill_description} (release the upright object onto the support)"
                if skill_description
                else "release the upright object onto the support"
            )
        )
        step_sim = getattr(getattr(self._skills, "robot", None), "step_sim", None)
        if callable(step_sim):
            for _ in range(settle_steps):
                step_sim()

        retreat_position = release_position.copy()
        retreat_position[2] += max(approach_offset * 0.5, 0.04)
        self._move_tool_to_pose(
            retreat_position,
            target_rotation_world=upright_rotation,
            allow_position_only_fallback=False,
        )
        self._clear_held_state()
        return True

    def execute_pick_and_insert_into_target(
        self,
        object_name: str,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object, align it, and insert it into a named target."""

        object_entry = self._resolve_named_target_entry(object_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False

        return self._execute_insert_held_object_into_target(
            target_name,
            skill_description=skill_description,
        )

    def _execute_insert_held_object_into_target(
        self,
        target_name: str,
        *,
        skill_description: str | None = None,
    ) -> bool:
        """Insert the currently held object into a named insertion target."""

        target_entry = self._resolve_named_target_entry(target_name)
        object_name = self._held_object_name or target_name

        entry_position = np.asarray(
            target_entry.get("entry_position", target_entry["position"]),
            dtype=np.float64,
        )
        target_position = np.asarray(
            target_entry.get("target_position", target_entry["position"]),
            dtype=np.float64,
        )
        object_entry = self._resolve_named_target_entry(object_name)
        insertion_subject_offset = self._metadata_vector(object_entry, "insertion_subject_offset_local")
        if insertion_subject_offset is not None:
            object_rotation = self._entry_rotation_matrix(object_entry)
            subject_offset_world = object_rotation @ insertion_subject_offset
            entry_position = entry_position - subject_offset_world
            target_position = target_position - subject_offset_world
        insertion_axis = self._metadata_vector(target_entry, "insertion_axis_world")
        if insertion_axis is None:
            insertion_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        insertion_axis = self._normalize_vector(insertion_axis)
        target_rotation = self._metadata_rotation(target_entry)
        if target_rotation is None:
            target_rotation = self._rotation_with_tool_z(insertion_axis)
        held_xy_offset = self._held_xy_offset_world
        if held_xy_offset is not None:
            held_xy_offset = np.asarray(held_xy_offset, dtype=np.float64).reshape(-1)
            if held_xy_offset.size >= 2 and np.all(np.isfinite(held_xy_offset[:2])):
                entry_position = entry_position.copy()
                target_position = target_position.copy()
                entry_position[:2] -= held_xy_offset[:2]
                target_position[:2] -= held_xy_offset[:2]
        pre_insert = entry_position - insertion_axis * 0.05

        if not self._move_tool_to_pose(
            pre_insert,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False
        if not self._move_tool_to_pose(
            entry_position,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False

        runtime_object_name = self._resolve_runtime_object_name(object_name)
        try:
            live_obj_pos = self._current_object_position(runtime_object_name)
        except Exception:
            live_obj_pos = None
        if live_obj_pos is not None:
            for _ in range(2):
                lateral_error = np.asarray(live_obj_pos, dtype=np.float64) - target_position
                lateral_error = lateral_error - insertion_axis * float(np.dot(lateral_error, insertion_axis))
                lateral_norm = float(np.linalg.norm(lateral_error))
                if not (0.002 < lateral_norm < 0.15):
                    break
                correction = 0.85 * lateral_error
                entry_position = entry_position - correction
                target_position = target_position - correction
                if not self._move_tool_to_pose(
                    entry_position,
                    target_rotation_world=target_rotation,
                    allow_position_only_fallback=False,
                ):
                    return False
                try:
                    live_obj_pos = self._current_object_position(runtime_object_name)
                except Exception:
                    break

        if not self._move_tool_to_pose(
            target_position,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False

        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper to release the inserted object)"
                if skill_description
                else "open the gripper to release the inserted object"
            )
        )
        self._clear_held_state()
        retreat = target_position - insertion_axis * 0.03
        self._move_tool_to_pose(
            retreat,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        )
        self._refresh_related_articulated_targets(str(target_entry.get("articulation_name", "")))
        return True

    def execute_pick_and_fit_into_slot(
        self,
        object_name: str,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object and fit it into a named slot or cutout."""

        object_entry = self._resolve_named_target_entry(object_name)
        target_entry = self._resolve_named_target_entry(target_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False

        slot_position = np.asarray(
            target_entry.get("slot_position", target_entry["position"]),
            dtype=np.float64,
        )
        slot_yaw_deg = float(target_entry.get("slot_yaw_deg", 0.0))
        target_rotation = np.asarray(self._skills.PALM_DOWN_ROTATION, dtype=np.float64) @ self._axis_angle_rotation(
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.radians(slot_yaw_deg),
        )
        return self.execute_place_object(
            slot_position,
            target_name=target_name,
            is_table=True,
            skill_description=skill_description,
            target_rotation_world=target_rotation,
            required_tool_axis_world=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            allow_position_only_fallback=False,
        )

    def _drive_articulation_joint(
        self,
        *,
        articulation_name: str,
        joint_name: str,
        target_position: float,
        settle_steps: int = 60,
    ) -> bool:
        env = getattr(getattr(self._skills, "robot", None), "env", None)
        if env is None:
            return False

        try:
            articulation = env.scene[articulation_name]
        except Exception:
            articulation = getattr(env.scene, articulation_name, None)
        if articulation is None or not hasattr(articulation, "data"):
            return False

        joint_idx = self._resolve_articulation_joint_index(articulation, joint_name)
        if joint_idx is None:
            return False

        env_idx = int(getattr(getattr(self._skills, "robot", None), "env_idx", 0))
        joint_ids = articulation.data.joint_pos.new_tensor([joint_idx]).long()
        env_ids = articulation.data.joint_pos.new_tensor([env_idx]).long()
        joint_limits = articulation.data.joint_pos_limits[env_idx, joint_idx]
        target_value = float(np.clip(float(target_position), float(joint_limits[0]), float(joint_limits[1])))
        target = articulation.data.joint_pos.new_tensor([[target_value]])
        zero_vel = articulation.data.joint_vel.new_zeros((1, 1))
        try:
            strong_stiffness = articulation.data.joint_stiffness.new_tensor([[max(float(articulation.data.joint_stiffness[env_idx, joint_idx].item()), 5_000.0)]])
            strong_damping = articulation.data.joint_damping.new_tensor([[max(float(articulation.data.joint_damping[env_idx, joint_idx].item()), 250.0)]])
            strong_effort = articulation.data.joint_effort_limits.new_tensor([[max(float(articulation.data.joint_effort_limits[env_idx, joint_idx].item()), 2_000.0)]])
            articulation.write_joint_stiffness_to_sim(strong_stiffness, joint_ids=joint_ids, env_ids=env_ids)
            articulation.write_joint_damping_to_sim(strong_damping, joint_ids=joint_ids, env_ids=env_ids)
            articulation.write_joint_effort_limit_to_sim(strong_effort, joint_ids=joint_ids, env_ids=env_ids)
            if hasattr(articulation, "write_joint_friction_coefficient_to_sim"):
                articulation.write_joint_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
        except Exception:
            # Some articulated assets do not expose all writable drive parameters.
            pass

        if hasattr(articulation.data, "default_joint_pos"):
            articulation.data.default_joint_pos[env_idx, joint_idx] = target_value
        if hasattr(articulation.data, "joint_pos_target"):
            articulation.data.joint_pos_target[env_idx, joint_idx] = target_value
        if hasattr(articulation.data, "joint_vel_target"):
            articulation.data.joint_vel_target[env_idx, joint_idx] = 0.0

        articulation.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids, env_ids=env_ids)
        articulation.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
        articulation.set_joint_velocity_target(zero_vel, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_data_to_sim()

        tolerance = 0.01
        final_pos = float(articulation.data.joint_pos[env_idx, joint_idx].item())
        if abs(final_pos - target_value) <= tolerance:
            return True

        for _ in range(max(int(settle_steps), 1)):
            articulation.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
            articulation.set_joint_velocity_target(zero_vel, joint_ids=joint_ids, env_ids=env_ids)
            articulation.write_data_to_sim()
            self._skills.robot.step_sim()
            final_pos = float(articulation.data.joint_pos[env_idx, joint_idx].item())
            if abs(final_pos - target_value) <= tolerance:
                return True

        articulation.write_joint_state_to_sim(target, zero_vel, joint_ids=joint_ids, env_ids=env_ids)
        articulation.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
        articulation.set_joint_velocity_target(zero_vel, joint_ids=joint_ids, env_ids=env_ids)
        articulation.write_data_to_sim()
        final_pos = float(articulation.data.joint_pos[env_idx, joint_idx].item())
        return abs(final_pos - target_value) <= tolerance

    def _resolve_articulation_joint_index(self, articulation, joint_name: str) -> int | None:
        joint_names = list(getattr(articulation, "joint_names", []) or [])
        if joint_name in joint_names:
            return joint_names.index(joint_name)
        if hasattr(articulation, "find_joints"):
            try:
                result = articulation.find_joints(joint_name)
            except Exception:
                result = None
            if isinstance(result, tuple) and result and len(result[0]) > 0:
                return int(result[0][0])
        return None

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
        self._held_grasp_lateral_bias = None

    def _is_precise_table_target(self, target_name: str | None) -> bool:
        """Return whether the placement target needs a tighter tabletop release."""

        if not target_name:
            return False
        target_lower = str(target_name).lower()
        return any(token in target_lower for token in self._PRECISE_TABLE_TARGET_TOKENS)

    def _current_object_position(self, object_name: str) -> np.ndarray:
        """Resolve the latest center position for a named object."""

        try:
            current_pos = self._detector.get_object_position(self._resolve_runtime_object_name(object_name))
        except Exception as exc:
            raise PolicyFallbackBlockedError(
                f"unable to read live pose for '{object_name}' during placement: {exc}"
            ) from exc
        return np.asarray(current_pos, dtype=np.float64)

    def _resolve_named_target_entry(self, target_name: str) -> dict:
        """Return translated metadata for a named object or target."""

        info = self._lookup_translated_entry(target_name)
        if info is None:
            raise PolicyFallbackBlockedError(
                f"target '{target_name}' missing from translated positions"
            )
        self._refresh_dynamic_target_entry(target_name, info)
        return info

    @staticmethod
    def _is_dynamic_articulated_container(info: dict | None) -> bool:
        """Return whether the translated entry represents a dynamic articulated container target."""

        if not isinstance(info, dict):
            return False
        return (
            info.get("articulation_name") is not None
            and info.get("joint_name") is not None
            and info.get("closed_center_position") is not None
            and info.get("slide_axis_world") is not None
        )

    @staticmethod
    def _is_dynamic_articulated_handle(info: dict | None) -> bool:
        """Return whether the translated entry represents a dynamic articulated handle target."""

        if not isinstance(info, dict):
            return False
        return (
            info.get("articulation_name") is not None
            and info.get("joint_name") is not None
            and info.get("closed_position") is not None
            and info.get("slide_axis_world") is not None
            and info.get("pull_axis_world") is not None
        )

    def _read_articulation_joint_position(self, articulation_name: str, joint_name: str) -> float | None:
        """Read a named articulation joint position from the active env."""

        env = getattr(getattr(self._skills, "robot", None), "env", None)
        if env is None:
            return None

        try:
            articulation = env.scene[articulation_name]
        except Exception:
            articulation = getattr(env.scene, articulation_name, None)
        if articulation is None or not hasattr(articulation, "data"):
            return None

        joint_idx = self._resolve_articulation_joint_index(articulation, joint_name)
        if joint_idx is None:
            return None
        env_idx = int(getattr(getattr(self._skills, "robot", None), "env_idx", 0))
        return float(articulation.data.joint_pos[env_idx, joint_idx].item())

    def _refresh_related_articulated_targets(self, articulation_name: str) -> None:
        """Refresh translated entries that depend on an articulated asset state."""

        if not articulation_name:
            return
        for name, info in self._positions.items():
            if not isinstance(info, dict):
                continue
            if str(info.get("articulation_name", "")) != articulation_name:
                continue
            self._refresh_dynamic_target_entry(name, info)

    def _refresh_dynamic_target_entry(self, target_name: str, info: dict) -> None:
        """Refresh a translated target from current articulated state when applicable."""

        articulation_name = str(info.get("articulation_name", "") or "")
        joint_name = str(info.get("joint_name", "") or "")
        if not articulation_name or not joint_name:
            return

        joint_pos = self._read_articulation_joint_position(articulation_name, joint_name)
        if joint_pos is None:
            return

        if self._is_dynamic_articulated_handle(info):
            closed_position = self._metadata_vector(info, "closed_position")
            slide_axis = self._metadata_vector(info, "slide_axis_world")
            if closed_position is not None and slide_axis is not None:
                slide_axis = self._normalize_vector(slide_axis)
                close_target = float(info.get("close_target_joint_position", 0.0) or 0.0)
                delta = float(joint_pos - close_target)
                current_position = closed_position + slide_axis * delta
                info["position"] = current_position.tolist()
                info["current_joint_position"] = float(joint_pos)
            return

        if not self._is_dynamic_articulated_container(info):
            return

        closed_floor = self._metadata_vector(info, "closed_floor_position")
        closed_center = self._metadata_vector(info, "closed_center_position")
        slide_axis = self._metadata_vector(info, "slide_axis_world")
        half_extents = self._metadata_vector(info, "container_half_extents")
        if closed_floor is None or closed_center is None or slide_axis is None:
            return

        slide_axis = self._normalize_vector(slide_axis)
        close_target = float(info.get("close_target_joint_position", 0.0) or 0.0)
        delta = float(joint_pos - close_target)
        current_floor = closed_floor + slide_axis * delta
        current_center = closed_center + slide_axis * delta
        insertion_axis = self._metadata_vector(info, "insertion_axis_world")
        if insertion_axis is None:
            insertion_axis = -slide_axis
        insertion_axis = self._normalize_vector(insertion_axis)
        if half_extents is None:
            half_extents = np.zeros(3, dtype=np.float64)
        insertion_extent = max(float(np.dot(np.abs(insertion_axis), np.abs(half_extents))), 0.0)
        entry_clearance = float(
            info.get("entry_clearance", max(insertion_extent + 0.03, 0.06))
            or max(insertion_extent + 0.03, 0.06)
        )
        entry_position = current_center - insertion_axis * entry_clearance

        info["position"] = current_center.tolist()
        info["support_top_z"] = float(current_floor[2])
        info["target_position"] = current_center.tolist()
        info["entry_position"] = entry_position.tolist()
        info["insertion_axis_world"] = insertion_axis.tolist()
        info["current_joint_position"] = float(joint_pos)

    def _lookup_translated_entry(self, object_name: str | None) -> dict | None:
        """Return the translated metadata entry for a semantic or concrete object name."""

        if not object_name:
            return None
        info = self._positions.get(object_name)
        if isinstance(info, dict):
            return info
        lookup = str(object_name).strip().lower()
        for candidate in self._positions.values():
            if not isinstance(candidate, dict):
                continue
            if candidate.get("grasp_object_name") == object_name:
                return candidate
            if candidate.get("source_object_name") == object_name:
                return candidate
            aliases = candidate.get("aliases") or []
            if any(isinstance(alias, str) and alias.strip().lower() == lookup for alias in aliases):
                return candidate
            asset_label = candidate.get("asset_label")
            if isinstance(asset_label, str) and asset_label.strip().lower() == lookup:
                return candidate
        return None

    def _resolve_handle_contact_position(
        self,
        handle_name: str,
        handle_entry: dict,
        *,
        stage: str,
    ) -> np.ndarray:
        """Resolve the current handle contact point.

        Dynamic articulated handles use asset metadata plus the current joint
        state as the source of truth. Vision grounding is reserved for generic
        non-articulated handles.
        """

        if self._is_dynamic_articulated_handle(handle_entry):
            self._refresh_dynamic_target_entry(handle_name, handle_entry)
            return np.asarray(handle_entry["position"], dtype=np.float64)

        if self._target_grounder is None:
            return np.asarray(handle_entry["position"], dtype=np.float64)

        grounded = self._target_grounder.ground_target(
            handle_name,
            handle_entry,
            stage=f"handle_{stage}",
            camera_names=("front", "top", "wrist"),
        )
        if grounded is None:
            raise PolicyFallbackBlockedError(
                f"unable to visually ground handle target '{handle_name}' for {stage}"
            )
        return np.asarray(grounded["position_world"], dtype=np.float64)

    def _resolve_runtime_object_name(self, object_name: str) -> str:
        """Resolve a semantic object alias to the concrete scene object to grasp."""

        info = self._lookup_translated_entry(object_name)
        if not isinstance(info, dict):
            return object_name
        runtime_name = info.get("grasp_object_name") or info.get("source_object_name")
        return str(runtime_name) if isinstance(runtime_name, str) else object_name

    @staticmethod
    def _metadata_vector(info: dict, field: str) -> np.ndarray | None:
        """Return a vector-valued metadata field if present."""

        value = info.get(field)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            return np.asarray(value, dtype=np.float64)
        return None

    @staticmethod
    def _metadata_rotation(info: dict, field: str = "target_rotation_world") -> np.ndarray | None:
        """Return a rotation-matrix metadata field if present."""

        value = info.get(field)
        if isinstance(value, (list, tuple)) and len(value) == 3 and all(
            isinstance(row, (list, tuple)) and len(row) == 3 for row in value
        ):
            return np.asarray(value, dtype=np.float64)
        return None

    @staticmethod
    def _entry_rotation_matrix(info: dict) -> np.ndarray:
        """Return the entry quaternion as a rotation matrix when available."""

        quat = info.get("quaternion", [1.0, 0.0, 0.0, 0.0])
        quat_arr = np.asarray(quat, dtype=np.float64)
        if quat_arr.shape != (4,) or not np.all(np.isfinite(quat_arr)):
            return np.eye(3, dtype=np.float64)
        return BaseTabletopCaPSkills._quat_to_rotation_matrix(quat_arr)

    @staticmethod
    def _rotation_with_tool_z(z_axis_world: np.ndarray) -> np.ndarray:
        """Build a right-handed world rotation whose tool-Z points along ``z_axis_world``."""

        z_axis = BaseTabletopCaPSkills._normalize_vector(z_axis_world)
        up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = up_hint - z_axis * float(np.dot(up_hint, z_axis))
        if np.linalg.norm(y_axis) < 1e-6:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y_axis = up_hint - z_axis * float(np.dot(up_hint, z_axis))
        y_axis = BaseTabletopCaPSkills._normalize_vector(y_axis)
        x_axis = BaseTabletopCaPSkills._normalize_vector(np.cross(y_axis, z_axis))
        return np.column_stack((x_axis, y_axis, z_axis))

    def _move_tool_to_pose(
        self,
        target_position: np.ndarray,
        *,
        target_rotation_world: np.ndarray | None = None,
        allow_position_only_fallback: bool = True,
    ) -> bool:
        """Move to a pose when available, otherwise fall back to position-only moves."""

        target_position = np.asarray(target_position, dtype=np.float64)
        if target_rotation_world is not None and hasattr(self._skills, "move_to_pose"):
            return bool(
                self._skills.move_to_pose(
                    target_position,
                    np.asarray(target_rotation_world, dtype=np.float64),
                    allow_position_only_fallback=allow_position_only_fallback,
                )
            )
        return bool(self._skills.move_to_position(target_position))

    def _resolve_container_anchor_name(self, container_name: str) -> str:
        """Resolve a semantic tray/bin/container name to a concrete placement anchor."""

        direct_entry = self._positions.get(container_name)
        if isinstance(direct_entry, dict) and self._is_dynamic_articulated_container(direct_entry):
            return container_name

        preferred_anchor = f"{container_name}_anchor"
        if preferred_anchor in self._positions:
            return preferred_anchor
        if container_name in self._positions:
            if container_name == "tray_base" and "tray_anchor" in self._positions:
                return "tray_anchor"
            return container_name

        lowered = str(container_name).lower()
        if "tray" in lowered and "tray_anchor" in self._positions:
            return "tray_anchor"

        for name, info in self._positions.items():
            if info.get("task_role") != "placement_target":
                continue
            name_lower = str(name).lower()
            if any(token in name_lower for token in ("tray", "bin", "container", "collection")):
                return name

        raise PolicyFallbackBlockedError(
            f"container target '{container_name}' missing from translated positions"
        )

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

        clip_limit = 0.04
        info = self._lookup_translated_entry(object_name) or {}
        extents = info.get("estimated_half_extents")
        if isinstance(extents, (list, tuple)) and len(extents) == 3:
            try:
                planar_extent = max(abs(float(extents[0])), abs(float(extents[1])))
                clip_limit = max(clip_limit, min(planar_extent * 3.0, 0.12))
            except Exception:
                pass
        goal_roles = info.get("goal_roles") or []
        if any(str(role).endswith(":inserted_into") or str(role).endswith(":height_below") for role in goal_roles):
            clip_limit = max(clip_limit, 0.12)
        return np.clip(xy_offset, -clip_limit, clip_limit)

    def _resolve_half_height(self, object_name: str | None) -> float:
        """Resolve the best available half-height estimate for an object."""

        if not object_name:
            raise PolicyFallbackBlockedError(
                "half-height lookup requires explicit object_name"
            )

        runtime_name = self._resolve_runtime_object_name(object_name)
        info = self._lookup_translated_entry(object_name) or self._lookup_translated_entry(runtime_name)
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

    def _resolve_upright_vertical_extent(self, object_entry: dict) -> float:
        """Return the object's support-to-center distance for upright placement."""

        extents = object_entry.get("estimated_half_extents")
        axis_local = object_entry.get("upright_axis_local", [1.0, 0.0, 0.0])
        if not (
            isinstance(extents, (list, tuple))
            and len(extents) == 3
            and isinstance(axis_local, (list, tuple))
            and len(axis_local) == 3
        ):
            return max(self._held_half_height or 0.02, 0.02)
        half_extents = np.asarray(extents, dtype=np.float64)
        axis = np.asarray(axis_local, dtype=np.float64)
        axis = np.abs(axis) / max(np.linalg.norm(axis), 1e-9)
        vertical_extent = max(float(np.dot(axis, half_extents)), 0.02)
        return vertical_extent

    def _resolve_upright_grasp_offset_local(self, object_entry: dict) -> np.ndarray:
        """Return the local grasp offset used for upright placement."""

        offset = object_entry.get("upright_grasp_offset_local")
        if isinstance(offset, (list, tuple)) and len(offset) == 3:
            return np.asarray(offset, dtype=np.float64)
        axis_local = np.asarray(object_entry.get("upright_axis_local", [1.0, 0.0, 0.0]), dtype=np.float64)
        axis_local = self._normalize_vector(axis_local)
        vertical_extent = self._resolve_upright_vertical_extent(object_entry)
        grasp_distance = min(max(vertical_extent * 0.60, 0.04), max(vertical_extent - 0.03, 0.04))
        return axis_local * grasp_distance

    def _resolve_upright_grasp_offset_candidates_local(self, object_entry: dict) -> list[np.ndarray]:
        """Return candidate local grasp offsets for upright placement."""

        candidates = object_entry.get("upright_grasp_offset_candidates_local")
        resolved: list[np.ndarray] = []
        if isinstance(candidates, (list, tuple)):
            for candidate in candidates:
                if isinstance(candidate, (list, tuple)) and len(candidate) == 3:
                    resolved.append(np.asarray(candidate, dtype=np.float64))
        if resolved:
            return resolved
        return [self._resolve_upright_grasp_offset_local(object_entry)]

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

        info = self._lookup_translated_entry(object_name)
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
