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
    _PRECISE_TABLE_TARGET_TOKENS = ("marker", "zone", "anchor", "command_pose")

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
        del object_position
        if not object_name:
            raise PolicyFallbackBlockedError(
                "execute_pick_object requires explicit object_name; runtime inference is disabled"
            )
        runtime_object_name = self._resolve_runtime_object_name(object_name)

        pick_ok = self._skills.execute_pick(
            runtime_object_name,
            approach_offset=self.PICK_APPROACH_OFFSET,
            approach_angle_deg=approach_angle_deg,
            target_rotation_world=target_rotation_world,
            approach_direction_world=approach_direction_world,
            required_tool_axis_world=required_tool_axis_world,
            allow_position_only_fallback=allow_position_only_fallback,
            skill_description=skill_description,
        )
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
        approach_offset = self.PLACE_APPROACH_OFFSET
        drop_offset = self.PLACE_DROP_OFFSET
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
        handle_pos = np.asarray(handle_entry["position"], dtype=np.float64)
        pull_axis = self._metadata_vector(handle_entry, "pull_axis_world")
        if pull_axis is None:
            raise PolicyFallbackBlockedError(
                f"handle target '{handle_name}' is missing pull_axis_world metadata"
            )
        pull_axis = self._normalize_vector(pull_axis)
        pull_distance = float(handle_entry.get("pull_distance", 0.25))
        if open_fraction is not None:
            pull_distance *= float(np.clip(open_fraction, 0.0, 1.0))
        target_rotation = self._metadata_rotation(handle_entry)

        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper before grasping {handle_name})"
                if skill_description
                else f"open the gripper before grasping {handle_name}"
            )
        )

        approach_pos = handle_pos + pull_axis * 0.06
        if not self._move_tool_to_pose(
            approach_pos,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False
        if not self._move_tool_to_pose(
            handle_pos,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False

        self.gripper_close(
            skill_description=(
                f"{skill_description} (close the gripper to grasp {handle_name})"
                if skill_description
                else f"close the gripper to grasp {handle_name}"
            )
        )

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
        return bool(pulled)

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
        target_position = np.asarray(support_entry["position"], dtype=np.float64).copy()
        if support_name == "table_surface":
            target_position[:2] = current_position[:2]
        upright_rotation = self._metadata_rotation(object_entry, field="upright_target_rotation_world")
        if upright_rotation is None:
            palm_down = np.asarray(self._skills.PALM_DOWN_ROTATION, dtype=np.float64)
            upright_rotation = palm_down @ self._axis_angle_rotation(np.array([0.0, 1.0, 0.0]), -np.pi / 2.0)

        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False

        return self.execute_place_object(
            target_position,
            target_name=support_name,
            is_table=True,
            target_rotation_world=upright_rotation,
            allow_position_only_fallback=False,
            skill_description=skill_description,
        )

    def execute_pick_and_insert_into_target(
        self,
        object_name: str,
        target_name: str,
        skill_description: str | None = None,
    ) -> bool:
        """Pick a named object, align it, and insert it into a named target."""

        object_entry = self._resolve_named_target_entry(object_name)
        target_entry = self._resolve_named_target_entry(target_name)
        picked = self.execute_pick_object(
            object_entry["position"],
            object_name=object_name,
            skill_description=skill_description,
        )
        if not picked:
            return False

        entry_position = np.asarray(
            target_entry.get("entry_position", target_entry["position"]),
            dtype=np.float64,
        )
        target_position = np.asarray(
            target_entry.get("target_position", target_entry["position"]),
            dtype=np.float64,
        )
        insertion_axis = self._metadata_vector(target_entry, "insertion_axis_world")
        if insertion_axis is None:
            insertion_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        insertion_axis = self._normalize_vector(insertion_axis)
        target_rotation = self._metadata_rotation(target_entry)
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
        pre_slot = slot_position.copy()
        pre_slot[2] += 0.05
        if not self._move_tool_to_pose(
            pre_slot,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False
        if not self._move_tool_to_pose(
            slot_position,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        ):
            return False

        self.gripper_open(
            skill_description=(
                f"{skill_description} (open the gripper to place the object into the slot)"
                if skill_description
                else "open the gripper to place the object into the slot"
            )
        )
        self._clear_held_state()
        self._move_tool_to_pose(
            pre_slot,
            target_rotation_world=target_rotation,
            allow_position_only_fallback=False,
        )
        return True

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
        return info

    def _lookup_translated_entry(self, object_name: str | None) -> dict | None:
        """Return the translated metadata entry for a semantic or concrete object name."""

        if not object_name:
            return None
        info = self._positions.get(object_name)
        if isinstance(info, dict):
            return info
        for candidate in self._positions.values():
            if not isinstance(candidate, dict):
                continue
            if candidate.get("grasp_object_name") == object_name:
                return candidate
            if candidate.get("source_object_name") == object_name:
                return candidate
        return None

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
        return np.clip(xy_offset, -0.04, 0.04)

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
