"""
Scene Graph object position query module for IsaacLab environments.

Replaces camera-based Grounding DINO detection with direct scene graph queries.
Faster, more accurate, and requires no vision model dependencies.

Runs INSIDE the IsaacLab conda subprocess where torch and isaaclab are available.
"""

from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _normalize_name(value: str) -> str:
    """Return a compact normalized identifier for fuzzy stage matching."""

    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


class SimDetector:
    """Query object positions directly from IsaacLab scene graph.

    Args:
        env: IsaacLab ManagerBasedRLEnv (or ManagerBasedEnv) instance.
        task_doc: Parsed task YAML document dict.
        env_idx: Environment index to query (default 0 for single-env).
    """

    def __init__(self, env, task_doc: dict, env_idx: int = 0):
        self._env = env
        self._task_doc = task_doc
        self._env_idx = env_idx

        # Parse asset metadata from task YAML
        self._asset_meta: dict[str, dict] = {}
        for asset in task_doc.get("assets", []):
            self._asset_meta[asset["name"]] = asset

        # Build name -> scene entity key mapping
        self._entity_map: dict[str, str] = {}
        self._robot_name: Optional[str] = None
        self._ee_frame_body: Optional[str] = None
        self._ee_offset: Optional[list[float]] = None

        self._build_entity_map()

    def _build_entity_map(self) -> None:
        """Map asset names from task YAML to IsaacLab scene entity keys."""
        scene = self._env.scene

        # Collect available scene entity keys
        available_rigid = set()
        available_artic = set()

        if hasattr(scene, "rigid_objects"):
            for key in scene.rigid_objects:
                available_rigid.add(key)
        if hasattr(scene, "articulations"):
            for key in scene.articulations:
                available_artic.add(key)

        all_keys = available_rigid | available_artic
        logger.info(f"Scene entities: rigid={available_rigid}, articulations={available_artic}")

        for asset_name, meta in self._asset_meta.items():
            asset_type = meta.get("type", "")

            if asset_type == "articulation":
                # This is the robot
                self._robot_name = asset_name
                self._ee_frame_body = meta.get("ee_frame", {}).get("body")
                self._ee_offset = meta.get("ee_frame", {}).get("offset_position")

                # Try to find the robot in scene articulations
                key = self._resolve_entity_key(asset_name, meta, available_artic)
                if key:
                    self._entity_map[asset_name] = key
                continue

            # Rigid or static objects
            prefer_stage_pose = self._should_use_stage_pose_mapping(asset_name, meta)
            key = self._resolve_entity_key(
                asset_name,
                meta,
                all_keys,
                allow_singleton_fallback=not prefer_stage_pose,
            )
            if key:
                self._entity_map[asset_name] = key
            else:
                if prefer_stage_pose:
                    logger.info(
                        "Will resolve asset '%s' via stage pose instead of scene entity mapping",
                        asset_name,
                    )
                    continue
                logger.warning(f"Could not map asset '{asset_name}' to any scene entity. "
                               f"Available: {all_keys}")

    def _resolve_entity_key(
        self,
        asset_name: str,
        meta: dict,
        candidates: set[str],
        *,
        allow_singleton_fallback: bool = True,
    ) -> Optional[str]:
        """Resolve a YAML asset name to an IsaacLab scene entity key.

        Matching strategy (in priority order):
        1. Exact match
        2. Case-insensitive match
        3. Substring match (asset_name contained in key or vice versa)
        4. Prim path basename match
        5. Normalized match (underscores/hyphens removed)
        6. Sole non-robot object fallback for singleton scenes
        """
        if not candidates:
            return None

        # 1. Exact match
        if asset_name in candidates:
            return asset_name

        # 2. Case-insensitive
        name_lower = asset_name.lower()
        for key in candidates:
            if key.lower() == name_lower:
                return key

        # 3. Substring match
        for key in candidates:
            if name_lower in key.lower() or key.lower() in name_lower:
                return key

        prim_path = str((meta or {}).get("prim_path", "") or "").strip()
        prim_basename = ""
        if prim_path:
            prim_basename = PurePosixPath(prim_path).name
            if prim_basename:
                prim_lower = prim_basename.lower()
                for key in candidates:
                    if key.lower() == prim_lower:
                        return key
                for key in candidates:
                    if prim_lower in key.lower() or key.lower() in prim_lower:
                        return key

        # 5. Normalized (strip underscores/hyphens)
        def normalize(s: str) -> str:
            return s.lower().replace("_", "").replace("-", "")

        norm_name = normalize(asset_name)
        for key in candidates:
            if normalize(key) == norm_name:
                return key
        if prim_basename:
            norm_prim = normalize(prim_basename)
            for key in candidates:
                if normalize(key) == norm_prim:
                    return key

        # 6. Final fallback: if there is only one non-robot candidate, assume it.
        if allow_singleton_fallback:
            non_robot_candidates = {
                key for key in candidates if key.lower() not in {"robot", "franka", "openarm", "so101", "ur10e"}
            }
            if len(non_robot_candidates) == 1:
                return next(iter(non_robot_candidates))

        return None

    @staticmethod
    def _should_use_stage_pose_mapping(asset_name: str, meta: dict) -> bool:
        """Return whether a target should bypass loose scene-entity fallback.

        Static or visual-only placement targets often do not appear as IsaacLab
        rigid entities after env sanitization. In those cases, singleton
        fallback can incorrectly alias them to the only movable object in the
        scene, causing false-positive goal verification. Prefer stage-pose
        lookup for these assets instead.
        """

        asset_type = str(meta.get("type", "")).lower()
        physics = meta.get("physics", {}) if isinstance(meta.get("physics"), dict) else {}
        name_lower = str(asset_name or "").lower()

        if asset_type == "static":
            return True
        if physics.get("rigid_body") is False or physics.get("collision") is False:
            return True
        if any(
            token in name_lower
            for token in (
                "marker",
                "zone",
                "tray",
                "bin",
                "anchor",
                "target",
                "command_pose",
                "handle",
                "hole",
                "slot",
                "cutout",
                "receptacle",
            )
        ):
            return True
        return False

    def get_object_position(self, name: str) -> np.ndarray:
        """Get world-frame position [x, y, z] for a named object.

        Args:
            name: Asset name as defined in task YAML.

        Returns:
            Position array of shape (3,).

        Raises:
            KeyError: If object name not found in scene.
        """
        pos, _ = self.get_object_pose(name)
        return pos

    def get_object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get world-frame pose for a named object.

        Args:
            name: Asset name as defined in task YAML.

        Returns:
            Tuple of (position[3], quaternion_wxyz[4]).

        Raises:
            KeyError: If object name not found in scene.
        """
        prefer_stage_pose = self._should_prefer_stage_pose(name)

        scene_pose = self._get_scene_entity_pose(name)
        if scene_pose is not None:
            if not prefer_stage_pose or self._is_pose_plausible(scene_pose[0]):
                return scene_pose

        if prefer_stage_pose:
            stage_pose = self._get_stage_prim_pose(name)
            if stage_pose is not None:
                return stage_pose
            if scene_pose is not None:
                return scene_pose

        entity_key = self._entity_map.get(name)
        if entity_key is None:
            stage_pose = self._get_stage_prim_pose(name)
            if stage_pose is not None:
                return stage_pose
            raise KeyError(
                f"Object '{name}' not mapped to any scene entity. "
                f"Mapped objects: {list(self._entity_map.keys())}"
            )
        if scene_pose is not None:
            return scene_pose

        stage_pose = self._get_stage_prim_pose(name)
        if stage_pose is not None:
            return stage_pose
        raise KeyError(
            f"Failed to query entity '{entity_key}' for object '{name}'"
        )

    def _get_scene_entity_pose(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the pose from the mapped IsaacLab scene entity when available."""

        entity_key = self._entity_map.get(name)
        if entity_key is None:
            return None

        scene = self._env.scene
        idx = self._env_idx

        def _extract_pose(obj) -> tuple[np.ndarray, np.ndarray] | None:
            try:
                pos = obj.data.root_pos_w[idx].cpu().numpy()
                quat = obj.data.root_quat_w[idx].cpu().numpy()
            except Exception:
                return None
            return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)

        if hasattr(scene, "rigid_objects") and entity_key in scene.rigid_objects:
            pose = _extract_pose(scene[entity_key])
            if pose is not None:
                return pose

        if hasattr(scene, "articulations") and entity_key in scene.articulations:
            pose = _extract_pose(scene[entity_key])
            if pose is not None:
                return pose

        try:
            return _extract_pose(scene[entity_key])
        except Exception:
            return None

    @staticmethod
    def _is_pose_plausible(pos: np.ndarray) -> bool:
        """Return whether a scene pose looks usable for live execution logic."""

        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if pos.size < 3 or not np.all(np.isfinite(pos[:3])):
            return False
        if np.linalg.norm(pos[:3]) > 10.0:
            return False
        if pos[2] < -0.5:
            return False
        return True

    def _should_prefer_stage_pose(self, name: str) -> bool:
        meta = self._asset_meta.get(name) or {}
        asset_path = str(meta.get("asset_path", "") or "").lower()
        return any(
            token in asset_path
            for token in (
                "factory_gear_",
                "m16_nut",
                "assets/assembling_kits/shape_",
            )
        )

    def _get_stage_prim_pose(self, name: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Resolve a pose directly from the USD stage when scene-entity mapping is unavailable."""

        meta = self._asset_meta.get(name)
        if not isinstance(meta, dict):
            return None

        stage = getattr(getattr(self._env, "sim", None), "stage", None)
        if stage is None:
            return None

        try:
            from pxr import Gf, UsdGeom
        except Exception:
            return None

        xform_cache = UsdGeom.XformCache()
        for prim in self._iter_stage_root_candidates(stage, name, meta):
            for candidate in self._iter_pose_candidate_prims(prim):
                try:
                    matrix = xform_cache.GetLocalToWorldTransform(candidate)
                    translation = matrix.ExtractTranslation()
                    rotation = matrix.ExtractRotationQuat()
                except Exception:
                    continue
                pos = np.array(
                    [float(translation[0]), float(translation[1]), float(translation[2])],
                    dtype=np.float64,
                )
                quat = np.array(
                    [
                        float(rotation.GetReal()),
                        float(rotation.GetImaginary()[0]),
                        float(rotation.GetImaginary()[1]),
                        float(rotation.GetImaginary()[2]),
                    ],
                    dtype=np.float64,
                )
                norm = np.linalg.norm(quat)
                if norm > 1e-9:
                    quat = quat / norm
                logger.info(
                    "Resolved stage pose for %s from prim %s (candidate=%s)",
                    name,
                    prim.GetPath(),
                    candidate.GetPath(),
                )
                return pos, quat
        return None

    def _iter_stage_root_candidates(self, stage, name: str, meta: dict):
        """Yield likely root prims for stage-pose lookup.

        Exact prim-path guesses are tried first, then a stage-wide basename
        search restricted to the active env namespace when available.
        """

        yielded: set[str] = set()

        def _yield(prim):
            if prim is None or not prim.IsValid():
                return
            path = str(prim.GetPath())
            if path in yielded:
                return
            yielded.add(path)
            yield prim

        for prim_path in self._candidate_stage_prim_paths(meta):
            prim = stage.GetPrimAtPath(prim_path)
            for yielded_prim in _yield(prim):
                yield yielded_prim

        search_names = {str(name or "").strip()}
        prim_path = str((meta or {}).get("prim_path", "") or "").strip()
        if prim_path:
            search_names.add(PurePosixPath(prim_path).name)
        asset_path = str((meta or {}).get("asset_path", "") or "").strip()
        if asset_path:
            search_names.add(PurePosixPath(asset_path).stem)
        search_names = {candidate for candidate in search_names if candidate}
        search_names_lower = {candidate.lower() for candidate in search_names}
        search_names_normalized = {_normalize_name(candidate) for candidate in search_names}

        env_prim_paths = getattr(self._env.scene, "env_prim_paths", None)
        env_prefix: str | None = None
        if isinstance(env_prim_paths, (list, tuple)) and len(env_prim_paths) > self._env_idx:
            env_prefix = str(env_prim_paths[self._env_idx])

        traverse = stage.TraverseAll() if hasattr(stage, "TraverseAll") else stage.Traverse()
        for prim in traverse:
            if not prim or not prim.IsValid():
                continue
            path = str(prim.GetPath())
            if env_prefix and not path.startswith(env_prefix):
                continue
            basename = prim.GetName()
            if not basename:
                continue
            if (
                basename in search_names
                or basename.lower() in search_names_lower
                or _normalize_name(basename) in search_names_normalized
            ):
                for yielded_prim in _yield(prim):
                    yield yielded_prim

    def _iter_pose_candidate_prims(self, root_prim):
        """Yield likely prims whose world transform best represents the visible object pose."""

        yielded: set[str] = set()

        def _yield(prim):
            path = str(prim.GetPath())
            if path in yielded:
                return
            yielded.add(path)
            yield prim

        for prim in _yield(root_prim):
            yield prim

        preferred = []
        meshes = []
        stack = list(root_prim.GetChildren())
        while stack:
            prim = stack.pop(0)
            stack.extend(list(prim.GetChildren()))
            if not prim.IsValid():
                continue
            name = prim.GetName().lower()
            type_name = prim.GetTypeName()
            if name == "world" or name.endswith("_gear") or "visual" in name:
                preferred.append(prim)
            if type_name == "Mesh":
                meshes.append(prim)

        for prim in preferred + meshes:
            for yielded_prim in _yield(prim):
                yield yielded_prim

    def _candidate_stage_prim_paths(self, meta: dict) -> list[str]:
        """Return likely stage prim paths for a task asset."""

        prim_path = str((meta or {}).get("prim_path", "") or "").strip()
        candidates: list[str] = []
        if prim_path:
            candidates.append(prim_path)

        env_prim_paths = getattr(self._env.scene, "env_prim_paths", None)
        env_prefix: str | None = None
        if isinstance(env_prim_paths, (list, tuple)) and len(env_prim_paths) > self._env_idx:
            env_prefix = str(env_prim_paths[self._env_idx])
        if env_prefix and prim_path:
            basename = PurePosixPath(prim_path).name
            namespaced = f"{env_prefix}/{basename}"
            if namespaced not in candidates:
                candidates.append(namespaced)
            asset_name = str((meta or {}).get("name", "") or "").strip()
            if asset_name:
                for variant in (asset_name, asset_name.replace("_", "").capitalize()):
                    alt = f"{env_prefix}/{variant}"
                    if alt not in candidates:
                        candidates.append(alt)

        return candidates

    def get_all_objects(self) -> dict[str, dict]:
        """Get positions and quaternions for all mapped non-robot objects.

        Returns:
            Dict mapping asset name -> {"position": [x,y,z], "quaternion": [w,x,y,z]}.
        """
        result = {}
        for name, meta in self._asset_meta.items():
            # Skip robot articulation
            if name == self._robot_name or str(meta.get("type", "")).lower() == "articulation":
                continue
            try:
                pos, quat = self.get_object_pose(name)
                result[name] = {
                    "position": pos.tolist(),
                    "quaternion": quat.tolist(),
                }
            except KeyError as e:
                logger.warning(f"Skipping object '{name}': {e}")
        return result

    def get_robot_ee_position(self) -> np.ndarray:
        """Get world-frame end-effector position [x, y, z].

        Resolution order:
        1. FrameTransformer "ee_frame" in scene
        2. Articulation body position + ee_frame offset from task YAML

        Returns:
            EE position array of shape (3,).

        Raises:
            RuntimeError: If EE position cannot be determined.
        """
        scene = self._env.scene
        idx = self._env_idx

        # Method 1: FrameTransformer (standard IsaacLab pattern)
        try:
            ee_frame = scene["ee_frame"]
            # FrameTransformer stores target frame positions
            pos = ee_frame.data.target_pos_w[idx, 0].cpu().numpy()
            return pos
        except (KeyError, AttributeError, IndexError):
            pass

        # Method 2: Articulation body pose + offset
        if self._robot_name and self._robot_name in self._entity_map:
            robot_key = self._entity_map[self._robot_name]
            try:
                robot = scene[robot_key]
                if self._ee_frame_body:
                    body_idx = robot.find_bodies(self._ee_frame_body)
                    if body_idx is not None:
                        # find_bodies returns (indices, names)
                        if isinstance(body_idx, tuple):
                            body_indices = body_idx[0]
                        else:
                            body_indices = body_idx

                        pos = robot.data.body_pos_w[idx, body_indices[0]].cpu().numpy()

                        # Apply TCP offset if defined
                        if self._ee_offset:
                            offset = np.array(self._ee_offset, dtype=np.float64)
                            # Simple translation offset (ignores rotation for now)
                            pos = pos + offset

                        return pos
            except Exception as e:
                logger.warning(f"Failed to get EE from articulation body: {e}")

        # Method 3: Fall back to robot root position (very rough)
        if self._robot_name and self._robot_name in self._entity_map:
            logger.warning("Falling back to robot root position as EE estimate")
            pos, _ = self.get_object_pose(self._robot_name)
            return pos

        raise RuntimeError(
            "Cannot determine EE position: no ee_frame in scene and "
            "no robot articulation body available."
        )

    def get_scene_description(self) -> str:
        """Generate natural language scene description for LLM consumption.

        Returns:
            Multi-line string describing all objects and robot EE.
        """
        lines = ["Scene contains:"]

        for name, meta in self._asset_meta.items():
            asset_type = meta.get("type", "unknown")

            if asset_type == "articulation":
                continue  # Described separately as robot

            # Build description parts
            parts = [asset_type]

            props = meta.get("properties", {})
            if props.get("color"):
                color = props["color"]
                color_name = self._color_to_name(color)
                if color_name:
                    parts.append(color_name)

            # Detect shape from primitive or asset path
            primitive = meta.get("primitive")
            if primitive:
                parts.append(primitive)
            else:
                asset_path = meta.get("asset_path", "")
                for shape in ("cube", "block", "sphere", "cylinder", "cone", "mug", "can", "bowl"):
                    if shape in asset_path.lower() or shape in name.lower():
                        parts.append(shape)
                        break

            # Size info
            size = props.get("size")
            if size and isinstance(size, list) and len(size) >= 1:
                dim_cm = size[0] * 100  # meters to cm
                parts.append(f"{dim_cm:.0f}cm")

            desc = ", ".join(parts)

            # Position
            try:
                pos = self.get_object_position(name)
                pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
            except KeyError:
                # Use YAML default position
                yaml_pos = meta.get("position", [0, 0, 0])
                pos_str = f"[{yaml_pos[0]:.3f}, {yaml_pos[1]:.3f}, {yaml_pos[2]:.3f}] (initial)"

            lines.append(f"  - {name} ({desc}): position {pos_str}")

        # Robot info
        robot_meta = self._asset_meta.get(self._robot_name, {}) if self._robot_name else {}
        robot_type = robot_meta.get("robot_type", "unknown")

        try:
            ee_pos = self.get_robot_ee_position()
            ee_str = f"[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
        except RuntimeError:
            ee_str = "[unavailable]"

        lines.append(f"Robot: {robot_type}, EE at {ee_str}")

        return "\n".join(lines)

    @staticmethod
    def _color_to_name(color: list[float]) -> Optional[str]:
        """Map RGB [0-1] color to approximate name."""
        if len(color) < 3:
            return None

        r, g, b = color[0], color[1], color[2]
        threshold = 0.5

        if r > threshold and g < threshold and b < threshold:
            return "red"
        if r < threshold and g > threshold and b < threshold:
            return "green"
        if r < threshold and g < threshold and b > threshold:
            return "blue"
        if r > threshold and g > threshold and b < threshold:
            return "yellow"
        if r > threshold and g > threshold and b > threshold:
            return "white"
        if r < 0.2 and g < 0.2 and b < 0.2:
            return "black"
        if r > threshold and g < threshold and b > threshold:
            return "purple"
        if r > threshold and g > 0.3 and b < threshold:
            return "orange"

        return None
