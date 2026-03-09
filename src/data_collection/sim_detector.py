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
            key = self._resolve_entity_key(asset_name, meta, all_keys)
            if key:
                self._entity_map[asset_name] = key
            else:
                logger.warning(f"Could not map asset '{asset_name}' to any scene entity. "
                               f"Available: {all_keys}")

    def _resolve_entity_key(self, asset_name: str, meta: dict, candidates: set[str]) -> Optional[str]:
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
        non_robot_candidates = {
            key for key in candidates if key.lower() not in {"robot", "franka", "openarm", "so101", "ur10e"}
        }
        if len(non_robot_candidates) == 1:
            return next(iter(non_robot_candidates))

        return None

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
        entity_key = self._entity_map.get(name)
        if entity_key is None:
            raise KeyError(
                f"Object '{name}' not mapped to any scene entity. "
                f"Mapped objects: {list(self._entity_map.keys())}"
            )

        scene = self._env.scene
        idx = self._env_idx

        # Try rigid objects first, then articulations
        if hasattr(scene, "rigid_objects") and entity_key in scene.rigid_objects:
            obj = scene[entity_key]
            pos = obj.data.root_pos_w[idx].cpu().numpy()
            quat = obj.data.root_quat_w[idx].cpu().numpy()
            return pos, quat

        if hasattr(scene, "articulations") and entity_key in scene.articulations:
            obj = scene[entity_key]
            pos = obj.data.root_pos_w[idx].cpu().numpy()
            quat = obj.data.root_quat_w[idx].cpu().numpy()
            return pos, quat

        # Fallback: try direct scene access
        try:
            obj = scene[entity_key]
            pos = obj.data.root_pos_w[idx].cpu().numpy()
            quat = obj.data.root_quat_w[idx].cpu().numpy()
            return pos, quat
        except Exception as e:
            raise KeyError(
                f"Failed to query entity '{entity_key}' for object '{name}': {e}"
            )

    def get_all_objects(self) -> dict[str, dict]:
        """Get positions and quaternions for all mapped non-robot objects.

        Returns:
            Dict mapping asset name -> {"position": [x,y,z], "quaternion": [w,x,y,z]}.
        """
        result = {}
        for name in self._entity_map:
            # Skip robot articulation
            if name == self._robot_name:
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
