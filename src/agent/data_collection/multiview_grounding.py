"""Multi-view RGB-D target grounding for articulated handle interactions.

This module uses simulator cameras plus depth reprojection to refine a named
target's world-space position from multiple viewpoints. It is intentionally
simple: the translated target position acts as a prior, each camera proposes a
nearby RGB-D point, and the valid proposals are fused with view-dependent
weights after outlier rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class _ViewObservation:
    camera_name: str
    position_world: np.ndarray | None
    pixel_coords: tuple[int, int] | None
    confidence: float
    status: str
    depth_m: float | None = None
    prior_pixel_coords: tuple[float, float] | None = None
    prior_distance_m: float | None = None


class MultiViewTargetGrounder:
    """Fuse `front`/`top`/`wrist` RGB-D evidence for a named target.

    The service is designed for simulator use. It does not require a learned
    detector; instead it refines a semantic target's translated prior using the
    available depth maps from multiple cameras.
    """

    DEFAULT_CAMERA_ORDER = ("front", "top", "wrist")
    BASE_VIEW_WEIGHTS = {
        "front": 1.0,
        "top": 0.7,
        "wrist": 0.85,
    }

    def __init__(
        self,
        cameras: Any,
        robot_interface: Any,
        robot_cfg: Any | None = None,
        *,
        search_radius_px: int = 18,
        outlier_threshold_m: float = 0.08,
    ) -> None:
        self._cameras = cameras
        self._robot = robot_interface
        self._robot_cfg = robot_cfg
        self._search_radius_px = int(search_radius_px)
        self._outlier_threshold_m = float(outlier_threshold_m)
        self._events: list[dict[str, Any]] = []
        self._episode_idx: int | None = None

    def begin_episode(self, episode_idx: int) -> None:
        self._episode_idx = int(episode_idx)
        self._events = []

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def ground_target(
        self,
        target_name: str,
        target_entry: dict[str, Any],
        *,
        query: str | None = None,
        stage: str,
        camera_names: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, Any] | None:
        if self._cameras is None:
            return None

        prior_world = np.asarray(target_entry.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
        if prior_world.shape != (3,) or not np.all(np.isfinite(prior_world)):
            return None

        camera_order = tuple(camera_names or self.DEFAULT_CAMERA_ORDER)
        observations: list[_ViewObservation] = []
        for camera_name in camera_order:
            observations.append(self._ground_single_view(camera_name, prior_world))

        valid = [obs for obs in observations if obs.position_world is not None]
        if not valid:
            self._record_event(
                target_name=target_name,
                query=query or self._build_query(target_name, target_entry),
                stage=stage,
                prior_world=prior_world,
                observations=observations,
                fused=None,
                status="no_valid_views",
            )
            return None

        fused_world, filtered = self._fuse_observations(valid)
        target_rotation = self._target_rotation_quat(target_entry)
        robot_pos, robot_quat = self._robot.world_pose_to_robot_pose(fused_world, target_rotation)

        fused = {
            "target_name": target_name,
            "query": query or self._build_query(target_name, target_entry),
            "position_world": fused_world.tolist(),
            "position_robot": robot_pos.tolist(),
            "orientation_robot_wxyz": robot_quat.tolist(),
            "contributing_views": [obs.camera_name for obs in filtered],
        }
        self._record_event(
            target_name=target_name,
            query=fused["query"],
            stage=stage,
            prior_world=prior_world,
            observations=observations,
            fused=fused,
            status="ok",
        )
        return fused

    def _ground_single_view(self, camera_name: str, prior_world: np.ndarray) -> _ViewObservation:
        rgb, depth = self._cameras.capture_rgb_depth(camera_name)
        if rgb is None or depth is None:
            return _ViewObservation(camera_name, None, None, 0.0, "capture_failed")

        try:
            intrinsics = self._cameras.get_intrinsics(camera_name)
            transform_world = self._cameras.get_camera_transform_world(camera_name)
        except Exception:
            return _ViewObservation(camera_name, None, None, 0.0, "camera_metadata_failed")

        prior_pixel = self._project_world_to_pixel(prior_world, intrinsics, transform_world)
        if prior_pixel is None:
            return _ViewObservation(camera_name, None, None, 0.0, "prior_out_of_view")

        u_prior, v_prior, _ = prior_pixel
        candidate = self._search_depth_candidate(
            prior_world=prior_world,
            depth=depth,
            intrinsics=intrinsics,
            transform_world=transform_world,
            center_u=u_prior,
            center_v=v_prior,
        )
        if candidate is None:
            return _ViewObservation(
                camera_name,
                None,
                None,
                0.0,
                "no_depth_candidate",
                prior_pixel_coords=(float(u_prior), float(v_prior)),
            )

        position_world, depth_m, pixel_coords, confidence = candidate
        return _ViewObservation(
            camera_name=camera_name,
            position_world=position_world,
            pixel_coords=pixel_coords,
            confidence=confidence,
            status="ok",
            depth_m=depth_m,
            prior_pixel_coords=(float(u_prior), float(v_prior)),
            prior_distance_m=float(np.linalg.norm(position_world - prior_world)),
        )

    def _search_depth_candidate(
        self,
        *,
        prior_world: np.ndarray,
        depth: np.ndarray,
        intrinsics: dict[str, float],
        transform_world: np.ndarray,
        center_u: float,
        center_v: float,
    ) -> tuple[np.ndarray, float, tuple[int, int], float] | None:
        height, width = depth.shape[:2]
        radius = self._search_radius_px
        u_min = max(int(round(center_u)) - radius, 0)
        u_max = min(int(round(center_u)) + radius, width - 1)
        v_min = max(int(round(center_v)) - radius, 0)
        v_max = min(int(round(center_v)) + radius, height - 1)
        if u_min > u_max or v_min > v_max:
            return None

        best: tuple[np.ndarray, float, tuple[int, int], float] | None = None
        best_score = float("inf")
        for v in range(v_min, v_max + 1):
            for u in range(u_min, u_max + 1):
                depth_m = float(depth[v, u])
                if not np.isfinite(depth_m) or depth_m <= 0.02 or depth_m > 5.0:
                    continue
                point_world = self._deproject_pixel_to_world(float(u), float(v), depth_m, intrinsics, transform_world)
                prior_distance = float(np.linalg.norm(point_world - prior_world))
                pixel_distance = float(np.hypot(float(u) - center_u, float(v) - center_v))
                score = prior_distance + 0.002 * pixel_distance
                if score >= best_score:
                    continue
                confidence = max(1e-6, 1.0 / (1.0 + prior_distance * 20.0 + pixel_distance * 0.1))
                best = (point_world, depth_m, (int(u), int(v)), confidence)
                best_score = score
        return best

    def _fuse_observations(
        self,
        observations: list[_ViewObservation],
    ) -> tuple[np.ndarray, list[_ViewObservation]]:
        positions = np.asarray([obs.position_world for obs in observations if obs.position_world is not None], dtype=np.float64)
        weights = np.asarray(
            [
                float(self.BASE_VIEW_WEIGHTS.get(obs.camera_name, 0.5)) * max(float(obs.confidence), 1e-6)
                for obs in observations
                if obs.position_world is not None
            ],
            dtype=np.float64,
        )
        if len(positions) == 1:
            return positions[0], observations

        centroid = np.average(positions, axis=0, weights=weights)
        distances = np.linalg.norm(positions - centroid[None, :], axis=1)
        keep_mask = distances <= self._outlier_threshold_m
        if not np.any(keep_mask):
            keep_mask[:] = True
        filtered = [obs for obs, keep in zip(observations, keep_mask.tolist()) if keep]
        filtered_positions = positions[keep_mask]
        filtered_weights = weights[keep_mask]
        fused = np.average(filtered_positions, axis=0, weights=filtered_weights)
        return fused, filtered

    @staticmethod
    def _build_query(target_name: str, target_entry: dict[str, Any]) -> str:
        aliases = target_entry.get("aliases") or []
        if isinstance(aliases, (list, tuple)):
            for alias in aliases:
                if isinstance(alias, str) and alias.strip():
                    return alias.strip()
        asset_label = target_entry.get("asset_label")
        if isinstance(asset_label, str) and asset_label.strip():
            return asset_label.strip()
        return str(target_name).replace("_", " ")

    def _target_rotation_quat(self, target_entry: dict[str, Any]) -> np.ndarray:
        rotation = target_entry.get("target_rotation_world")
        if (
            isinstance(rotation, (list, tuple))
            and len(rotation) == 3
            and all(isinstance(row, (list, tuple)) and len(row) == 3 for row in rotation)
        ):
            return self._rotation_matrix_to_quat(np.asarray(rotation, dtype=np.float64))
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _record_event(
        self,
        *,
        target_name: str,
        query: str,
        stage: str,
        prior_world: np.ndarray,
        observations: list[_ViewObservation],
        fused: dict[str, Any] | None,
        status: str,
    ) -> None:
        self._events.append(
            {
                "episode": self._episode_idx,
                "target_name": target_name,
                "query": query,
                "stage": stage,
                "status": status,
                "prior_world": prior_world.tolist(),
                "per_view": [
                    {
                        "camera_name": obs.camera_name,
                        "status": obs.status,
                        "confidence": float(obs.confidence),
                        "position_world": None if obs.position_world is None else obs.position_world.tolist(),
                        "pixel_coords": None if obs.pixel_coords is None else [int(obs.pixel_coords[0]), int(obs.pixel_coords[1])],
                        "prior_pixel_coords": None
                        if obs.prior_pixel_coords is None
                        else [float(obs.prior_pixel_coords[0]), float(obs.prior_pixel_coords[1])],
                        "prior_distance_m": obs.prior_distance_m,
                        "depth_m": obs.depth_m,
                    }
                    for obs in observations
                ],
                "fused": fused,
            }
        )

    @staticmethod
    def _project_world_to_pixel(
        position_world: np.ndarray,
        intrinsics: dict[str, float],
        transform_world: np.ndarray,
    ) -> tuple[float, float, float] | None:
        world_to_camera = np.linalg.inv(np.asarray(transform_world, dtype=np.float64))
        point_world_h = np.concatenate([np.asarray(position_world, dtype=np.float64), np.array([1.0], dtype=np.float64)])
        point_camera = world_to_camera @ point_world_h
        x_cam, y_cam, z_cam = point_camera[:3]
        depth = -float(z_cam)
        if not np.isfinite(depth) or depth <= 1e-6:
            return None
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        u = fx * (float(x_cam) / depth) + cx
        v = cy - fy * (float(y_cam) / depth)
        width = float(intrinsics["width"])
        height = float(intrinsics["height"])
        if not (0.0 <= u < width and 0.0 <= v < height):
            return None
        return float(u), float(v), depth

    @staticmethod
    def _deproject_pixel_to_world(
        u: float,
        v: float,
        depth: float,
        intrinsics: dict[str, float],
        transform_world: np.ndarray,
    ) -> np.ndarray:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        x_cam = (u - cx) * depth / fx
        y_cam = -(v - cy) * depth / fy
        z_cam = -depth
        point_camera_h = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float64)
        point_world_h = np.asarray(transform_world, dtype=np.float64) @ point_camera_h
        return point_world_h[:3]

    @staticmethod
    def _rotation_matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
        m = np.asarray(rotation, dtype=np.float64)
        trace = np.trace(m)
        if trace > 0.0:
            s = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        quat = np.array([w, x, y, z], dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return quat / norm
