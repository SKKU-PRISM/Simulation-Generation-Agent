"""
Coordinate Frame Transformations for SO-101 Robot

Transforms target positions from external coordinate frames (e.g., world)
to the robot's base_link frame for IK computation.

Usage:
    transformer = FrameTransformer("robot_configs/world2robot_matrices/robot3_matrix.json")

    # Transform position from world frame to base_link
    target_base = transformer.transform_position([0.5, 0.2, 0.1], from_frame="world")
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Create rotation matrix from roll-pitch-yaw angles (in degrees).

    Convention: ZYX (yaw around Z, then pitch around Y, then roll around X)

    Args:
        roll: Rotation around X axis (degrees)
        pitch: Rotation around Y axis (degrees)
        yaw: Rotation around Z axis (degrees)

    Returns:
        3x3 rotation matrix
    """
    # Convert to radians
    r = np.radians(roll)
    p = np.radians(pitch)
    y = np.radians(yaw)

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])

    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0],
        [np.sin(y), np.cos(y), 0],
        [0, 0, 1]
    ])

    # ZYX order: R = Rz @ Ry @ Rx
    return Rz @ Ry @ Rx


def make_homogeneous_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Create 4x4 homogeneous transformation matrix.

    Args:
        rotation: 3x3 rotation matrix
        translation: 3x1 or (3,) translation vector

    Returns:
        4x4 homogeneous transformation matrix
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation.flatten()
    return T


def invert_homogeneous_matrix(T: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 homogeneous transformation matrix.

    For T = [R | t], T_inv = [R^T | -R^T @ t]

    Args:
        T: 4x4 homogeneous transformation matrix

    Returns:
        4x4 inverse transformation matrix
    """
    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


class FrameTransformer:
    """
    Coordinate frame transformer for robot control.

    Transforms positions from external coordinate frames to robot's base_link frame.

    Configuration format (JSON):
    {
        "robot_id": "robot3",
        "frames": {
            "world": {
                "translation": [x, y, z],           # world 기준 로봇 위치 (미터)
                "rotation_rpy": [roll, pitch, yaw]  # world 기준 로봇 회전 (도)
            }
        }
    }

    Example:
        로봇이 world 좌표계에서 (1.0, 0.5, 0.0) 위치에 yaw 90도 회전:
        {
            "translation": [1.0, 0.5, 0.0],
            "rotation_rpy": [0.0, 0.0, 90.0]
        }

    Transformation:
        p_base = T_base_from_world @ p_world
        where T_base_from_world = inverse(T_world_from_base)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize frame transformer.

        Args:
            config_path: Path to frames configuration JSON file
        """
        self.frames = {}
        self.config_path = config_path
        self.robot_id = "unknown"

        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def load_config(self, config_path: str):
        """Load frame configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)

        self.robot_id = config.get("robot_id", "unknown")

        for frame_name, frame_data in config.get("frames", {}).items():
            # User provides: robot position/orientation in world frame
            robot_position = np.array(frame_data.get("translation", [0, 0, 0]))
            robot_rpy = frame_data.get("rotation_rpy", [0, 0, 0])

            # Compute T_world_from_base (robot pose in world frame)
            R_world_from_base = rotation_matrix_from_rpy(robot_rpy[0], robot_rpy[1], robot_rpy[2])
            T_world_from_base = make_homogeneous_matrix(R_world_from_base, robot_position)

            # Compute inverse: T_base_from_world (for transforming world points to base_link)
            T_base_from_world = invert_homogeneous_matrix(T_world_from_base)

            self.frames[frame_name] = {
                "T_base_from_frame": T_base_from_world,
                "T_frame_from_base": T_world_from_base,
                "robot_position": robot_position,
                "robot_rpy": robot_rpy,
                "description": frame_data.get("description", ""),
            }

    def reload(self):
        """Reload configuration from file (for hot-reloading)."""
        if self.config_path:
            self.load_config(self.config_path)

    def get_available_frames(self) -> list:
        """Get list of available coordinate frames."""
        return list(self.frames.keys())

    def has_frame(self, frame_name: str) -> bool:
        """Check if a frame is defined."""
        return frame_name in self.frames

    def transform_position(
        self,
        position: Union[np.ndarray, list, tuple],
        from_frame: str,
    ) -> np.ndarray:
        """
        Transform position from external frame to base_link.

        Args:
            position: [x, y, z] position in the source frame (meters)
            from_frame: Name of the source coordinate frame

        Returns:
            [x, y, z] position in base_link frame (meters)

        Raises:
            ValueError: If frame is not defined
        """
        if from_frame == "base_link":
            return np.array(position)

        if from_frame not in self.frames:
            raise ValueError(f"Unknown frame: '{from_frame}'. Available: {self.get_available_frames()}")

        T = self.frames[from_frame]["T_base_from_frame"]

        # Convert to homogeneous coordinates
        p_frame = np.array([position[0], position[1], position[2], 1.0])

        # Transform: p_base = T_base_from_world @ p_world
        p_base = T @ p_frame

        return p_base[:3]

    def transform_pose(
        self,
        position: Union[np.ndarray, list, tuple],
        orientation: np.ndarray,
        from_frame: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform pose (position + orientation) from external frame to base_link.

        Args:
            position: [x, y, z] position in the source frame (meters)
            orientation: 3x3 rotation matrix in the source frame
            from_frame: Name of the source coordinate frame

        Returns:
            Tuple of (position, orientation) in base_link frame
        """
        if from_frame == "base_link":
            return np.array(position), orientation

        if from_frame not in self.frames:
            raise ValueError(f"Unknown frame: '{from_frame}'. Available: {self.get_available_frames()}")

        T = self.frames[from_frame]["T_base_from_frame"]
        R = T[:3, :3]

        # Transform position
        p_base = self.transform_position(position, from_frame)

        # Transform orientation
        orientation_base = R @ orientation

        return p_base, orientation_base

    def get_frame_info(self, frame_name: str) -> dict:
        """Get information about a specific frame."""
        if frame_name not in self.frames:
            return None

        frame = self.frames[frame_name]
        return {
            "name": frame_name,
            "robot_position": frame["robot_position"].tolist(),
            "robot_rpy": frame["robot_rpy"],
            "description": frame["description"],
        }

    def print_info(self):
        """Print information about all configured frames."""
        print(f"FrameTransformer (robot: {self.robot_id})")
        print(f"  Config: {self.config_path}")
        print(f"  Available frames: {self.get_available_frames()}")
        for name in self.frames:
            info = self.get_frame_info(name)
            print(f"    {name}:")
            print(f"      Robot position in {name}: {info['robot_position']} m")
            print(f"      Robot rotation in {name}: {info['robot_rpy']} deg")
