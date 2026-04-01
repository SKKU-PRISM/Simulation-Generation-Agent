"""Generate capture_scene.py script for general IsaacLab tasks.

This module creates a standalone capture script that:
1. Loads the generated env_cfg.py
2. Injects 3 cameras (front, top, wrist) into the scene
3. Captures screenshots from each angle
4. Saves images to a debug/ folder
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Default camera configurations for 3 angles
DEFAULT_CAMERAS = {
    "front": {
        "position": [1.5, 0.0, 0.8],
        "target": [0.5, 0.0, 0.3],
    },
    "top": {
        "position": [0.5, 0.0, 1.8],
        "target": [0.5, 0.0, 0.0],
    },
    "wrist": {
        "cam_type": "body_mounted",
        "parent_body": "panda_hand",
        "offset_pos": [0.04, 0.0, 0.04],
        "offset_rot": [0.0, 0.7071, 0.7071, 0.0],  # wxyz: looking down
        "convention": "world",
    },
}

# Body name mapping per robot type
WRIST_BODY_MAP = {
    "franka": "panda_hand",
    "ur10e": "ee_link",
    "openarm": "openarm_inner_finger_left",
    "so101": "moving_jaw",
}


def build_capture_script(env_class: str, robot: str = "franka",
                         camera_overrides: dict | None = None) -> str:
    """Generate a capture_scene.py script for any IsaacLab task.

    Args:
        env_class: Name of the environment config class (e.g., "FrankaStackEnvCfg").
        robot: Robot type for wrist camera body mapping.
        camera_overrides: Optional per-camera overrides for position/target.

    Returns:
        Python source code string for capture_scene.py.
    """
    cameras = dict(DEFAULT_CAMERAS)
    if camera_overrides:
        for cam_name, overrides in camera_overrides.items():
            if cam_name in cameras:
                cameras[cam_name].update(overrides)

    wrist_body = WRIST_BODY_MAP.get(robot, "panda_hand")

    front = cameras["front"]
    top = cameras["top"]
    wrist = cameras["wrist"]

    script = f'''"""Capture debug screenshots (front, top, wrist) for scene verification."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, {str(PROJECT_ROOT)!r})

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from PIL import Image
from types import SimpleNamespace
from isaaclab.envs import ManagerBasedRLEnv

from env_cfg import {env_class}
from src.agent.data_collection.config import CameraConfig
from src.agent.data_collection.sim_camera import SceneCameraManager, inject_cameras_into_scene


def main():
    env_cfg = {env_class}()
    env_cfg.scene.num_envs = 1

    # Define 3 cameras: front, top, wrist
    camera_configs = {{
        "front": CameraConfig(
            name="front",
            cam_type="fixed",
            position={repr(front["position"])},
            target={repr(front["target"])},
            resolution=(1280, 720),
        ),
        "top": CameraConfig(
            name="top",
            cam_type="fixed",
            position={repr(top["position"])},
            target={repr(top["target"])},
            resolution=(1280, 720),
        ),
        "wrist": CameraConfig(
            name="wrist",
            cam_type="body_mounted",
            parent_body="{wrist_body}",
            offset_pos={repr(wrist.get("offset_pos", [0.04, 0.0, 0.04]))},
            offset_rot={repr(wrist.get("offset_rot", [0.0, 0.7071, 0.7071, 0.0]))},
            convention="{wrist.get("convention", "world")}",
            resolution=(640, 480),
        ),
    }}

    camera_attr_names = inject_cameras_into_scene(
        env_cfg,
        SimpleNamespace(cameras=camera_configs),
        camera_names=["front", "top", "wrist"],
    )

    env = ManagerBasedRLEnv(cfg=env_cfg)
    post_setup = globals().get("post_env_setup")
    if callable(post_setup):
        post_setup(env)
    env.reset()
    cameras = SceneCameraManager(env, camera_attr_names, camera_configs)

    # Step simulation to let scene settle
    for _ in range(6):
        actions = torch.zeros_like(env.action_manager.action)
        env.step(actions)

    # Capture from all cameras
    images = cameras.capture_all()

    # Save to debug/ folder
    debug_dir = Path(__file__).resolve().parent / "debug"
    debug_dir.mkdir(exist_ok=True)

    saved = []
    for attr_name, img in images.items():
        if img is None:
            print(f"WARNING: capture failed for {{attr_name}}", flush=True)
            continue
        # Map attr_name (e.g., "front_cam") back to logical name
        logical_name = attr_name[:-4] if attr_name.endswith("_cam") else attr_name
        out_path = debug_dir / f"{{logical_name}}.png"
        Image.fromarray(img).save(out_path)
        saved.append(str(out_path))
        print(f"SAVED: {{out_path}}", flush=True)

    print(f"DEBUG_CAPTURE: {{len(saved)}} images saved to {{debug_dir}}", flush=True)

    # Signal success
    marker = os.environ.get("ISAACLAB_SUCCESS_MARKER")
    if marker:
        Path(marker).write_text("DEBUG_CAPTURE")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
'''
    return script.strip() + "\n"
