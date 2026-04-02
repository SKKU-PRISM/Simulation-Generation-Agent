"""Factory env adapter: translates skill calls to 6D task-space actions.

Each Factory task requires a precise 3-phase control strategy:
  Phase 1 (Approach): Move above the insertion target
  Phase 2 (Align): Fine XY alignment directly above target
  Phase 3 (Insert/Thread): Descend (+ rotate for NutThread)

Success conditions (all tasks require XY < 2.5mm):
  PegInsert:  Z descent < 1.0mm into hole
  GearMesh:   Z descent < 1.5mm onto gear base shaft
  NutThread:  Z descent < 0.75mm + yaw ≤ 0 rad

6D action space: [dx, dy, dz, droll, dpitch, dyaw] ∈ [-1, 1]
  - Position: scaled by pos_threshold = 0.02m/step, then EMA(0.2)
  - Rotation: scaled by rot_threshold = 0.097 rad/step (~5.6°)
  - Roll/Pitch locked to (π, 0), only yaw effective
  - NutThread: yaw clamped to [-1, 0] (unidirectional)
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from typing import Optional

logger = logging.getLogger(__name__)


class FactoryDetector:
    """Reads object positions directly from Factory env tensors."""

    def __init__(self, env):
        self.env = env

    def get_object_position(self, name: str) -> np.ndarray:
        name_lower = name.lower()
        if any(k in name_lower for k in ("hole", "base", "bolt", "fixed")):
            pos = self.env.fixed_pos[0].cpu().numpy()
        elif any(k in name_lower for k in ("peg", "gear", "nut", "held")):
            pos = self.env.held_pos[0].cpu().numpy()
        else:
            pos = self.env.fixed_pos[0].cpu().numpy()
        return pos - self.env.scene.env_origins[0].cpu().numpy()

    def get_all_objects(self) -> dict:
        fixed_pos = self.get_object_position("fixed")
        held_pos = self.get_object_position("held")
        task_name = self.env.cfg_task.name
        if task_name == "peg_insert":
            return {"hole": {"position": fixed_pos}, "peg": {"position": held_pos}}
        elif task_name == "gear_mesh":
            return {"gear_base": {"position": fixed_pos}, "medium_gear": {"position": held_pos}}
        elif task_name == "nut_thread":
            return {"bolt": {"position": fixed_pos}, "nut": {"position": held_pos}}
        return {"fixed": {"position": fixed_pos}, "held": {"position": held_pos}}


class FactorySimSkills:
    """3-phase insertion/threading controller for Factory tasks.

    Uses Factory's internal tensors to compute the exact insertion target,
    then executes approach → align → insert/thread phases.
    """

    POS_THRESHOLD = np.array([0.02, 0.02, 0.02])

    def __init__(self, env, cameras=None):
        self.env = env
        self.device = env.device
        self.detector = FactoryDetector(env)
        self.cameras = cameras or {}
        self._frames = {"front": [], "top": []}
        self._step_count = 0
        self._action_history = []

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def fingertip_pos(self) -> np.ndarray:
        """Current fingertip midpoint position (env-local frame)."""
        return (self.env.fingertip_midpoint_pos[0] - self.env.scene.env_origins[0]).cpu().numpy()

    @property
    def insertion_target_pos(self) -> np.ndarray:
        """Exact position where held asset base must reach for success.

        Uses Factory's own utility function which accounts for task-specific
        offsets (gear base offset, bolt thread height, etc.).
        """
        from src.agent.isaac_lab.factory import factory_utils
        target_pos, _ = factory_utils.get_target_held_base_pose(
            self.env.fixed_pos, self.env.fixed_quat,
            self.env.cfg_task.name, self.env.cfg_task.fixed_asset_cfg,
            self.env.num_envs, self.env.device,
        )
        return (target_pos[0] - self.env.scene.env_origins[0]).cpu().numpy()

    @property
    def held_base_pos(self) -> np.ndarray:
        """Current held asset base position (with task-specific offset)."""
        from src.agent.isaac_lab.factory import factory_utils
        held_pos, _ = factory_utils.get_held_base_pose(
            self.env.held_pos, self.env.held_quat,
            self.env.cfg_task.name, self.env.cfg_task.fixed_asset_cfg,
            self.env.num_envs, self.env.device,
        )
        return (held_pos[0] - self.env.scene.env_origins[0]).cpu().numpy()

    # ------------------------------------------------------------------
    # Low-level step
    # ------------------------------------------------------------------

    def _step(self, action_6d: np.ndarray) -> dict:
        """Send one 6D action to the Factory env."""
        action_tensor = torch.tensor(
            action_6d, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        obs, reward, terminated, truncated, info = self.env.step(action_tensor)
        self._step_count += 1
        self._action_history.append(action_6d.copy())
        return {
            "reward": reward.item(),
            "terminated": terminated.item(),
            "truncated": truncated.item(),
        }

    def _pos_to_action(self, target_pos: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """Proportional position controller → normalized 6D action."""
        delta = target_pos - self.fingertip_pos
        normalized = np.clip(delta / self.POS_THRESHOLD * gain, -1.0, 1.0)
        return np.array([normalized[0], normalized[1], normalized[2], 0.0, 0.0, 0.0])

    # ------------------------------------------------------------------
    # Phase controllers
    # ------------------------------------------------------------------

    def _phase_approach(self, target_xy: np.ndarray, target_z: float,
                        height_above: float = 0.02, max_steps: int = 150) -> bool:
        """Phase 1: Move above target with coarse XY alignment."""
        approach_pos = np.array([target_xy[0], target_xy[1], target_z + height_above])
        print(f"    [approach] target={approach_pos}, fingertip={self.fingertip_pos}", flush=True)
        for i in range(max_steps):
            action = self._pos_to_action(approach_pos, gain=2.0)
            self._step(action)
            dist = np.linalg.norm(self.fingertip_pos - approach_pos)
            if dist < 0.015:
                print(f"    [approach] reached at step {i}, dist={dist:.4f}m", flush=True)
                return True
        print(f"    [approach] max_steps, dist={dist:.4f}m", flush=True)
        return True  # proceed anyway

    def _phase_align(self, target_xy: np.ndarray, hold_z: float,
                     max_steps: int = 80) -> bool:
        """Phase 2: Fine XY alignment while holding Z constant."""
        print(f"    [align] target_xy={target_xy}, hold_z={hold_z:.4f}", flush=True)
        for i in range(max_steps):
            # XY proportional, Z hold
            delta_xy = target_xy - self.fingertip_pos[:2]
            xy_action = np.clip(delta_xy / self.POS_THRESHOLD[:2] * 1.0, -1.0, 1.0)
            # Small Z correction to hold height
            delta_z = hold_z - self.fingertip_pos[2]
            z_action = np.clip(delta_z / self.POS_THRESHOLD[2] * 0.5, -0.3, 0.3)
            action = np.array([xy_action[0], xy_action[1], z_action, 0.0, 0.0, 0.0])
            self._step(action)

            xy_dist = np.linalg.norm(self.fingertip_pos[:2] - target_xy)
            if xy_dist < 0.002:
                print(f"    [align] aligned at step {i}, xy_dist={xy_dist:.4f}m", flush=True)
                return True
        xy_dist = np.linalg.norm(self.fingertip_pos[:2] - target_xy)
        print(f"    [align] max_steps, xy_dist={xy_dist:.4f}m", flush=True)
        return xy_dist < 0.004  # still proceed if close enough

    def _phase_descend(self, target_pos: np.ndarray, max_steps: int = 120) -> bool:
        """Phase 3a: Z descent with XY correction (PegInsert, GearMesh).

        Maintains XY alignment while descending toward target Z.
        """
        print(f"    [descend] target={target_pos}", flush=True)
        for i in range(max_steps):
            # XY correction + strong Z descent
            delta_xy = target_pos[:2] - self.fingertip_pos[:2]
            xy_action = np.clip(delta_xy / self.POS_THRESHOLD[:2] * 1.0, -0.5, 0.5)
            action = np.array([xy_action[0], xy_action[1], -1.0, 0.0, 0.0, 0.0])
            self._step(action)
            if self._check_success():
                print(f"    [descend] SUCCESS at step {i}", flush=True)
                self._hold_position(20)
                return True
            if i % 30 == 0:
                held = self.held_base_pos
                tgt = self.insertion_target_pos
                xy_err = np.linalg.norm(held[:2] - tgt[:2])
                z_gap = held[2] - tgt[2]
                print(f"    [descend] step={i} xy_err={xy_err:.4f}m z_gap={z_gap:.4f}m", flush=True)
        print(f"    [descend] max_steps reached", flush=True)
        return False

    def _phase_thread(self, target_pos: np.ndarray, max_steps: int = 180) -> bool:
        """Phase 3b: Descent + yaw rotation with XY correction (NutThread)."""
        print(f"    [thread] target={target_pos}, starting descent + rotation", flush=True)
        for i in range(max_steps):
            # XY correction + descent + yaw rotation
            delta_xy = target_pos[:2] - self.fingertip_pos[:2]
            xy_action = np.clip(delta_xy / self.POS_THRESHOLD[:2] * 1.0, -0.5, 0.5)
            action = np.array([xy_action[0], xy_action[1], -0.5, 0.0, 0.0, -0.8])
            self._step(action)
            if self._check_success():
                print(f"    [thread] SUCCESS at step {i}", flush=True)
                self._hold_position(20)
                return True
        print(f"    [thread] max_steps reached", flush=True)
        return False

    def _hold_position(self, steps: int = 20):
        """Hold current position (zero action) for stability."""
        for _ in range(steps):
            self._step(np.zeros(6))

    def _check_success(self) -> bool:
        """Check Factory's built-in success criteria."""
        check_rot = self.env.cfg_task.name == "nut_thread"
        successes = self.env._get_curr_successes(
            success_threshold=self.env.cfg_task.success_threshold,
            check_rot=check_rot,
        )
        return successes[0].item()

    # ------------------------------------------------------------------
    # High-level skills
    # ------------------------------------------------------------------

    @property
    def _fingertip_to_held_offset(self) -> np.ndarray:
        """Offset from fingertip to held asset base (fingertip is above held base)."""
        return self.fingertip_pos - self.held_base_pos

    def _target_to_fingertip_target(self, held_target: np.ndarray) -> np.ndarray:
        """Convert held-asset target to fingertip target by adding the offset."""
        offset = self._fingertip_to_held_offset
        return held_target + offset

    def execute_insert(self, target_name: str = "", **kwargs) -> dict:
        """Insert already-grasped object into target (PegInsert, GearMesh).

        3-phase strategy:
          1. Approach: move fingertip above insertion target (accounting for held-to-fingertip offset)
          2. Align: fine XY centering
          3. Descend: push down with XY correction until Factory success check passes
        """
        held_target = self.insertion_target_pos  # where held_base must go
        held = self.held_base_pos
        offset = self._fingertip_to_held_offset  # fingertip is above held base
        fingertip_target = held_target + offset   # where fingertip must go

        print(f"  [execute_insert] held_target={held_target}", flush=True)
        print(f"  [execute_insert] held_base={held}", flush=True)
        print(f"  [execute_insert] fingertip={self.fingertip_pos}", flush=True)
        print(f"  [execute_insert] offset={offset} (fingertip above held)", flush=True)
        print(f"  [execute_insert] fingertip_target={fingertip_target}", flush=True)
        xy_err = np.linalg.norm(held[:2] - held_target[:2])
        z_gap = held[2] - held_target[2]
        print(f"  [execute_insert] xy_err={xy_err:.4f}m, z_gap={z_gap:.4f}m", flush=True)

        # Phase 1: Approach — move fingertip above the target
        self._phase_approach(fingertip_target[:2], fingertip_target[2], height_above=0.02)

        # Phase 2: Fine XY alignment at approach height
        align_z = fingertip_target[2] + 0.005
        self._phase_align(fingertip_target[:2], hold_z=align_z)

        # Phase 3: Descend — push fingertip down (held base follows)
        descend_target = fingertip_target.copy()
        descend_target[2] -= 0.01  # push below target to ensure insertion
        success = self._phase_descend(descend_target, max_steps=200)

        return {"success": success, "steps": self._step_count}

    def execute_thread(self, target_name: str = "", **kwargs) -> dict:
        """Thread already-grasped nut onto bolt (NutThread).

        3-phase strategy:
          1. Approach: move above bolt (fingertip target = held target + offset)
          2. Align: fine XY centering
          3. Thread: descend + rotate yaw simultaneously
        """
        held_target = self.insertion_target_pos
        held = self.held_base_pos
        offset = self._fingertip_to_held_offset
        fingertip_target = held_target + offset

        print(f"  [execute_thread] held_target={held_target}, fingertip_target={fingertip_target}", flush=True)

        # Phase 1: Approach
        self._phase_approach(fingertip_target[:2], fingertip_target[2], height_above=0.015)

        # Phase 2: Align
        align_z = fingertip_target[2] + 0.003
        self._phase_align(fingertip_target[:2], hold_z=align_z)

        # Phase 3: Thread
        thread_target = fingertip_target.copy()
        thread_target[2] -= 0.005
        success = self._phase_thread(thread_target, max_steps=300)

        return {"success": success, "steps": self._step_count}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def move_to_position(self, target_xyz, tolerance=0.015, max_steps=60, gain=2.0, **kwargs) -> bool:
        """Generic position move (used by CaP generated code)."""
        target = np.array(target_xyz, dtype=np.float64)
        for i in range(max_steps):
            action = self._pos_to_action(target, gain=gain)
            self._step(action)
            dist = np.linalg.norm(self.fingertip_pos - target)
            if dist < tolerance:
                return True
        return False

    def is_success(self) -> bool:
        """Check current success state."""
        return self._check_success()

    def _capture_frame(self):
        """Capture frames from scene-integrated cameras."""
        for cam_name, cam in self.cameras.items():
            try:
                cam.update(dt=self.env.physics_dt)
                data = cam.data.output.get("rgb")
                if data is not None:
                    frame = data[0, ..., :3].cpu().numpy()
                    if cam_name in self._frames:
                        self._frames[cam_name].append(frame.copy())
            except Exception:
                pass

    def replay_with_cameras(self) -> dict[str, list]:
        """Replay action history with camera capture."""
        if not self._action_history or not self.cameras:
            return self._frames
        logger.info(f"Replaying {len(self._action_history)} actions with cameras...")
        self.env.reset()
        self._frames = {"front": [], "top": []}
        for _ in range(10):
            self.env.step(torch.zeros(1, 6, device=self.device))
        for i, action in enumerate(self._action_history):
            action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.env.step(action_tensor)
            if i % 10 == 0:
                self._capture_frame()
        return self._frames

    def get_frames(self) -> dict[str, list]:
        return self._frames
