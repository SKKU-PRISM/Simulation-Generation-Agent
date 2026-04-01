"""Franka CaP runtime skills for generated code."""

from .base import BaseTabletopCaPSkills


class FrankaSkills(BaseTabletopCaPSkills):
    """Franka-specific tabletop CaP skill wrapper.

    Includes Factory task skills for insertion/threading where
    the robot starts with the object already grasped.
    """

    def execute_insert(
        self,
        target_name: str,
        approach_height: float = 0.03,
        insertion_speed: float = 0.002,
        max_steps: int = 300,
        skill_description: str = "",
        **kwargs,
    ) -> dict:
        """Insert already-grasped object into a target (hole/socket/shaft).

        The robot must already be holding the object (gripper closed).
        Do NOT call execute_pick_object() before this.

        Args:
            target_name: Name of the fixed target (e.g. "hole", "gear_base")
            approach_height: Height above target for approach (m)
            insertion_speed: Descent speed per step (m)
            max_steps: Maximum insertion steps

        Returns:
            dict with "success" and "message"
        """
        return self._skills.execute_insert(
            target_name=target_name,
            approach_height=approach_height,
            insertion_speed=insertion_speed,
            max_steps=max_steps,
        )

    def execute_thread(
        self,
        target_name: str,
        rotation_per_step: float = 0.05,
        descent_per_rotation: float = 0.0005,
        total_rotations: float = 2.0,
        max_steps: int = 500,
        skill_description: str = "",
        **kwargs,
    ) -> dict:
        """Thread already-grasped nut onto a bolt.

        The robot must already be holding the nut (gripper closed).
        Do NOT call execute_pick_object() before this.

        Args:
            target_name: Name of the bolt entity
            rotation_per_step: Radians to rotate per step
            descent_per_rotation: Descent per full rotation (m)
            total_rotations: Target number of rotations
            max_steps: Maximum threading steps

        Returns:
            dict with "success" and "message"
        """
        return self._skills.execute_thread(
            target_name=target_name,
            rotation_per_step=rotation_per_step,
            descent_per_rotation=descent_per_rotation,
            total_rotations=total_rotations,
            max_steps=max_steps,
        )
