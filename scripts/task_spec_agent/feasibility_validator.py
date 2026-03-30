"""
Physical Feasibility Validation Module for Task Specification Agent

This module provides functionality to validate whether a robot can physically
execute a given task plan based on its kinematic constraints, workspace limits,
and capabilities.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml

from task_decomposer import TaskPlan, AtomicAction


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RobotProfile:
    """
    Represents a robot's physical and kinematic capabilities.

    This profile contains all the information needed to validate whether
    a robot can execute specific manipulation tasks.
    """

    name: str
    """Robot identifier (e.g., 'franka', 'openarm', 'ur10', 'so101')"""

    dof: int
    """Degrees of freedom (total DOFs including gripper)"""

    max_reach: float
    """Maximum reach distance in meters"""

    workspace: Dict[str, List[float]]
    """Workspace boundaries: {x: [min, max], y: [min, max], z: [min, max]}"""

    joint_limits: Dict[str, List[float]]
    """Joint position limits: {joint_name: [min, max]}"""

    gripper_type: str
    """Type of gripper: 'parallel_jaw', 'suction', 'claw', 'none'"""

    payload_capacity: float = 5.0
    """Maximum payload capacity in kilograms"""

    gripper_max_opening: Optional[float] = None
    """Maximum gripper opening in meters (None for suction)"""

    tabletop_reach: Optional[float] = None
    """Practical horizontal reach at tabletop height in meters"""

    arm_dofs: Optional[int] = None
    """Number of arm DOFs (excluding gripper)"""

    def __str__(self) -> str:
        return (f"RobotProfile(name={self.name}, dof={self.dof}, "
                f"max_reach={self.max_reach}m, gripper={self.gripper_type})")

    def is_position_in_workspace(self, x: float, y: float, z: float) -> bool:
        """
        Check if a 3D position is within the robot's workspace.

        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            z: Z coordinate in meters

        Returns:
            True if position is in workspace, False otherwise
        """
        x_range = self.workspace.get('x', [-float('inf'), float('inf')])
        y_range = self.workspace.get('y', [-float('inf'), float('inf')])
        z_range = self.workspace.get('z', [-float('inf'), float('inf')])

        return (x_range[0] <= x <= x_range[1] and
                y_range[0] <= y <= y_range[1] and
                z_range[0] <= z <= z_range[1])

    def is_reachable(self, x: float, y: float, z: float) -> bool:
        """
        Check if a 3D position is reachable by the robot.

        This checks both workspace boundaries and distance from base.

        Args:
            x: X coordinate in meters
            y: Y coordinate in meters
            z: Z coordinate in meters

        Returns:
            True if position is reachable, False otherwise
        """
        # Check workspace boundaries first
        if not self.is_position_in_workspace(x, y, z):
            return False

        # Check distance from base (assuming base at origin)
        distance = math.sqrt(x**2 + y**2 + z**2)
        return distance <= self.max_reach

    def supports_action_type(self, action_type: str) -> bool:
        """
        Check if the robot can perform the specified action type.

        Args:
            action_type: Type of action (reach, grasp, release, etc.)

        Returns:
            True if action is supported, False otherwise
        """
        # All robots support basic motion actions
        motion_actions = ['reach', 'lift', 'place', 'push', 'pull']
        if action_type in motion_actions:
            return True

        # Grasping actions depend on gripper type
        grasp_actions = ['grasp', 'release']
        if action_type in grasp_actions:
            return self.gripper_type != 'none'

        # Open/close actions are for articulated grippers or drawers
        articulation_actions = ['open', 'close']
        if action_type in articulation_actions:
            # Can open/close if has gripper (for pushing/pulling) or for drawer operations
            return self.gripper_type != 'none'

        # Unknown action type
        logger.warning(f"Unknown action type: {action_type}")
        return False

    def can_grasp_object(self, object_size: Optional[float] = None) -> bool:
        """
        Check if the robot can grasp an object of given size.

        Args:
            object_size: Size of object in meters (diameter or width)

        Returns:
            True if object can be grasped, False otherwise
        """
        if self.gripper_type == 'none':
            return False

        if self.gripper_type == 'suction':
            # Suction grippers can handle various sizes
            return True

        if object_size is not None and self.gripper_max_opening is not None:
            # Check if object fits in gripper
            return object_size <= self.gripper_max_opening

        # If size unknown or no limit specified, assume it can grasp
        return True


@dataclass
class ValidationError:
    """
    Represents a critical error that prevents task execution.
    """

    action_id: str
    """ID of the action that failed validation"""

    error_type: str
    """Type of error: 'workspace', 'reachability', 'action_compatibility', 'gripper'"""

    message: str
    """Detailed error message"""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional error details"""

    def __str__(self) -> str:
        return f"ValidationError({self.action_id}): [{self.error_type}] {self.message}"


@dataclass
class ValidationWarning:
    """
    Represents a warning that suggests caution but doesn't prevent execution.
    """

    action_id: str
    """ID of the action that triggered the warning"""

    warning_type: str
    """Type of warning: 'near_limit', 'tight_constraint', 'suboptimal'"""

    message: str
    """Detailed warning message"""

    details: Dict[str, Any] = field(default_factory=dict)
    """Additional warning details"""

    def __str__(self) -> str:
        return f"ValidationWarning({self.action_id}): [{self.warning_type}] {self.message}"


@dataclass
class ValidationResult:
    """
    Result of task plan validation including errors and warnings.
    """

    is_valid: bool
    """Whether the task plan can be executed"""

    errors: List[ValidationError] = field(default_factory=list)
    """Critical errors that prevent execution"""

    warnings: List[ValidationWarning] = field(default_factory=list)
    """Warnings that suggest caution"""

    robot_profile: Optional[RobotProfile] = None
    """Robot profile used for validation"""

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return (f"ValidationResult({status}): "
                f"{len(self.errors)} errors, {len(self.warnings)} warnings")

    def add_error(self, action_id: str, error_type: str, message: str,
                  details: Optional[Dict[str, Any]] = None):
        """Add a validation error."""
        self.errors.append(ValidationError(
            action_id=action_id,
            error_type=error_type,
            message=message,
            details=details or {}
        ))
        self.is_valid = False

    def add_warning(self, action_id: str, warning_type: str, message: str,
                   details: Optional[Dict[str, Any]] = None):
        """Add a validation warning."""
        self.warnings.append(ValidationWarning(
            action_id=action_id,
            warning_type=warning_type,
            message=message,
            details=details or {}
        ))

    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        lines = []
        lines.append(f"Validation Result: {'PASS' if self.is_valid else 'FAIL'}")
        lines.append(f"Robot: {self.robot_profile.name if self.robot_profile else 'Unknown'}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)


class FeasibilityValidator:
    """
    Validator for checking physical feasibility of robot manipulation tasks.

    This class validates task plans against robot capabilities to ensure
    that planned actions are physically possible given kinematic constraints,
    workspace limits, and gripper capabilities.
    """

    def __init__(self, robot_type: str = "franka"):
        """
        Initialize the FeasibilityValidator.

        Args:
            robot_type: Type of robot to validate for (franka, openarm, ur10, so101)

        Raises:
            ValueError: If robot profile cannot be loaded
        """
        self.robot_type = robot_type
        self.robot_profile = self._load_robot_profile(robot_type)
        logger.info(f"FeasibilityValidator initialized for robot: {robot_type}")

    def _load_robot_profile(self, robot_type: str) -> RobotProfile:
        """
        Load robot profile from YAML configuration file.

        Args:
            robot_type: Type of robot (franka, openarm, ur10, so101)

        Returns:
            RobotProfile object with robot specifications

        Raises:
            ValueError: If profile file doesn't exist or is invalid
        """
        # Find profile file
        config_dir = Path(__file__).parent.parent.parent / "configs" / "robot_profiles"
        profile_path = config_dir / f"{robot_type}.yaml"

        if not profile_path.exists():
            raise ValueError(f"Robot profile not found: {profile_path}")

        try:
            with open(profile_path, 'r') as f:
                config = yaml.safe_load(f)

            # Extract robot info
            robot_info = config.get('robot', {})
            kinematics = config.get('kinematics', {})
            joints = config.get('joints', {})
            gripper = config.get('gripper', {})

            # Build joint limits dictionary
            joint_limits = {}
            arm_joints = joints.get('arm', {})
            if 'limits' in arm_joints:
                joint_limits.update(arm_joints['limits'])

            finger_joints = joints.get('finger', {})
            if 'limits' in finger_joints:
                joint_limits.update(finger_joints['limits'])

            # Create workspace dict
            workspace_envelope = kinematics.get('workspace_envelope', {})
            workspace = {
                'x': workspace_envelope.get('x', [-1.0, 1.0]),
                'y': workspace_envelope.get('y', [-1.0, 1.0]),
                'z': workspace_envelope.get('z', [-0.5, 1.5])
            }

            # Get gripper info
            gripper_type = gripper.get('type', 'none')
            gripper_max_opening = gripper.get('max_opening')

            # Estimate payload capacity (default 5kg if not specified)
            # Could be derived from effort limits, but use conservative default
            payload_capacity = config.get('physics', {}).get('payload_capacity', 5.0)

            # Create profile
            profile = RobotProfile(
                name=robot_info.get('name', robot_type),
                dof=robot_info.get('total_dofs', 0),
                max_reach=kinematics.get('max_reach', 1.0),
                workspace=workspace,
                joint_limits=joint_limits,
                gripper_type=gripper_type,
                payload_capacity=payload_capacity,
                gripper_max_opening=gripper_max_opening,
                tabletop_reach=kinematics.get('tabletop_reach'),
                arm_dofs=robot_info.get('arm_dofs')
            )

            logger.info(f"Loaded robot profile: {profile}")
            return profile

        except Exception as e:
            raise ValueError(f"Failed to load robot profile for {robot_type}: {str(e)}")

    def validate(self, task_plan: TaskPlan) -> ValidationResult:
        """
        Validate a task plan for physical feasibility.

        This method checks:
        - Workspace range validation
        - Reachability of target positions
        - Action type compatibility with robot capabilities
        - Gripper compatibility with objects

        Args:
            task_plan: TaskPlan to validate

        Returns:
            ValidationResult with errors and warnings
        """
        logger.info(f"Validating task plan: {task_plan.task_name}")

        result = ValidationResult(
            is_valid=True,
            robot_profile=self.robot_profile
        )

        # Validate each action
        for action in task_plan.actions:
            self._validate_action(action, result)

        # Additional global validations
        self._validate_task_sequence(task_plan, result)

        logger.info(f"Validation complete: {result}")
        return result

    def _validate_action(self, action: AtomicAction, result: ValidationResult):
        """
        Validate a single atomic action.

        Args:
            action: AtomicAction to validate
            result: ValidationResult to update with findings
        """
        # Validate action type compatibility
        if not self.robot_profile.supports_action_type(action.action_type):
            result.add_error(
                action_id=action.id,
                error_type='action_compatibility',
                message=f"Robot '{self.robot_profile.name}' does not support "
                       f"action type '{action.action_type}'",
                details={'robot_gripper': self.robot_profile.gripper_type}
            )
            return

        # Validate gripper compatibility for grasp/release actions
        if action.action_type in ['grasp', 'release']:
            self._validate_gripper_action(action, result)

        # Validate workspace and reachability for position-based actions
        if action.action_type in ['reach', 'place', 'push', 'pull']:
            self._validate_position_action(action, result)

        # Validate lift action
        if action.action_type == 'lift':
            self._validate_lift_action(action, result)

        # Validate open/close actions
        if action.action_type in ['open', 'close']:
            self._validate_articulation_action(action, result)

    def _validate_gripper_action(self, action: AtomicAction, result: ValidationResult):
        """Validate gripper-related actions (grasp, release)."""
        # Check if robot has a gripper
        if self.robot_profile.gripper_type == 'none':
            result.add_error(
                action_id=action.id,
                error_type='gripper',
                message=f"Robot '{self.robot_profile.name}' has no gripper",
                details={'action_type': action.action_type}
            )
            return

        # Check object size if specified in parameters
        object_size = action.parameters.get('object_size')
        if object_size is not None:
            if not self.robot_profile.can_grasp_object(object_size):
                result.add_error(
                    action_id=action.id,
                    error_type='gripper',
                    message=f"Object size ({object_size}m) exceeds gripper capacity "
                           f"({self.robot_profile.gripper_max_opening}m)",
                    details={
                        'object_size': object_size,
                        'max_opening': self.robot_profile.gripper_max_opening
                    }
                )
                return

            # Warning if object is close to gripper limit
            if (self.robot_profile.gripper_max_opening and
                object_size > 0.9 * self.robot_profile.gripper_max_opening):
                result.add_warning(
                    action_id=action.id,
                    warning_type='tight_constraint',
                    message=f"Object size ({object_size}m) is close to gripper limit "
                           f"({self.robot_profile.gripper_max_opening}m)",
                    details={'size_ratio': object_size / self.robot_profile.gripper_max_opening}
                )

        # Validate target object is specified
        if not action.target_object:
            result.add_warning(
                action_id=action.id,
                warning_type='suboptimal',
                message=f"Grasp action has no target_object specified",
                details={'action_type': action.action_type}
            )

    def _validate_position_action(self, action: AtomicAction, result: ValidationResult):
        """Validate actions that involve reaching a position."""
        # Extract position from parameters
        position = action.parameters.get('position')
        target_position = action.parameters.get('target_position')

        pos = position or target_position

        if pos is None:
            # No explicit position - will be determined at execution time
            result.add_warning(
                action_id=action.id,
                warning_type='suboptimal',
                message=f"No position specified for {action.action_type} action",
                details={'action': action.action_type}
            )
            return

        # Position should be [x, y, z]
        if not isinstance(pos, (list, tuple)) or len(pos) != 3:
            result.add_error(
                action_id=action.id,
                error_type='workspace',
                message=f"Invalid position format: {pos}",
                details={'position': pos}
            )
            return

        x, y, z = pos

        # Check workspace boundaries
        if not self.robot_profile.is_position_in_workspace(x, y, z):
            result.add_error(
                action_id=action.id,
                error_type='workspace',
                message=f"Position ({x:.3f}, {y:.3f}, {z:.3f}) is outside workspace",
                details={
                    'position': [x, y, z],
                    'workspace': self.robot_profile.workspace
                }
            )
            return

        # Check reachability
        if not self.robot_profile.is_reachable(x, y, z):
            result.add_error(
                action_id=action.id,
                error_type='reachability',
                message=f"Position ({x:.3f}, {y:.3f}, {z:.3f}) is not reachable",
                details={
                    'position': [x, y, z],
                    'max_reach': self.robot_profile.max_reach,
                    'distance': math.sqrt(x**2 + y**2 + z**2)
                }
            )
            return

        # Warning if near workspace limits
        distance = math.sqrt(x**2 + y**2 + z**2)
        if distance > 0.9 * self.robot_profile.max_reach:
            result.add_warning(
                action_id=action.id,
                warning_type='near_limit',
                message=f"Position ({x:.3f}, {y:.3f}, {z:.3f}) is near reach limit",
                details={
                    'distance': distance,
                    'max_reach': self.robot_profile.max_reach,
                    'ratio': distance / self.robot_profile.max_reach
                }
            )

    def _validate_lift_action(self, action: AtomicAction, result: ValidationResult):
        """Validate lift action."""
        # Check if object weight is specified
        object_weight = action.parameters.get('object_weight')
        if object_weight is not None:
            if object_weight > self.robot_profile.payload_capacity:
                result.add_error(
                    action_id=action.id,
                    error_type='action_compatibility',
                    message=f"Object weight ({object_weight}kg) exceeds payload capacity "
                           f"({self.robot_profile.payload_capacity}kg)",
                    details={
                        'object_weight': object_weight,
                        'payload_capacity': self.robot_profile.payload_capacity
                    }
                )
                return

            # Warning if close to payload limit
            if object_weight > 0.8 * self.robot_profile.payload_capacity:
                result.add_warning(
                    action_id=action.id,
                    warning_type='near_limit',
                    message=f"Object weight ({object_weight}kg) is close to payload limit",
                    details={'weight_ratio': object_weight / self.robot_profile.payload_capacity}
                )

        # Check lift distance if specified
        lift_distance = action.parameters.get('distance')
        if lift_distance is not None and lift_distance > 0.5:
            result.add_warning(
                action_id=action.id,
                warning_type='suboptimal',
                message=f"Large lift distance ({lift_distance}m) may be slow or unstable",
                details={'lift_distance': lift_distance}
            )

    def _validate_articulation_action(self, action: AtomicAction, result: ValidationResult):
        """Validate open/close actions (drawers, doors, etc.)."""
        # These actions typically involve pushing/pulling, which requires a gripper or suction
        if self.robot_profile.gripper_type == 'none':
            result.add_error(
                action_id=action.id,
                error_type='action_compatibility',
                message=f"Robot '{self.robot_profile.name}' cannot perform "
                       f"{action.action_type} action without end effector",
                details={'robot_gripper': self.robot_profile.gripper_type}
            )
            return

        # Check opening distance for drawer/door operations
        opening_distance = action.parameters.get('distance')
        if opening_distance is not None and opening_distance > 0.5:
            result.add_warning(
                action_id=action.id,
                warning_type='suboptimal',
                message=f"Large opening distance ({opening_distance}m) may require "
                        f"multiple movements",
                details={'opening_distance': opening_distance}
            )

    def _validate_task_sequence(self, task_plan: TaskPlan, result: ValidationResult):
        """
        Validate the overall task sequence for feasibility.

        Args:
            task_plan: Complete task plan
            result: ValidationResult to update
        """
        # Check for grasp before place pattern
        has_grasp = False
        for action in task_plan.actions:
            if action.action_type == 'place' and not has_grasp:
                result.add_warning(
                    action_id=action.id,
                    warning_type='suboptimal',
                    message="Place action without preceding grasp action",
                    details={'task_name': task_plan.task_name}
                )

            if action.action_type == 'grasp':
                has_grasp = True
            elif action.action_type == 'release':
                has_grasp = False

        # Check for excessively long action sequences
        if len(task_plan.actions) > 20:
            result.add_warning(
                action_id='task_plan',
                warning_type='suboptimal',
                message=f"Task has many actions ({len(task_plan.actions)}), "
                        f"execution may be time-consuming",
                details={'num_actions': len(task_plan.actions)}
            )

        # Validate that required objects/locations are consistent
        self._validate_object_consistency(task_plan, result)

    def _validate_object_consistency(self, task_plan: TaskPlan, result: ValidationResult):
        """Validate that objects and locations are used consistently."""
        grasped_objects = set()

        for action in task_plan.actions:
            # Track grasped objects
            if action.action_type == 'grasp' and action.target_object:
                if action.target_object in grasped_objects:
                    result.add_warning(
                        action_id=action.id,
                        warning_type='suboptimal',
                        message=f"Object '{action.target_object}' may already be grasped",
                        details={'object': action.target_object}
                    )
                grasped_objects.add(action.target_object)

            # Track released objects
            if action.action_type == 'release' and action.target_object:
                if action.target_object not in grasped_objects:
                    result.add_warning(
                        action_id=action.id,
                        warning_type='suboptimal',
                        message=f"Releasing object '{action.target_object}' that wasn't grasped",
                        details={'object': action.target_object}
                    )
                grasped_objects.discard(action.target_object)


if __name__ == "__main__":
    """
    Example usage and testing of the feasibility validator.
    """
    import sys
    from task_decomposer import TaskPlan, AtomicAction

    # Test with different robots
    robot_types = ['franka', 'openarm', 'ur10', 'so101']

    if len(sys.argv) > 1:
        robot_types = [sys.argv[1]]

    for robot_type in robot_types:
        print("\n" + "="*60)
        print(f"Testing Feasibility Validator for: {robot_type.upper()}")
        print("="*60)

        try:
            # Create validator
            validator = FeasibilityValidator(robot_type=robot_type)
            print(f"\nRobot Profile: {validator.robot_profile}")
            print(f"  DOF: {validator.robot_profile.dof}")
            print(f"  Max Reach: {validator.robot_profile.max_reach}m")
            print(f"  Gripper: {validator.robot_profile.gripper_type}")
            print(f"  Workspace: {validator.robot_profile.workspace}")

            # Example task 1: Simple pick and place
            print("\n" + "-"*60)
            print("Example 1: Valid Pick and Place Task")
            print("-"*60)

            task1 = TaskPlan(
                task_name="pick_and_place",
                description="Pick a cube and place it on target",
                actions=[
                    AtomicAction(
                        id="action_1",
                        action_type="reach",
                        target_object="cube",
                        parameters={"position": [0.5, 0.0, 0.3]},
                        prerequisites=[]
                    ),
                    AtomicAction(
                        id="action_2",
                        action_type="grasp",
                        target_object="cube",
                        parameters={"object_size": 0.05},
                        prerequisites=["action_1"]
                    ),
                    AtomicAction(
                        id="action_3",
                        action_type="lift",
                        target_object="cube",
                        parameters={"distance": 0.1, "object_weight": 0.5},
                        prerequisites=["action_2"]
                    ),
                    AtomicAction(
                        id="action_4",
                        action_type="place",
                        target_location="target_marker",
                        parameters={"target_position": [0.6, 0.2, 0.3]},
                        prerequisites=["action_3"]
                    ),
                    AtomicAction(
                        id="action_5",
                        action_type="release",
                        target_object="cube",
                        parameters={},
                        prerequisites=["action_4"]
                    )
                ],
                dependency_graph={
                    "action_1": [],
                    "action_2": ["action_1"],
                    "action_3": ["action_2"],
                    "action_4": ["action_3"],
                    "action_5": ["action_4"]
                }
            )

            result1 = validator.validate(task1)
            print(f"\n{result1.get_summary()}")

            # Example task 2: Out of reach position
            print("\n" + "-"*60)
            print("Example 2: Invalid Task - Position Out of Reach")
            print("-"*60)

            task2 = TaskPlan(
                task_name="unreachable_task",
                description="Try to reach a position outside workspace",
                actions=[
                    AtomicAction(
                        id="action_1",
                        action_type="reach",
                        target_location="far_point",
                        parameters={"position": [5.0, 5.0, 2.0]},  # Way out of reach
                        prerequisites=[]
                    )
                ],
                dependency_graph={"action_1": []}
            )

            result2 = validator.validate(task2)
            print(f"\n{result2.get_summary()}")

            # Example task 3: Gripper compatibility
            print("\n" + "-"*60)
            print("Example 3: Gripper Compatibility Check")
            print("-"*60)

            task3 = TaskPlan(
                task_name="large_object_grasp",
                description="Try to grasp object larger than gripper",
                actions=[
                    AtomicAction(
                        id="action_1",
                        action_type="grasp",
                        target_object="large_box",
                        parameters={"object_size": 0.5},  # Very large object
                        prerequisites=[]
                    )
                ],
                dependency_graph={"action_1": []}
            )

            result3 = validator.validate(task3)
            print(f"\n{result3.get_summary()}")

        except Exception as e:
            print(f"Error testing {robot_type}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
