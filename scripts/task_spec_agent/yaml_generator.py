"""
YAML Generator Module for Task Specification Agent

This module provides functionality to generate Isaac Sim task specification YAML files
from task plans and validation results. It uses robot-specific templates and asset mappings
to create properly formatted YAML configurations compatible with the existing task framework.
"""

import logging
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from task_decomposer import TaskPlan, AtomicAction
from feasibility_validator import ValidationResult


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YAMLGenerator:
    """
    Generator for creating Isaac Sim task YAML specifications.

    This class takes a TaskPlan and ValidationResult and generates a complete
    YAML task specification file compatible with the existing Isaac Sim task framework.
    """

    def __init__(self, robot_type: str = "franka"):
        """
        Initialize the YAMLGenerator.

        Args:
            robot_type: Type of robot ("franka", "openarm", or "ur10")

        Raises:
            ValueError: If robot_type is not supported
        """
        supported_robots = ["franka", "openarm", "ur10"]
        if robot_type not in supported_robots:
            raise ValueError(f"Unsupported robot type: {robot_type}. "
                           f"Supported types: {supported_robots}")

        self.robot_type = robot_type
        self.template = self._load_template(robot_type)
        self.asset_mappings = self._load_asset_mappings()
        logger.info(f"YAMLGenerator initialized for robot: {robot_type}")

    def _load_template(self, robot_type: str) -> dict:
        """
        Load the robot-specific YAML template.

        Args:
            robot_type: Type of robot

        Returns:
            Template dictionary

        Raises:
            FileNotFoundError: If template file doesn't exist
        """
        # Find template file relative to project root
        template_path = (Path(__file__).parent.parent.parent /
                        "configs" / "task_templates" / f"{robot_type}_template.yaml")

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r') as f:
            template = yaml.safe_load(f)

        logger.info(f"Loaded template from: {template_path}")
        return template

    def _load_asset_mappings(self) -> dict:
        """
        Load the asset mappings configuration.

        Returns:
            Asset mappings dictionary

        Raises:
            FileNotFoundError: If asset mappings file doesn't exist
        """
        mappings_path = (Path(__file__).parent.parent.parent /
                        "configs" / "asset_mappings.yaml")

        if not mappings_path.exists():
            raise FileNotFoundError(f"Asset mappings not found: {mappings_path}")

        with open(mappings_path, 'r') as f:
            mappings = yaml.safe_load(f)

        logger.info(f"Loaded asset mappings from: {mappings_path}")
        return mappings

    def _map_objects_to_assets(self, objects: List[str]) -> List[dict]:
        """
        Map object names to asset configurations.

        Args:
            objects: List of object names from task plan

        Returns:
            List of asset dictionaries with full configuration
        """
        assets = []
        object_mappings = self.asset_mappings.get('objects', {})

        for i, obj_name in enumerate(objects):
            # Normalize object name
            obj_key = obj_name.lower().replace(' ', '_').replace('-', '_')

            # Check if object exists in mappings
            if obj_key in object_mappings:
                mapping = object_mappings[obj_key]
                asset = self._create_asset_from_mapping(obj_name, mapping, i)
            else:
                # Create default primitive cube for unmapped objects
                logger.warning(f"Object '{obj_name}' not in asset mappings, using default cube")
                asset = self._create_default_asset(obj_name, i)

            assets.append(asset)

        return assets

    def _create_asset_from_mapping(self, obj_name: str, mapping: dict, index: int) -> dict:
        """
        Create an asset configuration from a mapping entry.

        Args:
            obj_name: Name of the object
            mapping: Mapping configuration
            index: Index for unique naming

        Returns:
            Asset configuration dictionary
        """
        # Base asset configuration
        asset = {
            'name': obj_name.lower().replace(' ', '_'),
            'type': 'rigid' if mapping.get('physics', {}).get('rigid_body', True) else 'static',
            'source': mapping['source'],
            'prim_path': f"/World/{obj_name.replace(' ', '')}_{index}"
        }

        # Add source-specific fields
        if mapping['source'] == 'usd':
            asset['asset_path'] = mapping['asset_path']
        elif mapping['source'] == 'primitive':
            asset['primitive'] = mapping['primitive']

        # Add position (default to table surface)
        asset['position'] = [0.5, 0.0, 0.055]

        # Add rotation
        asset['rotation'] = [1, 0, 0, 0]

        # Add scale
        asset['scale'] = mapping.get('scale', [1.0, 1.0, 1.0])

        # Add color if specified
        if 'color' in mapping:
            asset['color'] = mapping['color']

        # Add physics properties
        if 'physics' in mapping:
            asset['physics'] = mapping['physics'].copy()

        return asset

    def _create_default_asset(self, obj_name: str, index: int) -> dict:
        """
        Create a default asset configuration for unmapped objects.

        Args:
            obj_name: Name of the object
            index: Index for unique naming

        Returns:
            Default asset configuration dictionary
        """
        return {
            'name': obj_name.lower().replace(' ', '_'),
            'type': 'rigid',
            'source': 'primitive',
            'primitive': 'cube',
            'prim_path': f"/World/{obj_name.replace(' ', '')}_{index}",
            'position': [0.5, 0.0, 0.055],
            'rotation': [1, 0, 0, 0],
            'scale': [0.05, 0.05, 0.05],
            'physics': {
                'rigid_body': True,
                'collision': True
            }
        }

    def _extract_objects_from_actions(self, actions: List[AtomicAction]) -> List[str]:
        """
        Extract unique object names from action list.

        Args:
            actions: List of atomic actions

        Returns:
            List of unique object names
        """
        objects = set()
        for action in actions:
            if action.target_object:
                objects.add(action.target_object)
            if action.target_location:
                # Extract object name from location (e.g., "on blue_block" → "blue_block")
                loc = action.target_location
                for prefix in ["on ", "above ", "near ", "at ", "inside ", "onto "]:
                    if loc.lower().startswith(prefix):
                        loc = loc[len(prefix):]
                        break
                loc = loc.strip()
                if loc and not loc.replace("_", "").replace("-", "").replace(".", "").replace(" ", "").isdigit():
                    objects.add(loc)
        return list(objects)

    def _generate_goal_conditions(self, task_plan: TaskPlan) -> List[dict]:
        """
        Generate goal conditions based on task plan actions.

        Args:
            task_plan: The task plan to analyze

        Returns:
            List of goal condition dictionaries
        """
        conditions = []

        # Analyze actions to determine goal type
        action_types = [action.action_type for action in task_plan.actions]

        # Check for pick and place task
        if 'grasp' in action_types and 'place' in action_types:
            # Find the target object and location
            place_actions = [a for a in task_plan.actions if a.action_type == 'place']
            if place_actions:
                place_action = place_actions[-1]  # Use last place action
                if place_action.target_object and place_action.target_location:
                    conditions.append({
                        'subject': place_action.target_object,
                        'relation': 'at_position',
                        'target': place_action.target_location,
                        'tolerance': 0.03
                    })

        # Check for reach task
        elif 'reach' in action_types:
            reach_actions = [a for a in task_plan.actions if a.action_type == 'reach']
            if reach_actions:
                reach_action = reach_actions[-1]
                if reach_action.target_location:
                    conditions.append({
                        'subject': 'robot_ee',
                        'relation': 'at_position',
                        'target': reach_action.target_location,
                        'tolerance': 0.02
                    })

        # Check for open/close tasks
        elif 'open' in action_types or 'close' in action_types:
            open_actions = [a for a in task_plan.actions if a.action_type in ['open', 'close']]
            if open_actions:
                action = open_actions[-1]
                if action.target_object:
                    state = 'open' if action.action_type == 'open' else 'closed'
                    conditions.append({
                        'subject': action.target_object,
                        'relation': 'state',
                        'target': state
                    })

        # Default condition if nothing specific found
        if not conditions:
            conditions.append({
                'subject': 'task',
                'relation': 'completed',
                'target': 'success'
            })

        return conditions

    def _determine_episode_length(self, task_plan: TaskPlan) -> float:
        """
        Determine appropriate episode length based on task complexity.

        Args:
            task_plan: The task plan to analyze

        Returns:
            Episode length in seconds
        """
        # Base time per action type
        base_time = {
            'reach': 2.0,
            'grasp': 2.0,
            'release': 1.0,
            'lift': 2.0,
            'place': 2.0,
            'open': 3.0,
            'close': 3.0,
            'push': 2.0,
            'pull': 2.0
        }

        # Calculate total time needed
        total_time = sum(base_time.get(action.action_type, 2.0)
                        for action in task_plan.actions)

        # Add buffer (50%)
        total_time *= 1.5

        # Apply robot-specific multipliers
        robot_multipliers = {
            'franka': 1.0,
            'openarm': 1.5,  # OpenArm is slower
            'ur10': 1.2
        }
        total_time *= robot_multipliers.get(self.robot_type, 1.0)

        # Clamp to reasonable range
        return max(5.0, min(total_time, 30.0))

    def generate(self, task_plan: TaskPlan, validation_result: Optional[ValidationResult] = None) -> str:
        """
        Generate a complete YAML task specification.

        Args:
            task_plan: The task plan to convert to YAML
            validation_result: Optional validation result (for metadata/warnings)

        Returns:
            YAML string with complete task specification

        Raises:
            ValueError: If task plan is invalid
        """
        if not task_plan.validate():
            raise ValueError("Task plan validation failed")

        logger.info(f"Generating YAML for task: {task_plan.task_name}")

        # Start with template
        spec = self.template.copy()

        # Update task information
        spec['task']['name'] = task_plan.task_name
        spec['task']['description'] = task_plan.description

        # Update episode length based on task complexity
        episode_length = self._determine_episode_length(task_plan)
        spec['simulation']['episode_length'] = episode_length

        # Extract objects from actions
        objects = self._extract_objects_from_actions(task_plan.actions)

        # Map objects to assets
        object_assets = self._map_objects_to_assets(objects)

        # Add object assets to the spec (after robot and table)
        spec['assets'].extend(object_assets)

        # Generate goal conditions
        conditions = self._generate_goal_conditions(task_plan)
        spec['goal']['description'] = task_plan.description
        spec['goal']['conditions'] = conditions

        # Add validation metadata if provided
        if validation_result:
            spec['notes'] = [
                f"Generated from TaskPlan with {len(task_plan.actions)} actions",
                f"Validation result: {'VALID' if validation_result.is_valid else 'INVALID'}",
                f"Errors: {len(validation_result.errors)}, Warnings: {len(validation_result.warnings)}"
            ]

            # Add warnings if any
            if validation_result.errors:
                spec['notes'].append(
                    f"WARNING: {len(validation_result.errors)} validation errors detected"
                )

        # Convert to YAML string
        yaml_str = yaml.dump(spec, default_flow_style=False, sort_keys=False,
                            allow_unicode=True, width=100)

        logger.info(f"YAML generation complete ({len(yaml_str)} characters)")
        return yaml_str

    def save_to_file(self, yaml_content: str, file_path: str) -> None:
        """
        Save YAML content to a file.

        Args:
            yaml_content: YAML string to save
            file_path: Path where to save the file

        Raises:
            IOError: If file cannot be written
        """
        output_path = Path(file_path)

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        with open(output_path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"YAML saved to: {output_path}")


if __name__ == "__main__":
    """
    Example usage and testing of the YAML generator.
    """
    import sys
    from task_decomposer import AtomicAction, TaskPlan
    from feasibility_validator import ValidationResult, ValidationIssue

    print("="*60)
    print("Testing YAMLGenerator")
    print("="*60)

    # Create a sample pick and place task plan
    actions = [
        AtomicAction(
            id="action_1",
            action_type="reach",
            target_object="cube",
            prerequisites=[]
        ),
        AtomicAction(
            id="action_2",
            action_type="grasp",
            target_object="cube",
            prerequisites=["action_1"]
        ),
        AtomicAction(
            id="action_3",
            action_type="lift",
            target_object="cube",
            prerequisites=["action_2"]
        ),
        AtomicAction(
            id="action_4",
            action_type="place",
            target_object="cube",
            target_location="target_marker",
            prerequisites=["action_3"]
        ),
        AtomicAction(
            id="action_5",
            action_type="release",
            target_object="cube",
            prerequisites=["action_4"]
        )
    ]

    task_plan = TaskPlan(
        task_name="PickAndPlace",
        description="Pick a cube from the table and place it on a target marker",
        actions=actions,
        dependency_graph={
            "action_1": [],
            "action_2": ["action_1"],
            "action_3": ["action_2"],
            "action_4": ["action_3"],
            "action_5": ["action_4"]
        }
    )

    # Create sample validation result
    validation_result = ValidationResult(
        is_valid=True,
        errors=[],
        warnings=[]
    )

    # Test with different robot types
    robot_types = ["franka", "openarm", "ur10"]

    for robot_type in robot_types:
        print(f"\n{'-'*60}")
        print(f"Testing with robot: {robot_type}")
        print(f"{'-'*60}")

        try:
            # Create generator
            generator = YAMLGenerator(robot_type=robot_type)

            # Generate YAML
            yaml_content = generator.generate(task_plan, validation_result)

            # Print first 50 lines
            lines = yaml_content.split('\n')
            print("\nGenerated YAML (first 50 lines):")
            print('\n'.join(lines[:50]))
            if len(lines) > 50:
                print(f"... ({len(lines) - 50} more lines)")

            # Test save functionality
            output_dir = Path(__file__).parent / "test_output"
            output_file = output_dir / f"test_{robot_type}_pickplace.yaml"
            generator.save_to_file(yaml_content, str(output_file))
            print(f"\nSaved to: {output_file}")

        except Exception as e:
            print(f"Error testing {robot_type}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
