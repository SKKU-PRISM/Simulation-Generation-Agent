"""
Task Decomposer Module for Task Specification Agent

This module provides functionality to decompose high-level robot manipulation tasks
into sequences of atomic actions using LLM-based reasoning.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

from nl_parser import ParsedTask
from llm_client import BaseLLMClient, LLMConnectionError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AtomicAction:
    """
    Represents a single atomic action in a robot manipulation task.

    Each atomic action corresponds to a primitive operation that the robot
    can execute, such as reaching, grasping, lifting, or placing an object.
    """

    id: str
    """Unique identifier for this action (e.g., 'action_1')"""

    action_type: str
    """Type of action: reach, grasp, release, lift, place, open, close, push, pull"""

    target_object: Optional[str] = None
    """Name of the target object (if applicable)"""

    target_location: Optional[str] = None
    """Name of the target location (if applicable)"""

    parameters: Dict[str, Any] = field(default_factory=dict)
    """Additional parameters (speed, force, distance, direction, etc.)"""

    prerequisites: List[str] = field(default_factory=list)
    """List of action IDs that must complete before this action can start"""

    def __str__(self) -> str:
        target = self.target_object or self.target_location or "N/A"
        return (f"AtomicAction(id={self.id}, type={self.action_type}, "
                f"target={target}, prereqs={self.prerequisites})")

    def validate(self) -> bool:
        """
        Validate that the atomic action has all required fields.

        Returns:
            True if valid, False otherwise
        """
        # Check action type
        valid_types = ['reach', 'grasp', 'release', 'lift', 'place',
                       'open', 'close', 'push', 'pull']
        if self.action_type not in valid_types:
            logger.error(f"Invalid action type: {self.action_type}")
            return False

        # Check that at least one target is specified for most actions
        if self.action_type not in ['lift'] and not self.target_object and not self.target_location:
            logger.error(f"Action {self.id} ({self.action_type}) requires a target")
            return False

        return True


@dataclass
class TaskPlan:
    """
    Represents a complete task plan with sequenced atomic actions.

    A task plan contains the full decomposition of a high-level task into
    atomic actions, including their dependencies and execution order.
    """

    task_name: str
    """Name of the task"""

    description: str
    """Description of the task"""

    actions: List[AtomicAction] = field(default_factory=list)
    """List of atomic actions in execution order"""

    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    """
    Dependency graph mapping action IDs to their prerequisite action IDs.
    Format: {action_id: [prerequisite_id_1, prerequisite_id_2, ...]}
    """

    def __str__(self) -> str:
        return (f"TaskPlan(name={self.task_name}, "
                f"num_actions={len(self.actions)})")

    def validate(self) -> bool:
        """
        Validate the task plan structure and dependencies.

        Returns:
            True if valid, False otherwise
        """
        # Validate each action
        for action in self.actions:
            if not action.validate():
                logger.error(f"Invalid action: {action.id}")
                return False

        # Check that all action IDs are unique
        action_ids = [action.id for action in self.actions]
        if len(action_ids) != len(set(action_ids)):
            logger.error("Duplicate action IDs found in task plan")
            return False

        # Validate dependency graph
        for action_id, prereqs in self.dependency_graph.items():
            # Check that action_id exists
            if action_id not in action_ids:
                logger.error(f"Dependency graph references non-existent action: {action_id}")
                return False

            # Check that all prerequisites exist
            for prereq in prereqs:
                if prereq not in action_ids:
                    logger.error(f"Action {action_id} has non-existent prerequisite: {prereq}")
                    return False

        # Check for circular dependencies
        if self._has_circular_dependencies():
            logger.error("Circular dependencies detected in task plan")
            return False

        return True

    def _has_circular_dependencies(self) -> bool:
        """
        Check if the dependency graph contains circular dependencies.

        Returns:
            True if circular dependencies exist, False otherwise
        """
        visited = set()
        rec_stack = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Visit all prerequisites
            for prereq in self.dependency_graph.get(node, []):
                if prereq not in visited:
                    if dfs(prereq):
                        return True
                elif prereq in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        # Check each node
        for action_id in self.dependency_graph.keys():
            if action_id not in visited:
                if dfs(action_id):
                    return True

        return False

    def get_execution_order(self) -> List[str]:
        """
        Get the topologically sorted execution order of actions.

        Returns:
            List of action IDs in execution order

        Raises:
            ValueError: If the task plan has circular dependencies
        """
        if self._has_circular_dependencies():
            raise ValueError("Cannot compute execution order: circular dependencies detected")

        # Topological sort using Kahn's algorithm
        in_degree = {action.id: 0 for action in self.actions}
        for prereqs in self.dependency_graph.values():
            for prereq in prereqs:
                in_degree[prereq] = in_degree.get(prereq, 0)

        # Calculate in-degrees
        for action_id, prereqs in self.dependency_graph.items():
            in_degree[action_id] = len(prereqs)

        # Find all nodes with in-degree 0
        queue = [action_id for action_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            # Sort to ensure deterministic ordering
            queue.sort()
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree for dependent nodes
            for action_id, prereqs in self.dependency_graph.items():
                if node in prereqs:
                    in_degree[action_id] -= 1
                    if in_degree[action_id] == 0:
                        queue.append(action_id)

        return result


class TaskDecomposerChain:
    """
    Chain for decomposing high-level tasks into atomic actions using an LLM.

    This class uses a language model to analyze task descriptions and generate
    detailed action sequences for robot manipulation tasks.
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize the TaskDecomposerChain.

        Args:
            llm_client: An instance of BaseLLMClient for LLM interactions
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()
        logger.info("TaskDecomposerChain initialized")

    def _load_prompt_template(self) -> str:
        """
        Load the task decomposition prompt template.

        Returns:
            The prompt template as a string

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
        """
        # Find prompt file relative to this module
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "task_decomposition.md"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Task decomposition prompt not found: {prompt_path}")

        with open(prompt_path, 'r') as f:
            template = f.read()

        logger.info(f"Loaded task decomposition prompt from: {prompt_path}")
        return template

    def _prepare_task_input(self, parsed_task: ParsedTask) -> str:
        """
        Prepare the task input string for the prompt.

        Args:
            parsed_task: The parsed task to decompose

        Returns:
            Formatted task input string
        """
        # Format constraints as a readable string
        constraints_str = ""
        if parsed_task.constraints:
            for key, value in parsed_task.constraints.items():
                constraints_str += f"\n  - {key}: {value}"

        task_input = f"""
**Task Information:**
- Raw Description: {parsed_task.raw_text}
- Extracted Actions: {', '.join(parsed_task.actions) if parsed_task.actions else 'None'}
- Objects: {', '.join(parsed_task.objects) if parsed_task.objects else 'None'}
- Locations: {', '.join(parsed_task.locations) if parsed_task.locations else 'None'}
- Constraints: {constraints_str if constraints_str else 'None'}
"""
        return task_input.strip()

    def _create_output_schema(self) -> dict:
        """
        Create the JSON schema for structured output.

        Returns:
            JSON schema dictionary
        """
        return {
            "type": "object",
            "properties": {
                "task_name": {"type": "string"},
                "description": {"type": "string"},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "action_type": {
                                "type": "string",
                                "enum": ["reach", "grasp", "release", "lift", "place",
                                        "open", "close", "push", "pull"]
                            },
                            "target_object": {"type": ["string", "null"]},
                            "target_location": {"type": ["string", "null"]},
                            "parameters": {"type": "object"},
                            "prerequisites": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["id", "action_type", "prerequisites"]
                    }
                },
                "dependency_graph": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "required": ["task_name", "description", "actions", "dependency_graph"]
        }

    def decompose(self, parsed_task: ParsedTask) -> TaskPlan:
        """
        Decompose a parsed task into a sequence of atomic actions.

        This method uses the LLM to analyze the task and generate a detailed
        action plan with dependencies.

        Args:
            parsed_task: The parsed task to decompose

        Returns:
            TaskPlan containing the decomposed actions

        Raises:
            LLMConnectionError: If LLM invocation fails
            ValueError: If the generated plan is invalid
        """
        logger.info(f"Decomposing task: {parsed_task.raw_text[:50]}...")

        try:
            # Prepare the prompt
            task_input = self._prepare_task_input(parsed_task)
            prompt = self.prompt_template.replace("{task_input}", task_input)

            # Get structured response from LLM
            schema = self._create_output_schema()
            logger.debug("Invoking LLM for task decomposition...")
            response = self.llm_client.invoke_with_schema(prompt, schema)

            # Parse the response into TaskPlan
            task_plan = self._parse_response(response)

            # Validate the task plan
            if not task_plan.validate():
                raise ValueError("Generated task plan failed validation")

            logger.info(f"Successfully decomposed task into {len(task_plan.actions)} actions")
            return task_plan

        except LLMConnectionError as e:
            logger.error(f"LLM error during task decomposition: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during task decomposition: {str(e)}")
            raise ValueError(f"Failed to decompose task: {str(e)}")

    def _parse_response(self, response: dict) -> TaskPlan:
        """
        Parse the LLM response into a TaskPlan object.

        Args:
            response: Dictionary response from LLM

        Returns:
            TaskPlan object

        Raises:
            ValueError: If response structure is invalid
        """
        try:
            # Extract basic info
            task_name = response.get("task_name", "")
            description = response.get("description", "")

            # Parse actions
            actions = []
            for action_dict in response.get("actions", []):
                action = AtomicAction(
                    id=action_dict.get("id", ""),
                    action_type=action_dict.get("action_type", ""),
                    target_object=action_dict.get("target_object"),
                    target_location=action_dict.get("target_location"),
                    parameters=action_dict.get("parameters", {}),
                    prerequisites=action_dict.get("prerequisites", [])
                )
                actions.append(action)

            # Get dependency graph
            dependency_graph = response.get("dependency_graph", {})

            # Create TaskPlan
            task_plan = TaskPlan(
                task_name=task_name,
                description=description,
                actions=actions,
                dependency_graph=dependency_graph
            )

            return task_plan

        except Exception as e:
            raise ValueError(f"Failed to parse LLM response: {str(e)}")


if __name__ == "__main__":
    """
    Example usage and testing of the task decomposer.
    """
    import sys
    from llm_client import get_llm_client

    # Create LLM client
    provider = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        print(f"Creating LLM client (provider: {provider or 'default'})...")
        client = get_llm_client(provider=provider)

        print("Testing LLM connection...")
        if not client.test_connection():
            print("Connection test failed!")
            sys.exit(1)

        # Create decomposer
        print("\nInitializing TaskDecomposerChain...")
        decomposer = TaskDecomposerChain(client)

        # Example task 1: Pick and Place
        print("\n" + "="*60)
        print("Example 1: Pick and Place Task")
        print("="*60)

        task1 = ParsedTask(
            raw_text="Pick a cube from the table and place it on a target marker",
            actions=["pick", "place"],
            objects=["cube", "table", "target_marker"],
            locations=["on the table", "on the target marker"],
            constraints={"tolerance": "3cm", "robot": "UR10"}
        )

        print(f"\nInput: {task1}")
        plan1 = decomposer.decompose(task1)
        print(f"\nResult: {plan1}")
        print(f"\nActions ({len(plan1.actions)}):")
        for action in plan1.actions:
            print(f"  - {action}")
        print(f"\nExecution Order: {plan1.get_execution_order()}")

        # Example task 2: Open Drawer
        print("\n" + "="*60)
        print("Example 2: Open Drawer Task")
        print("="*60)

        task2 = ParsedTask(
            raw_text="Open the top drawer of a cabinet by pulling the handle",
            actions=["open", "pull"],
            objects=["cabinet", "drawer", "drawer_handle_top"],
            locations=["top drawer"],
            constraints={"opening_distance": "0.2m", "robot": "UR10"}
        )

        print(f"\nInput: {task2}")
        plan2 = decomposer.decompose(task2)
        print(f"\nResult: {plan2}")
        print(f"\nActions ({len(plan2.actions)}):")
        for action in plan2.actions:
            print(f"  - {action}")
        print(f"\nExecution Order: {plan2.get_execution_order()}")

        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
