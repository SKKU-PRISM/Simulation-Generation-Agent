"""
Natural Language Parser Module for Task Specification Agent

This module provides natural language parsing capabilities for robot task descriptions.
It converts free-form text descriptions into structured ParsedTask objects that can be
used for task specification and scene generation.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

from llm_client import BaseLLMClient, get_llm_client, LLMConnectionError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ParsedTask:
    """
    Data class representing a parsed robot task.

    This structure holds the extracted components from a natural language
    task description, including actions, objects, locations, and constraints.

    Attributes:
        actions: List of actions/verbs extracted from the task (e.g., "grasp", "place", "open")
        objects: List of objects mentioned in the task (e.g., "drawer", "cup", "table")
        locations: List of spatial references and locations (e.g., "on the table", "inside drawer")
        constraints: Dictionary of constraints and additional semantic information
        raw_text: The original input text
    """
    actions: List[str] = field(default_factory=list)
    objects: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""

    def __str__(self) -> str:
        """Pretty string representation of the parsed task."""
        return (
            f"ParsedTask(\n"
            f"  raw_text: {self.raw_text}\n"
            f"  actions: {self.actions}\n"
            f"  objects: {self.objects}\n"
            f"  locations: {self.locations}\n"
            f"  constraints: {self.constraints}\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ParsedTask to dictionary format."""
        return {
            "actions": self.actions,
            "objects": self.objects,
            "locations": self.locations,
            "constraints": self.constraints,
            "raw_text": self.raw_text
        }


class NLParserChain:
    """
    Natural Language Parser Chain for robot task descriptions.

    This class uses an LLM to parse natural language descriptions of robot tasks
    into structured components (actions, objects, locations, constraints).
    It supports both English and Korean inputs.

    Example:
        >>> llm_client = get_llm_client()
        >>> parser = NLParserChain(llm_client)
        >>> parsed = parser.parse("서랍에서 물건을 꺼내 테이블에 놓아라")
        >>> print(parsed.actions)
        ['open', 'grasp', 'pick', 'place']
    """

    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize the NLParserChain with an LLM client.

        Args:
            llm_client: An instance of BaseLLMClient for LLM interactions
        """
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()
        logger.info("NLParserChain initialized successfully")

    def _load_prompt_template(self) -> str:
        """
        Load the task parsing prompt template from the prompts directory.

        Returns:
            The prompt template as a string

        Raises:
            FileNotFoundError: If the prompt template file doesn't exist
        """
        # Get the path to the prompt template
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / "task_parsing.md"

        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        # Load the prompt template
        with open(prompt_path, 'r', encoding='utf-8') as f:
            template = f.read()

        logger.info(f"Loaded prompt template from: {prompt_path}")
        return template

    def parse(self, text: str) -> ParsedTask:
        """
        Parse a natural language task description into structured components.

        This method takes a free-form text description of a robot task and uses
        an LLM to extract structured information including actions, objects,
        locations, and constraints.

        Args:
            text: Natural language description of the robot task (English or Korean)

        Returns:
            ParsedTask object containing extracted components

        Raises:
            LLMConnectionError: If the LLM call fails
            ValueError: If the LLM response cannot be parsed

        Example:
            >>> parser = NLParserChain(llm_client)
            >>> result = parser.parse("Stack the red block on top of the blue block")
            >>> print(result.actions)
            ['grasp', 'pick', 'place', 'stack']
        """
        logger.info(f"Parsing task description: {text}")

        # Construct the full prompt
        full_prompt = f"{self.prompt_template}\n\n## Task Description to Parse\n\n{text}"

        # Define the expected output schema
        output_schema = {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of actions/verbs in base form"
                },
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of physical objects mentioned"
                },
                "locations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of spatial references and locations"
                },
                "constraints": {
                    "type": "object",
                    "description": "Dictionary of constraints and semantic information"
                },
                "raw_text": {
                    "type": "string",
                    "description": "The original input text"
                }
            },
            "required": ["actions", "objects", "locations", "constraints", "raw_text"]
        }

        try:
            # Call the LLM with structured output
            response = self.llm_client.invoke_with_schema(full_prompt, output_schema)

            # Create ParsedTask from the response
            parsed_task = ParsedTask(
                actions=response.get("actions", []),
                objects=response.get("objects", []),
                locations=response.get("locations", []),
                constraints=response.get("constraints", {}),
                raw_text=response.get("raw_text", text)
            )

            logger.info(f"Successfully parsed task. Found {len(parsed_task.actions)} actions, "
                       f"{len(parsed_task.objects)} objects, {len(parsed_task.locations)} locations")
            logger.debug(f"Parsed result: {parsed_task}")

            return parsed_task

        except LLMConnectionError as e:
            logger.error(f"LLM connection error during parsing: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing required field in LLM response: {e}")
            raise ValueError(f"Invalid LLM response format: missing field {e}")
        except Exception as e:
            logger.error(f"Unexpected error during parsing: {e}")
            raise ValueError(f"Failed to parse task description: {e}")


if __name__ == "__main__":
    """
    Test the NLParserChain with example inputs in Korean and English.
    """
    import sys
    import json

    print("=" * 80)
    print("Natural Language Parser Chain - Test Suite")
    print("=" * 80)

    # Initialize LLM client
    try:
        print("\n1. Initializing LLM client...")
        provider = sys.argv[1] if len(sys.argv) > 1 else None
        llm_client = get_llm_client(provider=provider)

        print("2. Testing LLM connection...")
        if not llm_client.test_connection():
            print("ERROR: LLM connection test failed!")
            sys.exit(1)
        print("   Connection test passed!")

    except Exception as e:
        print(f"ERROR: Failed to initialize LLM client: {e}")
        sys.exit(1)

    # Initialize parser
    try:
        print("\n3. Initializing NLParserChain...")
        parser = NLParserChain(llm_client)
        print("   Parser initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize parser: {e}")
        sys.exit(1)

    # Test cases
    test_cases = [
        {
            "name": "Korean - Drawer to Table",
            "text": "서랍에서 물건을 꺼내 테이블에 놓아라"
        },
        {
            "name": "Korean - Stacking Blocks",
            "text": "빨간 블록을 파란 블록 위에 쌓아라"
        },
        {
            "name": "English - Cabinet Task",
            "text": "Open the cabinet drawer, grasp the red cup, and place it carefully on the table"
        },
        {
            "name": "English - Simple Stacking",
            "text": "Stack the red block on top of the blue block"
        },
        {
            "name": "English - Pick and Place",
            "text": "Pick up the gear from the table and place it on the mounting post"
        },
        {
            "name": "Korean - Reaching Task",
            "text": "로봇을 테이블 위의 목표 지점으로 이동시켜라"
        }
    ]

    # Run test cases
    print("\n" + "=" * 80)
    print("Running Test Cases")
    print("=" * 80)

    successful_tests = 0
    failed_tests = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        print(f"Input: {test_case['text']}")
        print()

        try:
            # Parse the task
            parsed = parser.parse(test_case['text'])

            # Display results
            print("Parsed Result:")
            print(f"  Actions: {parsed.actions}")
            print(f"  Objects: {parsed.objects}")
            print(f"  Locations: {parsed.locations}")
            print(f"  Constraints: {json.dumps(parsed.constraints, indent=4, ensure_ascii=False)}")
            print(f"  Raw Text: {parsed.raw_text}")

            successful_tests += 1
            print("\n✓ Test passed!")

        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            failed_tests += 1

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Total tests: {len(test_cases)}")
    print(f"Passed: {successful_tests}")
    print(f"Failed: {failed_tests}")

    if failed_tests == 0:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print(f"\n⚠ {failed_tests} test(s) failed.")
        sys.exit(1)
