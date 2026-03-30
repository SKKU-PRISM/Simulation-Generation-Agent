"""
Task Specification Agent - Integrated Pipeline and CLI

This module provides a complete end-to-end pipeline for converting natural language
task descriptions into validated YAML task specifications for Isaac Sim robotics tasks.

Pipeline:
    Natural Language Input
    -> NLParser (structured task extraction)
    -> TaskDecomposer (atomic action planning)
    -> FeasibilityValidator (physical validation)
    -> YAMLGenerator (YAML specification output)

Usage:
    # Python API
    agent = TaskSpecAgent(robot_type="franka")
    result = agent.process("Pick up the cube and place it on the target")

    # CLI
    python3 task_spec_agent.py "Pick up the cube" --robot franka --output task.yaml
"""

import os
import sys
import logging
import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Import all pipeline components
from llm_client import get_llm_client, LLMConnectionError
from nl_parser import NLParserChain, ParsedTask
from task_decomposer import TaskDecomposerChain, TaskPlan
from feasibility_validator import FeasibilityValidator, ValidationResult
from yaml_generator import YAMLGenerator
from rag_match_yaml_generator import RAGYAMLGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TaskSpecResult:
    """
    Complete result from the task specification pipeline.

    This data class contains the outputs from all stages of processing,
    including any errors encountered along the way. Even if processing fails
    partway through, partial results are preserved for debugging.

    Attributes:
        success: Whether the complete pipeline succeeded
        yaml_content: Generated YAML specification (if successful)
        yaml_path: Path where YAML was saved (if saved)
        parsed_task: Result from natural language parsing stage
        task_plan: Result from task decomposition stage
        validation_result: Result from feasibility validation stage
        errors: List of error messages from any stage
        processing_time: Total time taken to process (seconds)
    """

    success: bool
    """Whether the complete pipeline succeeded"""

    yaml_content: Optional[str] = None
    """Generated YAML specification content"""

    yaml_path: Optional[str] = None
    """Path where YAML file was saved"""

    parsed_task: Optional[ParsedTask] = None
    """Parsed natural language task"""

    task_plan: Optional[TaskPlan] = None
    """Decomposed task plan with atomic actions"""

    validation_result: Optional[ValidationResult] = None
    """Physical feasibility validation result"""

    errors: List[str] = field(default_factory=list)
    """List of error messages from any stage"""

    processing_time: float = 0.0
    """Total processing time in seconds"""

    def __str__(self) -> str:
        """Human-readable summary of the result."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"TaskSpecResult: {status}",
            f"  Processing time: {self.processing_time:.2f}s"
        ]

        if self.parsed_task:
            lines.append(f"  Parsed task: {len(self.parsed_task.actions)} actions, "
                        f"{len(self.parsed_task.objects)} objects")

        if self.task_plan:
            lines.append(f"  Task plan: {len(self.task_plan.actions)} atomic actions")

        if self.validation_result:
            valid_status = "VALID" if self.validation_result.is_valid else "INVALID"
            lines.append(f"  Validation: {valid_status} "
                        f"({len(self.validation_result.errors)} errors, "
                        f"{len(self.validation_result.warnings)} warnings)")

        if self.yaml_content:
            lines.append(f"  YAML: {len(self.yaml_content)} characters")

        if self.yaml_path:
            lines.append(f"  Saved to: {self.yaml_path}")

        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for error in self.errors:
                lines.append(f"    - {error}")

        return "\n".join(lines)

    def get_summary(self) -> str:
        """Get a concise summary suitable for printing."""
        return str(self)


class TaskSpecAgent:
    """
    End-to-end task specification agent.

    This class orchestrates the complete pipeline from natural language input
    to validated YAML task specification. It manages all intermediate stages
    and provides comprehensive error handling and logging.

    Example:
        >>> agent = TaskSpecAgent(robot_type="franka")
        >>> result = agent.process("Pick up the red cube and place it on the blue block")
        >>> if result.success:
        ...     print(f"YAML saved to: {result.yaml_path}")
        >>> else:
        ...     print(f"Errors: {result.errors}")
    """

    def __init__(self, robot_type: str = "franka", llm_provider: Optional[str] = None,
                 use_rag: bool = True):
        """
        Initialize the TaskSpecAgent.

        Args:
            robot_type: Type of robot ("franka", "openarm", "ur10", "so101")
            llm_provider: LLM provider to use ("azure", "huggingface", None for default)
            use_rag: Use RAG-based YAML generation (True) or template-based (False)

        Raises:
            ValueError: If robot_type is not supported
            LLMConnectionError: If LLM client initialization fails
        """
        self.robot_type = robot_type
        self.llm_provider = llm_provider
        self.use_rag = use_rag

        logger.info(f"Initializing TaskSpecAgent for robot: {robot_type}")
        logger.info(f"LLM provider: {llm_provider or 'default'}")
        logger.info(f"YAML generation: {'RAG-based' if use_rag else 'template-based'}")

        # Validate robot type
        supported_robots = ["franka", "openarm", "ur10", "so101"]
        if robot_type not in supported_robots:
            raise ValueError(f"Unsupported robot type: {robot_type}. "
                           f"Supported types: {supported_robots}")

        # Initialize LLM client
        try:
            logger.info("Initializing LLM client...")
            self.llm_client = get_llm_client(provider=llm_provider)
            logger.info("LLM client initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize LLM client: {str(e)}"
            logger.error(error_msg)
            raise LLMConnectionError(error_msg)

        # Initialize pipeline components
        try:
            logger.info("Initializing pipeline components...")

            self.nl_parser = NLParserChain(self.llm_client)
            logger.info("  - NLParserChain initialized")

            self.task_decomposer = TaskDecomposerChain(self.llm_client)
            logger.info("  - TaskDecomposerChain initialized")

            self.feasibility_validator = FeasibilityValidator(robot_type=robot_type)
            logger.info("  - FeasibilityValidator initialized")

            if use_rag:
                self.rag_generator = RAGYAMLGenerator(robot_type=robot_type)
                logger.info("  - RAGYAMLGenerator initialized")
            else:
                self.yaml_generator = YAMLGenerator(robot_type=robot_type)
                logger.info("  - YAMLGenerator (template) initialized")

            logger.info("All pipeline components initialized successfully")

        except Exception as e:
            error_msg = f"Failed to initialize pipeline components: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    def process(self, natural_language: str, output_path: Optional[str] = None) -> TaskSpecResult:
        """
        Process a natural language task description through the complete pipeline.

        This method executes all stages of the pipeline in sequence:
        1. Natural language parsing
        2. Task decomposition
        3. Feasibility validation
        4. YAML generation
        5. Optional file saving

        If any stage fails, processing stops and returns a result with partial
        outputs and error messages.

        Args:
            natural_language: Natural language task description
            output_path: Optional path to save the generated YAML file

        Returns:
            TaskSpecResult containing all outputs and any errors

        Example:
            >>> agent = TaskSpecAgent("franka")
            >>> result = agent.process("Pick the cube", output_path="task.yaml")
            >>> print(result.get_summary())
        """
        start_time = time.time()
        result = TaskSpecResult(success=False)

        logger.info("="*80)
        logger.info("Starting Task Specification Pipeline")
        logger.info("="*80)
        logger.info(f"Input: {natural_language}")
        logger.info(f"Robot: {self.robot_type}")

        try:
            # Stage 1: Natural Language Parsing
            logger.info("\n" + "-"*80)
            logger.info("Stage 1: Natural Language Parsing")
            logger.info("-"*80)

            try:
                parsed_task = self.nl_parser.parse(natural_language)
                result.parsed_task = parsed_task

                logger.info(f"Parsing complete:")
                logger.info(f"  Actions: {parsed_task.actions}")
                logger.info(f"  Objects: {parsed_task.objects}")
                logger.info(f"  Locations: {parsed_task.locations}")

            except Exception as e:
                error_msg = f"Natural language parsing failed: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.processing_time = time.time() - start_time
                return result

            # Stage 2: Task Decomposition
            logger.info("\n" + "-"*80)
            logger.info("Stage 2: Task Decomposition")
            logger.info("-"*80)

            try:
                task_plan = self.task_decomposer.decompose(parsed_task)
                result.task_plan = task_plan

                logger.info(f"Decomposition complete:")
                logger.info(f"  Task name: {task_plan.task_name}")
                logger.info(f"  Number of actions: {len(task_plan.actions)}")
                logger.info(f"  Action sequence:")
                for action in task_plan.actions:
                    logger.info(f"    - {action}")

            except Exception as e:
                error_msg = f"Task decomposition failed: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.processing_time = time.time() - start_time
                return result

            # Stage 3: Feasibility Validation
            logger.info("\n" + "-"*80)
            logger.info("Stage 3: Feasibility Validation")
            logger.info("-"*80)

            try:
                validation_result = self.feasibility_validator.validate(task_plan)
                result.validation_result = validation_result

                logger.info(f"Validation complete:")
                logger.info(f"  Valid: {validation_result.is_valid}")
                logger.info(f"  Errors: {len(validation_result.errors)}")
                logger.info(f"  Warnings: {len(validation_result.warnings)}")

                if validation_result.errors:
                    logger.warning("Validation errors found:")
                    for error in validation_result.errors:
                        logger.warning(f"  - {error}")

                if validation_result.warnings:
                    logger.info("Validation warnings:")
                    for warning in validation_result.warnings:
                        logger.info(f"  - {warning}")

                # Add validation errors to result
                if not validation_result.is_valid:
                    for error in validation_result.errors:
                        result.errors.append(f"Validation: {str(error)}")

            except Exception as e:
                error_msg = f"Feasibility validation failed: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.processing_time = time.time() - start_time
                return result

            # Stage 4: YAML Generation
            logger.info("\n" + "-"*80)
            logger.info(f"Stage 4: YAML Generation ({'RAG-based' if self.use_rag else 'template-based'})")
            logger.info("-"*80)

            try:
                if self.use_rag:
                    yaml_content = self.rag_generator.generate(
                        task_description=natural_language,
                        parsed_task=parsed_task,
                        task_plan=task_plan,
                        validation_result=validation_result,
                    )
                else:
                    yaml_content = self.yaml_generator.generate(task_plan, validation_result)

                result.yaml_content = yaml_content

                logger.info(f"YAML generation complete:")
                logger.info(f"  Size: {len(yaml_content)} characters")
                logger.info(f"  Lines: {len(yaml_content.splitlines())}")

            except Exception as e:
                error_msg = f"YAML generation failed: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.processing_time = time.time() - start_time
                return result

            # Stage 5: Save to file (if output path provided)
            if output_path:
                logger.info("\n" + "-"*80)
                logger.info("Stage 5: Saving YAML to File")
                logger.info("-"*80)

                try:
                    if self.use_rag:
                        self.rag_generator.save(yaml_content, output_path)
                    else:
                        self.yaml_generator.save_to_file(yaml_content, output_path)
                    result.yaml_path = output_path
                    logger.info(f"YAML saved to: {output_path}")

                except Exception as e:
                    error_msg = f"Failed to save YAML file: {str(e)}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    result.processing_time = time.time() - start_time
                    return result

            # Success!
            result.success = True
            result.processing_time = time.time() - start_time

            logger.info("\n" + "="*80)
            logger.info("Pipeline Complete - SUCCESS")
            logger.info("="*80)
            logger.info(f"Total processing time: {result.processing_time:.2f}s")

            return result

        except Exception as e:
            # Catch-all for unexpected errors
            error_msg = f"Unexpected error in pipeline: {str(e)}"
            logger.error(error_msg)
            import traceback
            logger.error(traceback.format_exc())
            result.errors.append(error_msg)
            result.processing_time = time.time() - start_time
            return result


def create_cli_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Task Specification Agent - Convert natural language to YAML task specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with Franka robot
  python3 task_spec_agent.py "Pick up the cube and place it on the target"

  # Specify robot type and output file
  python3 task_spec_agent.py "Open the drawer" --robot ur10 --output drawer_task.yaml

  # Use different LLM provider
  python3 task_spec_agent.py "Stack blocks" --provider huggingface

  # Enable verbose logging
  python3 task_spec_agent.py "Reach the goal" --robot openarm --verbose
        """
    )

    # Positional argument: task description
    parser.add_argument(
        "task",
        type=str,
        help="Natural language task description (e.g., 'Pick up the cube')"
    )

    # Optional arguments
    parser.add_argument(
        "--robot",
        type=str,
        default="franka",
        choices=["franka", "openarm", "ur10", "so101"],
        help="Robot type (default: franka)"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output YAML file path (default: print to stdout)"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["azure", "huggingface", "bedrock"],
        help="LLM provider (default: from config)"
    )

    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Use template-based YAML generation instead of RAG"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )

    return parser


def main():
    """
    Main CLI entry point.

    Parses command-line arguments and executes the task specification pipeline.
    Exits with code 0 on success, 1 on failure.
    """
    # Parse arguments
    parser = create_cli_parser()
    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    # Print header
    print("="*80)
    print("Task Specification Agent - CLI")
    print("="*80)
    print(f"Task: {args.task}")
    print(f"Robot: {args.robot}")
    print(f"LLM Provider: {args.provider or 'default'}")
    if args.output:
        print(f"Output: {args.output}")
    print("="*80)

    try:
        # Initialize agent
        print("\nInitializing agent...")
        agent = TaskSpecAgent(robot_type=args.robot, llm_provider=args.provider,
                              use_rag=not args.no_rag)
        print("Agent initialized successfully!\n")

        # Process task
        print("Processing task through pipeline...\n")
        result = agent.process(args.task, output_path=args.output)

        # Print results
        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(result.get_summary())
        print("="*80)

        # Print YAML content if no output file specified
        if result.success and result.yaml_content and not args.output:
            print("\nGenerated YAML:")
            print("-"*80)
            print(result.yaml_content)
            print("-"*80)

        # Exit with appropriate code
        if result.success:
            print("\nStatus: SUCCESS")
            sys.exit(0)
        else:
            print("\nStatus: FAILED")
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
