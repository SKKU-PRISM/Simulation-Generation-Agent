#!/usr/bin/env python3
"""CLI entry point for standalone IsaacLab evaluation.

Usage:
    python scripts/evaluate.py outputs/isaaclab/FrankaStack tasks/franka/stack/franka_stack.yaml
    python scripts/evaluate.py --skip-runtime outputs/isaaclab/FrankaStack tasks/franka/stack/franka_stack.yaml
"""

from src.agent.isaac_lab.evaluator import main

if __name__ == "__main__":
    main()
