#!/usr/bin/env python3
"""CLI entry point for IsaacLab + LLM code generation.

Usage:
    python scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml
    python scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --evaluate
    python scripts/run_isaac_lab.py tasks/franka/stack/franka_stack.yaml --dry-run
    python scripts/run_isaac_lab.py --batch tasks/franka/
    python scripts/run_isaac_lab.py --eval-only outputs/isaaclab/FrankaStack tasks/franka/stack/franka_stack.yaml
"""

from src.agent.isaac_lab.agent import main

if __name__ == "__main__":
    main()
