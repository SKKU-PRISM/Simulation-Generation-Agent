#!/usr/bin/env python3
"""CLI entry point for Isaac Sim automated pipeline runner.

Usage:
    python scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml
    python scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm
"""

from src.isaac_sim.runner import main


if __name__ == "__main__":
    main()
