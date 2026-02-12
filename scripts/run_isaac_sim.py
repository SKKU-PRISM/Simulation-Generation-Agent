#!/usr/bin/env python3
"""CLI entry point for Isaac Sim + VLM evaluation loop.

Usage:
    python scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml
    python scripts/run_isaac_sim.py tasks/franka/stack/franka_stack.yaml --skip-vlm

Note: Requires running Isaac Sim with MCP extension on localhost:8766.
      The iterative refinement loop (build → screenshot → eval → retry)
      is not yet fully implemented as an automated pipeline runner.
      Use build_scene.py for individual scene building.
"""

import sys

# TODO: Import from src.isaac_sim.runner once implemented
print("Isaac Sim automated runner is not yet implemented.")
print("Use individual components:")
print("  python scripts/build_scene.py <yaml_path>       # Build scene")
print("  python -m src.isaac_sim.screenshot <output.png>  # Capture screenshot")
print("  python -m src.isaac_sim.vlm_evaluator ...        # VLM evaluation")
sys.exit(1)
