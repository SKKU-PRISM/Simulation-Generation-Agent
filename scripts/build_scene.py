#!/usr/bin/env python3
"""CLI entry point for Isaac Sim scene building via MCP.

Usage:
    python scripts/build_scene.py tasks/franka/stack/franka_stack.yaml
    python scripts/build_scene.py tasks/franka/stack/franka_stack.yaml --host localhost --port 8766
"""

from src.isaac_sim.scene_builder import main

if __name__ == "__main__":
    main()
