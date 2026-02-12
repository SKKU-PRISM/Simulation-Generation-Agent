#!/usr/bin/env python3
"""CLI entry point for Pipeline 1: Isaac Sim scene building via MCP.

Usage:
    python scripts/build_scene.py tasks/franka/stack/franka_stack.yaml
    python scripts/build_scene.py tasks/franka/stack/franka_stack.yaml --host localhost --port 8766
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.isaac_sim.scene_builder import main

if __name__ == "__main__":
    main()
