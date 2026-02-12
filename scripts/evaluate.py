#!/usr/bin/env python3
"""CLI entry point for standalone IsaacLab evaluation.

Usage:
    python scripts/evaluate.py outputs/isaaclab/FrankaStack tasks/franka/stack/franka_stack.yaml
    python scripts/evaluate.py --skip-runtime outputs/isaaclab/FrankaStack tasks/franka/stack/franka_stack.yaml
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.isaac_lab.evaluator import main

if __name__ == "__main__":
    main()
