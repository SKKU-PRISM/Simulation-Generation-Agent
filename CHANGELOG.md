# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-03

### Added
- NL-to-YAML task specification pipeline with RAG matching
- IsaacLab environment code generation via LLM (scene, MDP, rewards, terminations)
- Self-refinement loop with static analysis and runtime evaluation
- Automated data collection pipeline (pick-and-place, stacking)
- Multi-camera system (top, wrist, front) with headless rendering
- VLM-based episode success judge (multi-view)
- IK-based manipulation with Pinocchio (6-DOF + orientation fallback)
- Robot support: Franka Panda, UR10e + Robotiq 2F-85, SO-101, OpenARM
- Docker support with NVIDIA Isaac Sim 5.1.0 + IsaacLab v2.3.2
- LeRobot-compatible dataset export
- Multi-language documentation (EN, KO, ZH, JA, DE)
- 82 predefined task YAML templates
