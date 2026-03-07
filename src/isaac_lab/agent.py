"""IsaacLab Environment Generation Agent.

Converts YAML task documents into IsaacLab ManagerBasedRLEnv Python code
and validates by executing with IsaacLab.

Usage:
    python scripts/isaaclab_agent.py tasks/franka/stack/franka_stack.yaml
    python scripts/isaaclab_agent.py tasks/franka/lift/franka_lift.yaml --dry-run
    python scripts/isaaclab_agent.py --batch tasks/franka/
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from src.common.llm_client import AzureOpenAIClient
from src.common.robot_names import contains_robot_name, normalize_robot_name

console = Console()


# ---------------------------------------------------------------------------
# Reference map: YAML task category -> IsaacLab source files (code pattern examples)
# ---------------------------------------------------------------------------

REFERENCE_MAP = {
    "stack": {
        "base": "stack/stack_env_cfg.py",
        "robot": {
            "default": ["stack/config/franka/stack_joint_pos_env_cfg.py"],
            "ur10e": [
                "stack/config/ur10_gripper/stack_joint_pos_env_cfg.py",
                "deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py",
            ],
        },
        "mdp": ["stack/mdp/terminations.py", "stack/mdp/observations.py"],
    },
    "lift": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["lift/mdp/rewards.py", "lift/mdp/terminations.py"],
    },
    "reach": {
        "base": "reach/reach_env_cfg.py",
        "robot": {
            "default": ["reach/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["reach/config/openarm/unimanual/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/reach/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": [],
    },
    "cabinet": {
        "base": "cabinet/cabinet_env_cfg.py",
        "robot": {
            "default": ["cabinet/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["cabinet/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": [],
    },
    # Categories without direct IsaacLab equivalent — use closest match
    "pick_place": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": [
                "deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py",
                "deploy/reach/config/ur_10e/joint_pos_env_cfg.py",
            ],
        },
        "mdp": ["lift/mdp/rewards.py", "pick_place/mdp/observations.py", "pick_place/mdp/terminations.py"],
    },
    "sort": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "openarm": ["lift/config/openarm/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["lift/mdp/rewards.py"],
    },
    "peg_insert": {
        "base": "lift/lift_env_cfg.py",
        "robot": {
            "default": ["lift/config/franka/joint_pos_env_cfg.py"],
            "ur10e": ["deploy/gear_assembly/config/ur_10e/joint_pos_env_cfg.py"],
        },
        "mdp": ["lift/mdp/rewards.py"],
    },
}


class IsaacLabAgent:
    """Agent that converts YAML task documents to IsaacLab environment code."""

    def __init__(self, config_path: str | None = None):
        config_path = config_path or str(PROJECT_ROOT / "configs" / "isaaclab_agent_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.isaaclab_path = Path(
            self.config["isaaclab"].get("path")
            or os.environ.get("ISAACLAB_PATH")
            or os.path.expanduser("~/workspace/IsaacLab")
        )
        self.reference_base = self.isaaclab_path / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "manipulation"
        self.output_dir = PROJECT_ROOT / self.config["agent"]["output_dir"]
        self.max_retries = self.config["agent"]["max_retries"]
        self.execution_timeout = self.config["agent"]["execution_timeout"]
        self.headless = self.config["agent"]["headless"]
        self.num_envs = self.config["agent"]["validation_num_envs"]
        self.save_intermediate = self.config["agent"]["save_intermediate"]

        # Load prompt templates
        self.system_prompt = (PROJECT_ROOT / "prompts" / "isaaclab_generation.md").read_text()
        self.error_fix_template = (PROJECT_ROOT / "prompts" / "isaaclab_error_fix.md").read_text()

        # Initialize LLM client
        self.llm = AzureOpenAIClient()

    # ----- YAML Loading -----

    def load_task_yaml(self, yaml_path: str) -> dict:
        """Load and return parsed YAML task document."""
        with open(yaml_path) as f:
            return yaml.safe_load(f)

    def _detect_category(self, yaml_path: str) -> str:
        """Detect task category from path (tasks/{robot}/{category}/...)."""
        parts = Path(yaml_path).parts
        # Look for known categories
        for part in parts:
            if part in REFERENCE_MAP:
                return part
        # Fallback: guess from filename
        name = Path(yaml_path).stem.lower()
        for cat in REFERENCE_MAP:
            if cat in name:
                return cat
        return "lift"  # default fallback

    def _detect_robot(self, task_doc: dict, yaml_path: str | None = None) -> str:
        """Detect robot type from task document."""
        for asset in task_doc.get("assets", []):
            if asset.get("type") == "articulation":
                robot = contains_robot_name(
                    [
                        str(asset.get("robot_type", "")),
                        str(asset.get("name", "")),
                        str(asset.get("asset_path", "")),
                    ]
                )
                if robot:
                    return robot

        task_meta = task_doc.get("task", {})
        robot = contains_robot_name([str(task_meta.get("name", "")), str(task_meta.get("description", ""))])
        if robot:
            return robot

        if yaml_path:
            robot = normalize_robot_name(str(Path(yaml_path)))
            if robot:
                return robot

        return "franka"

    # ----- Reference Code Loading -----

    def _robot_reference_paths(self, category: str, robot: str) -> list[str]:
        """Return ordered robot-specific reference paths."""
        ref_info = REFERENCE_MAP.get(category, REFERENCE_MAP["lift"])
        robot_refs = ref_info.get("robot", {})
        if isinstance(robot_refs, str):
            return [robot_refs]
        return robot_refs.get(robot, robot_refs.get("default", []))

    def select_reference(self, category: str, robot: str) -> str:
        """Read IsaacLab reference source files for the given task category.

        These are used as CODE PATTERN EXAMPLES in the LLM prompt.
        The actual task content comes from the YAML document.
        """
        ref_info = REFERENCE_MAP.get(category, REFERENCE_MAP["lift"])
        parts = []

        # Base config
        base_path = self.reference_base / ref_info["base"]
        if base_path.exists():
            parts.append(f"### Reference: {ref_info['base']}\n```python\n{base_path.read_text()}\n```")

        # Robot-specific config
        for robot_ref in self._robot_reference_paths(category, robot):
            robot_path = self.reference_base / robot_ref
            if robot_path.exists():
                content = robot_path.read_text()
                if len(content) > 6000:
                    content = content[:6000] + "\n# ... (truncated)"
                parts.append(f"### Reference: {robot_ref}\n```python\n{content}\n```")

        # MDP files (rewards, terminations)
        for mdp_file in ref_info.get("mdp", []):
            mdp_path = self.reference_base / mdp_file
            if mdp_path.exists():
                content = mdp_path.read_text()
                # Truncate if too long (save tokens for gpt-5-mini)
                if len(content) > 3000:
                    content = content[:3000] + "\n# ... (truncated)"
                parts.append(f"### Reference: {mdp_file}\n```python\n{content}\n```")

        return "\n\n".join(parts)

    def _build_robot_guidance(self, robot: str) -> str:
        """Return robot-specific implementation rules to inject into the prompt."""
        if robot != "ur10e":
            return ""

        return """## Robot-Specific Rules (UR10e + Robotiq 2F-85)

- This task MUST use `UR10e_ROBOTIQ_2F_85_CFG`, not `FRANKA_PANDA_CFG`, `UR10_CFG`, or suction gripper configs.
- Add this import when `robot_type` is `ur10e`:
  `from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG`
- Robot setup pattern:
  ```python
  self.scene.robot = UR10e_ROBOTIQ_2F_85_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
  ```
- UR10e arm action should target the 6 arm joints only:
  `["shoulder_.*", "elbow_joint", "wrist_.*"]`
- Use `wrist_3_link` as the end-effector body for observations / frame transformers.
- Never use `panda_hand`, `panda_link0`, `ee_link`, or `panda_finger.*` in UR10e code.
- Prefer a binary gripper action with the Robotiq joints:
  ```python
  self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
      asset_name="robot",
      joint_names=[
          "finger_joint",
          "right_outer_knuckle_joint",
          "left_inner_finger_joint",
          "right_inner_finger_joint",
          "left_inner_finger_knuckle_joint",
          "right_inner_finger_knuckle_joint",
      ],
      open_command_expr={
          "finger_joint": 0.0,
          "right_outer_knuckle_joint": 0.0,
          "left_inner_finger_joint": 0.0,
          "right_inner_finger_joint": 0.0,
          "left_inner_finger_knuckle_joint": 0.0,
          "right_inner_finger_knuckle_joint": 0.0,
      },
      close_command_expr={
          "finger_joint": 0.65,
          "right_outer_knuckle_joint": 0.65,
          "left_inner_finger_joint": -0.65,
          "right_inner_finger_joint": 0.65,
          "left_inner_finger_knuckle_joint": -0.65,
          "right_inner_finger_knuckle_joint": -0.65,
      },
  )
  ```
- For UR10e observations, use `SceneEntityCfg("robot", body_names=["wrist_3_link"])`.
"""

    # ----- Prompt Building -----

    def build_prompt(self, yaml_path: str, task_doc: dict, reference_code: str) -> str:
        """Build the user prompt with YAML content and reference code."""
        yaml_content = Path(yaml_path).read_text()
        category = self._detect_category(yaml_path)
        robot = self._detect_robot(task_doc, yaml_path)
        task_name = task_doc.get("task", {}).get("name", "UnknownTask")
        robot_guidance = self._build_robot_guidance(robot)

        prompt = f"""## Task

Convert the following YAML task document into IsaacLab ManagerBasedRLEnv Python code.

**Task name**: {task_name}
**Robot**: {robot}
**Category**: {category}
**Config class name**: `{task_name}EnvCfg`

## YAML Task Document (THIS IS THE INPUT — use these values)

```yaml
{yaml_content}
```

## IsaacLab Reference Code (USE AS CODE PATTERN EXAMPLES ONLY)

The following IsaacLab source files show the correct Python code structure.
Use this structure/pattern, but fill in values from the YAML above.

{reference_code}

{robot_guidance}

## Instructions

1. Generate `env_cfg.py` containing the full environment configuration:
   - Scene class with robot, table, ground plane, lighting, and all objects from YAML
   - Actions, observations, events, rewards, and terminations config classes
   - Main `{task_name}EnvCfg(ManagerBasedRLEnvCfg)` class

2. Generate `run_env.py` for validation (follow the exact AppLauncher pattern from the system prompt)

3. Map ALL YAML values to IsaacLab equivalents:
   - Robot: use the correct pre-defined config for the detected robot (`FRANKA_PANDA_CFG` for franka, `UR10e_ROBOTIQ_2F_85_CFG` for ur10e)
   - Objects: map asset_path, position, rotation, scale, physics properties
   - Simulation: map timestep, decimation, episode_length, physx settings
   - Randomization: convert to EventTermCfg
   - Goals: convert to termination conditions

4. For goal.success_criteria or goal.conditions that need custom logic,
   generate mdp/terminations.py with the custom function.
"""
        return prompt

    # ----- Code Parsing -----

    def parse_generated_code(self, response: str) -> dict[str, str]:
        """Parse LLM response into {filename: code} dict.

        Expects format:
            ```python:filename.py
            ...code...
            ```
        """
        files = {}
        # Match ```python:filename.py ... ``` blocks
        pattern = r"```python:([^\n]+)\n(.*?)```"
        matches = re.findall(pattern, response, re.DOTALL)

        for filename, code in matches:
            filename = filename.strip()
            files[filename] = code.strip()

        # Fallback: if no filename annotations, try to detect by content
        if not files:
            # Try plain ```python blocks
            plain_blocks = re.findall(r"```python\n(.*?)```", response, re.DOTALL)
            for i, block in enumerate(plain_blocks):
                block = block.strip()
                if "AppLauncher" in block:
                    files["run_env.py"] = block
                elif "ManagerBasedRLEnvCfg" in block or "@configclass" in block:
                    files["env_cfg.py"] = block
                elif "def " in block and i > 1:
                    # Likely a custom MDP function
                    if "reward" in block.lower():
                        files["mdp/rewards.py"] = block
                    else:
                        files["mdp/terminations.py"] = block

        return files

    # ----- File Writing -----

    def write_output(self, code_dict: dict[str, str], output_dir: Path) -> Path:
        """Write generated code files to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for filename, code in code_dict.items():
            filepath = output_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(code)
            console.print(f"  [green]Wrote[/green] {filepath.relative_to(PROJECT_ROOT)}")

        # ALWAYS ensure mdp/ package exists (env_cfg.py uses `import mdp as mdp`)
        mdp_dir = output_dir / "mdp"
        mdp_dir.mkdir(parents=True, exist_ok=True)

        # Regenerate __init__.py to match actual custom files
        init_file = mdp_dir / "__init__.py"
        exports = []
        if (mdp_dir / "rewards.py").exists():
            exports.append("from .rewards import *  # noqa: F401, F403")
        if (mdp_dir / "terminations.py").exists():
            exports.append("from .terminations import *  # noqa: F401, F403")
        if (mdp_dir / "observations.py").exists():
            exports.append("from .observations import *  # noqa: F401, F403")
        init_file.write_text(
            '"""MDP functions — re-exports isaaclab.envs.mdp + custom functions."""\n\n'
            "from isaaclab.envs.mdp import *  # noqa: F401, F403\n\n"
            + "\n".join(exports)
            + "\n"
        )

        # Auto-patch env_cfg.py: ensure it uses local mdp/ package
        self._patch_mdp_import(output_dir)

        # Clear __pycache__ to avoid stale bytecode
        pycache = mdp_dir / "__pycache__"
        if pycache.exists():
            import shutil
            shutil.rmtree(pycache)

        return output_dir

    def _patch_mdp_import(self, output_dir: Path):
        """Ensure env_cfg.py imports local mdp/ package instead of isaaclab.envs.mdp."""
        env_cfg_path = output_dir / "env_cfg.py"
        if not env_cfg_path.exists():
            return

        content = env_cfg_path.read_text()
        original = content

        # Remove ALL variants of isaaclab.envs.mdp imports
        for pattern in [
            "import isaaclab.envs.mdp as mdp\n",
            "import isaaclab.envs.mdp as mdp  # noqa: F401, F403\n",
            "from isaaclab.envs.mdp import *  # noqa: F401, F403\n",
            "from isaaclab.envs.mdp import *\n",
        ]:
            content = content.replace(pattern, "")

        # Remove `from mdp import *` (wrong pattern — need `import mdp as mdp`)
        content = content.replace("from mdp import *  # noqa: F401, F403\n", "")
        content = content.replace("from mdp import *\n", "")
        # Also handle the "corrected import" comment variant
        for line in content.split("\n"):
            if line.strip().startswith("from mdp import *"):
                content = content.replace(line + "\n", "")

        # Ensure exactly one `import mdp as mdp` line exists
        if "import mdp as mdp" not in content:
            # Insert after the last isaaclab import
            lines = content.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("from isaaclab") or stripped.startswith("import isaaclab"):
                    insert_idx = i + 1
            lines.insert(insert_idx, "import mdp as mdp  # local mdp/ package")
            content = "\n".join(lines)

        if content != original:
            env_cfg_path.write_text(content)
            console.print("  [yellow]Auto-patched[/yellow] env_cfg.py → import mdp as mdp (local)")

    # ----- Post-Generation Validation -----

    def _validate_generated_code(self, output_dir: Path, robot: str) -> list[str]:
        """Post-generation validation: auto-fix known mistakes before execution."""
        fixes = []

        # Fix: `in env.scene:` → `in env.scene.keys():`
        # InteractiveScene has no __contains__, so `in` operator causes KeyError
        for py_file in output_dir.rglob("*.py"):
            try:
                text = py_file.read_text()
            except Exception:
                continue
            original = text
            # Match patterns like `not in env.scene:` or `in env.scene:`
            # but not already `in env.scene.keys():`
            text = re.sub(
                r'\bnot in env\.scene\b(?!\.keys)',
                'not in env.scene.keys()',
                text,
            )
            text = re.sub(
                r'(?<!not )\bin env\.scene\b(?!\.keys|\.articulations|\.rigid_objects|\.sensors|\[)',
                'in env.scene.keys()',
                text,
            )
            if text != original:
                py_file.write_text(text)
                fixes.append(f"Patched {py_file.name}: env.scene → env.scene.keys()")

        fixes.extend(self._apply_robot_specific_fixes(output_dir, robot))
        return fixes

    def _apply_robot_specific_fixes(self, output_dir: Path, robot: str) -> list[str]:
        """Patch predictable robot-specific mistakes from the LLM output."""
        if robot != "ur10e":
            return []

        env_cfg_path = output_dir / "env_cfg.py"
        if not env_cfg_path.exists():
            return []

        original = env_cfg_path.read_text()
        content = original

        replacements = [
            (
                "from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # if robot_type is franka",
                "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG",
            ),
            (
                "from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG",
                "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG",
            ),
            ("FRANKA_PANDA_CFG", "UR10e_ROBOTIQ_2F_85_CFG"),
            ('["panda_joint.*"]', '["shoulder_.*", "elbow_joint", "wrist_.*"]'),
            (
                '["panda_finger.*"]',
                '["finger_joint", "right_outer_knuckle_joint", "left_inner_finger_joint", '
                '"right_inner_finger_joint", "left_inner_finger_knuckle_joint", '
                '"right_inner_finger_knuckle_joint"]',
            ),
            ('{"panda_finger_.*": 0.04}', '{"finger_joint": 0.0, "right_outer_knuckle_joint": 0.0, '
             '"left_inner_finger_joint": 0.0, "right_inner_finger_joint": 0.0, '
             '"left_inner_finger_knuckle_joint": 0.0, "right_inner_finger_knuckle_joint": 0.0}'),
            ('{"panda_finger_.*": 0.0}', '{"finger_joint": 0.65, "right_outer_knuckle_joint": 0.65, '
             '"left_inner_finger_joint": -0.65, "right_inner_finger_joint": 0.65, '
             '"left_inner_finger_knuckle_joint": -0.65, "right_inner_finger_knuckle_joint": -0.65}'),
            ('gripper_joint_names = ["panda_finger_.*"]',
             'gripper_joint_names = ["finger_joint", "right_outer_knuckle_joint", '
             '"left_inner_finger_joint", "right_inner_finger_joint", '
             '"left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint"]'),
            ("gripper_open_val = 0.04", "gripper_open_val = 0.0"),
            ("gripper_threshold = 0.005", "gripper_threshold = 0.05"),
            ("panda_hand", "wrist_3_link"),
            ("panda_link0", "base_link"),
            ("ee_link", "wrist_3_link"),
        ]
        for source, target in replacements:
            content = content.replace(source, target)

        if "UR10e_ROBOTIQ_2F_85_CFG" in content and "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG" not in content:
            lines = content.split("\n")
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.startswith("from isaaclab_assets") or line.startswith("from isaaclab.markers"):
                    insert_idx = i
                    break
            lines.insert(insert_idx, "from isaaclab_assets.robots.universal_robots import UR10e_ROBOTIQ_2F_85_CFG")
            content = "\n".join(lines)

        if content == original:
            return []

        env_cfg_path.write_text(content)
        return ["Patched env_cfg.py: normalized UR10e Robotiq 2F-85 config and body/joint names"]

    # ----- Execution -----

    def execute_isaaclab(self, output_dir: Path) -> tuple[bool, str]:
        """Execute generated run_env.py via isaaclab.sh inside the conda env."""
        run_script = output_dir / "run_env.py"
        if not run_script.exists():
            return False, "run_env.py not found in output directory"

        launcher = self.isaaclab_path / "isaaclab.sh"
        if not launcher.exists():
            return False, f"isaaclab.sh not found at {launcher}"

        conda_env = self.config["isaaclab"]["conda_env"]

        # Build the isaaclab.sh command
        isaaclab_cmd = f"{launcher} -p {run_script} --num_envs {self.num_envs}"
        if self.headless:
            isaaclab_cmd += " --headless"

        # Wrap with conda run to ensure correct Python environment
        cmd = [
            "conda", "run", "-n", conda_env, "--no-capture-output",
            "bash", "-c", isaaclab_cmd,
        ]

        console.print(f"  [blue]Executing[/blue]: conda run -n {conda_env} -- {isaaclab_cmd}")

        # Marker file for reliable SUCCESS detection (stdout can be lost when simulation_app.close() hangs)
        marker_file = output_dir / ".success_marker"
        if marker_file.exists():
            marker_file.unlink()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"  # Force unbuffered stdout in subprocess
        env["ISAACLAB_SUCCESS_MARKER"] = str(marker_file)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.execution_timeout,
                cwd=str(output_dir),
                env=env,
            )
            output = result.stdout + "\n" + result.stderr
            # Check for SUCCESS in stdout/stderr OR via marker file
            success = "SUCCESS" in output or marker_file.exists()
            return success, output

        except subprocess.TimeoutExpired as e:
            # If timed out, check if SUCCESS was printed before the hang OR marker file exists
            partial = ""
            if e.stdout:
                partial += e.stdout if isinstance(e.stdout, str) else e.stdout.decode()
            if e.stderr:
                partial += e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
            if "SUCCESS" in partial or marker_file.exists():
                return True, partial
            return False, f"Execution timed out after {self.execution_timeout}s\n{partial[-2000:]}"
        except Exception as e:
            return False, f"Execution error: {e}"

    # ----- Error Parsing -----

    def _extract_traceback(self, output: str) -> str:
        """Extract the Python traceback from IsaacLab output, stripping GPU info/warnings."""
        lines = output.split("\n")
        # Find "Traceback (most recent call last):" and take everything from there
        traceback_start = None
        for i, line in enumerate(lines):
            if "Traceback (most recent call last):" in line:
                traceback_start = i
                break

        if traceback_start is not None:
            traceback_text = "\n".join(lines[traceback_start:])
            # Truncate if still too long
            if len(traceback_text) > 3000:
                traceback_text = traceback_text[:3000] + "\n... (truncated)"
            return traceback_text

        # No traceback found — look for ERROR lines or return last 2000 chars
        error_lines = [l for l in lines if "Error" in l or "error" in l.lower() or "FAILED" in l]
        if error_lines:
            return "\n".join(error_lines[-20:])

        # Fallback: return last 2000 chars
        if len(output) > 2000:
            return "... (no traceback found, last 2000 chars)\n" + output[-2000:]
        return output

    # ----- Error Fix -----

    def fix_errors(self, code_dict: dict[str, str], error_msg: str, attempt: int) -> dict[str, str]:
        """Ask LLM to fix errors in generated code."""
        console.print(f"  [yellow]Attempt {attempt}[/yellow]: Asking LLM to fix errors...")

        # Build original code section
        original_parts = []
        for filename, code in code_dict.items():
            original_parts.append(f"```python:{filename}\n{code}\n```")
        original_code = "\n\n".join(original_parts)

        # Extract only the traceback from the error output (skip GPU info, warnings)
        error_msg = self._extract_traceback(error_msg)

        prompt = self.error_fix_template.replace("{ORIGINAL_CODE}", original_code)
        prompt = prompt.replace("{ERROR_TRACEBACK}", error_msg)

        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            temperature=0.1,
        )

        fixed_files = self.parse_generated_code(response)

        # Merge: update only changed files, keep unchanged
        merged = dict(code_dict)
        merged.update(fixed_files)
        return merged

    # ----- Main Pipeline -----

    def run(self, yaml_path: str, dry_run: bool = False,
            evaluate: bool = False) -> dict:
        """Run the full pipeline: YAML → code generation → execution → error fix loop.

        Args:
            yaml_path: Path to the YAML task document.
            dry_run: If True, generate code but don't execute.
            evaluate: If True, run quality evaluation after successful execution.

        Returns:
            dict with keys: success, output_dir, attempts, error, evaluation (optional)
        """
        yaml_path = str(Path(yaml_path).resolve())
        task_doc = self.load_task_yaml(yaml_path)
        task_name = task_doc.get("task", {}).get("name", "UnknownTask")
        category = self._detect_category(yaml_path)
        robot = self._detect_robot(task_doc, yaml_path)

        console.print(Panel(
            f"[bold]Task[/bold]: {task_name}\n"
            f"[bold]Robot[/bold]: {robot}\n"
            f"[bold]Category[/bold]: {category}\n"
            f"[bold]YAML[/bold]: {yaml_path}",
            title="IsaacLab Agent",
            border_style="blue",
        ))

        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_slug = task_name.lower().replace(" ", "_")
        output_dir = self.output_dir / f"{task_slug}_{timestamp}"

        # Step 1: Load reference code (pattern examples)
        console.print("[bold]Step 1:[/bold] Loading reference code patterns...")
        reference_code = self.select_reference(category, robot)

        # Step 2: Build prompt and generate code
        console.print("[bold]Step 2:[/bold] Generating IsaacLab code via LLM...")
        user_prompt = self.build_prompt(yaml_path, task_doc, reference_code)
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=self.config["agent"]["temperature"],
        )

        code_dict = self.parse_generated_code(response)
        if not code_dict:
            console.print("[red]Error:[/red] LLM returned no parseable code blocks")
            return {"success": False, "output_dir": None, "attempts": 0, "error": "No code generated"}

        console.print(f"  Generated {len(code_dict)} file(s): {list(code_dict.keys())}")

        # Step 3: Write files
        console.print("[bold]Step 3:[/bold] Writing generated code...")
        self.write_output(code_dict, output_dir)

        if dry_run:
            console.print("[green]Dry run complete.[/green] Code saved to:", str(output_dir))
            return {"success": True, "output_dir": str(output_dir), "attempts": 0, "error": None}

        # Step 3.5: Validate and auto-fix known mistakes
        fixes = self._validate_generated_code(output_dir, robot)
        for fix in fixes:
            console.print(f"  [yellow]Auto-fix[/yellow]: {fix}")

        # Step 4: Execute and iterate
        for attempt in range(1, self.max_retries + 1):
            console.print(f"[bold]Step 4:[/bold] Execution attempt {attempt}/{self.max_retries}...")
            success, output = self.execute_isaaclab(output_dir)

            if success:
                console.print(f"[green bold]SUCCESS[/green bold] on attempt {attempt}")
                result = {"success": True, "output_dir": str(output_dir), "attempts": attempt, "error": None}

                if evaluate:
                    console.print("[bold]Step 5:[/bold] Running quality evaluation...")
                    from src.isaac_lab.evaluator import IsaacLabEvaluator
                    evaluator = IsaacLabEvaluator()
                    eval_result = evaluator.evaluate(
                        output_dir=str(output_dir), yaml_path=yaml_path,
                    )
                    result["evaluation"] = eval_result
                    result["eval_score"] = eval_result.get("total_score", 0)

                return result

            console.print(f"  [red]Failed[/red] (attempt {attempt})")

            # Save intermediate output
            if self.save_intermediate:
                error_file = output_dir / f"error_attempt_{attempt}.txt"
                error_file.write_text(output)

            # Fix errors (except on last attempt)
            if attempt < self.max_retries:
                code_dict = self.fix_errors(code_dict, output, attempt + 1)
                self.write_output(code_dict, output_dir)
                # Re-validate after error fix
                fixes = self._validate_generated_code(output_dir, robot)
                for fix in fixes:
                    console.print(f"  [yellow]Auto-fix[/yellow]: {fix}")

        console.print(f"[red bold]FAILED[/red bold] after {self.max_retries} attempts")
        return {
            "success": False,
            "output_dir": str(output_dir),
            "attempts": self.max_retries,
            "error": output[-500:] if output else "Unknown error",
        }

    def run_batch(self, task_dir: str, dry_run: bool = False) -> list[dict]:
        """Run the agent on all YAML files in a directory (recursively)."""
        task_dir = Path(task_dir)
        yaml_files = sorted(task_dir.rglob("*.yaml"))
        yaml_files = [f for f in yaml_files if "template" not in str(f)]

        console.print(f"[bold]Batch mode:[/bold] Found {len(yaml_files)} task YAML files in {task_dir}")

        results = []
        for i, yaml_file in enumerate(yaml_files, 1):
            console.print(f"\n{'='*60}")
            console.print(f"[bold][{i}/{len(yaml_files)}][/bold] {yaml_file.name}")
            console.print(f"{'='*60}")

            result = self.run(str(yaml_file), dry_run=dry_run)
            result["yaml_path"] = str(yaml_file)
            results.append(result)

        # Summary
        successes = sum(1 for r in results if r["success"])
        console.print(f"\n[bold]Batch Summary:[/bold] {successes}/{len(results)} succeeded")
        for r in results:
            status = "[green]OK[/green]" if r["success"] else "[red]FAIL[/red]"
            console.print(f"  {status} {Path(r['yaml_path']).name}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert YAML task documents to IsaacLab ManagerBasedRLEnv code"
    )
    parser.add_argument("yaml_path", nargs="?", help="Path to task YAML file")
    parser.add_argument("--batch", type=str, help="Directory of YAML files to process")
    parser.add_argument("--dry-run", action="store_true", help="Generate code without executing")
    parser.add_argument("--evaluate", action="store_true", help="Run quality evaluation after success")
    parser.add_argument("--eval-only", type=str, help="Evaluate existing output directory (skip generation)")
    parser.add_argument("--output-dir", type=str, help="Override output directory")
    parser.add_argument("--config", type=str, help="Path to agent config YAML")
    args = parser.parse_args()

    # Eval-only mode: evaluate existing output without re-generating
    if args.eval_only:
        if not args.yaml_path:
            parser.error("--eval-only requires yaml_path")
        from src.isaac_lab.evaluator import IsaacLabEvaluator
        evaluator = IsaacLabEvaluator()
        report = evaluator.evaluate(output_dir=args.eval_only, yaml_path=args.yaml_path)
        sys.exit(0 if report["total_score"] >= 70 else 1)

    if not args.yaml_path and not args.batch:
        parser.error("Either yaml_path or --batch is required")

    agent = IsaacLabAgent(config_path=args.config)

    if args.output_dir:
        agent.output_dir = Path(args.output_dir)

    if args.batch:
        agent.run_batch(args.batch, dry_run=args.dry_run)
    else:
        result = agent.run(args.yaml_path, dry_run=args.dry_run,
                           evaluate=args.evaluate)
        sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
