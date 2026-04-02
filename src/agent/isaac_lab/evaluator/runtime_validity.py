"""Category 4: Runtime Validity (25 pts) - Generate and execute eval_runner.py, parse results."""

import json
import os
import subprocess
from pathlib import Path

from rich.console import Console

from .parser import _check, EnvCfgParser

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[4]


# ---------------------------------------------------------------------------
# Category 4: Runtime Validity (25 pts)
# ---------------------------------------------------------------------------

class RuntimeValidityChecker:
    """Generate and execute eval_runner.py, parse results."""

    CAT = "runtime_validity"

    def __init__(self, output_dir: Path, parser: EnvCfgParser,
                 yaml_doc: dict, config: dict):
        self.output_dir = Path(output_dir).resolve()
        self.parser = parser
        self.yaml_doc = yaml_doc
        self.cfg = config
        self.results: dict | None = None

    def generate_eval_runner(self) -> Path:
        """Generate eval_runner.py in output_dir."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        env_cfg_class = self.parser.find_env_cfg_class() or "EnvCfg"

        # Collect rigid object names for position checking
        rigid_names = [a["name"] for a in self.yaml_doc.get("assets", [])
                       if a.get("type") == "rigid"]
        obj_names_str = json.dumps(rigid_names)

        eval_steps = self.cfg.get("eval_steps", 20)
        num_envs = self.cfg.get("num_envs", 2)

        code = f'''"""IsaacLab environment evaluation runner (auto-generated)."""
import argparse
import json
import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default={num_envs})
parser.add_argument("--eval_steps", type=int, default={eval_steps})
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from env_cfg import {env_cfg_class}

OBJECT_NAMES = {obj_names_str}

def main():
    results = {{
        "env_created": False,
        "reset_success": False,
        "steps_completed": 0,
        "obs_shapes": {{}},
        "obs_ranges": {{}},
        "reward_stats": {{}},
        "object_positions": {{}},
        "errors": [],
    }}

    try:
        env_cfg = {env_cfg_class}()
        env_cfg.scene.num_envs = args.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)
        results["env_created"] = True

        obs, _ = env.reset()
        results["reset_success"] = True

        # Record obs shapes
        if isinstance(obs, dict):
            for key, val in obs.items():
                if hasattr(val, "shape"):
                    results["obs_shapes"][key] = list(val.shape)

        all_rewards = []
        for step in range(args.eval_steps):
            actions = torch.zeros_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(actions)
            all_rewards.append(rew.detach().cpu())
            results["steps_completed"] = step + 1

        # Obs ranges from last obs
        if isinstance(obs, dict):
            for key, val in obs.items():
                if hasattr(val, "shape"):
                    t = val.detach().cpu().float()
                    results["obs_ranges"][key] = {{
                        "min": float(t.min()),
                        "max": float(t.max()),
                        "has_nan": bool(torch.isnan(t).any()),
                        "has_inf": bool(torch.isinf(t).any()),
                    }}

        # Reward stats
        if all_rewards:
            rt = torch.stack(all_rewards)
            results["reward_stats"] = {{
                "mean": float(rt.mean()),
                "std": float(rt.std()),
                "min": float(rt.min()),
                "max": float(rt.max()),
                "all_zero": bool((rt == 0).all()),
            }}

        # Object positions
        for name in OBJECT_NAMES:
            try:
                entity = env.scene[name]
                pos = entity.data.root_pos_w[0].detach().cpu().tolist()
                results["object_positions"][name] = pos
            except Exception:
                pass

        env.close()

    except Exception as e:
        results["errors"].append({{
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()[-2000:],
        }})

    # Write results BEFORE simulation_app.close() (which may hang)
    out_path = Path(os.path.dirname(os.path.abspath(__file__))) / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"EVAL_COMPLETE: {{out_path}}", flush=True)

    marker = os.environ.get("ISAACLAB_SUCCESS_MARKER")
    if marker:
        Path(marker).write_text("EVAL_COMPLETE")
    simulation_app.close()

if __name__ == "__main__":
    main()
'''
        runner_path = (self.output_dir / "eval_runner.py").resolve()
        runner_path.write_text(code)
        return runner_path

    def execute(self) -> bool:
        """Execute eval_runner.py via conda subprocess."""
        runner = self.generate_eval_runner()
        if not runner.exists():
            console.print(f"  [red]eval_runner.py not found[/red]: {runner}")
            return False

        isaaclab_path = Path(
            self.cfg.get("isaaclab_path")
            or os.environ.get("ISAACLAB_PATH")
            or str((PROJECT_ROOT.parent / "IsaacLab").resolve())
        ).expanduser().resolve()
        conda_env = self.cfg.get("conda_env", "env_isaaclab")
        launcher = (isaaclab_path / "isaaclab.sh").resolve()
        if not launcher.exists():
            console.print(f"  [red]isaaclab.sh not found[/red]: {launcher}")
            return False

        timeout = self.cfg.get("timeout", 300)

        # Run from output_dir and pass runner filename to avoid relative-path duplication.
        isaaclab_cmd = f"{launcher} -p {runner.name} --num_envs {self.cfg.get('num_envs', 2)}"
        if self.cfg.get("headless", True):
            isaaclab_cmd += " --headless"

        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output",
               "bash", "-c", isaaclab_cmd]

        marker_file = (self.output_dir / ".eval_marker").resolve()
        if marker_file.exists():
            marker_file.unlink()

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["ISAACLAB_SUCCESS_MARKER"] = str(marker_file)

        console.print(f"  [blue]Running eval_runner.py[/blue] (timeout={timeout}s)...")

        try:
            subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, cwd=str(self.output_dir), env=env,
            )
        except subprocess.TimeoutExpired:
            console.print("  [yellow]Eval runner timed out[/yellow] (results may still be available)")
        except Exception as e:
            console.print(f"  [red]Eval runner error[/red]: {e}")
            return False

        return self._read_results()

    def _read_results(self) -> bool:
        results_file = (self.output_dir / "eval_results.json").resolve()
        if results_file.exists():
            with open(results_file) as f:
                self.results = json.load(f)
            return True
        return False

    def score_env_creation(self) -> dict:
        """(8 pts) Env created successfully."""
        if not self.results:
            return _check(self.CAT, "env_creation", 0, 8, "No runtime results")
        if self.results.get("env_created"):
            return _check(self.CAT, "env_creation", 8, 8, "Environment created")
        errors = self.results.get("errors", [])
        msg = errors[0]["message"][:100] if errors else "unknown error"
        return _check(self.CAT, "env_creation", 0, 8, f"Failed: {msg}")

    def score_reset_step(self) -> dict:
        """(7 pts) N+ steps completed."""
        if not self.results:
            return _check(self.CAT, "reset_step_cycle", 0, 7, "No runtime results")
        steps = self.results.get("steps_completed", 0)
        target = self.cfg.get("eval_steps", 20)
        if steps >= target:
            return _check(self.CAT, "reset_step_cycle", 7, 7,
                           f"{steps} steps completed")
        if steps > 0:
            score = round(7 * steps / target)
            return _check(self.CAT, "reset_step_cycle", max(score, 1), 7,
                           f"Only {steps}/{target} steps")
        return _check(self.CAT, "reset_step_cycle", 0, 7, "No steps completed")

    def score_reward_computation(self) -> dict:
        """(5 pts) Finite reward values, detect non-zero signal."""
        if not self.results:
            return _check(self.CAT, "reward_computation", 0, 5, "No runtime results")

        stats = self.results.get("reward_stats", {})
        if not stats:
            # rewards=None case
            if self.parser.rewards_is_none():
                return _check(self.CAT, "reward_computation", 3, 5,
                               "rewards=None (acceptable for validation)")
            return _check(self.CAT, "reward_computation", 0, 5, "No reward data")

        import math
        if math.isnan(stats.get("mean", 0)) or math.isinf(stats.get("mean", 0)):
            return _check(self.CAT, "reward_computation", 0, 5, "NaN/Inf rewards")

        if stats.get("all_zero"):
            return _check(self.CAT, "reward_computation", 3, 5,
                           "All rewards zero (no learning signal)")

        return _check(self.CAT, "reward_computation", 5, 5,
                       f"mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    def score_physics_stability(self) -> dict:
        """(5 pts) Object positions bounded."""
        if not self.results:
            return _check(self.CAT, "physics_stability", 0, 5, "No runtime results")

        positions = self.results.get("object_positions", {})
        if not positions:
            return _check(self.CAT, "physics_stability", 3, 5, "No object positions collected")

        bound = self.cfg.get("position_bound", 10.0)
        exploded = []
        for name, pos in positions.items():
            if any(abs(p) > bound for p in pos[:3]):
                exploded.append(name)

        # Check obs for NaN
        obs_ranges = self.results.get("obs_ranges", {})
        has_nan = any(r.get("has_nan") for r in obs_ranges.values() if isinstance(r, dict))

        pts = 5
        details = []
        if exploded:
            pts -= 3
            details.append(f"exploded: {exploded}")
        if has_nan:
            pts -= 2
            details.append("NaN in observations")

        return _check(self.CAT, "physics_stability", max(pts, 0), 5,
                       ". ".join(details) or "stable")

    def run_all(self) -> list[dict]:
        return [
            self.score_env_creation(),
            self.score_reset_step(),
            self.score_reward_computation(),
            self.score_physics_stability(),
        ]
