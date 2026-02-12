"""IsaacLab Environment Evaluation System.

Evaluates LLM-generated IsaacLab ManagerBasedRLEnv code against
the source YAML task document across 4 categories (100 points):
  1. Scene Fidelity (30)   -- YAML assets vs generated scene
  2. MDP Correctness (25)  -- obs/action/reward/term/event config
  3. Task Alignment (25)   -- YAML goal mapped to code
  4. Runtime Validity (20) -- actual execution in IsaacLab

Usage:
    python -m src.isaac_lab.evaluator <output_dir> <yaml_path>
    python -m src.isaac_lab.evaluator --skip-runtime <output_dir> <yaml_path>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .parser import _check, GoalNormalizer, EnvCfgParser
from .scene_fidelity import SceneFidelityChecker
from .mdp_correctness import MDPCorrectnessChecker
from .task_alignment import TaskAlignmentChecker
from .runtime_validity import RuntimeValidityChecker

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
console = Console()


# ---------------------------------------------------------------------------
# Main Evaluator
# ---------------------------------------------------------------------------

class IsaacLabEvaluator:
    """Orchestrates all evaluation phases."""

    def __init__(self, config_path: str | None = None):
        config_path = config_path or str(PROJECT_ROOT / "configs" / "isaaclab_eval_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def evaluate(self, output_dir: str, yaml_path: str,
                 skip_runtime: bool = False) -> dict:
        """Run full evaluation. Returns report dict."""
        output_dir = Path(output_dir)
        yaml_path = str(Path(yaml_path).resolve())

        # Load inputs
        with open(yaml_path) as f:
            yaml_doc = yaml.safe_load(f)

        env_cfg_path = output_dir / "env_cfg.py"
        if not env_cfg_path.exists():
            return {"total_score": 0, "checklist": [], "breakdown": {},
                    "issues": ["env_cfg.py not found"]}

        parser = EnvCfgParser(env_cfg_path)
        eval_cfg = self.config.get("evaluation", {})
        runtime_cfg = self.config.get("runtime", {})
        runtime_cfg["isaaclab_path"] = self.config.get("isaaclab", {}).get("path", "")
        runtime_cfg["conda_env"] = self.config.get("isaaclab", {}).get("conda_env", "env_isaaclab")

        # Normalize goals
        goals = GoalNormalizer.normalize(
            yaml_doc.get("goal", {}), yaml_doc.get("assets", [])
        )

        console.print(Panel(
            f"[bold]Output[/bold]: {output_dir}\n"
            f"[bold]YAML[/bold]: {yaml_path}\n"
            f"[bold]Goals[/bold]: {len(goals)} conditions\n"
            f"[bold]Runtime[/bold]: {'skip' if skip_runtime else 'enabled'}",
            title="IsaacLab Evaluator", border_style="cyan",
        ))

        # Phase 1: Static analysis
        console.print("[bold]Phase 1:[/bold] Static analysis...")
        scene_checker = SceneFidelityChecker(yaml_doc, parser, eval_cfg)
        mdp_checker = MDPCorrectnessChecker(yaml_doc, parser, eval_cfg)
        task_checker = TaskAlignmentChecker(yaml_doc, parser, goals, output_dir)

        checks = []
        checks.extend(scene_checker.run_all())
        checks.extend(mdp_checker.run_all())
        checks.extend(task_checker.run_all())

        # Phase 2: Runtime analysis
        runtime_checks = []
        if not skip_runtime:
            console.print("[bold]Phase 2:[/bold] Runtime analysis...")
            runtime_checker = RuntimeValidityChecker(
                output_dir, parser, yaml_doc, runtime_cfg
            )
            runtime_checker.execute()
            runtime_checks = runtime_checker.run_all()
            checks.extend(runtime_checks)

            # Also update obs_validity with runtime data
            if runtime_checker.results:
                self._update_obs_validity(checks, runtime_checker.results)
        else:
            # Give partial credit for runtime if skipped
            for name, max_pts in [("env_creation", 5), ("reset_step_cycle", 5),
                                   ("reward_computation", 5), ("physics_stability", 5)]:
                checks.append(_check("runtime_validity", name, 0, max_pts, "skipped"))

        # Phase 3: Aggregate
        report = self._aggregate(checks, runtime_cfg if not skip_runtime else None)
        self._print_report(report)

        # Save report
        report_path = output_dir / "eval_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\n  Report saved to: {report_path}")

        return report

    def _update_obs_validity(self, checks: list[dict], results: dict):
        """Update obs_validity check with runtime NaN/Inf data."""
        obs_ranges = results.get("obs_ranges", {})
        has_nan = any(r.get("has_nan") for r in obs_ranges.values() if isinstance(r, dict))
        has_inf = any(r.get("has_inf") for r in obs_ranges.values() if isinstance(r, dict))

        for check in checks:
            if check["check"] == "observation_validity":
                if has_nan or has_inf:
                    check["score"] = max(check["score"] - 2, 0)
                    check["status"] = "FAIL" if check["score"] == 0 else "WARN"
                    check["details"] += f". Runtime: NaN={has_nan}, Inf={has_inf}"
                break

    def _aggregate(self, checks: list[dict],
                   runtime_checker_results: dict | None) -> dict:
        """Combine all checks into final report."""
        breakdown = {}
        for cat in ("scene_fidelity", "mdp_correctness", "task_alignment", "runtime_validity"):
            cat_checks = [c for c in checks if c["category"] == cat]
            breakdown[cat] = {
                "score": sum(c["score"] for c in cat_checks),
                "max": sum(c["max"] for c in cat_checks),
            }

        total = sum(b["score"] for b in breakdown.values())
        issues = [c["details"] for c in checks if c["status"] == "FAIL"]

        return {
            "total_score": total,
            "checklist": checks,
            "breakdown": breakdown,
            "issues": issues,
        }

    def _print_report(self, report: dict):
        """Pretty-print evaluation report."""
        table = Table(title=f"Evaluation Score: {report['total_score']}/100")
        table.add_column("Category", style="bold")
        table.add_column("Check")
        table.add_column("Status", justify="center")
        table.add_column("Score", justify="right")
        table.add_column("Details", max_width=50)

        status_style = {"PASS": "green", "WARN": "yellow", "FAIL": "red"}

        for check in report["checklist"]:
            style = status_style.get(check["status"], "white")
            table.add_row(
                check["category"],
                check["check"],
                f"[{style}]{check['status']}[/{style}]",
                f"{check['score']}/{check['max']}",
                check["details"][:50],
            )

        console.print(table)

        # Breakdown summary
        console.print("\n[bold]Breakdown:[/bold]")
        for cat, scores in report["breakdown"].items():
            pct = round(100 * scores["score"] / scores["max"]) if scores["max"] else 0
            bar = "\u2588" * (pct // 5) + "\u2591" * (20 - pct // 5)
            console.print(f"  {cat:25s} {scores['score']:2d}/{scores['max']:2d}  {bar}  {pct}%")

        if report["issues"]:
            console.print(f"\n[red bold]Issues ({len(report['issues'])}):[/red bold]")
            for issue in report["issues"][:10]:
                console.print(f"  - {issue}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM-generated IsaacLab environment against YAML task document"
    )
    parser.add_argument("output_dir", help="Path to generated code directory")
    parser.add_argument("yaml_path", help="Path to source YAML task document")
    parser.add_argument("--skip-runtime", action="store_true",
                        help="Skip runtime evaluation (static analysis only)")
    parser.add_argument("--config", type=str, help="Path to eval config YAML")
    args = parser.parse_args()

    evaluator = IsaacLabEvaluator(config_path=args.config)
    report = evaluator.evaluate(
        output_dir=args.output_dir,
        yaml_path=args.yaml_path,
        skip_runtime=args.skip_runtime,
    )
    sys.exit(0 if report["total_score"] >= 70 else 1)


if __name__ == "__main__":
    main()
