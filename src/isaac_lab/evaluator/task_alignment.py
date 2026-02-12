"""Category 3: Task Alignment (25 pts) - Check if YAML goal conditions are reflected in generated code."""

from pathlib import Path

from .parser import _check, EnvCfgParser


# ---------------------------------------------------------------------------
# Category 3: Task Alignment (25 pts)
# ---------------------------------------------------------------------------

class TaskAlignmentChecker:
    """Check if YAML goal conditions are reflected in generated code."""

    CAT = "task_alignment"

    def __init__(self, yaml_doc: dict, parser: EnvCfgParser,
                 normalized_goals: list[dict], output_dir: Path):
        self.yaml_doc = yaml_doc
        self.parser = parser
        self.goals = normalized_goals
        self.output_dir = output_dir

    def check_goal_condition_mapping(self) -> dict:
        """(10 pts) Each YAML condition maps to termination or reward."""
        if not self.goals:
            return _check(self.CAT, "goal_condition_mapping", 10, 10,
                           "No goals defined (skip)")

        terms = self.parser.extract_termination_terms()
        rewards = self.parser.extract_reward_terms()
        custom_funcs = self.parser.extract_custom_mdp_functions(self.output_dir / "mdp")
        custom_names = {f["name"] for f in custom_funcs}

        # All func names across terminations and rewards
        all_funcs = {t["func"] for t in terms} | {r["func"] for r in rewards}
        # Also check custom mdp funcs used
        all_funcs |= {f"mdp.{n}" for n in custom_names}

        mapped = 0
        unmapped = []
        for goal in self.goals:
            rel = goal.get("relation", "")
            subj = goal.get("subject", "")

            if rel in ("stacked_below", "xy_aligned"):
                # Need custom stacking function
                if any("stack" in f for f in all_funcs | custom_names):
                    mapped += 1
                else:
                    unmapped.append(f"{rel}({subj})")
            elif rel == "height_above":
                if any("height" in f for f in all_funcs):
                    mapped += 1
                else:
                    unmapped.append(f"height_above({subj})")
            elif rel in ("at_position", "on_surface", "above", "position_above"):
                # Any custom reward/termination referencing subject
                entity_terms = {t.get("asset_cfg", "") for t in terms}
                reward_funcs = {r["func"] for r in rewards}
                if subj in entity_terms or any(subj in f for f in reward_funcs | custom_names):
                    mapped += 1
                else:
                    unmapped.append(f"{rel}({subj})")
            elif rel == "open":
                # Gripper check - usually implicit
                mapped += 1
            else:
                # Unknown relation, be lenient
                mapped += 0.5

        total = len(self.goals)
        score = round(10 * mapped / total) if total else 10
        details = f"{int(mapped)}/{total} conditions mapped"
        if unmapped:
            details += f". Unmapped: {unmapped[:5]}"
        return _check(self.CAT, "goal_condition_mapping", score, 10, details)

    def check_threshold_preservation(self) -> dict:
        """(8 pts) Numeric thresholds from YAML preserved in code."""
        if not self.goals:
            return _check(self.CAT, "threshold_preservation", 8, 8,
                           "No goals (skip)")

        source = self.parser.source
        # Also check mdp/ files
        mdp_dir = self.output_dir / "mdp"
        if mdp_dir.exists():
            for py in mdp_dir.glob("*.py"):
                source += "\n" + py.read_text()

        found = 0
        total = 0
        missing = []
        for goal in self.goals:
            for key in ("value", "tolerance"):
                val = goal.get(key)
                if val is not None and isinstance(val, (int, float)):
                    total += 1
                    # Search for this value in code (with some float tolerance)
                    val_str = str(val)
                    if val_str in source:
                        found += 1
                    elif f"{val:.4f}" in source or f"{val:.3f}" in source:
                        found += 1
                    else:
                        missing.append(f"{goal.get('relation','?')}:{key}={val}")

        score = round(8 * found / total) if total else 8
        details = f"{found}/{total} thresholds found"
        if missing:
            details += f". Missing: {missing[:5]}"
        return _check(self.CAT, "threshold_preservation", score, 8, details)

    def check_custom_mdp_validity(self) -> dict:
        """(7 pts) Custom functions exist with correct signatures."""
        mdp_dir = self.output_dir / "mdp"
        custom_funcs = self.parser.extract_custom_mdp_functions(mdp_dir)

        # If no custom functions needed (no complex goals), that's OK
        needs_custom = any(
            g.get("relation") in ("stacked_below", "xy_aligned", "at_position", "position_above")
            for g in self.goals
        )

        if not needs_custom:
            return _check(self.CAT, "custom_mdp_validity", 7, 7,
                           "No custom MDP needed for this task")

        if not custom_funcs:
            return _check(self.CAT, "custom_mdp_validity", 0, 7,
                           "Custom MDP functions needed but none found")

        pts = 3  # functions exist
        details = [f"{len(custom_funcs)} custom functions"]

        # Check signatures: first param should be 'env'
        valid_sig = all(
            len(f["params"]) >= 1 and f["params"][0] == "env"
            for f in custom_funcs
        )
        if valid_sig:
            pts += 2
            details.append("valid signatures")

        # Check __init__.py imports them
        init_file = mdp_dir / "__init__.py"
        if init_file.exists():
            init_content = init_file.read_text()
            for f in custom_funcs:
                if f["file"].replace(".py", "") in init_content:
                    pts += 2
                    details.append("properly imported")
                    break

        return _check(self.CAT, "custom_mdp_validity", min(pts, 7), 7,
                       ". ".join(details))

    def run_all(self) -> list[dict]:
        return [
            self.check_goal_condition_mapping(),
            self.check_threshold_preservation(),
            self.check_custom_mdp_validity(),
        ]
