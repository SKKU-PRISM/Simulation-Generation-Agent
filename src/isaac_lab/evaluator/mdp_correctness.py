"""Category 2: MDP Correctness (25 pts) - Check observation/action/reward/termination/event config."""

from .parser import _check, EnvCfgParser


# ---------------------------------------------------------------------------
# Category 2: MDP Correctness (25 pts)
# ---------------------------------------------------------------------------

class MDPCorrectnessChecker:
    """Check observation/action/reward/termination/event config."""

    CAT = "mdp_correctness"

    def __init__(self, yaml_doc: dict, parser: EnvCfgParser, config: dict):
        self.yaml_doc = yaml_doc
        self.parser = parser
        self.cfg = config
        self.yaml_assets = yaml_doc.get("assets", [])
        self.rigid_objects = [a for a in self.yaml_assets if a.get("type") == "rigid"]

    def check_observation_coverage(self) -> dict:
        """(7 pts) All rigid objects have obs terms."""
        obs_terms = self.parser.extract_observation_terms()
        observed_entities = {t["asset_cfg"] for t in obs_terms if t.get("asset_cfg")}

        covered = 0
        missing = []
        for obj in self.rigid_objects:
            name = obj["name"]
            if name in observed_entities:
                covered += 1
            else:
                missing.append(name)

        # Also check robot obs (joint_pos, joint_vel, last_action)
        robot_funcs = {t["func"] for t in obs_terms}
        has_robot_obs = any("joint_pos" in f or "joint_vel" in f for f in robot_funcs)

        total = len(self.rigid_objects) + (1 if self.rigid_objects else 0)  # +1 for robot
        covered += 1 if has_robot_obs else 0

        score = round(7 * covered / total) if total else 7
        details = f"{covered}/{total} entities observed"
        if missing:
            details += f". Missing obs: {missing}"
        return _check(self.CAT, "observation_coverage", score, 7, details)

    def check_observation_validity(self) -> dict:
        """(3 pts) Obs functions are valid mdp names (runtime NaN check deferred)."""
        obs_terms = self.parser.extract_observation_terms()
        valid_funcs = set(self.cfg.get("valid_obs_functions", []))
        # Also accept custom mdp.* functions
        invalid = []
        for t in obs_terms:
            func = t["func"]
            if func not in valid_funcs and not func.startswith("mdp."):
                invalid.append(func)

        if not invalid:
            return _check(self.CAT, "observation_validity", 3, 3, "All obs funcs valid")
        score = max(0, 3 - len(invalid))
        return _check(self.CAT, "observation_validity", score, 3,
                       f"Invalid funcs: {invalid}")

    def check_action_space(self) -> dict:
        """(5 pts) Matches robot DOF."""
        actions = self.parser.extract_action_config()
        robot_asset = next((a for a in self.yaml_assets if a.get("type") == "articulation"), None)
        if not robot_asset:
            return _check(self.CAT, "action_space", 5, 5, "No robot (skip)")

        pts = 0
        details = []
        if "arm" in actions:
            pts += 3
            details.append("arm action found")
        if "gripper" in actions:
            pts += 2
            details.append("gripper action found")

        return _check(self.CAT, "action_space", pts, 5, ", ".join(details) or "no actions")

    def check_reward_structure(self) -> dict:
        """(5 pts) Has task-relevant rewards."""
        if self.parser.rewards_is_none():
            if self.cfg.get("allow_none_rewards", True):
                return _check(
                    self.CAT,
                    "reward_structure",
                    3,
                    5,
                    "rewards=None — treated as validation-only setup",
                )
            return _check(self.CAT, "reward_structure", 0, 5, "rewards=None — no reward signal")

        rewards = self.parser.extract_reward_terms()
        if not rewards:
            return _check(self.CAT, "reward_structure", 1, 5,
                           "Rewards section exists but no RewTerm found")

        # Check weights
        all_have_weight = all(r.get("weight") is not None for r in rewards)
        has_task_reward = any(
            "action_rate" not in r["func"] and "joint_vel" not in r["func"]
            for r in rewards
        )

        pts = 2  # at least some rewards
        if all_have_weight:
            pts += 1
        if has_task_reward:
            pts += 2
        return _check(self.CAT, "reward_structure", min(pts, 5), 5,
                       f"{len(rewards)} rewards, task_relevant={has_task_reward}")

    def check_termination_coverage(self) -> dict:
        """(3 pts) time_out + safety terminations."""
        terms = self.parser.extract_termination_terms()
        pts = 0
        details = []

        has_timeout = any(t.get("time_out") or "time_out" in t.get("func", "") for t in terms)
        if has_timeout:
            pts += 1
            details.append("time_out")

        has_safety = any("height_below" in t.get("func", "") or "illegal_contact" in t.get("func", "")
                         for t in terms)
        if has_safety:
            pts += 1
            details.append("safety")

        if len(terms) >= 2:
            pts = min(pts + 1, 3)
            details.append(f"{len(terms)} total")

        return _check(self.CAT, "termination_coverage", pts, 3, ", ".join(details) or "none")

    def check_event_coverage(self) -> dict:
        """(2 pts) Reset events for robot + objects."""
        events = self.parser.extract_event_terms()
        reset_events = [e for e in events if e.get("mode") == "reset"]

        has_robot_reset = any(
            e.get("asset_cfg") == "robot" or "scene_to_default" in e.get("func", "")
            for e in reset_events
        )
        object_resets = {e.get("asset_cfg") for e in reset_events if e.get("asset_cfg") and e.get("asset_cfg") != "robot"}

        pts = 0
        if has_robot_reset:
            pts += 1
        if object_resets:
            pts += 1

        return _check(self.CAT, "event_coverage", pts, 2,
                       f"robot_reset={has_robot_reset}, object_resets={object_resets or 'none'}")

    def run_all(self) -> list[dict]:
        return [
            self.check_observation_coverage(),
            self.check_observation_validity(),
            self.check_action_space(),
            self.check_reward_structure(),
            self.check_termination_coverage(),
            self.check_event_coverage(),
        ]
