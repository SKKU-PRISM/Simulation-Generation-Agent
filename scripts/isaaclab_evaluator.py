"""IsaacLab Environment Evaluation System.

Evaluates LLM-generated IsaacLab ManagerBasedRLEnv code against
the source YAML task document across 4 categories (100 points):
  1. Scene Fidelity (30)   — YAML assets vs generated scene
  2. MDP Correctness (25)  — obs/action/reward/term/event config
  3. Task Alignment (25)   — YAML goal mapped to code
  4. Runtime Validity (20) — actual execution in IsaacLab

Usage:
    python scripts/isaaclab_evaluator.py <output_dir> <yaml_path>
    python scripts/isaaclab_evaluator.py --skip-runtime <output_dir> <yaml_path>
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
console = Console()


# ---------------------------------------------------------------------------
# Helper: Check result dataclass
# ---------------------------------------------------------------------------

def _check(category: str, name: str, score: int, max_score: int,
           details: str) -> dict:
    """Create a standardised check result."""
    return {
        "category": category,
        "check": name,
        "status": "PASS" if score == max_score else ("WARN" if score > 0 else "FAIL"),
        "score": score,
        "max": max_score,
        "details": details,
    }


# ---------------------------------------------------------------------------
# Goal Normalizer
# ---------------------------------------------------------------------------

class GoalNormalizer:
    """Convert both YAML goal formats to a unified condition list."""

    @staticmethod
    def normalize(goal: dict, assets: list[dict]) -> list[dict]:
        """Return list of dicts with keys: subject, relation, target, value, tolerance."""
        if not goal:
            return []

        # Format A: conditions list (lift, pick_place, sort, cabinet)
        if "conditions" in goal:
            return [dict(c) for c in goal["conditions"]]

        # Format B: success_criteria dict (stack tasks)
        if "success_criteria" in goal:
            return GoalNormalizer._normalize_stack_criteria(
                goal["success_criteria"], goal.get("description", ""), assets
            )

        return []

    @staticmethod
    def _normalize_stack_criteria(criteria: dict, description: str,
                                  assets: list[dict]) -> list[dict]:
        """Convert stack success_criteria to condition list."""
        # Find stackable objects (rigid, not robot/table)
        cubes = [a["name"] for a in assets
                 if a.get("type") == "rigid"]

        # Parse order from description if possible (e.g. "Cube_1 bottom -> Cube_2 middle -> Cube_3 top")
        # Default: assets list order (bottom to top)
        ordered = cubes  # already in YAML order

        conditions = []
        xy_thr = criteria.get("xy_threshold", 0.04)
        h_diff = criteria.get("height_diff", 0.0468)
        h_thr = criteria.get("height_threshold", 0.005)

        for i in range(len(ordered) - 1):
            bottom, top = ordered[i], ordered[i + 1]
            conditions.append({
                "subject": bottom, "relation": "stacked_below", "target": top,
                "value": h_diff, "tolerance": h_thr,
            })
            conditions.append({
                "subject": bottom, "relation": "xy_aligned", "target": top,
                "tolerance": xy_thr,
            })

        if criteria.get("gripper_must_be_open"):
            conditions.append({
                "subject": "gripper", "relation": "open", "value": True,
            })

        return conditions


# ---------------------------------------------------------------------------
# AST-based env_cfg.py Parser
# ---------------------------------------------------------------------------

class EnvCfgParser:
    """Parse env_cfg.py using Python AST to extract structured information."""

    def __init__(self, env_cfg_path: str | Path):
        self.source = Path(env_cfg_path).read_text()
        self.tree = ast.parse(self.source)
        self._lines = self.source.splitlines()

    # --- Class finding ---

    def find_env_cfg_class(self) -> str | None:
        """Find class name inheriting from ManagerBasedRLEnvCfg."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name and "ManagerBasedRLEnvCfg" in base_name:
                        return node.name
        return None

    def find_scene_class(self) -> str | None:
        """Find class inheriting from InteractiveSceneCfg."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = self._get_name(base)
                    if base_name and "InteractiveSceneCfg" in base_name:
                        return node.name
        return None

    # --- Scene entity extraction ---

    def extract_scene_entities(self) -> dict[str, dict]:
        """Extract scene entity definitions from InteractiveSceneCfg subclass.

        Returns: {name: {type, prim_path, pos, rot, usd_path, scale}}
        """
        scene_cls = self._find_class_node("InteractiveSceneCfg")
        if not scene_cls:
            return {}

        entities = {}
        for stmt in scene_cls.body:
            info = self._parse_assignment(stmt)
            if info:
                entities[info["name"]] = info
        return entities

    def _parse_assignment(self, stmt) -> dict | None:
        """Parse a class-level assignment like `cube_1 = RigidObjectCfg(...)`."""
        name = None
        value_node = None

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            name = stmt.target.id
            value_node = stmt.value

        if not name or not isinstance(value_node, ast.Call):
            return None

        call_name = self._get_name(value_node.func)
        if not call_name:
            return None

        # Only parse known config types
        known_types = ("RigidObjectCfg", "AssetBaseCfg", "ArticulationCfg",
                       "FrameTransformerCfg")
        if not any(t in call_name for t in known_types):
            return None

        info = {"name": name, "type": call_name}
        kw = self._extract_keywords(value_node)

        info["prim_path"] = kw.get("prim_path", "")

        # init_state → pos, rot
        init_state = kw.get("init_state")
        if isinstance(init_state, dict):
            info["pos"] = init_state.get("pos")
            info["rot"] = init_state.get("rot")

        # spawn → usd_path, scale
        spawn = kw.get("spawn")
        if isinstance(spawn, dict):
            info["usd_path"] = spawn.get("usd_path", "")
            info["scale"] = spawn.get("scale")
            info["spawn_type"] = spawn.get("_call_name", "")

        return info

    # --- Observation extraction ---

    def extract_observation_terms(self) -> list[dict]:
        """Extract ObsTerm definitions. Returns [{name, func, asset_cfg}]."""
        terms = []
        # Find ObsTerm(...) calls in the source via regex (more reliable for nested classes)
        pattern = r"(\w+)\s*=\s*ObsTerm\(func\s*=\s*([\w.]+)(?:,\s*params\s*=\s*\{[^}]*\"asset_cfg\"\s*:\s*SceneEntityCfg\(\"(\w+)\"\))?"
        for m in re.finditer(pattern, self.source):
            terms.append({
                "name": m.group(1),
                "func": m.group(2),
                "asset_cfg": m.group(3),  # None if no asset_cfg
            })
        return terms

    # --- Reward extraction ---

    def extract_reward_terms(self) -> list[dict]:
        """Extract RewTerm definitions. Returns [{name, func, weight}]."""
        terms = []
        pattern = r"(\w+)\s*=\s*RewTerm\(func\s*=\s*([\w.]+).*?weight\s*=\s*([+-]?[\d.eE+-]+)"
        for m in re.finditer(pattern, self.source, re.DOTALL):
            terms.append({
                "name": m.group(1),
                "func": m.group(2),
                "weight": float(m.group(3)),
            })
        return terms

    # --- Termination extraction ---

    def extract_termination_terms(self) -> list[dict]:
        """Extract DoneTerm definitions. Returns [{name, func, params}]."""
        terms = []
        # Match multi-line: name = DoneTerm(\n  func=mdp.xxx, ...)
        pattern = r"(\w+)\s*=\s*DoneTerm\(\s*func\s*=\s*([\w.]+)"
        for m in re.finditer(pattern, self.source, re.DOTALL):
            term = {"name": m.group(1), "func": m.group(2)}
            # Get the full DoneTerm(...) block by finding matching paren
            block = self._get_call_block(m.start())
            if "time_out=True" in block:
                term["time_out"] = True
            asset_m = re.search(r'SceneEntityCfg\("(\w+)"\)', block)
            if asset_m:
                term["asset_cfg"] = asset_m.group(1)
            terms.append(term)
        return terms

    # --- Event extraction ---

    def extract_event_terms(self) -> list[dict]:
        """Extract EventTerm definitions. Returns [{name, func, mode, asset_cfg}]."""
        terms = []
        pattern = r"(\w+)\s*=\s*EventTerm\(\s*func\s*=\s*([\w.]+)"
        for m in re.finditer(pattern, self.source, re.DOTALL):
            block = self._get_call_block(m.start())
            mode_m = re.search(r'mode\s*=\s*"(\w+)"', block)
            mode = mode_m.group(1) if mode_m else "unknown"
            term = {"name": m.group(1), "func": m.group(2), "mode": mode}
            asset_m = re.search(r'SceneEntityCfg\("(\w+)"\)', block)
            if asset_m:
                term["asset_cfg"] = asset_m.group(1)
            terms.append(term)
        return terms

    # --- Action extraction ---

    def extract_action_config(self) -> dict:
        """Extract action configuration. Returns {arm: {joint_names}, gripper: {joint_names}}."""
        result = {}
        arm_m = re.search(r'JointPositionActionCfg\(.*?joint_names\s*=\s*\[([^\]]+)\]',
                          self.source, re.DOTALL)
        if arm_m:
            result["arm"] = {"joint_names": arm_m.group(1).strip()}

        grip_m = re.search(r'BinaryJointPositionActionCfg\(.*?joint_names\s*=\s*\[([^\]]+)\]',
                           self.source, re.DOTALL)
        if grip_m:
            result["gripper"] = {"joint_names": grip_m.group(1).strip()}

        return result

    # --- Simulation params ---

    def extract_simulation_params(self) -> dict:
        """Extract simulation params from __post_init__."""
        params = {}
        # self.decimation = N
        m = re.search(r"self\.decimation\s*=\s*(\d+)", self.source)
        if m:
            params["decimation"] = int(m.group(1))
        # self.episode_length_s = N
        m = re.search(r"self\.episode_length_s\s*=\s*([\d.]+)", self.source)
        if m:
            params["episode_length_s"] = float(m.group(1))
        # self.sim.dt = N
        m = re.search(r"self\.sim\.dt\s*=\s*([\d.]+)", self.source)
        if m:
            params["dt"] = float(m.group(1))
        return params

    # --- Custom MDP functions ---

    def extract_custom_mdp_functions(self, mdp_dir: str | Path) -> list[dict]:
        """Parse mdp/*.py for custom function signatures."""
        mdp_dir = Path(mdp_dir)
        funcs = []
        for py_file in mdp_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue
            try:
                tree = ast.parse(py_file.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    params = [a.arg for a in node.args.args]
                    funcs.append({
                        "file": py_file.name,
                        "name": node.name,
                        "params": params,
                    })
        return funcs

    # --- Check rewards is None ---

    def rewards_is_none(self) -> bool:
        """Check if rewards = None in the env cfg class."""
        return bool(re.search(r"rewards\s*=\s*None", self.source))

    # --- Check env_spacing ---

    def has_env_spacing(self) -> bool:
        return "env_spacing" in self.source

    # --- Check ground plane ---

    def has_ground_plane(self) -> bool:
        return "GroundPlaneCfg" in self.source

    # --- Check lighting ---

    def has_lighting(self) -> bool:
        return "DomeLightCfg" in self.source or "DistantLightCfg" in self.source

    # --- Internal helpers ---

    def _find_class_node(self, base_name: str) -> ast.ClassDef | None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    name = self._get_name(base)
                    if name and base_name in name:
                        return node
        return None

    def _get_name(self, node) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            obj = self._get_name(node.value)
            if obj:
                return f"{obj}.{node.attr}"
            return node.attr
        return None

    def _extract_keywords(self, call_node: ast.Call) -> dict:
        """Recursively extract keyword arguments from an ast.Call."""
        result = {}
        if hasattr(call_node, 'func'):
            result["_call_name"] = self._get_name(call_node.func) or ""
        for kw in call_node.keywords:
            if kw.arg:
                result[kw.arg] = self._eval_node(kw.value)
        return result

    def _eval_node(self, node):
        """Best-effort evaluation of AST node to Python value."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(e) for e in node.elts]
        if isinstance(node, ast.Call):
            return self._extract_keywords(node)
        if isinstance(node, ast.Dict):
            return {
                self._eval_node(k): self._eval_node(v)
                for k, v in zip(node.keys, node.values)
                if k is not None
            }
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            val = self._eval_node(node.operand)
            if isinstance(val, (int, float)):
                return -val
        if isinstance(node, ast.JoinedStr):
            # f-string: return raw source text
            return ast.get_source_segment(self.source, node) or "<f-string>"
        if isinstance(node, ast.Name):
            return f"<{node.id}>"
        if isinstance(node, ast.Attribute):
            return self._get_name(node) or "<attr>"
        return None

    def _get_call_block(self, char_offset: int) -> str:
        """Extract the full parenthesized block starting near char_offset."""
        # Find opening paren
        start = self.source.index("(", char_offset)
        depth = 0
        for i in range(start, len(self.source)):
            if self.source[i] == "(":
                depth += 1
            elif self.source[i] == ")":
                depth -= 1
                if depth == 0:
                    return self.source[start:i + 1]
        return self.source[start:start + 500]

    def _get_line_context(self, char_offset: int, num_lines: int = 3) -> str:
        """Get num_lines of source starting at char_offset."""
        line_no = self.source[:char_offset].count("\n")
        end = min(line_no + num_lines, len(self._lines))
        return "\n".join(self._lines[line_no:end])


# ---------------------------------------------------------------------------
# Category 1: Scene Fidelity (30 pts)
# ---------------------------------------------------------------------------

class SceneFidelityChecker:
    """Compare generated scene against YAML task document."""

    CAT = "scene_fidelity"

    def __init__(self, yaml_doc: dict, parser: EnvCfgParser, config: dict):
        self.yaml_doc = yaml_doc
        self.parser = parser
        self.cfg = config
        self.yaml_assets = yaml_doc.get("assets", [])
        self.scene_entities = parser.extract_scene_entities()

    def check_asset_completeness(self) -> dict:
        """(10 pts) Every YAML asset has a corresponding scene entity."""
        entity_names_lower = {n.lower().replace("-", "_") for n in self.scene_entities}
        # Also check prim_path segments
        prim_segments = set()
        for e in self.scene_entities.values():
            pp = e.get("prim_path", "")
            seg = pp.split("/")[-1].lower().replace("-", "_") if "/" in pp else ""
            if seg:
                prim_segments.add(seg)

        matched = 0
        missing = []
        for asset in self.yaml_assets:
            name = asset["name"].lower().replace("-", "_")
            if name in entity_names_lower or name in prim_segments:
                matched += 1
            # Also check if robot is assigned in __post_init__
            elif asset.get("type") == "articulation" and "FRANKA_PANDA_CFG" in self.parser.source:
                matched += 1
            else:
                missing.append(asset["name"])

        total = len(self.yaml_assets)
        score = round(10 * matched / total) if total else 10
        details = f"{matched}/{total} assets found"
        if missing:
            details += f". Missing: {missing}"
        return _check(self.CAT, "asset_completeness", score, 10, details)

    def check_asset_config(self) -> dict:
        """(8 pts) USD paths, positions, rotations match YAML."""
        tol = self.cfg.get("position_tolerance", 0.02)
        correct = 0
        total = 0
        issues = []

        for asset in self.yaml_assets:
            name = asset["name"].lower().replace("-", "_")
            entity = self._find_entity(name)
            if not entity:
                continue

            # Check position
            yaml_pos = asset.get("position")
            cfg_pos = entity.get("pos")
            if yaml_pos and cfg_pos:
                total += 1
                if self._pos_match(yaml_pos, cfg_pos, tol):
                    correct += 1
                else:
                    issues.append(f"{asset['name']} pos mismatch: yaml={yaml_pos} cfg={cfg_pos}")

            # Check USD path
            yaml_path = asset.get("asset_path", "")
            cfg_usd = entity.get("usd_path", "")
            if yaml_path and cfg_usd and isinstance(cfg_usd, str):
                total += 1
                # Normalize: strip template vars
                yaml_norm = yaml_path.split("}")[-1] if "}" in yaml_path else yaml_path
                cfg_norm = cfg_usd.split("}")[-1] if "}" in cfg_usd else cfg_usd
                # For f-strings, extract the path suffix
                if "/" in cfg_norm:
                    cfg_norm = "/" + cfg_norm.split("/", 1)[-1] if not cfg_norm.startswith("/") else cfg_norm
                if yaml_norm in cfg_norm or cfg_norm in yaml_norm:
                    correct += 1
                else:
                    issues.append(f"{asset['name']} path mismatch")

        score = round(8 * correct / total) if total else 8
        details = f"{correct}/{total} attributes match"
        if issues:
            details += f". Issues: {issues[:3]}"
        return _check(self.CAT, "asset_config", score, 8, details)

    def check_physics_config(self) -> dict:
        """(5 pts) timestep, decimation, episode_length match."""
        yaml_sim = self.yaml_doc.get("simulation", {})
        cfg_sim = self.parser.extract_simulation_params()

        matches = 0
        total = 0
        issues = []

        for key, yaml_key in [("decimation", "decimation"), ("dt", "timestep"),
                               ("episode_length_s", "episode_length")]:
            yaml_val = yaml_sim.get(yaml_key)
            cfg_val = cfg_sim.get(key)
            if yaml_val is not None:
                total += 1
                if cfg_val is not None and abs(float(yaml_val) - float(cfg_val)) < 0.001:
                    matches += 1
                else:
                    issues.append(f"{yaml_key}: yaml={yaml_val} cfg={cfg_val}")

        score = round(5 * matches / total) if total else 5
        details = f"{matches}/{total} params match"
        if issues:
            details += f". {issues}"
        return _check(self.CAT, "physics_config", score, 5, details)

    def check_robot_config(self) -> dict:
        """(4 pts) Correct robot config used."""
        robot_asset = next((a for a in self.yaml_assets if a.get("type") == "articulation"), None)
        if not robot_asset:
            return _check(self.CAT, "robot_config", 4, 4, "No robot in YAML (skip)")

        robot_type = robot_asset.get("robot_type", "franka")
        cfg_map = {"franka": "FRANKA_PANDA_CFG"}
        expected = cfg_map.get(robot_type)

        if expected and expected in self.parser.source:
            return _check(self.CAT, "robot_config", 4, 4, f"{expected} found")

        # Check if any robot entity exists
        if any("Articulation" in str(e.get("type", "")) for e in self.scene_entities.values()):
            return _check(self.CAT, "robot_config", 2, 4, "Robot entity exists but not standard cfg")

        return _check(self.CAT, "robot_config", 0, 4, f"Expected {expected}, not found")

    def check_scene_structure(self) -> dict:
        """(3 pts) Ground plane, lighting, env_spacing set."""
        pts = 0
        details = []
        if self.parser.has_ground_plane():
            pts += 1
            details.append("ground")
        if self.parser.has_lighting():
            pts += 1
            details.append("light")
        if self.parser.has_env_spacing():
            pts += 1
            details.append("env_spacing")
        return _check(self.CAT, "scene_structure", pts, 3, ", ".join(details) or "none")

    def run_all(self) -> list[dict]:
        return [
            self.check_asset_completeness(),
            self.check_asset_config(),
            self.check_physics_config(),
            self.check_robot_config(),
            self.check_scene_structure(),
        ]

    # --- Helpers ---

    def _find_entity(self, name: str) -> dict | None:
        name_l = name.lower().replace("-", "_")
        for ename, einfo in self.scene_entities.items():
            if ename.lower().replace("-", "_") == name_l:
                return einfo
        return None

    @staticmethod
    def _pos_match(a, b, tol: float) -> bool:
        if not a or not b:
            return False
        try:
            a = [float(x) for x in a[:3]]
            b = [float(x) for x in b[:3]]
            return all(abs(ai - bi) < tol for ai, bi in zip(a, b))
        except (TypeError, ValueError):
            return False


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
            return _check(self.CAT, "reward_structure", 0, 5,
                           "rewards=None — no reward signal")

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


# ---------------------------------------------------------------------------
# Category 4: Runtime Validity (20 pts)
# ---------------------------------------------------------------------------

class RuntimeValidityChecker:
    """Generate and execute eval_runner.py, parse results."""

    CAT = "runtime_validity"

    def __init__(self, output_dir: Path, parser: EnvCfgParser,
                 yaml_doc: dict, config: dict):
        self.output_dir = output_dir
        self.parser = parser
        self.yaml_doc = yaml_doc
        self.cfg = config
        self.results: dict | None = None

    def generate_eval_runner(self) -> Path:
        """Generate eval_runner.py in output_dir."""
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
        runner_path = self.output_dir / "eval_runner.py"
        runner_path.write_text(code)
        return runner_path

    def execute(self) -> bool:
        """Execute eval_runner.py via conda subprocess."""
        runner = self.generate_eval_runner()
        isaaclab_path = Path(
            self.cfg.get("isaaclab_path")
            or os.environ.get("ISAACLAB_PATH")
            or os.path.expanduser("~/workspace/IsaacLab")
        )
        conda_env = self.cfg.get("conda_env", "env_isaaclab")
        launcher = isaaclab_path / "isaaclab.sh"
        timeout = self.cfg.get("timeout", 300)

        isaaclab_cmd = f"{launcher} -p {runner} --num_envs {self.cfg.get('num_envs', 2)}"
        if self.cfg.get("headless", True):
            isaaclab_cmd += " --headless"

        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output",
               "bash", "-c", isaaclab_cmd]

        marker_file = self.output_dir / ".eval_marker"
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
        results_file = self.output_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                self.results = json.load(f)
            return True
        return False

    def score_env_creation(self) -> dict:
        """(5 pts) Env created successfully."""
        if not self.results:
            return _check(self.CAT, "env_creation", 0, 5, "No runtime results")
        if self.results.get("env_created"):
            return _check(self.CAT, "env_creation", 5, 5, "Environment created")
        errors = self.results.get("errors", [])
        msg = errors[0]["message"][:100] if errors else "unknown error"
        return _check(self.CAT, "env_creation", 0, 5, f"Failed: {msg}")

    def score_reset_step(self) -> dict:
        """(5 pts) N+ steps completed."""
        if not self.results:
            return _check(self.CAT, "reset_step_cycle", 0, 5, "No runtime results")
        steps = self.results.get("steps_completed", 0)
        target = self.cfg.get("eval_steps", 20)
        if steps >= target:
            return _check(self.CAT, "reset_step_cycle", 5, 5,
                           f"{steps} steps completed")
        if steps > 0:
            score = round(5 * steps / target)
            return _check(self.CAT, "reset_step_cycle", max(score, 1), 5,
                           f"Only {steps}/{target} steps")
        return _check(self.CAT, "reset_step_cycle", 0, 5, "No steps completed")

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
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
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
