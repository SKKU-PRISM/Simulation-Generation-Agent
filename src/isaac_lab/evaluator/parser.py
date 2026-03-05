"""YAML goal normalization and AST-based env_cfg.py parsing."""

import ast
import json
import re
from pathlib import Path


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
        entities = {}
        scene_cls = self._find_class_node("InteractiveSceneCfg")
        if scene_cls:
            for stmt in scene_cls.body:
                info = self._parse_assignment(stmt)
                if info:
                    entities[info["name"]] = info

        # Also parse dynamic scene assignments in ManagerBasedRLEnvCfg.__post_init__:
        #   self.scene.<entity_name> = <CfgCall>(...)
        env_cfg_cls = self._find_class_node("ManagerBasedRLEnvCfg")
        if env_cfg_cls:
            entities.update(self._extract_post_init_scene_entities(env_cfg_cls))

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

        if not name:
            return None
        return self._parse_scene_call_assignment(name, value_node)

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

    def _parse_scene_call_assignment(self, name: str, value_node) -> dict | None:
        """Parse a scene assignment call into normalized entity metadata."""
        if not isinstance(value_node, ast.Call):
            return None

        call_name = self._get_name(value_node.func)
        if not call_name:
            return None

        known_types = ("RigidObjectCfg", "AssetBaseCfg", "ArticulationCfg", "FrameTransformerCfg")
        if not any(t in call_name for t in known_types):
            return None

        info = {"name": name, "type": call_name}
        kw = self._extract_keywords(value_node)

        info["prim_path"] = kw.get("prim_path", "")

        init_state = kw.get("init_state")
        if isinstance(init_state, dict):
            info["pos"] = init_state.get("pos")
            info["rot"] = init_state.get("rot")

        spawn = kw.get("spawn")
        if isinstance(spawn, dict):
            info["usd_path"] = spawn.get("usd_path", "")
            info["scale"] = spawn.get("scale")
            info["spawn_type"] = spawn.get("_call_name", "")

        return info

    def _extract_post_init_scene_entities(self, env_cfg_cls: ast.ClassDef) -> dict[str, dict]:
        """Extract `self.scene.<name> = ...` entities from ManagerBasedRLEnvCfg.__post_init__."""
        post_init = next(
            (
                stmt
                for stmt in env_cfg_cls.body
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__post_init__"
            ),
            None,
        )
        if post_init is None:
            return {}

        entities = {}
        for stmt in ast.walk(post_init):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    entity_name = self._scene_entity_name_from_target(target)
                    if entity_name:
                        info = self._parse_scene_call_assignment(entity_name, stmt.value)
                        if info:
                            entities[entity_name] = info
            elif isinstance(stmt, ast.AnnAssign):
                entity_name = self._scene_entity_name_from_target(stmt.target)
                if entity_name and stmt.value is not None:
                    info = self._parse_scene_call_assignment(entity_name, stmt.value)
                    if info:
                        entities[entity_name] = info

        return entities

    @staticmethod
    def _scene_entity_name_from_target(target) -> str | None:
        """Return entity name for `self.scene.<name>` assignment targets."""
        if not isinstance(target, ast.Attribute):
            return None
        scene_attr = target.value
        if not isinstance(scene_attr, ast.Attribute):
            return None
        root = scene_attr.value
        if not isinstance(root, ast.Name):
            return None
        if root.id != "self" or scene_attr.attr != "scene":
            return None
        return target.attr

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
