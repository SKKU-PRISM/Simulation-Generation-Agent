"""Category 1: Scene Fidelity (30 pts) - Compare generated scene against YAML task document."""

from .parser import _check, EnvCfgParser


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

    @staticmethod
    def _expected_robot_cfg(robot_type: str | None) -> str | None:
        cfg_map = {
            "franka": "FRANKA_PANDA_CFG",
            "ur10": "UR10e_ROBOTIQ_2F_85_CFG",
            "ur10e": "UR10e_ROBOTIQ_2F_85_CFG",
        }
        return cfg_map.get((robot_type or "franka").lower())

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
            expected_robot_cfg = self._expected_robot_cfg(asset.get("robot_type"))
            if name in entity_names_lower or name in prim_segments:
                matched += 1
            # Also check if robot is assigned in __post_init__
            elif asset.get("type") == "articulation" and expected_robot_cfg and expected_robot_cfg in self.parser.source:
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
        """(8 pts) Key asset attributes (path/pose/scale/color) match YAML."""
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

            yaml_scale = asset.get("scale")
            cfg_scale = entity.get("scale")
            if yaml_scale and cfg_scale:
                total += 1
                if self._pos_match(yaml_scale, cfg_scale, 0.001):
                    correct += 1
                else:
                    issues.append(f"{asset['name']} scale mismatch: yaml={yaml_scale} cfg={cfg_scale}")

            yaml_color = asset.get("color")
            cfg_color = entity.get("color")
            if yaml_color and cfg_color:
                total += 1
                if self._pos_match(yaml_color, cfg_color, 0.05):
                    correct += 1
                else:
                    issues.append(f"{asset['name']} color mismatch: yaml={yaml_color} cfg={cfg_color}")

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
        expected = self._expected_robot_cfg(robot_type)

        if expected is None:
            if any("Articulation" in str(e.get("type", "")) for e in self.scene_entities.values()):
                return _check(self.CAT, "robot_config", 4, 4, f"Robot entity exists for {robot_type}")
            return _check(self.CAT, "robot_config", 0, 4, f"No robot entity found for {robot_type}")

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
