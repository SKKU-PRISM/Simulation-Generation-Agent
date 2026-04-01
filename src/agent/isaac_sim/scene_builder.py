#!/usr/bin/env python3
"""
Scene Builder - Builds Isaac Sim scenes from Task Documents via MCP

This module reads YAML task documents and creates the corresponding
simulation environment using MCP tools.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Any

from src.agent.common.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class SceneBuilder:
    """Builds Isaac Sim scenes from Task Documents."""

    def __init__(self, mcp_client: MCPClient = None):
        self.mcp = mcp_client or MCPClient()

    def build_from_yaml(self, yaml_path: str | Path) -> dict:
        """Build scene from a YAML task document."""
        yaml_path = Path(yaml_path)

        if not yaml_path.exists():
            return {"success": False, "error": f"File not found: {yaml_path}"}

        with open(yaml_path, "r") as f:
            document = yaml.safe_load(f)

        return self.build(document)

    def build(self, document: dict) -> dict:
        """Build scene from a task document dictionary."""
        results = {
            "success": True,
            "created_prims": [],
            "errors": [],
            "warnings": []
        }

        # Reset randomization state
        self._placed_positions = []

        # 1. Verify connection
        logger.info("Verifying Isaac Sim connection...")
        scene_info = self.mcp.get_scene_info()
        if "error" in scene_info:
            results["success"] = False
            results["errors"].append(f"Connection failed: {scene_info['error']}")
            return results

        logger.info("Connection verified")

        # 2. Clear existing scene and create new stage
        logger.info("Creating new stage...")
        clear_result = self._clear_scene()
        if not clear_result.get("success", True):
            results["warnings"].append("Failed to clear scene, continuing anyway")

        # 3. Create physics scene
        logger.info("Setting up physics...")
        physics_config = document.get("simulation", {})
        physics_result = self._create_physics_scene(physics_config)
        if physics_result.get("success"):
            results["created_prims"].append("/World/PhysicsScene")

        # 4. Create ground plane
        ground_config = document.get("scene", {}).get("ground", {})
        if ground_config.get("enabled", True):
            logger.info("Creating ground plane...")
            ground_result = self._create_ground(ground_config)
            if ground_result.get("success"):
                results["created_prims"].append("/World/GroundPlane")

        # 5. Setup lighting
        lighting_config = document.get("scene", {}).get("lighting", {})
        if lighting_config:
            logger.info("Setting up lighting...")
            self._setup_lighting(lighting_config)

        # 6. Create assets
        assets = document.get("assets", [])
        for asset in assets:
            logger.info(f"Creating asset: {asset.get('name', 'unnamed')}")
            asset_result = self._create_asset(asset)
            if asset_result.get("success"):
                results["created_prims"].append(asset.get("prim_path", "unknown"))
            else:
                results["errors"].append(
                    f"Failed to create {asset.get('name')}: {asset_result.get('error', 'unknown')}"
                )

        # 7. Setup camera
        camera_config = document.get("camera", {})
        if camera_config:
            logger.info("Setting up camera...")
            cam_result = self._setup_camera(camera_config)
            if cam_result.get("success"):
                results["created_prims"].append(camera_config.get("prim_path", "/World/Camera"))

        # Check for errors
        if results["errors"]:
            results["success"] = False

        return results

    def _clear_scene(self) -> dict:
        """Clear the current scene using reset_scene command."""
        result = self.mcp.reset_scene(keep_physics=False)
        return {"success": result.get("status") == "success"}

    def _create_physics_scene(self, config: dict) -> dict:
        """Create physics scene with gravity, timestep, and PhysX settings."""
        gravity = config.get("gravity", [0, 0, -9.81])
        timestep = config.get("timestep", 0.01)
        physx = config.get("physx", {})
        bounce = physx.get("bounce_threshold", 0.5)
        friction_dist = physx.get("friction_correlation_distance", 0.025)
        gpu_agg_pairs = physx.get("gpu_found_lost_aggregate_pairs_capacity", None)
        gpu_total_agg = physx.get("gpu_total_aggregate_pairs_capacity", None)

        gpu_lines = ""
        if gpu_agg_pairs is not None:
            gpu_lines += f"\nphysx_scene.CreateGpuFoundLostAggregatePairsCapacityAttr().Set({gpu_agg_pairs})"
        if gpu_total_agg is not None:
            gpu_lines += f"\nphysx_scene.CreateGpuTotalAggregatePairsCapacityAttr().Set({gpu_total_agg})"

        code = f'''
from pxr import UsdPhysics, Gf, PhysxSchema
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create physics scene
physics_scene_path = "/World/PhysicsScene"
physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
physics_scene.GetGravityDirectionAttr().Set(Gf.Vec3f({gravity[0]}, {gravity[1]}, {gravity[2]}).GetNormalized())
physics_scene.GetGravityMagnitudeAttr().Set(abs({gravity[2]}))

# PhysX settings
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
physx_scene.CreateTimeStepsPerSecondAttr().Set(int(1.0 / {timestep}))
physx_scene.CreateBounceThresholdAttr().Set({bounce})
physx_scene.CreateFrictionCorrelationDistanceAttr().Set({friction_dist})
{gpu_lines}

print("SUCCESS: Physics scene created")
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _create_ground(self, config: dict) -> dict:
        """Create grid ground plane by loading Isaac Sim's default_environment.usd."""
        position = config.get("position", [0, 0, 0])

        code = f'''
from pxr import UsdGeom, Gf
import omni.usd
from omni.isaac.core.utils.nucleus import get_assets_root_path

stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()

prim_path = "/World/GroundPlane"
prim = stage.DefinePrim(prim_path)
prim.GetReferences().AddReference(assets_root + "/Isaac/Environments/Grid/default_environment.usd")

xformable = UsdGeom.Xformable(prim)
xformable.ClearXformOpOrder()
xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))

print("SUCCESS: Grid ground plane created")
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _setup_lighting(self, config: dict) -> dict:
        """Setup scene lighting."""
        light_type = config.get("type", "dome")
        intensity = config.get("intensity", 3000)
        color = config.get("color", [0.75, 0.75, 0.75])

        code = f'''
from pxr import UsdLux, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create dome light
light_path = "/World/DomeLight"
dome_light = UsdLux.DomeLight.Define(stage, light_path)
dome_light.GetIntensityAttr().Set({intensity})
dome_light.GetColorAttr().Set(Gf.Vec3f({color[0]}, {color[1]}, {color[2]}))

print("SUCCESS: Lighting created")
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _apply_randomization(self, asset: dict) -> dict:
        """Apply domain randomization to asset config based on 'randomize' section.

        Modifies position and rotation in-place based on randomize ranges.
        Tracks placed positions for min_separation enforcement.
        """
        import random, math

        randomize = asset.get("randomize", {})
        if not randomize:
            return asset

        # Position randomization
        pos_rand = randomize.get("position", {})
        if pos_rand:
            rand_type = pos_rand.get("type", "absolute")
            base_pos = asset.get("position", [0, 0, 0])
            x_range = pos_rand.get("x", None)
            y_range = pos_rand.get("y", None)
            min_sep = randomize.get("min_separation", 0)

            # z handling: scalar = fixed, list = range to randomize
            z_raw = pos_rand.get("z", None)
            if isinstance(z_raw, list) and len(z_raw) == 2:
                z_val = random.uniform(z_raw[0], z_raw[1])
                if rand_type == "relative":
                    z_val = base_pos[2] + z_val
            elif rand_type == "relative":
                z_val = base_pos[2] + (z_raw if z_raw is not None else 0.0)
            else:
                z_val = z_raw if z_raw is not None else base_pos[2]

            if x_range and y_range:
                for _ in range(100):  # max attempts for min_separation
                    if rand_type == "relative":
                        # Offset from initial position
                        x = base_pos[0] + random.uniform(x_range[0], x_range[1])
                        y = base_pos[1] + random.uniform(y_range[0], y_range[1])
                    else:
                        # Absolute position within range
                        x = random.uniform(x_range[0], x_range[1])
                        y = random.uniform(y_range[0], y_range[1])
                    # Check min_separation against already placed objects
                    if min_sep > 0 and hasattr(self, '_placed_positions'):
                        too_close = False
                        for px, py in self._placed_positions:
                            if math.sqrt((x - px) ** 2 + (y - py) ** 2) < min_sep:
                                too_close = True
                                break
                        if too_close:
                            continue
                    break
                asset["position"] = [x, y, z_val]
                if not hasattr(self, '_placed_positions'):
                    self._placed_positions = []
                self._placed_positions.append((x, y))

        # Orientation randomization (yaw only)
        orient_rand = randomize.get("orientation", {})
        if orient_rand:
            yaw_range = orient_rand.get("yaw", None)
            if yaw_range:
                yaw = random.uniform(yaw_range[0], yaw_range[1])
                # Convert yaw to quaternion (wxyz): rotation around Z axis
                w = math.cos(yaw / 2)
                z = math.sin(yaw / 2)
                asset["rotation"] = [w, 0, 0, z]

        return asset

    def _create_asset(self, asset: dict) -> dict:
        """Create a single asset in the scene."""
        # Apply domain randomization before creating
        asset = self._apply_randomization(asset)

        asset_type = asset.get("type", "rigid")
        prim_path = asset.get("prim_path", "/World/Asset")
        position = asset.get("position", [0, 0, 0])
        rotation = asset.get("rotation", [1, 0, 0, 0])  # wxyz quaternion
        scale = asset.get("scale", [1, 1, 1])

        if asset_type == "articulation":
            return self._create_articulation(asset)
        elif asset_type == "static":
            # Check if asset_url or asset_path is provided (USD reference)
            if asset.get("asset_url") or asset.get("asset_path"):
                return self._load_usd_asset(asset, is_static=True)
            elif "primitive" in asset:
                return self._create_primitive(asset)
            return self._create_static_asset(asset)
        elif asset_type == "rigid":
            # Check if asset_url or asset_path is provided (USD reference)
            if asset.get("asset_url") or asset.get("asset_path"):
                return self._load_usd_asset(asset, is_static=False)
            elif "primitive" in asset:
                return self._create_primitive(asset)
            else:
                return self._create_rigid_asset(asset)
        else:
            return {"success": False, "error": f"Unknown asset type: {asset_type}"}

    @staticmethod
    def _resolve_asset_path(asset_path: str) -> str:
        """Resolve template variables in asset_path to relative paths.

        Task documents use IsaacLab-style template variables like
        {ISAAC_NUCLEUS_DIR} and {ISAACLAB_NUCLEUS_DIR}. These need to
        be converted to relative paths that can be appended to
        get_assets_root_path().

        Paths starting with 'assets/' are treated as project-local and
        converted to absolute paths.

        Examples:
            "{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd"
                -> "/Isaac/Props/Blocks/blue_block.usd"
            "{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
                -> "/IsaacLab/Robots/FrankaEmika/panda_instanceable.usd"
            "assets/robots/so101/so101.usd"
                -> "/home/.../AutoEnvConstruction/assets/robots/so101/so101.usd"
        """
        asset_path = asset_path.replace("{ISAAC_NUCLEUS_DIR}", "/Isaac")
        asset_path = asset_path.replace("{ISAACLAB_NUCLEUS_DIR}", "/Isaac/IsaacLab")
        # Local assets/ paths -> project absolute paths
        if asset_path.startswith("assets/"):
            project_root = Path(__file__).resolve().parents[3]
            asset_path = str(project_root / asset_path)
        return asset_path

    def _load_usd_asset(self, asset: dict, is_static: bool = True) -> dict:
        """Load USD asset from path or URL (e.g., SeattleLabTable).

        Args:
            asset: Asset configuration dict with asset_url or asset_path, position, rotation, scale
                   - asset_url: Full HTTPS URL to the USD asset (takes precedence)
                   - asset_path: Relative path appended to get_assets_root_path()
            is_static: If True, add collision only. If False, add rigid body physics.

        Returns:
            dict with success status
        """
        asset_url = asset.get("asset_url", "")
        asset_path = asset.get("asset_path", "")
        prim_path = asset.get("prim_path", "/World/Asset")
        position = asset.get("position", [0, 0, 0])
        rotation = asset.get("rotation", [1, 0, 0, 0])  # wxyz quaternion
        scale = asset.get("scale", [1, 1, 1])
        physics = asset.get("physics", {})

        # Physics settings for rigid bodies
        mass = physics.get("mass", 1.0)
        add_collision = physics.get("collision", True)
        add_rigid_body = not is_static and physics.get("rigid_body", True)

        # Resolve template variables in asset_path
        asset_path = self._resolve_asset_path(asset_path)

        # Determine whether to use asset_url directly or construct from assets_root
        if asset_url:
            # Use full URL directly (e.g., for Isaac 4.2 assets on Isaac 5.1)
            path_code = f'''
# Using direct asset URL
full_path = "{asset_url}"
'''
        else:
            # Determine path construction: local absolute vs Nucleus relative
            is_nucleus_path = asset_path.startswith("/Isaac") or asset_path.startswith("/IsaacLab")
            if is_nucleus_path:
                path_code = f'''
# Get the nucleus path
from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root = get_assets_root_path()
asset_path = "{asset_path}"
full_path = assets_root + asset_path
'''
            else:
                # Local absolute path (e.g., project-local assets/)
                path_code = f'''
# Local asset path
full_path = "{asset_path}"
'''

        code = f'''
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

{path_code}

print(f"Loading USD asset from: {{full_path}}")

# Add reference to USD
prim_path = "{prim_path}"
prim = stage.DefinePrim(prim_path)
prim.GetReferences().AddReference(full_path)

# Check if prim was created
if not prim.IsValid():
    print(f"ERROR: Failed to load asset from {{full_path}}")
else:
    # Set transform — clear pre-existing xform ops to prevent compound rotation.
    # USD assets may have ops like xformOp:rotateXYZ that don't match "orient",
    # causing a second orient op to be added on top → double rotation.
    # Use opSuffix to avoid precision conflicts with pre-existing float3 ops
    # (e.g. YCB Physics assets have xformOp:translate as float3).
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="yaml").Set(
        Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))
    orientOp = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="yaml")
    orientOp.Set(Gf.Quatd({rotation[0]}, {rotation[1]}, {rotation[2]}, {rotation[3]}))
    scaleOp = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, opSuffix="yaml")
    scaleOp.Set(Gf.Vec3d({scale[0]}, {scale[1]}, {scale[2]}))

    # Add physics
    {"UsdPhysics.CollisionAPI.Apply(prim)" if add_collision else "# No collision"}
    {"UsdPhysics.RigidBodyAPI.Apply(prim)" if add_rigid_body else "# Static object (no rigid body)"}
    {"UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(" + str(mass) + ")" if add_rigid_body else "# No mass for static"}

    print(f"SUCCESS: USD asset loaded at {{prim_path}}")
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _create_articulation(self, asset: dict) -> dict:
        """Create an articulation (robot).

        Prefers direct USD loading when asset_path is provided.
        Falls back to MCP create_robot for known robot types without asset_path.
        """
        asset_path = self._resolve_asset_path(asset.get("asset_path", ""))
        prim_path = asset.get("prim_path", "/World/Robot")
        position = asset.get("position", [0, 0, 0])
        rotation = asset.get("rotation", [1, 0, 0, 0])
        robot_type = asset.get("robot_type", None)

        known_robots = ["franka", "jetbot", "carter", "g1", "go1"]

        # Prefer direct USD loading when asset_path is available
        if asset_path:
            logger.info(f"Direct USD loading for articulation: {asset_path}")
            # Fall through to direct USD loading code below
        elif robot_type and robot_type.lower() in known_robots:
            # Fallback: MCP create_robot only when no asset_path
            logger.info(f"Using create_robot fallback for {robot_type} (no asset_path)")
            result = self.mcp.create_robot(robot_type.lower(), position)
            if result.get("status") != "success":
                return {"success": False, "error": result.get("message", "create_robot failed")}

            # Apply initial joints from YAML (create_robot puts Franka at /Franka)
            if robot_type.lower() == "franka":
                initial_joints = asset.get("initial_joints", {})
                joint_values = self._parse_franka_joints(initial_joints)
                self._set_franka_initial_joints(joint_values, "/Franka")

            return {"success": True}
        else:
            return {"success": False, "error": "No asset_path and no known robot_type"}

        # Build variant selection code if variant_sets specified in YAML
        variant_sets = asset.get("variant_sets", {})
        variant_code = ""
        if variant_sets:
            variant_lines = []
            for vs_name, vs_value in variant_sets.items():
                variant_lines.append(
                    f'prim.GetVariantSets().GetVariantSet("{vs_name}").SetVariantSelection("{vs_value}")'
                )
            variant_code = "\n".join(variant_lines)
            variant_code = f"\n# Set USD variant selections\n{variant_code}\n"

        scale = asset.get("scale", [1, 1, 1])

        # Determine path construction: local absolute vs Nucleus relative
        is_nucleus_path = asset_path.startswith("/Isaac") or asset_path.startswith("/IsaacLab")
        if is_nucleus_path:
            path_code = f'''from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root = get_assets_root_path()
full_path = assets_root + "{asset_path}"'''
        else:
            # Local absolute path (e.g., /home/.../assets/robots/so101/so101.usd)
            path_code = f'full_path = "{asset_path}"'

        code = f'''
from pxr import Usd, UsdGeom, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()
{path_code}

print(f"Loading articulation from: {{full_path}}")
prim_path = "{prim_path}"
prim = stage.DefinePrim(prim_path)
prim.GetReferences().AddReference(full_path)
{variant_code}
xformable = UsdGeom.Xformable(prim)
xformable.ClearXformOpOrder()
xformable.AddTranslateOp().Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))
orientOp = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
orientOp.Set(Gf.Quatd({rotation[0]}, {rotation[1]}, {rotation[2]}, {rotation[3]}))
scaleOp = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
scaleOp.Set(Gf.Vec3d({scale[0]}, {scale[1]}, {scale[2]}))

print("SUCCESS: Articulation created at " + prim_path)
'''
        result = self.mcp.execute_script(code)
        if result.get("status") != "success":
            return {"success": False, "error": result.get("message", "USD load failed")}

        # Apply initial joints if specified (for robots loaded via direct USD)
        initial_joints = asset.get("initial_joints", {})
        if initial_joints:
            self._set_articulation_initial_joints(initial_joints, prim_path)

        return {"success": True}

    @staticmethod
    def _parse_franka_joints(initial_joints: dict) -> list:
        """Convert YAML initial_joints dict to ordered array [j1..j7, finger_l, finger_r]."""
        # Default: FRANKA_PANDA_CFG from IsaacLab franka.py
        defaults = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741, 0.04, 0.04]
        if not initial_joints:
            return defaults
        values = [
            initial_joints.get("panda_joint1", defaults[0]),
            initial_joints.get("panda_joint2", defaults[1]),
            initial_joints.get("panda_joint3", defaults[2]),
            initial_joints.get("panda_joint4", defaults[3]),
            initial_joints.get("panda_joint5", defaults[4]),
            initial_joints.get("panda_joint6", defaults[5]),
            initial_joints.get("panda_joint7", defaults[6]),
            initial_joints.get("panda_finger_joint", defaults[7]),
            initial_joints.get("panda_finger_joint", defaults[8]),
        ]
        return values

    def _set_franka_initial_joints(self, joint_values: list, prim_path: str) -> None:
        """Set Franka Panda to initial joint positions from YAML.

        Uses Articulation API during simulation. Does NOT pause timeline
        afterwards so the visual state reflects the physics state.
        """
        joints_str = str(joint_values)
        code = f'''
import omni.timeline, asyncio, numpy as np

timeline = omni.timeline.get_timeline_interface()
timeline.play()

async def _apply():
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    from omni.isaac.core.articulations import Articulation
    robot = Articulation(prim_path="{prim_path}")
    robot.initialize()

    target = np.array({joints_str})
    robot.set_joint_positions(target)
    robot.set_joint_position_targets(target)

    # Let simulation settle with targets held
    for _ in range(120):
        await omni.kit.app.get_app().next_update_async()

    actual = robot.get_joint_positions()
    print("Target:", [round(t,3) for t in target[:7]])
    print("Actual:", [round(float(a),3) for a in actual[:7]])
    # DO NOT pause timeline - visual state must match physics state
    print("SUCCESS: Franka joints applied (timeline still running)")

asyncio.ensure_future(_apply())
'''
        result = self.mcp.execute_script(code)
        if result.get("status") != "success":
            logger.warning(f"Failed to set Franka joints: {result.get('message', '')}")
        else:
            import time
            time.sleep(5)  # Wait for async settle + visual update

    def _set_articulation_initial_joints(self, initial_joints: dict, prim_path: str) -> None:
        """Set initial joint positions for any articulation loaded via direct USD.

        Args:
            initial_joints: dict mapping joint_name -> value (radians)
            prim_path: USD prim path of the articulation root
        """
        if not initial_joints:
            return

        # Build a dict of {joint_name: value} for Isaac Sim
        joints_dict = {k: float(v) for k, v in initial_joints.items()}
        joints_dict_str = str(joints_dict)
        code = f'''
import omni.timeline, asyncio, numpy as np

timeline = omni.timeline.get_timeline_interface()
timeline.play()

async def _apply():
    for _ in range(10):
        await omni.kit.app.get_app().next_update_async()

    from omni.isaac.core.articulations import Articulation
    robot = Articulation(prim_path="{prim_path}")
    robot.initialize()

    joints_dict = {joints_dict_str}
    dof_names = robot.dof_names
    target = robot.get_joint_positions()

    for name, val in joints_dict.items():
        if name in dof_names:
            idx = dof_names.index(name)
            target[idx] = val

    robot.set_joint_positions(target)
    robot.set_joint_position_targets(target)

    for _ in range(120):
        await omni.kit.app.get_app().next_update_async()

    actual = robot.get_joint_positions()
    print("DOF names:", dof_names)
    print("Target:", [round(float(t),3) for t in target])
    print("Actual:", [round(float(a),3) for a in actual])
    print("SUCCESS: Articulation joints applied at {prim_path}")

asyncio.ensure_future(_apply())
'''
        result = self.mcp.execute_script(code)
        if result.get("status") != "success":
            logger.warning(f"Failed to set joints at {prim_path}: {result.get('message', '')}")
        else:
            import time
            time.sleep(5)

    def _create_static_asset(self, asset: dict) -> dict:
        """Create a static (non-physics) asset."""
        asset_path = self._resolve_asset_path(asset.get("asset_path", ""))
        prim_path = asset.get("prim_path", "/World/Static")
        position = asset.get("position", [0, 0, 0])
        rotation = asset.get("rotation", [1, 0, 0, 0])
        scale = asset.get("scale", [1, 1, 1])

        code = f'''
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Get the nucleus path
from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root = get_assets_root_path()
asset_path = "{asset_path}"
full_path = assets_root + asset_path

# Add reference to USD
prim_path = "{prim_path}"
prim = stage.DefinePrim(prim_path)
prim.GetReferences().AddReference(full_path)

# Set transform
xformable = UsdGeom.Xformable(prim)
xformable.ClearXformOpOrder()
xformable.AddTranslateOp().Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))
xformable.AddScaleOp().Set(Gf.Vec3f({scale[0]}, {scale[1]}, {scale[2]}))

# Set rotation (quaternion wxyz) - use Quatd with PrecisionDouble to match USD assets
orientOp = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
orientOp.Set(Gf.Quatd({rotation[0]}, {rotation[1]}, {rotation[2]}, {rotation[3]}))

# Add collision
UsdPhysics.CollisionAPI.Apply(prim)

print("SUCCESS: Static asset created at " + prim_path)
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _create_primitive(self, asset: dict) -> dict:
        """Create a primitive shape (cube, sphere, cylinder)."""
        primitive = asset.get("primitive", "cube")
        prim_path = asset.get("prim_path", "/World/Primitive")
        position = asset.get("position", [0, 0, 0])
        scale = asset.get("scale", [1, 1, 1])
        color = asset.get("color", [0.5, 0.5, 0.5])
        physics = asset.get("physics", {})

        mass = physics.get("mass", 1.0)
        collision = physics.get("collision", True)
        rigid_body = physics.get("rigid_body", True)

        if primitive == "cube":
            code = f'''
from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create cube
prim_path = "{prim_path}"
cube = UsdGeom.Cube.Define(stage, prim_path)

# Set size (cube uses single size attribute)
cube.GetSizeAttr().Set(1.0)

# Set transform
xformable = UsdGeom.Xformable(cube.GetPrim())
xformable.ClearXformOpOrder()
xformable.AddTranslateOp().Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))
xformable.AddScaleOp().Set(Gf.Vec3f({scale[0]}, {scale[1]}, {scale[2]}))

# Set color
cube.GetDisplayColorAttr().Set([({color[0]}, {color[1]}, {color[2]})])

# Add physics
{"UsdPhysics.CollisionAPI.Apply(cube.GetPrim())" if collision else ""}
{"UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())" if rigid_body else ""}
{"UsdPhysics.MassAPI.Apply(cube.GetPrim()).GetMassAttr().Set(" + str(mass) + ")" if rigid_body else ""}

print("SUCCESS: Cube created at " + prim_path)
'''
        elif primitive == "sphere":
            code = f'''
from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create sphere
prim_path = "{prim_path}"
sphere = UsdGeom.Sphere.Define(stage, prim_path)
sphere.GetRadiusAttr().Set({scale[0] / 2})

# Set transform
xformable = UsdGeom.Xformable(sphere.GetPrim())
xformable.ClearXformOpOrder()
xformable.AddTranslateOp().Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))

# Set color
sphere.GetDisplayColorAttr().Set([({color[0]}, {color[1]}, {color[2]})])

# Add physics
{"UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())" if collision else ""}
{"UsdPhysics.RigidBodyAPI.Apply(sphere.GetPrim())" if rigid_body else ""}
{"UsdPhysics.MassAPI.Apply(sphere.GetPrim()).GetMassAttr().Set(" + str(mass) + ")" if rigid_body else ""}

print("SUCCESS: Sphere created at " + prim_path)
'''
        elif primitive == "cylinder":
            code = f'''
from pxr import UsdGeom, UsdPhysics, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Create cylinder
prim_path = "{prim_path}"
cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
cylinder.GetRadiusAttr().Set({scale[0] / 2})
cylinder.GetHeightAttr().Set({scale[1]})

# Set transform
xformable = UsdGeom.Xformable(cylinder.GetPrim())
xformable.ClearXformOpOrder()
xformable.AddTranslateOp().Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))

# Set color
cylinder.GetDisplayColorAttr().Set([({color[0]}, {color[1]}, {color[2]})])

# Add physics
{"UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())" if collision else ""}
{"UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())" if rigid_body else ""}
{"UsdPhysics.MassAPI.Apply(cylinder.GetPrim()).GetMassAttr().Set(" + str(mass) + ")" if rigid_body else ""}

print("SUCCESS: Cylinder created at " + prim_path)
'''
        else:
            return {"success": False, "error": f"Unknown primitive: {primitive}"}

        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _create_rigid_asset(self, asset: dict) -> dict:
        """Create a rigid body asset from USD file."""
        asset_path = self._resolve_asset_path(asset.get("asset_path", ""))
        prim_path = asset.get("prim_path", "/World/Rigid")
        position = asset.get("position", [0, 0, 0])
        rotation = asset.get("rotation", [1, 0, 0, 0])
        scale = asset.get("scale", [1, 1, 1])
        physics = asset.get("physics", {})

        mass = physics.get("mass", 1.0)

        code = f'''
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd

stage = omni.usd.get_context().get_stage()

# Get the nucleus path
from omni.isaac.core.utils.nucleus import get_assets_root_path
assets_root = get_assets_root_path()
asset_path = "{asset_path}"
full_path = assets_root + asset_path

# Add reference to USD
prim_path = "{prim_path}"
prim = stage.DefinePrim(prim_path)
prim.GetReferences().AddReference(full_path)

# Set transform
xformable = UsdGeom.Xformable(prim)
xformable.ClearXformOpOrder()
xformable.AddTranslateOp().Set(Gf.Vec3d({position[0]}, {position[1]}, {position[2]}))
xformable.AddScaleOp().Set(Gf.Vec3f({scale[0]}, {scale[1]}, {scale[2]}))

# Set rotation (quaternion wxyz) - use Quatd with PrecisionDouble to match USD assets
orientOp = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
orientOp.Set(Gf.Quatd({rotation[0]}, {rotation[1]}, {rotation[2]}, {rotation[3]}))

# Add physics
UsdPhysics.CollisionAPI.Apply(prim)
UsdPhysics.RigidBodyAPI.Apply(prim)
UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set({mass})

print("SUCCESS: Rigid asset created at " + prim_path)
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}

    def _setup_camera(self, config: dict) -> dict:
        """Set OmniverseKit_Persp camera position and target using Isaac Sim's set_camera_view."""
        position = config.get("position", [2, 2, 1.5])
        target = config.get("target", [0, 0, 0.3])

        code = f'''
from isaacsim.core.utils.viewports import set_camera_view
set_camera_view(
    eye=[{position[0]}, {position[1]}, {position[2]}],
    target=[{target[0]}, {target[1]}, {target[2]}],
    camera_prim_path="/OmniverseKit_Persp"
)
print("SUCCESS: Perspective camera set to eye={position}, target={target}")
'''
        result = self.mcp.execute_script(code)
        return {"success": result.get("status") == "success"}


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Build Isaac Sim scene from Task Document")
    parser.add_argument("document", type=str, help="Path to YAML task document")
    parser.add_argument("--host", type=str, default="localhost", help="MCP host")
    parser.add_argument("--port", type=int, default=8766, help="MCP port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = MCPClient(host=args.host, port=args.port)
    builder = SceneBuilder(client)

    result = builder.build_from_yaml(args.document)

    if result["success"]:
        print(f"\nScene built successfully!")
        print(f"Created prims: {', '.join(result['created_prims'])}")
    else:
        print(f"\nScene build failed!")
        print(f"Errors: {result['errors']}")

    if result["warnings"]:
        print(f"Warnings: {result['warnings']}")

    client.disconnect()


if __name__ == "__main__":
    main()
