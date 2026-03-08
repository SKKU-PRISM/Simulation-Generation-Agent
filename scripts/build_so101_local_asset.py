#!/usr/bin/env python3
"""Build the canonical repo-local SO-101 USD from the ADC URDF.

The generated USD is written in-place to ``assets/robots/so101/so101.usd``.
Post-processing adds two simple non-instanceable jaw-pad collision proxies so
small-cube grasps do not rely on the imported mesh collisions alone.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = (
    PROJECT_ROOT / "external" / "AutoDataCollector" / "assets" / "urdf" / "so101_robot3.urdf"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "robots" / "so101"
DEFAULT_OUTPUT_FILE = "so101.usd"
DEFAULT_MESH_ROOT_CANDIDATES = (
    PROJECT_ROOT / "external" / "AutoDataCollector" / "assets" / "urdf" / "assets",
    PROJECT_ROOT.parent / "SO-ARM100" / "Simulation" / "SO101" / "assets",
)

# Pad proxies are expressed in the local frame of the parent link.
# The fixed jaw proxy lives on gripper_frame_link, which already sits at the
# claw mouth. The moving jaw proxy is expressed in moving_jaw_so101_v1_link and
# was back-computed from the runtime relative transform to gripper_frame_link.
SO101_FIXED_PAD_CENTER = (0.0, -0.0105, 0.0)
SO101_FIXED_PAD_HALF_SIZE = (0.008, 0.0025, 0.010)
SO101_MOVING_PAD_CENTER = (0.0281, -0.0085, 0.0747)
SO101_MOVING_PAD_HALF_SIZE = (0.008, 0.0025, 0.010)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Source SO-101 URDF")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the converted USD")
parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="USD filename inside output-dir")
parser.add_argument("--force", action="store_true", help="Force USD regeneration")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def _ensure_cube_collision(stage, link_path: str, name: str, center: tuple[float, float, float], half_size: tuple[float, float, float]) -> str:
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    prim_path = f"{link_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, prim_path)
    prim = cube.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(center)
    xformable.AddScaleOp().Set(half_size)
    UsdGeom.Imageable(prim).MakeInvisible()
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        physx_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    else:
        physx_api = PhysxSchema.PhysxCollisionAPI(prim)
    physx_api.GetContactOffsetAttr().Set(0.005)
    physx_api.GetRestOffsetAttr().Set(0.0)
    return prim_path


def _postprocess_so101_usd(usd_path: Path) -> list[str]:
    from pxr import Usd

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD for post-process: {usd_path}")

    root_children = [child for child in stage.GetPseudoRoot().GetChildren() if child.IsValid()]
    if not root_children:
        raise RuntimeError(f"No root prim found in {usd_path}")
    root_path = root_children[0].GetPath().pathString

    fixed_link_path = f"{root_path}/gripper_frame_link"
    moving_link_path = f"{root_path}/moving_jaw_so101_v1_link"
    if not stage.GetPrimAtPath(fixed_link_path).IsValid():
        raise RuntimeError(f"Missing fixed jaw link in USD: {fixed_link_path}")
    if not stage.GetPrimAtPath(moving_link_path).IsValid():
        raise RuntimeError(f"Missing moving jaw link in USD: {moving_link_path}")

    added = [
        _ensure_cube_collision(
            stage,
            fixed_link_path,
            "fixed_pad_proxy",
            SO101_FIXED_PAD_CENTER,
            SO101_FIXED_PAD_HALF_SIZE,
        ),
        _ensure_cube_collision(
            stage,
            moving_link_path,
            "moving_pad_proxy",
            SO101_MOVING_PAD_CENTER,
            SO101_MOVING_PAD_HALF_SIZE,
        ),
    ]
    stage.Save()
    return added


def main() -> int:
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_text = urdf_path.read_text()
    mesh_root = next((path for path in DEFAULT_MESH_ROOT_CANDIDATES if path.exists()), None)
    if mesh_root is None:
        raise FileNotFoundError(
            "Could not find SO-101 mesh assets. Checked: "
            + ", ".join(str(path) for path in DEFAULT_MESH_ROOT_CANDIDATES)
        )
    rewritten_urdf_text = re.sub(
        r'filename="assets/([^"]+)"',
        lambda match: f'filename="{(mesh_root / match.group(1)).resolve()}"',
        urdf_text,
    )

    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.01))
    try:
        prev_cwd = Path.cwd()
        os.chdir(urdf_path.parent)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_urdf_path = Path(tmpdir) / urdf_path.name
                temp_urdf_path.write_text(rewritten_urdf_text)
                cfg = UrdfConverterCfg(
                    asset_path=str(temp_urdf_path),
                    usd_dir=str(output_dir),
                    usd_file_name=args.output_file,
                    force_usd_conversion=args.force,
                    make_instanceable=False,
                    fix_base=True,
                    merge_fixed_joints=True,
                    convert_mimic_joints_to_normal_joints=False,
                    self_collision=False,
                    collision_from_visuals=False,
                    joint_drive=None,
                )
                converter = UrdfConverter(cfg)
        finally:
            os.chdir(prev_cwd)
        usd_path = Path(converter.usd_path).resolve()
        added = _postprocess_so101_usd(usd_path)
        print(f"Built local SO-101 asset: {usd_path}")
        for path in added:
            print(f"  added collision proxy: {path}")
        return 0
    finally:
        sim._disable_app_control_on_stop_handle = True
        sim.stop()
        sim.clear()
        sim.clear_all_callbacks()
        sim.clear_instance()
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
