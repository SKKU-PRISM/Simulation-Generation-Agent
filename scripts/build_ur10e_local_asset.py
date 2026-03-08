#!/usr/bin/env python3
"""Build a repo-local UR10e + Robotiq 2F-85 USD from the Isaac Sim URDF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "robots" / "ur10e_robotiq_2f85"


def _default_urdf_path() -> Path:
    isaac_sim_dir = Path(
        os.environ.get("ISAAC_SIM_DIR", str((PROJECT_ROOT.parent / "isaac-sim").resolve()))
    ).expanduser().resolve()
    return (
        isaac_sim_dir
        / "standalone_examples"
        / "api"
        / "isaacsim.robot.manipulators"
        / "ur10e"
        / "rmpflow"
        / "ur10e.urdf"
    )


def _sanitize_converter_config(config_path: Path, urdf_path: Path, output_dir: Path) -> None:
    if not config_path.exists():
        return
    isaac_sim_dir = Path(
        os.environ.get("ISAAC_SIM_DIR", str((PROJECT_ROOT.parent / "isaac-sim").resolve()))
    ).expanduser().resolve()
    try:
        asset_path_value = f'{{ISAAC_SIM_DIR}}/{urdf_path.relative_to(isaac_sim_dir).as_posix()}'
    except ValueError:
        try:
            asset_path_value = urdf_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            asset_path_value = urdf_path.name
    sanitized_lines = []
    for line in config_path.read_text().splitlines():
        if line.startswith("asset_path:"):
            line = f"asset_path: {asset_path_value}"
        elif line.startswith("usd_dir:"):
            line = f'usd_dir: {output_dir.relative_to(PROJECT_ROOT)}'
        sanitized_lines.append(line)
    config_path.write_text("\n".join(sanitized_lines) + "\n")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--urdf", type=Path, default=_default_urdf_path(), help="Source URDF with the 2F-85 gripper baked in")
parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for the converted USD")
parser.add_argument("--force", action="store_true", help="Force USD regeneration")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main() -> int:
    urdf_path = args.urdf.expanduser().resolve()
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.01))
    try:
        cfg = UrdfConverterCfg(
            asset_path=str(urdf_path),
            usd_dir=str(output_dir),
            usd_file_name="ur10e_robotiq_2f85.usd",
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
        _sanitize_converter_config(output_dir / "config.yaml", urdf_path, output_dir)
        print(f"Built local UR10e asset: {converter.usd_path}")
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
