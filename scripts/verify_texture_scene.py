#!/usr/bin/env python3
"""Render texture-sensitive task assets in isolated Isaac Sim scenes and save screenshots."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify that texture-sensitive task assets load with materials in scene.")
    parser.add_argument("tasks", nargs="+", help="Task YAML paths to inspect.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory for screenshots and manifests.")
    parser.add_argument("--width", type=int, default=1024, help="Screenshot width.")
    parser.add_argument("--height", type=int, default=768, help="Screenshot height.")
    parser.add_argument(
        "--asset-name",
        action="append",
        default=[],
        help="Optional asset name filter; can be repeated.",
    )
    return parser


_PARSER = _build_arg_parser()
_ARGS = _PARSER.parse_args()
_APP_PARSER = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_APP_PARSER)
_APP_ARGS = _APP_PARSER.parse_args([])
_APP_ARGS.headless = True
_APP_ARGS.enable_cameras = True
_APP_ARGS.rendering_mode = "performance"
_APP = AppLauncher(_APP_ARGS).app


import numpy as np
from PIL import Image
from pxr import Usd, UsdShade

import omni.replicator.core as rep
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.task_docs import load_task_document
from src.data_collection.pipeline import build_task_texture_audit


def _resolve_asset_path(asset_path: str) -> str:
    asset_path = str(asset_path or "")
    asset_path = asset_path.replace("{ISAAC_NUCLEUS_DIR}", ISAAC_NUCLEUS_DIR)
    asset_path = asset_path.replace("{ISAACLAB_NUCLEUS_DIR}", ISAACLAB_NUCLEUS_DIR)
    if asset_path.startswith("assets/"):
        return str((PROJECT_ROOT / asset_path).resolve())
    return asset_path


def _asset_spawn_pose(asset_family: str) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    del asset_family
    return (0.0, 0.0, 0.06), (1.0, 0.0, 0.0, 0.0)


def _camera_view(asset_family: str) -> tuple[list[float], list[float]]:
    if asset_family in {"tuna_can", "tomato_soup_can", "mustard_bottle"}:
        return [0.65, 0.45, 0.28], [0.0, 0.0, 0.08]
    if asset_family in {"mug", "ycb_mug"}:
        return [0.60, 0.48, 0.30], [0.0, 0.0, 0.10]
    return [0.62, 0.46, 0.27], [0.0, 0.0, 0.08]


def _collect_material_bindings(stage: Usd.Stage, root_path: str) -> list[dict[str, str]]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []

    records: list[dict[str, str]] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsValid():
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct = binding_api.GetDirectBinding()
        material_path = str(direct.GetMaterialPath())
        if material_path:
            records.append(
                {
                    "prim_path": str(prim.GetPath()),
                    "prim_type": prim.GetTypeName(),
                    "material_path": material_path,
                    "purpose": str(direct.GetMaterialPurpose() or ""),
                }
            )
    return records


def _collect_material_prims(stage: Usd.Stage, root_path: str) -> list[str]:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return []
    return [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.IsValid() and prim.GetTypeName() == "Material"
    ]


def _render_rgb(sim: SimulationContext, resolution: tuple[int, int], eye: list[float], target: list[float]) -> np.ndarray:
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    render_product = rep.create.render_product("/OmniverseKit_Persp", resolution=resolution)
    if not isinstance(render_product, str):
        render_product = render_product.path
    annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    annotator.attach(render_product)

    for _ in range(8):
        sim.step()

    rgb_data = annotator.get_data()
    annotator.detach(render_product)
    if rgb_data.shape[-1] == 4:
        rgb_data = rgb_data[..., :3]
    return rgb_data


def _spawn_visual_scene(asset_path: str, asset_family: str) -> SimulationContext:
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.01))

    ground_cfg = sim_utils.GroundPlaneCfg(color=(0.22, 0.22, 0.24), size=(6.0, 6.0))
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 0.95))
    light_cfg.func("/World/DomeLight", light_cfg)
    key_light_cfg = sim_utils.DistantLightCfg(intensity=3500.0, color=(0.95, 0.95, 0.95))
    key_light_cfg.func("/World/KeyLight", key_light_cfg)

    pedestal_cfg = sim_utils.CuboidCfg(
        size=(0.28, 0.28, 0.02),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.58, 0.60)),
    )
    pedestal_cfg.func("/World/Pedestal", pedestal_cfg, translation=(0.0, 0.0, 0.01))

    usd_cfg = sim_utils.UsdFileCfg(usd_path=asset_path)
    translation, orientation = _asset_spawn_pose(asset_family)
    usd_cfg.func("/World/Asset", usd_cfg, translation=translation, orientation=orientation)
    sim_utils.update_stage()
    return sim


def _gather_cases(tasks: list[str], asset_names: set[str]) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for task in tasks:
        task_path = (PROJECT_ROOT / task).resolve() if not Path(task).is_absolute() else Path(task)
        task_doc = load_task_document(task_path)
        assets_by_name = {asset.get("name"): asset for asset in task_doc.get("assets", []) if asset.get("name")}
        for record in build_task_texture_audit(task_doc):
            asset_name = str(record.get("asset_name", ""))
            if asset_names and asset_name not in asset_names:
                continue
            asset = assets_by_name.get(asset_name, {})
            cases.append(
                {
                    "task_name": str(record.get("task_name", "unknown")),
                    "yaml_path": str(task_path),
                    "asset_name": asset_name,
                    "asset_family": str(record.get("asset_family", "")),
                    "asset_path": _resolve_asset_path(str(asset.get("asset_path", record.get("asset_path", "")))),
                    "loader_policy": str(record.get("loader_policy", "")),
                }
            )
    return cases


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(_ARGS.output_dir).resolve()
        if _ARGS.output_dir
        else (PROJECT_ROOT / ".local_docs" / "adc" / f"texture_scene_smoke_{timestamp}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = _gather_cases(_ARGS.tasks, set(_ARGS.asset_name))
    resolution = (_ARGS.width, _ARGS.height)
    report: dict[str, object] = {
        "timestamp": timestamp,
        "resolution": list(resolution),
        "cases": [],
    }

    for case in cases:
        case_dir = output_dir / f"{case['task_name']}_{case['asset_name']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        sim = None
        try:
            sim = _spawn_visual_scene(case["asset_path"], case["asset_family"])
            stage = sim.stage
            bindings = _collect_material_bindings(stage, "/World/Asset")
            material_prims = _collect_material_prims(stage, "/World/Asset")
            eye, target = _camera_view(case["asset_family"])
            rgb = _render_rgb(sim, resolution, eye, target)
            screenshot_path = case_dir / "scene_rgb.png"
            Image.fromarray(rgb).save(screenshot_path)
            record = {
                **case,
                "scene_ok": True,
                "screenshot_path": str(screenshot_path),
                "material_binding_count": len(bindings),
                "material_prim_count": len(material_prims),
                "material_bindings": bindings,
                "material_prims": material_prims,
            }
            (case_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            report["cases"].append(record)
        except Exception as exc:
            record = {
                **case,
                "scene_ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            (case_dir / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            report["cases"].append(record)
        finally:
            if sim is not None:
                sim.stop()
                sim.clear()
                sim.clear_all_callbacks()
                sim.clear_instance()

    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    for case in report["cases"]:
        print(
            f"[{'OK' if case.get('scene_ok') else 'FAIL'}] "
            f"{case.get('task_name')}::{case.get('asset_name')} -> {case.get('screenshot_path', case.get('error'))}"
        )
    _APP.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
