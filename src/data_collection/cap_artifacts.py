"""Helpers for saving forward-only CaP execution artifacts."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


def make_episode_run_dir(output_dir: str | Path, episode_idx: int) -> Path:
    """Return and create the per-episode CaP artifact directory."""

    run_dir = Path(output_dir) / "cap_runs" / f"episode_{episode_idx:06d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_code_generation_artifacts(
    run_dir: str | Path,
    generated_code: str,
    raw_response: str,
    translated_positions: dict[str, Any],
) -> dict[str, str]:
    """Persist generated code, raw LLM text, and translated positions."""

    run_path = Path(run_dir)
    code_path = run_path / "generated_code.py"
    response_path = run_path / "raw_llm_response.txt"
    positions_path = run_path / "scene_positions.json"

    code_path.write_text(generated_code, encoding="utf-8")
    response_path.write_text(raw_response, encoding="utf-8")
    positions_path.write_text(
        json.dumps(translated_positions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "generated_code": str(code_path),
        "raw_response": str(response_path),
        "scene_positions": str(positions_path),
    }


def save_execution_context(
    run_dir: str | Path,
    instruction: str,
    object_positions: dict[str, Any],
    generated_code: str,
    execution_success: bool,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Persist a lightweight ADC-compatible execution context JSON."""

    run_path = Path(run_dir)
    context_path = run_path / "execution_context.json"
    payload = {
        "instruction": instruction,
        "object_positions": object_positions,
        "generated_spec": {},
        "generated_code": generated_code,
        "execution_success": bool(execution_success),
        "timestamp": datetime.now().isoformat(),
        "robot_id": 0,
        "metadata": metadata or {},
    }
    context_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(context_path)


def save_judge_images(
    run_dir: str | Path,
    prefix: str,
    images: dict[str, Any] | None,
) -> list[str]:
    """Save judge images from a multi-view capture dict."""

    if not images:
        return []

    from PIL import Image

    saved_paths: list[str] = []
    run_path = Path(run_dir)
    for view_name, image in images.items():
        if image is None:
            continue
        out_path = run_path / f"{prefix}_{view_name}.png"
        Image.fromarray(image).save(out_path)
        saved_paths.append(str(out_path))
    return saved_paths
