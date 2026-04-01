"""Shared IsaacLab runtime helpers for local and container execution."""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path


CONTAINER_ISAACLAB_PATH = Path("/workspace/IsaacLab")
KNOWN_NVIDIA_VULKAN_ICD_PATHS = (
    Path("/usr/share/vulkan/icd.d/nvidia_icd.json"),
    Path("/etc/vulkan/icd.d/nvidia_icd.json"),
)


def is_docker_runtime() -> bool:
    raw = str(os.environ.get("SIMGEN_IN_DOCKER", "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return Path("/.dockerenv").exists()


def resolve_isaaclab_path(project_root: Path, configured_path: str | None = None) -> Path:
    env_path = os.environ.get("ISAACLAB_PATH")
    candidate = env_path or configured_path
    if candidate:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = project_root / path
        return path.resolve()
    if is_docker_runtime() and CONTAINER_ISAACLAB_PATH.exists():
        return CONTAINER_ISAACLAB_PATH.resolve()
    return (project_root.parent / "IsaacLab").resolve()


def resolve_output_path(project_root: Path, raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        artifact_root = os.environ.get("SIMGEN_ARTIFACT_ROOT")
        if is_docker_runtime() and artifact_root:
            relative_path = path
            if relative_path.parts and relative_path.parts[0] == "outputs":
                relative_path = Path(*relative_path.parts[1:]) if len(relative_path.parts) > 1 else Path()
            path = Path(artifact_root).expanduser() / relative_path
        else:
            path = project_root / path
    return path.resolve()


def resolve_vulkan_icd_path() -> str | None:
    """Return a valid NVIDIA Vulkan ICD path when one is available."""

    override = os.environ.get("SIMGEN_VK_ICD_FILENAMES")
    if override:
        return override

    for candidate in KNOWN_NVIDIA_VULKAN_ICD_PATHS:
        if candidate.exists():
            return str(candidate)
    return None


def apply_headless_gpu_env(env: dict[str, str]) -> None:
    """Normalize environment variables for headless GPU execution."""

    env.pop("DISPLAY", None)
    icd_path = resolve_vulkan_icd_path()
    if icd_path:
        env["VK_ICD_FILENAMES"] = icd_path
    else:
        env.pop("VK_ICD_FILENAMES", None)


def _normalize_launch_mode(launch_mode: str | None) -> str:
    mode = str(launch_mode or "auto").strip().lower()
    if mode not in {"auto", "direct", "conda"}:
        raise ValueError(f"Unsupported IsaacLab launch mode: {launch_mode}")
    return mode


def build_isaaclab_runner_command(
    script_path: Path,
    *,
    isaaclab_path: Path,
    launch_mode: str | None = None,
    conda_env: str | None = None,
    enable_cameras: bool = False,
    headless: bool = False,
    num_envs: int | None = None,
    use_launcher: bool = True,
    python_executable: str | None = None,
) -> tuple[list[str], str]:
    """Build a subprocess command for running an IsaacLab script."""

    mode = _normalize_launch_mode(launch_mode)
    if mode == "auto":
        mode = "direct" if is_docker_runtime() else "conda"

    launcher = isaaclab_path / "isaaclab.sh"
    use_launcher = bool(use_launcher and launcher.exists())

    python_executable = python_executable or os.environ.get("SIMGEN_ISAACLAB_PYTHON") or sys.executable

    base_cmd: list[str]
    if use_launcher:
        base_cmd = [str(launcher), "-p", str(script_path)]
    else:
        base_cmd = [python_executable, str(script_path)]

    if num_envs is not None:
        base_cmd.extend(["--num_envs", str(num_envs)])
    if headless:
        base_cmd.append("--headless")
    if enable_cameras:
        base_cmd.append("--enable_cameras")

    display_cmd = shlex.join(base_cmd)

    if mode == "direct":
        return base_cmd, display_cmd

    if not conda_env:
        raise ValueError("conda launch mode requires a conda environment name")
    if shutil.which("conda") is None:
        raise RuntimeError("conda launch mode requested but 'conda' is not available")

    wrapped_cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "--no-capture-output",
        "bash",
        "-lc",
        display_cmd,
    ]
    return wrapped_cmd, shlex.join(wrapped_cmd)
