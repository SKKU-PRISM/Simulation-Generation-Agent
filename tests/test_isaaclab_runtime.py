from __future__ import annotations

from pathlib import Path

from src.agent.common.isaaclab_runtime import (
    build_isaaclab_runner_command,
    resolve_isaaclab_path,
    resolve_output_path,
)


def test_resolve_isaaclab_path_prefers_env(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    env_path = tmp_path / "isaaclab-env"
    env_path.mkdir()

    monkeypatch.setenv("ISAACLAB_PATH", str(env_path))

    resolved = resolve_isaaclab_path(project_root, None)

    assert resolved == env_path.resolve()


def test_resolve_output_path_handles_relative_path(tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()

    resolved = resolve_output_path(project_root, "artifacts/run")

    assert resolved == (project_root / "artifacts" / "run").resolve()


def test_resolve_output_path_prefers_artifact_root_in_docker(monkeypatch, tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    artifact_root = tmp_path / "mounted-artifacts"
    artifact_root.mkdir()

    monkeypatch.setenv("SIMGEN_IN_DOCKER", "1")
    monkeypatch.setenv("SIMGEN_ARTIFACT_ROOT", str(artifact_root))

    resolved = resolve_output_path(project_root, "outputs/isaaclab")

    assert resolved == (artifact_root / "isaaclab").resolve()


def test_build_isaaclab_runner_command_direct_with_launcher(tmp_path: Path):
    isaaclab_root = tmp_path / "IsaacLab"
    isaaclab_root.mkdir()
    launcher = isaaclab_root / "isaaclab.sh"
    launcher.write_text("#!/usr/bin/env bash\n")
    launcher.chmod(0o755)

    script_path = tmp_path / "runner.py"
    script_path.write_text("print('ok')\n")

    cmd, display = build_isaaclab_runner_command(
        script_path,
        isaaclab_path=isaaclab_root,
        launch_mode="direct",
        headless=True,
        num_envs=2,
        use_launcher=True,
    )

    assert cmd[:3] == [str(launcher), "-p", str(script_path)]
    assert "--headless" in cmd
    assert "--num_envs" in cmd
    assert str(script_path) in display


def test_build_isaaclab_runner_command_direct_without_launcher(tmp_path: Path):
    isaaclab_root = tmp_path / "IsaacLab"
    isaaclab_root.mkdir()
    script_path = tmp_path / "runner.py"
    script_path.write_text("print('ok')\n")

    cmd, _ = build_isaaclab_runner_command(
        script_path,
        isaaclab_path=isaaclab_root,
        launch_mode="direct",
        use_launcher=False,
        python_executable="python3",
    )

    assert cmd == ["python3", str(script_path)]
