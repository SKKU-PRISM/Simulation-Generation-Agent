#!/usr/bin/env python3
"""End-to-end Docker release validator."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "docker_release_validation"
DEFAULT_IMAGE_TAG = "simgen-isaaclab:release"
DEFAULT_CONTAINER_ARTIFACT_ROOT = "/workspace/artifacts"
# At least one provider group must be set (OpenAI platform OR Azure)
DOCKER_REQUIRED_ENVS_OPENAI = ("OPENAI_API_KEY",)
DOCKER_REQUIRED_ENVS_AZURE = ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL")
OPTIONAL_DOCKER_ENVS = (
    "OPENAI_BASE_URL", "HF_TOKEN", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY",
)
SENSITIVE_DOCKER_ENVS = set(
    DOCKER_REQUIRED_ENVS_OPENAI + DOCKER_REQUIRED_ENVS_AZURE + OPTIONAL_DOCKER_ENVS
)


@dataclass
class PhaseResult:
    name: str
    status: str
    command: str | None
    returncode: int | None
    log_path: str | None
    note: str | None = None


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize_command_for_log(cmd: list[str]) -> str:
    sanitized: list[str] = []
    for arg in cmd:
        if "=" in arg:
            key, value = arg.split("=", 1)
            if key in SENSITIVE_DOCKER_ENVS and value:
                sanitized.append(f"{key}=<redacted>")
                continue
        sanitized.append(arg)
    return " ".join(sanitized)


def _run_command(
    name: str,
    cmd: list[str],
    *,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    log_dir: Path,
) -> PhaseResult:
    log_path = log_dir / f"{name}.log"
    started = datetime.now(timezone.utc).isoformat()
    logged_command = _sanitize_command_for_log(cmd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
        body = [
            f"# Command: {logged_command}",
            f"# Started: {started}",
            f"# Return code: {proc.returncode}",
            "",
            "## STDOUT",
            proc.stdout or "",
            "",
            "## STDERR",
            proc.stderr or "",
        ]
        _write_text(log_path, "\n".join(body))
        return PhaseResult(
            name=name,
            status="passed" if proc.returncode == 0 else "failed",
            command=logged_command,
            returncode=proc.returncode,
            log_path=str(log_path),
        )
    except subprocess.TimeoutExpired as exc:
        body = [
            f"# Command: {logged_command}",
            f"# Started: {started}",
            "# Result: timeout",
            "",
            "## STDOUT",
            exc.stdout or "",
            "",
            "## STDERR",
            exc.stderr or "",
        ]
        _write_text(log_path, "\n".join(body))
        return PhaseResult(
            name=name,
            status="failed",
            command=logged_command,
            returncode=None,
            log_path=str(log_path),
            note=f"Timed out after {timeout}s",
        )
    except Exception as exc:
        _write_text(log_path, f"# Command: {logged_command}\n# Error: {exc}\n")
        return PhaseResult(
            name=name,
            status="failed",
            command=logged_command,
            returncode=None,
            log_path=str(log_path),
            note=str(exc),
        )


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def _docker_info_access() -> tuple[bool, str | None]:
    try:
        proc = subprocess.run(
            ["docker", "info"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except Exception as exc:
        return False, str(exc)

    if proc.returncode == 0:
        return True, None

    text = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    if "permission denied" in text.lower():
        return False, "Docker daemon is installed but this shell cannot access /var/run/docker.sock. Start a fresh login shell or run `newgrp docker`."
    return False, text or f"docker info failed with return code {proc.returncode}"


def _build_docker_run_base(
    *,
    image_tag: str,
    host_artifact_dir: Path,
    include_llm_env: bool,
    extra_docker_args: list[str] | None = None,
) -> list[str]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "ACCEPT_EULA=Y",
        "-e",
        "PRIVACY_CONSENT=Y",
        "-e",
        "OMNI_KIT_ACCEPT_EULA=YES",
        "-v",
        f"{host_artifact_dir}:{DEFAULT_CONTAINER_ARTIFACT_ROOT}",
    ]
    if extra_docker_args:
        cmd.extend(extra_docker_args)
    if include_llm_env:
        for key in DOCKER_REQUIRED_ENVS_OPENAI + DOCKER_REQUIRED_ENVS_AZURE:
            value = os.environ.get(key)
            if value:
                cmd.extend(["-e", key])
        for optional_key in OPTIONAL_DOCKER_ENVS:
            value = os.environ.get(optional_key)
            if value:
                cmd.extend(["-e", optional_key])
    cmd.append(image_tag)
    return cmd


def _expect_failure(phase: PhaseResult, *, success_note: str, failure_note: str) -> PhaseResult:
    if phase.returncode and phase.returncode != 0:
        phase.status = "passed"
        phase.note = success_note
    else:
        phase.status = "failed"
        phase.note = failure_note
    return phase


def _scan_text_for_leaks(text: str) -> list[str]:
    findings: list[str] = []
    patterns = (
        (r"/home/[A-Za-z0-9._-]+", "home_path"),
        (rf"\b{re.escape(Path.home().name)}[A-Za-z0-9._-]*\b", "personal_identifier"),
        (r"\bsk-[A-Za-z0-9]{20,}\b", "openai_key"),
        (r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "github_token"),
        (r"\bhf_[A-Za-zA-Z0-9]{20,}\b", "huggingface_token"),
    )
    for pattern, label in patterns:
        for match in re.finditer(pattern, text):
            findings.append(f"{label}:{match.group(0)}")
    return findings


def _write_summary(output_dir: Path, phases: list[PhaseResult], extra: dict[str, object]) -> tuple[Path, Path]:
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "phases": [asdict(p) for p in phases],
        "summary": {
            "passed": sum(1 for p in phases if p.status == "passed"),
            "failed": sum(1 for p in phases if p.status == "failed"),
            "skipped": sum(1 for p in phases if p.status == "skipped"),
        },
        "extra": extra,
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Docker Release Validation Summary",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Passed: `{summary['summary']['passed']}`",
        f"- Failed: `{summary['summary']['failed']}`",
        f"- Skipped: `{summary['summary']['skipped']}`",
        "",
        "| Phase | Status | Command | Note |",
        "| --- | --- | --- | --- |",
    ]
    for phase in phases:
        lines.append(
            f"| {phase.name} | {phase.status} | `{phase.command or ''}` | {phase.note or ''} |"
        )
    md_path = output_dir / "summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Docker release end-to-end.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG)
    parser.add_argument("--build-no-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-build", action="store_true", help="Reuse an existing image tag instead of rebuilding it")
    parser.add_argument("--run-full-soak", action="store_true")
    parser.add_argument(
        "--resume-kill-delay",
        type=int,
        default=90,
        help="Seconds to wait before killing the representative resume probe container",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve() / _timestamp()
    logs_dir = output_dir / "logs"
    host_artifact_dir = output_dir / "artifacts"
    logs_dir.mkdir(parents=True, exist_ok=True)
    host_artifact_dir.mkdir(parents=True, exist_ok=True)

    phases: list[PhaseResult] = []
    extra: dict[str, object] = {
        "docker_available": _docker_available(),
        "gpu_available": shutil.which("nvidia-smi") is not None,
        "full_soak_requested": bool(args.run_full_soak),
    }

    phases.append(
        _run_command(
            "repo_audit",
            [sys.executable, "scripts/audit_release_repo.py", "--output-dir", str(output_dir / "release_audit")],
            log_dir=logs_dir,
        )
    )

    if not _docker_available():
        phases.append(
            PhaseResult(
                name="docker_prerequisite",
                status="failed",
                command="docker --version",
                returncode=127,
                log_path=None,
                note="Docker CLI is not installed on this machine",
            )
        )
        json_path, md_path = _write_summary(output_dir, phases, extra)
        print("Docker validation incomplete: Docker CLI not available")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        return 2

    docker_access_ok, docker_access_note = _docker_info_access()
    extra["docker_accessible"] = docker_access_ok
    if not docker_access_ok:
        phases.append(
            PhaseResult(
                name="docker_access_prerequisite",
                status="failed",
                command="docker info",
                returncode=1,
                log_path=None,
                note=docker_access_note,
            )
        )
        json_path, md_path = _write_summary(output_dir, phases, extra)
        print("Docker validation incomplete: Docker daemon access is not ready in this shell")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        return 2

    phases.append(_run_command("docker_version", ["docker", "--version"], log_dir=logs_dir))
    phases.append(_run_command("docker_info", ["docker", "info"], log_dir=logs_dir, timeout=300))
    phases.append(_run_command("docker_compose_version", ["docker", "compose", "version"], log_dir=logs_dir, timeout=300))

    if args.skip_build:
        image_present = _run_command(
            "docker_image_present",
            ["docker", "image", "inspect", args.image_tag],
            log_dir=logs_dir,
            timeout=300,
        )
        phases.append(image_present)
        if image_present.status != "passed":
            json_path, md_path = _write_summary(output_dir, phases, extra)
            print("Docker validation blocked: requested --skip-build but image tag does not exist")
            print(f"JSON report: {json_path}")
            print(f"Markdown report: {md_path}")
            return 1
        phases.append(
            PhaseResult(
                name="docker_build",
                status="skipped",
                command=None,
                returncode=None,
                log_path=None,
                note=f"Skipped rebuild and reused image tag {args.image_tag}",
            )
        )
    else:
        build_cmd = ["docker", "build", "-t", args.image_tag]
        if args.build_no_cache:
            build_cmd.append("--no-cache")
        build_cmd.append(".")
        phases.append(_run_command("docker_build", build_cmd, log_dir=logs_dir, timeout=14400))
        if phases[-1].status != "passed":
            json_path, md_path = _write_summary(output_dir, phases, extra)
            print("Docker build failed")
            print(f"JSON report: {json_path}")
            print(f"Markdown report: {md_path}")
            return 1

    inspect_phase = _run_command("docker_inspect", ["docker", "inspect", args.image_tag], log_dir=logs_dir)
    history_phase = _run_command("docker_history", ["docker", "history", "--no-trunc", args.image_tag], log_dir=logs_dir)
    phases.extend([inspect_phase, history_phase])
    leak_findings: list[str] = []
    for phase in (inspect_phase, history_phase):
        if phase.log_path:
            leak_findings.extend(_scan_text_for_leaks(Path(phase.log_path).read_text(encoding="utf-8", errors="ignore")))
    extra["image_audit_findings"] = leak_findings
    phases.append(
        PhaseResult(
            name="image_audit",
            status="passed" if not leak_findings else "failed",
            command=None,
            returncode=0 if not leak_findings else 1,
            log_path=None,
            note="No obvious leaks found in docker inspect/history output" if not leak_findings else "Potential secret or host path found in image metadata",
        )
    )

    phases.append(
        _run_command(
            "entrypoint_help",
            _build_docker_run_base(image_tag=args.image_tag, host_artifact_dir=host_artifact_dir, include_llm_env=False) + ["--help"],
            log_dir=logs_dir,
            timeout=300,
        )
    )
    phases.append(
        _expect_failure(
            _run_command(
                "entrypoint_invalid_mode",
                _build_docker_run_base(image_tag=args.image_tag, host_artifact_dir=host_artifact_dir, include_llm_env=False) + ["--mode", "mcp"],
                log_dir=logs_dir,
                timeout=300,
            ),
            success_note="Unsupported mode failed fast as expected",
            failure_note="Unsupported mode unexpectedly succeeded",
        )
    )
    phases.append(
        _expect_failure(
            _run_command(
                "missing_env_failfast",
                ["docker", "run", "--rm", args.image_tag, "--mode", "e2e-batch"],
                log_dir=logs_dir,
                timeout=300,
            ),
            success_note="Missing env vars failed fast as expected",
            failure_note="Container unexpectedly ran without required env vars",
        )
    )
    phases.append(
        _run_command(
            "container_base_env",
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "bash",
                args.image_tag,
                "-lc",
                "source /etc/os-release && "
                "echo ID=$ID && echo VERSION_ID=$VERSION_ID && "
                "python --version && "
                "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
            ],
            log_dir=logs_dir,
            timeout=900,
        )
    )
    phases.append(
        _run_command(
            "isaaclab_import",
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "python",
                args.image_tag,
                "-c",
                "import isaaclab, isaacsim; print(isaaclab.__file__); print(isaacsim.__file__)",
            ],
            log_dir=logs_dir,
            timeout=900,
        )
    )

    has_openai = all(os.environ.get(k) for k in DOCKER_REQUIRED_ENVS_OPENAI)
    has_azure = all(os.environ.get(k) for k in DOCKER_REQUIRED_ENVS_AZURE)
    missing_envs: list[str] = []
    if not has_openai and not has_azure:
        missing_envs = list(DOCKER_REQUIRED_ENVS_OPENAI)
    extra["missing_runtime_envs"] = missing_envs
    if missing_envs:
        phases.append(
            PhaseResult(
                name="functional_prerequisites",
                status="failed",
                command=None,
                returncode=2,
                log_path=None,
                note=(
                    "Missing API key. Set OPENAI_API_KEY (OpenAI platform) "
                    "or AZURE_OPENAI_API_KEY + AZURE_OPENAI_BASE_URL (Azure)."
                ),
            )
        )
        json_path, md_path = _write_summary(output_dir, phases, extra)
        print("Functional Docker validation blocked by missing environment variables")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        return 1

    runtime_base = _build_docker_run_base(
        image_tag=args.image_tag,
        host_artifact_dir=host_artifact_dir,
        include_llm_env=True,
    )
    functional_steps = [
        (
            "isaac_lab_franka_lift",
            runtime_base + ["--mode", "isaac-lab", "--task", "tasks/franka/lift/franka_lift.yaml", "--", "--evaluate"],
            3600,
        ),
        (
            "isaac_lab_franka_pick_place_drawer",
            runtime_base + ["--mode", "isaac-lab", "--task", "tasks/franka/pick_place/franka_pick_place_drawer.yaml", "--", "--evaluate"],
            3600,
        ),
        (
            "isaac_lab_franka_stack_tray",
            runtime_base + ["--mode", "isaac-lab", "--task", "tasks/franka/stack/franka_stack_tray.yaml", "--", "--evaluate"],
            3600,
        ),
        (
            "data_collection_franka_lift",
            runtime_base + ["--mode", "data-collection", "--task", "tasks/franka/lift/franka_lift.yaml", "--", "--target-success", "2", "--max-attempts", "8"],
            5400,
        ),
        (
            "data_collection_franka_stack",
            runtime_base + ["--mode", "data-collection", "--task", "tasks/franka/stack/franka_stack.yaml", "--", "--target-success", "2", "--max-attempts", "10"],
            5400,
        ),
        (
            "data_collection_franka_color_sort",
            runtime_base + ["--mode", "data-collection", "--task", "tasks/franka/sort/franka_color_sort.yaml", "--", "--target-success", "2", "--max-attempts", "10"],
            5400,
        ),
        (
            "representative_e2e_batch",
            runtime_base + ["--mode", "e2e-batch", "--config", "configs/docker/e2e_batch_representative.yaml"],
            14400,
        ),
    ]
    for name, cmd, timeout in functional_steps:
        phases.append(_run_command(name, cmd, log_dir=logs_dir, timeout=timeout))

    resume_name = f"simgen-release-resume-{int(time.time())}"
    resume_cmd = _build_docker_run_base(
        image_tag=args.image_tag,
        host_artifact_dir=host_artifact_dir,
        include_llm_env=True,
        extra_docker_args=["--name", resume_name],
    ) + ["--mode", "e2e-batch", "--config", "configs/docker/e2e_batch_representative.yaml"]
    resume_log = logs_dir / "resume_interruption.log"
    try:
        proc = subprocess.Popen(
            resume_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(max(args.resume_kill_delay, 5))
        subprocess.run(["docker", "kill", resume_name], capture_output=True, text=True)
        stdout = proc.communicate(timeout=120)[0] if proc.stdout else ""
        _write_text(resume_log, stdout)
        phases.append(
            PhaseResult(
                name="resume_interruption",
                status="passed",
                command=" ".join(resume_cmd),
                returncode=0,
                log_path=str(resume_log),
                note=f"Killed container {resume_name} after {args.resume_kill_delay}s",
            )
        )
    except Exception as exc:
        _write_text(resume_log, f"Resume interruption failed: {exc}\n")
        phases.append(
            PhaseResult(
                name="resume_interruption",
                status="failed",
                command=" ".join(resume_cmd),
                returncode=None,
                log_path=str(resume_log),
                note=str(exc),
            )
        )
    finally:
        subprocess.run(["docker", "rm", "-f", resume_name], capture_output=True, text=True)

    phases.append(
        _run_command(
            "resume_recovery",
            runtime_base + ["--mode", "e2e-batch", "--config", "configs/docker/e2e_batch_representative.yaml", "--resume"],
            log_dir=logs_dir,
            timeout=14400,
        )
    )

    if args.run_full_soak:
        phases.append(
            _run_command(
                "full_soak_release_batch",
                runtime_base + ["--mode", "e2e-batch", "--config", "configs/docker/e2e_batch_release.yaml", "--resume"],
                log_dir=logs_dir,
                timeout=86400,
            )
        )
    else:
        phases.append(
            PhaseResult(
                name="full_soak_release_batch",
                status="skipped",
                command=None,
                returncode=None,
                log_path=None,
                note="Not executed. Re-run with --run-full-soak for final acceptance.",
            )
        )

    json_path, md_path = _write_summary(output_dir, phases, extra)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    failed = sum(1 for p in phases if p.status == "failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
