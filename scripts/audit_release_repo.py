#!/usr/bin/env python3
"""Static release audit for secrets, public wording, and Docker runtime targets."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "release_audit"
CURRENT_USERNAME = Path.home().name
PUBLIC_CRITICAL_PREFIXES = (
    "Dockerfile",
    ".dockerignore",
    "README.docker.md",
    "run_agent.sh",
    "configs/docker/",
    "docs/release/",
    "scripts/audit_release_repo.py",
    "scripts/validate_docker_release.py",
)
REQUIRED_PATHS = (
    "Dockerfile",
    ".dockerignore",
    "README.docker.md",
    "run_agent.sh",
    "requirements.txt",
    "src/main.py",
    "data/input_sample.json",
    "configs/docker/data_collection_release.yaml",
    "configs/docker/e2e_batch_release.yaml",
    "configs/docker/e2e_batch_representative.yaml",
    "configs/docker/e2e_batch_smoke.yaml",
    "configs/docker/isaaclab_agent_release.yaml",
    "docs/release/docker_release_checklist.md",
    "scripts/audit_release_repo.py",
    "scripts/validate_docker_release.py",
)
DOCKERIGNORE_REQUIRED_TOKENS = (".git", "outputs", "artifacts", ".env")
README_REQUIRED_HEADINGS = (
    "## Build",
    "## Help / Healthcheck",
    "## Representative Validation",
    "## Full Validation",
)
README_REQUIRED_SNIPPETS = (
    "docker build -t simgen-isaaclab:release .",
    "docker run --rm simgen-isaaclab:release --help",
    "python3 scripts/validate_docker_release.py",
    "python3 scripts/validate_docker_release.py --run-full-soak",
    "https://docs.docker.com/engine/install/ubuntu/",
    "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html",
)
DOCKERFILE_REQUIRED_SUBSTRINGS = (
    "FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04",
    "ARG PYTHON_VERSION=3.11",
    'isaacsim[all,extscache]==${ISAACSIM_PIP_VERSION}',
    'ARG ISAACSIM_PIP_VERSION=5.1.0',
    'ARG ISAACLAB_VERSION=v2.3.2',
    "./isaaclab.sh -i none",
)
DOCKER_CONFIG_REQUIRED_TOKENS = (
    ('configs/docker/data_collection_release.yaml', 'launch_mode: "direct"'),
    ('configs/docker/data_collection_release.yaml', "use_launcher: false"),
    ('configs/docker/isaaclab_agent_release.yaml', 'launch_mode: "direct"'),
    ('configs/docker/isaaclab_agent_release.yaml', "use_launcher: false"),
)
PLACEHOLDER_VALUES = {
    "...",
    "<repo-url>",
    "<org>",
    "<dataset_name>",
    "your-key",
    "your-key-here",
    "your-anthropic-key-here",
    "your-google-key-here",
    "your-azure-openai-key-here",
    "your-hf-namespace",
}
ALLOWLIST_ABSOLUTE_PREFIXES = (
    "/workspace/IsaacLab",
    "/workspace/artifacts",
    "/workspace/Simulation-Generation-Agent",
    "/workspace/isaaclab",
    "/Isaac",
    "/IsaacLab",
    "/isaac-sim",
    "/usr/bin/env",
    "/opt/isaaclab-env",
)
PUBLIC_WORDING_PATTERN = re.compile(
    "|".join(
        [
            r"p" r"df",
            "sub" "mission" " guideline",
            "제" "출 " "기" "준",
            "docs/" "sub" "mission/",
            "docker_" "p" "df_" "compliance",
        ]
    ),
    re.IGNORECASE,
)
SECRET_TOKEN_PATTERNS = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-zA-Z0-9]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b")),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
)
ENV_ASSIGNMENT_PATTERN = re.compile(
    r"(?m)^\s*([A-Z0-9_]*(?:API_KEY|_TOKEN|_SECRET))\s*[:=]\s*([^\s#]+)"
)
HOME_PATH_PATTERN = re.compile(r"/home/([A-Za-z0-9._-]+)")


@dataclass
class Finding:
    severity: str
    code: str
    path: str
    line: int | None
    message: str
    excerpt: str | None = None


def _run_git_ls_files() -> list[str]:
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(PROJECT_ROOT),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            check=True,
            capture_output=True,
        )
        return [p for p in proc.stdout.decode("utf-8", errors="ignore").split("\0") if p]
    except Exception:
        return []


def _iter_repo_files() -> list[Path]:
    tracked = _run_git_ls_files()
    if tracked:
        return [PROJECT_ROOT / rel for rel in tracked if rel and not rel.startswith("outputs/")]
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith(("outputs/", ".git/", "external/AutoDataCollector/")):
            continue
        files.append(path)
    return sorted(files)


def _is_public_critical(rel_path: str) -> bool:
    return any(rel_path == prefix or rel_path.startswith(prefix) for prefix in PUBLIC_CRITICAL_PREFIXES)


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="ignore")


def _looks_like_placeholder(value: str) -> bool:
    lowered = value.strip().strip("\"'").lower()
    if not lowered:
        return True
    if lowered in {item.lower() for item in PLACEHOLDER_VALUES}:
        return True
    if lowered.startswith(("your-", "example", "<", "${", "$", "local/", "http://localhost")):
        return True
    return False


def _excerpt(line: str) -> str:
    line = line.strip()
    return line if len(line) <= 180 else f"{line[:177]}..."


def _add_finding(
    findings: list[Finding],
    *,
    severity: str,
    code: str,
    path: Path,
    line: int | None,
    message: str,
    excerpt: str | None = None,
) -> None:
    findings.append(
        Finding(
            severity=severity,
            code=code,
            path=path.relative_to(PROJECT_ROOT).as_posix(),
            line=line,
            message=message,
            excerpt=excerpt,
        )
    )


def audit_repo() -> list[Finding]:
    findings: list[Finding] = []

    for rel in REQUIRED_PATHS:
        if not (PROJECT_ROOT / rel).exists():
            _add_finding(
                findings,
                severity="blocker",
                code="missing_required_path",
                path=PROJECT_ROOT / rel,
                line=None,
                message=f"Required release path is missing: {rel}",
            )

    dockerfile = PROJECT_ROOT / "Dockerfile"
    docker_text = _read_text(dockerfile) or ""
    for token in DOCKERFILE_REQUIRED_SUBSTRINGS:
        if token not in docker_text:
            _add_finding(
                findings,
                severity="blocker",
                code="dockerfile_runtime_target",
                path=dockerfile,
                line=None,
                message=f"Dockerfile is missing required runtime token: {token}",
            )

    dockerignore = PROJECT_ROOT / ".dockerignore"
    dockerignore_text = _read_text(dockerignore) or ""
    for token in DOCKERIGNORE_REQUIRED_TOKENS:
        if token not in dockerignore_text:
            _add_finding(
                findings,
                severity="blocker",
                code="dockerignore_required_token",
                path=dockerignore,
                line=None,
                message=f".dockerignore must exclude '{token}'",
            )

    readme = PROJECT_ROOT / "README.docker.md"
    readme_text = _read_text(readme) or ""
    for heading in README_REQUIRED_HEADINGS:
        if heading not in readme_text:
            _add_finding(
                findings,
                severity="blocker",
                code="readme_required_heading",
                path=readme,
                line=None,
                message=f"README.docker.md must include heading: {heading}",
            )
    for snippet in README_REQUIRED_SNIPPETS:
        if snippet not in readme_text:
            _add_finding(
                findings,
                severity="blocker",
                code="readme_required_snippet",
                path=readme,
                line=None,
                message=f"README.docker.md must include command example: {snippet}",
            )

    for rel, token in DOCKER_CONFIG_REQUIRED_TOKENS:
        path = PROJECT_ROOT / rel
        text = _read_text(path) or ""
        if token not in text:
            _add_finding(
                findings,
                severity="blocker",
                code="docker_config_runtime_mode",
                path=path,
                line=None,
                message=f"{rel} must include `{token}`",
            )

    for path in _iter_repo_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        if rel.startswith(("outputs/", ".git/", "external/AutoDataCollector/")):
            continue

        text = _read_text(path)
        if text is None:
            continue
        lines = text.splitlines()
        critical = _is_public_critical(rel)
        personal_pattern = re.compile(rf"\b{re.escape(CURRENT_USERNAME)}[A-Za-z0-9._-]*\b")

        for idx, line in enumerate(lines, start=1):
            if critical and PUBLIC_WORDING_PATTERN.search(line):
                _add_finding(
                    findings,
                    severity="blocker",
                    code="public_wording",
                    path=path,
                    line=idx,
                    message="Public critical file contains release-internal wording that should not be published",
                    excerpt=_excerpt(line),
                )

            for match in personal_pattern.finditer(line):
                _add_finding(
                    findings,
                    severity="blocker" if critical else "warning",
                    code="personal_identifier",
                    path=path,
                    line=idx,
                    message=f"Personal identifier '{match.group(0)}' found",
                    excerpt=_excerpt(line),
                )

            for match in HOME_PATH_PATTERN.finditer(line):
                user = match.group(1)
                if any(token in line for token in ('r"/home/', "HOME_PATH_PATTERN", "/home/<", "/home/[", "Use ${HOME}")):
                    continue
                if user in {"$USER", "you", "username", "example"}:
                    _add_finding(
                        findings,
                        severity="warning",
                        code="placeholder_home_path",
                        path=path,
                        line=idx,
                        message="Use ${HOME} or /path/to/... instead of /home/<placeholder>",
                        excerpt=_excerpt(line),
                    )
                    continue
                if any(line.strip().startswith(prefix) for prefix in ALLOWLIST_ABSOLUTE_PREFIXES):
                    continue
                _add_finding(
                    findings,
                    severity="blocker" if critical else "warning",
                    code="home_path",
                    path=path,
                    line=idx,
                    message=f"Host absolute home path detected: {match.group(0)}",
                    excerpt=_excerpt(line),
                )

            for code, pattern in SECRET_TOKEN_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                _add_finding(
                    findings,
                    severity="blocker",
                    code=code,
                    path=path,
                    line=idx,
                    message=f"Potential secret detected by pattern '{code}'",
                    excerpt=_excerpt(line),
                )

            assign_match = ENV_ASSIGNMENT_PATTERN.search(line)
            if assign_match:
                key, value = assign_match.groups()
                if _looks_like_placeholder(value):
                    continue
                _add_finding(
                    findings,
                    severity="blocker",
                    code="env_secret_assignment",
                    path=path,
                    line=idx,
                    message=f"Possible hardcoded secret assignment for {key}",
                    excerpt=_excerpt(line),
                )

    return findings


def write_reports(output_dir: Path, findings: list[Finding]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    blockers = [f for f in findings if f.severity == "blocker"]
    warnings = [f for f in findings if f.severity == "warning"]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(PROJECT_ROOT),
        "summary": {
            "blockers": len(blockers),
            "warnings": len(warnings),
            "passed": len(blockers) == 0,
        },
        "findings": [asdict(f) for f in findings],
    }
    json_path = output_dir / "audit_report.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Release Audit Report",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Repo root: `{PROJECT_ROOT}`",
        f"- Blockers: `{len(blockers)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
    ]
    if not findings:
        lines.append("No findings.")
    else:
        lines.extend(
            [
                "| Severity | Code | Path | Line | Message |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for finding in findings:
            lines.append(
                f"| {finding.severity} | {finding.code} | `{finding.path}` | "
                f"{finding.line or ''} | {finding.message} |"
            )
            if finding.excerpt:
                lines.append(f"|  | excerpt |  |  | `{finding.excerpt}` |")
    md_path = output_dir / "audit_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the repository for Docker release readiness.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    findings = audit_repo()
    json_path, md_path = write_reports(output_dir, findings)

    blockers = sum(1 for finding in findings if finding.severity == "blocker")
    warnings = sum(1 for finding in findings if finding.severity == "warning")
    print(f"Release audit complete: blockers={blockers}, warnings={warnings}")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0 if blockers == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
