"""API token usage tracker for LLM/VLM calls across the pipeline.

Tracks input/output tokens per API call with step labels.
Supports cross-process aggregation via JSONL file (TOKEN_USAGE_FILE env var).

Set TOKEN_USAGE_LOG=1 to print real-time per-call logs to stderr.

Usage::

    from src.agent.common.token_tracker import tracker

    tracker.record("cap_codegen", "gpt-5-mini", input_tokens=1200, output_tokens=800)
    print(tracker.summary())
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Step display names for prettier output
_STEP_DISPLAY_NAMES: dict[str, str] = {
    "task_spec_agent": "NL Parse/Decompose",
    "rag_yaml_gen": "RAG YAML Gen",
    "isaaclab_codegen": "IsaacLab CodeGen",
    "isaaclab_error_fix": "IsaacLab ErrorFix",
    "cap_codegen": "CaP CodeGen",
    "vlm_judge": "VLM Judge",
    "llm_generate": "LLM Generate",
}


@dataclass
class APICallRecord:
    timestamp: str
    step: str
    model: str
    input_tokens: int
    output_tokens: int


class TokenTracker:
    """Thread-safe token usage tracker with file-based cross-process aggregation."""

    def __init__(self) -> None:
        self._records: list[APICallRecord] = []
        self._lock = threading.Lock()

    def record(
        self,
        step: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        in_tok = int(input_tokens)
        out_tok = int(output_tokens)
        rec = APICallRecord(
            timestamp=datetime.now().isoformat(),
            step=step,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )
        with self._lock:
            self._records.append(rec)
        logger.debug(
            "Token usage: step=%s model=%s in=%d out=%d",
            step, model, in_tok, out_tok,
        )
        # Real-time stderr log when TOKEN_USAGE_LOG is set
        if os.environ.get("TOKEN_USAGE_LOG"):
            display = _STEP_DISPLAY_NAMES.get(step, step)
            total = in_tok + out_tok
            print(
                f"  [API] {display:<20s} | {model:<14s} "
                f"| in: {in_tok:>7,} | out: {out_tok:>6,} | total: {total:>8,}",
                file=sys.stderr,
                flush=True,
            )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)

        total_input = sum(r.input_tokens for r in records)
        total_output = sum(r.output_tokens for r in records)
        total_calls = len(records)

        by_step: dict[str, dict[str, int]] = {}
        for r in records:
            if r.step not in by_step:
                by_step[r.step] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            by_step[r.step]["input_tokens"] += r.input_tokens
            by_step[r.step]["output_tokens"] += r.output_tokens
            by_step[r.step]["calls"] += 1

        by_model: dict[str, dict[str, int]] = {}
        for r in records:
            if r.model not in by_model:
                by_model[r.model] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["calls"] += 1

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_api_calls": total_calls,
            "by_step": by_step,
            "by_model": by_model,
        }

    def report(self) -> str:
        s = self.summary()
        if s["total_api_calls"] == 0:
            return "No API calls recorded."

        sep = "=" * 66
        lines = [
            sep,
            "  TOKEN USAGE REPORT",
            sep,
            f"  Total: {s['total_tokens']:,} tokens "
            f"({s['total_input_tokens']:,} in + {s['total_output_tokens']:,} out) "
            f"| {s['total_api_calls']} API calls",
            "",
        ]

        # --- By Step table ---
        lines.append("  By Pipeline Stage:")
        hdr = f"  {'Stage':<22s} {'Calls':>6s} {'Input':>10s} {'Output':>10s} {'Total':>10s}"
        lines.append(f"  {'-' * 62}")
        lines.append(hdr)
        lines.append(f"  {'-' * 62}")
        for step, data in sorted(s["by_step"].items()):
            display = _STEP_DISPLAY_NAMES.get(step, step)
            total = data["input_tokens"] + data["output_tokens"]
            lines.append(
                f"  {display:<22s} {data['calls']:>6,} "
                f"{data['input_tokens']:>10,} {data['output_tokens']:>10,} {total:>10,}"
            )
        lines.append(f"  {'-' * 62}")
        lines.append("")

        # --- By Model table ---
        lines.append("  By Model:")
        lines.append(f"  {'-' * 46}")
        lines.append(f"  {'Model':<22s} {'Calls':>6s} {'Total Tokens':>14s}")
        lines.append(f"  {'-' * 46}")
        for model, data in sorted(s["by_model"].items()):
            total = data["input_tokens"] + data["output_tokens"]
            lines.append(f"  {model:<22s} {data['calls']:>6,} {total:>14,}")
        lines.append(f"  {'-' * 46}")
        lines.append(sep)

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "summary": self.summary(),
                "records": [asdict(r) for r in self._records],
            }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info("Token usage saved to %s", path)

    def append_to_file(self, path: Path) -> None:
        """Append records as JSONL for cross-process aggregation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            records = list(self._records)
        if not records:
            return
        with open(path, "a") as f:
            for r in records:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def reset(self) -> None:
        with self._lock:
            self._records.clear()

    @staticmethod
    def aggregate_from_file(path: str | Path) -> dict[str, Any]:
        """Read JSONL file and produce aggregated summary."""
        path = Path(path)
        if not path.exists():
            return {"total_tokens": 0, "total_api_calls": 0}

        records: list[APICallRecord] = []
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            records.append(APICallRecord(**d))

        tmp = TokenTracker()
        tmp._records = records
        return tmp.summary()

    @staticmethod
    def report_from_file(path: str | Path) -> str:
        """Read JSONL file and produce formatted report."""
        path = Path(path)
        if not path.exists():
            return "No token usage data found."
        tmp = TokenTracker()
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            d = json.loads(line)
            tmp._records.append(APICallRecord(**d))
        return tmp.report()


# Module-level singleton
tracker = TokenTracker()


def _save_on_exit() -> None:
    usage_file = os.environ.get("TOKEN_USAGE_FILE")
    if usage_file and tracker._records:
        tracker.append_to_file(Path(usage_file))


atexit.register(_save_on_exit)
