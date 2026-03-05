"""Isaac Sim pipeline runner: build scene -> capture screenshot -> optional VLM loop."""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from src.common.mcp_client import MCPClient
from src.isaac_sim.scene_builder import SceneBuilder
from src.isaac_sim.screenshot import ScreenshotCapture
from src.isaac_sim.vlm_evaluator import create_evaluator, load_task_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class IsaacSimRunner:
    """Orchestrates the Isaac Sim pipeline for one task YAML."""

    def __init__(self, config_path: str | None = None):
        config_path = Path(config_path) if config_path else (PROJECT_ROOT / "configs" / "pipeline_config.yaml")
        with open(config_path) as f:
            self.config = yaml.safe_load(f) or {}

        vlm_cfg_path = PROJECT_ROOT / "configs" / "vlm_config.yaml"
        vlm_cfg = {}
        if vlm_cfg_path.exists():
            with open(vlm_cfg_path) as f:
                vlm_cfg = yaml.safe_load(f) or {}
        retry_cfg = vlm_cfg.get("retry", {})
        self.vlm_retry_max_retries = int(retry_cfg.get("max_retries", 3))
        self.vlm_retry_delay = float(retry_cfg.get("retry_delay", 2.0))

    def _resolution(self) -> tuple[int, int]:
        screenshot_cfg = self.config.get("screenshot", {})
        resolution = screenshot_cfg.get("resolution", [1280, 720])
        if isinstance(resolution, (list, tuple)) and len(resolution) == 2:
            return int(resolution[0]), int(resolution[1])
        return 1280, 720

    @staticmethod
    def _make_run_dir(yaml_path: Path, output_root: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_slug = yaml_path.stem.lower()
        run_dir = output_root / f"{task_slug}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def run(
        self,
        yaml_path: str,
        skip_vlm: bool = False,
        backend: str = "auto",
        max_iterations: int | None = None,
        threshold: int | None = None,
        output_dir: str | None = None,
    ) -> dict:
        """Run the full pipeline and return a machine-readable report."""
        yaml_path_obj = Path(yaml_path).expanduser().resolve()
        if not yaml_path_obj.exists():
            return {
                "success": False,
                "error": f"Task YAML not found: {yaml_path_obj}",
            }

        output_root = Path(output_dir).expanduser().resolve() if output_dir else (PROJECT_ROOT / "outputs" / "isaac_sim")
        run_dir = self._make_run_dir(yaml_path_obj, output_root)

        pipeline_cfg = self.config.get("pipeline", {})
        iteration_delay = float(pipeline_cfg.get("iteration_delay", 2.0))
        configured_max_iterations = int(pipeline_cfg.get("max_iterations", 5))
        effective_max_iterations = int(max_iterations or configured_max_iterations)
        effective_max_iterations = max(1, effective_max_iterations)

        score_threshold = int(threshold if threshold is not None else pipeline_cfg.get("threshold", 80))
        if skip_vlm:
            # No scoring loop without VLM evaluation.
            effective_max_iterations = 1

        mcp_cfg = self.config.get("mcp", {})
        host = mcp_cfg.get("host", "localhost")
        port = int(mcp_cfg.get("port", 8766))
        timeout = float(mcp_cfg.get("timeout", 300))
        resolution = self._resolution()

        client = MCPClient(host=host, port=port, timeout=timeout)
        builder = SceneBuilder(client)
        capture = ScreenshotCapture(client)
        document = load_task_document(yaml_path_obj)
        evaluator = None if skip_vlm else create_evaluator(backend=backend)

        scene_info = client.get_scene_info()
        if "error" in scene_info:
            return {
                "success": False,
                "error": f"MCP connection failed: {scene_info['error']}",
                "output_dir": str(run_dir),
            }

        iteration_reports = []
        success = False
        best_score = -1
        best_iteration = None

        try:
            for iteration in range(1, effective_max_iterations + 1):
                iter_report = {"iteration": iteration}

                build_result = builder.build_from_yaml(yaml_path_obj)
                iter_report["build"] = build_result

                if not build_result.get("success", False):
                    iter_report["status"] = "build_failed"
                    iteration_reports.append(iter_report)
                    if iteration < effective_max_iterations:
                        time.sleep(iteration_delay)
                    continue

                screenshot_path = run_dir / f"iter_{iteration:02d}.png"
                screenshot_result = capture.capture(screenshot_path, resolution=resolution)
                iter_report["screenshot"] = screenshot_result
                if not screenshot_result.get("success", False):
                    iter_report["status"] = "screenshot_failed"
                    iteration_reports.append(iter_report)
                    if iteration < effective_max_iterations:
                        time.sleep(iteration_delay)
                    continue

                if skip_vlm:
                    iter_report["status"] = "success"
                    iteration_reports.append(iter_report)
                    success = True
                    break

                screenshot_file = screenshot_result.get("path", str(screenshot_path))
                eval_result = evaluator.evaluate_with_retry(
                    screenshot_file,
                    document,
                    max_retries=self.vlm_retry_max_retries,
                    retry_delay=self.vlm_retry_delay,
                )
                iter_report["evaluation"] = eval_result

                if not eval_result.get("success", False):
                    iter_report["status"] = "evaluation_failed"
                    iter_report["score"] = 0
                    iteration_reports.append(iter_report)
                    if iteration < effective_max_iterations:
                        time.sleep(iteration_delay)
                    continue

                score = int(eval_result.get("score", 0))
                iter_report["score"] = score
                iter_report["status"] = "scored"
                iteration_reports.append(iter_report)

                if score > best_score:
                    best_score = score
                    best_iteration = iteration

                if score >= score_threshold:
                    success = True
                    break

                if iteration < effective_max_iterations:
                    time.sleep(iteration_delay)
        finally:
            client.disconnect()

        report = {
            "success": success if not skip_vlm else (len(iteration_reports) > 0 and iteration_reports[-1]["status"] == "success"),
            "skip_vlm": skip_vlm,
            "backend": None if skip_vlm else backend,
            "yaml_path": str(yaml_path_obj),
            "output_dir": str(run_dir),
            "iterations": iteration_reports,
            "max_iterations": effective_max_iterations,
            "threshold": None if skip_vlm else score_threshold,
            "best_score": None if skip_vlm or best_score < 0 else best_score,
            "best_iteration": best_iteration,
        }

        report_path = run_dir / "run_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["report_path"] = str(report_path)

        return report


def main():
    parser = argparse.ArgumentParser(description="Run Isaac Sim pipeline with optional VLM loop")
    parser.add_argument("yaml_path", type=str, help="Path to task YAML document")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM evaluation and run one build/capture pass")
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "azure", "claude", "gemini", "ollama", "mock"],
        help="VLM backend",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Override max iterations")
    parser.add_argument("--threshold", type=int, default=None, help="Override score threshold")
    parser.add_argument("--output-dir", type=str, default=None, help="Output root directory for run artifacts")
    parser.add_argument("--config", type=str, default=None, help="Path to pipeline config YAML")
    args = parser.parse_args()

    runner = IsaacSimRunner(config_path=args.config)
    result = runner.run(
        yaml_path=args.yaml_path,
        skip_vlm=args.skip_vlm,
        backend=args.backend,
        max_iterations=args.max_iterations,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )

    if result.get("success"):
        print("Pipeline succeeded")
        print(f"Output dir: {result.get('output_dir')}")
        if result.get("best_score") is not None:
            print(f"Best score: {result['best_score']}")
    else:
        print("Pipeline failed")
        if result.get("error"):
            print(f"Error: {result['error']}")
        print(f"Output dir: {result.get('output_dir')}")

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
