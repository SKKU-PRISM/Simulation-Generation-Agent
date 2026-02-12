#!/usr/bin/env python3
"""
Test Components - Quick tests for individual pipeline components.

Tests each component independently without requiring the full pipeline.

Usage:
    python tests/test_components.py connection
    python tests/test_components.py scene tasks/franka/stack/franka_stack.yaml
    python tests/test_components.py screenshot outputs/test.png
    python tests/test_components.py full --skip-vlm
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_mcp_connection(host: str = "localhost", port: int = 8766) -> bool:
    """Test MCP connection to Isaac Sim."""
    from src.common.mcp_client import MCPClient

    logger.info(f"Testing MCP connection to {host}:{port}...")
    client = MCPClient(host=host, port=port)

    result = client.get_scene_info()
    if "error" in result:
        logger.error(f"Connection failed: {result['error']}")
        return False

    logger.info(f"Connection successful!")
    logger.info(f"Scene info: {result}")
    client.disconnect()
    return True


def test_scene_builder(document_path: str, host: str = "localhost", port: int = 8766) -> bool:
    """Test scene builder with a document."""
    from src.common.mcp_client import MCPClient
    from src.isaac_sim.scene_builder import SceneBuilder

    logger.info(f"Testing scene builder with {document_path}...")

    client = MCPClient(host=host, port=port)
    builder = SceneBuilder(client)

    result = builder.build_from_yaml(document_path)

    if result["success"]:
        logger.info("Scene built successfully!")
        logger.info(f"Created prims: {result['created_prims']}")
    else:
        logger.error(f"Scene build failed: {result['errors']}")

    if result.get("warnings"):
        logger.warning(f"Warnings: {result['warnings']}")

    client.disconnect()
    return result["success"]


def test_screenshot_capture(output_path: str, host: str = "localhost", port: int = 8766) -> bool:
    """Test screenshot capture."""
    from src.common.mcp_client import MCPClient
    from src.isaac_sim.screenshot import ScreenshotCapture

    logger.info(f"Testing screenshot capture to {output_path}...")

    client = MCPClient(host=host, port=port)
    capture = ScreenshotCapture(client)

    result = capture.capture(output_path)

    if result["success"]:
        logger.info(f"Screenshot saved to: {result['path']}")
    else:
        logger.error(f"Screenshot capture failed: {result.get('error')}")

    client.disconnect()
    return result.get("success", False)


def test_vlm_evaluator(screenshot_path: str, document_path: str) -> bool:
    """Test VLM evaluator."""
    from src.isaac_sim.vlm_evaluator import create_evaluator, load_task_document, MockVLMEvaluator

    logger.info(f"Testing VLM evaluator...")

    evaluator = create_evaluator(backend="auto")
    if isinstance(evaluator, MockVLMEvaluator):
        logger.warning("No VLM backend available (set AZURE_OPENAI_API_KEY or ANTHROPIC_API_KEY or GOOGLE_API_KEY)")
        return False

    logger.info(f"Using backend: {type(evaluator).__name__}")
    logger.info(f"Screenshot: {screenshot_path}")
    logger.info(f"Document: {document_path}")

    document = load_task_document(document_path)

    result = evaluator.evaluate(screenshot_path, document)

    if result.get("success"):
        logger.info(f"Evaluation score: {result['score']}/100")
        logger.info(f"Breakdown: {result.get('breakdown', {})}")
    else:
        logger.error(f"Evaluation failed: {result.get('error')}")

    return result.get("success", False)


def main():
    parser = argparse.ArgumentParser(description="Test pipeline components")
    subparsers = parser.add_subparsers(dest="command", help="Component to test")

    # Connection test
    conn_parser = subparsers.add_parser("connection", help="Test MCP connection")
    conn_parser.add_argument("--host", default="localhost")
    conn_parser.add_argument("--port", type=int, default=8766)

    # Scene builder test
    scene_parser = subparsers.add_parser("scene", help="Test scene builder")
    scene_parser.add_argument("document", help="Path to task document")
    scene_parser.add_argument("--host", default="localhost")
    scene_parser.add_argument("--port", type=int, default=8766)

    # Screenshot test
    screenshot_parser = subparsers.add_parser("screenshot", help="Test screenshot capture")
    screenshot_parser.add_argument("output", help="Output path for screenshot")
    screenshot_parser.add_argument("--host", default="localhost")
    screenshot_parser.add_argument("--port", type=int, default=8766)

    # VLM test
    vlm_parser = subparsers.add_parser("vlm", help="Test VLM evaluator")
    vlm_parser.add_argument("screenshot", help="Path to screenshot")
    vlm_parser.add_argument("document", help="Path to task document")

    # Full test
    full_parser = subparsers.add_parser("full", help="Run full component test")
    full_parser.add_argument("--host", default="localhost")
    full_parser.add_argument("--port", type=int, default=8766)
    full_parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM test")
    full_parser.add_argument("--document", default=None, help="Task document to build (default: franka/stack)")

    args = parser.parse_args()

    if args.command == "connection":
        success = test_mcp_connection(args.host, args.port)
    elif args.command == "scene":
        success = test_scene_builder(args.document, args.host, args.port)
    elif args.command == "screenshot":
        success = test_screenshot_capture(args.output, args.host, args.port)
    elif args.command == "vlm":
        success = test_vlm_evaluator(args.screenshot, args.document)
    elif args.command == "full":
        logger.info("=" * 50)
        logger.info("Running full component test")
        logger.info("=" * 50)

        # 1. Test connection
        if not test_mcp_connection(args.host, args.port):
            logger.error("Connection test failed, aborting")
            return 1

        # 2. Use existing task document
        base_dir = Path(__file__).resolve().parent.parent
        if args.document:
            test_doc = Path(args.document)
        else:
            test_doc = base_dir / "tasks" / "franka" / "stack" / "franka_stack.yaml"

        if not test_doc.exists():
            logger.error(f"Task document not found: {test_doc}")
            return 1
        logger.info(f"Using task document: {test_doc}")

        # 3. Build scene
        if not test_scene_builder(str(test_doc), args.host, args.port):
            logger.error("Scene builder test failed")
            return 1

        # 4. Capture screenshot
        import time
        time.sleep(2)  # Wait for scene to render
        test_screenshot = base_dir / "outputs" / "test_screenshot.png"
        test_screenshot.parent.mkdir(parents=True, exist_ok=True)
        if not test_screenshot_capture(str(test_screenshot), args.host, args.port):
            logger.warning("Screenshot capture test failed (may need different capture method)")

        # 5. VLM test (optional)
        if not args.skip_vlm and test_screenshot.exists():
            if not test_vlm_evaluator(str(test_screenshot), str(test_doc)):
                logger.warning("VLM test failed (check ANTHROPIC_API_KEY)")

        logger.info("=" * 50)
        logger.info("Full component test complete!")
        logger.info("=" * 50)
        success = True
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
