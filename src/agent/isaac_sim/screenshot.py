#!/usr/bin/env python3
"""
Screenshot Capture - Captures viewport screenshots from Isaac Sim via MCP

This module captures screenshots of the current Isaac Sim viewport for
VLM evaluation.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.agent.common.mcp_client import MCPClient

logger = logging.getLogger(__name__)


class ScreenshotCapture:
    """Captures screenshots from Isaac Sim viewport."""

    def __init__(self, mcp_client: MCPClient = None):
        self.mcp = mcp_client or MCPClient()

    def capture(self, output_path: str | Path, resolution: tuple = (1280, 720),
                camera_eye: list = None, camera_target: list = None) -> dict:
        """
        Capture a screenshot of the current viewport using Replicator.

        Args:
            output_path: Path to save the screenshot
            resolution: (width, height) tuple
            camera_eye: Optional camera position [x, y, z]
            camera_target: Optional camera look-at target [x, y, z]

        Returns:
            dict with 'success', 'path', and optional 'error' keys
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Default camera view if not specified (captures robot + table + cubes)
        # Robot is at origin, table is at (0.5, 0, z), cubes are on table
        if camera_eye is None:
            camera_eye = [1.5, 1.2, 1.0]  # Higher and further back to see robot
        if camera_target is None:
            camera_target = [0.25, 0, 0.3]  # Look between robot and table

        # Use Replicator-based capture which works reliably in streaming mode
        output_dir = str(output_path.parent.absolute())

        code = f'''
import omni.replicator.core as rep
import asyncio

# Set camera view first
from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(
    eye={camera_eye},
    target={camera_target},
    camera_prim_path="/OmniverseKit_Persp"
)

# Create render product from viewport camera
rp = rep.create.render_product("/OmniverseKit_Persp", ({resolution[0]}, {resolution[1]}))

# Create output writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="{output_dir}", rgb=True)
writer.attach([rp])

# Capture frame
async def do_capture():
    await rep.orchestrator.step_async()
    print("SUCCESS: Camera capture complete")

asyncio.ensure_future(do_capture())
'''
        result = self.mcp.execute_script(code)

        # Wait for async capture to complete
        time.sleep(2.0)

        # Check for replicator output (rgb_*.png)
        rgb_files = list(output_path.parent.glob("rgb_*.png"))
        if rgb_files:
            # Get the latest file and rename to expected path
            latest = max(rgb_files, key=lambda p: p.stat().st_mtime)
            if latest != output_path:
                latest.rename(output_path)
            logger.info(f"Screenshot saved to {output_path}")
            return {"success": True, "path": str(output_path)}

        # Check if file was created directly
        if output_path.exists():
            logger.info(f"Screenshot saved to {output_path}")
            return {"success": True, "path": str(output_path)}

        # Try alternative capture method as fallback
        return self._capture_alternative(output_path, resolution)

    def _capture_alternative(self, output_path: Path, resolution: tuple) -> dict:
        """Alternative capture method using different API."""
        abs_path = str(output_path.absolute())

        code = f'''
import omni.renderer_capture
import carb

# Configure capture
capture_settings = {{
    "width": {resolution[0]},
    "height": {resolution[1]},
    "file_path": "{abs_path}",
    "render_product_path": ""
}}

# Try capture
try:
    omni.renderer_capture.capture_next_frame_swapchain(capture_settings["file_path"])
    print("SUCCESS: Alternative capture initiated")
except Exception as e:
    print(f"ERROR: {{e}}")
'''
        result = self.mcp.execute_script(code)

        # Wait for capture
        time.sleep(1.0)

        if output_path.exists():
            logger.info(f"Screenshot saved to {output_path}")
            return {"success": True, "path": str(output_path)}
        else:
            # Try yet another method
            return self._capture_replicator(output_path, resolution)

    def _capture_replicator(self, output_path: Path, resolution: tuple) -> dict:
        """Capture using Replicator API (more reliable)."""
        abs_path = str(output_path.absolute())

        code = f'''
import omni.replicator.core as rep
import asyncio

# Create render product
render_product = rep.create.render_product("/OmniverseKit_Persp", ({resolution[0]}, {resolution[1]}))

# Create writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir="{output_path.parent}", rgb=True)
writer.attach([render_product])

# Capture single frame
async def capture():
    await rep.orchestrator.step_async()
    print("SUCCESS: Replicator capture complete")

asyncio.ensure_future(capture())
'''
        result = self.mcp.execute_script(code)

        # Wait for capture
        time.sleep(2.0)

        # Check for output (replicator may use different naming)
        if output_path.exists():
            logger.info(f"Screenshot saved to {output_path}")
            return {"success": True, "path": str(output_path)}

        # Check for replicator output pattern
        rgb_files = list(output_path.parent.glob("rgb_*.png"))
        if rgb_files:
            # Rename to expected path
            latest = max(rgb_files, key=lambda p: p.stat().st_mtime)
            latest.rename(output_path)
            logger.info(f"Screenshot saved to {output_path}")
            return {"success": True, "path": str(output_path)}

        return {
            "success": False,
            "error": "Failed to capture screenshot with all methods"
        }

    def capture_with_camera(
        self,
        output_path: str | Path,
        camera_prim_path: str = "/World/Camera",
        resolution: tuple = (1280, 720),
        camera_position: list = None,
        camera_target: list = None
    ) -> dict:
        """
        Capture screenshot using viewport camera positioned to see the scene.

        Args:
            output_path: Path to save the screenshot
            camera_prim_path: USD path to camera prim (used for position hints if available)
            resolution: (width, height) tuple
            camera_position: Optional camera position override
            camera_target: Optional camera look-at target override

        Returns:
            dict with 'success', 'path', and optional 'error' keys
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_path.parent.absolute())

        # Default camera view for Franka workspace (captures robot + table + cubes)
        # Robot is at origin, table is at (0.5, 0, z), cubes are on table
        if camera_position is None:
            camera_position = [1.5, 1.2, 1.0]  # Higher and further back to see robot
        if camera_target is None:
            camera_target = [0.25, 0, 0.3]  # Look between robot and table

        code = f'''
import omni.replicator.core as rep
import asyncio

# Set viewport camera position
from omni.isaac.core.utils.viewports import set_camera_view
set_camera_view(
    eye={camera_position},
    target={camera_target},
    camera_prim_path="/OmniverseKit_Persp"
)

# Create render product from viewport camera (not the task camera)
render_product = rep.create.render_product("/OmniverseKit_Persp", ({resolution[0]}, {resolution[1]}))

# Create basic writer
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(
    output_dir="{output_dir}",
    rgb=True
)
writer.attach([render_product])

# Capture
async def capture():
    await rep.orchestrator.step_async()
    print("SUCCESS: Camera capture complete")

asyncio.ensure_future(capture())
'''
        result = self.mcp.execute_script(code)

        # Wait for async capture to complete (needs more time for USD loading)
        time.sleep(3.0)

        # Check for output - look for rgb_*.png files
        rgb_files = list(output_path.parent.glob("rgb_*.png"))
        if rgb_files:
            latest = max(rgb_files, key=lambda p: p.stat().st_mtime)
            try:
                if latest != output_path and latest.exists():
                    # Move the file instead of rename to handle cross-device moves
                    import shutil
                    shutil.move(str(latest), str(output_path))
                logger.info(f"Screenshot saved to {output_path}")
                return {"success": True, "path": str(output_path)}
            except Exception as e:
                logger.warning(f"File move failed: {e}, using original path")
                return {"success": True, "path": str(latest)}

        if output_path.exists():
            return {"success": True, "path": str(output_path)}

        return {
            "success": False,
            "error": f"Failed to capture screenshot"
        }

    def get_image_base64(self, image_path: str | Path) -> Optional[str]:
        """Read an image file and return base64 encoded string."""
        image_path = Path(image_path)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            return None

        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Capture Isaac Sim viewport screenshot")
    parser.add_argument("output", type=str, help="Output path for screenshot")
    parser.add_argument("--host", type=str, default="localhost", help="MCP host")
    parser.add_argument("--port", type=int, default=8766, help="MCP port")
    parser.add_argument("--width", type=int, default=1280, help="Image width")
    parser.add_argument("--height", type=int, default=720, help="Image height")
    parser.add_argument("--camera", type=str, default=None, help="Camera prim path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = MCPClient(host=args.host, port=args.port)
    capture = ScreenshotCapture(client)

    if args.camera:
        result = capture.capture_with_camera(
            args.output,
            camera_prim_path=args.camera,
            resolution=(args.width, args.height)
        )
    else:
        result = capture.capture(args.output, resolution=(args.width, args.height))

    if result["success"]:
        print(f"Screenshot saved to: {result['path']}")
    else:
        print(f"Failed: {result.get('error', 'unknown error')}")

    client.disconnect()


if __name__ == "__main__":
    main()
