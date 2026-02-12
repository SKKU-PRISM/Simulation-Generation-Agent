"""MCP client for Isaac Sim communication via TCP socket."""

import json
import logging
import socket

logger = logging.getLogger(__name__)


class MCPClient:
    """Simple MCP client that communicates with Isaac Sim extension via socket."""

    def __init__(self, host: str = "localhost", port: int = 8766, timeout: float = 300.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.socket = None

    def connect(self) -> bool:
        """Connect to Isaac Sim MCP extension."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to Isaac Sim at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    def disconnect(self):
        """Disconnect from Isaac Sim."""
        if self.socket:
            self.socket.close()
            self.socket = None

    def send_command(self, command: dict) -> dict:
        """Send a command and receive response."""
        if not self.socket:
            if not self.connect():
                return {"error": "Not connected"}

        try:
            message = json.dumps(command) + "\n"
            self.socket.sendall(message.encode())

            response_data = b""
            while True:
                chunk = self.socket.recv(8192)
                if not chunk:
                    break
                response_data += chunk
                try:
                    response = json.loads(response_data.decode().strip())
                    return response
                except json.JSONDecodeError:
                    continue

            response = json.loads(response_data.decode().strip())
            return response
        except Exception as e:
            logger.error(f"Command failed: {e}")
            return {"error": str(e)}

    def execute_script(self, code: str) -> dict:
        """Execute Python code in Isaac Sim."""
        return self.send_command({
            "type": "execute_script",
            "params": {"code": code}
        })

    def get_scene_info(self) -> dict:
        """Get scene information to verify connection."""
        return self.send_command({
            "type": "get_scene_info",
            "params": {}
        })

    def reset_scene(self, keep_physics: bool = False) -> dict:
        """Reset scene using extension's reset_scene command."""
        return self.send_command({
            "type": "reset_scene",
            "params": {"keep_physics": keep_physics}
        })

    def create_robot(self, robot_type: str, position: list) -> dict:
        """Create robot using extension's create_robot command."""
        return self.send_command({
            "type": "create_robot",
            "params": {"robot_type": robot_type, "position": position}
        })
