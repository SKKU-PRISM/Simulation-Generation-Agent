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

    @staticmethod
    def _parse_json_payload(payload: str) -> dict:
        """Parse a JSON payload, allowing optional non-JSON prefixes."""
        payload = payload.strip()
        if not payload:
            raise json.JSONDecodeError("Empty payload", payload, 0)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            start = payload.find("{")
            end = payload.rfind("}")
            if start >= 0 and end > start:
                return json.loads(payload[start:end + 1])
            raise

    def _receive_response(self) -> dict:
        """Receive one MCP JSON response (newline-delimited preferred)."""
        buffer = ""
        while True:
            chunk = self.socket.recv(8192)
            if not chunk:
                raise ConnectionError("MCP server closed the socket")

            buffer += chunk.decode("utf-8", errors="replace")

            # Primary mode: newline-delimited JSON responses.
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    return self._parse_json_payload(line)
                except json.JSONDecodeError:
                    continue

            # Fallback: some servers may send a single JSON blob without newline.
            stripped = buffer.strip()
            if stripped:
                try:
                    return self._parse_json_payload(stripped)
                except json.JSONDecodeError:
                    continue

    def send_command(self, command: dict) -> dict:
        """Send a command and receive response."""
        command_type = command.get("type", "unknown")

        for attempt in range(2):
            if not self.socket and not self.connect():
                if attempt == 0:
                    continue
                return {"error": f"MCP command '{command_type}' failed: not connected"}

            try:
                message = json.dumps(command) + "\n"
                self.socket.sendall(message.encode("utf-8"))
                return self._receive_response()
            except (BrokenPipeError, ConnectionError, ConnectionResetError, socket.timeout, OSError) as e:
                logger.warning(
                    "MCP command '%s' failed on attempt %d/2: %s",
                    command_type,
                    attempt + 1,
                    e,
                )
                self.disconnect()
                if attempt == 0:
                    logger.info("Reconnecting and retrying MCP command '%s' once", command_type)
                    continue
                return {"error": f"MCP command '{command_type}' failed: {e}"}
            except Exception as e:
                logger.error("MCP command '%s' failed: %s", command_type, e)
                return {"error": f"MCP command '{command_type}' failed: {e}"}

        return {"error": f"MCP command '{command_type}' failed: retries exhausted"}

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
