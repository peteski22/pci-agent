"""
PCI Agent HTTP Server

Simple HTTP server for the PCI Agent API.
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler


class AgentHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the PCI Agent API."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/health":
            self._send_json({"status": "healthy", "service": "pci-agent"})
        elif self.path == "/":
            self._send_json({
                "service": "pci-agent",
                "version": "0.1.0",
                "endpoints": ["/health", "/process"],
            })
        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/process":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")

            try:
                request = json.loads(body) if body else {}
                # Placeholder response - in production this would use the Agent
                response = {
                    "status": "success",
                    "message": "Agent processing is not yet implemented",
                    "request_received": request,
                }
                self._send_json(response)
            except json.JSONDecodeError:
                self._send_error(400, "Invalid JSON")
        else:
            self._send_error(404, "Not found")

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(self, status: int, message: str) -> None:
        """Send an error response."""
        self._send_json({"error": message}, status)

    def log_message(self, format: str, *args) -> None:
        """Log HTTP requests."""
        print(f"[Agent] {args[0]} {args[1]} {args[2]}")


def main() -> None:
    """Start the PCI Agent HTTP server."""
    port = int(os.environ.get("PORT", "8082"))
    server = HTTPServer(("0.0.0.0", port), AgentHandler)
    print(f"PCI Agent starting on port {port}")
    print(f"Health check: http://localhost:{port}/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
