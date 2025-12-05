"""
PCI Agent HTTP Server

HTTP server for the PCI Agent API that coordinates verification requests
between business apps and user apps.

Flow:
1. User initiates a service request (e.g., "I want to buy alcohol")
2. Business receives the request and responds with ZKP verification requirements
3. User sees the verification request and approves/denies
4. Business receives the proof and completes the transaction

Services:
- PCI ZKP Service (Midnight): http://localhost:8084
- Cardano Devnet (Yaci): http://localhost:8080
"""

import json
import os
import uuid
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

# Service configuration
ZKP_SERVICE_URL = os.environ.get("ZKP_SERVICE_URL", "http://localhost:8084")
CARDANO_API_URL = os.environ.get("CARDANO_API_URL", "http://localhost:8080")

# In-memory storage for demo (would be persistent in production)
SERVICE_REQUESTS: dict[str, dict[str, Any]] = {}  # User -> Business requests
VERIFICATION_REQUESTS: dict[str, dict[str, Any]] = {}  # Business -> User requests


def call_zkp_service(proof_type: str, data: dict) -> dict[str, Any]:
    """Call the PCI ZKP service to generate a real ZK proof via Midnight."""
    url = f"{ZKP_SERVICE_URL}/proofs/{proof_type}"
    try:
        req = Request(
            url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as e:
        print(f"[Agent] ZKP service error: {e}")
        return {"error": str(e), "fallback": True}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from ZKP service", "fallback": True}


def query_cardano_api(endpoint: str) -> dict[str, Any]:
    """Query the Cardano devnet API (Yaci Store)."""
    url = f"{CARDANO_API_URL}{endpoint}"
    try:
        with urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as e:
        print(f"[Agent] Cardano API error: {e}")
        return {"error": str(e)}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON from Cardano API"}


class AgentHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the PCI Agent API."""

    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"status": "healthy", "service": "pci-agent"})

        elif path == "/services":
            # Check status of connected services
            services = self._check_services_status()
            self._send_json(services)

        elif path == "/cardano/tip":
            # Get Cardano chain tip from devnet
            tip = query_cardano_api("/api/v1/blocks/latest")
            self._send_json(tip)

        elif path == "/":
            self._send_json(
                {
                    "service": "pci-agent",
                    "version": "0.1.0",
                    "description": "Coordinates verification flow between users and businesses",
                    "services": {
                        "zkp": ZKP_SERVICE_URL,
                        "cardano": CARDANO_API_URL,
                    },
                    "endpoints": [
                        "GET /health",
                        "GET /services - check connected service status",
                        "GET /cardano/tip - get Cardano chain tip",
                        "",
                        "# Service Requests (User -> Business)",
                        "GET /service-requests - list service requests for business",
                        "POST /service-requests - user initiates a service request",
                        "",
                        "# Verification Requests (Business -> User)",
                        "GET /requests - list verification requests for user",
                        "GET /requests/:id - get single request",
                        "POST /requests - business creates verification request",
                        "POST /requests/:id/approve - user approves request (generates ZK proof)",
                        "POST /requests/:id/deny - user denies request",
                    ],
                }
            )

        elif path == "/service-requests":
            # Business polls for incoming service requests
            status_filter = query.get("status", [None])[0]
            requests = list(SERVICE_REQUESTS.values())
            if status_filter:
                requests = [r for r in requests if r["status"] == status_filter]
            self._send_json({"requests": requests})

        elif path == "/requests":
            # User polls for verification requests
            status_filter = query.get("status", [None])[0]
            requests = list(VERIFICATION_REQUESTS.values())
            if status_filter:
                requests = [r for r in requests if r["status"] == status_filter]
            self._send_json({"requests": requests})

        elif path.startswith("/requests/") and path.count("/") == 2:
            request_id = path.split("/")[2]
            if request_id in VERIFICATION_REQUESTS:
                self._send_json(VERIFICATION_REQUESTS[request_id])
            else:
                self._send_error(404, "Request not found")

        elif path.startswith("/service-requests/") and path.count("/") == 2:
            request_id = path.split("/")[2]
            if request_id in SERVICE_REQUESTS:
                self._send_json(SERVICE_REQUESTS[request_id])
            else:
                self._send_error(404, "Service request not found")

        else:
            self._send_error(404, "Not found")

    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/service-requests":
            # User initiates a service request
            self._create_service_request()

        elif path == "/requests":
            # Business creates a verification request (in response to service request)
            self._create_verification_request()

        elif path.endswith("/approve"):
            parts = path.split("/")
            if "requests" in parts:
                request_id = parts[2]
                self._approve_request(request_id)
            else:
                self._send_error(404, "Not found")

        elif path.endswith("/deny"):
            parts = path.split("/")
            if "requests" in parts:
                request_id = parts[2]
                self._deny_request(request_id)
            else:
                self._send_error(404, "Not found")

        elif path.endswith("/complete"):
            # Business marks service request as complete
            request_id = path.split("/")[2]
            self._complete_service_request(request_id)

        else:
            self._send_error(404, "Not found")

    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _create_service_request(self) -> None:
        """User initiates a service request to a business."""
        body = self._read_body()
        if body is None:
            return

        request_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow()

        request = {
            "id": request_id,
            "userId": body.get("userId", "demo-user"),
            "userName": body.get("userName", "Demo User"),
            "businessId": body.get("businessId", "demo-business"),
            "serviceType": body.get("serviceType", "purchase"),
            "serviceName": body.get("serviceName", "Age-restricted product"),
            "status": "pending",  # pending -> verification_required -> completed/denied
            "createdAt": now.isoformat() + "Z",
            "expiresAt": (now + timedelta(minutes=10)).isoformat() + "Z",
            "verificationRequestId": None,  # Will be set when business responds
        }

        SERVICE_REQUESTS[request_id] = request
        user_name = request["userName"]
        service_name = request["serviceName"]
        print(f"[Agent] User {user_name} initiated service request {request_id}: {service_name}")
        self._send_json(request, 201)

    def _create_verification_request(self) -> None:
        """Business creates a verification request (typically in response to a service request)."""
        body = self._read_body()
        if body is None:
            return

        request_id = str(uuid.uuid4())[:8]
        now = datetime.utcnow()

        request = {
            "id": request_id,
            "type": body.get("type", "age"),
            "businessId": body.get("businessId", "unknown"),
            "businessName": body.get("businessName", "Unknown Business"),
            "claim": body.get("claim", {"type": "age", "minAge": 18}),
            "status": "pending",
            "serviceRequestId": body.get("serviceRequestId"),  # Links to original service request
            "createdAt": now.isoformat() + "Z",
            "expiresAt": (now + timedelta(minutes=5)).isoformat() + "Z",
            "response": None,
        }

        VERIFICATION_REQUESTS[request_id] = request

        # Update the service request if linked
        service_request_id = body.get("serviceRequestId")
        if service_request_id and service_request_id in SERVICE_REQUESTS:
            SERVICE_REQUESTS[service_request_id]["status"] = "verification_required"
            SERVICE_REQUESTS[service_request_id]["verificationRequestId"] = request_id

        biz_name = request["businessName"]
        print(f"[Agent] Business {biz_name} created verification request {request_id}")
        self._send_json(request, 201)

    def _approve_request(self, request_id: str) -> None:
        """User approves a verification request."""
        if request_id not in VERIFICATION_REQUESTS:
            self._send_error(404, "Request not found")
            return

        request = VERIFICATION_REQUESTS[request_id]
        if request["status"] != "pending":
            self._send_error(400, f"Request already {request['status']}")
            return

        # Read user's private data from request body (agent doesn't interpret this)
        user_data = self._read_body() or {}

        # Get the claim requirements from the verification request
        claim = request["claim"]
        proof_type = claim.get("type", "unknown")

        # Build proof request - merge claim requirements with user's private data
        # Agent is generic: just passes data through to ZKP service
        proof_data = {
            **claim,  # Business requirements (e.g., minAge)
            **user_data,  # User's private data (e.g., birthDate)
            "requestId": request_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

        # Call ZKP service - all verification logic lives there, not here
        print(f"[Agent] Calling ZKP service at {ZKP_SERVICE_URL}/proofs/{proof_type}")
        zkp_result = call_zkp_service(proof_type, proof_data)

        if zkp_result.get("error"):
            # ZKP service unavailable - cannot generate proof
            print(f"[Agent] ZKP service error: {zkp_result.get('error')}")
            proof_response = {
                "verified": False,
                "error": "ZKP service unavailable",
                "source": "error",
            }
        else:
            # Use proof from ZKP service
            # The verified status is inside publicSignals
            public_signals = zkp_result.get("proof", {}).get("publicSignals", {})
            proof_response = {
                "verified": public_signals.get("verified", False),
                "publicSignals": public_signals,
                "proof": zkp_result.get("proof", {}),
                "source": "zkp",
            }
            print(f"[Agent] ZK proof generated, verified={proof_response['verified']}")

        is_verified = proof_response.get("verified", False)
        # "approved" = proof passed, "rejected" = proof failed (user didn't meet criteria)
        request["status"] = "approved" if is_verified else "rejected"
        request["response"] = proof_response
        request["respondedAt"] = datetime.utcnow().isoformat() + "Z"

        # Update linked service request based on verification result
        service_request_id = request.get("serviceRequestId")
        if service_request_id and service_request_id in SERVICE_REQUESTS:
            new_status = "verified" if is_verified else "rejected"
            SERVICE_REQUESTS[service_request_id]["status"] = new_status

        status_str = "APPROVED" if is_verified else "REJECTED (verification failed)"
        print(f"[Agent] Request {request_id} {status_str}")
        self._send_json(request)

    def _deny_request(self, request_id: str) -> None:
        """User denies a verification request."""
        if request_id not in VERIFICATION_REQUESTS:
            self._send_error(404, "Request not found")
            return

        request = VERIFICATION_REQUESTS[request_id]
        if request["status"] != "pending":
            self._send_error(400, f"Request already {request['status']}")
            return

        request["status"] = "denied"
        request["response"] = {"verified": False, "reason": "User denied request"}
        request["respondedAt"] = datetime.utcnow().isoformat() + "Z"

        # Update linked service request
        service_request_id = request.get("serviceRequestId")
        if service_request_id and service_request_id in SERVICE_REQUESTS:
            SERVICE_REQUESTS[service_request_id]["status"] = "denied"

        print(f"[Agent] Request {request_id} DENIED")
        self._send_json(request)

    def _complete_service_request(self, request_id: str) -> None:
        """Business marks a service request as complete."""
        if request_id not in SERVICE_REQUESTS:
            self._send_error(404, "Service request not found")
            return

        request = SERVICE_REQUESTS[request_id]
        if request["status"] != "verified":
            self._send_error(400, f"Service request not verified (status: {request['status']})")
            return

        request["status"] = "completed"
        request["completedAt"] = datetime.utcnow().isoformat() + "Z"

        print(f"[Agent] Service request {request_id} COMPLETED")
        self._send_json(request)

    def _check_services_status(self) -> dict[str, Any]:
        """Check status of connected services (ZKP and Cardano)."""
        agent_port = os.environ.get("PORT", "8082")
        services = {
            "agent": {"status": "healthy", "url": f"http://localhost:{agent_port}"},
            "zkp": {"status": "unknown", "url": ZKP_SERVICE_URL},
            "cardano": {"status": "unknown", "url": CARDANO_API_URL},
        }

        # Check ZKP service
        try:
            with urlopen(f"{ZKP_SERVICE_URL}/health", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                services["zkp"]["status"] = data.get("status", "healthy")
        except Exception as e:
            services["zkp"]["status"] = "unavailable"
            services["zkp"]["error"] = str(e)

        # Check Cardano devnet
        try:
            with urlopen(f"{CARDANO_API_URL}/api/v1/blocks/latest", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
                services["cardano"]["status"] = "healthy"
                services["cardano"]["latestBlock"] = data.get("number") or data.get("height")
        except Exception as e:
            services["cardano"]["status"] = "unavailable"
            services["cardano"]["error"] = str(e)

        return services

    def _read_body(self) -> dict[str, Any] | None:
        """Read and parse JSON body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            body = self.rfile.read(content_length).decode("utf-8")
            return json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return None

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_cors_headers(self) -> None:
        """Add CORS headers for browser access."""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

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
    print(f"API docs: http://localhost:{port}/")
    print("")
    print("Connected Services:")
    print(f"  ZKP (Midnight): {ZKP_SERVICE_URL}")
    print(f"  Cardano Devnet: {CARDANO_API_URL}")
    print(f"  Service status: http://localhost:{port}/services")
    print("")
    print("Flow:")
    print("  1. User sends service request -> POST /service-requests")
    print("  2. Business polls for requests -> GET /service-requests")
    print("  3. Business creates verification request -> POST /requests")
    print("  4. User polls for verification requests -> GET /requests")
    print("  5. User approves/denies -> POST /requests/:id/approve or /deny")
    print("     (Approval triggers ZK proof generation via Midnight)")
    print("  6. Business polls for response and completes -> POST /service-requests/:id/complete")
    server.serve_forever()


if __name__ == "__main__":
    main()
