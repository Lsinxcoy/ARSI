"""ARSI HTTP Interface — bidirectional API for agent integration.

Endpoints:
  POST /arsi/brief   — agent asks for guidance before task
  POST /arsi/report  — agent reports result after task
  GET  /arsi/stats   — interface statistics
  GET  /arsi/health  — health check

Usage:
    python scripts/arsi_interface.py --port 9300
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.adapters.bidirectional_interface import ARSIInterface, ARSIReport

logger = logging.getLogger("arsi.interface")

# Global instances
arsi: ARSI = None
interface: ARSIInterface = None


class ARSIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ARSI interface."""

    def do_GET(self):
        if self.path == "/arsi/health":
            self._respond(200, {"status": "ok", "llm": arsi.llm.available if arsi.llm else False})
        elif self.path == "/arsi/stats":
            self._respond(200, {
                "interface": interface.stats,
                "arsi": arsi.get_stats(),
            })
        else:
            self._respond(404, {"error": "not_found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
            return

        if self.path == "/arsi/brief":
            self._handle_brief(data)
        elif self.path == "/arsi/report":
            self._handle_report(data)
        else:
            self._respond(404, {"error": "not_found"})

    def _handle_brief(self, data: dict):
        """Handle brief request from agent."""
        task = data.get("task", "")
        agent_id = data.get("agent_id", "unknown")

        if not task:
            self._respond(400, {"error": "task_required"})
            return

        brief = interface.brief(task, agent_id)
        self._respond(200, brief.to_dict())

    def _handle_report(self, data: dict):
        """Handle report from agent."""
        required = ["agent_id", "task", "outcome"]
        for field in required:
            if field not in data:
                self._respond(400, {"error": f"{field}_required"})
                return

        report = ARSIReport(
            brief_id=data.get("brief_id", ""),
            agent_id=data["agent_id"],
            task_description=data["task"],
            outcome=data["outcome"],
            effect=data.get("effect", 0.5),
            skills_used=data.get("skills_used", []),
            recommendations_followed=data.get("recommendations_followed", []),
            recommendations_ignored=data.get("recommendations_ignored", []),
            notes=data.get("notes", ""),
        )

        result = interface.report(report)
        self._respond(200, result)

    def _respond(self, code: int, data: dict):
        """Send JSON response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False, default=str).encode())

    def log_message(self, format, *args):
        logger.info(f"{self.address_string()} {format % args}")


def main():
    global arsi, interface

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ARSI HTTP Interface")
    parser.add_argument("--port", type=int, default=9300)
    parser.add_argument("--config", default="config/arsi.yaml")
    args = parser.parse_args()

    logger.info("Building ARSI...")
    arsi = ARSI.from_config(args.config)
    interface = ARSIInterface(arsi)

    server = HTTPServer(("0.0.0.0", args.port), ARSIHandler)
    logger.info(f"ARSI Interface listening on port {args.port}")
    logger.info(f"Endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/arsi/brief")
    logger.info(f"  POST http://localhost:{args.port}/arsi/report")
    logger.info(f"  GET  http://localhost:{args.port}/arsi/stats")
    logger.info(f"  GET  http://localhost:{args.port}/arsi/health")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        arsi.close()


if __name__ == "__main__":
    main()
