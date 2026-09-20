"""ARSI HTTP Interface — bidirectional API for agent integration.

Endpoints:
  POST /arsi/brief   — agent asks for guidance before task
  POST /arsi/report  — agent reports result after task
  POST /arsi/ma/register  — register host agent for multi-agent protocol
  POST /arsi/ma/dispatch  — ARSI assigns task + structured brief
  POST /arsi/ma/result    — host posts task result into orchestrator
  POST /arsi/ma/cycle     — one orchestration pass
  GET  /arsi/ma/health    — multi-agent registry health
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
            ma = None
            if getattr(arsi, "multi_agent", None) is not None:
                ma = arsi.multi_agent.health().get("multi_agent")
            self._respond(200, {
                "status": "ok",
                "llm": arsi.llm.available if arsi.llm else False,
                "model": getattr(getattr(arsi, "llm", None), "config", None) and arsi.llm.config.model,
                "multi_agent": ma,
            })
        elif self.path == "/arsi/ma/health":
            if getattr(arsi, "multi_agent", None) is None:
                self._respond(503, {"error": "multi_agent_not_attached"})
                return
            self._respond(200, arsi.multi_agent.health())
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
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid_json"})
            return

        if self.path == "/arsi/brief":
            self._handle_brief(data)
        elif self.path == "/arsi/report":
            self._handle_report(data)
        elif self.path == "/arsi/ma/register":
            self._handle_ma_register(data)
        elif self.path == "/arsi/ma/dispatch":
            self._handle_ma_dispatch(data)
        elif self.path == "/arsi/ma/result":
            self._handle_ma_result(data)
        elif self.path == "/arsi/ma/cycle":
            self._handle_ma_cycle(data)
        else:
            self._respond(404, {"error": "not_found"})

    def _ma(self):
        return getattr(arsi, "multi_agent", None)

    def _handle_ma_register(self, data: dict):
        orch = self._ma()
        if orch is None:
            self._respond(503, {"error": "multi_agent_not_attached"})
            return
        agent_id = data.get("agent_id")
        if not agent_id:
            self._respond(400, {"error": "agent_id_required"})
            return
        rec = orch.register(
            agent_id,
            role=data.get("role", "worker"),
            capabilities=data.get("capabilities", []),
        )
        self._respond(200, rec.to_dict())

    def _handle_ma_dispatch(self, data: dict):
        orch = self._ma()
        if orch is None:
            self._respond(503, {"error": "multi_agent_not_attached"})
            return
        agent_id = data.get("agent_id")
        task = data.get("task") or data.get("task_description") or ""
        if not agent_id or not task:
            self._respond(400, {"error": "agent_id_and_task_required"})
            return
        # ensure brief path uses live ARSIInterface
        if orch.interface is None:
            orch.interface = interface
        res = orch.dispatch(agent_id, task, payload=data.get("payload") or {})
        self._respond(200 if res.get("dispatched") else 400, res)

    def _handle_ma_result(self, data: dict):
        orch = self._ma()
        if orch is None:
            self._respond(503, {"error": "multi_agent_not_attached"})
            return
        required = ["agent_id", "task_id", "task", "outcome"]
        for field in required:
            if field not in data:
                self._respond(400, {"error": f"{field}_required"})
                return
        if orch.interface is None:
            orch.interface = interface
        res = orch.submit_result(
            agent_id=data["agent_id"],
            task_id=data["task_id"],
            task_description=data["task"],
            outcome=data["outcome"],
            effect=float(data.get("effect", 0.0) or 0.0),
            skills_used=data.get("skills_used", []),
            notes=data.get("notes", ""),
            recommendations_followed=data.get("recommendations_followed", []),
            recommendations_ignored=data.get("recommendations_ignored", []),
        )
        self._respond(200 if res.get("accepted") else 400, res)

    def _handle_ma_cycle(self, data: dict):
        orch = self._ma()
        if orch is None:
            self._respond(503, {"error": "multi_agent_not_attached"})
            return
        if orch.interface is None:
            orch.interface = interface
        self._respond(200, orch.cycle())

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
    # Wire multi-agent orchestrator to live brief/report interface
    if getattr(arsi, "multi_agent", None) is not None:
        arsi.multi_agent.interface = interface
        arsi.multi_agent.arsi = arsi
        logger.info("Multi-agent orchestrator attached to ARSIInterface")

    server = HTTPServer(("0.0.0.0", args.port), ARSIHandler)
    logger.info(f"ARSI Interface listening on port {args.port}")
    logger.info(f"Endpoints:")
    logger.info(f"  POST http://localhost:{args.port}/arsi/brief")
    logger.info(f"  POST http://localhost:{args.port}/arsi/report")
    logger.info(f"  POST http://localhost:{args.port}/arsi/ma/register")
    logger.info(f"  POST http://localhost:{args.port}/arsi/ma/dispatch")
    logger.info(f"  POST http://localhost:{args.port}/arsi/ma/result")
    logger.info(f"  GET  http://localhost:{args.port}/arsi/ma/health")
    logger.info(f"  GET  http://localhost:{args.port}/arsi/health")
    server.serve_forever()


if __name__ == "__main__":
    main()
