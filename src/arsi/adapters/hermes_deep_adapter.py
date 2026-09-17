"""Hermes Deep Adapter — extracts from the main Hermes state database.

Reads from: C:/Users/<user>/AppData/Local/hermes/state.db
Contains: 3400+ sessions, 750K+ messages, token/cost tracking
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class HermesDeepAdapter:
    """Deep extraction from Hermes main state database."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = Path(db_path)
        else:
            # Auto-detect
            for env_var in ["USERPROFILE", "HOME"]:
                base = os.environ.get(env_var, "")
                if base and "Users" in base and "SYSTEM" not in base:
                    candidate = Path(base) / "AppData" / "Local" / "hermes" / "state.db"
                    if candidate.exists():
                        self.db_path = candidate
                        break
            else:
                self.db_path = Path("C:/Users/41228/AppData/Local/hermes/state.db")

        self._extracted_count = 0

    def extract_traces(self, limit_sessions: int = 100, limit_messages: int = 500) -> list[dict]:
        """Extract behavior traces from Hermes state database."""
        traces = []

        if not self.db_path.exists():
            logger.warning(f"Hermes state.db not found: {self.db_path}")
            return traces

        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row

            traces.extend(self._extract_sessions(conn, limit_sessions))
            traces.extend(self._extract_tool_calls(conn, limit_messages))
            traces.extend(self._extract_model_usage(conn))
            traces.extend(self._extract_delivery(conn))

            conn.close()
        except Exception as e:
            logger.error(f"Failed to read Hermes state.db: {e}")

        self._extracted_count += len(traces)
        return traces

    def _extract_sessions(self, conn, limit: int) -> list[dict]:
        """Extract session-level traces."""
        traces = []
        cursor = conn.execute(
            """SELECT id, title, model, message_count, tool_call_count,
                      input_tokens, output_tokens, estimated_cost_usd,
                      started_at, ended_at, end_reason, cwd, tool_names
               FROM sessions
               WHERE archived = 0 AND hidden = 0
               ORDER BY started_at DESC
               LIMIT ?""",
            (limit,),
        )

        for row in cursor.fetchall():
            # Classify session outcome
            msg_count = row["message_count"] or 0
            tool_count = row["tool_call_count"] or 0
            end_reason = row["end_reason"] or ""

            if end_reason == "error":
                outcome = "failure"
                effect = 0.2
            elif msg_count > 10 and tool_count > 5:
                outcome = "success"
                effect = 0.8
            elif msg_count > 0:
                outcome = "partial"
                effect = 0.5
            else:
                outcome = "recorded"
                effect = 0.3

            traces.append({
                "action": f"hermes_session:{(row['title'] or 'untitled')[:50]}",
                "outcome": outcome,
                "effect": effect,
                "params": {
                    "source": "hermes_state_db",
                    "session_id": row["id"],
                    "model": row["model"],
                    "message_count": msg_count,
                    "tool_call_count": tool_count,
                    "input_tokens": row["input_tokens"] or 0,
                    "output_tokens": row["output_tokens"] or 0,
                    "cost_usd": row["estimated_cost_usd"] or 0,
                    "end_reason": end_reason,
                    "tool_names": row["tool_names"] or "",
                    "cwd": row["cwd"] or "",
                },
            })

        return traces

    def _extract_tool_calls(self, conn, limit: int) -> list[dict]:
        """Extract tool call traces from messages."""
        traces = []
        cursor = conn.execute(
            """SELECT session_id, tool_name, content, timestamp, token_count
               FROM messages
               WHERE role = 'tool' AND tool_name IS NOT NULL
               ORDER BY timestamp DESC
               LIMIT ?""",
            (limit,),
        )

        for row in cursor.fetchall():
            tool_name = row["tool_name"]
            content = row["content"] or ""

            # Classify tool call outcome
            content_lower = content.lower()
            if any(w in content_lower for w in ["error", "failed", "exception", "traceback"]):
                outcome = "failure"
                effect = 0.2
            elif any(w in content_lower for w in ["success", "complete", "done", "pass"]):
                outcome = "success"
                effect = 0.7
            else:
                outcome = "recorded"
                effect = 0.5

            traces.append({
                "action": f"hermes_tool:{tool_name}",
                "outcome": outcome,
                "effect": effect,
                "params": {
                    "source": "hermes_tool_call",
                    "tool_name": tool_name,
                    "session_id": row["session_id"],
                    "content_length": len(content),
                    "token_count": row["token_count"] or 0,
                },
            })

        return traces

    def _extract_model_usage(self, conn) -> list[dict]:
        """Extract model usage traces."""
        traces = []
        cursor = conn.execute(
            """SELECT model, task, api_call_count, input_tokens, output_tokens,
                      estimated_cost_usd, first_seen, last_seen
               FROM session_model_usage
               ORDER BY last_seen DESC
               LIMIT 50""",
        )

        for row in cursor.fetchall():
            traces.append({
                "action": f"hermes_model:{row['model']}",
                "outcome": "recorded",
                "effect": 0.5,
                "params": {
                    "source": "hermes_model_usage",
                    "model": row["model"],
                    "task": row["task"],
                    "api_calls": row["api_call_count"] or 0,
                    "input_tokens": row["input_tokens"] or 0,
                    "output_tokens": row["output_tokens"] or 0,
                    "cost_usd": row["estimated_cost_usd"] or 0,
                },
            })

        return traces

    def _extract_delivery(self, conn) -> list[dict]:
        """Extract delivery obligation traces."""
        traces = []
        cursor = conn.execute(
            """SELECT platform, state, attempts, content, created_at
               FROM delivery_obligations
               ORDER BY created_at DESC
               LIMIT 30""",
        )

        for row in cursor.fetchall():
            state = row["state"] or "unknown"
            traces.append({
                "action": f"hermes_delivery:{row['platform']}",
                "outcome": "success" if state == "delivered" else "partial" if state == "pending" else "failure",
                "effect": 0.7 if state == "delivered" else 0.4,
                "params": {
                    "source": "hermes_delivery",
                    "platform": row["platform"],
                    "state": state,
                    "attempts": row["attempts"] or 0,
                },
            })

        return traces

    def get_stats(self) -> dict:
        """Get adapter statistics."""
        stats = {
            "db_path": str(self.db_path),
            "db_exists": self.db_path.exists(),
            "extracted_count": self._extracted_count,
        }

        if self.db_path.exists():
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.execute("SELECT COUNT(*) FROM sessions WHERE archived = 0")
                stats["active_sessions"] = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
                stats["total_messages"] = cursor.fetchone()[0]
                cursor = conn.execute("SELECT COUNT(DISTINCT tool_name) FROM messages WHERE tool_name IS NOT NULL")
                stats["unique_tools"] = cursor.fetchone()[0]
                conn.close()
            except Exception:
                pass

        return stats
