"""MiMo Session Trace Extractor — reads MiMo Desktop session memory.

Extracts behavior traces from MiMo Desktop's session memory files.
These traces represent what the agent actually did (tool calls, decisions, outcomes).

Memory location: C:/Users/<user>/.local/share/mimocode/memory/sessions/<session_id>/
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MiMoSessionExtractor:
    """Extracts behavior traces from MiMo Desktop session memory."""

    def __init__(self, memory_base: Optional[str] = None):
        self.memory_base = Path(memory_base or self._default_memory_path())
        self._extracted_count = 0

    @staticmethod
    def _default_memory_path() -> str:
        """Default MiMo Desktop memory path."""
        # Try multiple user detection methods
        for env_var in ["USERPROFILE", "HOME", "HOMEPATH"]:
            path = os.environ.get(env_var, "")
            if path and "Users" in path and "SYSTEM" not in path:
                return str(Path(path) / ".local" / "share" / "mimocode" / "memory" / "sessions")

        # Fallback: scan C:/Users for non-system directories
        users_dir = Path("C:/Users")
        if users_dir.exists():
            for user_dir in users_dir.iterdir():
                if user_dir.is_dir() and user_dir.name not in ("SYSTEM", "Public", "Default", "All Users"):
                    candidate = user_dir / ".local" / "share" / "mimocode" / "memory" / "sessions"
                    if candidate.exists():
                        return str(candidate)

        # Last resort
        user = os.environ.get("USERNAME", "user")
        return f"C:\\Users\\{user}\\.local\\share\\mimocode\\memory\\sessions"

    def list_sessions(self) -> list[dict]:
        """List available MiMo Desktop sessions."""
        sessions = []
        if not self.memory_base.exists():
            return sessions

        for session_dir in self.memory_base.iterdir():
            if not session_dir.is_dir():
                continue
            # Look for memory files
            memory_files = list(session_dir.glob("*.md")) + list(session_dir.glob("*.json"))
            if memory_files:
                sessions.append({
                    "session_id": session_dir.name,
                    "path": str(session_dir),
                    "file_count": len(memory_files),
                    "latest_mtime": max(f.stat().st_mtime for f in memory_files),
                })

        sessions.sort(key=lambda x: x["latest_mtime"], reverse=True)
        return sessions

    def extract_traces(self, session_id: Optional[str] = None) -> list[dict]:
        """Extract behavior traces from a session.

        If session_id is None, uses the most recent session.
        """
        if session_id is None:
            sessions = self.list_sessions()
            if not sessions:
                logger.warning("No MiMo sessions found")
                return []
            session_id = sessions[0]["session_id"]

        session_path = self.memory_base / session_id
        if not session_path.exists():
            logger.warning(f"Session not found: {session_id}")
            return []

        traces = []

        # Extract from memory files
        for mem_file in session_path.glob("*.md"):
            content = mem_file.read_text(encoding="utf-8", errors="ignore")
            traces.extend(self._parse_memory_file(mem_file.name, content))

        for mem_file in session_path.glob("*.json"):
            try:
                content = json.loads(mem_file.read_text(encoding="utf-8", errors="ignore"))
                traces.extend(self._parse_json_file(mem_file.name, content))
            except json.JSONDecodeError:
                continue

        self._extracted_count += len(traces)
        logger.info(f"Extracted {len(traces)} traces from session {session_id}")
        return traces

    def _parse_memory_file(self, filename: str, content: str) -> list[dict]:
        """Parse a markdown memory file into behavior traces."""
        traces = []
        lines = content.split("\n")

        current_action = None
        current_content = []

        for line in lines:
            # Detect action patterns
            if line.startswith("# ") or line.startswith("## "):
                # Save previous action
                if current_action:
                    traces.append(self._make_trace(current_action, "\n".join(current_content), filename))
                current_action = line.strip("# ").strip()
                current_content = []
            elif line.startswith("- ") or line.startswith("* "):
                current_content.append(line.strip("- * "))
            elif line.strip():
                current_content.append(line.strip())

        # Don't forget the last action
        if current_action:
            traces.append(self._make_trace(current_action, "\n".join(current_content), filename))

        return traces

    def _parse_json_file(self, filename: str, data: dict | list) -> list[dict]:
        """Parse a JSON memory file into behavior traces."""
        traces = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    traces.append({
                        "action": item.get("action", item.get("type", "unknown")),
                        "outcome": item.get("outcome", item.get("status", "unknown")),
                        "effect": item.get("effect", item.get("score", 0.5)),
                        "params": item.get("params", item.get("data", {})),
                        "source": filename,
                    })
        elif isinstance(data, dict):
            traces.append({
                "action": data.get("action", "memory_snapshot"),
                "outcome": "recorded",
                "effect": 0.5,
                "params": {"keys": list(data.keys())[:10]},
                "source": filename,
            })

        return traces

    def _make_trace(self, action: str, content: str, source: str) -> dict:
        """Create a behavior trace from parsed content."""
        # Classify action type
        action_lower = action.lower()
        if any(w in action_lower for w in ["create", "write", "build", "implement"]):
            outcome = "success"
            effect = 0.8
        elif any(w in action_lower for w in ["fix", "bug", "error", "fail"]):
            outcome = "failure"
            effect = 0.2
        elif any(w in action_lower for w in ["test", "verify", "check"]):
            outcome = "success" if "pass" in content.lower() else "partial"
            effect = 0.6
        elif any(w in action_lower for w in ["analyze", "research", "study"]):
            outcome = "success"
            effect = 0.7
        else:
            outcome = "success"
            effect = 0.5

        return {
            "action": action[:50],
            "outcome": outcome,
            "effect": effect,
            "params": {
                "content_length": len(content),
                "content_preview": content[:200],
                "source_file": source,
            },
        }

    def extract_from_path(self, path: str) -> list[dict]:
        """Extract traces from an arbitrary path (e.g., task list, progress file)."""
        p = Path(path)
        if not p.exists():
            return []

        traces = []
        if p.is_file():
            content = p.read_text(encoding="utf-8", errors="ignore")
            if p.suffix == ".json":
                try:
                    data = json.loads(content)
                    traces.extend(self._parse_json_file(p.name, data))
                except json.JSONDecodeError:
                    pass
            else:
                traces.extend(self._parse_memory_file(p.name, content))
        elif p.is_dir():
            for f in p.rglob("*.md"):
                content = f.read_text(encoding="utf-8", errors="ignore")
                traces.extend(self._parse_memory_file(f.name, content))

        return traces

    @property
    def stats(self) -> dict:
        return {
            "memory_base": str(self.memory_base),
            "extracted_count": self._extracted_count,
            "sessions_found": len(self.list_sessions()),
        }
