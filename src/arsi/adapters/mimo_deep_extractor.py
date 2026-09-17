"""MiMo Deep Session Extractor — extracts rich behavior traces from session memory.

Parses checkpoint.md, notes.md, and tasks/*/progress.md to extract:
- Task completion records (what was done, what succeeded/failed)
- Active intent and directives (what the agent was trying to do)
- Session notes (execution decisions and outcomes)
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


class MiMoDeepExtractor:
    """Deep extraction from MiMo Desktop session memory."""

    def __init__(self, memory_base: Optional[str] = None):
        self.memory_base = Path(memory_base or self._default_memory_path())
        self._extracted_count = 0

    @staticmethod
    def _default_memory_path() -> str:
        for env_var in ["USERPROFILE", "HOME"]:
            path = os.environ.get(env_var, "")
            if path and "Users" in path and "SYSTEM" not in path:
                return str(Path(path) / ".local" / "share" / "mimocode" / "memory" / "sessions")
        users_dir = Path("C:/Users")
        if users_dir.exists():
            for user_dir in users_dir.iterdir():
                if user_dir.is_dir() and user_dir.name not in ("SYSTEM", "Public", "Default"):
                    candidate = user_dir / ".local" / "share" / "mimocode" / "memory" / "sessions"
                    if candidate.exists():
                        return str(candidate)
        return ""

    def list_sessions(self) -> list[dict]:
        sessions = []
        if not self.memory_base or not self.memory_base.exists():
            return sessions
        for session_dir in self.memory_base.iterdir():
            if not session_dir.is_dir():
                continue
            files = list(session_dir.rglob("*.md"))
            if files:
                sessions.append({
                    "session_id": session_dir.name,
                    "path": str(session_dir),
                    "file_count": len(files),
                    "latest_mtime": max(f.stat().st_mtime for f in files),
                })
        sessions.sort(key=lambda x: x["latest_mtime"], reverse=True)
        return sessions

    def extract_traces(self, session_id: Optional[str] = None) -> list[dict]:
        """Extract deep behavior traces from session memory.

        If session_id is None, extracts from ALL sessions.
        """
        if session_id is not None:
            session_ids = [session_id]
        else:
            session_ids = [s["session_id"] for s in self.list_sessions()]

        all_traces = []
        for sid in session_ids:
            all_traces.extend(self._extract_from_session(sid))

        self._extracted_count += len(all_traces)
        return all_traces

    def _extract_from_session(self, session_id: str) -> list[dict]:
        """Extract traces from a single session."""

        session_path = self.memory_base / session_id
        if not session_path.exists():
            return []

        traces = []

        # Extract from checkpoint.md (richest source)
        checkpoint = session_path / "checkpoint.md"
        if checkpoint.exists():
            content = checkpoint.read_text(encoding="utf-8", errors="ignore")
            traces.extend(self._parse_checkpoint(session_id, content))

        # Extract from notes.md
        notes = session_path / "notes.md"
        if notes.exists():
            content = notes.read_text(encoding="utf-8", errors="ignore")
            traces.extend(self._parse_notes(session_id, content))

        # Extract from tasks/*/progress.md
        for progress_file in session_path.rglob("progress.md"):
            content = progress_file.read_text(encoding="utf-8", errors="ignore")
            traces.extend(self._parse_progress(session_id, progress_file, content))

        # Extract from checkpoint-moderately-silent.md
        silent = session_path / "checkpoint-moderately-silent.md"
        if silent.exists():
            content = silent.read_text(encoding="utf-8", errors="ignore")
            traces.extend(self._parse_notes(session_id, content))

        self._extracted_count += len(traces)
        return traces

    def _parse_checkpoint(self, session_id: str, content: str) -> list[dict]:
        """Parse checkpoint.md into behavior traces."""
        traces = []

        # Extract topic
        topic_match = re.search(r'^Topic:\s*(.+)$', content, re.MULTILINE)
        topic = topic_match.group(1).strip() if topic_match else "unknown"

        # Extract active intent (§1)
        intent = self._extract_section(content, "Active intent")
        if intent:
            traces.append({
                "action": f"session_intent:{topic[:40]}",
                "outcome": "recorded",
                "effect": 0.6,
                "params": {
                    "source": "checkpoint_intent",
                    "session": session_id,
                    "topic": topic,
                    "intent_preview": intent[:200],
                },
            })

        # Extract directives (§3)
        directives = self._extract_section(content, "Directives")
        if directives:
            directive_lines = [l.strip() for l in directives.split("\n") if l.strip().startswith("- ")]
            traces.append({
                "action": f"session_directives:{len(directive_lines)}",
                "outcome": "recorded",
                "effect": 0.5,
                "params": {
                    "source": "checkpoint_directives",
                    "session": session_id,
                    "directive_count": len(directive_lines),
                    "directives": directive_lines[:5],
                },
            })

        # Extract task tree (§4) — this is the richest data
        task_tree = self._extract_section(content, "Task tree")
        if task_tree:
            traces.extend(self._parse_task_tree(session_id, task_tree, topic))

        return traces

    def _extract_section(self, content: str, section_name: str) -> str:
        """Extract a section from checkpoint markdown."""
        pattern = rf'##\s*§\d+\s+{re.escape(section_name)}.*?\n(.*?)(?=\n##\s*§|\n##\s*[A-Z]|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _parse_task_tree(self, session_id: str, task_tree: str, topic: str) -> list[dict]:
        """Parse task tree into individual task traces."""
        traces = []
        lines = task_tree.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect completed tasks
            if line.startswith("✅"):
                task_name = line.lstrip("✅").strip()
                traces.append({
                    "action": f"task_complete:{task_name[:50]}",
                    "outcome": "success",
                    "effect": 0.8,
                    "params": {
                        "source": "task_tree",
                        "session": session_id,
                        "topic": topic,
                        "status": "completed",
                    },
                })
            # Detect failed tasks
            elif line.startswith("❌"):
                task_name = line.lstrip("❌").strip()
                traces.append({
                    "action": f"task_fail:{task_name[:50]}",
                    "outcome": "failure",
                    "effect": 0.2,
                    "params": {
                        "source": "task_tree",
                        "session": session_id,
                        "topic": topic,
                        "status": "failed",
                    },
                })
            # Detect in-progress tasks
            elif line.startswith("🔄") or line.startswith("⏳"):
                task_name = line.lstrip("🔄⏳").strip()
                traces.append({
                    "action": f"task_progress:{task_name[:50]}",
                    "outcome": "partial",
                    "effect": 0.5,
                    "params": {
                        "source": "task_tree",
                        "session": session_id,
                        "topic": topic,
                        "status": "in_progress",
                    },
                })

        return traces

    def _parse_notes(self, session_id: str, content: str) -> list[dict]:
        """Parse notes.md into behavior traces."""
        traces = []
        entries = re.split(r'##\s*\[turn\s+\d+', content)

        for entry in entries[1:]:  # Skip first (before first turn marker)
            entry = entry.strip()
            if not entry:
                continue

            # Extract turn number
            turn_match = re.match(r'(\d+)', entry)
            turn = turn_match.group(1) if turn_match else "?"

            # Classify the note
            entry_lower = entry.lower()
            if any(w in entry_lower for w in ["完成", "成功", "done", "success", "complete"]):
                outcome = "success"
                effect = 0.7
            elif any(w in entry_lower for w in ["失败", "错误", "fail", "error", "bug"]):
                outcome = "failure"
                effect = 0.3
            elif any(w in entry_lower for w in ["发现", "学习", "发现", "learn", "discover"]):
                outcome = "success"
                effect = 0.8
            else:
                outcome = "recorded"
                effect = 0.5

            traces.append({
                "action": f"session_note_turn{turn}",
                "outcome": outcome,
                "effect": effect,
                "params": {
                    "source": "session_notes",
                    "session": session_id,
                    "turn": turn,
                    "content_preview": entry[:200],
                },
            })

        return traces

    def _parse_progress(self, session_id: str, progress_file: Path, content: str) -> list[dict]:
        """Parse tasks/*/progress.md into behavior traces."""
        traces = []

        # Extract task ID from path
        task_match = re.search(r'tasks[/\\](T\d+)', str(progress_file))
        task_id = task_match.group(1) if task_match else "unknown"

        lines = content.split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "✅" in line or "完成" in line or "done" in line.lower():
                traces.append({
                    "action": f"task_step:{task_id}:{line[:40]}",
                    "outcome": "success",
                    "effect": 0.7,
                    "params": {
                        "source": "task_progress",
                        "session": session_id,
                        "task_id": task_id,
                    },
                })
            elif "❌" in line or "失败" in line or "fail" in line.lower():
                traces.append({
                    "action": f"task_step_fail:{task_id}:{line[:40]}",
                    "outcome": "failure",
                    "effect": 0.2,
                    "params": {
                        "source": "task_progress",
                        "session": session_id,
                        "task_id": task_id,
                    },
                })

        return traces

    @property
    def stats(self) -> dict:
        return {
            "memory_base": str(self.memory_base),
            "extracted_count": self._extracted_count,
            "sessions_found": len(self.list_sessions()),
        }
