"""SYNTHEX Autopoiesis (母巢) adapter — extract traces from the Mother Nest.

Data sources:
1. synthex_main.db — mechanism status + memory store
2. state/ — mechanism manifest, progress, rules
3. WAL logs — write-ahead log entries
4. Source modules — mechanism inventory
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SynthexAdapter:
    """Extracts behavior traces from SYNTHEX Autopoiesis (母巢)."""

    def __init__(self, synthex_home: Optional[str] = None):
        self.home = Path(synthex_home or "E:/SYNTHEX Autopoiesis")
        self._extracted_count = 0

    def extract_traces(self) -> list[dict]:
        """Extract behavior traces from all SYNTHEX data sources."""
        traces = []
        traces.extend(self._extract_from_db())
        traces.extend(self._extract_from_state())
        traces.extend(self._extract_from_wal())
        traces.extend(self._extract_from_mechanisms())
        traces.extend(self._extract_from_admissions())
        traces.extend(self._extract_from_rules())
        traces.extend(self._extract_from_depth_progress())
        self._extracted_count += len(traces)
        return traces

    def _extract_from_admissions(self) -> list[dict]:
        """Extract traces from v2_admissions.jsonl (gate decisions)."""
        traces = []
        admissions_path = self.home / "state" / "v2_admissions.jsonl"
        if not admissions_path.exists():
            return traces

        try:
            with open(admissions_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Parse last 50 admissions
            for line in lines[-50:]:
                try:
                    data = json.loads(line.strip())
                    decision = data.get("context", {}).get("decision", "unknown")
                    traces.append({
                        "action": f"synthex_gate:{data.get('name', 'unknown')}",
                        "outcome": "success" if decision == "admit" else "partial" if decision == "hold" else "failure",
                        "effect": 0.8 if decision == "admit" else 0.4 if decision == "hold" else 0.2,
                        "params": {
                            "source": "v2_admissions",
                            "decision": decision,
                            "rid": data.get("rid"),
                            "n_records": data.get("context", {}).get("n_records", 0),
                        },
                    })
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.warning(f"Failed to read admissions: {e}")

        return traces

    def _extract_from_rules(self) -> list[dict]:
        """Extract traces from rimrule_rules.jsonl."""
        traces = []
        rules_path = self.home / "state" / "rimrule_rules.jsonl"
        if not rules_path.exists():
            return traces

        try:
            with open(rules_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        traces.append({
                            "action": f"synthex_rule:{data.get('condition', 'unknown')}",
                            "outcome": "success" if data.get("confidence", 0) > 0.8 else "partial",
                            "effect": data.get("confidence", 0.5),
                            "params": {
                                "source": "rimrule",
                                "condition": data.get("condition", ""),
                                "action": data.get("action", ""),
                                "confidence": data.get("confidence", 0),
                                "support": data.get("support", 0),
                                "mdl_score": data.get("mdl_score", 0),
                            },
                        })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Failed to read rules: {e}")

        return traces

    def _extract_from_depth_progress(self) -> list[dict]:
        """Extract traces from depth_progress.json."""
        traces = []
        progress_path = self.home / "state" / "depth_progress.json"
        if not progress_path.exists():
            return traces

        try:
            data = json.loads(progress_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, value in list(data.items())[:20]:
                    if isinstance(value, dict):
                        traces.append({
                            "action": f"synthex_progress:{key}",
                            "outcome": "recorded",
                            "effect": 0.5,
                            "params": {
                                "source": "depth_progress",
                                "key": key,
                                "value_keys": list(value.keys())[:5],
                            },
                        })
                    elif isinstance(value, (int, float)):
                        traces.append({
                            "action": f"synthex_progress:{key}",
                            "outcome": "recorded",
                            "effect": min(1.0, max(0.0, value / 100)) if isinstance(value, (int, float)) else 0.5,
                            "params": {"source": "depth_progress", "key": key, "value": value},
                        })
        except Exception as e:
            logger.warning(f"Failed to read depth progress: {e}")

        return traces

    def _extract_from_db(self) -> list[dict]:
        """Extract traces from synthex_main.db."""
        traces = []
        db_path = self.home / "synthex_main.db"
        if not db_path.exists():
            return traces

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row

            # Mechanism status
            cursor = conn.execute("SELECT * FROM mechanism_status")
            for row in cursor.fetchall():
                traces.append({
                    "action": "synthex_mechanism_status",
                    "outcome": "recorded",
                    "effect": 0.5,
                    "params": {"source": "mechanism_status", "data": dict(row)},
                })

            # Memory store
            cursor = conn.execute("SELECT * FROM memory_store LIMIT 20")
            for row in cursor.fetchall():
                traces.append({
                    "action": "synthex_memory",
                    "outcome": "recorded",
                    "effect": 0.5,
                    "params": {"source": "memory_store", "data": dict(row)},
                })

            conn.close()
        except Exception as e:
            logger.warning(f"Failed to read SYNTHEX DB: {e}")

        return traces

    def _extract_from_state(self) -> list[dict]:
        """Extract traces from state files."""
        traces = []
        state_dir = self.home / "state"
        if not state_dir.exists():
            return traces

        # Mechanism manifest
        manifest_path = state_dir / "mechanism_manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                mechanisms = data.get("mechanisms", [])
                traces.append({
                    "action": "synthex_mechanism_manifest",
                    "outcome": "success",
                    "effect": 0.8,
                    "params": {
                        "source": "mechanism_manifest",
                        "count": len(mechanisms),
                        "mechanisms": mechanisms[:10],
                    },
                })
            except Exception as e:
                logger.warning(f"Failed to read mechanism manifest: {e}")

        # Other state files
        for state_file in state_dir.glob("*.json"):
            if state_file.name == "mechanism_manifest.json":
                continue
            try:
                data = json.loads(state_file.read_text(encoding="utf-8"))
                traces.append({
                    "action": f"synthex_state:{state_file.stem}",
                    "outcome": "recorded",
                    "effect": 0.5,
                    "params": {
                        "source": "state_file",
                        "filename": state_file.name,
                        "keys": list(data.keys())[:5] if isinstance(data, dict) else [],
                    },
                })
            except Exception:
                continue

        return traces

    def _extract_from_wal(self) -> list[dict]:
        """Extract traces from WAL logs."""
        traces = []
        wal_dir = self.home / "wal_logs"
        if not wal_dir.exists():
            return traces

        wal_files = sorted(wal_dir.glob("*.log"), reverse=True)[:5]
        for wf in wal_files:
            try:
                content = wf.read_text(encoding="utf-8", errors="ignore")
                lines = content.strip().split("\n")
                traces.append({
                    "action": "synthex_wal",
                    "outcome": "recorded",
                    "effect": 0.4,
                    "params": {
                        "source": "wal_log",
                        "filename": wf.name,
                        "line_count": len(lines),
                    },
                })
            except Exception:
                continue

        return traces

    def _extract_from_mechanisms(self) -> list[dict]:
        """Extract mechanism inventory from source code."""
        traces = []
        mech_dir = self.home / "src" / "synthex_autopoiesis" / "mechanisms"
        if not mech_dir.exists():
            return traces

        mech_files = list(mech_dir.glob("*.py"))
        mech_files = [f for f in mech_files if f.name != "__init__.py" and f.name != "__pycache__"]

        for mf in mech_files[:20]:
            try:
                content = mf.read_text(encoding="utf-8", errors="ignore")
                # Check if it has actual implementation (not just stub)
                has_class = "class " in content
                has_method = "def " in content
                lines = len(content.split("\n"))

                traces.append({
                    "action": f"synthex_mechanism:{mf.stem}",
                    "outcome": "success" if has_class and has_method and lines > 20 else "partial",
                    "effect": min(1.0, lines / 200),
                    "params": {
                        "source": "mechanism_source",
                        "filename": mf.name,
                        "lines": lines,
                        "has_class": has_class,
                        "has_method": has_method,
                    },
                })
            except Exception:
                continue

        return traces

    def get_stats(self) -> dict:
        """Get SYNTHEX adapter statistics."""
        return {
            "home": str(self.home),
            "extracted_count": self._extracted_count,
            "db_exists": (self.home / "synthex_main.db").exists(),
            "state_exists": (self.home / "state").exists(),
            "wal_exists": (self.home / "wal_logs").exists(),
            "src_exists": (self.home / "src" / "synthex_autopoiesis").exists(),
        }
