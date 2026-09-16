"""SQLite storage engine with WAL mode and CAS write tokens.

Inherited from SYNTHEX's MinervaStore design.
Three-zone isolation: SELF / EXPERIENCE / PROXY.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from arsi.foundation.schema import (
    CausalLink,
    MemoryRecord,
    MemoryStatus,
    MemoryZone,
    BehaviorTrace,
)

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    zone TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    agent_id TEXT,
    generation INTEGER DEFAULT 0,
    importance REAL DEFAULT 0.5,
    utility_score REAL DEFAULT 0.0,
    freshness REAL DEFAULT 1.0,
    created_at TEXT NOT NULL,
    last_accessed TEXT,
    access_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'active'
);

CREATE INDEX IF NOT EXISTS idx_memories_zone ON memories(zone);
CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_id);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_generation ON memories(generation);

CREATE TABLE IF NOT EXISTS causal_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experience_id TEXT NOT NULL,
    source_traces TEXT DEFAULT '[]',
    distilled_at TEXT NOT NULL,
    verified_by TEXT,
    applied_to TEXT DEFAULT '[]',
    freshness_eta REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS traces (
    id TEXT PRIMARY KEY,
    agent_id TEXT,
    action TEXT,
    action_params TEXT DEFAULT '{}',
    outcome TEXT,
    effect REAL DEFAULT 0.0,
    generation INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL,
    state_before TEXT,
    state_after TEXT
);

CREATE TABLE IF NOT EXISTS self_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    data TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS effect_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism TEXT NOT NULL,
    effect REAL NOT NULL,
    context TEXT,
    generation INTEGER DEFAULT 0,
    timestamp TEXT NOT NULL
);
"""


class CASLock:
    """Compare-and-Swap write token for concurrent safety."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._version = 0

    def acquire(self, expected_version: int) -> bool:
        with self._lock:
            if self._version == expected_version:
                self._version += 1
                return True
            return False

    @property
    def version(self) -> int:
        return self._version


class MnemosyneStore:
    """Unified memory store with three-zone isolation.

    Based on SYNTHEX MinervaStore (CAS + WAL) upgraded for ARSI.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._cas = CASLock()
        self._generation = 0
        logger.info(f"MnemosyneStore initialized at {db_path}")

    @property
    def current_generation(self) -> int:
        return self._generation

    def bump_generation(self) -> int:
        self._generation += 1
        return self._generation

    # ── Memory CRUD ─────────────────────────────────────────────

    def write_memory(self, record: MemoryRecord) -> bool:
        """Write a memory record to the appropriate zone."""
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (id, zone, content, tags, agent_id, generation,
                    importance, utility_score, freshness, created_at,
                    last_accessed, access_count, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.id,
                    record.zone.value,
                    record.content,
                    json.dumps(record.tags),
                    record.agent_id,
                    record.generation,
                    record.importance,
                    record.utility_score,
                    record.freshness,
                    record.created_at.isoformat(),
                    record.last_accessed.isoformat() if record.last_accessed else None,
                    record.access_count,
                    record.status.value,
                ),
            )
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to write memory: {e}")
            return False

    def read_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row:
            return self._row_to_memory(row)
        return None

    def search_memories(
        self,
        zone: Optional[MemoryZone] = None,
        agent_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        query = "SELECT * FROM memories WHERE status = 'active'"
        params: list[Any] = []

        if zone:
            query += " AND zone = ?"
            params.append(zone.value)
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)

        query += " ORDER BY utility_score DESC, freshness DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        results = [self._row_to_memory(r) for r in rows]

        if tags:
            tag_set = set(tags)
            results = [r for r in results if tag_set & set(r.tags)]

        return results

    def update_utility(self, memory_id: str, delta: float) -> None:
        """Update utility score (useful for utility_decay)."""
        self._conn.execute(
            "UPDATE memories SET utility_score = utility_score + ?, "
            "last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (delta, datetime.now().isoformat(), memory_id),
        )
        self._conn.commit()

    def archive_memory(self, memory_id: str) -> None:
        self._conn.execute(
            "UPDATE memories SET status = 'archived' WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    def quarantine_memory(self, memory_id: str) -> None:
        self._conn.execute(
            "UPDATE memories SET status = 'quarantined' WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    # ── Causal Links ────────────────────────────────────────────

    def write_causal_link(self, link: CausalLink) -> int:
        cursor = self._conn.execute(
            """INSERT INTO causal_links
               (experience_id, source_traces, distilled_at, verified_by,
                applied_to, freshness_eta)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                link.experience_id,
                json.dumps(link.source_traces),
                link.distilled_at.isoformat(),
                link.verified_by,
                json.dumps(link.applied_to),
                link.freshness_eta,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_causal_link(self, experience_id: str) -> Optional[CausalLink]:
        row = self._conn.execute(
            "SELECT * FROM causal_links WHERE experience_id = ?",
            (experience_id,),
        ).fetchone()
        if row:
            return CausalLink(
                experience_id=row["experience_id"],
                source_traces=json.loads(row["source_traces"]),
                distilled_at=datetime.fromisoformat(row["distilled_at"]),
                verified_by=row["verified_by"],
                applied_to=json.loads(row["applied_to"]),
                freshness_eta=row["freshness_eta"],
            )
        return None

    # ── Behavior Traces ─────────────────────────────────────────

    def write_trace(self, trace: BehaviorTrace) -> bool:
        try:
            self._conn.execute(
                """INSERT OR REPLACE INTO traces
                   (id, agent_id, action, action_params, outcome, effect,
                    generation, timestamp, state_before, state_after)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace.id,
                    trace.agent_id,
                    trace.action,
                    json.dumps(trace.action_params),
                    trace.outcome,
                    trace.effect,
                    trace.generation,
                    trace.timestamp.isoformat(),
                    trace.state_before.model_dump_json(),
                    trace.state_after.model_dump_json(),
                ),
            )
            self._conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to write trace: {e}")
            return False

    def get_recent_traces(self, n: int = 50, agent_id: Optional[str] = None) -> list[dict]:
        query = "SELECT * FROM traces"
        params: list[Any] = []
        if agent_id:
            query += " WHERE agent_id = ?"
            params.append(agent_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(n)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Self Records (Ψ, η, dream sessions, reports) ───────────

    def write_self_record(self, kind: str, data: dict) -> int:
        cursor = self._conn.execute(
            "INSERT INTO self_records (kind, data, timestamp) VALUES (?, ?, ?)",
            (kind, json.dumps(data, default=str), datetime.now().isoformat()),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_self_records(self, kind: str, limit: int = 10) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM self_records WHERE kind = ? ORDER BY timestamp DESC LIMIT ?",
            (kind, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Effect Ledger ───────────────────────────────────────────

    def record_effect(self, mechanism: str, effect: float, context: str = "") -> None:
        self._conn.execute(
            "INSERT INTO effect_ledger (mechanism, effect, context, generation, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (mechanism, effect, context, self._generation, datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_effect_ledger(self, mechanism: Optional[str] = None, limit: int = 100) -> list[dict]:
        query = "SELECT * FROM effect_ledger"
        params: list[Any] = []
        if mechanism:
            query += " WHERE mechanism = ?"
            params.append(mechanism)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ── Stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = {}
        for zone in MemoryZone:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM memories WHERE zone = ? AND status = 'active'",
                (zone.value,),
            ).fetchone()
            stats[f"{zone.value}_count"] = row["cnt"]

        row = self._conn.execute("SELECT COUNT(*) as cnt FROM traces").fetchone()
        stats["trace_count"] = row["cnt"]

        row = self._conn.execute("SELECT COUNT(*) as cnt FROM causal_links").fetchone()
        stats["causal_link_count"] = row["cnt"]

        stats["generation"] = self._generation
        return stats

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            zone=MemoryZone(row["zone"]),
            content=row["content"],
            tags=json.loads(row["tags"]),
            agent_id=row["agent_id"],
            generation=row["generation"],
            importance=row["importance"],
            utility_score=row["utility_score"],
            freshness=row["freshness"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=(
                datetime.fromisoformat(row["last_accessed"])
                if row["last_accessed"]
                else None
            ),
            access_count=row["access_count"],
            status=MemoryStatus(row["status"]),
        )
