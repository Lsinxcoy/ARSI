"""Structured audit events for harness evolution (stage / gate / commit)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class AuditLog:
    def __init__(self, root: str | Path):
        self.path = Path(root) / "audit.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        stage: str,
        event: str,
        gate: str = "",
        commit: str = "",
        ok: bool = True,
        **payload: Any,
    ) -> dict:
        row = {
            "ts": datetime.now().isoformat(),
            "stage": stage,
            "event": event,
            "gate": gate,
            "commit": commit,
            "ok": bool(ok),
            **payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        return row

    def tail(self, n: int = 20) -> list[dict]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        return rows[-n:]
