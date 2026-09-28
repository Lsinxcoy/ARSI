"""Change-Manifest — typed harness edit artifact (HarnessX §10.3 style)."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from arsi.harness.taxonomy import dimension_of_symbol, is_edit_allowed


@dataclass
class ChangeManifest:
    manifest_id: str = ""
    dimension: str = ""
    module: str = ""
    symbol: str = ""
    edit_type: str = "prompt"  # prompt | processor | tool | config | control
    summary: str = ""
    diff: str = ""
    evidence: list = field(default_factory=list)
    inverse_op: str = ""  # how to roll back
    status: str = "proposed"  # proposed | accepted | rejected | rolled_back
    reject_reason: str = ""
    created_at: str = ""
    variant_id: str = "default"

    def __post_init__(self):
        if not self.manifest_id:
            self.manifest_id = f"CM-{uuid.uuid4().hex[:10]}"
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.dimension:
            self.dimension = dimension_of_symbol(self.symbol)

    def validate_scope(self) -> tuple[bool, str]:
        return is_edit_allowed(self.dimension, self.module, self.symbol)

    def to_dict(self) -> dict:
        return asdict(self)


class ManifestStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "change_manifests.jsonl"

    def append(self, m: ChangeManifest) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")

    def list(self, status: Optional[str] = None) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if status is None or d.get("status") == status:
                out.append(d)
        return out

    def update_status(self, manifest_id: str, status: str, reason: str = "") -> bool:
        rows = self.list()
        hit = False
        for r in rows:
            if r.get("manifest_id") == manifest_id:
                r["status"] = status
                r["reject_reason"] = reason
                hit = True
        if hit:
            with open(self.path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return hit
