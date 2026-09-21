"""ObservationPack — SoL-Pi B-line for host-facing large artifacts.

Large payloads are archived locally; hosts receive a stable handle + excerpt.
Exact retrieval remains possible via get_observation(handle).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from arsi.foundation.paths import archive_dir, write_json_once

PACK_SCHEMA = "arsi.observation_pack.v1"
DEFAULT_THRESHOLD = 2048  # bytes; hosts get handle beyond this


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


@dataclass
class ObservationHandle:
    handle: str
    kind: str = ""
    path: str = ""
    size: int = 0
    hash: str = ""
    excerpt: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def brief_lines(self) -> list[str]:
        return [
            f"handle={self.handle}",
            f"kind={self.kind}",
            f"size={self.size}",
            f"hash={self.hash}",
            f"excerpt={self.excerpt[:200]}",
            f"retrieve=arsi.foundation.observation_pack.get_observation('{self.handle}')",
        ]


class ObservationPack:
    def __init__(self, threshold: int = DEFAULT_THRESHOLD, root: Optional[Path] = None):
        self.threshold = int(threshold)
        self.root = Path(root) if root else (archive_dir() / "observation_pack")
        self._index: dict[str, dict] = {}

    def pack(self, payload: str, kind: str = "observation") -> ObservationHandle | str:
        """Return handle if large, else return original payload unchanged."""
        if payload is None:
            payload = ""
        raw = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, default=str)
        data = raw.encode("utf-8", errors="replace")
        if len(data) < self.threshold:
            return raw
        h = _hash(data)
        handle_id = f"obs_{kind}_{h}"
        path = self.root / f"{handle_id}.txt"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        except Exception:
            # fallback: cannot pack
            return raw
        excerpt = raw[:240].replace("\n", " ") + ("…" if len(raw) > 240 else "")
        rec = ObservationHandle(
            handle=handle_id,
            kind=kind,
            path=str(path),
            size=len(data),
            hash=h,
            excerpt=excerpt,
            created_at=datetime.now().isoformat(),
        )
        self._index[handle_id] = rec.to_dict()
        try:
            write_json_once(
                self.root / "index.json",
                {"schema": PACK_SCHEMA, "items": self._index},
                writer_id="observation_pack",
            )
        except Exception:
            pass
        return rec

    def format_handle(self, obj: ObservationHandle | str) -> str:
        if isinstance(obj, str):
            return obj
        lines = [
            "# ARSI Observation Handle",
            *[f"- {x}" for x in obj.brief_lines()],
        ]
        return "\n".join(lines)

    def get_observation(self, handle: str) -> Optional[str]:
        meta = self._index.get(handle)
        if not meta:
            p = self.root / f"{handle}.txt"
            if p.exists():
                return p.read_text(encoding="utf-8", errors="replace")
            return None
        p = Path(meta.get("path") or "")
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
        return None


# process-wide pack for host channel
PACK = ObservationPack()


def pack_for_host(payload: str, kind: str = "brief") -> str:
    """Host-facing: large → handle block; small → unchanged."""
    return PACK.format_handle(PACK.pack(payload, kind=kind))


def get_observation(handle: str) -> Optional[str]:
    return PACK.get_observation(handle)
