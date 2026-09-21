"""Evidence-preserving receipts — SoL-Pi B-line for ARSI host empowerment.

SoL-Pi Evidence-Preserving Reducer (adapted, no Pi dependency):
  1. Archive exact source
  2. Build a compact receipt with quotes + hashes
  3. Deterministic verify
  4. On failure → fall back to original (never invent facts)

Aligns with VacuumGate: consolidation/empowerment must not invent ungrounded facts.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from arsi.foundation.paths import archive_dir, project_root, write_json_once

RECEIPT_SCHEMA = "arsi.evidence_receipt.v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _safe_quotes(text: str, n: int = 3, max_len: int = 160) -> list[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    quotes = []
    for ln in lines[:n]:
        quotes.append(ln[:max_len])
    # also last non-empty line
    if lines and lines[-1][:max_len] not in quotes:
        quotes.append(lines[-1][:max_len])
    return quotes[: n + 1]


@dataclass
class EvidenceReceipt:
    schema: str = RECEIPT_SCHEMA
    receipt_id: str = ""
    source_kind: str = ""
    source_path: str = ""
    source_hash: str = ""
    source_bytes: int = 0
    receipt_bytes: int = 0
    quotes: list[str] = field(default_factory=list)
    fields: dict = field(default_factory=dict)
    verified: bool = False
    reason: str = ""
    fallback_used: bool = False
    timestamp: str = ""
    writer_id: str = "arsi.foundation.evidence_receipt"

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.receipt_id:
            self.receipt_id = f"rcpt_{_sha256_text(self.timestamp + self.source_kind)}"

    @property
    def compression_ok(self) -> bool:
        return self.receipt_bytes > 0 and self.receipt_bytes <= max(self.source_bytes, 1)

    @property
    def grounded(self) -> bool:
        """Every quote must appear in archived source when source is local text."""
        if not self.quotes:
            return False
        p = Path(self.source_path) if self.source_path else None
        if p and p.exists():
            try:
                src = p.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return False
            return all(q in src for q in self.quotes)
        # no local source text → treat quotes as unverified
        return False

    def verify(self) -> bool:
        ok = True
        reasons = []
        if not self.source_hash:
            ok = False
            reasons.append("missing_source_hash")
        if not self.quotes:
            ok = False
            reasons.append("no_quotes")
        if not self.source_path:
            ok = False
            reasons.append("no_source_archive_ungrounded")
        elif Path(self.source_path).exists():
            try:
                data = Path(self.source_path).read_bytes()
                h = hashlib.sha256(data).hexdigest()[:16]
                if h != self.source_hash:
                    ok = False
                    reasons.append("source_hash_mismatch")
            except Exception as e:
                ok = False
                reasons.append(f"source_read_error:{e}")
            if not self.grounded:
                ok = False
                reasons.append("quote_not_in_source")
        else:
            ok = False
            reasons.append("source_path_missing")
        if self.compression_ok is False and self.source_bytes > 0:
            reasons.append("receipt_not_smaller")
        self.verified = ok
        self.reason = ";".join(reasons) if reasons else "ok"
        return ok

    def to_dict(self) -> dict:
        return asdict(self)

    def brief_snippet(self, max_chars: int = 500) -> str:
        """Host-facing compact form — evidence first, no invented advice."""
        q = "\n".join(f"  - {x}" for x in self.quotes[:3])
        return (
            f"# ARSI Evidence Receipt {self.receipt_id}\n"
            f"- kind: {self.source_kind}\n"
            f"- source: {self.source_path}\n"
            f"- hash: {self.source_hash} bytes={self.source_bytes}\n"
            f"- verified: {self.verified} ({self.reason})\n"
            f"## Grounded quotes\n{q}\n"
            f"## Structured fields\n{json.dumps(self.fields, ensure_ascii=False)[:max_chars]}\n"
        )


def build_receipt_from_text(
    text: str,
    source_kind: str = "brief",
    fields: Optional[dict] = None,
    archive_name: Optional[str] = None,
    persist: bool = True,
) -> EvidenceReceipt:
    """Archive exact text + compact receipt + verify."""
    text = text or ""
    raw_hash = _sha256_text(text)
    src_path = ""
    if persist:
        name = archive_name or f"{source_kind}_{raw_hash}.txt"
        src = archive_dir() / "evidence" / name
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            src.write_text(text, encoding="utf-8")
            src_path = str(src)
            raw_hash = hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        except Exception:
            src_path = ""
    quotes = _safe_quotes(text)
    receipt_body = "\n".join(quotes)
    rec = EvidenceReceipt(
        source_kind=source_kind,
        source_path=src_path,
        source_hash=raw_hash,
        source_bytes=len(text.encode("utf-8")),
        receipt_bytes=len(receipt_body.encode("utf-8")),
        quotes=quotes,
        fields=dict(fields or {}),
    )
    ok = rec.verify()
    if not ok and src_path:
        rec.fallback_used = True
        rec.reason = rec.reason or "verify_failed_fallback_original"
    if persist:
        try:
            out = archive_dir() / "evidence" / f"{rec.receipt_id}.json"
            write_json_once(out, rec.to_dict(), writer_id="evidence_receipt.build")
        except Exception:
            pass
    return rec


def fusion_write_and_verify(path: Path, content: str, receipt_kind: str = "skill") -> dict:
    """Action Fusion (SoL-Pi): write host artifact + verify + receipt in one call."""
    path = Path(path)
    before = path.stat().st_size if path.exists() else 0
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        after = path.stat().st_size if path.exists() else 0
        write_ok = after >= len(content.encode("utf-8")) * 0.5 and after > 0
        rec = build_receipt_from_text(
            content,
            source_kind=receipt_kind,
            fields={"path": str(path), "bytes_before": before, "bytes_after": after},
            archive_name=f"{receipt_kind}_{path.name}_{_sha256_text(str(path))}.txt",
        )
        return {
            "fused_write": True,
            "path": str(path),
            "write_ok": write_ok,
            "bytes_before": before,
            "bytes_after": after,
            "receipt": rec.to_dict(),
            "verified": rec.verified,
        }
    except Exception as e:
        return {
            "fused_write": True,
            "path": str(path),
            "write_ok": False,
            "error": str(e),
            "receipt": {"verified": False, "reason": str(e)},
        }
