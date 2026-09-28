"""P1-1 Compile-gate accounting (CM-7b86cd1d1f 收账).

Every CodeVerifier compile reject/pass is counted so landscape can prove the
gate is working (invalid_or_compile cluster shrink) without guessing.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CompileGateLedger:
    n_pass: int = 0
    n_reject: int = 0
    n_syntax: int = 0
    last_reason: str = ""
    sources: dict = field(default_factory=dict)  # source → counts

    def record(self, *, ok: bool, reason: str = "", source: str = "code_verifier", syntax: bool = False) -> None:
        if syntax:
            self.n_syntax += 1
        if ok:
            self.n_pass += 1
        else:
            self.n_reject += 1
            self.last_reason = str(reason or "")[:200]
        row = self.sources.setdefault(str(source or "unknown"), {"pass": 0, "reject": 0})
        row["pass" if ok else "reject"] += 1

    @property
    def reject_rate(self) -> float:
        tot = self.n_pass + self.n_reject
        return round(self.n_reject / tot, 6) if tot else 0.0

    @property
    def total(self) -> int:
        return self.n_pass + self.n_reject

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reject_rate"] = self.reject_rate
        d["total"] = self.total
        d["note"] = "cm_7b86cd1d1f_compile_gate_ledger"
        return d

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


# process-wide ledger (daemon / tests)
_LEDGER = CompileGateLedger()
# P: same candidate_id compile fails → abandon after max retries
_CANDIDATE_FAILS: dict[str, int] = {}
MAX_CANDIDATE_COMPILES = 2  # 1 initial + 1 repair
_PERSIST_PATH = None


def _persist_path():
    global _PERSIST_PATH
    if _PERSIST_PATH is None:
        try:
            from arsi.foundation.paths import archive_dir

            _PERSIST_PATH = archive_dir() / "harness" / "candidate_fails.json"
        except Exception:
            _PERSIST_PATH = Path("archive/harness/candidate_fails.json")
    return _PERSIST_PATH


def _load_persist() -> None:
    p = _persist_path()
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _CANDIDATE_FAILS.update({str(k): int(v) for k, v in data.items()})
    except Exception:
        pass


def _save_persist() -> None:
    p = _persist_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_CANDIDATE_FAILS, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


_load_persist()


def candidate_reject(candidate_id: str) -> tuple[bool, str]:
    """True if this candidate_id has exhausted compile attempts (retry ring)."""
    cid = str(candidate_id or "")
    if not cid:
        return False, ""
    n = int(_CANDIDATE_FAILS.get(cid, 0))
    if n >= MAX_CANDIDATE_COMPILES:
        return True, f"compile_retry_ring_abandon:{cid}:n={n}"
    return False, ""


def bump_candidate_fail(candidate_id: str) -> int:
    cid = str(candidate_id or "")
    if not cid:
        return 0
    _CANDIDATE_FAILS[cid] = int(_CANDIDATE_FAILS.get(cid, 0)) + 1
    _save_persist()
    return _CANDIDATE_FAILS[cid]


def extract_candidate_id(blob: str) -> str:
    import re

    m = re.search(r"compiled_mech_ext_([0-9a-fA-F]+)", str(blob or ""))
    return m.group(1) if m else ""


def note_trace_candidate(blob: str) -> tuple[str, bool, str]:
    """P: ingest/digester path — synthex failures never hit CodeVerifier.

    Returns (candidate_id, should_abandon, reason). Bumps fail count per id.
    """
    cid = extract_candidate_id(blob)
    if not cid:
        return "", False, ""
    skip, why = candidate_reject(cid)
    if skip:
        return cid, True, why
    bump_candidate_fail(cid)
    n = _CANDIDATE_FAILS.get(cid, 0)
    if n >= MAX_CANDIDATE_COMPILES:
        return cid, True, f"compile_retry_ring_abandon:{cid}:n={n}"
    return cid, False, f"compile_fail_n={n}"


def abandoned_candidates() -> list[str]:
    return sorted([cid for cid, n in _CANDIDATE_FAILS.items() if n >= MAX_CANDIDATE_COMPILES])


def is_reversion_blocked(blob: str) -> tuple[bool, str]:
    """P: hard-abandon — re-version of exhausted candidate_id is refused."""
    cid = extract_candidate_id(blob)
    if cid and cid in set(abandoned_candidates()):
        return True, f"reversion_blocked_abandoned:{cid}"
    return False, ""


def get_compile_ledger() -> CompileGateLedger:
    return _LEDGER


def reset_compile_ledger() -> None:
    global _LEDGER
    _LEDGER = CompileGateLedger()


def record_compile(*, ok: bool, reason: str = "", source: str = "code_verifier", syntax: bool = False) -> dict:
    _LEDGER.record(ok=ok, reason=reason, source=source, syntax=syntax)
    return _LEDGER.to_dict()


def compile_gate_effect(label_counts_now: dict, label_counts_before: dict) -> dict:
    """P1 收账: did invalid_or_compile shrink after the gate?"""
    now = float((label_counts_now or {}).get("invalid_or_compile") or 0)
    before = float((label_counts_before or {}).get("invalid_or_compile") or 0)
    delta = now - before
    return {
        "invalid_or_compile_before": before,
        "invalid_or_compile_now": now,
        "delta": round(delta, 4),
        "shrank": delta < 0,
        "ledger": _LEDGER.to_dict(),
        "note": "compile_gate_effect_on_cluster",
    }
