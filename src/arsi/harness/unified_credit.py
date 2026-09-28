"""P-R4 Unified credit ledger L_t for dual-Ω (policy + harness).

Problem: Dream-RSI policy credit and Evolver harness credit are separate books;
a single improvement often moves both — ΔS cannot be attributed.

Rule: one table for manifest edits and dream policy revisions.
source_tag ∈ {policy, harness, both}.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

SOURCE_POLICY = "policy"
SOURCE_HARNESS = "harness"
SOURCE_BOTH = "both"
VALID_SOURCES = (SOURCE_POLICY, SOURCE_HARNESS, SOURCE_BOTH)


@dataclass
class UnifiedEditRecord:
    t: int
    source: str  # policy | harness | both
    omega_id: str  # policy:beta/v | harness:CM-xxx | both:pair
    component: str = ""
    hypothesis: str = ""
    dS: float = 0.0
    dC: float = 0.0
    accepted: bool = False
    attribution: dict = field(default_factory=dict)  # {policy: x, harness: y} shares
    note: str = ""

    def __post_init__(self):
        if self.source not in VALID_SOURCES:
            self.source = SOURCE_BOTH
        if not self.attribution and self.source == SOURCE_BOTH:
            self.attribution = {"policy": 0.5, "harness": 0.5}

    def to_dict(self) -> dict:
        return asdict(self)


def _default_attr(source: str) -> dict:
    if source == SOURCE_POLICY:
        return {"policy": 1.0, "harness": 0.0}
    if source == SOURCE_HARNESS:
        return {"policy": 0.0, "harness": 1.0}
    return {"policy": 0.5, "harness": 0.5}


class UnifiedCreditLedger:
    """Shared L_t across Ω_policy and Ω_harness."""

    def __init__(self):
        self.records: list[UnifiedEditRecord] = []

    def log(
        self,
        t: int,
        source: str,
        omega_id: str,
        *,
        component: str = "",
        hypothesis: str = "",
        dS: float = 0.0,
        dC: float = 0.0,
        accepted: bool = False,
        attribution: Optional[dict] = None,
        note: str = "",
    ) -> UnifiedEditRecord:
        rec = UnifiedEditRecord(
            t=int(t),
            source=source if source in VALID_SOURCES else SOURCE_BOTH,
            omega_id=str(omega_id),
            component=str(component or ""),
            hypothesis=str(hypothesis or ""),
            dS=float(dS),
            dC=float(dC),
            accepted=bool(accepted),
            attribution=dict(attribution) if attribution else _default_attr(source),
            note=str(note or ""),
        )
        self.records.append(rec)
        return rec

    def log_manifest(self, t: int, manifest_id: str, **kw) -> UnifiedEditRecord:
        kw.setdefault("component", "harness.manifest")
        return self.log(t, SOURCE_HARNESS, manifest_id, **kw)

    def log_dream(self, t: int, policy_id: str, **kw) -> UnifiedEditRecord:
        kw.setdefault("component", "dream.policy")
        return self.log(t, SOURCE_POLICY, policy_id, **kw)

    def log_both(self, t: int, pair_id: str, *, policy_share: float = 0.5, **kw) -> UnifiedEditRecord:
        share = max(0.0, min(1.0, float(policy_share)))
        kw.setdefault("attribution", {"policy": share, "harness": round(1.0 - share, 6)})
        kw.setdefault("component", "dual_omega")
        return self.log(t, SOURCE_BOTH, pair_id, **kw)

    def attribution_totals(self) -> dict:
        """Credit mass split by Ω — accepted ΔS weighted by attribution shares."""
        pol = 0.0
        har = 0.0
        for r in self.records:
            if not r.accepted:
                continue
            a = r.attribution or _default_attr(r.source)
            pol += float(r.dS) * float(a.get("policy", 0.0) or 0.0)
            har += float(r.dS) * float(a.get("harness", 0.0) or 0.0)
        return {
            "policy_credit": round(pol, 6),
            "harness_credit": round(har, 6),
            "total": round(pol + har, 6),
            "n": len(self.records),
            "n_accepted": sum(1 for r in self.records if r.accepted),
        }

    def by_source(self) -> dict:
        out = {s: {"n": 0, "accepted": 0, "dS": 0.0} for s in VALID_SOURCES}
        for r in self.records:
            row = out.setdefault(r.source, {"n": 0, "accepted": 0, "dS": 0.0})
            row["n"] += 1
            row["dS"] += float(r.dS)
            if r.accepted:
                row["accepted"] += 1
        for row in out.values():
            row["dS"] = round(row["dS"], 6)
        return out

    def to_dict(self) -> dict:
        return {
            "n": len(self.records),
            "attribution": self.attribution_totals(),
            "by_source": self.by_source(),
            "records": [r.to_dict() for r in self.records[-50:]],
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps([r.to_dict() for r in self.records], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def unify_from_pair(
    t: int,
    *,
    manifest_id: str = "",
    policy_id: str = "",
    dS: float = 0.0,
    dC: float = 0.0,
    accepted: bool = False,
    policy_share: float = 0.5,
    hypothesis: str = "",
) -> UnifiedEditRecord:
    """One improvement that moved both Ω books → single attributed row."""
    ledger = UnifiedCreditLedger()
    pair = f"{policy_id or 'policy'}×{manifest_id or 'harness'}"
    return ledger.log_both(
        t,
        pair,
        policy_share=policy_share,
        dS=dS,
        dC=dC,
        accepted=accepted,
        hypothesis=hypothesis or f"manifest={manifest_id}|policy={policy_id}",
    )
