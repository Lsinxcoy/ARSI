"""Self-Harness constraints (arXiv:2606.09498): minimal edit + dual regression.

Weakness Mining → minimal Proposal → Validation (regression before accept).
Dual regression: held-in AND held-out must not regress (held-out = private split).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass
class DualRegressionVerdict:
    ok: bool
    held_in_ok: bool
    held_out_ok: bool
    reason: str = ""
    detail: dict = field(default_factory=dict)


def minimal_edit_score(diff: str) -> float:
    """Lower is more minimal. Lines changed + chars / 80."""
    lines = [ln for ln in (diff or "").splitlines() if ln.strip() and not ln.strip().startswith("###")]
    changed = sum(1 for ln in lines if ln.lstrip().startswith(("+", "-", "!")))
    return float(changed) + len(diff or "") / 80.0


def prefer_minimal(cand_diff: str, other_diff: str) -> bool:
    """True if cand is at least as minimal as other."""
    return minimal_edit_score(cand_diff) <= minimal_edit_score(other_diff) + 1e-9


def dual_regression(
    held_in_before: float,
    held_in_after: float,
    held_out_before: float,
    held_out_after: float,
    eps: float = 1e-6,
) -> DualRegressionVerdict:
    """Accept only if neither split regresses (Self-Harness Proposal Validation)."""
    hi_ok = float(held_in_after) >= float(held_in_before) - eps
    ho_ok = float(held_out_after) >= float(held_out_before) - eps
    if hi_ok and ho_ok:
        return DualRegressionVerdict(True, hi_ok, ho_ok, "ok")
    bad = []
    if not hi_ok:
        bad.append("held_in_regress")
    if not ho_ok:
        bad.append("held_out_regress")
    return DualRegressionVerdict(
        False,
        hi_ok,
        ho_ok,
        "+".join(bad),
        {
            "held_in": [held_in_before, held_in_after],
            "held_out": [held_out_before, held_out_after],
        },
    )


def self_harness_round(
    *,
    traces: Sequence[dict],
    proposals: Sequence[dict],
    held_in_before: float,
    held_in_after: float,
    held_out_before: float,
    held_out_after: float,
) -> dict:
    """Self-Harness loop glue: mine → prefer minimal → dual regression gate.

    proposals: [{diff, summary, ...}]
    """
    from arsi.harness.digester import Digester

    clusters = Digester().digest(list(traces or []))
    ranked = sorted(
        list(proposals or []),
        key=lambda p: minimal_edit_score(str((p or {}).get("diff") or "")),
    )
    best = ranked[0] if ranked else None
    dual = dual_regression(held_in_before, held_in_after, held_out_before, held_out_after)
    return {
        "weakness_clusters": [{"label": c.label, "n": c.n} for c in clusters[:8]],
        "n_proposals": len(list(proposals or [])),
        "minimal_diff_score": minimal_edit_score(str((best or {}).get("diff") or "")),
        "dual_regression": dual.to_dict() if hasattr(dual, "to_dict") else {
            "ok": dual.ok, "reason": dual.reason, "held_in_ok": dual.held_in_ok, "held_out_ok": dual.held_out_ok,
        },
        "accepted": bool(dual.ok and best is not None),
        "note": "self_harness_mine_minimal_dual_regression",
    }
