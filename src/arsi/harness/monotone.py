"""P-R7 V*≥V₀ generalization to harness manifests (Dream-RSI × RRSI).

Dream-RSI: current policy is always in the candidate set → V(m*) ≥ V(0).
RRSI noise floor only protects local S*. Rumination: ChangeManifest selection
must also be monotone — admitted best harness score never declines.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass
class MonotoneSelection:
    best_id: str
    best_score: float
    baseline_id: str
    baseline_score: float
    monotone_ok: bool
    candidates: list = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def select_manifest_monotone(
    baseline_id: str,
    baseline_score: float,
    candidates: Sequence[dict],
    score_key: str = "score",
    id_key: str = "manifest_id",
) -> MonotoneSelection:
    """V*≥V₀ for manifests: baseline always in set; pick max score.

    candidates: [{manifest_id, score, ...}, ...]
    """
    pool = [
        {"manifest_id": str(baseline_id), "score": float(baseline_score), "role": "baseline"}
    ]
    for c in candidates or []:
        mid = str((c or {}).get(id_key) or (c or {}).get("id") or "")
        sc = float((c or {}).get(score_key, float("-inf")) or float("-inf"))
        if mid and mid != str(baseline_id):
            pool.append({"manifest_id": mid, "score": sc, "role": "candidate", **{k: v for k, v in (c or {}).items() if k not in (id_key, score_key)}})
        elif mid == str(baseline_id):
            # allow re-stating baseline score
            pool[0]["score"] = max(pool[0]["score"], sc)

    best = max(pool, key=lambda x: x["score"])
    ok = best["score"] >= float(baseline_score) - 1e-12
    return MonotoneSelection(
        best_id=best["manifest_id"],
        best_score=best["score"],
        baseline_id=str(baseline_id),
        baseline_score=float(baseline_score),
        monotone_ok=ok,
        candidates=pool,
        reason="V_star_ge_V0" if ok else "candidate_below_baseline_rejected",
    )


def harness_v_star_ge_v0(
    admitted_best_history: Sequence[float],
    eps: float = 1e-9,
) -> dict:
    """History of admitted-best harness scores must be non-decreasing."""
    xs = [float(x) for x in (admitted_best_history or [])]
    if len(xs) < 2:
        return {"ok": True, "n": len(xs), "note": "insufficient_history"}
    drops = []
    for i in range(1, len(xs)):
        if xs[i] + eps < xs[i - 1]:
            drops.append({"i": i, "from": xs[i - 1], "to": xs[i]})
    return {
        "ok": not drops,
        "n": len(xs),
        "first": xs[0],
        "last": xs[-1],
        "drops": drops,
        "note": "harness_admitted_best_monotone" if not drops else "harness_best_regressed",
    }


def extend_admitted_history(
    history: Sequence[float],
    new_best: float,
    use_max: bool = True,
) -> list[float]:
    """Append only if it keeps the sequence monotone (max of prior best)."""
    xs = [float(x) for x in (history or [])]
    val = float(new_best)
    if use_max and xs:
        val = max(xs[-1], val)
    xs.append(val)
    return xs


@dataclass
class MonotoneLedger:
    """Running admitted-best for harness manifests (isomorphic to Dream-RSI)."""

    history: list[float] = field(default_factory=list)
    best_id: str = ""

    def observe(self, score: float, manifest_id: str = "") -> dict:
        before = self.history[-1] if self.history else float("-inf")
        self.history = extend_admitted_history(self.history, score)
        after = self.history[-1]
        if after >= before - 1e-12 and (manifest_id or after == score):
            if after == float(score) or not self.best_id:
                self.best_id = str(manifest_id or self.best_id)
        return {
            "monotone_ok": after >= before - 1e-12,
            "best": after,
            "best_id": self.best_id,
            "history_n": len(self.history),
        }

    def report(self) -> dict:
        r = harness_v_star_ge_v0(self.history)
        r["best_id"] = self.best_id
        return r
