"""AIDE² evaluation protocol — pub/priv signal split + hidden grade (arXiv:2609.26457).

Inner loop optimizes r_pub; outer selection uses g = mean r_priv on **hidden**
held-out data the rewritten agent never observes. Fixed budget ⇒ gains are
algorithm, not extra compute.

ARSI mapping:
  selection tasks  → host evolve slice (may drive Digester/Evolver)
  private grade    → sealed_tasks / I6 / held-out host slice  (NEVER in evolve loss)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence


@dataclass
class TaskSplit:
    selection: list[str] = field(default_factory=list)
    private: list[str] = field(default_factory=list)
    ood: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def is_selection(self, task_id: str) -> bool:
        return task_id in set(self.selection)

    def is_private(self, task_id: str) -> bool:
        return task_id in set(self.private) or task_id in set(self.ood)


DEFAULT_ARSI_SPLIT_NOTE = (
    "sealed_tasks + i6_gates + eval_loop must live in private/blacklist — never selection"
)


def split_tasks(
    task_ids: Iterable[str],
    *,
    private_ratio: float = 0.2,
    blacklist: Sequence[str] = (),
    ood: Sequence[str] = (),
    seed: int = 17,
) -> TaskSplit:
    """Deterministic split; blacklist always private; ood is external (never selection)."""
    ids = sorted({str(t) for t in task_ids or [] if t})
    bl = {str(b) for b in blacklist or ()}
    ood_set = {str(t) for t in (ood or ())}
    rest = [t for t in ids if t not in bl and t not in ood_set]
    # stable pseudo-shuffle
    def key(t: str) -> int:
        h = seed
        for ch in t:
            h = (h * 131 + ord(ch)) & 0xFFFFFFFF
        return h

    rest_sorted = sorted(rest, key=key)
    n_priv = max(1, int(round(len(rest_sorted) * float(private_ratio)))) if rest_sorted else 0
    private = rest_sorted[:n_priv]
    selection = rest_sorted[n_priv:]
    return TaskSplit(
        selection=selection,
        private=private,
        ood=sorted(ood_set),
        blacklist=sorted(bl),
    )


def private_grade(priv_scores: Sequence[float]) -> float:
    """g(a) = mean of private held-out scores (Eq.2)."""
    xs = [float(x) for x in (priv_scores or [])]
    return round(sum(xs) / len(xs), 6) if xs else 0.0


def accept_rewrite(
    *,
    public_gain: float,
    private_gain: float,
    budget_delta: float = 0.0,
    require_private: bool = True,
) -> dict:
    """AIDE²: keep rewrite only if **private** grade improves.

    A candidate that wins on r_pub but loses on r_priv is rejected
    (~1/4 of rejects in the paper were public-winners).
    """
    if require_private and private_gain <= 0:
        return {
            "accepted": False,
            "reason": "private_grade_not_improved",
            "public_gain": public_gain,
            "private_gain": private_gain,
        }
    if budget_delta > 0:
        # fixed budget discipline — extra spend must not be the source of gain
        return {
            "accepted": False,
            "reason": "budget_increase",
            "public_gain": public_gain,
            "private_gain": private_gain,
            "budget_delta": budget_delta,
        }
    return {
        "accepted": True,
        "reason": "private_grade_improved",
        "public_gain": public_gain,
        "private_gain": private_gain,
    }


def first_order_vs_second_order(
    selection_gain: float,
    holdout_gain: float,
) -> str:
    """AIDE²: 1st-order = selection; 2nd-order = external held-out transfer."""
    if selection_gain > 0 and holdout_gain > 0:
        return "transfers"
    if selection_gain > 0 and holdout_gain <= 0:
        return "selection_only_overfit"
    if selection_gain <= 0 and holdout_gain > 0:
        return "holdout_only_noise"
    return "no_gain"


def outer_loop_select(
    candidates: Sequence[dict],
    *,
    incumbent_id: str = "",
    key_g: str = "private_grade",
    key_pub: str = "public_gain",
) -> dict:
    """AIDE² outer loop: a* = argmax g(a) among accepted rewrites.

    Public-win / private-lose candidates must already be rejected upstream
    (accept_rewrite). This is the argmax over **hidden** grade only.
    """
    rows = []
    for c in candidates or []:
        cid = str((c or {}).get("id") or (c or {}).get("candidate_id") or "")
        g = float((c or {}).get(key_g, float("-inf")))
        pub = float((c or {}).get(key_pub, 0.0) or 0.0)
        rows.append({"id": cid, "g": g, "public_gain": pub})
    if not rows:
        return {"best_id": incumbent_id, "best_g": None, "n": 0, "rule": "argmax_private_grade"}
    best = max(rows, key=lambda r: r["g"])
    # if incumbent listed, never pick below it without explicit higher g
    if incumbent_id:
        for r in rows:
            if r["id"] == incumbent_id and r["g"] >= best["g"] - 1e-12:
                best = r
                break
    return {
        "best_id": best["id"] or incumbent_id,
        "best_g": best["g"],
        "n": len(rows),
        "candidates": rows,
        "rule": "argmax_private_grade",
    }
