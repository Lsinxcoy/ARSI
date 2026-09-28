"""Recuris R2 — component-scoped skill patches + statistical validation gate.

Gate is not a good/bad filter; it refuses to commit when evidence cannot
separate from zero (paper case: REJECT despite source-task repair).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from arsi.harness.skill_trace import (
    COMPONENT_EXPERIENTIAL,
    COMPONENT_HARNESS,
    COMPONENT_WORKING,
)


@dataclass
class SkillPatch:
    patch_id: str
    component: str
    skill_id: str
    skill_title: str
    skill_body: str
    source_tasks: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateVerdict:
    accepted: bool
    reason: str
    net_pp: float = 0.0
    ci: tuple = (0.0, 0.0)
    min_n: int = 0
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ci"] = list(self.ci)
        return d


def propose_patch(
    component: str,
    *,
    skill_id: str,
    skill_title: str,
    skill_body: str,
    source_tasks: Sequence[str] = (),
    implicated_only: Sequence[str] = (),
) -> Optional[SkillPatch]:
    """Patch only the diagnosed component; harness faults get no patch."""
    if component == COMPONENT_HARNESS:
        return None
    if implicated_only and component not in set(implicated_only):
        return None
    return SkillPatch(
        patch_id=f"SP-{skill_id}",
        component=component,
        skill_id=skill_id,
        skill_title=skill_title,
        skill_body=skill_body,
        source_tasks=list(source_tasks or []),
        note="recuris_component_scoped_patch",
    )


def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in (xs or [])]
    return sum(xs) / len(xs) if xs else 0.0


def validation_gate(
    *,
    source_before: Sequence[float],
    source_after: Sequence[float],
    dev_before: Sequence[float],
    dev_after: Sequence[float],
    min_n: int = 8,
    z: float = 1.96,
) -> GateVerdict:
    """Accept only if source improves AND dev does not regress, with CI power.

    If n < min_n or CI of dev delta contains 0 with |net| small → REJECT wait.
    """
    sb, sa = list(source_before or []), list(source_after or [])
    db, da = list(dev_before or []), list(dev_after or [])
    n_dev = min(len(db), len(da))
    n_src = min(len(sb), len(sa))
    if n_src == 0:
        return GateVerdict(False, "no_source_scores", min_n=min_n)
    src_delta = _mean(sa[:n_src]) - _mean(sb[:n_src])
    if src_delta <= 0:
        return GateVerdict(False, "source_not_repaired", round(src_delta, 6), min_n=min_n)

    if n_dev < min_n:
        return GateVerdict(
            False,
            "wait_more_evidence_dev_n_too_small",
            round(src_delta, 6),
            (float("nan"), float("nan")),
            min_n=min_n,
            detail={"n_dev": n_dev, "src_delta": round(src_delta, 6)},
        )

    deltas = [float(a) - float(b) for b, a in zip(db[:n_dev], da[:n_dev])]
    mu = _mean(deltas)
    var = sum((d - mu) ** 2 for d in deltas) / max(1, n_dev - 1)
    se = math.sqrt(max(var, 0.0) / n_dev)
    lo, hi = mu - z * se, mu + z * se
    # dev must not regress; CI should not be dominated by possible regression
    if mu < -1e-9:
        return GateVerdict(
            False,
            "dev_regressed",
            round(mu, 6),
            (round(lo, 6), round(hi, 6)),
            min_n=min_n,
        )
    if lo < 0 and abs(mu) < 0.05:
        return GateVerdict(
            False,
            "ci_cannot_separate_from_zero",
            round(mu, 6),
            (round(lo, 6), round(hi, 6)),
            min_n=min_n,
            detail={"note": "conservative_reject_like_paper"},
        )
    return GateVerdict(
        True,
        "accepted_dev_stable",
        round(mu, 6),
        (round(lo, 6), round(hi, 6)),
        min_n=min_n,
        detail={"src_delta": round(src_delta, 6), "n_dev": n_dev},
    )
