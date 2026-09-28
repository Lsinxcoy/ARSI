"""Gates — Critic / regression / seesaw (AEGIS stage 4 + pathologies)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional, Sequence


@dataclass
class GateResult:
    ok: bool
    gate: str
    reason: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def critic_check(
    manifest_diff: str,
    summary: str = "",
    forbidden: Sequence[str] = (
        "answer_key",
        "gold_",
        "ground_truth",
        "secret_key",
        "ARSI_API_KEY",
        "nvapi-",
    ),
) -> GateResult:
    """Anti reward-hacking: reject edits that embed keys/answers or self-verify scores."""
    blob = f"{summary}\n{manifest_diff}".lower()
    for bad in forbidden:
        if bad.lower() in blob:
            return GateResult(False, "critic", f"forbidden_pattern:{bad}")
    if "score = 1" in blob or "success = true" in blob and "assert" not in blob:
        return GateResult(False, "critic", "hardcoded_success_claim")
    return GateResult(True, "critic", "ok")


def regression_check(
    before: dict,
    after: dict,
    eps: float = 1e-6,
) -> GateResult:
    """Catastrophic forgetting: any named metric that drops fails the gate.

    before/after: {metric: float, higher_is_better via key suffix optional}
    Keys ending with `_lower_better` treat decrease as improvement.
    """
    drops = []
    gains = []
    for k, v0 in (before or {}).items():
        if k not in (after or {}):
            continue
        try:
            a, b = float(v0), float(after[k])
        except Exception:
            continue
        lower_better = str(k).endswith("_lower_better")
        delta = b - a
        worse = (delta < -eps) if not lower_better else (delta > eps)
        better = (delta > eps) if not lower_better else (delta < -eps)
        if worse:
            drops.append({"metric": k, "before": a, "after": b, "delta": round(delta, 6)})
        if better:
            gains.append({"metric": k, "delta": round(delta, 6)})
    if drops:
        return GateResult(False, "regression", "metrics_regressed", {"drops": drops, "gains": gains})
    return GateResult(True, "regression", "ok", {"gains": gains})


def seesaw_check(
    cluster_before: dict,
    cluster_after: dict,
    eps: float = 1e-6,
) -> GateResult:
    """Seesaw: improve some task/cluster rates while regressing others.

    cluster_*: {cluster_id: success_rate}
    """
    gains, drops = [], []
    for k, v0 in (cluster_before or {}).items():
        if k not in (cluster_after or {}):
            continue
        try:
            a, b = float(v0), float(cluster_after[k])
        except Exception:
            continue
        if b - a > eps:
            gains.append({"cluster": k, "delta": round(b - a, 6)})
        elif a - b > eps:
            drops.append({"cluster": k, "delta": round(b - a, 6)})
    if gains and drops:
        return GateResult(
            False,
            "seesaw",
            "improve_some_regress_others_fork_candidate",
            {"gains": gains, "drops": drops},
        )
    if drops and not gains:
        return GateResult(False, "seesaw", "net_regression", {"drops": drops})
    return GateResult(True, "seesaw", "ok", {"gains": gains, "drops": drops})
