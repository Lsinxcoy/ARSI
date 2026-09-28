"""Grader co-evolution (arXiv:2607.12790) — metrics as compositions of drawback detectors.

Stance: we rarely know what good is; given an output we can find drawbacks.
Clean verdict = no known drawback found — not certified correctness.

Validity lives in the **anchor**, not the lifecycle. Task score cannot validate
a self-evolved evaluator (a vacuous always-pass metric trains skills just as well).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

# verdicts: 1=drawback, 0=clean, None=abstain
DrawbackFn = Callable[[dict, object, dict], Optional[bool]]


@dataclass
class DrawbackOp:
    name: str
    kind: str = "static"  # static | execution | judge
    detects: str = ""
    fn: DrawbackFn = None  # type: ignore
    status: str = "active"  # active | shadow | retired
    marginal: float = 0.0

    def verdict(self, t: dict, y, c: dict) -> Optional[bool]:
        if self.fn is None:
            return None
        try:
            return self.fn(t, y, c)
        except Exception:
            return None


@dataclass
class MetricVerdict:
    verdict: str  # pass | fail | abstain
    named_drawbacks: list[str] = field(default_factory=list)
    expr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _combine_any(vs: list[Optional[bool]]) -> Optional[bool]:
    if any(v is True for v in vs):
        return True
    if all(v is False for v in vs):
        return False
    return None


def _combine_all(vs: list[Optional[bool]]) -> Optional[bool]:
    known = [v for v in vs if v is not None]
    if any(v is False for v in known):
        return False
    if len(known) == len(vs) and known and all(known):
        return True
    return None


class MetricExpression:
    """Tree of drawback ops: any / all / vote. Root finds no drawback ⇒ pass."""

    def __init__(self, ops: Sequence[DrawbackOp], mode: str = "any", k_of_n: int = 1):
        self.ops = list(ops or [])
        self.mode = mode
        self.k_of_n = int(k_of_n)

    def evaluate(self, t: dict, y, c: Optional[dict] = None) -> MetricVerdict:
        c = c or {}
        named = []
        votes: list[Optional[bool]] = []
        for op in self.ops:
            if op.status == "retired":
                continue
            v = op.verdict(t, y, c)
            votes.append(v)
            if v is True:
                named.append(f"{op.name}:{op.detects or op.kind}")
        known = [v for v in votes if v is not None]
        if not known:
            return MetricVerdict("abstain", [], self.expr_str())
        if self.mode == "all":
            raw = _combine_all(votes)
        elif self.mode == "vote":
            raw = sum(1 for v in known if v) >= min(self.k_of_n, len(known))
        else:
            raw = _combine_any(votes)
        if raw is None:
            return MetricVerdict("abstain", named, self.expr_str())
        return MetricVerdict("fail" if raw else "pass", named, self.expr_str())

    def expr_str(self) -> str:
        return f"({self.mode} {' '.join(o.name for o in self.ops)})"


def validity_gate(v: MetricVerdict) -> bool:
    """Drop vacuous metrics: all-pass / all-fail / all-abstain on a batch is invalid."""
    return v.verdict in ("pass", "fail")


def is_vacuous_metric(verdicts: Sequence[MetricVerdict]) -> bool:
    kinds = {v.verdict for v in verdicts or []}
    return kinds.issubset({"pass"}) or kinds.issubset({"fail"}) or kinds.issubset({"abstain"})


def recall_weighted_adreement(
    pred: Sequence[str],
    labels: Sequence[str],
    w_fail: float = 2.0,
    w_pass: float = 1.0,
) -> float:
    """A_dev: missed drawback costs 2× a false alarm (paper)."""
    n = min(len(pred), len(labels))
    if n == 0:
        return 0.0
    r_fail_num = r_fail_den = r_pass_num = r_pass_den = 0
    for i in range(n):
        p, g = pred[i], labels[i]
        if g == "fail":
            r_fail_den += 1
            if p == "fail":
                r_fail_num += 1
        elif g == "pass":
            r_pass_den += 1
            if p == "pass":
                r_pass_num += 1
    r_fail = (r_fail_num / r_fail_den) if r_fail_den else 1.0
    r_pass = (r_pass_num / r_pass_den) if r_pass_den else 1.0
    return (w_fail * r_fail + w_pass * r_pass) / (w_fail + w_pass)


def metric_fitness(
    a_dev: float,
    a_train: float,
    complexity: float,
    w: float = 0.5,
    lam: float = 0.01,
) -> float:
    """S(e) = A_dev · A_train^w − λ·C(e)   (Eq.2)"""
    return float(a_dev) * (float(a_train) ** float(w)) - float(lam) * float(complexity)


def birth_gate(
    fires_on_cluster: int,
    cluster_size: int,
    false_fires_on_good: int,
) -> bool:
    """Fire on ≥ half its cluster and stay clean on known-good."""
    if cluster_size <= 0:
        return False
    return fires_on_cluster * 2 >= cluster_size and false_fires_on_good == 0


def leave_one_out_marginal(
    with_op: float,
    without_op: float,
) -> float:
    return float(with_op) - float(without_op)


def curate_pool(
    ops: list[DrawbackOp],
    marginals: dict[str, float],
    grace_left: dict[str, int],
) -> list[DrawbackOp]:
    """Retire non-positive marginals after grace; promote shadows that raise A_dev."""
    for op in ops:
        m = float(marginals.get(op.name, 0.0))
        op.marginal = m
        g = int(grace_left.get(op.name, 0))
        if op.status == "shadow" and m > 0:
            op.status = "active"
        elif op.status == "active" and m <= 0 and g <= 0:
            op.status = "retired"
    return ops


def reliability_weighted_consensus(
    votes,
    kappa: float = 0.5,
):
    """Grader: vote scaled by 1+kappa*max(0, marginal) so anchor-proven ops lead."""
    pos = 0.0
    neg = 0.0
    for _name, v, marg in votes or []:
        if v is None:
            continue
        w = 1.0 + float(kappa) * max(0.0, float(marg))
        if v:
            pos += w
        else:
            neg += w
    if pos == 0.0 and neg == 0.0:
        return None
    return pos > neg


def grader_fidelity_check(
    *,
    clean_is_not_correct: bool = True,
    validity_on_anchor: bool = True,
    task_score_validates_evaluator: bool = False,
    has_lifecycle: bool = True,
) -> dict:
    """Grader invariants: clean≠correct; validity on anchor; task score cannot validate evaluator."""
    ok = (
        bool(clean_is_not_correct)
        and bool(validity_on_anchor)
        and not task_score_validates_evaluator
        and bool(has_lifecycle)
    )
    return {
        "ok": ok,
        "clean_is_not_correct": bool(clean_is_not_correct),
        "validity_on_anchor": bool(validity_on_anchor),
        "task_score_validates_evaluator": bool(task_score_validates_evaluator),
        "has_lifecycle": bool(has_lifecycle),
        "note": "grader_negative_epistemology_2607_12790",
    }
