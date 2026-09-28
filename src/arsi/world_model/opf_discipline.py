"""Statistical OPF discipline for ARSI introspective world (JEPA-Anything inspired).

Not neural OPF — measurement-only on Z_BLOCKS:
  J0 cross-block orthogonality (organ × live × pool × env)
  J1 factor activity (projected variance floor / dead-block alarm)
  J2 intervention → block response (who gets pushed)

Thin-data discipline: no learned projectors; diagnostic only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

from arsi.world_model.capability_flow import Z_BLOCKS, Z_KEYS


def _mean(xs: Sequence[float]) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    a = [float(x) for x in xs[:n]]
    b = [float(y) for y in ys[:n]]
    ma, mb = _mean(a), _mean(b)
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a) or 1e-12
    vb = sum((y - mb) ** 2 for y in b) or 1e-12
    r = num / ((va * vb) ** 0.5)
    if r != r:
        return 0.0
    return max(-1.0, min(1.0, r))


def _block_series(rows: Sequence[dict], block: str) -> list[float]:
    """Collapse a block to one activity series (mean of |Δ| across keys)."""
    keys = Z_BLOCKS.get(block) or ()
    out = []
    for r in rows or []:
        vals = [abs(float((r or {}).get(k) or 0.0)) for k in keys if (r or {}).get(k) is not None]
        out.append(_mean(vals) if vals else 0.0)
    return out


@dataclass
class OPFDisciplineReport:
    ok: bool
    orth_max_abs: float = 0.0
    orth_pairs: dict = field(default_factory=dict)
    activity: dict = field(default_factory=dict)
    dead_blocks: list = field(default_factory=list)
    intervention_response: dict = field(default_factory=dict)
    orth_alerts: list = field(default_factory=list)
    note: str = "statistical_opf_j0_j1_j2"

    def to_dict(self) -> dict:
        return asdict(self)


def cross_block_orthogonality(z_rows: Sequence[dict], thr: float = 0.85) -> dict:
    """J0: block mean-activity series should not be collinear (max |ρ| < thr)."""
    series = {b: _block_series(z_rows, b) for b in Z_BLOCKS}
    pairs = {}
    worst = 0.0
    names = list(series)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r = abs(_pearson(series[a], series[b]))
            pairs[f"{a}~{b}"] = round(r, 4)
            worst = max(worst, r)
    return {
        "orth_max_abs": round(worst, 4),
        "pairs": pairs,
        "ok": worst < thr,
        "thr": thr,
        "n_rows": len(list(z_rows or [])),
        "note": "j0_cross_block_orthogonality",
    }


def factor_activity(z_rows: Sequence[dict], min_var: float = 1e-6) -> dict:
    """J1: each block must retain non-trivial variance (else dead factor)."""
    act = {}
    dead = []
    for b in Z_BLOCKS:
        xs = _block_series(z_rows, b)
        if len(xs) < 2:
            act[b] = {"var": 0.0, "n": len(xs), "alive": False}
            dead.append(b)
            continue
        mu = _mean(xs)
        var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
        alive = var >= min_var
        act[b] = {"var": round(var, 8), "n": len(xs), "alive": alive}
        if not alive:
            dead.append(b)
    return {
        "activity": act,
        "dead_blocks": dead,
        "min_var": min_var,
        "ok": not dead,
        "note": "j1_factor_activity",
    }


def intervention_response(
    z_before: dict,
    z_after: dict,
    blocks: Optional[Sequence[str]] = None,
) -> dict:
    """J2: which blocks moved under intervention (|Δ| per block)."""
    use = list(blocks or Z_BLOCKS)
    resp = {}
    for b in use:
        keys = Z_BLOCKS.get(b) or ()
        deltas = []
        for k in keys:
            if k in z_before or k in z_after:
                deltas.append(abs(float(z_after.get(k, 0.0) or 0.0) - float(z_before.get(k, 0.0) or 0.0)))
        resp[b] = {"delta_l1": round(sum(deltas), 6), "n_keys": len(deltas)}
    ranked = sorted(resp.items(), key=lambda kv: -kv[1]["delta_l1"])
    return {
        "response": resp,
        "most_pushed": ranked[0][0] if ranked else "",
        "ranking": [b for b, _ in ranked],
        "note": "j2_intervention_block_response",
    }


def orthogonality_alerts(pairs: dict, thr: float = 0.85) -> list[str]:
    """Pairs above thr are capacity-coupled — OPF wants them on separate factors."""
    out = []
    for k, v in (pairs or {}).items():
        try:
            if float(v) >= thr:
                out.append(f"{k}={float(v):.3f}>={thr}")
        except Exception:
            continue
    return out


def opf_discipline(z_rows: Sequence[dict], z_before: Optional[dict] = None, z_after: Optional[dict] = None) -> OPFDisciplineReport:
    j0 = cross_block_orthogonality(z_rows)
    j1 = factor_activity(z_rows)
    j2 = intervention_response(z_before or {}, z_after or {}) if (z_before or z_after) else {"response": {}, "most_pushed": "", "ranking": []}
    ok = bool(j0.get("ok")) and bool(j1.get("ok"))
    return OPFDisciplineReport(
        ok=ok,
        orth_max_abs=float(j0.get("orth_max_abs") or 0.0),
        orth_pairs=dict(j0.get("pairs") or {}),
        activity=dict(j1.get("activity") or {}),
        dead_blocks=list(j1.get("dead_blocks") or []),
        intervention_response=j2,
        orth_alerts=orthogonality_alerts(j0.get("pairs") or {}),
    )
