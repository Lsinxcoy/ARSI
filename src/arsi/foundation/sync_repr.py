"""Synchronization representation — CTM-inspired temporal co-fluctuation.

Source: arXiv:2505.05522 Continuous Thought Machines (App. H recursive sync).
Not neural CTM training. Statistical pairwise temporal structure of ARSI dyn
features as a **second-order** complement to first-order velocity v(z).

Rules (ODEWorld + CTM):
- block-separated (organ / live / pool / env) — never mix tracks into one narrative
- multi-scale via exponential decay half-lives (short / med / long)
- dead = near-zero variance or isolated (max |rho| ~ 0) under enough samples
- diagnostic only — does not change scores, iron laws, or Dream-RSI selection
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

# re-export for callers
__all__ = [
    "HALF_LIVES_S",
    "PairSync",
    "action_conditioned_sync",
    "block_series_from_z_rows",
    "channel_stats",
    "corr_matrix",
    "dead_channels",
    "half_life_to_decay",
    "ctm_fidelity_check",
    "multi_scale_pair_sync",
    "pearson",
    "spectral_entropy_sync",
    "sync_report_from_z_rows",
    "top_pairs",
]

# CTM-inspired multi-scale half-lives (wall-clock seconds)
HALF_LIVES_S = {
    "short": 30.0,
    "med": 300.0,
    "long": 1800.0,
}

DEAD_VAR_EPS = 1e-10
DEAD_CORR_THR = 1e-3
MIN_SAMPLES_FOR_DEAD = 4
TOP_PAIR_DEFAULT = 5


def half_life_to_decay(half_life_s: float) -> float:
    """exp(-r * t_half) = 0.5  =>  r = ln2 / t_half."""
    return math.log(2.0) / max(float(half_life_s), 1e-6)


def ctm_fidelity_check(
    *,
    nlm_present: bool = True,
    sync_present: bool = True,
    diagnostic_only: bool = True,
    half_lives: Optional[dict] = None,
) -> dict:
    """CTM ablation lesson: NLM **and** sync must both be present; diagnostic-only."""
    hl = half_lives or HALF_LIVES_S
    scales_ok = set(hl.keys()) >= {"short", "med", "long"}
    both = bool(nlm_present and sync_present)
    ok = both and bool(diagnostic_only) and scales_ok
    return {
        "ok": ok,
        "nlm_present": nlm_present,
        "sync_present": sync_present,
        "both_required": both,
        "diagnostic_only": bool(diagnostic_only),
        "scales_ok": scales_ok,
        "note": "ctm_nlm_and_sync_together_app_h",
    }


class PairSync:
    """Recursive rescaled co-activation for one channel pair (CTM App. H).

    alpha^{t+1} = e^{-r dt} alpha^t + z_i z_j
    beta^{t+1}  = e^{-r dt} beta^t  + 1
    S_ij = alpha / sqrt(beta)
    """

    def __init__(self, half_life_s: float = 300.0):
        self.r = half_life_to_decay(half_life_s)
        self.alpha = 0.0
        self.beta = 0.0
        self.n = 0

    def update(self, zi: float, zj: float, dt: float = 1.0) -> None:
        decay = math.exp(-self.r * max(float(dt), 0.0))
        self.alpha = decay * self.alpha + float(zi) * float(zj)
        self.beta = decay * self.beta + 1.0
        self.n += 1

    def value(self) -> float:
        if self.beta <= 0.0:
            return 0.0
        return self.alpha / math.sqrt(self.beta)

    def to_dict(self) -> dict:
        return {
            "S": round(self.value(), 6),
            "alpha": round(self.alpha, 6),
            "beta": round(self.beta, 6),
            "n": self.n,
            "half_life_s": round(math.log(2.0) / self.r, 3) if self.r > 0 else None,
        }


def _mean_std(xs: Sequence[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    aa = [float(a[i]) for i in range(n)]
    bb = [float(b[i]) for i in range(n)]
    ma, sa = _mean_std(aa)
    mb, sb = _mean_std(bb)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    cov = sum((aa[i] - ma) * (bb[i] - mb) for i in range(n)) / n
    return max(-1.0, min(1.0, cov / (sa * sb)))


def corr_matrix(cols: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Pairwise Pearson over equal-length columns."""
    keys = list(cols.keys())
    out: dict[str, dict[str, float]] = {k: {} for k in keys}
    for i, ki in enumerate(keys):
        out[ki][ki] = 1.0 if _mean_std(cols[ki])[1] > 1e-12 else 0.0
        for kj in keys[i + 1 :]:
            r = pearson(cols[ki], cols[kj])
            out[ki][kj] = round(r, 6)
            out[kj][ki] = round(r, 6)
    return out


def spectral_entropy_sync(corr: dict[str, dict[str, float]]) -> dict:
    """Entropy of |eigenvalues| proxy via row L1 mass of |S| — cheap collapse alarm."""
    keys = list(corr.keys())
    if not keys:
        return {"entropy": 0.0, "collapse": True, "note": "empty"}
    masses = []
    for k in keys:
        s = sum(abs(corr[k].get(j, 0.0)) for j in keys)
        masses.append(s)
    total = sum(masses) + 1e-12
    p = [m / total for m in masses if m > 1e-12]
    if not p:
        return {"entropy": 0.0, "collapse": True, "dim": len(keys), "note": "zero_mass"}
    ent = -sum(pi * math.log(pi) for pi in p)
    max_ent = math.log(len(keys))
    return {
        "entropy": round(ent, 4),
        "max_entropy": round(max_ent, 4),
        "norm_entropy": round(ent / max_ent, 4) if max_ent > 0 else 0.0,
        "dim": len(keys),
        "collapse": bool(ent < 0.2 * max_ent),
        "note": "sync_row_l1_spectral_entropy_proxy",
    }


def top_pairs(
    corr: dict[str, dict[str, float]], k: int = TOP_PAIR_DEFAULT
) -> list[dict]:
    keys = list(corr.keys())
    pairs = []
    for i, ki in enumerate(keys):
        for kj in keys[i + 1 :]:
            r = float(corr[ki].get(kj, 0.0) or 0.0)
            pairs.append({"a": ki, "b": kj, "rho": round(r, 6)})
    pairs.sort(key=lambda p: abs(p["rho"]), reverse=True)
    return pairs[: max(0, int(k))]


def channel_stats(cols: dict[str, list[float]], n_min: int = MIN_SAMPLES_FOR_DEAD) -> dict:
    out = {}
    for k, xs in cols.items():
        m, s = _mean_std(xs)
        out[k] = {
            "n": len(xs),
            "mean": round(m, 6),
            "std": round(s, 6),
            "frozen": bool(len(xs) >= n_min and s < DEAD_VAR_EPS),
        }
    return out


def dead_channels(
    cols: dict[str, list[float]],
    corr: Optional[dict[str, dict[str, float]]] = None,
    n_min: int = MIN_SAMPLES_FOR_DEAD,
    corr_thr: float = DEAD_CORR_THR,
) -> list[str]:
    """Frozen (no variance) or isolated (no coupling) channels — CTM dead-neuron analog."""
    if corr is None:
        corr = corr_matrix(cols)
    stats = channel_stats(cols, n_min=n_min)
    dead = []
    for k, st in stats.items():
        if st["n"] < n_min:
            continue
        if st["frozen"]:
            dead.append(k)
            continue
        others = [abs(float(corr[k].get(j, 0.0) or 0.0)) for j in corr.get(k, {}) if j != k]
        if others and max(others) < corr_thr:
            dead.append(k)
    return dead


def block_series_from_z_rows(
    z_rows: Sequence[dict], blocks: dict[str, Sequence[str]]
) -> dict[str, dict[str, list[float]]]:
    """Split z maps into per-block columns."""
    out: dict[str, dict[str, list[float]]] = {}
    for bname, keys in blocks.items():
        cols: dict[str, list[float]] = {k: [] for k in keys}
        for z in z_rows:
            for k in keys:
                cols[k].append(float((z or {}).get(k, 0.0) or 0.0))
        out[bname] = cols
    return out


def multi_scale_pair_sync(
    cols_i: Sequence[float],
    cols_j: Sequence[float],
    dts: Optional[Sequence[float]] = None,
    half_lives: Optional[dict[str, float]] = None,
) -> dict[str, dict]:
    """S_ij at short/med/long time scales (CTM learnable r_ij → fixed grid first)."""
    half_lives = half_lives or HALF_LIVES_S
    n = min(len(cols_i), len(cols_j))
    trackers = {name: PairSync(hl) for name, hl in half_lives.items()}
    for t in range(n):
        dt = float(dts[t]) if dts and t < len(dts) else (1.0 if t == 0 else 1.0)
        # first sample: dt from 0
        if dts and t < len(dts):
            dt = max(float(dts[t]), 0.0)
        else:
            dt = 1.0
        for tr in trackers.values():
            tr.update(float(cols_i[t]), float(cols_j[t]), dt=dt)
    return {name: tr.to_dict() for name, tr in trackers.items()}


def sync_report_from_z_rows(
    z_rows: Sequence[dict],
    blocks: dict[str, Sequence[str]],
    dts: Optional[Sequence[float]] = None,
    actions: Optional[Sequence[str]] = None,
    top_k: int = TOP_PAIR_DEFAULT,
    window: int = 64,
) -> dict:
    """Full Sync-1 report: per-block corr, dead, top pairs, multi-scale, entropy."""
    rows = list(z_rows)[-max(2, int(window)) :]
    dt_win = list(dts or [])[-len(rows) :] if dts else None
    act_win = list(actions or [])[-len(rows) :] if actions else None
    series = block_series_from_z_rows(rows, blocks)
    report: dict = {
        "n_samples": len(rows),
        "window": min(len(rows), int(window)),
        "blocks": {},
        "note": "sync1_statistical_not_ctm_training",
    }
    all_dead: list[str] = []
    for bname, cols in series.items():
        if not cols:
            continue
        corr = corr_matrix(cols)
        dead = dead_channels(cols, corr=corr)
        all_dead.extend(dead)
        stats = channel_stats(cols)
        # multi-scale on organ block self-pairs + top pair only (cheap)
        pair_sync = {}
        keys = list(cols.keys())
        for i, ki in enumerate(keys):
            for kj in keys[i + 1 :]:
                pair_sync[f"{ki}~{kj}"] = multi_scale_pair_sync(
                    cols[ki], cols[kj], dts=dt_win
                )
        report["blocks"][bname] = {
            "keys": keys,
            "corr": corr,
            "stats": stats,
            "dead": dead,
            "top_pairs": top_pairs(corr, k=top_k),
            "entropy": spectral_entropy_sync(corr),
            "pair_sync_sample": {
                k: pair_sync[k] for k in list(pair_sync)[: min(6, len(pair_sync))]
            },
        }
    report["dead_channels"] = sorted(set(all_dead))
    report["dead_organs"] = sorted(
        set(all_dead) & set(blocks.get("organ") or ())
    )
    # whole-z entropy proxy
    flat_cols = {}
    for cols in series.values():
        flat_cols.update(cols)
    report["entropy_all"] = spectral_entropy_sync(corr_matrix(flat_cols)) if flat_cols else {}
    if act_win:
        report["by_action"] = action_conditioned_sync(rows, act_win, blocks)
    return report


def action_conditioned_sync(
    z_rows: Sequence[dict],
    actions: Sequence[str],
    blocks: dict[str, Sequence[str]],
    min_n: int = 3,
    top_k: int = 3,
) -> dict:
    """C1-3: S(a) — co-fluctuation stratified by causal action window.

    Complements v(z;a): first-order push vs *which organs move together* under an op.
    Actions with < min_n samples stay unmeasured (no invented coupling).
    """
    rows = list(z_rows)
    act_list = list(actions or [])
    acts = [(str(act_list[i]) if i < len(act_list) else "unknown") for i in range(len(rows))]
    by_act: dict[str, list[dict]] = {}
    for z, a in zip(rows, acts):
        by_act.setdefault(a or "unknown", []).append(z)
    out: dict = {"by_action": {}, "note": "action_conditioned_sync_S_a", "min_n": min_n}
    for a, subset in sorted(by_act.items()):
        if len(subset) < min_n:
            out["by_action"][a] = {"n": len(subset), "trusted": False, "note": "unmeasured"}
            continue
        series = block_series_from_z_rows(subset, blocks)
        block_rep = {}
        for bname, cols in series.items():
            corr = corr_matrix(cols)
            block_rep[bname] = {
                "top_pairs": top_pairs(corr, k=top_k),
                "dead": dead_channels(cols, corr=corr),
                "entropy": spectral_entropy_sync(corr),
            }
        out["by_action"][a] = {"n": len(subset), "trusted": True, "blocks": block_rep}
    return out
