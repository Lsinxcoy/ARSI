"""Capability Velocity Field — physical-time dyn flow (arXiv:2607.27924 PT-Flow adapted).

Not a video ODE. Statistical first-order flow over ARSI dyn state z:
  z = compact dyn features (organ/mem/eta/live/pool/D_T)
  static context c = host identity / laws / beta (time-invariant within a term)
  v_gt = Δz / Δt_wall   (physical wall-clock seconds, not tick index)
  v_hat = EWMA / ridge velocity field fit on recent (z, v_gt) pairs
  integrate(z0, c, T) → subgoal z_τ for planning (super-resolution)

Rules (ODEWorld lessons):
- static features must NOT enter velocity fit (decoupling)
- supervise v_hat against v_gt directly (first-order), not multi-step endpoint consistency
- keep live / pool / D_T on separate vector blocks — never mix tracks into one narrative
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from arsi.foundation.rankme import centered_effective_rank
from arsi.foundation.sync_memory import SyncWorkingMemory
from arsi.foundation.sync_repr import sync_report_from_z_rows
from arsi.world_model.cognitive_map import CognitiveMap
from arsi.world_model.kernel_flow import KernelVelocityField
from arsi.world_model.nlm_filter import NLMBank

logger = logging.getLogger(__name__)

# Stable dyn feature order (blocks must stay separable)
Z_KEYS = (
    # organ / self (IWM)
    "self_trust",
    "memory_trust",
    "behavior_predictor_trust",
    "eta",
    # live track (never pool)
    "live_last",
    "live_ema",
    # pool track (never live)
    "pool_last",
    "pool_ema",
    # environment difficulty track
    "d_t_mean",
    "d_t_spread",
    "el_pass_mean",
)

Z_BLOCKS = {
    "organ": ("self_trust", "memory_trust", "behavior_predictor_trust", "eta"),
    "live": ("live_last", "live_ema"),
    "pool": ("pool_last", "pool_ema"),
    "env": ("d_t_mean", "d_t_spread", "el_pass_mean"),
}

STATIC_KEYS = ("beta", "grid_branch", "pool_size", "host_measured_frac")

# JEPA-Anything OPF decoupling policy (update rates must differ)
DECOUPLE_POLICY = {
    "organ": "iwm_calibrator_events",  # self_trust / memory / predictor
    "live": "live_capability_eval_only",
    "pool": "dream_rsi_deploy_only",  # _pool_scores append site
    "env": "env_evolution_health",
    "note": "blocks must not share one ingest cadence (J0 orthogonality)",
}


def flow_fidelity_check(
    z_keys: Sequence[str] = Z_KEYS,
    static_keys: Sequence[str] = STATIC_KEYS,
    v_is_first_order: bool = True,
    mixed_track_scores: bool = False,
) -> dict:
    """ODEWorld fidelity invariants (engineering, not a proof).

    1) static keys must not appear in dyn z
    2) supervision is first-order only
    3) live/pool/D_T never mixed into one score track
    """
    zk = set(str(k) for k in (z_keys or ()))
    sk = set(str(k) for k in (static_keys or ()))
    overlap = sorted(zk & sk)
    blocks_ok = not mixed_track_scores
    # block disjointness
    seen: set = set()
    block_overlap = []
    for name, keys in Z_BLOCKS.items():
        for k in keys:
            if k in seen:
                block_overlap.append(k)
            seen.add(k)
    ok = (not overlap) and bool(v_is_first_order) and blocks_ok and (not block_overlap)
    report = {
        "ok": ok,
        "static_in_z": overlap,
        "first_order_only": bool(v_is_first_order),
        "tracks_mixed": bool(mixed_track_scores),
        "block_overlap": block_overlap,
        "note": "ode_world_pt_flow_fidelity",
    }
    # JEPA-Anything statistical OPF (J0 orthogonality when rows provided)
    try:
        if z_keys is Z_KEYS and static_keys is STATIC_KEYS:
            from arsi.world_model.opf_discipline import cross_block_orthogonality

            report["opf_j0"] = {}
    except Exception:
        pass
    return report


@dataclass
class FlowSample:
    t_wall: float
    z: list[float]
    v_gt: list[float] = field(default_factory=list)
    dt: float = 0.0
    static: dict = field(default_factory=dict)
    note: str = ""
    action: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlowObservation:
    t_wall: float = 0.0
    dt: float = 0.0
    z: dict = field(default_factory=dict)
    v_gt: dict = field(default_factory=dict)
    v_hat: dict = field(default_factory=dict)
    v_error: dict = field(default_factory=dict)
    negative_organs: list[str] = field(default_factory=list)
    rankme: dict = field(default_factory=dict)
    integrated: dict = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def encode_dyn_state(stats: dict) -> dict:
    """Extract dyn z from ARSI get_stats() / health-like dict. Static context excluded."""
    s = stats or {}
    iwm = s.get("iwm") or {}
    if isinstance(iwm, dict) and isinstance(iwm.get("iwm"), dict):
        iwm = iwm["iwm"]
    iwm = iwm if isinstance(iwm, dict) else {}
    organ = iwm.get("organ_trust") or iwm.get("organs") or {}
    if not isinstance(organ, dict):
        organ = {}
    env = s.get("env_evolution") or {}
    diff = (env.get("difficulty") or {}) if isinstance(env, dict) else {}
    el = (env.get("el") or {}) if isinstance(env, dict) else {}
    el_lins = el.get("lineages") or {}
    pass_rates = []
    if isinstance(el_lins, dict):
        for st in el_lins.values():
            if isinstance(st, dict) and st.get("pass_rate") is not None:
                pass_rates.append(float(st.get("pass_rate") or 0.0))
    live_hist = s.get("_live_capability_scores") or s.get("live_history") or []
    pool_hist = s.get("_pool_scores") or s.get("pool_history") or []

    def _last(xs):
        return float(xs[-1]) if xs else 0.0

    def _ema(xs, alpha=0.3):
        if not xs:
            return 0.0
        e = float(xs[0])
        for x in xs[1:]:
            e = alpha * float(x) + (1 - alpha) * e
        return float(e)

    z = {
        "self_trust": float(iwm.get("self_trust") or 0.0),
        "memory_trust": float(iwm.get("memory_trust") or organ.get("memory") or 0.0),
        "behavior_predictor_trust": float(
            iwm.get("behavior_predictor_accuracy")
            or iwm.get("layer1_holdout")
            or organ.get("behavior_predictor")
            or 0.0
        ),
        "eta": float(s.get("eta") or 0.0),
        "live_last": _last(list(live_hist)),
        "live_ema": _ema(list(live_hist)),
        "pool_last": _last(list(pool_hist)),
        "pool_ema": _ema(list(pool_hist)),
        "d_t_mean": float(diff.get("d_t_mean") or 0.0),
        "d_t_spread": float(diff.get("d_t_spread") or 0.0),
        "el_pass_mean": (sum(pass_rates) / len(pass_rates)) if pass_rates else 0.0,
    }
    return {k: round(float(z.get(k, 0.0)), 6) for k in Z_KEYS}


def encode_static_context(stats: dict) -> dict:
    """Time-invariant-ish context c — must NOT enter velocity fit (ODEWorld decoupling)."""
    s = stats or {}
    env = s.get("env_evolution") or {}
    ma = s.get("multi_agent") or {}
    agents = ma.get("agents") or {}
    measured = 0
    total = 0
    if isinstance(agents, dict):
        for a in agents.values():
            total += 1
            if isinstance(a, dict) and (a.get("tasks_completed") or 0) > 0:
                measured += 1
    return {
        "beta": float(s.get("beta") or 0.0),
        "grid_branch": float((s.get("grid_plan") or {}).get("branch_count") or 0.0),
        "pool_size": float(s.get("world_pool_size") or 0.0),
        "host_measured_frac": (measured / total) if total else 0.0,
    }


def _finite_clamped(x, lo: float = -1e3, hi: float = 1e3) -> float:
    """P0-3: kill Inf/NaN before they poison the online field."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v or v in (float("inf"), float("-inf")):  # NaN / Inf
        return 0.0
    return max(lo, min(hi, v))


def velocity_gt(z_prev: Sequence[float], z_now: Sequence[float], dt: float, min_dt: float = 1e-3) -> list[float]:
    """First-order finite difference over physical time (Savitzky–Golay light: 2-pt)."""
    dt_eff = max(_finite_clamped(dt, 1e-3, 1e6), min_dt)
    return [
        _finite_clamped((_finite_clamped(b) - _finite_clamped(a)) / dt_eff, -1e3, 1e3)
        for a, b in zip(z_prev, z_now)
    ]


class VelocityField:
    """Lightweight v_hat: ridge regression v ≈ W·[z; z²] + b (diagonal-ish EWMA fallback).

    Direct first-order supervision only — fit against v_gt, never multi-step consistency loss.
    """

    def __init__(self, n_dims: int = len(Z_KEYS), lr: float = 0.15, l2: float = 1e-3):
        self.n = n_dims
        self.lr = lr
        self.l2 = l2
        # diagonal online fit: v_i ≈ a_i * z_i + b_i  (compact, underdetermined-data safe)
        self.a = [0.0] * n_dims
        self.b = [0.0] * n_dims
        self.se = [0.0] * n_dims
        self.n_updates = 0

    def fit_sample(self, z: Sequence[float], v_gt: Sequence[float]) -> float:
        """One online gradient step on ||v_hat - v_gt||² + λ||a||² (first-order loss)."""
        zc = [_finite_clamped(z[i] if i < len(z) else 0.0) for i in range(self.n)]
        v_gt_c = [_finite_clamped(v_gt[i] if i < len(v_gt) else 0.0) for i in range(self.n)]
        v_hat = self.predict_vec(zc)
        err = 0.0
        for i in range(self.n):
            e = _finite_clamped(v_hat[i] - v_gt_c[i], -1e3, 1e3)
            err += e * e
            gi = 2.0 * e
            self.a[i] = _finite_clamped(self.a[i] - self.lr * (gi * zc[i] + self.l2 * self.a[i]), -1e3, 1e3)
            self.b[i] = _finite_clamped(self.b[i] - self.lr * gi, -1e3, 1e3)
            self.se[i] = 0.9 * self.se[i] + 0.1 * e * e
        self.n_updates += 1
        return err / max(1, self.n)

    def predict_vec(self, z: Sequence[float]) -> list[float]:
        out = []
        for i in range(self.n):
            zi = _finite_clamped(z[i] if i < len(z) else 0.0)
            out.append(_finite_clamped(self.a[i] * zi + self.b[i], -1e3, 1e3))
        return out

    def predict(self, z_map: dict) -> dict:
        vec = [float(z_map.get(k, 0.0) or 0.0) for k in Z_KEYS]
        v = self.predict_vec(vec)
        return {k: round(v[i], 6) for i, k in enumerate(Z_KEYS)}

    def mse(self) -> float:
        v = sum(_finite_clamped(s, 0.0, 1e6) for s in self.se) / max(1, self.n)
        return v if v == v else 0.0

    def to_dict(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "mse_ewma": round(self.mse(), 8),
            "a": [round(_finite_clamped(x), 5) for x in self.a],
            "b": [round(_finite_clamped(x), 5) for x in self.b],
        }


# P2-8: action vocabulary (ARSI empowerment / host ops)
ACTION_POOL = (
    "learn",
    "remember",
    "dream",
    "maintain",
    "evolve",
    "empower",
    "repair",
    "unknown",
)


def canon_action(action: Optional[str]) -> str:
    a = (action or "unknown").strip().lower()
    if a in ACTION_POOL:
        return a
    for known in ACTION_POOL:
        if a.startswith(known):
            return known
    return "unknown"


class ActionConditionedVelocityField(VelocityField):
    """v(z, t; c, a) ≈ global(z) + residual_a(z)  (diagonal residual, sparse-data safe).

    Action-specific residual only after min_action_samples; else fall back to global.
    Direct first-order supervision only (same as VelocityField).
    """

    def __init__(
        self,
        n_dims: int = len(Z_KEYS),
        lr: float = 0.15,
        l2: float = 1e-3,
        min_action_samples: int = 3,
        residual_lr: float = 0.12,
    ):
        super().__init__(n_dims=n_dims, lr=lr, l2=l2)
        self.min_action_samples = int(min_action_samples)
        self.residual_lr = float(residual_lr)
        self.action_a: dict[str, list[float]] = {}
        self.action_b: dict[str, list[float]] = {}
        self.action_n: dict[str, int] = {}
        self.last_action = "unknown"

    def _action_slot(self, action: str) -> tuple[list[float], list[float]]:
        if action not in self.action_a:
            self.action_a[action] = [0.0] * self.n
            self.action_b[action] = [0.0] * self.n
            self.action_n.setdefault(action, 0)
        return self.action_a[action], self.action_b[action]

    def fit_sample(
        self,
        z: Sequence[float],
        v_gt: Sequence[float],
        action: str = "unknown",
    ) -> float:
        """Fit global field, then action residual on first-order error."""
        action = canon_action(action)
        self.last_action = action
        base_err = super().fit_sample(z, v_gt)
        v_base = super().predict_vec(z)
        aa, ab = self._action_slot(action)
        for i in range(self.n):
            zi = _finite_clamped(z[i] if i < len(z) else 0.0)
            vgi = _finite_clamped(v_gt[i] if i < len(v_gt) else 0.0)
            e = _finite_clamped((v_base[i] + aa[i] * zi + ab[i]) - vgi, -1e3, 1e3)
            gi = 2.0 * e
            aa[i] = _finite_clamped(aa[i] - self.residual_lr * (gi * zi + self.l2 * aa[i]), -1e3, 1e3)
            ab[i] = _finite_clamped(ab[i] - self.residual_lr * gi, -1e3, 1e3)
        self.action_n[action] = self.action_n.get(action, 0) + 1
        return base_err

    def predict_vec(self, z: Sequence[float], action: Optional[str] = None) -> list[float]:
        base = super().predict_vec(z)
        act = canon_action(action if action is not None else self.last_action)
        if self.action_n.get(act, 0) < self.min_action_samples:
            return base
        aa, ab = self.action_slot_or_zero(act)
        return [
            _finite_clamped(base[i] + aa[i] * _finite_clamped(z[i] if i < len(z) else 0.0) + ab[i], -1e3, 1e3)
            for i in range(self.n)
        ]

    def action_slot_or_zero(self, action: str) -> tuple[list[float], list[float]]:
        aa = self.action_a.get(action)
        ab = self.action_b.get(action)
        if aa is None or ab is None:
            return [0.0] * self.n, [0.0] * self.n
        return aa, ab

    def predict(self, z_map: dict, action: Optional[str] = None) -> dict:
        vec = [float(z_map.get(k, 0.0) or 0.0) for k in Z_KEYS]
        v = self.predict_vec(vec, action=action)
        return {k: round(v[i], 6) for i, k in enumerate(Z_KEYS)}

    def action_support(self) -> dict:
        return {
            a: {"n": int(self.action_n.get(a, 0)), "trusted": self.action_n.get(a, 0) >= self.min_action_samples}
            for a in sorted(self.action_n)
        }

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["min_action_samples"] = self.min_action_samples
        d["last_action"] = self.last_action
        d["action_support"] = self.action_support()
        d["action_a"] = {a: [round(x, 5) for x in v] for a, v in self.action_a.items()}
        d["action_b"] = {a: [round(x, 5) for x in v] for a, v in self.action_b.items()}
        d["action_n"] = dict(self.action_n)
        return d


def _field_predict(field: VelocityField, z: dict, action: Optional[str] = None) -> dict:
    """Predict v_hat; pass action only if the field accepts it (P2-8)."""
    if action is None:
        return field.predict(z)
    try:
        return field.predict(z, action=action)
    except TypeError:
        return field.predict(z)


def integrate(z0: dict, field: VelocityField, horizon_s: float = 300.0, steps: int = 4, action: Optional[str] = None) -> dict:
    """Euler integrate v_hat from z0 over physical horizon → subgoal z_τ (super-res)."""
    z = {k: float(z0.get(k, 0.0) or 0.0) for k in Z_KEYS}
    if horizon_s <= 0 or steps <= 0:
        return {
            "z_goal": z,
            "horizon_s": 0.0,
            "steps": 0,
            "action": canon_action(action) if action is not None else None,
        }
    dt = float(horizon_s) / float(steps)
    path = [{**z}]
    for _ in range(steps):
        v = _field_predict(field, z, action=action)
        z = {k: z[k] + float(v.get(k, 0.0)) * dt for k in Z_KEYS}
        path.append({**z})
    return {
        "z_goal": {k: round(z[k], 6) for k in Z_KEYS},
        "path": [{k: round(p[k], 6) for k in Z_KEYS} for p in path],
        "horizon_s": float(horizon_s),
        "steps": int(steps),
        "action": canon_action(action) if action is not None else None,
        "note": "euler_super_resolution_subgoal",
    }


def negative_organs(v_hat: dict, thr: float = -1e-4) -> list[str]:
    """Organs whose trust-like components are decreasing — IWM focus candidates."""
    out = []
    for key in Z_BLOCKS["organ"]:
        if float(v_hat.get(key, 0.0) or 0.0) < thr:
            out.append(key)
    return out


def integrate_backward(
    z_fail: dict,
    field: VelocityField,
    lookback_s: float = 300.0,
    steps: int = 4,
    action: Optional[str] = None,
) -> dict:
    """Euler integrate −v_hat from failure state → z_pre (ODEWorld backward prediction).

    Reconstructs pre-failure dyn state under current velocity field.
    Marked as model reconstruction — not a substitute for observed history.
    action conditions which v(z; a) is used (P2-8).
    """
    z = {k: float(z_fail.get(k, 0.0) or 0.0) for k in Z_KEYS}
    if lookback_s <= 0 or steps <= 0:
        return {
            "z_pre": z,
            "lookback_s": 0.0,
            "steps": 0,
            "note": "empty_reverse",
            "action": canon_action(action) if action is not None else None,
            "evidence_status": "model_reconstruction",
        }
    dt = float(lookback_s) / float(steps)
    path = [{**z}]
    for _ in range(steps):
        v = _field_predict(field, z, action=action)
        # reverse: z_pre ≈ z − v·dt
        z = {k: z[k] - float(v.get(k, 0.0)) * dt for k in Z_KEYS}
        path.append({**z})
    return {
        "z_pre": {k: round(z[k], 6) for k in Z_KEYS},
        "path": [{k: round(p[k], 6) for k in Z_KEYS} for p in path],
        "lookback_s": float(lookback_s),
        "steps": int(steps),
        "action": canon_action(action) if action is not None else None,
        "note": "euler_reverse_integrate_z_pre",
        "evidence_status": "model_reconstruction",  # not observed — unverified until receipt
    }


def pre_failure_organs(
    z_pre: dict,
    z_fail: dict,
    thr: float = 1e-4,
) -> list[str]:
    """Organs that were already higher before failure than at failure (degraded)."""
    out = []
    for key in Z_BLOCKS["organ"]:
        pre = float((z_pre or {}).get(key, 0.0) or 0.0)
        fail = float((z_fail or {}).get(key, 0.0) or 0.0)
        if pre - fail > thr:
            out.append(key)
    return out


def reverse_from_failure(
    z_fail: dict,
    field: VelocityField,
    lookback_s: float = 300.0,
    steps: int = 4,
    note: str = "",
    action: Optional[str] = None,
) -> dict:
    """Full reverse package: z_pre + organs already degraded before failure."""
    rev = integrate_backward(z_fail, field, lookback_s=lookback_s, steps=steps, action=action)
    z_pre = rev.get("z_pre") or {}
    degraded = pre_failure_organs(z_pre, z_fail)
    v_now = _field_predict(field, z_fail or {}, action=action)
    return {
        "z_fail": {k: float((z_fail or {}).get(k, 0.0) or 0.0) for k in Z_KEYS},
        "z_pre": z_pre,
        "pre_failure_organs": degraded,
        "negative_organs_now": negative_organs(v_now),
        "action": canon_action(action) if action is not None else None,
        "reverse": rev,
        "note": note or "failure_reverse_integrate",
        "evidence_status": rev.get("evidence_status", "model_reconstruction"),
    }


def rankme_windows(history: list[dict]) -> dict:
    """Collapse diagnostics on sliding windows of z (and optional pool node rows)."""
    if not history:
        return {"z_window": {"effective_rank": 0.0, "collapse": True, "n": 0}}
    keys = list(Z_KEYS)
    rows = [[float((h.get("z") or {}).get(k, 0.0) or 0.0) for k in keys] for h in history]
    z_win = centered_effective_rank(rows[-64:])
    block_ranks = {}
    for name, block in Z_BLOCKS.items():
        brows = [[float((h.get("z") or {}).get(k, 0.0) or 0.0) for k in block] for h in history]
        block_ranks[name] = centered_effective_rank(brows[-64:])
    return {"z_window": z_win, "blocks": block_ranks}


class CapabilityFlowTracker:
    """Wall-clock sampled capability flow with first-order velocity field."""

    def __init__(self, history_size: int = 128, min_dt: float = 0.5):
        self.history_size = history_size
        self.min_dt = float(min_dt)
        self.field = ActionConditionedVelocityField()
        # C2: NLM private filters · kernel v · cognitive map · sync working memory
        self.nlm = NLMBank(Z_KEYS)
        self.kernel = KernelVelocityField()
        self.cogmap = CognitiveMap(Z_KEYS)
        self.sync_mem = SyncWorkingMemory()
        self._samples: deque[FlowSample] = deque(maxlen=history_size)
        self._obs_log: deque[dict] = deque(maxlen=64)
        self._last_t: Optional[float] = None
        self._last_z: Optional[list[float]] = None
        self._last_z_map: dict = {}
        self._last_static: dict = {}
        self._last_action: str = "unknown"

    def observe(
        self,
        stats: dict,
        t_wall: Optional[float] = None,
        note: str = "",
        action: Optional[str] = None,
    ) -> FlowObservation:
        """Record one wall-clock sample; fit v_hat on v_gt (first-order).

        action: causal host/ARSI op for this interval (P2-8 v(z; a)).
        """
        t = float(t_wall if t_wall is not None else time.time())
        act = canon_action(action)
        z_map = encode_dyn_state(stats)
        static = encode_static_context(stats)
        z_vec = [float(z_map[k]) for k in Z_KEYS]
        dt = 0.0
        v_gt = [0.0] * len(Z_KEYS)
        if self._last_t is not None and self._last_z is not None:
            dt = t - self._last_t
            if dt >= self.min_dt:
                v_gt = velocity_gt(self._last_z, z_vec, dt, min_dt=self.min_dt)
                self.field.fit_sample(z_vec, v_gt, action=act)
                self.kernel.fit_sample(z_vec, v_gt, action=act)
            else:
                # too fast — keep last, do not fake velocity
                dt = max(dt, 0.0)
        # C2-1/3: private NLM filter + cognitive map edge
        nlm_z = self.nlm.observe_z(z_map)
        if self._last_z_map:
            self.cogmap.observe(self._last_z_map, z_map, act)
        self._last_z_map = dict(z_map)
        sample = FlowSample(
            t_wall=t, z=z_vec, v_gt=v_gt, dt=dt, static=static,
            note=f"{note}|{act}", action=act,
        )
        self._samples.append(sample)
        self._last_t = t
        self._last_z = z_vec
        self._last_static = static
        self._last_action = act

        v_hat = self.field.predict(z_map, action=act)
        v_gt_map = {k: round(v_gt[i], 6) for i, k in enumerate(Z_KEYS)}
        v_err = {
            k: round(float(v_hat.get(k, 0.0) or 0.0) - float(v_gt_map.get(k, 0.0) or 0.0), 6)
            for k in Z_KEYS
        }
        hist_maps = [{"z": {k: s.z[i] for i, k in enumerate(Z_KEYS)}} for s in self._samples]
        rk = rankme_windows(hist_maps)
        z_maps = [dict(h["z"]) for h in hist_maps]
        dts = [float(s.dt) for s in self._samples]
        acts = [getattr(s, "action", "") or "unknown" for s in self._samples]
        try:
            sync_rep = sync_report_from_z_rows(z_maps, Z_BLOCKS, dts=dts, actions=acts)
        except Exception:
            sync_rep = {"n_samples": 0, "note": "sync_unavailable"}
        # only integrate after ≥2 updates so v_hat is not pure prior noise
        integ = {}
        if self.field.n_updates >= 2:
            integ = integrate(z_map, self.field, horizon_s=max(30.0, dt or 300.0) * 2, steps=4, action=act)

        obs = FlowObservation(
            t_wall=t,
            dt=round(dt, 4),
            z=z_map,
            v_gt=v_gt_map,
            v_hat=v_hat,
            v_error=v_err,
            negative_organs=negative_organs(v_hat),
            rankme=rk,
            integrated=integ,
            note=f"{note or 'capability_flow_observe'}|action={act}",
        )
        try:
            obs_d = obs.to_dict()
            obs_d["sync"] = sync_rep
            self._obs_log.append(obs_d)
        except Exception:
            self._obs_log.append(obs.to_dict())
        return obs

    def health(self) -> dict:
        last = self._obs_log[-1] if self._obs_log else {}
        opf = {}
        try:
            from arsi.world_model.opf_discipline import opf_discipline

            z_rows = []
            for o in list(self._obs_log)[-64:]:
                z = o.get("z") or {}
                if z:
                    z_rows.append(z)
            if z_rows:
                opf = opf_discipline(z_rows).to_dict()
        except Exception:
            opf = {}
        return {
            "n_samples": len(self._samples),
            "n_v_updates": self.field.n_updates,
            "field_mse_ewma": round(self.field.mse(), 8),
            "last_dt": last.get("dt"),
            "last_negative_organs": last.get("negative_organs") or [],
            "last_rankme": last.get("rankme") or {},
            "last_v_hat": last.get("v_hat") or {},
            "last_v_gt": last.get("v_gt") or {},
            "last_z": last.get("z") or {},
            "last_integrated_goal": (last.get("integrated") or {}).get("z_goal") or {},
            "static_context": self._last_static,
            "last_action": self._last_action,
            "action_support": getattr(self.field, "action_support", lambda: {})(),
            "field": self.field.to_dict() if hasattr(self.field, "to_dict") else {},
            "sync": last.get("sync") or {},
            "dead_channels": (last.get("sync") or {}).get("dead_channels") or [],
            "dead_organs": (last.get("sync") or {}).get("dead_organs") or [],
            "nlm": self.nlm.report(),
            "kernel_v": self.kernel.to_dict(),
            "cognitive_map": self.cogmap.report(),
            "sync_memory": self.sync_mem.report(),
            "nlm_pred": self.nlm.predict_z(),
            "track_policy": "live/pool/D_T/v split — never mix scores",
            "decoouple_policy": DECOUPLE_POLICY,
            "opf": opf,
            "note": "ode_world_pt_flow_adapted_statistical",
        }

    def flow_guidance(self, horizon_s: float = 600.0) -> dict:
        """P1 control surface: negative-velocity organs + z subgoal for planning.

        Evidence-based only — derived from observed v_hat / integrate, never invented.
        """
        last = self._obs_log[-1] if self._obs_log else {}
        neg = list(last.get("negative_organs") or [])
        v_hat = dict(last.get("v_hat") or {})
        z_now = dict(last.get("z") or {})
        goal = {}
        if self.field.n_updates >= 2 and z_now:
            try:
                goal = integrate(
                    z_now, self.field, horizon_s=horizon_s, steps=4, action=self._last_action
                ).get("z_goal") or {}
            except Exception:
                goal = {}
        if not goal:
            goal = (last.get("integrated") or {}).get("z_goal") or {}

        # P1 reverse: if last obs is a failure anchor, rebuild z_pre
        z_pre = {}
        pre_fail = []
        rev = {}
        if (last.get("note") or "").startswith("fail") or last.get("failure_anchor"):
            try:
                rev = reverse_from_failure(
                    z_now,
                    self.field,
                    lookback_s=max(60.0, horizon_s / 2),
                    action=self._last_action,
                )
                z_pre = rev.get("z_pre") or {}
                pre_fail = list(rev.get("pre_failure_organs") or [])
            except Exception:
                rev = {}
        # always expose last known reverse if recorded
        if not z_pre and last.get("z_pre"):
            z_pre = dict(last.get("z_pre") or {})
            pre_fail = list(last.get("pre_failure_organs") or [])

        focus = "execute_with_evidence"
        # pre-failure degradation is stronger signal than instantaneous negative v
        focus_src = list(pre_fail) + [x for x in neg if x not in pre_fail]
        if "memory_trust" in focus_src:
            focus = "reingest"
        elif "behavior_predictor_trust" in focus_src or "self_trust" in focus_src:
            focus = "calibrate"
        elif "live_last" in focus_src or "live_ema" in focus_src:
            focus = "explore_frontier"
        elif focus_src:
            focus = "repair_organs"

        sync = last.get("sync") or {}
        route = {}
        if goal and z_now:
            try:
                route = self.cogmap.route(z_now, goal)
            except Exception:
                route = {}
        return {
            "focus": focus,
            "negative_organs": neg,
            "pre_failure_organs": pre_fail,
            "dead_organs": list(sync.get("dead_organs") or []),
            "dead_channels": list(sync.get("dead_channels") or []),
            "z_pre": z_pre,
            "reverse": rev,
            "v_hat": v_hat,
            "kernel_v": (
                self.kernel.predict(z_now, action=self._last_action) if z_now else {}
            ),
            "z_now": z_now,
            "z_subgoal": goal,
            "cognitive_route": route,
            "horizon_s": float(horizon_s),
            "n_v_updates": self.field.n_updates,
            "source": "capability_flow_flow_guidance",
        }

    def reverse_last_failure(
        self,
        z_fail: Optional[dict] = None,
        lookback_s: float = 300.0,
        action: Optional[str] = None,
    ) -> dict:
        """Explicit reverse from a failure (host_reverse / IWM). action conditions v(z; a)."""
        z = z_fail or (self._obs_log[-1].get("z") if self._obs_log else {}) or {}
        act = action if action is not None else self._last_action
        out = reverse_from_failure(z, self.field, lookback_s=lookback_s, action=act)
        if self._obs_log:
            self._obs_log[-1]["z_pre"] = out.get("z_pre")
            self._obs_log[-1]["pre_failure_organs"] = out.get("pre_failure_organs")
            self._obs_log[-1]["failure_anchor"] = True
            self._obs_log[-1]["failure_action"] = out.get("action")
        return out

    def serialize(self) -> dict:
        return {
            "schema": "arsi.capability_flow.v1",
            "field": self.field.to_dict(),
            "history": [s.to_dict() for s in self._samples],
            "last_obs": self._obs_log[-1] if self._obs_log else {},
        }

    @classmethod
    def from_serialize(cls, data: dict) -> "CapabilityFlowTracker":
        tr = cls()
        fd = data.get("field") or {}
        try:
            n = int(fd.get("n_updates") or 0)
            if fd.get("a") and fd.get("b"):
                tr.field.a = [float(x) for x in fd["a"]]
                tr.field.b = [float(x) for x in fd["b"]]
                tr.field.n_updates = n
            for a, slot in (fd.get("action_a") or {}).items():
                tr.field.action_a[str(a)] = [float(x) for x in slot]
            for a, slot in (fd.get("action_b") or {}).items():
                tr.field.action_b[str(a)] = [float(x) for x in slot]
            for a, cnt in (fd.get("action_n") or {}).items():
                tr.field.action_n[str(a)] = int(cnt)
            if fd.get("last_action"):
                tr.field.last_action = canon_action(str(fd["last_action"]))
                tr._last_action = tr.field.last_action
        except Exception:
            pass
        for h in data.get("history") or []:
            try:
                tr._samples.append(
                    FlowSample(
                        t_wall=float(h.get("t_wall") or 0.0),
                        z=[float(x) for x in (h.get("z") or [])],
                        v_gt=[float(x) for x in (h.get("v_gt") or [])],
                        dt=float(h.get("dt") or 0.0),
                        static=dict(h.get("static") or {}),
                        note=str(h.get("note") or ""),
                    )
                )
            except Exception:
                continue
        if tr._samples:
            tr._last_t = tr._samples[-1].t_wall
            tr._last_z = list(tr._samples[-1].z)
        return tr
