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


@dataclass
class FlowSample:
    t_wall: float
    z: list[float]
    v_gt: list[float] = field(default_factory=list)
    dt: float = 0.0
    static: dict = field(default_factory=dict)
    note: str = ""

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


def velocity_gt(z_prev: Sequence[float], z_now: Sequence[float], dt: float, min_dt: float = 1e-3) -> list[float]:
    """First-order finite difference over physical time (Savitzky–Golay light: 2-pt)."""
    dt_eff = max(float(dt), min_dt)
    return [(float(b) - float(a)) / dt_eff for a, b in zip(z_prev, z_now)]


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
        v_hat = self.predict_vec(z)
        err = 0.0
        for i in range(self.n):
            e = v_hat[i] - float(v_gt[i])
            err += e * e
            gi = 2.0 * e
            self.a[i] -= self.lr * (gi * float(z[i]) + self.l2 * self.a[i])
            self.b[i] -= self.lr * gi
            self.se[i] = 0.9 * self.se[i] + 0.1 * e * e
        self.n_updates += 1
        return err / max(1, self.n)

    def predict_vec(self, z: Sequence[float]) -> list[float]:
        return [self.a[i] * float(z[i]) + self.b[i] for i in range(self.n)]

    def predict(self, z_map: dict) -> dict:
        vec = [float(z_map.get(k, 0.0) or 0.0) for k in Z_KEYS]
        v = self.predict_vec(vec)
        return {k: round(v[i], 6) for i, k in enumerate(Z_KEYS)}

    def mse(self) -> float:
        return sum(self.se) / max(1, self.n)

    def to_dict(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "mse_ewma": round(self.mse(), 8),
            "a": [round(x, 5) for x in self.a],
            "b": [round(x, 5) for x in self.b],
        }


def integrate(z0: dict, field: VelocityField, horizon_s: float = 300.0, steps: int = 4) -> dict:
    """Euler integrate v_hat from z0 over physical horizon → subgoal z_τ (super-res)."""
    z = {k: float(z0.get(k, 0.0) or 0.0) for k in Z_KEYS}
    if horizon_s <= 0 or steps <= 0:
        return {"z_goal": z, "horizon_s": 0.0, "steps": 0}
    dt = float(horizon_s) / float(steps)
    path = [{**z}]
    for _ in range(steps):
        v = field.predict(z)
        z = {k: z[k] + float(v.get(k, 0.0)) * dt for k in Z_KEYS}
        path.append({**z})
    return {
        "z_goal": {k: round(z[k], 6) for k in Z_KEYS},
        "path": [{k: round(p[k], 6) for k in Z_KEYS} for p in path],
        "horizon_s": float(horizon_s),
        "steps": int(steps),
        "note": "euler_super_resolution_subgoal",
    }


def negative_organs(v_hat: dict, thr: float = -1e-4) -> list[str]:
    """Organs whose trust-like components are decreasing — IWM focus candidates."""
    out = []
    for key in Z_BLOCKS["organ"]:
        if float(v_hat.get(key, 0.0) or 0.0) < thr:
            out.append(key)
    return out


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
        self.field = VelocityField()
        self._samples: deque[FlowSample] = deque(maxlen=history_size)
        self._obs_log: deque[dict] = deque(maxlen=64)
        self._last_t: Optional[float] = None
        self._last_z: Optional[list[float]] = None
        self._last_static: dict = {}

    def observe(self, stats: dict, t_wall: Optional[float] = None, note: str = "") -> FlowObservation:
        """Record one wall-clock sample; fit v_hat on v_gt (first-order)."""
        t = float(t_wall if t_wall is not None else time.time())
        z_map = encode_dyn_state(stats)
        static = encode_static_context(stats)
        z_vec = [float(z_map[k]) for k in Z_KEYS]
        dt = 0.0
        v_gt = [0.0] * len(Z_KEYS)
        if self._last_t is not None and self._last_z is not None:
            dt = t - self._last_t
            if dt >= self.min_dt:
                v_gt = velocity_gt(self._last_z, z_vec, dt, min_dt=self.min_dt)
                self.field.fit_sample(z_vec, v_gt)
            else:
                # too fast — keep last, do not fake velocity
                dt = max(dt, 0.0)
        sample = FlowSample(t_wall=t, z=z_vec, v_gt=v_gt, dt=dt, static=static, note=note)
        self._samples.append(sample)
        self._last_t = t
        self._last_z = z_vec
        self._last_static = static

        v_hat = self.field.predict(z_map)
        v_gt_map = {k: round(v_gt[i], 6) for i, k in enumerate(Z_KEYS)}
        v_err = {
            k: round(float(v_hat.get(k, 0.0) or 0.0) - float(v_gt_map.get(k, 0.0) or 0.0), 6)
            for k in Z_KEYS
        }
        hist_maps = [{"z": {k: s.z[i] for i, k in enumerate(Z_KEYS)}} for s in self._samples]
        rk = rankme_windows(hist_maps)
        # only integrate after ≥2 updates so v_hat is not pure prior noise
        integ = {}
        if self.field.n_updates >= 2:
            integ = integrate(z_map, self.field, horizon_s=max(30.0, dt or 300.0) * 2, steps=4)

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
            note=note or "capability_flow_observe",
        )
        self._obs_log.append(obs.to_dict())
        return obs

    def health(self) -> dict:
        last = self._obs_log[-1] if self._obs_log else {}
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
            "track_policy": "live/pool/D_T/v split — never mix scores",
            "note": "ode_world_pt_flow_adapted_statistical",
        }

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
