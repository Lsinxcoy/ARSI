"""C2-2 Kernel-regression velocity field v(z; a) — P2-8 neural/ODE upgrade (statistical).

Nadaraya–Watson kernel regression over recent (z, v_gt) pairs. First-order
supervision only (ODEWorld: no multi-step consistency loss). Optional action
conditioning by filtering the support set.

Data gate: min_samples before trusted; else fall back to zero / diagonal field.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional, Sequence

# local copy of action vocab to avoid circular import with capability_flow
_ACTION_POOL = (
    "learn",
    "remember",
    "dream",
    "maintain",
    "evolve",
    "empower",
    "repair",
    "unknown",
)
_DEFAULT_N = 11  # len(Z_KEYS)


def _canon_action(action: Optional[str]) -> str:
    a = (action or "unknown").strip().lower()
    if a in _ACTION_POOL:
        return a
    for known in _ACTION_POOL:
        if a.startswith(known):
            return known
    return "unknown"


__all__ = ["KernelVelocityField"]


def _sqdist(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


class KernelVelocityField:
    """v̂(z) = Σ w_i v_i / Σ w_i,  w_i = exp(-||z-z_i||² / (2σ²))."""

    def __init__(
        self,
        n_dims: int = _DEFAULT_N,
        bandwidth: float = 0.75,
        capacity: int = 64,
        min_samples: int = 6,
        min_action_samples: int = 3,
    ):
        self.n = n_dims
        self.bandwidth = max(1e-3, float(bandwidth))
        self.capacity = int(capacity)
        self.min_samples = int(min_samples)
        self.min_action_samples = int(min_action_samples)
        self._z: deque[list[float]] = deque(maxlen=self.capacity)
        self._v: deque[list[float]] = deque(maxlen=self.capacity)
        self._a: deque[str] = deque(maxlen=self.capacity)
        self.n_updates = 0
        self.se = 0.0
        self.last_action = "unknown"

    def fit_sample(
        self,
        z: Sequence[float],
        v_gt: Sequence[float],
        action: Optional[str] = None,
    ) -> float:
        act = _canon_action(action) if action is not None else "unknown"
        self.last_action = act
        zz = [float(x) for x in z]
        vv = [float(x) for x in v_gt]
        pred = self.predict_vec(zz, action=act if action is not None else None)
        err = sum((pred[i] - vv[i]) ** 2 for i in range(self.n)) / max(1, self.n)
        self.se = 0.9 * self.se + 0.1 * err
        self._z.append(zz)
        self._v.append(vv)
        self._a.append(act)
        self.n_updates += 1
        return err

    def _support(self, action: Optional[str]) -> tuple[list[list[float]], list[list[float]]]:
        if action is None:
            return list(self._z), list(self._v)
        act = _canon_action(action)
        zs, vs = [], []
        for z, v, a in zip(self._z, self._v, self._a):
            if a == act:
                zs.append(z)
                vs.append(v)
        if len(zs) < self.min_action_samples:
            return list(self._z), list(self._v)
        return zs, vs

    def predict_vec(self, z: Sequence[float], action: Optional[str] = None) -> list[float]:
        zz = [float(x) for x in z]
        zs, vs = self._support(action)
        if len(zs) < self.min_samples:
            return [0.0] * self.n
        h2 = 2.0 * self.bandwidth * self.bandwidth
        wsum = 0.0
        acc = [0.0] * self.n
        for zi, vi in zip(zs, vs):
            w = math.exp(-_sqdist(zz, zi) / h2)
            if w < 1e-8:
                continue
            wsum += w
            for i in range(self.n):
                acc[i] += w * vi[i]
        if wsum <= 0.0:
            return [0.0] * self.n
        return [acc[i] / wsum for i in range(self.n)]

    def predict(self, z_map: dict, action: Optional[str] = None) -> dict:
        keys = list(z_map.keys()) if z_map else [f"x{i}" for i in range(self.n)]
        # pad / trim to n
        vec = [float((z_map or {}).get(k, 0.0) or 0.0) for k in keys[: self.n]]
        if len(vec) < self.n:
            vec.extend([0.0] * (self.n - len(vec)))
        v = self.predict_vec(vec, action=action)
        out_keys = keys[: self.n] if keys else [f"x{i}" for i in range(self.n)]
        return {out_keys[i] if i < len(out_keys) else f"x{i}": round(v[i], 6) for i in range(self.n)}

    def trusted(self, action: Optional[str] = None) -> bool:
        if action is None:
            return self.n_updates >= self.min_samples
        zs, _ = self._support(action)
        return len(zs) >= self.min_action_samples

    def mse(self) -> float:
        return self.se

    def to_dict(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "support": len(self._z),
            "bandwidth": self.bandwidth,
            "mse_ewma": round(self.mse(), 8),
            "min_samples": self.min_samples,
            "trusted": self.trusted(),
            "last_action": self.last_action,
            "note": "nadaraya_watson_first_order_v",
        }
