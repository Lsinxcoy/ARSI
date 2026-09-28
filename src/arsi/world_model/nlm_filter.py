"""C2-1 NLM-style private history filters (CTM arXiv:2505.05522, linear first).

Each dyn channel has a **private** temporal filter over its own history A_d,
analogous to a Neuron-Level Model without a neural net yet:

    ẑ_d = g_d(A_d) = w_d · [x_{t-M+1}…x_t] + b_d

First-order online SGD only (predict next value, not multi-step consistency).
Data gate: channel trusted only after min_samples; else echo last.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

__all__ = ["ChannelNLM", "NLMBank"]


class ChannelNLM:
    def __init__(self, history_len: int = 8, lr: float = 0.12, l2: float = 1e-3, min_samples: int = 8):
        self.M = max(2, int(history_len))
        self.lr = float(lr)
        self.l2 = float(l2)
        self.min_samples = int(min_samples)
        self.w = [0.0] * self.M
        self.b = 0.0
        self.hist: deque[float] = deque(maxlen=self.M)
        self.n_updates = 0
        self.se = 0.0
        self._last: Optional[float] = None

    def _features(self) -> list[float]:
        # left-pad with oldest / zeros so length is always M
        raw = list(self.hist)
        if len(raw) < self.M:
            pad = [raw[0] if raw else 0.0] * (self.M - len(raw))
            raw = pad + raw
        return [float(x) for x in raw[-self.M :]]

    def predict(self) -> float:
        if not self.hist and self.n_updates == 0:
            return 0.0
        if self.n_updates < self.min_samples:
            return float(self._last) if self._last is not None else float(self.hist[-1] if self.hist else 0.0)
        phi = self._features()
        return self.b + sum(wi * xi for wi, xi in zip(self.w, phi))

    def observe(self, x: float) -> float:
        """Push sample; if we have a previous prediction target, fit; return ẑ after update."""
        try:
            x = float(x)
        except Exception:
            x = 0.0
        if x != x or x in (float("inf"), float("-inf")):
            x = 0.0
        x = max(-1e3, min(1e3, x))
        pred = self.predict()
        if self.hist or self._last is not None:
            # supervise: predict current x from history *before* push (first-order)
            phi = self._features()
            hat = self.b + sum(wi * xi for wi, xi in zip(self.w, phi))
            if self.n_updates >= self.min_samples or self.n_updates > 0:
                e = hat - x
                gi = 2.0 * e
                for i in range(self.M):
                    self.w[i] = max(-1e3, min(1e3, self.w[i] - self.lr * (gi * phi[i] + self.l2 * self.w[i])))
                self.b = max(-1e3, min(1e3, self.b - self.lr * gi))
                self.se = 0.9 * self.se + 0.1 * e * e
                self.n_updates += 1
            else:
                self.n_updates += 1
        self.hist.append(x)
        self._last = x
        return self.predict() if self.n_updates >= self.min_samples else x

    def trusted(self) -> bool:
        return self.n_updates >= self.min_samples

    def report(self) -> dict:
        return {
            "n_updates": self.n_updates,
            "trusted": self.trusted(),
            "mse_ewma": round(self.se, 8),
            "history_len": self.M,
            "last": self._last,
            "pred": round(self.predict(), 6) if self.trusted() else None,
            "w_norm": round(sum(abs(x) for x in self.w), 5),
        }


class NLMBank:
    """Private filter per Z channel (CTM NLM bank, linear)."""

    def __init__(self, keys: Sequence[str], history_len: int = 8, min_samples: int = 8):
        self.keys = list(keys)
        self.channels = {
            k: ChannelNLM(history_len=history_len, min_samples=min_samples) for k in self.keys
        }

    def observe_z(self, z: dict) -> dict:
        """Ingest snapshot; return per-channel ẑ (private history filter)."""
        out = {}
        for k in self.keys:
            x = float((z or {}).get(k, 0.0) or 0.0)
            out[k] = round(self.channels[k].observe(x), 6)
        return out

    def predict_z(self) -> dict:
        return {k: round(self.channels[k].predict(), 6) for k in self.keys}

    def report(self) -> dict:
        rep = {k: ch.report() for k, ch in self.channels.items()}
        trusted = [k for k, r in rep.items() if r.get("trusted")]
        return {
            "channels": rep,
            "trusted_keys": trusted,
            "trusted_frac": round(len(trusted) / max(1, len(self.keys)), 4),
            "note": "nlm_private_history_filter_linear",
        }
