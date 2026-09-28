"""C2-4 Sync working memory — pair-bound evidence IDs (CTM Q&A /超窗召回).

High-|ρ| organ pairs act as short-term bindings: evidence IDs stamped when
both channels co-move, recalled when current z re-enters that neighborhood.

Observation / retrieval only — never invents evidence, never scores.
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

from arsi.foundation.sync_repr import pearson

__all__ = ["SyncWorkingMemory"]


def _pair_key(a: str, b: str) -> str:
    x, y = sorted([str(a), str(b)])
    return f"{x}~{y}"


class SyncWorkingMemory:
    def __init__(
        self,
        capacity: int = 128,
        bind_rho: float = 0.5,
        min_bind_n: int = 3,
        recall_top_k: int = 5,
    ):
        self.capacity = int(capacity)
        self.bind_rho = float(bind_rho)
        self.min_bind_n = int(min_bind_n)
        self.recall_top_k = int(recall_top_k)
        # pair_key -> deque of {evidence_id, z, t}
        self._bind: dict[str, deque] = {}
        self._recent_z: deque[dict] = deque(maxlen=32)

    def bind(
        self,
        z: dict,
        evidence_id: str,
        channels: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Stamp evidence onto co-active channel pairs. Returns pair keys bound."""
        if not evidence_id:
            return []
        z = {k: float((z or {}).get(k, 0.0) or 0.0) for k in (channels or list(z.keys()))}
        keys = list(z.keys())
        bound = []
        # bind to pairs that currently co-move with history if available
        self._recent_z.append(dict(z))
        if len(self._recent_z) >= self.min_bind_n:
            hist = list(self._recent_z)
            for i, ki in enumerate(keys):
                for kj in keys[i + 1 :]:
                    xs = [h.get(ki, 0.0) for h in hist]
                    ys = [h.get(kj, 0.0) for h in hist]
                    r = pearson(xs, ys)
                    if abs(r) >= self.bind_rho:
                        pk = _pair_key(ki, kj)
                        self._bind.setdefault(pk, deque(maxlen=self.capacity))
                        self._bind[pk].append({"evidence_id": evidence_id, "z": dict(z), "rho": round(r, 4)})
                        bound.append(pk)
        else:
            # cold start: bind to self-pairs of provided channels only
            for k in keys:
                pk = _pair_key(k, k)
                self._bind.setdefault(pk, deque(maxlen=self.capacity))
                self._bind[pk].append({"evidence_id": evidence_id, "z": dict(z), "rho": 1.0})
                bound.append(pk)
        return sorted(set(bound))

    def active_pairs(self, z: dict) -> list[dict]:
        keys = list(z.keys()) if z else []
        out = []
        hist = list(self._recent_z) + [dict(z or {})]
        if len(hist) < 2:
            return out
        for i, ki in enumerate(keys):
            for kj in keys[i + 1 :]:
                xs = [h.get(ki, 0.0) for h in hist]
                ys = [h.get(kj, 0.0) for h in hist]
                r = pearson(xs, ys)
                if abs(r) >= self.bind_rho:
                    out.append({"pair": _pair_key(ki, kj), "rho": round(r, 4)})
        out.sort(key=lambda x: abs(x["rho"]), reverse=True)
        return out

    def recall(self, z: dict, top_k: Optional[int] = None) -> list[dict]:
        """Evidence IDs bound to currently active pairs (超窗 via pair, not single organ)."""
        k = int(top_k or self.recall_top_k)
        pairs = self.active_pairs(z or {})
        seen = set()
        out = []
        for p in pairs:
            for rec in self._bind.get(p["pair"], []):
                eid = rec.get("evidence_id")
                if not eid or eid in seen:
                    continue
                seen.add(eid)
                out.append(
                    {
                        "evidence_id": eid,
                        "pair": p["pair"],
                        "rho": p["rho"],
                    }
                )
                if len(out) >= k:
                    return out
        return out

    def report(self) -> dict:
        return {
            "pairs": len(self._bind),
            "n_bound": sum(len(v) for v in self._bind.values()),
            "bind_rho": self.bind_rho,
            "note": "sync_pair_working_memory",
        }
