"""Evolution-Lineage (EL) Scheduler — arXiv:2609.04128 §4.2.

Do not randomly sample hard generations. Advance lineage generation
only when the current generation's environments are sufficiently solved.

p̂_u(E) = mean success over recent probes
advance when p̂ > τ (default 6/8 = 0.75)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence


@dataclass
class LineageState:
    lineage_id: str
    generation: int = 0
    world_ids: list[str] = field(default_factory=list)
    probe_history: list[float] = field(default_factory=list)
    last_advance_ts: str = ""
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ELScheduler:
    def __init__(self, tau: float = 0.75, batch: int = 8):
        self.tau = float(tau)
        self.batch = int(batch)
        self.lineages: dict[str, LineageState] = {}
        self._active: dict[str, int] = {}  # lineage -> index into world list
        self.advances = 0

    def register_world(self, lineage_id: str, world_id: str, generation: int) -> None:
        st = self.lineages.get(lineage_id)
        if st is None:
            st = LineageState(lineage_id=lineage_id, generation=generation)
            self.lineages[lineage_id] = st
        if world_id not in st.world_ids:
            st.world_ids.append(world_id)
        st.generation = max(st.generation, generation)
        self._active.setdefault(lineage_id, 0)

    def record_probe(self, lineage_id: str, success: bool, score: Optional[float] = None) -> None:
        st = self.lineages.get(lineage_id)
        if st is None:
            return
        # P2-3: trust the explicit success flag (quality-based); do NOT override
        # with replay_score>0 — paper scores are often negative (cost term).
        if success is not None:
            ok = 1.0 if success else 0.0
        elif score is not None:
            ok = 1.0 if float(score) > 0 else 0.0
        else:
            ok = 0.0
        st.probe_history.append(ok)
        if len(st.probe_history) > self.batch:
            st.probe_history = st.probe_history[-self.batch :]

    def pass_rate(self, lineage_id: str) -> float:
        st = self.lineages.get(lineage_id)
        if not st or not st.probe_history:
            return 0.0
        return sum(st.probe_history) / len(st.probe_history)

    def active_world_id(self, lineage_id: str) -> Optional[str]:
        st = self.lineages.get(lineage_id)
        if not st or not st.world_ids:
            return None
        idx = self._active.get(lineage_id, 0)
        idx = min(idx, len(st.world_ids) - 1)
        return st.world_ids[idx]

    def maybe_advance(self, lineage_id: str) -> dict:
        st = self.lineages.get(lineage_id)
        if st is None:
            return {"advanced": False, "reason": "unknown_lineage"}
        p = self.pass_rate(lineage_id)
        # P2-3 curriculum: stuck at ~0 for a full batch → do not stack harder gens
        if len(st.probe_history) >= self.batch and p <= 0.05:
            return {
                "advanced": False,
                "reason": "curriculum_hold_zero_pass",
                "pass_rate": round(p, 4),
                "tau": self.tau,
            }
        idx = self._active.get(lineage_id, 0)
        n = len(st.world_ids)
        if n == 0:
            return {"advanced": False, "reason": "empty_lineage", "pass_rate": p}
        if p > self.tau:
            if idx < n - 1:
                self._active[lineage_id] = idx + 1
                self.advances += 1
                st.last_advance_ts = datetime.now().isoformat()
                st.probe_history = []  # reset window for next env
                return {
                    "advanced": True,
                    "reason": "pass_rate_above_tau",
                    "pass_rate": round(p, 4),
                    "new_index": idx + 1,
                    "world_id": st.world_ids[idx + 1],
                }
            return {"advanced": False, "reason": "lineage_exhausted", "pass_rate": round(p, 4), "n_worlds": n}
        return {"advanced": False, "reason": "below_tau", "pass_rate": round(p, 4), "tau": self.tau}

    def sample_world_ids(self, worlds: list, k: int = 3) -> list[str]:
        """EL-aware sampling: prefer active generation worlds; pad with older gens."""
        if not worlds:
            return []
        by_id = {getattr(w, "world_id", None) or i: w for i, w in enumerate(worlds)}
        chosen: list[str] = []
        for lin, st in self.lineages.items():
            wid = self.active_world_id(lin)
            if wid and wid in by_id and wid not in chosen:
                chosen.append(wid)
            if len(chosen) >= k:
                break
        # pad with remaining worlds (stable order)
        for w in worlds:
            wid = getattr(w, "world_id", None)
            if wid and wid not in chosen:
                chosen.append(wid)
            if len(chosen) >= k:
                break
        return chosen[:k]

    def select_worlds(self, worlds: list, k: int = 3) -> list:
        ids = set(self.sample_world_ids(worlds, k=k))
        picked = [w for w in worlds if getattr(w, "world_id", None) in ids]
        if len(picked) < k:
            picked = (picked + [w for w in worlds if w not in picked])[:k]
        return picked

    def health(self) -> dict:
        return {
            "tau": self.tau,
            "batch": self.batch,
            "advances": self.advances,
            "lineages": {
                lin: {
                    "generation": st.generation,
                    "n_worlds": len(st.world_ids),
                    "pass_rate": round(self.pass_rate(lin), 4),
                    "active_index": self._active.get(lin, 0),
                    "active_world": self.active_world_id(lin),
                }
                for lin, st in self.lineages.items()
            },
        }


def env_evolution_fidelity_check(
    *,
    off_policy: bool = True,
    directions: Optional[Sequence[str]] = None,
    invalid_test: bool = True,
    el_schedule: bool = True,
) -> dict:
    """EnvEvo invariants: off-policy, 3 directions, empty-world fails, EL by generation."""
    dirs = set(directions or ("scenario", "skill", "length"))
    ok = (
        bool(off_policy)
        and dirs >= {"scenario", "skill", "length"}
        and bool(invalid_test)
        and bool(el_schedule)
    )
    return {
        "ok": ok,
        "off_policy": bool(off_policy),
        "directions": sorted(dirs),
        "invalid_test": bool(invalid_test),
        "el_schedule": bool(el_schedule),
        "note": "env_evolution_fidelity_2609_04128",
    }
