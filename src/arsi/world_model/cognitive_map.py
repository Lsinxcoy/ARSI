"""C2-3 Cognitive map — no external coordinates (CTM maze lesson / D.6).

Edges are empirical (action → Δz) transitions. Query answers:
  which dimensions must move, and which actions historically moved them
  the right way — an imagined route in capability space, not a grid maze.

Diagnostic / planning aid only. Never invents deltas not observed.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional, Sequence

__all__ = ["CognitiveMap"]

DEFAULT_KEYS = (
    "self_trust",
    "memory_trust",
    "behavior_predictor_trust",
    "eta",
    "live_last",
    "live_ema",
    "pool_last",
    "pool_ema",
    "d_t_mean",
    "d_t_spread",
    "el_pass_mean",
)


class CognitiveMap:
    def __init__(self, keys: Sequence[str] = DEFAULT_KEYS, min_obs: int = 3, goal_eps: float = 0.05):
        self.keys = list(keys)
        self.min_obs = int(min_obs)
        self.goal_eps = float(goal_eps)
        # (action, dim) -> list of deltas
        self._obs: dict[tuple[str, str], list[float]] = defaultdict(list)

    def observe(self, z_before: dict, z_after: dict, action: str) -> dict:
        a = (action or "unknown").strip().lower()
        deltas = {}
        for k in self.keys:
            d = float((z_after or {}).get(k, 0.0) or 0.0) - float((z_before or {}).get(k, 0.0) or 0.0)
            deltas[k] = round(d, 6)
            if abs(d) > 1e-9:
                self._obs[(a, k)].append(d)
        return {"action": a, "deltas": deltas}

    def dimensions_to_change(self, z_now: dict, z_goal: dict) -> list[dict]:
        """Dims where |goal - now| > eps, with required direction."""
        out = []
        for k in self.keys:
            need = float((z_goal or {}).get(k, 0.0) or 0.0) - float((z_now or {}).get(k, 0.0) or 0.0)
            if abs(need) > self.goal_eps:
                out.append(
                    {
                        "dim": k,
                        "need": round(need, 6),
                        "direction": 1 if need > 0 else -1,
                    }
                )
        out.sort(key=lambda x: abs(x["need"]), reverse=True)
        return out

    def actions_for_dim(self, dim: str, direction: int = 0) -> list[dict]:
        """Historical mean Δdim per action; direction filter: +1 raise, -1 lower, 0 all."""
        rows = []
        for (a, k), deltas in self._obs.items():
            if k != dim or not deltas:
                continue
            mean_d = sum(deltas) / len(deltas)
            if direction > 0 and mean_d <= 0:
                continue
            if direction < 0 and mean_d >= 0:
                continue
            rows.append(
                {
                    "action": a,
                    "mean_delta": round(mean_d, 6),
                    "n": len(deltas),
                    "trusted": len(deltas) >= self.min_obs,
                }
            )
        rows.sort(key=lambda r: (r["trusted"], abs(r["mean_delta"]), r["n"]), reverse=True)
        return rows

    def route(self, z_now: dict, z_goal: dict, max_steps: int = 4) -> dict:
        """Imagined action sequence to walk z toward goal — episodic future thinking lite."""
        needs = self.dimensions_to_change(z_now, z_goal)
        steps = []
        used = []
        for need in needs:
            cands = self.actions_for_dim(need["dim"], direction=need["direction"])
            cands = [c for c in cands if c["trusted"]]
            if not cands:
                steps.append(
                    {
                        "dim": need["dim"],
                        "need": need["need"],
                        "action": None,
                        "reason": "unmeasured_no_trusted_action",
                    }
                )
                continue
            best = cands[0]
            steps.append(
                {
                    "dim": need["dim"],
                    "need": need["need"],
                    "action": best["action"],
                    "mean_delta": best["mean_delta"],
                    "n": best["n"],
                    "reason": "historical_mean_delta",
                }
            )
            used.append(best["action"])
        # compact to unique action order
        order = []
        for a in used:
            if a and a not in order:
                order.append(a)
        return {
            "needs": needs,
            "steps": steps[: max(1, int(max_steps)) * 2],
            "action_order": order[:max_steps],
            "coverage": round(
                sum(1 for s in steps if s.get("action")) / max(1, len(steps)), 4
            ),
            "note": "cognitive_map_no_external_coords",
        }

    def report(self) -> dict:
        by_action: dict[str, int] = {}
        for (a, _k), ds in self._obs.items():
            by_action[a] = by_action.get(a, 0) + len(ds)
        return {
            "edges": len(self._obs),
            "obs_per_action": by_action,
            "min_obs": self.min_obs,
            "note": "empirical_action_delta_graph",
        }
