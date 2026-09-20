"""Replay World — paper-faithful Dream-RSI discovery-tree simulator.

Based on arXiv:2609.14858 §3 + Appendix B.2.

Key formalism:
- Eligible A(T) = {root} ∪ {v ∈ T : v is a leaf of observed prefix}
- Action C ⊆ A(T; W), |C| ≤ W
- Child(v; T_full, T_obs):
  - v ≠ root: unique recorded child, if still unrevealed
  - v = root: earliest-created child not yet in observed (branch opening)
- Replay is deterministic and prefix-only: unrevealed scores stay unknown
- Score: pareto.reward = pareto.auc - λ · parallel_penalty
  (ARSI uses quality - β1·N + β2·N/k with the same three axes)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# Fail classes from Dream-RSI Appendix B.2 — which failures are worth retrying
REPAIRABLE_FAIL_CLASSES = {
    "compile_other",
    "runtime_error",
    "shape_error",
    "mask_error",
    "layout_error",
    "resource_limit",
    "timeout",
    "mismatch",
    "correctness_mismatch",
    "dependency",
}
HARD_FAIL_CLASSES = {
    "hard_unrecoverable",
    "invalid_input",
    "forbidden",
}


@dataclass
class Observation:
    """Prefix-observable cell outcome (Dream-RSI Observation)."""
    cell_id: str
    branch: int = 0
    attempt: int = 0
    parent_id: Optional[str] = None
    action: str = ""
    agent_id: str = ""
    score: float = 0.0
    evaluated: bool = True
    valid: bool = True
    fail_class: str = "ok"
    error: str = ""
    delta_vs_baseline: float = 0.0
    delta_vs_parent: float = 0.0
    n_valid: int = 0
    n_total: int = 0
    cost: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Dream-RSI success semantics: evaluated + error is None + fail_class=="ok"."""
        return self.evaluated and (self.error in ("", None)) and self.fail_class == "ok"

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "branch": self.branch,
            "attempt": self.attempt,
            "parent_id": self.parent_id,
            "action": self.action,
            "agent_id": self.agent_id,
            "score": self.score,
            "evaluated": self.evaluated,
            "valid": self.valid,
            "fail_class": self.fail_class,
            "error": self.error,
            "delta_vs_baseline": self.delta_vs_baseline,
            "delta_vs_parent": self.delta_vs_parent,
            "n_valid": self.n_valid,
            "n_total": self.n_total,
            "cost": self.cost,
            "tags": self.tags,
            "success": self.success,
        }


@dataclass
class ReplayResult:
    """One policy×world replay trajectory score."""
    world_id: str
    policy_name: str
    quality: float = 0.0
    cost: float = 0.0
    parallelism: float = 0.0
    replay_score: float = 0.0
    rounds: int = 0
    probes: int = 0
    opened_branches: int = 0
    successful_anchors: int = 0
    repairables: int = 0
    trajectory: list[dict] = field(default_factory=list)
    score_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "policy_name": self.policy_name,
            "quality": self.quality,
            "cost": self.cost,
            "parallelism": self.parallelism,
            "replay_score": self.replay_score,
            "rounds": self.rounds,
            "probes": self.probes,
            "opened_branches": self.opened_branches,
            "successful_anchors": self.successful_anchors,
            "repairables": self.repairables,
            "score_breakdown": self.score_breakdown,
        }


class ReplayWorld:
    """Frozen discovery tree usable as an off-policy replay simulator.

    Prefix-only API (Dream-RSI Appendix B.2 adapted to ARSI):
      reset(), observed(), legal_actions(), legal_roots(), opened_branches(),
      meta(cell_id), probe_batch(cells), baseline_score, max_parallelism
    """

    def __init__(
        self,
        world_id: str,
        nodes: dict[str, dict],
        baseline_score: float = 0.0,
        max_parallelism: int = 3,
        beta1: float = 0.1,
        beta2: float = 0.05,
        parallel_lambda: float = 0.05,
    ):
        self.world_id = world_id
        # Full recorded tree — policies must never peek past observed()
        self._full: dict[str, dict] = nodes
        self.baseline_score = baseline_score
        self.max_parallelism = max(1, int(max_parallelism))
        self.beta1 = beta1
        self.beta2 = beta2
        self.parallel_lambda = parallel_lambda
        self.score_mode = "quality_anchored"  # quality_anchored | paper
        self.min_quality_signal = 0.08
        self.weak_quality_cost_scale = 0.2
        self._observed: set[str] = set()
        self._obs_cache: dict[str, Observation] = {}
        self._probes = 0
        self._cost = 0.0
        self._rounds = 0
        self._branch_counter = 0
        self._root_open_order: list[str] = []

    # ── Dream-RSI question API ────────────────────────────────────
    def reset(self) -> None:
        self._observed = set()
        self._obs_cache = {}
        self._probes = 0
        self._cost = 0.0
        self._rounds = 0
        self._branch_counter = 0
        self._root_open_order = []
        if "root" in self._full:
            self._reveal("root")

    def observed(self) -> dict[str, Observation]:
        return dict(self._obs_cache)

    def legal_actions(self) -> list[str]:
        return self._eligible()

    def legal_roots(self) -> list[str]:
        """Unopened root-branch openings — ARSI maps this to root itself."""
        return ["root"] if "root" in self._full else []

    def opened_branches(self) -> list[int]:
        branches = {o.branch for o in self._obs_cache.values() if o.cell_id != "root"}
        return sorted(branches)

    def meta(self, cell_id: str) -> Optional[Observation]:
        return self._obs_cache.get(cell_id)

    @property
    def probe_count(self) -> int:
        return self._probes

    @property
    def total_cost(self) -> float:
        return self._cost

    # ── Transition ─────────────────────────────────────────────────
    def probe_batch(self, cells: Iterable[str]) -> list[Observation]:
        """Deterministically reveal recorded children (paper Child() semantics)."""
        batch = [c for c in cells if c in self._eligible()][: self.max_parallelism]
        if not batch:
            return []

        revealed: list[Observation] = []
        for cell_id in batch:
            child_ids = self._child(cell_id)
            for cid in child_ids:
                self._reveal(cid)
                if cid in self._obs_cache:
                    revealed.append(self._obs_cache[cid])
                    self._probes += 1
                    self._cost += float(self._full[cid].get("cost", 0.0))
        if revealed:
            self._rounds += 1
        return revealed

    def replay(
        self,
        policy_fn: Callable[[dict[str, Observation], list[str], int], list[str]],
        policy_name: str = "policy",
        max_rounds: int = 20,
        beta: float = 0.5,
    ) -> ReplayResult:
        """Run one prefix-only policy through this world."""
        self.reset()
        trajectory: list[dict] = []

        for _ in range(max_rounds):
            legal = self._eligible()
            if not legal:
                break
            batch = policy_fn(self.observed(), legal, self.max_parallelism)
            batch = [c for c in (batch or []) if c in legal]
            if not batch:
                break
            self.probe_batch(batch)
            trajectory.append({
                "round": self._rounds,
                "batch": batch,
                "probes": self._probes,
                "best_so_far": self._best_observed_score(),
            })

        quality = self._best_observed_score()
        # Paper Eq.1 axes: quality − β1·N + β2·(N/k)
        n = self._probes
        k = max(1, self._rounds)
        parallelism = n / k
        cost_term = self.beta1 * n
        bonus = self.beta2 * parallelism
        # Diagnosis: wall of -3.673 = near-zero quality − 0.1*~38 probes.
        # quality_anchored (default): when discovery finds almost no signal,
        # do not let raw probe cost dominate every policy into the same pit.
        mode = (getattr(self, "score_mode", None) or "quality_anchored").lower()
        min_signal = float(getattr(self, "min_quality_signal", 0.08) or 0.08)
        if mode == "paper":
            eff_cost = cost_term
        else:
            weak_scale = float(getattr(self, "weak_quality_cost_scale", 0.2) or 0.2)
            if quality < min_signal:
                eff_cost = cost_term * weak_scale
            else:
                # Progressive cost: full β₁·N only when discovery quality is strong.
                # Stops weak-quality pools from collapsing every policy to ~-3.7.
                scale = min(1.0, weak_scale + quality)
                eff_cost = cost_term * scale
        score = quality - eff_cost + bonus
        # Optional pareto-style parallel penalty term
        effective_seq = sum(
            max(1, (len(t["batch"]) + self.max_parallelism - 1) // self.max_parallelism)
            for t in trajectory
        )
        parallel_penalty = (effective_seq / max(n, 1)) if n else 0.0
        pareto = quality - self.parallel_lambda * parallel_penalty
        breakdown = {
            "mode": mode,
            "quality": round(quality, 4),
            "probes": n,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "raw_cost_term": round(cost_term, 4),
            "effective_cost_term": round(eff_cost, 4),
            "parallel_bonus": round(bonus, 4),
            "parallelism": round(parallelism, 4),
            "pareto": round(pareto, 4),
            "formula": "quality - effective_cost + parallel_bonus",
        }

        anchors = sum(1 for o in self._obs_cache.values() if o.success and o.score > self.baseline_score)
        repairables = sum(
            1 for o in self._obs_cache.values()
            if not o.success and o.fail_class in REPAIRABLE_FAIL_CLASSES
        )

        return ReplayResult(
            world_id=self.world_id,
            policy_name=policy_name,
            quality=round(quality, 4),
            cost=round(self._cost, 4),
            parallelism=round(parallelism, 4),
            replay_score=round(score, 4),
            rounds=self._rounds,
            probes=n,
            opened_branches=len(self.opened_branches()),
            successful_anchors=anchors,
            repairables=repairables,
            trajectory=trajectory,
            score_breakdown=breakdown,
        )

    def observation_signals(self) -> dict:
        """Helpers analogous to Dream-RSI see.policy.observation_signal."""
        branches: dict[int, list[Observation]] = {}
        for o in self._obs_cache.values():
            if o.cell_id == "root":
                continue
            branches.setdefault(o.branch, []).append(o)

        signals = {}
        for b, items in branches.items():
            items = sorted(items, key=lambda x: x.attempt)
            successes = [o for o in items if o.success]
            anchors = [o.score for o in successes]
            latest = items[-1] if items else None
            signals[b] = {
                "branch_promising": bool(anchors) and max(anchors) >= self.baseline_score,
                "branch_failed_hard": bool(latest) and latest.fail_class in HARD_FAIL_CLASSES,
                "has_repairable": any(o.fail_class in REPAIRABLE_FAIL_CLASSES for o in items),
                "best_anchor": max(anchors) if anchors else None,
                "attempts": len(items),
                "latest_fail_class": latest.fail_class if latest else "ok",
            }
        return signals

    # ── Internals ──────────────────────────────────────────────────
    def _eligible(self) -> list[str]:
        """A(T_obs) = {root} ∪ leaves of observed prefix that still have unobserved children."""
        if not self._observed:
            return ["root"] if "root" in self._full else []

        eligible: list[str] = []
        if "root" in self._full and self._has_unrevealed_root_child():
            eligible.append("root")

        for nid in self._observed:
            if nid == "root":
                continue
            node = self._full.get(nid)
            if not node:
                continue
            children = node.get("children", [])
            if any(c not in self._observed for c in children):
                eligible.append(nid)
        return eligible

    def _has_unrevealed_root_child(self) -> bool:
        root = self._full.get("root")
        if not root:
            return False
        return any(c not in self._observed for c in root.get("children", []))

    def _child(self, cell_id: str) -> list[str]:
        """Dream-RSI Child(v; T_full, T_obs)."""
        node = self._full.get(cell_id)
        if not node:
            return []
        children = node.get("children", [])

        if cell_id == "root":
            # Open earliest-created unrevealed child (one branch at a time)
            for c in self._root_creation_order(children):
                if c not in self._observed:
                    return [c]
            return []

        # Non-root: unique recorded child if still unrevealed
        for c in children:
            if c not in self._observed:
                return [c]
        return []

    def _root_creation_order(self, children: list[str]) -> list[str]:
        def ts(cid: str) -> str:
            return str(self._full.get(cid, {}).get("timestamp", ""))
        return sorted(children, key=ts)

    def _reveal(self, cell_id: str) -> None:
        if cell_id in self._observed or cell_id not in self._full:
            return
        node = self._full[cell_id]
        parent_id = node.get("parent_id")
        parent_score = 0.0
        parent_branch = 0
        if parent_id and parent_id in self._full:
            parent_score = float(self._full[parent_id].get("score", 0.0))
            parent_branch = int(self._full[parent_id].get("parallel_group", 0))

        branch = int(node.get("parallel_group", parent_branch or 0))
        if cell_id == "root":
            branch = 0
        elif parent_id == "root":
            self._branch_counter += 1
            branch = self._branch_counter

        score = float(node.get("score", 0.0))
        outcome = str(node.get("outcome", "unknown"))
        fail_class = str(node.get("fail_class", "ok" if outcome == "success" else "unknown"))
        error = str(node.get("error", "" if outcome == "success" else outcome))

        obs = Observation(
            cell_id=cell_id,
            branch=branch,
            attempt=int(node.get("depth", 0)),
            parent_id=parent_id,
            action=str(node.get("action", "")),
            agent_id=str(node.get("agent_id", "")),
            score=score,
            evaluated=True,
            valid=outcome != "failure",
            fail_class="ok" if outcome == "success" else fail_class,
            error="" if outcome == "success" else error,
            delta_vs_baseline=score - self.baseline_score,
            delta_vs_parent=score - parent_score,
            n_valid=1 if outcome == "success" else 0,
            n_total=1,
            cost=float(node.get("cost", 0.0)),
            tags=list(node.get("tags", [])),
        )
        self._observed.add(cell_id)
        self._obs_cache[cell_id] = obs
        if cell_id == "root":
            self._obs_cache[cell_id].branch = 0

    def _best_observed_score(self) -> float:
        scores = [o.score for o in self._obs_cache.values() if o.cell_id != "root"]
        return max(scores) if scores else 0.0

    @staticmethod
    def from_discovery_tree(tree, world_id: str = None, max_parallelism: int = 3) -> "ReplayWorld":
        """Build a ReplayWorld from an ARSI DiscoveryTree."""
        nodes = {}
        for nid, node in tree.nodes.items():
            nodes[nid] = {
                "parent_id": node.parent_id,
                "children": list(node.children),
                "action": node.action,
                "agent_id": node.agent_id,
                "outcome": node.outcome,
                "score": node.score,
                "cost": node.cost,
                "depth": node.depth,
                "timestamp": node.timestamp,
                "parallel_group": node.parallel_group,
                "fail_class": node.metadata.get("fail_class", "unknown"),
                "error": node.metadata.get("error", node.outcome),
                "tags": node.metadata.get("tags", []),
            }
        baseline = 0.0
        if nodes:
            baseline = max(
                (float(n["score"]) for nid, n in nodes.items() if nid != "root"),
                default=0.0,
            )
        return ReplayWorld(
            world_id=world_id or f"world_{id(tree)}",
            nodes=nodes,
            baseline_score=baseline,
            max_parallelism=max_parallelism,
        )
