"""Portfolio exploration policy — Dream-RSI Appendix B.2 dynamic batch.

Policy builds one portfolio batch per decision round:
  - exploitation: strong normal refinements
  - exploration: root openings / underexplored frontiers
  - recovery: at most one repairable failure

Hard rules from the paper prompt:
  - prefix-only signals
  - never sample randomly
  - never put parent+child in the same batch
  - no fixed widen-all/deepen-all wave
  - beta fixed within an episode; high beta = wider + more patient
"""
from __future__ import annotations

import logging
from typing import Optional

from arsi.world_model.replay_world import HARD_FAIL_CLASSES, REPAIRABLE_FAIL_CLASSES, Observation

logger = logging.getLogger(__name__)


def beta_schedule(beta: float) -> dict:
    """Route all behavioral thresholds through one beta knob (Appendix B.2)."""
    b = max(0.0, min(1.0, float(beta)))
    return {
        "beta": b,
        "width_bias": 0.3 + 0.7 * b,
        "patience": 2 + int(round(4 * b)),
        "prune_threshold": 0.55 - 0.25 * b,
        "reserve_threshold": 0.35 - 0.15 * b,
        "recovery_allowed": b >= 0.25,
        "max_batch_bonus": int(round(2 * b)),
    }


def default_beta_from_live(history: list[dict] | None = None) -> float:
    """Cross-cycle default-beta rule (Appendix B.2).

    - history is None → read disk ManifestStore (production loop)
    - history is a list (even empty/short) → use only those rows
    - fewer than 2 usable rows → bootstrap 0.6
    """
    if history is None:
        try:
            from arsi.meta.live_manifest import ManifestStore
            rows = ManifestStore().beta_history(n=5)
        except Exception:
            rows = []
    else:
        rows = list(history)
    if len(rows) < 2:
        return 0.6  # paper: moderately exploratory bootstrap
    recent = rows[-3:]
    scores = [float(h.get("best_score", 0.0)) for h in recent]
    betas = [float(h.get("beta", 0.6)) for h in recent]
    # Prefer sweep-selected beta when present in row
    if recent and "selected_beta" in recent[-1]:
        try:
            return max(0.0, min(1.0, float(recent[-1]["selected_beta"])))
        except Exception:
            pass
    if len(scores) >= 2 and scores[-1] > scores[-2] + 1e-6:
        return betas[-1]
    plateau = len(scores) >= 3 and abs(scores[-1] - scores[-2]) < 1e-3
    if plateau:
        return max(0.0, min(1.0, betas[-1] + 0.15))
    return betas[-1]


class PortfolioPolicy:
    """Prefix-only dynamic portfolio batch builder."""

    def __init__(self, beta: float = 0.6, max_workers: int = 3, name: str = "portfolio"):
        self.beta = beta
        self.max_workers = max_workers
        self.name = name
        self.schedule = beta_schedule(beta)

    def select(self, observed: dict[str, Observation], legal: list[str], max_parallelism: int = 3) -> list[str]:
        if not legal:
            return []
        W = max(1, min(max_parallelism or self.max_workers, self.max_workers or max_parallelism))
        sched = self.schedule

        # Classify legal cells
        roots = [c for c in legal if c == "root" or (c in observed and observed[c].parent_id == "root")]
        # legal frontiers are non-root cells with unexplored children
        frontiers = [c for c in legal if c != "root"]

        exploit_scores = []
        explore_scores = []
        recovery_scores = []

        baseline = 0.0
        if observed:
            baseline = max((o.score for cid, o in observed.items() if cid != "root"), default=0.0)

        for cell in frontiers:
            obs = observed.get(cell)
            if not obs:
                explore_scores.append((0.0, cell))
                continue
            if obs.fail_class in REPAIRABLE_FAIL_CLASSES and sched["recovery_allowed"]:
                recovery_scores.append((obs.score, cell))
            elif obs.success and obs.score >= baseline:
                exploit_scores.append((obs.score, cell))
            else:
                # underexplored / weak
                explore_scores.append((obs.score, cell))

        if "root" in legal:
            explore_scores.append((0.01, "root"))

        exploit_scores.sort(key=lambda x: -x[0])
        explore_scores.sort(key=lambda x: -x[0])
        recovery_scores.sort(key=lambda x: -x[0])

        batch: list[str] = []
        used = set()

        def add(cell: str) -> bool:
            if cell in used or cell not in legal:
                return False
            # never parent+child together
            parent = observed[cell].parent_id if cell in observed else None
            if parent and parent in used:
                return False
            if any(observed[c].parent_id == cell for c in used if c in observed):
                return False
            if len(batch) >= W:
                return False
            batch.append(cell)
            used.add(cell)
            return True

        # Portfolio: give exploration + justified recovery representation first
        # (paper: exploration and recovery before filling remaining by priority)
        n_explore = 1 if explore_scores else 0
        if W >= 3:
            n_explore = min(2, len(explore_scores))
        for _, cell in explore_scores[:n_explore]:
            add(cell)

        if recovery_scores and sched["recovery_allowed"]:
            add(recovery_scores[0][1])

        for _, cell in exploit_scores:
            if len(batch) >= W:
                break
            add(cell)

        # fill remaining
        for _, cell in explore_scores + exploit_scores + recovery_scores:
            if len(batch) >= W:
                break
            add(cell)

        return batch


def build_policy_fn(policy: PortfolioPolicy):
    """Adapter to ReplayWorld.replay(policy_fn)."""

    def _fn(observed: dict[str, Observation], legal: list[str], max_parallelism: int) -> list[str]:
        return policy.select(observed, legal, max_parallelism)

    return _fn
