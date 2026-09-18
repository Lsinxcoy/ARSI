"""Dream-RSI Deep Integration — six mechanisms from arXiv:2609.14858.

Mechanism 1: Batch decision interface (Governor selects multiple actions)
Mechanism 2: Replay objective with quality - cost + parallelism
Mechanism 3: Monotone improvement guarantee (current policy in candidate set)
Mechanism 4: History as simulator, not advice
Mechanism 5: Adaptive behavior curve (spend less when going well, more when stuck)
Mechanism 6: Multi-world evaluation (evaluate across ALL historical trees)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from arsi.foundation.schema import WorldState
from arsi.world_model.discovery_tree import DiscoveryTree

logger = logging.getLogger(__name__)


@dataclass
class ActionBatch:
    """Mechanism 1: Batch decision — multiple actions + parallelism.

    Dream-RSI: the exploration policy selects a batch C ⊆ A(T;W).
    The batch determines both WHERE to explore and HOW MANY in parallel.
    """
    actions: list[str]
    parallelism: int = 1
    scores: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.actions)


@dataclass
class ReplayResult:
    """Mechanism 2: Three-dimensional replay score."""
    discovery_quality: float = 0.0   # max(s_v) — best score found
    execution_cost: float = 0.0      # N — number of attempts
    parallelism: float = 0.0         # N/k — avg attempts per round
    replay_score: float = 0.0        # V = quality - β₁·cost + β₂·parallelism
    rounds: int = 0
    explored_count: int = 0


class DeepReplaySimulator:
    """Dream-RSI deep replay simulator with all six mechanisms."""

    def __init__(self, beta1: float = 0.1, beta2: float = 0.05):
        self.beta1 = beta1  # cost penalty coefficient
        self.beta2 = beta2  # parallelism bonus coefficient

    def evaluate_policy_on_tree(
        self,
        tree: DiscoveryTree,
        policy_fn,
        max_rounds: int = 10,
        max_parallel: int = 3,
    ) -> ReplayResult:
        """Mechanism 2+6: Evaluate policy on one tree with 3D scoring."""
        observed = {"root"}
        total_cost = 0.0
        rounds = 0
        explored = 0

        for _ in range(max_rounds):
            # Get eligible nodes (Mechanism 1: batch decision)
            eligible = self._get_eligible(tree, observed)
            if not eligible:
                break

            # Policy selects a batch (not just one node)
            batch = policy_fn(eligible, observed, max_parallel)
            if not batch:
                break

            # Replay: reveal recorded children deterministically
            for nid in batch[:max_parallel]:
                if nid not in tree.nodes:
                    continue
                node = tree.nodes[nid]
                for child_id in node.children:
                    if child_id not in observed and child_id in tree.nodes:
                        observed.add(child_id)
                        child = tree.nodes[child_id]
                        total_cost += child.cost if hasattr(child, 'cost') else 0
                        explored += 1

            rounds += 1

        # Mechanism 2: Three-dimensional score
        observed_nodes = [tree.nodes[nid] for nid in observed if nid in tree.nodes]
        quality = max((n.score for n in observed_nodes), default=0.0)
        parallelism = explored / max(rounds, 1)
        replay_score = quality - self.beta1 * total_cost + self.beta2 * parallelism

        return ReplayResult(
            discovery_quality=quality,
            execution_cost=total_cost,
            parallelism=parallelism,
            replay_score=replay_score,
            rounds=rounds,
            explored_count=explored,
        )

    def evaluate_across_all_trees(
        self,
        trees: list[DiscoveryTree],
        policy_fn,
        max_rounds: int = 10,
    ) -> dict:
        """Mechanism 6: Evaluate policy across ALL historical trees.

        Dream-RSI: each policy version is evaluated on every historical tree,
        not just the latest one. This ensures generalization.
        """
        results = []
        for tree in trees:
            result = self.evaluate_policy_on_tree(tree, policy_fn, max_rounds)
            results.append(result)

        # Average across all trees
        avg_score = sum(r.replay_score for r in results) / max(len(results), 1)
        avg_quality = sum(r.discovery_quality for r in results) / max(len(results), 1)

        return {
            "avg_replay_score": avg_score,
            "avg_discovery_quality": avg_quality,
            "tree_count": len(results),
            "per_tree": [
                {"score": r.replay_score, "quality": r.discovery_quality, "rounds": r.rounds}
                for r in results
            ],
        }

    def select_best_policy(
        self,
        current_policy_fn,
        candidate_policy_fns: list,
        trees: list[DiscoveryTree],
    ) -> tuple:
        """Mechanism 3: Monotone improvement guarantee.

        Candidate set includes current policy → V(m*) ≥ V(0).
        This is a FREE safety net.
        """
        all_policies = [current_policy_fn] + candidate_policy_fns

        best_policy = current_policy_fn
        best_score = -float("inf")

        for i, policy_fn in enumerate(all_policies):
            eval_result = self.evaluate_across_all_trees(trees, policy_fn)
            score = eval_result["avg_replay_score"]

            logger.info(f"Policy {i}: replay_score={score:.4f}")

            if score > best_score:
                best_score = score
                best_policy = policy_fn

        # Mechanism 3: since current policy is in the set,
        # best_score >= current_policy_score (monotone guarantee)
        return best_policy, best_score

    @staticmethod
    def _get_eligible(tree: DiscoveryTree, observed: set) -> list[str]:
        """Get eligible nodes: root + observed leaves with unexplored children."""
        eligible = []
        for nid in observed:
            if nid not in tree.nodes:
                continue
            node = tree.nodes[nid]
            if nid == "root":
                eligible.append(nid)
            elif node.children:
                unexplored = [c for c in node.children if c not in observed]
                if unexplored:
                    eligible.append(nid)
        return eligible


class AdaptiveBehaviorController:
    """Mechanism 5: Spend less when going well, more when stuck.

    Dream-RSI observation: learned policies reduce exploration during
    performance growth, then increase it when progress plateaus.
    """

    def __init__(self):
        self.performance_history: list[float] = []
        self.budget_history: list[int] = []
        self._base_budget = 3

    def record_performance(self, score: float):
        """Record a performance measurement."""
        self.performance_history.append(score)

    def recommend_budget(self) -> int:
        """Recommend exploration budget based on performance trend.

        Dream-RSI behavior curve:
        - Performance rising → reduce exploration (save resources)
        - Performance plateau → increase exploration (break through)
        - Performance declining → moderate exploration (stabilize)
        """
        if len(self.performance_history) < 3:
            return self._base_budget  # Not enough data

        recent = self.performance_history[-3:]
        older = self.performance_history[-6:-3] if len(self.performance_history) >= 6 else recent

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        trend = recent_avg - older_avg

        if trend > 0.05:
            # Performance rising → save resources
            budget = max(1, self._base_budget - 1)
            reason = "performance_rising"
        elif trend < -0.05:
            # Performance declining → moderate exploration
            budget = self._base_budget
            reason = "performance_declining"
        else:
            # Performance plateau → increase exploration
            budget = min(5, self._base_budget + 2)
            reason = "performance_plateau"

        self.budget_history.append(budget)
        logger.info(f"Budget recommendation: {budget} ({reason}, trend={trend:+.3f})")
        return budget

    @property
    def current_trend(self) -> str:
        if len(self.performance_history) < 3:
            return "insufficient_data"
        recent = self.performance_history[-3:]
        older = self.performance_history[-6:-3] if len(self.performance_history) >= 6 else recent
        trend = sum(recent)/len(recent) - sum(older)/len(older)
        if trend > 0.05:
            return "rising"
        elif trend < -0.05:
            return "declining"
        else:
            return "plateau"

    @property
    def stats(self) -> dict:
        return {
            "performance_count": len(self.performance_history),
            "current_trend": self.current_trend,
            "avg_budget": sum(self.budget_history) / max(len(self.budget_history), 1),
            "recent_performance": self.performance_history[-5:] if self.performance_history else [],
        }


class HistorySimulator:
    """Mechanism 4: History as interactive simulator, not advice.

    Dream-RSI ablation: using history as semantic guidance is WORSE
    than not using it. History must be a queryable simulator.
    """

    def __init__(self, tree: DiscoveryTree):
        self.tree = tree

    def query(self, task_pattern: str) -> dict:
        """Query historical performance for a task pattern.

        Returns STRUCTURED DATA, not advice text.
        """
        # Find relevant nodes
        relevant = []
        for node in self.tree.nodes.values():
            if node.action == "start":
                continue
            if any(w in node.action.lower() for w in task_pattern.lower().split()):
                relevant.append(node)

        if len(relevant) < 2:
            return {"available": False, "reason": "insufficient_data"}

        successes = [n for n in relevant if n.outcome == "success"]
        failures = [n for n in relevant if n.outcome == "failure"]

        # Structured data, NOT advice
        return {
            "available": True,
            "pattern": task_pattern,
            "total_attempts": len(relevant),
            "success_count": len(successes),
            "failure_count": len(failures),
            "success_rate": len(successes) / len(relevant),
            "avg_score": sum(n.score for n in relevant) / len(relevant),
            "best_score": max((n.score for n in relevant), default=0),
            "worst_score": min((n.score for n in relevant), default=0),
            "score_std": self._std([n.score for n in relevant]),
            "common_actions": self._top_actions(relevant),
        }

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _top_actions(nodes: list, n: int = 3) -> list[dict]:
        action_counts = {}
        for node in nodes:
            action_counts[node.action] = action_counts.get(node.action, 0) + 1
        sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
        return [{"action": a, "count": c} for a, c in sorted_actions[:n]]
