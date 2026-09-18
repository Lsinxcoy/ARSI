"""Autonomy Ladder — L3/L5 components.

Paper ① (arXiv:2609.11873): Five-level autonomy ladder for RSI.
  L3: Experience acquisition autonomy — AI decides what to learn from
  L5: Recursive inheritance autonomy — AI improves its own improvement mechanism

This module implements the L3 (AdaptiveTraceSelector) and L5 (MetaImprover)
components for ARSI.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.schema import EmpowermentDimension
from arsi.foundation.store import MnemosyneStore

logger = logging.getLogger(__name__)


class AdaptiveTraceSelector:
    """L3: ARSI autonomously decides which traces to analyze deeply.

    Instead of analyzing all traces equally, ARSI selects the most
    novel, impactful, and relevant traces for deep analysis.

    Based on: NeoHorse-1 insight — real production traffic is the
    strongest supervision, but not all traffic is equally valuable.
    """

    def __init__(self, store: MnemosyneStore):
        self.store = store
        self._selection_count = 0
        self._historical_patterns: dict[str, int] = {}

    def select_for_analysis(self, traces: list[dict], max_count: int = 50) -> list[dict]:
        """Select traces worth deep analysis based on novelty + impact + relevance."""
        if len(traces) <= max_count:
            return traces

        scored = []
        for t in traces:
            novelty = self._compute_novelty(t)
            impact = abs(t.get("effect", 0))
            relevance = self._compute_relevance(t)
            outcome_diversity = self._outcome_diversity_score(t)

            # Weighted score: novelty is most important for L3
            score = 0.35 * novelty + 0.25 * impact + 0.25 * relevance + 0.15 * outcome_diversity
            scored.append((t, score))

        # Sort by score, take top N
        scored.sort(key=lambda x: -x[1])
        selected = [t for t, _ in scored[:max_count]]

        # Record selection patterns
        self._selection_count += 1
        for t in selected:
            action = t.get("action", "unknown")
            self._historical_patterns[action] = self._historical_patterns.get(action, 0) + 1

        logger.info(f"L3 selection #{self._selection_count}: {len(selected)}/{len(traces)} traces")
        return selected

    def _compute_novelty(self, trace: dict) -> float:
        """How novel is this trace? (unseen action/outcome combinations)"""
        action = trace.get("action", "unknown")
        outcome = trace.get("outcome", "unknown")
        key = f"{action}:{outcome}"

        seen_count = self._historical_patterns.get(key, 0)
        # Novelty: unseen = 1.0, seen many times = low
        novelty = 1.0 / (1.0 + seen_count)
        return novelty

    def _compute_relevance(self, trace: dict) -> float:
        """How relevant is this trace to current system needs?"""
        # Check if action relates to known weak dimensions
        action = trace.get("action", "")
        weak_dims = ["knowledge", "calibration", "metacognition", "environment"]

        for dim in weak_dims:
            if dim in action.lower():
                return 0.8  # High relevance to weak dimension
        return 0.3  # Default relevance

    def _outcome_diversity_score(self, trace: dict) -> float:
        """Prefer traces with unusual outcomes (not just success/failure)."""
        outcome = trace.get("outcome", "")
        if outcome in ("success", "failure"):
            return 0.2  # Common outcomes
        return 0.8  # Unusual outcomes (partial, recorded, etc.)

    @property
    def stats(self) -> dict:
        return {
            "selection_count": self._selection_count,
            "pattern_count": len(self._historical_patterns),
            "top_patterns": dict(sorted(self._historical_patterns.items(), key=lambda x: -x[1])[:5]),
        }


class MetaImprover:
    """L5: Improves the improvement mechanism itself.

    MetaRSI: the outermost agent revises the scheduling policy.
    ARSI: the diagnostic algorithms themselves get improved.

    Key: track whether ARSI's diagnoses led to actual improvements.
    If not, revise HOW ARSI diagnoses, not just WHAT it recommends.
    """

    def __init__(self, store: MnemosyneStore):
        self.store = store
        self._diagnostic_history: list[dict] = []
        self._policy_revisions: list[dict] = []

    def evaluate_diagnostic_quality(self, dimension: str) -> dict:
        """How good is ARSI's diagnosis for this dimension?

        Tracks: when ARSI diagnosed X, did the agent actually improve?
        """
        # Get empowerment records for this dimension
        records = self.store.search_memories(
            zone=__import__("arsi.foundation.schema", fromlist=["MemoryZone"]).MemoryZone.EXPERIENCE,
            tags=[dimension, "empowerment"],
            limit=50,
        )

        if not records:
            return {
                "dimension": dimension,
                "diagnosis_count": 0,
                "accuracy": None,
                "status": "no_data",
            }

        # Count successful vs failed diagnoses
        successful = sum(1 for r in records if "success" in r.content.lower())
        total = len(records)
        accuracy = successful / max(total, 1)

        return {
            "dimension": dimension,
            "diagnosis_count": total,
            "successful": successful,
            "accuracy": round(accuracy, 3),
            "status": "tracked",
        }

    def revise_diagnostic_algorithm(self, dimension: str, feedback: dict) -> dict:
        """L5: Revise how ARSI diagnoses a dimension.

        If a dimension's diagnoses consistently fail to lead to improvements,
        change the diagnostic approach.
        """
        quality = self.evaluate_diagnostic_quality(dimension)

        if quality["accuracy"] is None:
            return {"status": "no_data_to_revise"}

        if quality["accuracy"] >= 0.6:
            return {"status": "diagnostic_quality_acceptable", "accuracy": quality["accuracy"]}

        # Low accuracy → propose revision
        revision = {
            "dimension": dimension,
            "old_accuracy": quality["accuracy"],
            "revision_type": "diagnostic_approach",
            "proposal": self._propose_revision(dimension, feedback),
            "timestamp": datetime.now().isoformat(),
        }

        self._policy_revisions.append(revision)
        logger.info(f"L5 revision proposed for {dimension}: accuracy={quality['accuracy']}")

        return revision

    def _propose_revision(self, dimension: str, feedback: dict) -> str:
        """Propose how to revise the diagnostic algorithm."""
        proposals = {
            "knowledge": "Switch from frequency-based to LLM-semantic gap detection",
            "calibration": "Use rolling window instead of all-time ECE",
            "metacognition": "Track consecutive failures per action, not global",
            "environment": "Analyze per-agent failure rates, not aggregate",
            "decomposition": "Use LLM to analyze task complexity, not just action counts",
            "attention": "Measure context-action correlation, not just param size",
        }
        return proposals.get(dimension, "Increase LLM involvement in diagnosis")

    @property
    def stats(self) -> dict:
        return {
            "diagnostic_history_count": len(self._diagnostic_history),
            "policy_revision_count": len(self._policy_revisions),
            "dimensions_tracked": len(set(r["dimension"] for r in self._policy_revisions)) if self._policy_revisions else 0,
        }
