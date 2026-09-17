"""Mnemosyne — unified memory substrate with three-zone isolation.

Based on: ARSI Whitepaper v0.8 §8 + SYNTHEX memory module inheritance.
Zones:
  - SELF: ARSI's identity (never deleted, improvement operators cannot rewrite core)
  - EXPERIENCE: distilled wisdom (cross-agent shared, can be archived)
  - PROXY: host agent raw traces (feed, can be cleaned after distillation)
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from arsi.foundation.schema import (
    CausalLink,
    MemoryRecord,
    MemoryZone,
    BehaviorTrace,
)
from arsi.foundation.store import MnemosyneStore

logger = logging.getLogger(__name__)


class DopamineGate:
    """Write gate — not all traces deserve to be stored.

    Inherited from SYNTHEX's dopamine.py.
    Decides whether a memory should be immediately consolidated
    to EXPERIENCE zone or stay in PROXY as raw material.
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

    def should_consolidate(self, record: MemoryRecord) -> bool:
        """High-importance or high-surprise records get consolidated immediately."""
        return record.importance >= self.threshold

    def score(self, record: MemoryRecord, effect: float = 0.0, surprise: float = 0.0) -> float:
        """Compute consolidation score: importance * alpha + surprise * beta."""
        return record.importance * 0.6 + surprise * 0.4


class UtilityDecay:
    """Usefulness-based forgetting — 'forget the useless', not 'forget the old'.

    Inherited from SYNTHEX's utility_decay.py.
    A memory accessed 5 times in 3 days is 'younger' than
    a 1-day-old memory never accessed.
    """

    def __init__(self, decay_rate: float = 0.01, min_utility: float = 0.05):
        self.decay_rate = decay_rate
        self.min_utility = min_utility

    def apply(self, store: MnemosyneStore, zone: MemoryZone = MemoryZone.PROXY) -> int:
        """Decay utility scores and archive memories below threshold.
        Returns number of archived memories."""
        records = store.search_memories(zone=zone, limit=10000)
        archived = 0
        for r in records:
            new_utility = r.utility_score - self.decay_rate * (1.0 - r.freshness)
            if new_utility < self.min_utility:
                store.archive_memory(r.id)
                archived += 1
            else:
                store.write_memory(r.model_copy(update={"utility_score": max(0, new_utility)}))
        return archived


class EdgeDiscovery:
    """Active association discovery — find hidden links between experiences.

    Inherited from SYNTHEX's edge_discovery.py.
    Periodically scans the EXPERIENCE zone for related patterns.
    """

    def __init__(self, store: MnemosyneStore):
        self.store = store

    def scan(self, min_shared_tags: int = 1) -> list[dict]:
        """Scan for experiences with shared tags (simplified association)."""
        experiences = self.store.search_memories(zone=MemoryZone.EXPERIENCE, limit=500)
        associations = []
        for i, a in enumerate(experiences):
            for b in experiences[i + 1:]:
                shared = set(a.tags) & set(b.tags)
                if len(shared) >= min_shared_tags:
                    associations.append({
                        "source": a.id,
                        "target": b.id,
                        "shared_tags": list(shared),
                        "strength": len(shared) / max(len(set(a.tags) | set(b.tags)), 1),
                    })
        associations.sort(key=lambda x: x["strength"], reverse=True)
        return associations[:20]


class Mnemosyne:
    """Unified memory substrate — the brain's memory system.

    Three zones + causal chains + cross-agent experience sharing + semantic retrieval.
    """

    def __init__(self, store: MnemosyneStore):
        self.store = store
        self.dopamine = DopamineGate()
        self.decay = UtilityDecay()
        self.edge_discovery = EdgeDiscovery(store)
        # Lazy-loaded semantic retriever
        self._retriever = None

    @property
    def retriever(self):
        """Semantic retriever (lazy-initialized)."""
        if self._retriever is None:
            from arsi.mnemosyne.semantic_retrieval import SemanticRetriever
            self._retriever = SemanticRetriever(self.store)
            self._retriever.index_all()
        return self._retriever

    @property
    def current_generation(self) -> int:
        return self.store.current_generation

    # ── PROXY zone (agent traces) ───────────────────────────────

    def ingest_trace(self, trace: BehaviorTrace) -> MemoryRecord:
        """Ingest a behavior trace into PROXY zone."""
        record = MemoryRecord(
            zone=MemoryZone.PROXY,
            content=f"[{trace.action}] {trace.outcome} (effect={trace.effect:.2f})",
            tags=[trace.agent_id, trace.action],
            agent_id=trace.agent_id,
            generation=trace.generation,
            importance=min(1.0, abs(trace.effect)),
        )
        self.store.write_memory(record)
        self.store.write_trace(trace)

        # Dopamine gate: should we consolidate immediately?
        if self.dopamine.should_consolidate(record):
            self._distill(record)

        return record

    # ── EXPERIENCE zone (distilled wisdom) ───────────────────────

    def _distill(self, proxy_record: MemoryRecord) -> MemoryRecord:
        """Distill a PROXY record into EXPERIENCE zone with causal chain."""
        experience = MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content=proxy_record.content,
            tags=proxy_record.tags + ["distilled"],
            agent_id=proxy_record.agent_id,
            generation=proxy_record.generation,
            importance=proxy_record.importance,
        )
        self.store.write_memory(experience)

        # Write causal chain
        self.store.write_causal_link(CausalLink(
            experience_id=experience.id,
            source_traces=[proxy_record.id],
        ))

        logger.info(f"Distilled {proxy_record.id} → {experience.id}")
        return experience

    def record_empowerment(self, op, causal_chain_id: str) -> None:
        """Record an empowerment operation into EXPERIENCE zone."""
        record = MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content=f"[empower:{op.dimension.value}] agent={op.target_agent} status={op.verification.value}",
            tags=[op.dimension.value, op.target_agent, "empowerment"],
            agent_id=op.target_agent,
            generation=self.current_generation,
            importance=0.7 if op.verification.value == "success" else 0.3,
        )
        self.store.write_memory(record)

    # ── SELF zone (identity) ─────────────────────────────────────

    def write_self_record(self, kind: str, data: dict) -> int:
        """Write to SELF zone (dream sessions, reports, etc.)."""
        return self.store.write_self_record(kind, data)

    # ── Cross-agent experience search ────────────────────────────

    def search_experience(self, query: str, agent_id: Optional[str] = None, k: int = 10) -> list[MemoryRecord]:
        """Search EXPERIENCE zone using semantic retrieval.

        Args:
            query: Natural language query (not just tags)
            agent_id: Boost results from this agent but include others
            k: Max results
        """
        try:
            # Use semantic retriever
            results_with_scores = self.retriever.search(
                query, k=k, zone=MemoryZone.EXPERIENCE
            )
            results = [r for r, s in results_with_scores]
        except Exception:
            # Fallback to tag matching
            results = self.store.search_memories(
                zone=MemoryZone.EXPERIENCE, limit=k
            )

        # Boost same-agent results
        if agent_id:
            results.sort(key=lambda r: (0 if r.agent_id == agent_id else 1, -r.utility_score))
        return results

    def search_cross_agent(self, query: str, exclude_agent: Optional[str] = None, k: int = 5) -> list[MemoryRecord]:
        """Search for experiences from OTHER agents (cross-agent sharing)."""
        try:
            results_with_scores = self.retriever.search_cross_agent(
                query, k=k, exclude_agent=exclude_agent
            )
            return [r for r, s in results_with_scores]
        except Exception:
            return self.store.search_memories(zone=MemoryZone.EXPERIENCE, limit=k)

    # ── Consolidation cycle ──────────────────────────────────────

    def consolidate(self) -> dict:
        """Run a consolidation cycle: decay + edge discovery."""
        archived = self.decay.apply(self.store, zone=MemoryZone.PROXY)
        associations = self.edge_discovery.scan()
        return {"archived": archived, "associations_found": len(associations)}

    # ── Stats ────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        stats = self.store.get_stats()
        stats["dopamine_threshold"] = self.dopamine.threshold
        stats["decay_rate"] = self.decay.decay_rate
        return stats
