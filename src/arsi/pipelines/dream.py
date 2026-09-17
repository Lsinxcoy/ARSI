"""Dream Pipeline — self-model refresh + memory consolidation.

When η ≥ θ_high, dream pipeline:
1. First-person observation (Ω_ε)
2. State re-parse (MindZero)
3. Belief reconstruction
4. Memory consolidation (decay + edge discovery)
5. η recomputation

Based on: ARSI Whitepaper v0.8 §7.3
"""
from __future__ import annotations

import logging
from datetime import datetime

from arsi.foundation.schema import MentalState, WorldState
from arsi.mnemosyne.core import Mnemosyne
from arsi.world_model.siwm import SIWM

logger = logging.getLogger(__name__)


class DreamSession:
    """Record of a dream cycle (auditable)."""

    def __init__(
        self,
        before_psi: MentalState,
        after_psi: MentalState,
        eta_before: float,
        eta_after: float,
    ):
        self.before_psi = before_psi
        self.after_psi = after_psi
        self.eta_before = eta_before
        self.eta_after = eta_after
        self.timestamp = datetime.now()

    def to_dict(self) -> dict:
        return {
            "belief_count_before": len(self.before_psi.beliefs),
            "belief_count_after": len(self.after_psi.beliefs),
            "eta_before": self.eta_before,
            "eta_after": self.eta_after,
            "eta_improved": self.eta_after < self.eta_before,
            "timestamp": self.timestamp.isoformat(),
        }


class DreamPipeline:
    """Dream pipeline — the system's sleep cycle.

    Not rest: deep processing of memory and self-model.
    Enhanced with LLM-powered belief reconciliation when available.
    """

    def __init__(self, siwm: SIWM, mnemosyne: Mnemosyne, llm=None):
        self.siwm = siwm
        self.mnemosyne = mnemosyne
        self.llm = llm
        self._dream_count = 0
        self._llm_reconcile_count = 0

    def execute(self, state: WorldState) -> WorldState:
        """Execute a dream cycle."""
        self._dream_count += 1
        eta_before = state.eta

        # 1. First-person observation: render current state
        observation = self._first_person_observe(state)

        # 2. Re-parse mental state from recent traces
        traces = self.mnemosyne.store.get_recent_traces(n=50)
        fresh_psi = self.siwm.mindzero.infer_mental_state(traces)

        # 3. Reconcile beliefs (LLM-powered when available)
        corrected_beliefs = self._reconcile_beliefs(
            state.psi.beliefs, fresh_psi.beliefs
        )

        # 4. Memory consolidation
        consolidation_result = self.mnemosyne.consolidate()

        # 5. Recompute η (reset to lower value after dream)
        # Dream reduces η because we've refreshed the self-model
        new_eta = max(0.0, eta_before * 0.5)  # Halve η after dream
        self.siwm.eta.eta_smooth = new_eta

        # 6. Record dream session
        session = DreamSession(
            before_psi=state.psi,
            after_psi=fresh_psi,
            eta_before=eta_before,
            eta_after=new_eta,
        )
        self.mnemosyne.write_self_record("dream_session", session.to_dict())

        logger.info(
            f" Dream #{self._dream_count}: η {eta_before:.3f} → {new_eta:.3f}, "
            f"consolidated {consolidation_result}"
        )

        # Return updated state
        return state.model_copy(update={
            "psi": fresh_psi.model_copy(update={"beliefs": corrected_beliefs}),
            "eta": new_eta,
        })

    def _first_person_observe(self, state: WorldState) -> dict:
        """Ω_ε — render system's first-person partial observation."""
        return {
            "generation": state.phi.generation,
            "belief_count": len(state.psi.beliefs),
            "eta": state.eta,
            "storage": state.phi.storage_stats,
        }

    def _reconcile_beliefs(self, old: list, new: list) -> list:
        """Reconcile old and new beliefs.

        Uses LLM for semantic comparison when available.
        Falls back to confidence-based heuristic.
        """
        # Try LLM-powered reconciliation
        if self.llm and self.llm.available and (old or new):
            try:
                result = self._llm_reconcile(old, new)
                if result:
                    self._llm_reconcile_count += 1
                    return result
            except Exception as e:
                logger.warning(f"LLM belief reconciliation failed, using heuristic: {e}")

        # Heuristic fallback: prefer new, keep high-confidence old
        reconciled = list(new)
        for old_belief in old:
            if old_belief.confidence > 0.7:
                contradicted = any(
                    self._beliefs_contradict(old_belief, nb) for nb in new
                )
                if not contradicted:
                    reconciled.append(old_belief)
        return reconciled

    def _llm_reconcile(self, old: list, new: list) -> list:
        """LLM-powered belief reconciliation."""
        import json

        old_str = json.dumps(
            [{"content": b.content, "confidence": b.confidence} for b in old[-10:]],
            ensure_ascii=False,
        )
        new_str = json.dumps(
            [{"content": b.content, "confidence": b.confidence} for b in new[-10:]],
            ensure_ascii=False,
        )

        prompt = f"""调和 AI 系统的新旧信念。保留准确的，修正过时的，合并重复的。

旧信念（之前推断的）：
{old_str}

新信念（从最近行为重新推断的）：
{new_str}

调和原则：
1. 新信念与实际行为一致时优先保留新信念
2. 旧信念置信度高且未被新证据推翻时保留
3. 新旧信念表达相同意图时合并（取更高置信度）
4. 矛盾时以新信念为准

输出 JSON：
{{"reconciled": [{{"content": "调和后的信念", "confidence": 0.0到1.0, "source": "old|new|merged"}}]}}"""

        resp = self.llm.chat(
            prompt,
            system="你是信念调和专家。只输出 JSON。",
            max_tokens=800,
        )

        if not resp.success:
            return None

        data = self._parse_json(resp.content)
        if not data or "reconciled" not in data:
            return None

        from arsi.foundation.schema import Belief
        reconciled = []
        for b in data["reconciled"]:
            if isinstance(b, dict) and "content" in b:
                reconciled.append(Belief(
                    content=b["content"],
                    confidence=max(0.0, min(1.0, b.get("confidence", 0.5))),
                    source=f"llm_dream:{b.get('source', 'unknown')}",
                ))
        return reconciled if reconciled else None

    @staticmethod
    def _parse_json(content: str):
        import json
        c = content.strip()
        if c.startswith("```json"):
            c = c[7:]
        if c.startswith("```"):
            c = c[3:]
        if c.endswith("```"):
            c = c[:-3]
        c = c.strip()
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            start = c.find("{")
            end = c.rfind("}")
            if start != -1 and end > start:
                try:
                    return json.loads(c[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return None

    @staticmethod
    def _beliefs_contradict(a, b) -> bool:
        """Simple contradiction check (simplified)."""
        # In production, would use semantic similarity
        return False

    @property
    def stats(self) -> dict:
        return {
            "dream_count": self._dream_count,
            "llm_reconcile_count": self._llm_reconcile_count,
        }
