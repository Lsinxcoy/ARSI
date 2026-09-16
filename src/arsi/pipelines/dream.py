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
    """

    def __init__(self, siwm: SIWM, mnemosyne: Mnemosyne):
        self.siwm = siwm
        self.mnemosyne = mnemosyne
        self._dream_count = 0

    def execute(self, state: WorldState) -> WorldState:
        """Execute a dream cycle."""
        self._dream_count += 1
        eta_before = state.eta

        # 1. First-person observation: render current state
        observation = self._first_person_observe(state)

        # 2. Re-parse mental state from recent traces
        traces = self.mnemosyne.store.get_recent_traces(n=50)
        fresh_psi = self.siwm.mindzero.infer_mental_state(traces)

        # 3. Reconcile beliefs (keep accurate, correct inaccurate)
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

        Keep old beliefs that are still consistent with new observations.
        Add new beliefs that explain recent behavior.
        """
        # Simple reconciliation: prefer new beliefs, keep high-confidence old ones
        reconciled = list(new)  # Start with new inferences

        for old_belief in old:
            # Keep old belief if it has high confidence and isn't contradicted
            if old_belief.confidence > 0.7:
                # Check if any new belief contradicts
                contradicted = any(
                    self._beliefs_contradict(old_belief, nb) for nb in new
                )
                if not contradicted:
                    reconciled.append(old_belief)

        return reconciled

    @staticmethod
    def _beliefs_contradict(a, b) -> bool:
        """Simple contradiction check (simplified)."""
        # In production, would use semantic similarity
        return False

    @property
    def stats(self) -> dict:
        return {"dream_count": self._dream_count}
