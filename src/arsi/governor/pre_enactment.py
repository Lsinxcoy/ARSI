"""Pre-enactment engine — predict consequences before executing.

Connects Governor's candidate actions to SIWM Layer 2 dynamics model.
Before executing an action, simulate its consequences and pick the best.

Based on: ARSI Whitepaper §7.2 (Branch Pre-enactment)
"""
from __future__ import annotations

import logging
from typing import Optional

from arsi.foundation.schema import WorldState
from arsi.world_model.dynamics import DynamicsModel

logger = logging.getLogger(__name__)


class PreEnactmentEngine:
    """Predicts consequences of candidate actions using Layer 2 dynamics.

    Governor calls this before executing to pick the best action.
    Falls back to heuristic scoring when dynamics model is untrained.
    """

    def __init__(self, dynamics: DynamicsModel):
        self.dynamics = dynamics
        self._prediction_count = 0
        self._pre_enactment_count = 0

    def evaluate_candidates(
        self,
        state: WorldState,
        candidates: list[str],
    ) -> list[dict]:
        """Evaluate all candidate actions via pre-enactment.

        Returns list of {action, predicted_delta, confidence, score} sorted by score.
        """
        self._pre_enactment_count += 1
        evaluated = []

        for action in candidates:
            pred = self.dynamics.predict_transition(state, action)
            self._prediction_count += 1

            # Score: how beneficial is the predicted delta?
            score = self._score_prediction(pred, state)

            evaluated.append({
                "action": action,
                "predicted_delta": pred.get("predicted_delta", {}),
                "confidence": pred.get("confidence", 0.0),
                "score": round(score, 4),
                "source": "pre_enactment" if pred.get("confidence", 0) > 0 else "no_data",
            })

        # Sort by score descending
        evaluated.sort(key=lambda x: x["score"], reverse=True)
        return evaluated

    def select_best(
        self,
        state: WorldState,
        candidates: list[str],
    ) -> dict:
        """Select the best action via pre-enactment.

        Returns {action, reason, confidence, score}.
        """
        evaluated = self.evaluate_candidates(state, candidates)

        if not evaluated:
            return {
                "action": candidates[0] if candidates else "remember",
                "reason": "no_candidates",
                "confidence": 0.0,
                "score": 0.0,
            }

        best = evaluated[0]

        # If confidence is too low, fall back to heuristic
        if best["confidence"] < 0.1:
            return self._heuristic_select(state, candidates, evaluated)

        return {
            "action": best["action"],
            "reason": f"pre_enactment: score={best['score']}, confidence={best['confidence']}",
            "confidence": best["confidence"],
            "score": best["score"],
            "alternatives": [
                {"action": e["action"], "score": e["score"]}
                for e in evaluated[1:3]
            ],
        }

    def _score_prediction(self, pred: dict, state: WorldState) -> float:
        """Score a predicted state transition.

        Higher score = more beneficial transition.
        """
        confidence = pred.get("confidence", 0.0)
        if confidence <= 0:
            return 0.0

        delta = pred.get("predicted_delta", {})

        # Scoring heuristics:
        # - eta decrease is good (self-model improving)
        # - experience_count increase is good (learning)
        # - trace_count increase is neutral (just more data)
        # - generation increase is neutral

        score = 0.0

        # η reduction is valuable
        eta_delta = delta.get("eta", 0)
        if eta_delta < 0:
            score += abs(eta_delta) * 3  # Reward η reduction
        elif eta_delta > 0:
            score -= eta_delta * 0.5  # Penalize η increase slightly

        # Experience growth is valuable
        exp_delta = delta.get("experience_count", 0)
        if exp_delta > 0:
            score += min(exp_delta * 0.1, 1.0)

        # Proxy reduction is good (distilled)
        proxy_delta = delta.get("proxy_count", 0)
        if proxy_delta < 0:
            score += abs(proxy_delta) * 0.05

        # Weight by confidence
        return score * confidence

    def _heuristic_select(
        self,
        state: WorldState,
        candidates: list[str],
        evaluated: list[dict],
    ) -> dict:
        """Fallback heuristic when dynamics model has low confidence."""
        # Priority: dream if η high, learn if many traces, else remember
        priority = {"dream": 0, "learn": 1, "maintain": 2, "evolve": 3, "remember": 4}

        if state.eta >= 0.3:
            preferred = "dream"
        elif state.phi.storage_stats.get("trace_count", 0) > state.phi.storage_stats.get("experience_count", 0) * 2:
            preferred = "learn"
        else:
            preferred = "remember"

        # Use preferred if it's in candidates
        if preferred in candidates:
            chosen = preferred
        else:
            chosen = min(candidates, key=lambda c: priority.get(c, 99))

        return {
            "action": chosen,
            "reason": "heuristic_fallback (low dynamics confidence)",
            "confidence": 0.0,
            "score": 0.0,
            "alternatives": [],
        }

    @property
    def stats(self) -> dict:
        return {
            "prediction_count": self._prediction_count,
            "pre_enactment_count": self._pre_enactment_count,
            "dynamics_trained": self.dynamics.transition_model._trained,
        }
