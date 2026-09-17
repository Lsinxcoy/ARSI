"""SIWM Layer 3 — Counterfactual Simulator.

Uses Layer 2 dynamics to iteratively expand multi-step predictions.
Enables Governor's high-risk decision simulation.

Based on: SIWM Technical Spec §6 (Layer 3)
"""
from __future__ import annotations

import logging
from typing import Optional

from arsi.foundation.llm import LLMClient
from arsi.foundation.schema import WorldState
from arsi.world_model.dynamics import DynamicsModel

logger = logging.getLogger(__name__)


class CounterfactualSimulator:
    """Simulates consequences of interventions via iterative Layer 2 expansion.

    Given current state + candidate intervention, predicts a trajectory
    of future states by repeatedly applying the dynamics model.
    """

    def __init__(self, dynamics: DynamicsModel, llm: Optional[LLMClient] = None):
        self.dynamics = dynamics
        self.llm = llm
        self._simulation_count = 0

    def simulate(
        self,
        state: WorldState,
        intervention: str,
        depth: int = 3,
    ) -> list[dict]:
        """Simulate a sequence of states after an intervention.

        Args:
            state: current world state
            intervention: the action to simulate
            depth: how many steps to expand

        Returns:
            List of predicted states with metadata.
        """
        self._simulation_count += 1
        trajectory = []

        current = state
        for step in range(depth):
            # Predict next state via Layer 2
            pred = self.dynamics.predict_transition(current, intervention)

            if pred.get("confidence", 0) <= 0:
                trajectory.append({
                    "step": step,
                    "action": intervention,
                    "predicted": False,
                    "reason": "no_dynamics_data",
                })
                break

            # Apply predicted delta
            delta = pred.get("predicted_delta", {})
            new_eta = max(0.0, min(1.0, current.eta + delta.get("eta", 0)))

            # Track storage changes
            old_storage = current.phi.storage_stats.copy()
            new_storage = old_storage.copy()
            for key in ["trace_count", "experience_count", "proxy_count"]:
                if key in delta:
                    new_storage[key] = max(0, int(old_storage.get(key, 0) + delta[key]))

            new_phi = current.phi.model_copy(update={
                "storage_stats": new_storage,
                "steps_since_change": current.phi.steps_since_change + 1,
            })

            current = current.model_copy(update={"phi": new_phi, "eta": new_eta})

            trajectory.append({
                "step": step + 1,
                "action": intervention,
                "predicted": True,
                "confidence": pred.get("confidence", 0),
                "eta": round(new_eta, 4),
                "storage": new_storage,
                "delta": {k: round(v, 4) for k, v in delta.items()},
            })

        return trajectory

    def compare_interventions(
        self,
        state: WorldState,
        interventions: list[str],
        depth: int = 2,
    ) -> list[dict]:
        """Compare multiple interventions by simulating each.

        Returns list of {intervention, final_eta, avg_confidence, score} sorted by score.
        """
        results = []

        for intervention in interventions:
            trajectory = self.simulate(state, intervention, depth)

            if not trajectory or not trajectory[-1].get("predicted"):
                results.append({
                    "intervention": intervention,
                    "final_eta": state.eta,
                    "avg_confidence": 0.0,
                    "score": 0.0,
                    "steps_simulated": 0,
                })
                continue

            final_eta = trajectory[-1]["eta"]
            confidences = [t.get("confidence", 0) for t in trajectory if t.get("predicted")]
            avg_conf = sum(confidences) / max(len(confidences), 1)

            # Score: lower final η is better, weighted by confidence
            eta_improvement = state.eta - final_eta
            score = eta_improvement * avg_conf

            results.append({
                "intervention": intervention,
                "final_eta": round(final_eta, 4),
                "avg_confidence": round(avg_conf, 4),
                "eta_improvement": round(eta_improvement, 4),
                "score": round(score, 4),
                "steps_simulated": len([t for t in trajectory if t.get("predicted")]),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def adaptive_depth(self, state: WorldState) -> int:
        """Determine simulation depth based on η (self-model freshness).

        Higher η → shorter simulation (less reliable predictions).
        """
        if state.eta < 0.15:
            return 5
        elif state.eta < 0.30:
            return 3
        elif state.eta < 0.50:
            return 1
        else:
            return 0  # Skip simulation entirely

    @property
    def stats(self) -> dict:
        return {
            "simulation_count": self._simulation_count,
            "dynamics_trained": self.dynamics.transition_model._trained,
        }
