"""SIWM — System Introspective World Model.

Three layers:
  Layer 1: Behavior Predictor (predict what the system will do next)
  Layer 2: Dynamics Model (predict state transitions)
  Layer 3: Counterfactual Simulator (predict consequences of interventions)

Plus: η (self-model mismatch) and MindZero (Theory of Mind).

Based on: SIWM Technical Spec + ARSI Whitepaper v0.8 §7
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

from arsi.foundation.schema import (
    Belief,
    MentalState,
    PhysicalState,
    WorldState,
)

logger = logging.getLogger(__name__)


class EtaTracker:
    """Self-model mismatch degree η — continuous physiological indicator.

    η = prediction error of Layer 1 (smoothed).
    η < 0.15 → normal
    0.15 ≤ η < 0.40 → trigger reflect
    η ≥ 0.40 → trigger dream
    """

    def __init__(self, alpha: float = 0.1, theta_low: float = 0.15, theta_high: float = 0.40):
        self.alpha = alpha
        self.theta_low = theta_low
        self.theta_high = theta_high
        self.eta_smooth = 0.0
        self.history: list[float] = []

    def update(self, predicted_action: str, actual_action: str) -> float:
        """Update η after each action."""
        error = 0.0 if predicted_action == actual_action else 1.0
        self.eta_smooth = self.alpha * error + (1 - self.alpha) * self.eta_smooth
        self.history.append(self.eta_smooth)
        return self.eta_smooth

    def should_dream(self) -> bool:
        return self.eta_smooth >= self.theta_high

    def should_reflect(self) -> bool:
        return self.theta_low <= self.eta_smooth < self.theta_high

    def adaptive_depth(self) -> int:
        """Simulation depth adapts to η."""
        if self.eta_smooth < 0.15:
            return 5
        if self.eta_smooth < 0.30:
            return 3
        if self.eta_smooth < 0.50:
            return 1
        return 0

    @property
    def value(self) -> float:
        return self.eta_smooth


class MindZero:
    """Theory of Mind — infers mental states from behavioral traces.

    Inherited from SYNTHEX's mind_zero.py (multi-hypothesis Bayesian).
    Infers beliefs, goals, and affect from action sequences.
    """

    def infer_mental_state(self, traces: list[dict]) -> MentalState:
        """Infer mental state from behavior traces."""
        beliefs = self._infer_beliefs(traces)
        affect = self._infer_affect(traces)
        return MentalState(
            beliefs=beliefs,
            affect=affect,
            norms=[],
        )

    def _infer_beliefs(self, traces: list[dict]) -> list[Belief]:
        """Infer beliefs from action patterns.

        Example: if learn pipeline fails 3 times then reflect is triggered,
        infer belief "learn is unreliable under current conditions".
        """
        beliefs = []
        action_counts: dict[str, int] = defaultdict(int)
        outcome_by_action: dict[str, list[str]] = defaultdict(list)

        for t in traces:
            action = t.get("action", "")
            outcome = t.get("outcome", "")
            action_counts[action] += 1
            outcome_by_action[action].append(outcome)

        for action, outcomes in outcome_by_action.items():
            if not outcomes:
                continue
            success_rate = sum(1 for o in outcomes if "success" in o.lower()) / len(outcomes)
            if success_rate < 0.5:
                beliefs.append(Belief(
                    content=f"{action} has low success rate ({success_rate:.0%})",
                    confidence=min(0.9, 1.0 - success_rate),
                    source="behavior_pattern",
                ))
            elif success_rate > 0.8:
                beliefs.append(Belief(
                    content=f"{action} is reliable ({success_rate:.0%})",
                    confidence=min(0.9, success_rate),
                    source="behavior_pattern",
                ))

        return beliefs

    def _infer_affect(self, traces: list[dict]) -> dict[str, float]:
        """Infer affect (emotional valence) from effect trends."""
        affect: dict[str, list[float]] = defaultdict(list)
        for t in traces:
            action = t.get("action", "")
            effect = t.get("effect", 0.0)
            affect[action].append(effect)

        result = {}
        for action, effects in affect.items():
            if len(effects) >= 2:
                trend = effects[-1] - effects[0]
                result[action] = max(-1.0, min(1.0, trend * 10))
            else:
                result[action] = effects[0] if effects else 0.0
        return result


class BehaviorPredictor:
    """Layer 1 — predicts what the system will do next.

    V1: rule-based (if-then + frequency table)
    V2: statistical (logistic regression / decision tree)
    V3: learned (shallow MLP / GBDT)
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.rules: dict[tuple, dict[str, int]] = {}
        self.global_prior: dict[str, int] = defaultdict(int)
        self.accuracy = 0.0
        self._total_predictions = 0
        self._correct_predictions = 0

    def fit(self, traces: list[dict]) -> None:
        """Train from behavior traces."""
        if self.version == "v1":
            self._fit_rules(traces)

    def _fit_rules(self, traces: list[dict]) -> None:
        for t in traces:
            pattern = self._discretize(t)
            action = t.get("action", "")
            if pattern not in self.rules:
                self.rules[pattern] = defaultdict(int)
            self.rules[pattern][action] += 1
            self.global_prior[action] += 1

    def predict(self, state: WorldState) -> str:
        """Predict next action."""
        features = self._extract_features(state)
        pattern = self._discretize_from_features(features)

        if pattern in self.rules and self.rules[pattern]:
            return max(self.rules[pattern], key=self.rules[pattern].get)

        if self.global_prior:
            return max(self.global_prior, key=self.global_prior.get)

        return "unknown"

    def evaluate(self, predicted: str, actual: str) -> float:
        """Evaluate prediction and update accuracy."""
        self._total_predictions += 1
        if predicted == actual:
            self._correct_predictions += 1
        self.accuracy = self._correct_predictions / max(self._total_predictions, 1)
        return self.accuracy

    def _extract_features(self, state: WorldState) -> dict:
        return {
            "eta_bucket": self._bucket(state.eta, [0.15, 0.40]),
            "gen_bucket": self._bucket(float(state.phi.generation), [1, 5, 20]),
            "steps_bucket": self._bucket(float(state.phi.steps_since_change), [5, 20, 100]),
            "belief_count_bucket": self._bucket(float(len(state.psi.beliefs)), [0, 3, 10]),
        }

    def _discretize(self, trace: dict) -> tuple:
        state_data = trace.get("state_before", {})
        if isinstance(state_data, dict):
            eta = state_data.get("eta", 0.0)
            gen = state_data.get("phi", {}).get("generation", 0)
        else:
            eta = 0.0
            gen = 0
        return (
            self._bucket(eta, [0.15, 0.40]),
            self._bucket(float(gen), [1, 5, 20]),
        )

    def _discretize_from_features(self, features: dict) -> tuple:
        return (
            features["eta_bucket"],
            features["gen_bucket"],
            features["steps_bucket"],
        )

    @staticmethod
    def _bucket(value: float, thresholds: list[float]) -> int:
        for i, t in enumerate(thresholds):
            if value < t:
                return i
        return len(thresholds)


class SIWM:
    """System Introspective World Model — the complete three-layer system."""

    def __init__(self, store, config=None):
        self.store = store
        self.eta = EtaTracker(
            alpha=getattr(config, "alpha", 0.1) if config else 0.1,
            theta_low=getattr(config, "theta_low", 0.15) if config else 0.15,
            theta_high=getattr(config, "theta_high", 0.40) if config else 0.40,
        )
        self.mindzero = MindZero()
        self.layer1 = BehaviorPredictor(version="v1")
        self._current_state: Optional[WorldState] = None

    def get_state(self) -> WorldState:
        """Get current world state."""
        if self._current_state is None:
            self._current_state = self._build_state()
        return self._current_state

    def refresh_state(self) -> WorldState:
        """Rebuild world state from store."""
        self._current_state = self._build_state()
        return self._current_state

    def _build_state(self) -> WorldState:
        """Build world state from store data."""
        stats = self.store.get_stats()
        traces = self.store.get_recent_traces(n=50)

        phi = PhysicalState(
            generation=stats.get("generation", 0),
            storage_stats={
                "proxy_count": stats.get("proxy_count", 0),
                "experience_count": stats.get("experience_count", 0),
                "trace_count": stats.get("trace_count", 0),
            },
        )

        psi = self.mindzero.infer_mental_state(traces)

        return WorldState(
            phi=phi,
            psi=psi,
            eta=self.eta.value,
        )

    def train_from_history(self) -> dict:
        """Train Layer 1 from historical traces."""
        traces = self.store.get_recent_traces(n=500)
        if not traces:
            return {"status": "no_data", "trace_count": 0}

        self.layer1.fit(traces)
        return {
            "status": "trained",
            "trace_count": len(traces),
            "rule_count": len(self.layer1.rules),
        }

    def predict_and_update(self, actual_action: str) -> dict:
        """Predict next action, compare with actual, update η."""
        state = self.get_state()
        predicted = self.layer1.predict(state)
        accuracy = self.layer1.evaluate(predicted, actual_action)
        eta = self.eta.update(predicted, actual_action)

        return {
            "predicted": predicted,
            "actual": actual_action,
            "correct": predicted == actual_action,
            "accuracy": accuracy,
            "eta": eta,
        }
