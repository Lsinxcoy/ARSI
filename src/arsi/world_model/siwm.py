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
        self._pair_rules: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.global_prior: dict[str, int] = defaultdict(int)
        self.accuracy = 0.0
        self._total_predictions = 0
        self._correct_predictions = 0

    @staticmethod
    def categorize_action(action: str) -> str:
        """Categorize action into a general type."""
        a = action.lower()
        if a.startswith(("learn", "synthex_rule")):
            return "learn"
        elif a.startswith(("evolve", "synthex_gate", "mstar")):
            return "evolve"
        elif a.startswith(("reflect", "session_note", "session_intent")):
            return "reflect"
        elif a.startswith(("dream",)):
            return "dream"
        elif a.startswith(("maintain", "synthex_state", "synthex_wal")):
            return "maintain"
        elif a.startswith(("remember", "session_directive", "task_complete")):
            return "remember"
        elif a.startswith(("hermes_tool", "hermes_session")):
            return "execute"
        elif a.startswith(("hermes_model", "hermes_delivery")):
            return "communicate"
        elif a.startswith(("synthex_mechanism", "hermes_skill", "hermes_plan")):
            return "configure"
        elif a.startswith(("empower",)):
            return "empower"
        else:
            return "other"

    def fit(self, traces: list[dict]) -> None:
        """Train from behavior traces."""
        if self.version == "v1":
            self._fit_rules(traces)

    def _fit_rules(self, traces: list[dict]) -> None:
        """Build sequential rules; merge into existing tables (cumulative Layer1)."""
        if getattr(self, "_pair_rules", None) is None:
            self._pair_rules = defaultdict(lambda: defaultdict(int))
        # accumulate — avoid rule_count collapse on thin refits
        for i in range(1, len(traces)):
            prev = traces[i - 1]
            curr = traces[i]

            prev_cat = self.categorize_action(prev.get("action", ""))
            prev_outcome = prev.get("outcome", "unknown")
            curr_cat = self.categorize_action(curr.get("action", ""))
            effect = float(prev.get("effect", 0.5) or 0.5)
            e_bucket = self._bucket(effect, [0.3, 0.6, 0.8])

            pattern = (prev_cat, prev_outcome)
            if pattern not in self.rules:
                self.rules[pattern] = defaultdict(int)
            self.rules[pattern][curr_cat] += 1
            self.global_prior[curr_cat] += 1

            # richer context: two-step action categories
            if i >= 2:
                prev2_cat = self.categorize_action(traces[i - 2].get("action", ""))
                pair = (prev2_cat, prev_cat, prev_outcome, e_bucket)
                self._pair_rules[pair][curr_cat] += 1

    def predict(self, state: WorldState, last_action: str = "", last_outcome: str = "", last_effect: float | None = None) -> str:
        """Predict next action category based on recent history."""
        if last_action:
            if last_effect is not None:
                e_bucket = self._bucket(float(last_effect), [0.3, 0.6, 0.8])
                # try pair context when available
            else:
                e_bucket = None
            pattern = (self.categorize_action(last_action), last_outcome)
            if pattern in self.rules and self.rules[pattern]:
                return max(self.rules[pattern], key=self.rules[pattern].get)

        if self.global_prior:
            return max(self.global_prior, key=self.global_prior.get)

        return "other"

    def evaluate(self, predicted: str, actual: str) -> float:
        """Evaluate prediction and update accuracy.

        Compares categories, not exact action names.
        """
        pred_cat = self.categorize_action(predicted)
        actual_cat = self.categorize_action(actual)
        self._total_predictions += 1
        if pred_cat == actual_cat:
            self._correct_predictions += 1
        self.accuracy = self._correct_predictions / max(self._total_predictions, 1)
        return self.accuracy

    def holdout_accuracy(self, traces: list[dict]) -> float:
        """Temporal holdout: predict t+1 from t on already-fitted rules.

        Does not mutate cumulative accuracy counters (live metrics stay separate).
        """
        traces = list(traces or [])
        if len(traces) < 3:
            return 0.0
        correct = 0
        total = 0
        for i in range(1, len(traces)):
            prev = traces[i - 1]
            curr = traces[i]
            last_action = prev.get("action", "")
            last_outcome = prev.get("outcome", "unknown")
            # majority fallback among recent context
            context = [self.categorize_action(t.get("action", "")) for t in traces[max(0, i - 4):i]]
            maj = max(set(context), key=context.count) if context else "other"
            pred = self.predict(WorldState(), last_action=last_action, last_outcome=last_outcome)
            pred_cat = self.categorize_action(pred)
            if pred_cat == "other" and maj != "other":
                pred_cat = maj
            actual_cat = self.categorize_action(curr.get("action", ""))
            total += 1
            if pred_cat == actual_cat:
                correct += 1
        return correct / max(total, 1)

    @property
    def live_accuracy(self) -> float:
        return self.accuracy

    def _extract_features(self, state: WorldState) -> dict:
        return {
            "action_cat": "unknown",  # Will be set by caller
            "outcome_bucket": "other",
            "effect_bucket": 0,
            "agent_cat": "unknown",
        }

    def _discretize(self, trace: dict) -> tuple:
        """Discretize trace into pattern bucket.

        Uses features available in ALL traces (not just ARSI's own):
        - action category
        - outcome type
        - effect level
        - agent category
        """
        action_cat = self.categorize_action(trace.get("action", ""))
        outcome = trace.get("outcome", "unknown")
        effect = trace.get("effect", 0.5)
        agent = trace.get("agent_id", trace.get("params", {}).get("source", "unknown"))

        # Outcome bucket
        if outcome == "success":
            outcome_bucket = "success"
        elif outcome == "failure":
            outcome_bucket = "failure"
        else:
            outcome_bucket = "other"

        # Effect bucket
        effect_bucket = self._bucket(effect, [0.3, 0.6, 0.8])

        # Agent category
        agent_cat = "hermes" if "hermes" in agent or "mstar" in agent else \
                    "synthex" if "synthex" in agent else \
                    "mimo" if "mimo" in agent or "session" in agent else "other"

        return (action_cat, outcome_bucket, effect_bucket, agent_cat)

    def _discretize_from_features(self, features: dict) -> tuple:
        return (
            features.get("action_cat", "other"),
            features.get("outcome_bucket", "other"),
            features.get("effect_bucket", 0),
            features.get("agent_cat", "other"),
        )

    @staticmethod
    def _bucket(value: float, thresholds: list[float]) -> int:
        for i, t in enumerate(thresholds):
            if value < t:
                return i
        return len(thresholds)


class SIWM:
    """System Introspective World Model — the complete three-layer system.

    Note: full introspective acceptance lives in arsi.iwm.IWM (Q1–Q6).
    SIWM provides η / MindZero / Layer1 predictors that IWM observes.
    """

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
        self.iwm = None  # attached by ARSI core when available

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
        """Train Layer 1 from historical traces + measure holdout accuracy."""
        traces = self.store.get_recent_traces(n=800)
        if not traces:
            return {"status": "no_data", "trace_count": 0, "holdout_accuracy": 0.0}

        # Keep cumulative rules across refits when new data is thin/repetitive
        split = max(3, int(len(traces) * 0.8))
        train, test = traces[:split], traces[split:] or traces[-max(3, len(traces) // 5):]
        prev_rules = len(self.layer1.rules)
        self.layer1.fit(train)
        # If refit collapsed rules on a flat batch, retain prior global priors
        if len(self.layer1.rules) < max(2, prev_rules // 4) and prev_rules > len(self.layer1.rules):
            # merge: refit on union of older history
            older = self.store.get_recent_traces(n=2000)
            if len(older) > len(train):
                self.layer1.fit(older[: max(split, 800)])
        holdout = self.layer1.holdout_accuracy(test)
        self._last_holdout_accuracy = holdout
        return {
            "status": "trained",
            "trace_count": len(traces),
            "rule_count": len(self.layer1.rules),
            "pair_rule_count": len(self.layer1._pair_rules),
            "holdout_accuracy": round(holdout, 4),
            "live_accuracy": round(self.layer1.live_accuracy, 4),
            "train_size": len(train),
            "test_size": len(test),
        }

    @property
    def last_holdout_accuracy(self) -> float:
        return float(getattr(self, "_last_holdout_accuracy", 0.0) or 0.0)

    def predict_and_update(self, actual_action: str) -> dict:
        """Predict next action category, compare with actual, update η."""
        state = self.get_state()

        # Sequential context from recent traces
        recent = self.store.get_recent_traces(n=6)
        last_action = recent[0].get("action", "") if recent else ""
        last_outcome = recent[0].get("outcome", "") if recent else ""
        context_cats = [
            self.layer1.categorize_action(t.get("action", "")) for t in recent[1:5]
        ] if len(recent) > 1 else []
        maj = max(set(context_cats), key=context_cats.count) if context_cats else None

        predicted = self.layer1.predict(state, last_action, last_outcome)
        pred_cat = self.layer1.categorize_action(predicted)
        if pred_cat == "other" and maj:
            predicted = f"recent_majority:{maj}"
            pred_cat = maj

        accuracy = self.layer1.evaluate(predicted, actual_action)
        actual_cat = self.layer1.categorize_action(actual_action)
        eta = self.eta.update(pred_cat, actual_cat)

        return {
            "predicted": predicted,
            "predicted_category": pred_cat,
            "actual": actual_action,
            "actual_category": actual_cat,
            "correct": pred_cat == actual_cat,
            "accuracy": accuracy,
            "live_accuracy": self.layer1.live_accuracy,
            "holdout_accuracy": self.last_holdout_accuracy,
            "eta": eta,
        }
