"""SIWM Layer 2 — Dynamics Model.

Learns state transition functions T(Φ_t, a) → Φ_{t+1} from behavior traces.
Enables Governor's pre-enactment: predict consequences before executing.

Approach:
  1. Statistical: group by action, average state deltas
  2. LLM-assisted: extract transition rules from patterns
  3. Prediction: given (state, candidate_action) → predicted next state

Based on: SIWM Technical Spec §5 (Layer 2)
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Optional

from arsi.foundation.llm import LLMClient
from arsi.foundation.schema import PhysicalState, WorldState
from arsi.foundation.store import MnemosyneStore

logger = logging.getLogger(__name__)


def _parse_llm_json(content: str) -> Optional[dict]:
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


class TransitionModel:
    """Statistical transition model: (state_features, action) → state_delta.

    Learns from behavior traces by grouping transitions by action
    and computing average state changes.
    """

    def __init__(self):
        # action → list of (before_features, after_features) pairs
        self._transitions: dict[str, list[dict]] = defaultdict(list)
        # action → average delta per feature
        self._avg_deltas: dict[str, dict[str, float]] = {}
        # action → count
        self._counts: dict[str, int] = defaultdict(int)
        self._trained = False

    def fit(self, traces: list[dict]) -> dict:
        """Learn transition patterns from behavior traces."""
        self._transitions.clear()
        self._avg_deltas.clear()
        self._counts.clear()

        for trace in traces:
            action = trace.get("action", "unknown")
            before = trace.get("state_before", {})
            after = trace.get("state_after", {})

            if not before or not after:
                continue

            # Extract numeric features
            before_features = self._extract_features(before)
            after_features = self._extract_features(after)

            if before_features and after_features:
                delta = {k: after_features.get(k, 0) - before_features.get(k, 0)
                         for k in before_features}
                self._transitions[action].append({
                    "before": before_features,
                    "after": after_features,
                    "delta": delta,
                })
                self._counts[action] += 1

        # Compute average deltas per action
        for action, transitions in self._transitions.items():
            if not transitions:
                continue
            avg_delta = {}
            keys = transitions[0]["delta"].keys()
            for k in keys:
                avg_delta[k] = sum(t["delta"].get(k, 0) for t in transitions) / len(transitions)
            self._avg_deltas[action] = avg_delta

        self._trained = True
        return {
            "trained": True,
            "actions": list(self._counts.keys()),
            "total_transitions": sum(self._counts.values()),
            "per_action_counts": dict(self._counts),
        }

    def predict(self, state: WorldState, action: str) -> dict:
        """Predict state delta for a given action.

        Returns dict with predicted feature changes and confidence.
        """
        if not self._trained or action not in self._avg_deltas:
            return {
                "action": action,
                "predicted_delta": {},
                "confidence": 0.0,
                "reason": "no_training_data" if not self._trained else f"unknown_action:{action}",
            }

        avg_delta = self._avg_deltas[action]
        count = self._counts[action]

        # Confidence based on sample size and consistency
        confidence = min(1.0, count / 20.0)  # Full confidence at 20+ samples

        # Check consistency (low variance = high confidence)
        if self._transitions[action]:
            variances = {}
            for k in avg_delta:
                values = [t["delta"].get(k, 0) for t in self._transitions[action]]
                mean = avg_delta[k]
                variances[k] = sum((v - mean) ** 2 for v in values) / max(len(values), 1)
            avg_variance = sum(variances.values()) / max(len(variances), 1)
            consistency = max(0, 1.0 - avg_variance)
            confidence *= consistency

        return {
            "action": action,
            "predicted_delta": {k: round(v, 4) for k, v in avg_delta.items()},
            "confidence": round(confidence, 4),
            "sample_count": count,
        }

    def _extract_features(self, state_data: dict) -> dict[str, float]:
        """Extract numeric features from state data."""
        features = {}
        if isinstance(state_data, dict):
            phi = state_data.get("phi", state_data)
            if isinstance(phi, dict):
                features["generation"] = float(phi.get("generation", 0))
                features["steps_since_change"] = float(phi.get("steps_since_change", 0))
                storage = phi.get("storage_stats", {})
                if isinstance(storage, dict):
                    features["trace_count"] = float(storage.get("trace_count", 0))
                    features["experience_count"] = float(storage.get("experience_count", 0))
                    features["proxy_count"] = float(storage.get("proxy_count", 0))
            features["eta"] = float(state_data.get("eta", 0))
            features["belief_count"] = float(len(state_data.get("psi", {}).get("beliefs", [])))
        return features

    @property
    def stats(self) -> dict:
        return {
            "trained": self._trained,
            "action_count": len(self._counts),
            "total_transitions": sum(self._counts.values()),
            "actions": dict(self._counts),
        }


class LLMDynamicsExtractor:
    """LLM-assisted transition rule extraction.

    Uses LLM to identify non-obvious transition patterns
    that simple averaging might miss.
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm
        self._rules: list[dict] = []
        self._llm_calls = 0

    def extract_rules(self, traces: list[dict]) -> list[dict]:
        """Extract transition rules using LLM analysis."""
        if not self.llm or not self.llm.available or len(traces) < 5:
            return []

        # Sample recent traces for analysis
        sample = traces[-20:]
        trace_summary = json.dumps(
            [{"action": t.get("action"), "outcome": t.get("outcome"),
              "effect": t.get("effect", 0)} for t in sample],
            ensure_ascii=False,
        )

        prompt = f"""分析以下行为轨迹，找出状态转移的规律。

轨迹：
{trace_summary}

找出：
1. 哪些动作倾向于增加/减少哪些指标
2. 动作之间的因果关系（A 导致 B）
3. 非线性效应（某些组合的效果不同于单独效果之和）

输出 JSON：
{{"rules": [{{"description": "规律描述", "action": "触发动作", "effect_pattern": "效果模式", "confidence": 0.0到1.0}}]}}"""

        resp = self.llm.chat(
            prompt,
            system="你是状态转移分析专家。只输出 JSON。",
            max_tokens=800,
        )

        if not resp.success:
            return []

        data = _parse_llm_json(resp.content)
        if data and "rules" in data:
            self._llm_calls += 1
            self._rules = data["rules"]
            return self._rules

        return []

    @property
    def stats(self) -> dict:
        return {"llm_calls": self._llm_calls, "rule_count": len(self._rules)}


class DynamicsModel:
    """Complete SIWM Layer 2: statistical + LLM-assisted dynamics.

    Combines TransitionModel (statistical) with LLMDynamicsExtractor (LLM)
    to provide state transition predictions for pre-enactment.
    """

    def __init__(self, store: MnemosyneStore, llm: Optional[LLMClient] = None):
        self.store = store
        self.transition_model = TransitionModel()
        self.llm_extractor = LLMDynamicsExtractor(llm)
        self._prediction_count = 0
        self._correct_predictions = 0

    def train(self) -> dict:
        """Train from behavior traces in the store."""
        traces = self.store.get_recent_traces(n=500)

        # Train statistical model
        stat_result = self.transition_model.fit(traces)

        # Extract LLM rules (if LLM available)
        llm_rules = self.llm_extractor.extract_rules(traces)

        return {
            "statistical": stat_result,
            "llm_rules_count": len(llm_rules),
            "total_traces": len(traces),
        }

    def predict_transition(self, state: WorldState, action: str) -> dict:
        """Predict the result of executing an action.

        Returns predicted state delta with confidence.
        """
        self._prediction_count += 1

        # Statistical prediction
        stat_pred = self.transition_model.predict(state, action)

        # If statistical model has low confidence, try LLM
        if stat_pred["confidence"] < 0.3 and self.llm_extractor.llm and self.llm_extractor.llm.available:
            llm_pred = self._llm_predict(state, action)
            if llm_pred:
                # Blend: prefer statistical when confident, LLM when not
                return {
                    **stat_pred,
                    "llm_enhanced": True,
                    "llm_prediction": llm_pred,
                    "source": "blended",
                }

        return stat_pred

    def simulate(self, state: WorldState, action: str, depth: int = 1) -> list[dict]:
        """Simulate a sequence of actions (pre-enactment).

        Returns list of predicted states.
        """
        trajectory = [{"state": state.model_dump(), "action": action, "predicted": True}]
        current_state = state

        for step in range(depth):
            pred = self.predict_transition(current_state, action)
            if pred["confidence"] > 0:
                # Apply predicted delta
                delta = pred.get("predicted_delta", {})
                new_phi = current_state.phi.model_copy(update={
                    "generation": current_state.phi.generation + int(delta.get("generation", 0)),
                    "steps_since_change": current_state.phi.steps_since_change + int(delta.get("steps_since_change", 1)),
                })
                new_eta = max(0, min(1, current_state.eta + delta.get("eta", 0)))
                current_state = current_state.model_copy(update={"phi": new_phi, "eta": new_eta})
                trajectory.append({
                    "state": current_state.model_dump(),
                    "action": action,
                    "predicted": True,
                    "confidence": pred["confidence"],
                    "step": step + 1,
                })
            else:
                break

        return trajectory

    def evaluate_prediction(self, predicted_delta: dict, actual_delta: dict) -> float:
        """Evaluate prediction accuracy against actual outcome."""
        if not predicted_delta or not actual_delta:
            return 0.0

        errors = []
        for k in predicted_delta:
            if k in actual_delta:
                pred = predicted_delta[k]
                actual = actual_delta[k]
                if abs(actual) > 0.001:
                    errors.append(abs(pred - actual) / abs(actual))
                else:
                    errors.append(abs(pred - actual))

        if not errors:
            return 0.0

        mape = sum(errors) / len(errors)
        accuracy = max(0, 1.0 - mape)
        self._correct_predictions += 1 if accuracy > 0.7 else 0
        return accuracy

    def _llm_predict(self, state: WorldState, action: str) -> Optional[dict]:
        """Use LLM to predict state transition."""
        llm = self.llm_extractor.llm
        if not llm or not llm.available:
            return None

        prompt = f"""预测执行动作后的状态变化。

当前状态：
- 代数: {state.phi.generation}
- η: {state.eta:.3f}
- 存储: {json.dumps(state.phi.storage_stats, ensure_ascii=False)}
- 信念数: {len(state.psi.beliefs)}

动作: {action}

输出 JSON：
{{"predicted_eta_change": -1.0到1.0, "predicted_storage_change": {{"trace_count": 整数, "experience_count": 整数}}, "reasoning": "推理过程"}}"""

        resp = llm.chat(prompt, system="你是状态转移预测专家。只输出 JSON。", max_tokens=400)
        if resp.success:
            return _parse_llm_json(resp.content)
        return None

    @property
    def stats(self) -> dict:
        return {
            "transition_model": self.transition_model.stats,
            "llm_extractor": self.llm_extractor.stats,
            "prediction_count": self._prediction_count,
            "correct_predictions": self._correct_predictions,
        }
