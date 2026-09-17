"""Tests for SIWM Layer 2 dynamics model."""
import pytest

from arsi.foundation.schema import WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.world_model.dynamics import TransitionModel, LLMDynamicsExtractor, DynamicsModel


@pytest.fixture
def store():
    s = MnemosyneStore(":memory:")
    yield s
    s.close()


def _make_traces_with_transitions(n=30):
    """Create traces with realistic state transitions."""
    traces = []
    for i in range(n):
        action = ["learn", "evolve", "dream", "maintain"][i % 4]
        before = {
            "phi": {
                "generation": i // 10,
                "steps_since_change": i,
                "storage_stats": {
                    "trace_count": i,
                    "experience_count": i // 2,
                    "proxy_count": i - i // 2,
                },
            },
            "eta": 0.1 + (i % 5) * 0.05,
            "psi": {"beliefs": [{"content": f"belief_{j}"} for j in range(i // 5)]},
        }
        after = {
            "phi": {
                "generation": (i + 1) // 10,
                "steps_since_change": i + 1,
                "storage_stats": {
                    "trace_count": i + 1,
                    "experience_count": (i + 1) // 2,
                    "proxy_count": (i + 1) - (i + 1) // 2,
                },
            },
            "eta": 0.1 + ((i + 1) % 5) * 0.05,
            "psi": {"beliefs": [{"content": f"belief_{j}"} for j in range((i + 1) // 5)]},
        }
        traces.append({
            "action": action,
            "outcome": "success" if i % 3 != 0 else "failure",
            "effect": 0.5,
            "state_before": before,
            "state_after": after,
        })
    return traces


class TestTransitionModel:
    def test_fit(self):
        model = TransitionModel()
        traces = _make_traces_with_transitions(30)
        result = model.fit(traces)
        assert result["trained"] is True
        assert result["total_transitions"] == 30
        assert "learn" in result["actions"]

    def test_predict_known_action(self):
        model = TransitionModel()
        traces = _make_traces_with_transitions(30)
        model.fit(traces)

        state = WorldState()
        pred = model.predict(state, "learn")
        assert pred["action"] == "learn"
        assert pred["sample_count"] > 0
        assert "predicted_delta" in pred

    def test_predict_unknown_action(self):
        model = TransitionModel()
        model.fit([])
        state = WorldState()
        pred = model.predict(state, "unknown_action")
        assert pred["confidence"] == 0.0

    def test_predict_empty_model(self):
        model = TransitionModel()
        state = WorldState()
        pred = model.predict(state, "learn")
        assert pred["confidence"] == 0.0

    def test_confidence_increases_with_samples(self):
        model = TransitionModel()
        few_traces = _make_traces_with_transitions(5)
        model.fit(few_traces)
        pred_few = model.predict(WorldState(), "learn")

        model2 = TransitionModel()
        many_traces = _make_traces_with_transitions(50)
        model2.fit(many_traces)
        pred_many = model2.predict(WorldState(), "learn")

        # More samples should give higher or equal confidence
        assert pred_many["confidence"] >= pred_few["confidence"]

    def test_stats(self):
        model = TransitionModel()
        traces = _make_traces_with_transitions(20)
        model.fit(traces)
        stats = model.stats
        assert stats["trained"] is True
        assert stats["total_transitions"] == 20


class TestLLMDynamicsExtractor:
    def test_no_llm(self):
        extractor = LLMDynamicsExtractor(llm=None)
        rules = extractor.extract_rules(_make_traces_with_transitions(10))
        assert rules == []

    def test_too_few_traces(self):
        extractor = LLMDynamicsExtractor(llm=None)
        rules = extractor.extract_rules([{"action": "learn"}])
        assert rules == []


class TestDynamicsModel:
    def test_train(self, store):
        # Write traces to store
        from arsi.foundation.schema import BehaviorTrace
        traces = _make_traces_with_transitions(30)
        for t in traces:
            store.write_trace(BehaviorTrace(
                agent_id="test",
                action=t["action"],
                outcome=t["outcome"],
                effect=t["effect"],
            ))

        model = DynamicsModel(store, llm=None)
        result = model.train()
        assert result["statistical"]["trained"] is True

    def test_predict_transition(self, store):
        from arsi.foundation.schema import BehaviorTrace
        traces = _make_traces_with_transitions(30)
        for t in traces:
            store.write_trace(BehaviorTrace(
                agent_id="test",
                action=t["action"],
                outcome=t["outcome"],
                effect=t["effect"],
            ))

        model = DynamicsModel(store, llm=None)
        model.train()

        pred = model.predict_transition(WorldState(), "learn")
        assert pred["action"] == "learn"

    def test_simulate(self, store):
        from arsi.foundation.schema import BehaviorTrace
        traces = _make_traces_with_transitions(30)
        for t in traces:
            store.write_trace(BehaviorTrace(
                agent_id="test",
                action=t["action"],
                outcome=t["outcome"],
                effect=t["effect"],
            ))

        model = DynamicsModel(store, llm=None)
        model.train()

        trajectory = model.simulate(WorldState(), "learn", depth=2)
        assert len(trajectory) >= 1
        assert trajectory[0]["action"] == "learn"

    def test_evaluate_prediction(self):
        model = DynamicsModel.__new__(DynamicsModel)
        model._prediction_count = 0
        model._correct_predictions = 0

        accuracy = model.evaluate_prediction(
            {"eta": 0.1, "trace_count": 5},
            {"eta": 0.12, "trace_count": 5},
        )
        assert 0 < accuracy <= 1

    def test_stats(self, store):
        model = DynamicsModel(store, llm=None)
        stats = model.stats
        assert "transition_model" in stats
        assert "llm_extractor" in stats
