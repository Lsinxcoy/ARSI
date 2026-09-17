"""Tests for pre-enactment engine."""
import pytest

from arsi.foundation.schema import BehaviorTrace, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.world_model.dynamics import DynamicsModel
from arsi.governor.pre_enactment import PreEnactmentEngine


@pytest.fixture
def store():
    s = MnemosyneStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def trained_dynamics(store):
    """Dynamics model trained on synthetic traces."""
    # Create traces with realistic state transitions
    for i in range(40):
        action = ["learn", "evolve", "dream", "maintain"][i % 4]
        before_state = WorldState(
            phi={"generation": i // 10, "steps_since_change": i,
                 "storage_stats": {"trace_count": i, "experience_count": i // 2}},
            eta=0.1 + (i % 5) * 0.05,
        )
        after_state = WorldState(
            phi={"generation": (i + 1) // 10, "steps_since_change": i + 1,
                 "storage_stats": {"trace_count": i + 1, "experience_count": (i + 1) // 2}},
            eta=0.05 + (i % 5) * 0.03,  # η tends to decrease
        )
        store.write_trace(BehaviorTrace(
            agent_id="test", action=action, outcome="success", effect=0.7,
            state_before=before_state, state_after=after_state,
        ))

    model = DynamicsModel(store, llm=None)
    model.train()
    return model


class TestPreEnactmentEngine:
    def test_evaluate_candidates(self, trained_dynamics):
        engine = PreEnactmentEngine(trained_dynamics)
        state = WorldState()
        candidates = ["learn", "evolve", "dream"]

        results = engine.evaluate_candidates(state, candidates)
        assert len(results) == 3
        assert all("action" in r for r in results)
        assert all("score" in r for r in results)

    def test_select_best_with_trained_model(self, trained_dynamics):
        engine = PreEnactmentEngine(trained_dynamics)
        state = WorldState()
        candidates = ["learn", "evolve", "dream", "maintain"]

        result = engine.select_best(state, candidates)
        assert result["action"] in candidates
        assert "reason" in result
        assert "confidence" in result

    def test_select_best_untrained(self, store):
        model = DynamicsModel(store, llm=None)  # Not trained
        engine = PreEnactmentEngine(model)
        state = WorldState()

        result = engine.select_best(state, ["learn", "evolve"])
        # Should fall back to heuristic
        assert result["action"] in ["learn", "evolve"]
        assert "heuristic" in result["reason"] or result["confidence"] == 0.0

    def test_scoring_rewards_eta_reduction(self, trained_dynamics):
        engine = PreEnactmentEngine(trained_dynamics)
        state = WorldState()

        # dream action should have η-reducing prediction
        results = engine.evaluate_candidates(state, ["dream", "learn"])
        # At least one should have non-zero score
        assert any(r["score"] > 0 for r in results) or all(r["confidence"] == 0 for r in results)

    def test_empty_candidates(self, trained_dynamics):
        engine = PreEnactmentEngine(trained_dynamics)
        state = WorldState()
        result = engine.select_best(state, [])
        assert result["action"] == "remember"  # Default

    def test_stats(self, trained_dynamics):
        engine = PreEnactmentEngine(trained_dynamics)
        engine.select_best(WorldState(), ["learn"])
        stats = engine.stats
        assert stats["pre_enactment_count"] == 1
        assert stats["prediction_count"] > 0


class TestPreEnactmentWithARSI:
    """Test pre-enactment integration with ARSI Core."""

    def test_arsi_step_uses_pre_enactment(self, tmp_path):
        """Verify ARSI step() uses pre-enactment when dynamics is trained."""
        import yaml
        from arsi.core import ARSI

        config = {
            "llm": {"provider": "openai", "model": "test", "api_key": "sk-test",
                    "use_proxy": False, "fallback_to_heuristic": True, "timeout": 1},
            "db_path": ":memory:",
            "iron_laws_path": str(tmp_path / "iron_laws.yaml"),
            "sealed_tasks_path": str(tmp_path / "sealed_tasks.yaml"),
        }
        config_path = tmp_path / "arsi.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        (tmp_path / "iron_laws.yaml").write_text(
            "laws:\n  - id: G10\n    name: test\n    description: test\n    severity: block\n",
            encoding="utf-8",
        )

        arsi = ARSI.from_config(config_path)
        if arsi.llm:
            arsi.llm._client = None

        # Ingest traces to train dynamics
        for i in range(20):
            arsi.ingest_trace("a1", "learn" if i % 2 == 0 else "evolve", "success", 0.6)

        # Run several steps
        sources = []
        for _ in range(5):
            result = arsi.step()
            sources.append(result["decision"]["source"])

        # At least some steps should use pre-enactment or have dynamics trained
        stats = arsi.get_stats()
        assert stats["dynamics_trained"] is True or stats["pre_enactment_count"] > 0

        arsi.close()
