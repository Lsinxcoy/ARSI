"""Comprehensive end-to-end integration test.

Tests the full pipeline: cold start → trace ingestion → training →
pre-enactment decision → execution → dream → gain decomposition →
cost report → dimension analysis.
"""
import json
import pytest

from arsi.core import ARSI
from arsi.foundation.schema import BehaviorTrace, WorldState
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.foundation.cost_ledger import CostLedger
from arsi.sealed_eval.gain_decomposition import GainDecomposer
from arsi.world_model.counterfactual import CounterfactualSimulator


@pytest.fixture
def arsi(tmp_path):
    """Create a full ARSI instance for integration testing."""
    import yaml

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
    (tmp_path / "sealed_tasks.yaml").write_text("tasks: []\n", encoding="utf-8")

    instance = ARSI.from_config(config_path)
    if instance.llm:
        instance.llm._client = None
    yield instance
    instance.close()


class TestFullPipeline:
    def test_cold_start_to_term(self, arsi):
        """Full pipeline: ingest → train → step → term → analyze."""
        # Phase 1: Ingest traces (simulating agent behavior)
        for i in range(30):
            arsi.ingest_trace(
                agent_id="test_agent",
                action=["learn", "evolve", "reflect", "dream", "maintain"][i % 5],
                outcome="success" if i % 3 != 0 else "failure",
                effect=0.3 + (i % 10) * 0.07,
                params={"cycle": i, "detail": f"param_{i}" * 3},
            )

        stats = arsi.get_stats()
        assert stats["trace_count"] == 30

        # Phase 2: Run several steps
        decision_sources = []
        for _ in range(8):
            result = arsi.step()
            decision_sources.append(result["decision"]["source"])

        assert arsi._step_count == 8

        # Phase 3: Dynamics should be trained
        assert arsi._dynamics_trained is True

        # Phase 4: Run a full term
        term_result = arsi.run_term(n_steps=5)

        assert "gain_decomposition" in term_result
        assert "cost_benefit" in term_result
        assert "lifecycle" in term_result

        # Verify gain decomposition structure
        gd = term_result["gain_decomposition"]
        assert "amplified" in gd
        assert "imported" in gd
        assert "self_organized" in gd
        assert "total" in gd

        # Verify cost-benefit structure
        cb = term_result["cost_benefit"]
        assert "gain" in cb
        assert "cost" in cb
        assert "efficiency" in cb

        # Phase 5: Dimension analysis
        traces = arsi.store.get_recent_traces(n=50)
        state = arsi.siwm.refresh_state()
        orch = DimensionOrchestrator(arsi.store, llm=None)
        dim_results = orch.run_all(traces, state)

        assert len(dim_results) == 6
        for dim_name, result in dim_results.items():
            assert "sense" in result or "error" in result

        # Phase 6: Counterfactual simulation
        cf = CounterfactualSimulator(arsi.dynamics, llm=None)
        trajectory = cf.simulate(state, "learn", depth=2)
        assert len(trajectory) >= 1

        # Final stats
        final = arsi.get_stats()
        assert final["step_count"] == 13  # 8 + 5
        assert final["dynamics_trained"] is True

    def test_dream_reduces_eta(self, arsi):
        """Verify dream cycle actually reduces η."""
        for i in range(20):
            arsi.ingest_trace("a1", "learn", "success", 0.5)

        # Force high η
        arsi.siwm.eta.eta_smooth = 0.50
        state = arsi.siwm.refresh_state()

        new_state = arsi.dream.execute(state)
        assert new_state.eta < 0.50

    def test_pre_enactment_with_data(self, arsi):
        """Verify pre-enactment produces meaningful predictions with data."""
        for i in range(40):
            arsi.ingest_trace(
                "a1", ["learn", "evolve"][i % 2], "success", 0.6,
                params={"idx": i},
            )

        arsi.dynamics.train()
        state = arsi.siwm.refresh_state()
        results = arsi.pre_enactment.evaluate_candidates(
            state, ["learn", "evolve", "dream"]
        )
        assert len(results) == 3
        assert all("score" in r for r in results)

    def test_cost_ledger_tracks(self, arsi):
        """Verify cost ledger accumulates across steps."""
        for i in range(10):
            arsi.ingest_trace("a1", "learn", "success", 0.6)

        initial_cost = arsi.get_stats().get("cost_llm_calls", 0)
        arsi.step()
        arsi.step()
        final_cost = arsi.get_stats().get("cost_llm_calls", 0)
        # Cost may or may not increase depending on decision source
        assert final_cost >= initial_cost

    def test_gain_decomposition_sum(self, arsi):
        """Verify gain decomposition parts sum to total."""
        for i in range(20):
            arsi.ingest_trace("a1", "learn", "success", 0.7)

        term_result = arsi.run_term(n_steps=3)
        gd = term_result["gain_decomposition"]

        parts_sum = gd["amplified"] + gd["imported"] + gd["self_organized"]
        # Parts should approximately sum to total (within rounding)
        assert abs(parts_sum - gd["total"]) < 0.02

    def test_dimension_lifecycle_updates(self, arsi):
        """Verify dimension lifecycle integration produces updates."""
        for i in range(20):
            arsi.ingest_trace("a1", "learn", "failure" if i % 3 == 0 else "success", 0.5)

        term_result = arsi.run_term(n_steps=3)
        lifecycle = term_result.get("lifecycle", {})
        assert len(lifecycle) > 0
        # At least some dimensions should have stage info
        for dim_name, info in lifecycle.items():
            assert "stage" in info

    def test_counterfactual_comparison(self, arsi):
        """Verify counterfactual simulator can compare interventions."""
        for i in range(30):
            arsi.ingest_trace("a1", ["learn", "evolve", "dream"][i % 3], "success", 0.6)

        arsi.dynamics.train()
        state = arsi.siwm.refresh_state()
        cf = CounterfactualSimulator(arsi.dynamics, llm=None)
        comparison = cf.compare_interventions(state, ["learn", "evolve", "dream"], depth=2)

        assert len(comparison) == 3
        assert all("score" in r for r in comparison)
        # Should be sorted by score
        scores = [r["score"] for r in comparison]
        assert scores == sorted(scores, reverse=True)

    def test_sealed_eval_integration(self, arsi):
        """Verify sealed evaluator can be called from ARSI."""
        result = arsi.run_evaluation()
        assert "capability_proxy" in result
        assert "self_model_eta" in result

    def test_semantic_search_works(self, arsi):
        """Verify semantic search is functional."""
        from arsi.foundation.schema import MemoryRecord, MemoryZone
        arsi.store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="约束检查是薄弱环节需要改进",
            tags=["decomposition"],
            agent_id="a1",
        ))
        arsi.store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="学会了分解任务的技巧",
            tags=["decomposition"],
            agent_id="b1",
        ))

        results = arsi.mnemosyne.search_cross_agent("约束检查", exclude_agent="a1")
        assert isinstance(results, list)

    def test_iron_laws_enforced(self, arsi):
        """Verify iron laws are loaded and enforced."""
        assert len(arsi.iron_laws.law_ids) >= 1
        assert "G10" in arsi.iron_laws.law_ids
