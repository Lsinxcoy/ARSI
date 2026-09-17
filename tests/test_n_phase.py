"""Tests for N1-N6 components."""
import pytest
import time

from arsi.foundation.schema import BehaviorTrace, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.sealed_eval.gain_decomposition import GainDecomposer
from arsi.foundation.cost_ledger import CostLedger
from arsi.empowerment.lifecycle_integration import DimensionLifecycleIntegrator
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.governor.core import DimensionManager
from arsi.world_model.dynamics import DynamicsModel
from arsi.world_model.counterfactual import CounterfactualSimulator
from arsi.mnemosyne.core import EdgeDiscovery


@pytest.fixture
def store():
    s = MnemosyneStore(":memory:")
    yield s
    s.close()


# ── N1: Gain Decomposition ────────────────────────────────────────

class TestGainDecomposer:
    def test_no_gain(self, store):
        decomposer = GainDecomposer(store, llm=None)
        result = decomposer.decompose(
            {"capability": 0.65}, {"capability": 0.65}, []
        )
        assert result["total"] == 0.0
        assert result["amplified"] == 0.0

    def test_positive_gain_rule_based(self, store):
        decomposer = GainDecomposer(store, llm=None)
        traces = [
            {"action": "evolve", "outcome": "success", "effect": 0.7},
            {"action": "evolve", "outcome": "success", "effect": 0.6},
            {"action": "learn", "outcome": "success", "effect": 0.8},
            {"action": "dream", "outcome": "success", "effect": 0.5},
        ]
        result = decomposer.decompose(
            {"capability": 0.60}, {"capability": 0.70}, traces
        )
        assert result["total"] == pytest.approx(0.10, abs=0.01)
        assert result["amplified"] > 0  # evolve actions
        assert result["imported"] > 0   # learn action
        assert result["self_organized"] > 0  # dream action
        # Sum should approximately equal total
        total_parts = result["amplified"] + result["imported"] + result["self_organized"]
        assert total_parts == pytest.approx(result["total"], abs=0.01)

    def test_record(self, store):
        decomposer = GainDecomposer(store, llm=None)
        result = decomposer.decompose(
            {"capability": 0.5}, {"capability": 0.6},
            [{"action": "learn", "outcome": "success", "effect": 0.8}],
        )
        decomposer.record(result, "term_1")
        # Should write to store
        memories = store.search_memories(limit=5)
        assert len(memories) > 0


# ── N2: Cost Ledger ───────────────────────────────────────────────

class TestCostLedger:
    def test_initial_state(self):
        ledger = CostLedger()
        snap = ledger.snapshot()
        assert snap["tokens"]["total"] == 0
        assert snap["verifier"]["queries"] == 0

    def test_record_llm_call(self):
        ledger = CostLedger()
        ledger.record_llm_call(input_tokens=100, output_tokens=50)
        snap = ledger.snapshot()
        assert snap["tokens"]["input"] == 100
        assert snap["tokens"]["output"] == 50
        assert snap["tokens"]["llm_calls"] == 1

    def test_record_verifier_query(self):
        ledger = CostLedger()
        ledger.record_verifier_query(3)
        snap = ledger.snapshot()
        assert snap["verifier"]["queries"] == 3

    def test_cost_benefit(self):
        ledger = CostLedger()
        ledger.record_llm_call(1000, 500)
        ledger.record_verifier_query(2)
        cb = ledger.cost_benefit(gain=0.15)
        assert cb["gain"] == 0.15
        assert cb["cost"]["tokens"] == 1500
        assert "gain_per_1k_tokens" in cb["efficiency"]

    def test_reset(self):
        ledger = CostLedger()
        ledger.record_llm_call(100, 50)
        ledger.reset()
        snap = ledger.snapshot()
        assert snap["tokens"]["total"] == 0

    def test_active_time_tracking(self):
        ledger = CostLedger()
        time.sleep(0.01)
        ledger.record_llm_call(10, 5)
        snap = ledger.snapshot()
        assert snap["time"]["active_seconds"] >= 0


# ── N3: Dimension Lifecycle Integration ───────────────────────────

class TestDimensionLifecycleIntegrator:
    def test_run_and_integrate(self, store):
        orch = DimensionOrchestrator(store, llm=None)
        dim_mgr = DimensionManager(max_per_term=2)
        integrator = DimensionLifecycleIntegrator(store, orch, dim_mgr, llm=None)

        traces = [
            {"action": "learn", "outcome": "success", "effect": 0.7, "action_params": {"x": 1}},
            {"action": "evolve", "outcome": "failure", "effect": 0.2, "action_params": {}},
        ]
        state = WorldState()
        result = integrator.run_and_integrate(traces, state)

        assert "dimension_results" in result
        assert "lifecycle_updates" in result
        assert "recommendation" in result
        assert result["integration_count"] == 1

    def test_lifecycle_report(self, store):
        orch = DimensionOrchestrator(store, llm=None)
        dim_mgr = DimensionManager(max_per_term=2)
        integrator = DimensionLifecycleIntegrator(store, orch, dim_mgr, llm=None)

        report = integrator.get_lifecycle_report()
        assert "knowledge" in report
        assert "calibration" in report

    def test_recommendation(self, store):
        orch = DimensionOrchestrator(store, llm=None)
        dim_mgr = DimensionManager(max_per_term=2)
        integrator = DimensionLifecycleIntegrator(store, orch, dim_mgr, llm=None)

        traces = [{"action": "learn", "outcome": "failure", "effect": 0.2}] * 10
        result = integrator.run_and_integrate(traces, WorldState())
        rec = result["recommendation"]
        assert "focus_dimensions" in rec
        assert len(rec["focus_dimensions"]) <= 2


# ── N4: Edge Discovery Semantic ──────────────────────────────────

class TestEdgeDiscoverySemantic:
    def test_tag_scan_fallback(self, store):
        from arsi.foundation.schema import MemoryRecord, MemoryZone
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE, content="test A", tags=["shared", "a"], agent_id="x"))
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE, content="test B", tags=["shared", "b"], agent_id="y"))

        ed = EdgeDiscovery(store)
        associations = ed.scan(min_shared_tags=1)
        assert len(associations) > 0
        assert associations[0].get("cross_agent") is True

    def test_enable_semantic(self, store):
        ed = EdgeDiscovery(store)
        ed.enable_semantic()
        assert ed._semantic_available is True

    def test_empty_store(self, store):
        ed = EdgeDiscovery(store)
        associations = ed.scan()
        assert associations == []


# ── N6: Counterfactual Simulator ─────────────────────────────────

class TestCounterfactualSimulator:
    @pytest.fixture
    def dynamics(self, store):
        # Create traces with state transitions
        for i in range(30):
            action = ["learn", "evolve", "dream"][i % 3]
            before = WorldState(
                phi={"generation": i // 10, "steps_since_change": i,
                     "storage_stats": {"trace_count": i, "experience_count": i // 2}},
                eta=0.2,
            )
            after = WorldState(
                phi={"generation": (i + 1) // 10, "steps_since_change": i + 1,
                     "storage_stats": {"trace_count": i + 1, "experience_count": (i + 1) // 2}},
                eta=0.15,
            )
            store.write_trace(BehaviorTrace(
                agent_id="test", action=action, outcome="success", effect=0.7,
                state_before=before, state_after=after,
            ))
        model = DynamicsModel(store, llm=None)
        model.train()
        return model

    def test_simulate(self, dynamics):
        sim = CounterfactualSimulator(dynamics, llm=None)
        state = WorldState()
        trajectory = sim.simulate(state, "learn", depth=2)
        assert len(trajectory) >= 1

    def test_compare_interventions(self, dynamics):
        sim = CounterfactualSimulator(dynamics, llm=None)
        state = WorldState()
        results = sim.compare_interventions(state, ["learn", "evolve", "dream"], depth=2)
        assert len(results) == 3
        assert all("score" in r for r in results)

    def test_adaptive_depth(self, dynamics):
        sim = CounterfactualSimulator(dynamics, llm=None)
        state = WorldState(eta=0.10)
        assert sim.adaptive_depth(state) == 5
        state2 = WorldState(eta=0.35)
        assert sim.adaptive_depth(state2) == 1
        state3 = WorldState(eta=0.60)
        assert sim.adaptive_depth(state3) == 0

    def test_stats(self, dynamics):
        sim = CounterfactualSimulator(dynamics, llm=None)
        sim.simulate(WorldState(), "learn", depth=1)
        stats = sim.stats
        assert stats["simulation_count"] == 1


# ── N5: Expanded Task Set ────────────────────────────────────────

class TestExpandedTasks:
    def test_task_set_loaded(self, tmp_path):
        from arsi.sealed_eval.evaluator import SealedEvaluator

        class NullAgent:
            def execute(self, prompt, timeout=60): return ""
            def declare_capability(self, task_id): return 0.5

        # Use real task file
        import shutil
        src = "config/sealed_tasks.yaml"
        dst = tmp_path / "sealed_tasks.yaml"
        shutil.copy(src, dst)

        evaluator = SealedEvaluator(dst, NullAgent())
        assert evaluator.task_count >= 15  # Expanded from 5 to 15+
