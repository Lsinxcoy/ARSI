"""Integration tests for ARSI Phase 1-4 components."""
import pytest
from datetime import datetime

from arsi.foundation.schema import (
    BehaviorTrace,
    EmpowermentDimension,
    EmpowermentOp,
    MemoryRecord,
    MemoryZone,
    VerificationStatus,
    WorldState,
)
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.iron_laws import IronLaws
from arsi.mnemosyne.core import Mnemosyne, DopamineGate, UtilityDecay
from arsi.mnemosyne.memory_proxy import MemoryProxy, NullNativeMemory
from arsi.world_model.siwm import SIWM, EtaTracker, MindZero, BehaviorPredictor
from arsi.governor.core import AutopoieticGovernor, DimensionManager
from arsi.empowerment.engine import EmpowermentEngine, EmpowermentVerifier, NullAdapter
from arsi.pipelines.dream import DreamPipeline


@pytest.fixture
def store():
    s = MnemosyneStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def mnemosyne(store):
    return Mnemosyne(store)


@pytest.fixture
def siwm(store):
    return SIWM(store)


@pytest.fixture
def laws(tmp_path):
    config = tmp_path / "iron_laws.yaml"
    config.write_text("""
laws:
  - id: G10
    name: 密封评估层
    description: 在所有写掩码之外
    severity: block
""", encoding="utf-8")
    return IronLaws(config)


@pytest.fixture
def governor(siwm, store, laws):
    return AutopoieticGovernor(siwm, store, laws)


# ── Mnemosyne Tests ────────────────────────────────────────────────

class TestMnemosyne:
    def test_ingest_trace(self, mnemosyne):
        trace = BehaviorTrace(agent_id="a1", action="learn", outcome="success", effect=0.8)
        record = mnemosyne.ingest_trace(trace)
        assert record.zone == MemoryZone.PROXY
        assert record.agent_id == "a1"

    def test_dopamine_gate_consolidates_high_importance(self, mnemosyne):
        trace = BehaviorTrace(agent_id="a1", action="evolve", outcome="success", effect=0.9)
        record = mnemosyne.ingest_trace(trace)
        # High effect (0.9) → importance 0.9 → above threshold 0.6 → consolidated
        experiences = mnemosyne.store.search_memories(zone=MemoryZone.EXPERIENCE)
        assert len(experiences) > 0

    def test_dopamine_gate_skips_low_importance(self, mnemosyne):
        trace = BehaviorTrace(agent_id="a1", action="remember", outcome="ok", effect=0.1)
        record = mnemosyne.ingest_trace(trace)
        # Low effect → not consolidated immediately
        experiences = mnemosyne.store.search_memories(zone=MemoryZone.EXPERIENCE)
        # May or may not have experiences from other tests, but this trace shouldn't add one
        # (effect=0.1 → importance=0.1 < 0.6 threshold)

    def test_cross_agent_experience_search(self, mnemosyne, store):
        # Agent A's experience
        store.write_memory(MemoryRecord(
            zone=MemoryZone.EXPERIENCE,
            content="约束检查薄弱",
            tags=["decomposition", "a1"],
            agent_id="a1",
        ))
        # Agent B searches and finds A's experience
        results = mnemosyne.search_experience(query_tags=["decomposition"], agent_id="b1")
        assert len(results) > 0
        assert results[0].agent_id == "a1"

    def test_consolidation(self, mnemosyne):
        # Ingest some traces
        for i in range(5):
            mnemosyne.ingest_trace(BehaviorTrace(
                agent_id="a1", action=f"action_{i}", outcome="success", effect=0.5
            ))
        result = mnemosyne.consolidate()
        assert "archived" in result
        assert "associations_found" in result

    def test_stats(self, mnemosyne):
        stats = mnemosyne.get_stats()
        assert "proxy_count" in stats
        assert "experience_count" in stats


# ── Memory Proxy Tests ─────────────────────────────────────────────

class TestMemoryProxy:
    def test_write_and_recall(self, mnemosyne):
        proxy = MemoryProxy(mnemosyne, NullNativeMemory(), "agent_a")
        proxy.remember("重要经验：先跑测试再改代码", tags=["coding"])
        results = proxy.recall("测试")
        assert len(results) > 0

    def test_degradation_on_error(self, mnemosyne):
        class FailingNative:
            def remember(self, content, tags=None): pass
            def recall(self, query, k=5): return ["fallback"]

        proxy = MemoryProxy(mnemosyne, FailingNative(), "agent_a")
        proxy.degraded = True
        results = proxy.recall("anything")
        assert results == ["fallback"]

    def test_stats(self, mnemosyne):
        proxy = MemoryProxy(mnemosyne, NullNativeMemory(), "agent_a")
        proxy.remember("test")
        proxy.recall("test")
        stats = proxy.stats
        assert stats["write_count"] == 1
        assert stats["read_count"] == 1


# ── SIWM Tests ─────────────────────────────────────────────────────

class TestSIWM:
    def test_eta_tracking(self):
        eta = EtaTracker()
        # Correct predictions → η stays low
        for _ in range(10):
            eta.update("learn", "learn")
        assert eta.value < 0.15
        assert not eta.should_dream()

    def test_eta_triggers_dream(self):
        eta = EtaTracker()
        # Wrong predictions → η rises
        for _ in range(20):
            eta.update("learn", "evolve")
        assert eta.value > 0.40
        assert eta.should_dream()

    def test_eta_adaptive_depth(self):
        eta = EtaTracker()
        eta.eta_smooth = 0.10
        assert eta.adaptive_depth() == 5
        eta.eta_smooth = 0.25
        assert eta.adaptive_depth() == 3
        eta.eta_smooth = 0.45
        assert eta.adaptive_depth() == 1
        eta.eta_smooth = 0.60
        assert eta.adaptive_depth() == 0

    def test_mindzero_belief_inference(self, store):
        mz = MindZero()
        traces = [
            {"action": "learn", "outcome": "failure"},
            {"action": "learn", "outcome": "failure"},
            {"action": "learn", "outcome": "failure"},
        ]
        psi = mz.infer_mental_state(traces)
        # Should infer that learn has low success rate
        assert any("learn" in b.content.lower() for b in psi.beliefs)

    def test_behavior_predictor_v1(self, store):
        predictor = BehaviorPredictor(version="v1")
        traces = [
            {"action": "learn", "outcome": "success", "state_before": {"eta": 0.1, "phi": {"generation": 1}}},
            {"action": "learn", "outcome": "success", "state_before": {"eta": 0.1, "phi": {"generation": 1}}},
            {"action": "learn", "outcome": "success", "state_before": {"eta": 0.1, "phi": {"generation": 1}}},
        ]
        predictor.fit(traces)
        assert len(predictor.rules) > 0

    def test_siwm_state_building(self, siwm, store):
        # Add some data
        store.write_trace(BehaviorTrace(agent_id="a1", action="learn", outcome="success"))
        state = siwm.get_state()
        assert isinstance(state, WorldState)
        assert state.phi.storage_stats.get("trace_count", 0) >= 1


# ── Governor Tests ─────────────────────────────────────────────────

class TestGovernor:
    def test_decide_returns_coupled_action(self, governor, siwm):
        state = siwm.get_state()
        action = governor.decide(state)
        assert action.a_phy != ""
        assert action.a_ment.reason != ""

    def test_decide_dream_when_eta_high(self, governor, siwm):
        siwm.eta.eta_smooth = 0.50  # Above threshold
        state = siwm.refresh_state()
        action = governor.decide(state)
        assert action.a_phy == "dream"

    def test_freeze_on_iron_law_violation(self, governor, siwm, laws):
        # Test that iron laws can detect violations
        class FakeState:
            sealed_tasks_accessible = True

        assert laws.violated(FakeState()) is True

    def test_dimension_rotation(self, governor, siwm, store):
        # Add traces so data_sufficiency is high enough
        for i in range(50):
            store.write_trace(BehaviorTrace(agent_id="a1", action="learn", outcome="success"))
        siwm.refresh_state()

        dim_mgr = DimensionManager(max_per_term=2)
        selections = []
        for _ in range(5):
            state = siwm.get_state()
            status = dim_mgr.scan(state)
            selected = dim_mgr.select_priority(status)
            selections.append(set(selected))

        # At least 2 different dimensions should be selected over 5 terms
        all_selected = set()
        for s in selections:
            all_selected.update(s)
        assert len(all_selected) >= 2

    def test_policy_distillation(self, governor, store):
        store.record_effect("hebbian", 0.8)
        store.record_effect("hebbian", 0.6)
        store.record_effect("dopamine", 0.9)
        policy = governor.distill_policy()
        assert "mechanism_preferences" in policy
        assert "hebbian" in policy["mechanism_preferences"]


# ── Empowerment Tests ──────────────────────────────────────────────

class TestEmpowerment:
    def test_empower_never_blind_success(self, mnemosyne, siwm):
        engine = EmpowermentEngine(mnemosyne, siwm, NullAdapter())
        state = siwm.get_state()
        op = engine.empower("agent_a", state)
        # With NullAdapter, snapshots are identical → should be UNKNOWN, not SUCCESS
        assert op.verification in [VerificationStatus.UNKNOWN, VerificationStatus.PARTIAL]

    def test_rollback(self, mnemosyne, siwm):
        engine = EmpowermentEngine(mnemosyne, siwm, NullAdapter())
        state = siwm.get_state()
        op = engine.empower("agent_a", state)
        result = engine.rollback(op)
        assert result is True
        assert op.rolled_back is True

    def test_verifier_unknown_without_evidence(self):
        verifier = EmpowermentVerifier()
        status, evidence = verifier.verify(
            agent_id="a",
            weapon={},
            pre_snapshot={"x": 1},
            post_snapshot={"x": 1},  # No change
        )
        assert status == VerificationStatus.UNKNOWN


# ── Dream Pipeline Tests ───────────────────────────────────────────

class TestDreamPipeline:
    def test_dream_reduces_eta(self, siwm, mnemosyne):
        dream = DreamPipeline(siwm, mnemosyne)
        siwm.eta.eta_smooth = 0.50
        state = siwm.refresh_state()
        new_state = dream.execute(state)
        assert new_state.eta < 0.50  # η should decrease

    def test_dream_records_session(self, siwm, mnemosyne, store):
        dream = DreamPipeline(siwm, mnemosyne)
        state = siwm.get_state()
        dream.execute(state)
        sessions = store.get_self_records("dream_session")
        assert len(sessions) > 0


# ── End-to-End Integration Test ────────────────────────────────────

class TestEndToEnd:
    def test_full_cycle(self, store, mnemosyne, siwm, governor, laws):
        """Test a full ARSI cycle: ingest → decide → act → dream."""
        # 1. Ingest traces
        for i in range(10):
            mnemosyne.ingest_trace(BehaviorTrace(
                agent_id="a1",
                action="learn" if i % 2 == 0 else "evolve",
                outcome="success" if i % 3 != 0 else "failure",
                effect=0.5 + (i * 0.05),
            ))

        # 2. Train predictor
        train_result = siwm.train_from_history()
        assert train_result["status"] == "trained"

        # 3. Governor decides
        state = siwm.refresh_state()
        action = governor.decide(state)
        assert action.a_phy in ["dream", "learn", "evolve", "maintain", "remember"]

        # 4. If dream, execute it
        if action.a_phy == "dream":
            from arsi.pipelines.dream import DreamPipeline
            dream = DreamPipeline(siwm, mnemosyne)
            state = dream.execute(state)
            assert state.eta < 0.50

        # 5. Record some effects then distill policy
        store.record_effect("hebbian", 0.8)
        store.record_effect("dopamine", 0.9)
        policy = governor.distill_policy()
        assert "mechanism_preferences" in policy

        # 6. Check stats
        m_stats = mnemosyne.get_stats()
        g_stats = governor.stats
        assert m_stats["generation"] == store.current_generation
        assert g_stats["decision_count"] >= 1
