"""Tests for foundation layer."""
import pytest
from datetime import datetime

from arsi.foundation.schema import (
    BehaviorTrace,
    Belief,
    CausalLink,
    CoupledAction,
    EmpowermentDimension,
    EmpowermentOp,
    MemoryRecord,
    MemoryZone,
    MentalState,
    PhysicalState,
    VerificationStatus,
    WorldState,
)
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.iron_laws import IronLaws, IronLawViolation


# ── Schema Tests ───────────────────────────────────────────────────

class TestSchema:
    def test_physical_state_defaults(self):
        phi = PhysicalState()
        assert phi.generation == 0
        assert phi.steps_since_change == 0
        assert phi.mechanisms == {}

    def test_mental_state_defaults(self):
        psi = MentalState()
        assert psi.beliefs == []
        assert psi.goals == []
        assert psi.norms == []

    def test_world_state(self):
        state = WorldState()
        assert state.eta == 0.0
        assert isinstance(state.phi, PhysicalState)
        assert isinstance(state.psi, MentalState)

    def test_belief_confidence_range(self):
        b = Belief(content="test", confidence=0.8)
        assert 0 <= b.confidence <= 1

    def test_coupled_action_freeze(self):
        action = CoupledAction.freeze("G3 铁律触发")
        assert action.a_phy == "freeze"
        assert "铁律" in action.a_ment.reason

    def test_empowerment_op_defaults(self):
        op = EmpowermentOp()
        assert op.verification == VerificationStatus.UNKNOWN
        assert op.rolled_back is False

    def test_memory_record_zones(self):
        for zone in MemoryZone:
            r = MemoryRecord(zone=zone, content="test")
            assert r.zone == zone
            assert r.status.value == "active"


# ── Store Tests ────────────────────────────────────────────────────

class TestStore:
    @pytest.fixture
    def store(self):
        s = MnemosyneStore(":memory:")
        yield s
        s.close()

    def test_write_and_read_memory(self, store):
        record = MemoryRecord(zone=MemoryZone.PROXY, content="hello", agent_id="a1")
        assert store.write_memory(record) is True
        loaded = store.read_memory(record.id)
        assert loaded is not None
        assert loaded.content == "hello"
        assert loaded.agent_id == "a1"

    def test_three_zone_isolation(self, store):
        store.write_memory(MemoryRecord(zone=MemoryZone.PROXY, content="p1", agent_id="a"))
        store.write_memory(MemoryRecord(zone=MemoryZone.EXPERIENCE, content="e1"))
        store.write_memory(MemoryRecord(zone=MemoryZone.SELF, content="s1"))

        proxy = store.search_memories(zone=MemoryZone.PROXY)
        exp = store.search_memories(zone=MemoryZone.EXPERIENCE)
        self_ = store.search_memories(zone=MemoryZone.SELF)

        assert len(proxy) == 1
        assert len(exp) == 1
        assert len(self_) == 1

    def test_generation_tracking(self, store):
        assert store.current_generation == 0
        g = store.bump_generation()
        assert g == 1
        assert store.current_generation == 1

    def test_causal_link(self, store):
        link = CausalLink(
            experience_id="exp_001",
            source_traces=["trace_1", "trace_2"],
        )
        link_id = store.write_causal_link(link)
        assert link_id > 0

        loaded = store.get_causal_link("exp_001")
        assert loaded is not None
        assert loaded.source_traces == ["trace_1", "trace_2"]

    def test_behavior_trace(self, store):
        trace = BehaviorTrace(
            agent_id="a1",
            action="learn",
            outcome="success",
            effect=0.5,
        )
        assert store.write_trace(trace) is True
        traces = store.get_recent_traces(n=10)
        assert len(traces) == 1
        assert traces[0]["action"] == "learn"

    def test_effect_ledger(self, store):
        store.record_effect("hebbian", 0.8, "test context")
        store.record_effect("hebbian", 0.6, "test context 2")
        ledger = store.get_effect_ledger(mechanism="hebbian")
        assert len(ledger) == 2

    def test_utility_update(self, store):
        record = MemoryRecord(zone=MemoryZone.EXPERIENCE, content="test")
        store.write_memory(record)
        store.update_utility(record.id, 0.3)
        loaded = store.read_memory(record.id)
        assert loaded.utility_score == pytest.approx(0.3)

    def test_archive(self, store):
        record = MemoryRecord(zone=MemoryZone.PROXY, content="old")
        store.write_memory(record)
        store.archive_memory(record.id)
        loaded = store.read_memory(record.id)
        assert loaded.status.value == "archived"

    def test_stats(self, store):
        store.write_memory(MemoryRecord(zone=MemoryZone.PROXY, content="p"))
        store.write_memory(MemoryRecord(zone=MemoryZone.EXPERIENCE, content="e"))
        stats = store.get_stats()
        assert stats["proxy_count"] == 1
        assert stats["experience_count"] == 1


# ── Iron Laws Tests ────────────────────────────────────────────────

class TestIronLaws:
    @pytest.fixture
    def laws(self, tmp_path):
        config = tmp_path / "iron_laws.yaml"
        config.write_text("""
laws:
  - id: G10
    name: 密封评估层
    description: 在所有写掩码之外
    severity: block
""", encoding="utf-8")
        return IronLaws(config)

    def test_load_laws(self, laws):
        assert "G10" in laws.law_ids

    def test_default_laws_when_no_config(self, tmp_path):
        laws = IronLaws(tmp_path / "nonexistent.yaml")
        assert len(laws.law_ids) == 10
        assert "G1" in laws.law_ids
        assert "G10" in laws.law_ids

    def test_budget_concentration_blocked(self, laws):
        with pytest.raises(IronLawViolation):
            laws.validate_policy({"budget_allocation": {"skill": 0.95}})

    def test_balanced_budget_ok(self, laws):
        laws.validate_policy({"budget_allocation": {"skill": 0.3, "harness": 0.3, "knowledge": 0.2}})

    def test_violated_returns_false_by_default(self, laws):
        class FakeState:
            pass
        assert laws.violated(FakeState()) is False
