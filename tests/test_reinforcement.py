"""Tests for quality gate and operator scheduler."""
import pytest

from arsi.foundation.quality_gate import TraceQualityGate, CurriculumScheduler, DifficultyLevel
from arsi.governor.operator_scheduler import OperatorScheduler, ImprovementOperator
from arsi.foundation.schema import EmpowermentDimension


class TestTraceQualityGate:
    @pytest.fixture
    def gate(self):
        return TraceQualityGate(llm=None)

    def test_gate1_pass(self, gate):
        ok, reason = gate.gate1_structural({"action": "learn", "outcome": "success", "effect": 0.8})
        assert ok is True

    def test_gate1_missing_action(self, gate):
        ok, reason = gate.gate1_structural({"outcome": "success", "effect": 0.5})
        assert ok is False
        assert reason == "missing_action"

    def test_gate1_effect_out_of_range(self, gate):
        ok, reason = gate.gate1_structural({"action": "learn", "outcome": "success", "effect": 1.5})
        assert ok is False

    def test_gate2_heuristic(self, gate):
        result = gate.gate2_semantic({"action": "learn", "outcome": "success", "effect": 0.9})
        assert result["goal_achievement"] == "PASS"

    def test_gate2_heuristic_failure(self, gate):
        result = gate.gate2_semantic({"action": "learn", "outcome": "failure", "effect": 0.2})
        assert result["goal_achievement"] == "FAIL"

    def test_gate3_difficulty_c0(self, gate):
        level = gate.gate3_difficulty({"action": "remember", "outcome": "success", "effect": 0.5, "params": {}})
        assert level == DifficultyLevel.C0

    def test_gate3_difficulty_c1(self, gate):
        level = gate.gate3_difficulty({"action": "learn", "outcome": "success", "effect": 0.9, "params": {}})
        assert level == DifficultyLevel.C1

    def test_gate3_difficulty_c2(self, gate):
        level = gate.gate3_difficulty({"action": "evolve", "outcome": "failure", "effect": 0.2, "params": {}})
        assert level == DifficultyLevel.C2

    def test_evaluate_full(self, gate):
        result = gate.evaluate({"action": "learn", "outcome": "success", "effect": 0.8, "params": {}})
        assert result["accepted"] is True
        assert result["gate3_difficulty"] in ["C0", "C1"]

    def test_evaluate_rejected(self, gate):
        result = gate.evaluate({"action": "", "outcome": "success", "effect": 0.5})
        assert result["accepted"] is False

    def test_stats(self, gate):
        gate.gate1_structural({"action": "learn", "outcome": "success", "effect": 0.5})
        gate.gate1_structural({"action": "", "outcome": "success", "effect": 0.5})
        stats = gate.stats
        assert stats["gate1_pass"] == 1
        assert stats["gate1_fail"] == 1


class TestCurriculumScheduler:
    @pytest.fixture
    def scheduler(self):
        return CurriculumScheduler()

    def test_phase1(self, scheduler):
        scheduler.set_term(1)
        assert "C0-C1" in scheduler.phase_description

    def test_phase3(self, scheduler):
        scheduler.set_term(6)
        assert "C0-C3" in scheduler.phase_description

    def test_select_traces(self, scheduler):
        scheduler.set_term(1)
        traces = [
            {"action": "a", "outcome": "success", "effect": 0.9},
            {"action": "b", "outcome": "failure", "effect": 0.1},
        ]
        quality = [
            {"accepted": True, "gate3_difficulty": "C1"},
            {"accepted": True, "gate3_difficulty": "C2"},
        ]
        selected = scheduler.select_traces(traces, quality)
        # C2 should be filtered out in phase 1
        assert len(selected) == 1
        assert selected[0]["action"] == "a"


class TestOperatorScheduler:
    @pytest.fixture
    def scheduler(self):
        return OperatorScheduler()

    def test_init(self, scheduler):
        assert len(scheduler.operators) == 6
        assert EmpowermentDimension.KNOWLEDGE in scheduler.operators

    def test_schedule(self, scheduler):
        selected = scheduler.schedule(budget=5.0, max_operators=3)
        assert len(selected) <= 3
        assert all(op.cost <= 5.0 for op in selected)

    def test_freshness_decay(self, scheduler):
        scheduler.mark_capability_change()
        for op in scheduler.operators.values():
            assert op.signal_freshness == 0.5

    def test_stale_filter(self, scheduler):
        # Make all operators stale
        for _ in range(5):
            scheduler.mark_capability_change()
        selected = scheduler.schedule(budget=100, max_operators=10)
        # All should be filtered out (freshness < 0.3)
        assert len(selected) == 0

    def test_record_application(self, scheduler):
        scheduler.record_application(EmpowermentDimension.KNOWLEDGE, success=True)
        op = scheduler.operators[EmpowermentDimension.KNOWLEDGE]
        assert op.total_applications == 1
        assert op.success_rate == 1.0

    def test_revise_policy(self, scheduler):
        # Simulate low success rate
        for _ in range(5):
            scheduler.record_application(EmpowermentDimension.KNOWLEDGE, success=False)
        result = scheduler.revise_proposal_policy(EmpowermentDimension.KNOWLEDGE)
        assert result["status"] == "revised"

    def test_stats(self, scheduler):
        scheduler.schedule(budget=5.0)
        stats = scheduler.stats
        assert stats["generation"] == 1
        assert "operators" in stats
