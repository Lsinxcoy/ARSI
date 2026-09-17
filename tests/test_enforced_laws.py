"""Tests for enforced iron laws (G5, G6, G10)."""
import pytest
import time

from arsi.foundation.enforced_laws import (
    G5ObjectiveUtility,
    G6CircuitBreaker,
    G10SealedProtection,
    EnforcedIronLaws,
    CircuitState,
)
from arsi.foundation.iron_laws import IronLawViolation


class TestG5ObjectiveUtility:
    def test_valid_effect(self):
        g5 = G5ObjectiveUtility()
        assert g5.validate_effect("hebbian", 0.8, "trace_123") is True

    def test_effect_out_of_range_high(self):
        g5 = G5ObjectiveUtility()
        with pytest.raises(IronLawViolation):
            g5.check_effect_range(1.5, "test")

    def test_effect_out_of_range_low(self):
        g5 = G5ObjectiveUtility()
        with pytest.raises(IronLawViolation):
            g5.check_effect_range(-1.5, "test")

    def test_effect_boundary_valid(self):
        g5 = G5ObjectiveUtility()
        g5.check_effect_range(1.0, "test")  # Should not raise
        g5.check_effect_range(-1.0, "test")  # Should not raise

    def test_self_eval_blocked(self):
        g5 = G5ObjectiveUtility()
        with pytest.raises(IronLawViolation):
            g5.check_no_self_eval("hebbian", "hebbian_internal_report")

    def test_self_eval_allowed_different_source(self):
        g5 = G5ObjectiveUtility()
        g5.check_no_self_eval("hebbian", "external_audit")  # Should not raise

    def test_validate_no_evidence_warns(self):
        g5 = G5ObjectiveUtility()
        result = g5.validate_effect("test", 0.5, evidence="")
        assert result is False  # Warns but doesn't raise

    def test_violation_count(self):
        g5 = G5ObjectiveUtility()
        try:
            g5.check_effect_range(2.0, "test")
        except IronLawViolation:
            pass
        assert g5.violation_count == 1


class TestG6CircuitBreaker:
    def test_initial_state(self):
        g6 = G6CircuitBreaker()
        assert g6.state == CircuitState.CLOSED
        assert g6.allow_call() is True

    def test_trips_after_threshold(self):
        g6 = G6CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            g6.record_failure()
        assert g6.state == CircuitState.OPEN
        assert g6.allow_call() is False

    def test_recovers_after_cooldown(self):
        g6 = G6CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        g6.record_failure()
        assert g6.state == CircuitState.OPEN
        time.sleep(0.15)
        assert g6.allow_call() is True  # Should transition to HALF_OPEN
        assert g6.state == CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        g6 = G6CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        g6.record_failure()
        time.sleep(0.15)
        g6.allow_call()  # Trigger HALF_OPEN
        g6.record_success()
        assert g6.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        g6 = G6CircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        g6.record_failure()
        time.sleep(0.15)
        g6.allow_call()  # Trigger HALF_OPEN
        g6.record_failure()
        assert g6.state == CircuitState.OPEN

    def test_manual_reset(self):
        g6 = G6CircuitBreaker(failure_threshold=1)
        g6.record_failure()
        assert g6.state == CircuitState.OPEN
        g6.reset()
        assert g6.state == CircuitState.CLOSED

    def test_stats(self):
        g6 = G6CircuitBreaker(failure_threshold=2)
        g6.record_failure()
        stats = g6.stats
        assert stats["state"] == "closed"
        assert stats["failure_count"] == 1


class TestG10SealedProtection:
    def test_normal_path_allowed(self):
        g10 = G10SealedProtection()
        assert g10.check_access("src/arsi/core.py", "read", "governor") is True

    def test_protected_path_blocked(self):
        g10 = G10SealedProtection()
        g10.register_protected_path("config/sealed_tasks.yaml")
        with pytest.raises(IronLawViolation):
            g10.check_access("config/sealed_tasks.yaml", "read", "governor")

    def test_sealed_evaluator_can_access(self):
        g10 = G10SealedProtection()
        g10.register_protected_path("config/sealed_tasks.yaml")
        assert g10.check_access("config/sealed_tasks.yaml", "read", "sealed_evaluator") is True

    def test_release_gate_blocked_no_eval(self):
        g10 = G10SealedProtection()
        with pytest.raises(IronLawViolation):
            g10.check_release_gate(eval_passed=False, eval_score=0.8)

    def test_release_gate_blocked_low_score(self):
        g10 = G10SealedProtection()
        with pytest.raises(IronLawViolation):
            g10.check_release_gate(eval_passed=True, eval_score=0.3, min_score=0.5)

    def test_release_gate_passed(self):
        g10 = G10SealedProtection()
        g10.check_release_gate(eval_passed=True, eval_score=0.8, min_score=0.5)  # No raise

    def test_access_log_append_only(self):
        g10 = G10SealedProtection()
        g10.register_protected_path("sealed.yaml")
        try:
            g10.check_access("sealed.yaml", "read", "hacker")
        except IronLawViolation:
            pass
        log = g10.access_log
        assert len(log) == 1
        assert log[0]["allowed"] is False

    def test_violation_count(self):
        g10 = G10SealedProtection()
        g10.register_protected_path("sealed.yaml")
        for _ in range(3):
            try:
                g10.check_access("sealed.yaml", "read", "hacker")
            except IronLawViolation:
                pass
        assert g10.violation_count == 3


class TestEnforcedIronLaws:
    def test_full_pipeline(self):
        laws = EnforcedIronLaws(sealed_task_path="config/sealed_tasks.yaml")

        # G5: validate effect
        assert laws.validate_effect("hebbian", 0.8, "evidence_1") is True

        # G6: check call allowed
        assert laws.check_call_allowed() is True

        # G10: sealed access blocked
        with pytest.raises(IronLawViolation):
            laws.check_sealed_access("config/sealed_tasks.yaml", "read", "mechanism")

    def test_violation_summary(self):
        laws = EnforcedIronLaws()
        try:
            laws.g5.check_effect_range(2.0, "test")
        except IronLawViolation:
            pass
        laws.record_call_result(False)

        summary = laws.get_violation_summary()
        assert summary["g5_violations"] == 1
        assert summary["g6_state"]["failure_count"] == 1
