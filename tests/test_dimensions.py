"""Tests for empowerment dimension implementations."""
import pytest

from arsi.empowerment.dimensions import (
    KnowledgeDimension,
    DecompositionDimension,
    CalibrationDimension,
    DimensionOrchestrator,
)
from arsi.foundation.schema import VerificationStatus, WorldState
from arsi.foundation.store import MnemosyneStore


@pytest.fixture
def store():
    s = MnemosyneStore(":memory:")
    yield s
    s.close()


@pytest.fixture
def state():
    return WorldState()


def _make_traces(n=20, fail_rate=0.3):
    traces = []
    for i in range(n):
        traces.append({
            "action": ["learn", "evolve", "reflect", "remember"][i % 4],
            "outcome": "failure" if (i % int(1 / fail_rate)) == 0 else "success",
            "effect": 0.3 + (i % 10) * 0.07,
            "action_params": {"step": i},
        })
    return traces


class TestKnowledgeDimension:
    def test_sense_no_failures(self, store, state):
        dim = KnowledgeDimension(store, llm=None)
        traces = [{"action": "learn", "outcome": "success", "effect": 0.8}] * 10
        result = dim.sense(traces, state)
        assert result["gap_detected"] is False

    def test_sense_with_failures(self, store, state):
        dim = KnowledgeDimension(store, llm=None)
        traces = _make_traces(20, fail_rate=0.5)
        result = dim.sense(traces, state)
        assert "high_fail_actions" in result
        assert "failure_count" in result

    def test_generate_no_gap(self, store, state):
        dim = KnowledgeDimension(store, llm=None)
        result = dim.generate({"gap_detected": False}, [])
        assert result["action"] == "none"

    def test_validate_success(self, store, state):
        dim = KnowledgeDimension(store, llm=None)
        status, evidence = dim.validate(
            {"action": "inject_knowledge", "count": 3},
            {"success_rate": 0.5},
            {"success_rate": 0.7},
        )
        assert status == VerificationStatus.SUCCESS
        assert evidence["delta"] == pytest.approx(0.2, abs=0.01)

    def test_validate_unknown(self, store, state):
        dim = KnowledgeDimension(store, llm=None)
        status, evidence = dim.validate(
            {"action": "inject_knowledge", "count": 3},
            {"success_rate": 0.5},
            {"success_rate": 0.5},
        )
        assert status == VerificationStatus.UNKNOWN


class TestDecompositionDimension:
    def test_sense_no_traces(self, store, state):
        dim = DecompositionDimension(store, llm=None)
        result = dim.sense([], state)
        assert result["gap_detected"] is False

    def test_sense_consecutive_fails(self, store, state):
        dim = DecompositionDimension(store, llm=None)
        traces = [
            {"action": "learn", "outcome": "success", "effect": 0.8},
            {"action": "evolve", "outcome": "failure", "effect": 0.2},
            {"action": "evolve", "outcome": "failure", "effect": 0.1},
            {"action": "evolve", "outcome": "failure", "effect": 0.1},
        ]
        result = dim.sense(traces, state)
        assert result["max_consecutive_fails"] >= 2

    def test_validate_improvement(self, store, state):
        dim = DecompositionDimension(store, llm=None)
        status, evidence = dim.validate(
            {"action": "recommend_decomposition"},
            {"max_consecutive_fails": 3},
            {"max_consecutive_fails": 1},
        )
        assert status == VerificationStatus.SUCCESS


class TestCalibrationDimension:
    def test_sense_well_calibrated(self, store, state):
        dim = CalibrationDimension(store, llm=None)
        # Effect matches outcome (well calibrated)
        traces = [
            {"action": "learn", "outcome": "success", "effect": 0.9},
            {"action": "learn", "outcome": "success", "effect": 0.8},
            {"action": "evolve", "outcome": "failure", "effect": 0.2},
            {"action": "evolve", "outcome": "failure", "effect": 0.1},
        ]
        result = dim.sense(traces, state)
        assert result["ece"] <= 0.15
        assert result["gap_detected"] is False

    def test_sense_miscalibrated(self, store, state):
        dim = CalibrationDimension(store, llm=None)
        # High effect but failures (overconfident)
        traces = [
            {"action": "learn", "outcome": "failure", "effect": 0.9},
            {"action": "learn", "outcome": "failure", "effect": 0.8},
            {"action": "evolve", "outcome": "failure", "effect": 0.7},
            {"action": "evolve", "outcome": "failure", "effect": 0.6},
        ]
        result = dim.sense(traces, state)
        assert result["ece"] > 0.3
        assert result["gap_detected"] is True

    def test_generate_recommendations(self, store, state):
        dim = CalibrationDimension(store, llm=None)
        sense_result = {
            "gap_detected": True,
            "ece": 0.4,
            "calibration_details": {
                "high": {"count": 10, "avg_predicted": 0.9, "avg_actual": 0.5, "gap": 0.4},
            },
        }
        result = dim.generate(sense_result, [])
        assert result["action"] == "adjust_calibration"
        assert len(result["recommendations"]) > 0
        assert result["recommendations"][0]["issue"] == "过度自信"

    def test_validate_ece_improvement(self, store, state):
        dim = CalibrationDimension(store, llm=None)
        status, evidence = dim.validate(
            {"action": "adjust_calibration"},
            {"ece": 0.4},
            {"ece": 0.2},
        )
        assert status == VerificationStatus.SUCCESS
        assert evidence["improvement"] == pytest.approx(0.2, abs=0.01)


class TestDimensionOrchestrator:
    def test_run_all(self, store, state):
        orch = DimensionOrchestrator(store, llm=None)
        traces = _make_traces(30, fail_rate=0.4)
        results = orch.run_all(traces, state)
        assert "knowledge" in results
        assert "decomposition" in results
        assert "calibration" in results

    def test_run_single(self, store, state):
        orch = DimensionOrchestrator(store, llm=None)
        traces = _make_traces(20, fail_rate=0.3)
        result = orch.run_dimension(
            __import__("arsi.foundation.schema", fromlist=["EmpowermentDimension"]).EmpowermentDimension.CALIBRATION,
            traces, state,
        )
        assert result["dimension"] == "calibration"
        assert "sense" in result
        assert "generate" in result
        assert "validation_status" in result

    def test_compute_stats(self, traces=None):
        traces = _make_traces(20, fail_rate=0.3)
        stats = DimensionOrchestrator._compute_stats(traces)
        assert 0 <= stats["success_rate"] <= 1
        assert stats["max_consecutive_fails"] >= 0
