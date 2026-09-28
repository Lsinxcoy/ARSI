"""Tests: armor actually wired into runtime paths."""
from __future__ import annotations


class TestRuntimeWiring:
    def test_admit_extras_domain_guard_blocks(self):
        from arsi.harness.runtime_wiring import admit_extras

        ok, reason, _ = admit_extras(valid_before=0.9, valid_after=0.8)
        assert ok is False
        assert reason == "valid_output_dropped"

    def test_admit_extras_everitt_blocks(self):
        from arsi.harness.runtime_wiring import admit_extras

        ok, reason, _ = admit_extras(
            utility_rewrite=True,
            value_anticipates_rewrite=False,
            evaluates_with_current_utility=True,
        )
        assert ok is False
        assert "everitt" in reason

    def test_armor_health_shape(self):
        from arsi.harness.runtime_wiring import armor_health

        h = armor_health([0.5, 0.8, 0.7, 0.6], gdi=0.5)
        assert "dual_sensor" in h
        assert "overshoot" in h
        assert "conformance" in h
        assert "shared_calib" in h
        assert h["dual_sensor"]["goal_drift_suspect"] is True

    def test_call_guard_logs_conformance(self):
        from arsi.harness.call_guard import guard_call
        from arsi.harness.runtime_wiring import call_conformance

        guard_call("llm", {}, lambda: {"text": "nvapi-secret"})
        r = call_conformance()
        assert r.n >= 1
        assert r.n_fallback >= 1

    def test_brief_compress(self):
        from arsi.harness.runtime_wiring import brief_compress

        s = brief_compress([f"lesson-{i} " + "x" * 100 for i in range(20)], max_chars=2000)
        assert len(s) <= 2000
        assert "lesson" in s

    def test_admit_with_detail_extras(self):
        from arsi.harness.accept import admit

        r = admit(
            diff="add retry",
            summary="ok",
            score_hat=0.9,
            s_star=0.8,
            delta=0.01,
            dS=0.05,
            dC=0.0,
            components=["prompt"],
            accepted_counts={},
            detail_extras={"valid_before": 0.9, "valid_after": 0.5},
        )
        assert r.admissible is False
        assert r.reason == "valid_output_dropped"
