"""SAHOO CAR frontier + stop priority tests."""
from __future__ import annotations


class TestCARFrontier:
    def test_alignment_cost_rising(self):
        from arsi.harness.sahoo import WEIGHTS_STATUS, car_frontier

        r = car_frontier([1.0, 0.95, 0.7, 0.65, 0.62, 0.6])
        assert r["trend"] == "alignment_cost_rising"
        assert WEIGHTS_STATUS["must_recalibrate"] is True

    def test_stop_priority_cps_zero(self):
        from arsi.harness.sahoo import decide_stop

        v = decide_stop(constraint_preservation=0.0, gdi=0.0, regression=0.0, quality_series=[1, 1, 1])
        assert v.stop is True
        assert v.rule == "constraint_zero"
