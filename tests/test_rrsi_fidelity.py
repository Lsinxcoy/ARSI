"""RRSI fidelity tests: relative cost rule, three-track, stall U_t, δ from H0."""
from __future__ import annotations


class TestCostRule:
    def test_relative_cost_and_branch_a(self):
        from arsi.harness.rrsi import cost_rule_ok, relative_cost_delta

        dC = relative_cost_delta(125.0, 100.0)
        assert abs(dC - 0.25) < 1e-9
        # coding: β0=0.10 β1=44.5 → for ΔS=0.01, limit=0.545
        ok, _ = cost_rule_ok(0.01, 0.25, 0.10, 44.5)
        assert ok is True
        bad, why = cost_rule_ok(0.01, 2.0, 0.10, 44.5)
        assert bad is False and "cost_exceeds" in why

    def test_coding_ws_zero(self):
        from arsi.harness.rrsi import WS

        assert WS["coding"] == 0.0


class TestThreeTrack:
    def test_table1_style(self):
        from arsi.harness.rrsi import three_track_eval

        r = three_track_eval([0.9, 0.91], [0.88, 0.89], [0.42, 0.45], h0_ood=0.397)
        assert r.evolve_mean > r.id_holdout_mean
        assert r.ood_gain_vs_h0 > 0
        assert r.evolve_n == 2


class TestRRSIRound:
    def test_floor_rejects(self):
        from arsi.harness.rrsi import rrsi_round

        v = rrsi_round(
            t=0, T=20,
            score_hat=0.5, score_now=0.5, score_w_ago=0.5, s_star=0.8, delta=0.03,
            dS=0.0, c_new=100, c_base=100,
        )
        assert v.admissible is False and v.reason == "floor"

    def test_above_delta_cost(self):
        from arsi.harness.rrsi import rrsi_round

        v = rrsi_round(
            t=5, T=20,
            score_hat=0.9, score_now=0.9, score_w_ago=0.5, s_star=0.85, delta=0.03,
            dS=0.1, c_new=150, c_base=100, domain="coding",
            components=["prompt"],
        )
        assert v.admissible is True
        assert v.branch == "above_delta"
        assert 1 <= v.b_t <= 4

    def test_stall_reserves_unexercised(self):
        from arsi.harness.rrsi import rrsi_round

        v = rrsi_round(
            t=10, T=20,
            score_hat=0.86, score_now=0.86, score_w_ago=0.85, s_star=0.85, delta=0.03,
            dS=0.01, c_new=100, c_base=100, domain="coding",
            components=["prompt"],
            all_components=["prompt", "skill", "memory", "client_tool"],
        )
        assert v.stalled is True
        assert v.m_draft == 1
        assert "skill" in v.unexercised

    def test_delta_only_from_h0(self):
        from arsi.harness.rrsi import delta_from_h0

        d = delta_from_h0([0.88, 0.90, 0.89, 0.91])
        assert d > 0
        assert abs(d - (0.91 - 0.88) / 2) < 1e-9
