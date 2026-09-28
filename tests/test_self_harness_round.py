"""Self-Harness loop: mine + minimal + dual regression."""
from __future__ import annotations


class TestSelfHarnessRound:
    def test_minimal_preferred_and_dual_gate(self):
        from arsi.harness.minimality import self_harness_round

        traces = [
            {"outcome": "failure", "action": "tool", "params": {"fail_class": "tool_error"}},
            {"outcome": "failure", "action": "tool", "params": {"fail_class": "tool_error"}},
        ]
        proposals = [
            {"id": "fat", "diff": "+" + "\n+".join(["x"] * 40)},
            {"id": "thin", "diff": "+ one_line_fix\n"},
        ]
        out = self_harness_round(
            traces=traces,
            proposals=proposals,
            held_in_before=0.5,
            held_in_after=0.52,
            held_out_before=0.5,
            held_out_after=0.51,
        )
        assert out["accepted"] is True
        assert out["dual_regression"]["ok"] is True
        assert out["minimal_diff_score"] < 5
        assert out["weakness_clusters"][0]["label"] == "tool_error"

    def test_dual_regression_blocks(self):
        from arsi.harness.minimality import self_harness_round

        out = self_harness_round(
            traces=[],
            proposals=[{"diff": "+ x"}],
            held_in_before=0.5,
            held_in_after=0.4,
            held_out_before=0.5,
            held_out_after=0.5,
        )
        assert out["accepted"] is False
        assert "held_in_regress" in out["dual_regression"]["reason"]
