"""AIDE² outer loop + OOD split tests."""
from __future__ import annotations


class TestOODSplit:
    def test_ood_never_in_selection(self):
        from arsi.harness.holdout import split_tasks

        ids = [f"t{i}" for i in range(20)]
        sp = split_tasks(ids, private_ratio=0.2, blacklist=["t0"], ood=["t19"], seed=17)
        assert "t19" in sp.ood
        assert "t19" not in sp.selection
        assert "t0" in sp.blacklist and "t0" not in sp.selection
        assert sp.is_private("t19") is True


class TestOuterLoop:
    def test_argmax_private_grade(self):
        from arsi.harness.holdout import accept_rewrite, outer_loop_select

        # public win private lose must be rejected before outer select
        bad = accept_rewrite(public_gain=0.2, private_gain=-0.1, budget_delta=0.0)
        assert bad["accepted"] is False
        rows = [
            {"id": "a1", "private_grade": 0.72, "public_gain": 0.05},
            {"id": "a2", "private_grade": 0.78, "public_gain": 0.02},
            {"id": "inc", "private_grade": 0.75, "public_gain": 0.0},
        ]
        sel = outer_loop_select(rows, incumbent_id="inc")
        assert sel["best_id"] == "a2"
        assert sel["rule"] == "argmax_private_grade"

    def test_incumbent_wins_ties(self):
        from arsi.harness.holdout import outer_loop_select

        rows = [
            {"id": "inc", "private_grade": 0.8},
            {"id": "new", "private_grade": 0.8},
        ]
        sel = outer_loop_select(rows, incumbent_id="inc")
        assert sel["best_id"] == "inc"
