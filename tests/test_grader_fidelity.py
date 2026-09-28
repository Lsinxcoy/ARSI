"""Grader fidelity: clean≠correct, validity on anchor."""
from __future__ import annotations


class TestGraderFidelity:
    def test_negative_epistemology(self):
        from arsi.harness.drawback import grader_fidelity_check
        from arsi.harness.epistemic import Claim, admit_claim

        assert grader_fidelity_check()["ok"] is True
        bad = grader_fidelity_check(task_score_validates_evaluator=True)
        assert bad["ok"] is False
        ok, why = admit_claim(
            Claim(kind="no_known_defect", subject="s", drawbacks_checked=["a"], raw="no_known_defect")
        )
        assert ok is True
        ok2, why2 = admit_claim(
            Claim(kind="no_known_defect", subject="s", raw="this is_correct")
        )
        assert ok2 is False and "forbidden" in why2
