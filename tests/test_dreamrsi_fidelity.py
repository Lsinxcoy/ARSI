"""Dream-RSI selection fidelity tests."""
from __future__ import annotations


class TestDreamSelectionFidelity:
    def test_paper_discipline(self):
        from arsi.world_model.dream_rsi_deep import dream_selection_fidelity_check

        assert dream_selection_fidelity_check()["ok"] is True
        r = dream_selection_fidelity_check(semantic_guidance_in_selection=True)
        assert r["ok"] is False
        r2 = dream_selection_fidelity_check(current_in_candidate_set=False)
        assert r2["V_star_ge_V0"] is False and r2["ok"] is False

    def test_monotone_in_pool(self):
        from arsi.harness.monotone import select_manifest_monotone

        sel = select_manifest_monotone("cur", 0.5, [{"manifest_id": "m", "score": 0.9}])
        assert sel.monotone_ok is True and sel.best_score >= sel.baseline_score
