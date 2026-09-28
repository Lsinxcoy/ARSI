"""CTM fidelity: NLM+sync together, diagnostic-only."""
from __future__ import annotations


class TestCTMFidelity:
    def test_both_required(self):
        from arsi.foundation.sync_repr import ctm_fidelity_check

        assert ctm_fidelity_check()["ok"] is True
        r = ctm_fidelity_check(nlm_present=True, sync_present=False)
        assert r["ok"] is False and r["both_required"] is False
        r2 = ctm_fidelity_check(diagnostic_only=False)
        assert r2["ok"] is False

    def test_half_lives(self):
        from arsi.foundation.sync_repr import HALF_LIVES_S, half_life_to_decay

        assert set(HALF_LIVES_S) >= {"short", "med", "long"}
        assert abs(half_life_to_decay(300.0) - (0.693147 / 300.0)) < 1e-4
