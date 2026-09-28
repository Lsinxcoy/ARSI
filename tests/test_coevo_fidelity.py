"""Co-evolution fidelity: stages + anchored meta."""
from __future__ import annotations


class TestCoEvoFidelity:
    def test_anchored_meta_blocks_rho(self):
        from arsi.harness.coevolution import (
            STAGE3_META,
            anchored_meta_ok,
            classify_coevolution,
            coevo_fidelity_check,
        )

        ok, why = anchored_meta_ok(True, True, True)
        assert ok is False and "rho" in why
        ok2, why2 = anchored_meta_ok(True, False, True)
        assert ok2 is True and why2 == "anchored_meta_coevolution"
        r = coevo_fidelity_check(stage=STAGE3_META, gamma_touches_rho=True)
        assert r["ok"] is False
        v = classify_coevolution(
            n_evolving_units=2, both_units_change=True, mutual_pressure=True, env_adapts=True
        )
        assert v.stage == "agent_environment"
