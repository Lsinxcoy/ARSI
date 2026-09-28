"""SEVerA well-formedness + FGGM chain tests."""
from __future__ import annotations


class TestFGGM:
    def test_fallback_satisfies_psi(self):
        from arsi.harness.contracts import DEFAULT_CONTRACTS, apply_contracts

        # G3: frozen + writes must fall back
        res = apply_contracts({"frozen": True}, {"writes": True, "apply": True})
        g3 = next(r for r in res if r.law_id == "G3")
        assert g3.used_fallback is True
        out = g3.output
        assert out.get("writes") is False

    def test_guarded_chain(self):
        from arsi.harness.contracts import guarded

        final, res = guarded({"frozen": False}, lambda: {"writes": True, "mutate_sealed": True})
        assert any(r.law_id == "G10" and r.used_fallback for r in res)
        assert final.get("mutate_sealed") is False


class TestWellFormedness:
    def test_default_contracts_wellformed(self):
        from arsi.harness.contracts import well_formedness

        r = well_formedness()
        assert r["note"].startswith("severa_wellformedness")
        assert set(r["laws"]) >= {"G3", "G5", "G6", "G10"}
        # at least block-severity laws must be fallback-valid
        for lid in ("G3", "G10"):
            assert r["laws"][lid]["fallback_valid"] is True
            assert r["laws"][lid]["checker_sound"] is True
