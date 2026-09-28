"""HarnessX AEGIS fidelity tests."""
from __future__ import annotations


class TestAEGISFidelity:
    def test_c7_frozen_pipeline(self):
        from arsi.harness.taxonomy import FROZEN_DIMS, aegis_fidelity_check

        assert "c7" in FROZEN_DIMS
        assert aegis_fidelity_check()["ok"] is True
        r = aegis_fidelity_check(frozen_dims=[], has_gates=False)
        assert r["ok"] is False
