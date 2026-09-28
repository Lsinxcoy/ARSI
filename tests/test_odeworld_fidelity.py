"""ODEWorld fidelity invariants tests."""
from __future__ import annotations


class TestFlowFidelity:
    def test_static_not_in_z(self):
        from arsi.world_model.capability_flow import (
            STATIC_KEYS,
            Z_BLOCKS,
            Z_KEYS,
            flow_fidelity_check,
        )

        r = flow_fidelity_check()
        assert r["ok"] is True
        assert set(Z_KEYS) & set(STATIC_KEYS) == set()
        assert r["static_in_z"] == []
        # blocks cover all z keys once
        covered = [k for keys in Z_BLOCKS.values() for k in keys]
        assert sorted(covered) == sorted(Z_KEYS)
        assert r["block_overlap"] == []

    def test_mixed_tracks_flagged(self):
        from arsi.world_model.capability_flow import flow_fidelity_check

        r = flow_fidelity_check(mixed_track_scores=True, v_is_first_order=False)
        assert r["ok"] is False
        assert r["first_order_only"] is False
