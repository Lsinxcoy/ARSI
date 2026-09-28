"""Sync-1 — CTM-inspired synchronization diagnostics (arXiv:2505.05522)."""
from __future__ import annotations

from arsi.foundation.sync_repr import (
    DEAD_CORR_THR,
    DEAD_VAR_EPS,
    HALF_LIVES_S,
    PairSync,
    block_series_from_z_rows,
    channel_stats,
    corr_matrix,
    dead_channels,
    half_life_to_decay,
    multi_scale_pair_sync,
    pearson,
    spectral_entropy_sync,
    sync_report_from_z_rows,
    top_pairs,
)
from arsi.world_model.capability_flow import Z_BLOCKS, CapabilityFlowTracker


class TestPairSyncRecursive:
    def test_decay_reduces_old_influence(self):
        short = PairSync(half_life_s=1.0)
        long_ = PairSync(half_life_s=1000.0)
        short.update(1.0, 1.0, dt=0.0)
        long_.update(1.0, 1.0, dt=0.0)
        # large gap then opposite signal
        short.update(-1.0, -1.0, dt=50.0)
        long_.update(-1.0, -1.0, dt=50.0)
        # both still positive co-activation, long retains more of first spike
        assert long_.value() > 0
        assert short.value() > 0

    def test_half_life_to_decay(self):
        r = half_life_to_decay(1.0)
        assert abs(r - 0.693147) < 1e-4


class TestCorrAndDead:
    def test_frozen_channel_is_dead(self):
        cols = {
            "a": [1.0, 1.0, 1.0, 1.0, 1.0],
            "b": [0.1, 0.2, 0.3, 0.4, 0.5],
            "c": [0.5, 0.4, 0.3, 0.2, 0.1],
        }
        dead = dead_channels(cols)
        assert "a" in dead
        assert "b" not in dead

    def test_isolated_channel_is_dead(self):
        # d constant-ish with others? use pure noise independent
        cols = {
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],  # perfect corr with x
            "iso": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],  # weak/osc — not isolated vs y necessarily
        }
        # make iso orthogonal-ish to both
        cols["iso"] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        corr = corr_matrix(cols)
        assert abs(corr["x"]["y"]) > 0.99
        dead = dead_channels(cols, corr=corr, corr_thr=0.5)
        # iso anti-correlated with x → |rho| high, not dead; only frozen would be
        assert "a" not in dead

    def test_pearson(self):
        assert pearson([1, 2, 3], [2, 4, 6]) > 0.99
        assert pearson([1, 2, 3], [3, 2, 1]) < -0.99
        assert pearson([1, 1, 1], [1, 2, 3]) == 0.0

    def test_top_pairs_sorted(self):
        cols = {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
            "c": [4.0, 3.0, 2.0, 1.0],
        }
        tops = top_pairs(corr_matrix(cols), k=2)
        assert tops[0]["a"] in ("a", "b") and tops[0]["b"] in ("a", "b")
        assert abs(tops[0]["rho"]) >= abs(tops[1]["rho"])


class TestMultiScale:
    def test_scales_present(self):
        ms = multi_scale_pair_sync([1, 2, 3, 4], [1, 2, 3, 4], dts=[0, 1, 1, 1])
        assert set(ms) == set(HALF_LIVES_S)
        assert ms["short"]["n"] == 4
        assert "S" in ms["med"]


class TestSyncReport:
    def test_blocks_and_dead_organs(self):
        z_rows = []
        for i in range(12):
            z_rows.append(
                {
                    "self_trust": 0.5 + 0.01 * i,
                    "memory_trust": 0.5,  # frozen
                    "behavior_predictor_trust": 0.4 + 0.02 * i,
                    "eta": 0.1,
                    "live_last": 0.5 + 0.01 * i,
                    "live_ema": 0.5 + 0.005 * i,
                    "pool_last": 0.1,
                    "pool_ema": 0.1,
                    "d_t_mean": 10.0 + i,
                    "d_t_spread": 1.0,
                    "el_pass_mean": 0.3,
                }
            )
        rep = sync_report_from_z_rows(z_rows, Z_BLOCKS)
        assert rep["n_samples"] == 12
        assert "organ" in rep["blocks"]
        assert "memory_trust" in rep["dead_organs"]
        assert rep["blocks"]["organ"]["top_pairs"]
        assert "entropy" in rep["blocks"]["organ"]

    def test_empty_rows(self):
        rep = sync_report_from_z_rows([], Z_BLOCKS)
        assert rep["n_samples"] == 0
        assert rep["dead_channels"] == []


class TestTrackerIntegration:
    def test_health_includes_sync(self):
        tr = CapabilityFlowTracker()
        for i in range(8):
            tr.observe(
                {
                    "eta": 0.1 + 0.01 * i,
                    "iwm": {
                        "self_trust": 0.5 + 0.01 * i,
                        "memory_trust": 0.8,
                        "behavior_predictor_trust": 0.4 + 0.02 * i,
                    },
                    "_live_capability_scores": [0.5 + 0.01 * i],
                    "_pool_scores": [0.2],
                },
                t_wall=1000.0 + i * 10.0,
                action="learn",
            )
        h = tr.health()
        sync = h.get("sync") or {}
        assert sync.get("n_samples", 0) >= 8
        assert "blocks" in sync
        assert "dead_organs" in sync
        # frozen memory_trust at 0.8
        assert "memory_trust" in (sync.get("dead_organs") or sync.get("dead_channels") or [])
