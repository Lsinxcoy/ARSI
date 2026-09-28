"""Statistical OPF discipline tests (JEPA-Anything J0–J2)."""
from __future__ import annotations


def _rows(n=12):
    rows = []
    for i in range(n):
        rows.append(
            {
                "self_trust": 0.1 * i,
                "memory_trust": 0.05 * ((i * 3) % 7),
                "behavior_predictor_trust": 0.02 * i,
                "eta": 0.01 * i,
                "live_last": 0.8 + 0.01 * i,
                "live_ema": 0.79 + 0.01 * i,
                "pool_last": -0.3 + 0.02 * ((i * 5) % 4),
                "pool_ema": -0.31,
                "d_t_mean": 13.0 + 0.01 * i,
                "d_t_spread": 0.2,
                "el_pass_mean": 0.01 * ((i * 2) % 5),
            }
        )
    return rows


class TestOPFDiscipline:
    def test_j0_orthogonality(self):
        from arsi.world_model.opf_discipline import cross_block_orthogonality

        r = cross_block_orthogonality(_rows())
        assert r["n_rows"] == 12
        assert "organ~live" in r["pairs"] or "live~organ" in r["pairs"]
        assert 0.0 <= r["orth_max_abs"] <= 1.0

    def test_j1_dead_block(self):
        from arsi.world_model.opf_discipline import factor_activity

        rows = _rows()
        for r in rows:
            r["pool_last"] = 0.5
            r["pool_ema"] = 0.5
        r = factor_activity(rows)
        assert "pool" in r["dead_blocks"]
        assert r["ok"] is False

    def test_j2_intervention_ranking(self):
        from arsi.world_model.opf_discipline import intervention_response

        before = {"live_last": 0.5, "live_ema": 0.5, "d_t_mean": 13.0, "eta": 0.1}
        after = {"live_last": 0.9, "live_ema": 0.8, "d_t_mean": 13.0, "eta": 0.11}
        r = intervention_response(before, after)
        assert r["most_pushed"] == "live"
        assert r["ranking"][0] == "live"

    def test_bundle(self):
        from arsi.world_model.opf_discipline import opf_discipline

        r = opf_discipline(_rows(), {"live_last": 0.1}, {"live_last": 0.9})
        assert r.intervention_response.get("most_pushed") == "live"
        assert r.note.startswith("statistical_opf")
