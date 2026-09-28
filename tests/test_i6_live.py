"""I6 live gate tests — claim ladder + evidence discipline."""
from __future__ import annotations

from arsi.foundation.verified import unverified, verified_from_exit_code
from arsi.iwm.i6_live import (
    DEFAULT_THRESHOLDS,
    I6LiveFacts,
    evaluate_i6_live,
    facts_from_checkpoint,
    facts_from_health_rows,
    merge_facts,
)


def _thick_ok_facts() -> I6LiveFacts:
    return I6LiveFacts(
        q_gate_ok=True,
        q_failed=[],
        q_pass={
            "Q1_organs": True,
            "Q2_frontier": True,
            "Q3_loop": True,
            "Q4_ledger": True,
            "Q5_provenance": True,
            "Q6_calibrate": True,
        },
        world_pool_size=6,
        trace_count=20000,
        experience_count=8000,
        manifest_cycles=50,
        n_v_updates=20,
        layer1_holdout=0.9,
        organ_trust={"memory": 0.8, "behavior_predictor": 0.7},
        organ_measured_frac=0.8,
        calibration_n=12,
        self_trust=0.7,
        degrade_to_baseline=False,
        paired_promote=True,
        live_regression=False,
        health_rows=30,
        host_success_rate=0.9,
        suite_verified=verified_from_exit_code("i6", 0, command="pytest").to_dict(),
    )


class TestClaimLadder:
    def test_skeleton_when_q1_q3_two_fail(self):
        f = _thick_ok_facts()
        f.q_pass = {"Q1_organs": False, "Q2_frontier": False, "Q3_loop": True}
        r = evaluate_i6_live(f)
        assert r.claim == "introspective_skeleton"
        assert r.live_v1 is False

    def test_q_not_run_is_candidate_not_skeleton(self):
        f = _thick_ok_facts()
        f.q_pass = {}
        f.q_gate_ok = False
        r = evaluate_i6_live(f)
        assert r.claim == "introspective_v1_candidate"
        assert r.gates["q_evaluated"] is False

    def test_candidate_when_thin(self):
        f = _thick_ok_facts()
        f.world_pool_size = 2
        f.n_v_updates = 1
        f.organ_measured_frac = 0.1
        r = evaluate_i6_live(f)
        assert r.claim == "introspective_v1_candidate"
        assert r.live_v1 is False
        assert "pool_thick" in r.failed or r.gates["pool_thick"] is False

    def test_live_v1_when_all_gates(self):
        f = _thick_ok_facts()
        r = evaluate_i6_live(f)
        assert r.claim == "introspective_v1_live"
        assert r.live_v1 is True
        assert r.verified.get("verified") is True

    def test_live_v1_downgrades_without_verified_suite(self):
        f = _thick_ok_facts()
        f.suite_verified = unverified("i6", "skipped").to_dict()
        r = evaluate_i6_live(f)
        assert r.live_v1 is False
        assert r.claim == "introspective_v1_candidate"
        assert "suite_not_verified" in r.failed or "suite_verified" in r.failed

    def test_regression_blocks_live(self):
        f = _thick_ok_facts()
        f.live_regression = True
        f.paired_promote = False
        r = evaluate_i6_live(f)
        assert r.live_v1 is False
        assert r.gates["no_live_regression"] is False


class TestFactsSources:
    def test_from_health_rows(self):
        rows = [
            {
                "world_pool_size": 3,
                "trace_count": 100,
                "manifest_cycles": 5,
                "capability_flow": {"n_v_updates": 2},
                "layer1": {"holdout_accuracy": 0.8},
                "iwm": {
                    "self_trust": 0.6,
                    "organ_trust": {"memory": 0.7, "dream": 0.0},
                    "calibration": {"n": 4},
                },
                "host_loop": {"hosts": {"a": {"success": True}, "b": {"success": True}}},
            }
        ] * 12
        f = facts_from_health_rows(rows)
        assert f.world_pool_size == 3
        assert f.health_rows == 12
        assert f.n_v_updates == 2
        assert f.layer1_holdout == 0.8
        assert f.organ_measured_frac == 0.5
        assert f.host_success_rate == 1.0

    def test_from_checkpoint(self):
        ckpt = {
            "tick": 9,
            "stats": {
                "world_pool_size": 5,
                "trace_count": 50,
                "capability_flow": {"n_v_updates": 3},
                "iwm": {"self_trust": 0.5, "organ_trust": {"memory": 1.0}},
            },
        }
        f = facts_from_checkpoint(ckpt)
        assert f.world_pool_size == 5
        assert f.trace_count == 50

    def test_empty_health(self):
        f = facts_from_health_rows([])
        assert f.health_rows == 0
        assert "no_health_rows" in f.note

    def test_merge_keeps_max(self):
        a = I6LiveFacts(world_pool_size=2, trace_count=10, live_regression=True)
        b = I6LiveFacts(world_pool_size=7, n_v_updates=4, paired_promote=True)
        m = merge_facts(a, b)
        assert m.world_pool_size == 7
        assert m.n_v_updates == 4
        assert m.live_regression is True
        assert m.paired_promote is True


class TestThresholds:
    def test_default_thresholds_shape(self):
        assert "min_world_pool" in DEFAULT_THRESHOLDS
        assert "min_n_v_updates" in DEFAULT_THRESHOLDS
        assert "min_organ_measured_frac" in DEFAULT_THRESHOLDS
        assert "min_calibration_samples" in DEFAULT_THRESHOLDS
        assert DEFAULT_THRESHOLDS["min_world_pool"] >= 5
