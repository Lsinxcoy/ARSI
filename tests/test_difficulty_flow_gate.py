"""P2-9 — difficulty × flow dual gate + same-generation pairing."""
from __future__ import annotations

from arsi.meta.difficulty_flow_gate import (
    apply_dual_gate_to_selection,
    d_t_is_rising,
    difficulty_flow_gate,
    extract_d_t_history,
    flow_progress_score,
    same_generation_pairs,
)
from arsi.meta.paired_ab import compare_paired, select_policy_paired


class TestFlowProgress:
    def test_toward_goal_positive(self):
        prog = flow_progress_score(
            z_now={"memory_trust": 0.2, "self_trust": 0.5},
            z_subgoal={"memory_trust": 0.8, "self_trust": 0.5},
            v_hat={"memory_trust": 0.05, "self_trust": 0.0},
        )
        assert prog["score"] > 0
        assert prog["positive_velocity"] is True
        assert prog["aligned"] >= 1

    def test_away_from_goal_negative(self):
        prog = flow_progress_score(
            z_now={"memory_trust": 0.2},
            z_subgoal={"memory_trust": 0.8},
            v_hat={"memory_trust": -0.1},
        )
        assert prog["score"] < 0
        assert prog["positive_velocity"] is False
        assert prog["anti"] == 1

    def test_empty(self):
        prog = flow_progress_score({}, {}, {})
        assert prog["score"] == 0.0
        assert prog["n_keys"] == 0


class TestDTRising:
    def test_rising(self):
        rising, reason = d_t_is_rising([1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
        assert rising is True
        assert "rising" in reason

    def test_stable(self):
        rising, _ = d_t_is_rising([2.0, 2.0, 2.0, 2.0])
        assert rising is False

    def test_insufficient(self):
        rising, reason = d_t_is_rising([1.0])
        assert rising is False
        assert "insufficient" in reason


class TestDualGate:
    def test_block_when_harder_without_progress(self):
        gate = difficulty_flow_gate(
            [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            {
                "z_now": {"memory_trust": 0.2},
                "z_subgoal": {"memory_trust": 0.8},
                "v_hat": {"memory_trust": -0.05},
            },
        )
        assert gate["allow_promote"] is False
        assert gate["d_t_rising"] is True
        assert "block_harder_env_without_z_progress" in gate["reason"]

    def test_pass_when_harder_with_progress(self):
        gate = difficulty_flow_gate(
            [1.0, 1.0, 1.0, 3.0, 3.0, 3.0],
            {
                "z_now": {"memory_trust": 0.2},
                "z_subgoal": {"memory_trust": 0.8},
                "v_hat": {"memory_trust": 0.05},
            },
        )
        assert gate["allow_promote"] is True
        assert "pass_harder_env_with_positive_z_progress" in gate["reason"]

    def test_hold_when_harder_no_flow_evidence(self):
        gate = difficulty_flow_gate([1.0, 1.0, 1.0, 3.0, 3.0, 3.0], {})
        assert gate["allow_promote"] is False
        assert "no_flow_evidence" in gate["reason"]

    def test_pass_when_not_rising(self):
        gate = difficulty_flow_gate([2.0, 2.0, 2.0, 2.0], {})
        assert gate["allow_promote"] is True
        assert "d_t_not_rising" in gate["reason"]


class TestSameGenerationPairs:
    def test_filter_by_generation(self):
        rows = [
            {"score": 1.0, "lineage": {"generation": 0}},
            {"score": 2.0, "lineage": {"generation": 1}},
            {"score": 3.0, "lineage": {"generation": 1}},
            {"score": 4.0, "lineage": {"generation": 0}},
        ]
        g1 = same_generation_pairs(rows, generation=1)
        assert len(g1) == 2
        assert all(r["lineage"]["generation"] == 1 for r in g1)

    def test_default_majority_generation(self):
        rows = [
            {"score": 1.0, "lineage": {"generation": 0}},
            {"score": 2.0, "lineage": {"generation": 1}},
            {"score": 3.0, "lineage": {"generation": 1}},
        ]
        out = same_generation_pairs(rows)
        assert all(r["lineage"]["generation"] == 1 for r in out)


class TestApplyDualGateToSelection:
    def test_revokes_promote(self):
        selection = {
            "best_name": "challenger",
            "current_name": "champion",
            "promoted": True,
            "paired_comparisons": [{"cand_name": "challenger", "promote": True, "reason": "promote_mean_delta=1"}],
            "all": [{"name": "challenger", "promote": True, "reason": "promote_mean_delta=1"}],
        }

        class _W:
            env_difficulty_dyn = {"d_t": 3.0}
            env_difficulty = {"d_t": 3.0}

        class _Pool:
            worlds = [_W(), _W(), _W(), _W(), _W(), _W()]

        pool = _Pool()
        # make rising d_t: override extract via mixed history — worlds all 3.0 is stable
        # force rising by monkey history
        out = apply_dual_gate_to_selection(
            selection,
            pool,
            flow_guidance={
                "z_now": {"memory_trust": 0.1},
                "z_subgoal": {"memory_trust": 0.9},
                "v_hat": {"memory_trust": -0.1},
            },
        )
        # d_t stable (all 3.0) → allow; so set worlds with rising pattern instead
        class _W2:
            def __init__(self, d):
                self.env_difficulty_dyn = {"d_t": d}
                self.env_difficulty = {"d_t": d}

        class _Pool2:
            worlds = [_W2(1), _W2(1), _W2(1), _W2(4), _W2(4), _W2(4)]

        selection2 = {
            "best_name": "challenger",
            "current_name": "champion",
            "promoted": True,
            "paired_comparisons": [{"cand_name": "challenger", "promote": True, "reason": "promote"}],
            "all": [{"name": "challenger", "promote": True, "reason": "promote"}],
        }
        out = apply_dual_gate_to_selection(
            selection2,
            _Pool2(),
            flow_guidance={
                "z_now": {"memory_trust": 0.1},
                "z_subgoal": {"memory_trust": 0.9},
                "v_hat": {"memory_trust": -0.1},
            },
        )
        assert out["gate_revoked"] is True
        assert out["best_name"] == "champion"
        assert out["promoted"] is False
        assert out["difficulty_flow_gate"]["allow_promote"] is False

    def test_keeps_promote_with_progress(self):
        class _W2:
            def __init__(self, d):
                self.env_difficulty_dyn = {"d_t": d}
                self.env_difficulty = {"d_t": d}

        class _Pool2:
            worlds = [_W2(1), _W2(1), _W2(1), _W2(4), _W2(4), _W2(4)]

        selection = {
            "best_name": "challenger",
            "current_name": "champion",
            "promoted": True,
            "paired_comparisons": [{"cand_name": "challenger", "promote": True, "reason": "promote"}],
            "all": [{"name": "challenger", "promote": True, "reason": "promote"}],
        }
        out = apply_dual_gate_to_selection(
            selection,
            _Pool2(),
            flow_guidance={
                "z_now": {"memory_trust": 0.1},
                "z_subgoal": {"memory_trust": 0.9},
                "v_hat": {"memory_trust": 0.2},
            },
        )
        assert out.get("gate_revoked") is not True
        assert out["best_name"] == "challenger"
        assert out["promoted"] is True


class TestCompareStillWorks:
    def test_compare_paired_unchanged(self):
        cmp = compare_paired(
            [0.1, 0.1, 0.1, 0.1, 0.1],
            [0.3, 0.3, 0.3, 0.3, 0.3],
            "a",
            "b",
        )
        assert cmp.promote is True or "promote" in cmp.reason or cmp.mean_delta > 0
