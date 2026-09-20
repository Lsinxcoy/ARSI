"""Runtime fix tests — score wall, PASS pool, beta freeze, policy sanitize, calibrator."""
from __future__ import annotations

from arsi.foundation.quality_gate import TraceQualityGate, QualityVerdict
from arsi.governor.exploration_policy import ExplorationPolicy, _parse_llm_json
from arsi.iwm.calibrate import IntrospectorCalibrator
from arsi.meta.beta_sweep import BetaSweepPoint, BetaSweepResult, sweep_beta
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import ReplayWorld
from arsi.world_model.world_pool import WorldPool
from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn


def _tree_with_success(effect=0.8, n=8):
    tree = DiscoveryTree()
    traces = []
    for i in range(n):
        traces.append({
            "action": "learn" if i % 2 == 0 else "remember",
            "outcome": "success",
            "effect": effect,
            "agent_id": "t",
            "params": {},
        })
    tree.build_from_traces(traces)
    return tree


class TestScoreWall:
    def test_quality_anchored_not_minus_three(self):
        tree = _tree_with_success(0.8, 12)
        world = ReplayWorld.from_discovery_tree(tree, world_id="w1", max_parallelism=3)
        world.score_mode = "quality_anchored"
        def pol(obs, legal, mp):
            return legal[:2]
        r = world.replay(pol, policy_name="t", max_rounds=10)
        assert r.score_breakdown.get("mode") == "quality_anchored"
        # success traces must produce non-trivial quality
        assert r.quality > 0.2
        # cost may dominate but quality_anchored should not stick at paper pit when quality present
        assert r.replay_score > -3.673 + 0.5 or r.quality >= 0.5

    def test_weak_quality_cost_scaled(self):
        tree = DiscoveryTree()
        traces = [{"action": "x", "outcome": "failure", "effect": 0.0, "params": {}} for _ in range(6)]
        tree.build_from_traces(traces)
        world = ReplayWorld.from_discovery_tree(tree, world_id="w2")
        world.score_mode = "quality_anchored"
        world.min_quality_signal = 0.08
        world.weak_quality_cost_scale = 0.2
        def pol(obs, legal, mp):
            return legal[:3]
        r = world.replay(pol, max_rounds=8)
        assert r.score_breakdown["effective_cost_term"] <= r.score_breakdown["raw_cost_term"] + 1e-6
        if r.quality < 0.08 and r.probes > 0:
            assert r.score_breakdown["effective_cost_term"] < r.score_breakdown["raw_cost_term"]

    def test_paper_mode_uses_raw_cost(self):
        tree = _tree_with_success(0.9, 6)
        world = ReplayWorld.from_discovery_tree(tree, world_id="w3")
        world.score_mode = "paper"
        def pol(obs, legal, mp):
            return legal[:2]
        r = world.replay(pol, max_rounds=6)
        assert r.score_breakdown["mode"] == "paper"
        assert abs(r.score_breakdown["effective_cost_term"] - r.score_breakdown["raw_cost_term"]) < 1e-9


class TestQualityGatePASS:
    def test_success_moderate_effect_is_pass(self):
        g = TraceQualityGate(llm=None)
        r = g.evaluate({"action": "learn", "outcome": "success", "effect": 0.6})
        assert r["accepted"]
        assert r["gate2"]["goal_achievement"] == QualityVerdict.PASS.value

    def test_failure_is_fail(self):
        g = TraceQualityGate(llm=None)
        r = g.evaluate({"action": "learn", "outcome": "failure", "effect": 0.2})
        assert r["gate2"]["goal_achievement"] == QualityVerdict.FAIL.value

    def test_filter_pass_first_quota(self):
        from arsi.core import ARSI
        from arsi.foundation.store import MnemosyneStore
        from arsi.mnemosyne.core import Mnemosyne
        from arsi.world_model.siwm import SIWM
        from arsi.foundation.iron_laws import IronLaws
        from arsi.governor.core import AutopoieticGovernor
        from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
        from arsi.pipelines.dream import DreamPipeline
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp())
        store = MnemosyneStore(str(tmp / "d.db"))
        mn = Mnemosyne(store)
        siwm = SIWM(store)
        lp = tmp / "laws.yaml"
        lp.write_text("laws: []\n", encoding="utf-8")
        laws = IronLaws(lp)
        arsi = ARSI(
            store=store, mnemosyne=mn, siwm=siwm,
            governor=AutopoieticGovernor(siwm, store, laws),
            empowerment=EmpowermentEngine(mn, siwm, NullAdapter()),
            dream=DreamPipeline(siwm, mn), iron_laws=laws, llm=None,
        )
        arsi._world_min_verdict = "PASS"
        traces = []
        for i in range(20):
            traces.append({"action": "learn", "outcome": "success", "effect": 0.8, "params": {}})
        for i in range(30):
            traces.append({"action": "remember", "outcome": "recorded", "effect": 0.2, "params": {}})
        kept, stats = arsi._filter_traces_for_world(traces)
        assert stats["pass"] >= 20
        assert stats["warn"] >= 1
        # PASS admitted first
        assert all(t.get("params", {}).get("gate_verdict") == "PASS" for t in kept[:20])
        if stats.get("warn_quota", 0) >= 0 and len(kept) > 20:
            assert stats["warn_admitted"] <= stats["warn_quota"]


class TestBetaFreeze:
    def test_degenerate_freezes_prev_beta(self):
        class FakePool:
            size = 2
            worlds = []
            def evaluate_policy_across_pool(self, fn, policy_name="p", max_rounds=5):
                return {
                    "available": True,
                    "avg_score": -3.673,
                    "avg_quality": 0.032,
                    "avg_probes": 38.0,
                    "per_world": [
                        {"probes": 38, "rounds": 20, "score": -3.673, "quality": 0.032},
                        {"probes": 38, "rounds": 20, "score": -3.673, "quality": 0.032},
                    ],
                }

        def factory(b):
            return lambda obs, legal, mp: list(legal)[:2]

        res = sweep_beta(FakePool(), factory, live_history=[{"best_score": -3.673, "beta": 0.65}])
        assert res.non_degenerate is False
        assert abs(res.selected_default_beta - 0.65) < 1e-6
        assert res.reason.startswith("degenerate_freeze_beta_")


class TestPolicySanitize:
    def test_parse_chinese_period(self):
        code = "Here is code:\n```python\ndef select_nodes(eligible, observed, tree_stats):\n    return eligible[:2]。\n```\n"
        parsed = _parse_llm_json(code)
        assert parsed is not None
        assert "def select_nodes" in parsed
        assert "。" not in parsed
        p = ExplorationPolicy(name="t", code=parsed)
        p._compile()
        out = p.select(["a", "b", "c"], set())
        assert out

    def test_reject_without_select_nodes(self):
        assert _parse_llm_json("hello") is None


class TestCalibratorStricter:
    def test_noop_not_success(self):
        cal = IntrospectorCalibrator(min_samples=2)
        cal.record("d1", used_iwm=True, success=True, note="heuristic|waiting_new_traces")
        cal.record("d2", used_iwm=True, success=True, note="learn_no_distill")
        assert cal.iwm_success_rate == 0.0

    def test_trust_capped_without_baseline(self):
        cal = IntrospectorCalibrator(min_samples=3)
        for i in range(6):
            cal.record(f"i{i}", used_iwm=True, success=True, note="ok_distilled")
        assert cal.self_trust() <= 0.75
        assert cal.report()["trust_capped_without_baseline"] is True

    def test_baseline_arm_raises_trust(self):
        cal = IntrospectorCalibrator(min_samples=3)
        for i in range(6):
            cal.record(f"i{i}", used_iwm=True, success=True, note="good")
        for i in range(6):
            cal.record(f"b{i}", used_iwm=False, success=True, note="good")
        assert cal.self_trust() > 0.75


class TestDiscoveryQuality:
    def test_success_scores_above_zero(self):
        tree = _tree_with_success(0.7, 5)
        scores = [n.score for n in tree.nodes.values() if n.id != "root" and n.outcome == "success"]
        assert scores and max(scores) >= 0.2
