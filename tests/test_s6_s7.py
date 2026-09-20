"""S6 paired A/B + S7 VacuumGate tests."""
from __future__ import annotations

from arsi.foundation.vacuum import VacuumGate, consolidation_vacuum
from arsi.meta.paired_ab import cohen_d_paired, compare_paired, select_policy_paired
from arsi.world_model.world_pool import WorldPool
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn
from arsi.meta.eval_loop import fixed_exploration_fn


class TestPairedAB:
    def test_negligible_holds(self):
        base = [0.1, 0.11, 0.09, 0.1, 0.12, 0.1]
        cand = [0.101, 0.109, 0.091, 0.102, 0.118, 0.101]
        cmp = compare_paired(base, cand)
        assert cmp.promote is False
        assert cmp.negligible is True
        assert "hold" in cmp.reason or "negligible" in cmp.reason

    def test_clear_effect_promotes(self):
        base = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        cand = [0.5, 0.55, 0.48, 0.52, 0.51, 0.53]
        cmp = compare_paired(base, cand)
        assert cmp.mean_delta > 0.3
        assert cmp.cohen_d >= 0.2
        assert cmp.promote is True

    def test_insufficient_samples_hold(self):
        cmp = compare_paired([0.1, 0.2], [0.9, 0.95], min_samples=5)
        assert cmp.promote is False
        assert "insufficient" in cmp.reason

    def test_cohen_d_zero_variance(self):
        assert cohen_d_paired([0.0, 0.0, 0.0]) == 0.0
        assert cohen_d_paired([0.2, 0.2, 0.2]) > 5

    def test_select_policy_paired_on_pool(self, tmp_path):
        pool = WorldPool()
        for i in range(6):
            traces = [
                {"action": "learn", "outcome": "success", "effect": 0.8, "params": {}},
                {"action": "remember", "outcome": "success", "effect": 0.7, "params": {}},
            ]
            pool.append_from_traces(traces, world_id=f"W{i}")
        current = fixed_exploration_fn(max_workers=2)
        challenge = build_policy_fn(PortfolioPolicy(beta=0.9, max_workers=3))
        out = select_policy_paired(
            pool, current, [challenge],
            current_name="fixed", candidate_names=["dream"],
            min_samples=3,
        )
        assert "paired_comparisons" in out
        assert out["gate"]["rule"].startswith("paired_delta")
        # champion always eligible
        assert out["best_name"] in ("fixed", "dream")


class TestVacuumGate:
    def test_allowed_ops(self):
        v = VacuumGate()
        assert v.check_operation("summarize_existing").allowed
        assert v.check_operation("extract").allowed
        assert v.check_operation("reconcile_existing").allowed
        assert v.check_operation("validate").allowed

    def test_forbidden_ops(self):
        v = VacuumGate()
        assert not v.check_operation("generate").allowed
        assert not v.check_operation("invent").allowed
        assert not v.check_operation("hypothesize").allowed
        assert not v.check_operation("unknown_op").allowed  # conservative

    def test_prompt_blocks_invention_language(self):
        v = VacuumGate()
        ok, reason = v.filter_llm_payload("请编造一些新的事实并输出", "")
        assert ok is False

    def test_reconcile_prompt_allowed(self):
        v = VacuumGate()
        ok, reason = v.filter_llm_payload(
            "调和 AI 系统的新旧信念。\n输出 JSON：\n{\"reconciled\": []}",
            "只输出 JSON",
        )
        assert ok is True

    def test_disabled_gate_allows(self):
        v = VacuumGate(enabled=False)
        assert v.check_operation("generate").allowed

    def test_singleton_stats(self):
        g = consolidation_vacuum()
        g.check_operation("extract")
        s = g.stats
        assert s["checked"] >= 1
        assert "summarize_existing" in s["allowed_ops"]


class TestWorldPoolPairedDefault:
    def test_select_uses_paired_when_enabled(self):
        pool = WorldPool()
        pool.append_from_traces(
            [{"action": "learn", "outcome": "success", "effect": 0.9, "params": {}}] * 3,
            world_id="A",
        )
        out = pool.select_best_policy(
            fixed_exploration_fn(2),
            [build_policy_fn(PortfolioPolicy(beta=0.5))],
            current_name="cur",
            candidate_names=["c1"],
        )
        assert "paired_comparisons" in out or out.get("gate", {}).get("rule") == "max_avg_score_fallback"
