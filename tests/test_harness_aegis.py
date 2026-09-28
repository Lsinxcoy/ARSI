"""Harness-1+X — taxonomy, manifest, AEGIS digester/planner/gates, variants."""
from __future__ import annotations

import tempfile
from pathlib import Path

from arsi.harness import (
    AuditLog,
    ChangeManifest,
    Digester,
    GateResult,
    ManifestStore,
    Planner,
    VariantPool,
    critic_check,
    dimension_of_symbol,
    is_edit_allowed,
    regression_check,
    seesaw_check,
)
from arsi.harness.taxonomy import DIMENSIONS, FROZEN_DIMS


class TestTaxonomy:
    def test_nine_dims(self):
        assert len(DIMENSIONS) == 9
        assert "c7" in FROZEN_DIMS

    def test_symbol_map(self):
        assert dimension_of_symbol("iron_laws.enforce") == "c7"
        assert dimension_of_symbol("memory.ingest") == "c3"
        assert dimension_of_symbol("empowerment.apply") == "c4"

    def test_scope_fence_frozen(self):
        ok, reason = is_edit_allowed("c7", "agent_loop", "iron_laws")
        assert ok is False
        assert "frozen" in reason


class TestManifestAndAudit:
    def test_manifest_scope_and_store(self, tmp_path):
        m = ChangeManifest(
            symbol="memory.decay",
            edit_type="config",
            summary="tune decay",
            diff="- 0.01\n+ 0.02",
            inverse_op="- 0.02\n+ 0.01",
        )
        ok, _ = m.validate_scope()
        assert ok is True
        assert m.dimension == "c3"
        store = ManifestStore(tmp_path)
        store.append(m)
        assert store.list()
        assert store.update_status(m.manifest_id, "rejected", "test")
        assert store.list(status="rejected")

    def test_frozen_manifest_rejected(self):
        m = ChangeManifest(symbol="iron_laws.G1", edit_type="control", summary="weaken")
        ok, reason = m.validate_scope()
        assert ok is False

    def test_audit(self, tmp_path):
        log = AuditLog(tmp_path)
        log.emit("critic", "ship", gate="critic", ok=True)
        assert log.tail(1)[0]["stage"] == "critic"


class TestDigesterPlanner:
    def test_digester_clusters(self):
        traces = [
            {"outcome": "failure", "fail_class": "tool_loop", "action": "tool"},
            {"outcome": "failure", "fail_class": "timeout", "action": "learn"},
            {"outcome": "success", "action": "learn"},
            {"outcome": "failure", "error": "timeout budget", "action": "dream"},
        ]
        clusters = Digester().digest(traces)
        labels = {c.label for c in clusters}
        assert clusters
        assert "timeout_or_budget" in labels or "tool_loop" in labels
        assert all(c.n >= 1 for c in clusters)

    def test_planner_untried_levers(self):
        dig = Digester()
        traces = [{"outcome": "failure", "fail_class": "invalid", "action": "empower"}]
        clusters = dig.digest(traces)
        land = Planner().landscape(clusters, prior_manifests=[])
        assert land.untried_edit_types
        assert "prompt" in land.untried_edit_types or "tool" in land.untried_edit_types
        assert land.recommendations

    def test_planner_tracks_tried(self):
        land = Planner().landscape(
            [],
            prior_manifests=[{"edit_type": "prompt", "dimension": "c2"}],
        )
        assert "prompt" not in land.untried_edit_types
        assert "c2" not in land.untried_dims


class TestGates:
    def test_critic_blocks_answers(self):
        r = critic_check("insert gold_ answer_key into prompt")
        assert r.ok is False

    def test_critic_ok(self):
        r = critic_check("add retry with backoff", summary="tool robustness")
        assert r.ok is True

    def test_regression(self):
        r = regression_check({"pass": 0.8, "cost_lower_better": 10}, {"pass": 0.7, "cost_lower_better": 8})
        assert r.ok is False
        r2 = regression_check({"pass": 0.8}, {"pass": 0.9})
        assert r2.ok is True

    def test_seesaw_fork_signal(self):
        r = seesaw_check({"a": 0.5, "b": 0.5}, {"a": 0.8, "b": 0.2})
        assert r.ok is False
        assert "fork" in r.reason


class TestVariants:
    def test_route_and_fork(self):
        pool = VariantPool(max_variants=3)
        assert pool.route("gaia") == "default"
        sw = seesaw_check({"a": 0.5, "b": 0.5}, {"a": 0.9, "b": 0.1})
        child = pool.fork_on_seesaw(sw, parent_id="default")
        assert child.variant_id in pool.variants
        pool.observe(child.variant_id, "a", True)
        assert pool.route("a") == child.variant_id
        rep = pool.report()
        assert "iron_laws_shared" in rep["note"]


class TestHostVariants:
    def test_bind_three_hosts(self):
        pool = VariantPool(max_variants=5)
        created = pool.ensure_hosts(["hermes", "mimo-desktop", "synthex-mothernest"])
        assert len(created) == 3
        assert pool.route_for_host("hermes") == "host:hermes"
        assert pool.route_for_host("mimo-desktop") == "host:mimo-desktop"
        assert pool.route_for_host("unknown") == "default"

    def test_observe_host_updates(self):
        pool = VariantPool(max_variants=5)
        pool.ensure_hosts(["hermes"])
        pool.observe_host("hermes", True)
        pool.observe_host("hermes", True)
        v = pool.variants["host:hermes"]
        assert v.cluster_success["default"] > 0.5

    def test_fork_still_works(self):
        pool = VariantPool(max_variants=4)
        pool.ensure_hosts(["hermes", "mimo"])
        sw = seesaw_check({"a": 0.4, "b": 0.4}, {"a": 0.9, "b": 0.1})
        child = pool.fork_on_seesaw(sw)
        assert child.variant_id in pool.variants


class TestC9TrainingBridge:
    def test_export_and_blacklist(self, tmp_path):
        from arsi.harness.training_bridge import export_training_set, trace_to_example

        traces = [
            {"agent_id": "hermes", "action": "learn", "outcome": "success", "effect": 0.5, "task_id": "t1"},
            {"agent_id": "synthex", "action": "compile", "outcome": "failure", "effect": 0.1,
             "task_id": "sealed_task_01"},
            {"agent_id": "mimo", "action": "dream", "outcome": "failure", "effect": 0.0, "task_id": "t2"},
        ]
        man = export_training_set(traces, tmp_path / "c9.jsonl", harness_variant="host:hermes")
        assert man["n_exported"] == 2
        assert man["n_excluded"] == 1
        assert man["grpo"] is False
        lines = (tmp_path / "c9.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert "sealed" not in (tmp_path / "c9.jsonl").read_text(encoding="utf-8")

    def test_task_level_fields(self):
        from arsi.harness.training_bridge import trace_to_example

        ex = trace_to_example({"agent_id": "h", "action": "tool", "outcome": "failure", "fail_class": "tool_loop"})
        assert ex.success is False
        assert ex.reward == 0.0
        assert ex.trajectory_digest
        assert ex.prompt_facts.get("fail_class") == "tool_loop"


class TestP0RRSI:
    def test_edit_budget_anneals(self):
        from arsi.harness.edit_budget import edit_budget

        assert edit_budget(0, 10, 1, 4) == 4
        assert edit_budget(10, 10, 1, 4) == 1
        assert edit_budget(5, 10, 1, 4) <= 4

    def test_delta_and_floor(self):
        from arsi.harness.noise_floor import calibrate_delta, passes_floor

        d = calibrate_delta([0.80, 0.84, 0.82, 0.81])
        assert d > 0
        # S*−δ = 0.84 − 0.02 = 0.82
        assert passes_floor(0.83, 0.84, d) is True
        assert passes_floor(0.70, 0.90, 0.02) is False

    def test_credit_ledger_and_prune(self):
        from arsi.harness.credit import CreditLedger

        led = CreditLedger(prune_window=3)
        led.log_candidate(1, "c1", [{"component": "prompt", "hypothesis": "h1", "diff": "x"}], dS=0.05, dC=0.0, accepted=True)
        led.log_candidate(2, "c2", [{"component": "skill", "hypothesis": "h2", "diff": "y"}], dS=0.0, dC=0.0, accepted=False)
        assert led.accepted_count("prompt") == 1
        assert "h2" in led.rejected_hypotheses()
        # skill recent gain 0 → prune target
        assert "skill" in led.prune_targets()

    def test_admit_cost_rule(self):
        from arsi.harness.accept import admit

        ok = admit(
            diff="add retry",
            summary="robustness",
            score_hat=0.85,
            s_star=0.84,
            delta=0.01,
            dS=0.05,
            dC=0.50,
            components=["prompt"],
            accepted_counts={},
            beta0=0.1,
            beta1=1.0,
        )
        assert ok.admissible is False
        assert ok.reason == "cost_rule"

    def test_admit_floor(self):
        from arsi.harness.accept import admit

        r = admit(
            diff="x",
            score_hat=0.5,
            s_star=0.9,
            delta=0.02,
            dS=-0.4,
            dC=-0.1,
            components=["prompt"],
            accepted_counts={},
        )
        assert r.admissible is False
        assert r.reason == "floor"

    def test_admit_within_band_ws0(self):
        from arsi.harness.accept import admit

        # score bump within noise, cost up → reject when ws=0
        r = admit(
            diff="tiny",
            score_hat=0.841,
            s_star=0.84,
            delta=0.02,
            dS=0.001,
            dC=0.2,
            components=["prompt"],
            accepted_counts={},
            ws=0.0,
            wn=0.1,
        )
        assert r.admissible is False
        assert r.branch == "within_delta"

    def test_exploration_directive(self):
        from arsi.harness.credit import CreditLedger
        from arsi.harness.prune import exploration_directive

        led = CreditLedger()
        led.log_candidate(1, "c", [{"component": "prompt", "hypothesis": "h"}], dS=0.1, dC=0.0, accepted=True)
        d = exploration_directive(led, {"prompt", "skill", "memory"}, 0.8, 0.79, 0.02)
        assert d["stalled"] is True
        assert d["m_draft"] == 1
        assert "skill" in d["unexercised"]


class TestSEVerAContracts:
    def test_g3_freeze_blocks_write(self):
        from arsi.harness.contracts import apply_contracts

        r = apply_contracts({"frozen": True}, {"apply": True, "writes": True})
        g3 = next(x for x in r if x.law_id == "G3")
        assert g3.used_fallback
        assert g3.output.get("apply") is False

    def test_g5_effect_clamped(self):
        from arsi.harness.contracts import apply_contracts

        r = apply_contracts({}, {"effect": 5.0})
        g5 = next(x for x in r if x.law_id == "G5")
        assert g5.output["effect"] == 1.0

    def test_g10_sealed_protected(self):
        from arsi.harness.contracts import apply_contracts

        r = apply_contracts({}, {"write_path": "config/sealed_tasks.yaml", "mutate_sealed": True})
        g10 = next(x for x in r if x.law_id == "G10")
        assert g10.used_fallback
        assert g10.output.get("sealed_protected") is True

    def test_guarded_wrapper(self):
        from arsi.harness.contracts import guarded

        out, res = guarded({"circuit": "OPEN"}, lambda: {"apply": True})
        assert out.get("apply") is False
        assert any(x.used_fallback for x in res)

    def test_g4_consolidate_requires_audit(self):
        from arsi.harness.contracts import apply_contracts

        r = apply_contracts({}, {"kind": "consolidate"})
        g4 = next(x for x in r if x.law_id == "G4")
        assert g4.used_fallback


class TestSAHOO:
    def test_gdi_zero_on_identical(self):
        from arsi.harness.sahoo import goal_drift_index

        r = goal_drift_index(text_a="hello world", text_b="hello world", tokens_a="hello world".split(), tokens_b="hello world".split())
        assert r.gdi < 0.05
        assert r.over_threshold is False

    def test_gdi_high_on_different(self):
        from arsi.harness.sahoo import goal_drift_index

        r = goal_drift_index(text_a="alpha beta gamma delta", text_b="zzz yyy xxx www")
        assert r.gdi > 0.3

    def test_constraint_zero_absolute_stop(self):
        from arsi.harness.sahoo import decide_stop

        v = decide_stop(constraint_preservation=0.0, gdi=0.1, regression=0.0, quality_series=[0.5, 0.6, 0.7])
        assert v.stop is True
        assert v.rule == "constraint_zero"
        assert v.priority == 1

    def test_gdi_stop(self):
        from arsi.harness.sahoo import decide_stop

        v = decide_stop(constraint_preservation=1.0, gdi=0.5, regression=0.0, quality_series=[0.5, 0.6, 0.7])
        assert v.rule == "gdi_threshold"

    def test_car(self):
        from arsi.harness.sahoo import capability_alignment_ratio, regression_risk

        assert capability_alignment_ratio(0.1, 0.05) == 2.0
        r = regression_risk([0.9, 0.5, 0.4, 0.3])
        assert 0.0 <= r <= 1.0


class TestAIDE2Holdout:
    def test_split_blacklist_private(self):
        from arsi.harness.holdout import split_tasks

        s = split_tasks(["a", "b", "c", "d", "e", "sealed_x"], private_ratio=0.3, blacklist=["sealed_x"])
        assert "sealed_x" in s.blacklist or "sealed_x" in s.private
        assert s.selection
        assert not set(s.selection) & {"sealed_x"}

    def test_private_grade_and_reject_public_only(self):
        from arsi.harness.holdout import accept_rewrite, first_order_vs_second_order, private_grade

        assert private_grade([0.8, 0.6]) == 0.7
        r = accept_rewrite(public_gain=0.05, private_gain=-0.01)
        assert r["accepted"] is False
        assert r["reason"] == "private_grade_not_improved"
        assert first_order_vs_second_order(0.05, 0.02) == "transfers"
        assert first_order_vs_second_order(0.05, -0.01) == "selection_only_overfit"

    def test_budget_increase_rejected(self):
        from arsi.harness.holdout import accept_rewrite

        r = accept_rewrite(public_gain=0.1, private_gain=0.05, budget_delta=0.2)
        assert r["accepted"] is False
        assert r["reason"] == "budget_increase"


class TestHackKPI:
    def test_hack_rate(self):
        from arsi.harness.hack_kpi import lineage_hack_trend, reward_hack_rate

        r = reward_hack_rate([0.2, 0.2, 0.0, 0.3], [0.0, 0.1, 0.0, -0.1])
        assert r.n == 4
        assert r.n_hack == 2
        assert lineage_hack_trend([0.55, 0.39, 0.32]) == "declining"
        assert lineage_hack_trend([0.3, 0.5]) == "rising"


class TestSelfHarnessMinimal:
    def test_minimal_and_dual_regression(self):
        from arsi.harness.minimality import dual_regression, minimal_edit_score, prefer_minimal

        assert minimal_edit_score("+ one\n- two\n") < minimal_edit_score("+ a\n+ b\n+ c\n+ d\n+ e\n- f\n")
        assert prefer_minimal("+ x\n", "+ a\n+ b\n+ c\n")
        ok = dual_regression(0.8, 0.85, 0.7, 0.72)
        assert ok.ok is True
        bad = dual_regression(0.8, 0.85, 0.7, 0.6)
        assert bad.ok is False
        assert "held_out_regress" in bad.reason


class TestGraderDrawback:
    def test_metric_names_drawback(self):
        from arsi.harness.drawback import DrawbackOp, MetricExpression

        ops = [
            DrawbackOp("crash", "execution", "crash", lambda t, y, c: None if y is None else (y.get("crash") or False)),
            DrawbackOp("spec", "judge", "spec_mismatch", lambda t, y, c: bool((y or {}).get("wrong_meaning"))),
        ]
        m = MetricExpression(ops, mode="any")
        v = m.evaluate({}, {"crash": False, "wrong_meaning": True})
        assert v.verdict == "fail"
        assert any("spec" in n for n in v.named_drawbacks)

    def test_vacuous_rejected(self):
        from arsi.harness.drawback import MetricVerdict, is_vacuous_metric, metric_fitness, recall_weighted_adreement

        assert is_vacuous_metric([MetricVerdict("pass"), MetricVerdict("pass")])
        # r_fail=0.5, r_pass=1.0 (no pass labels) → (2*0.5+1*1)/3
        assert abs(recall_weighted_adreement(["pass", "fail"], ["fail", "fail"]) - 2 / 3) < 1e-9
        assert metric_fitness(0.8, 0.8, 10, w=0.5, lam=0.01) > 0.5

    def test_birth_and_curate(self):
        from arsi.harness.drawback import DrawbackOp, birth_gate, curate_pool, leave_one_out_marginal

        assert birth_gate(3, 4, 0) is True
        assert birth_gate(1, 4, 0) is False
        assert birth_gate(3, 4, 1) is False
        assert abs(leave_one_out_marginal(0.8, 0.7) - 0.1) < 1e-9
        ops = [
            DrawbackOp("a", "static", "x", lambda t, y, c: False, status="shadow"),
            DrawbackOp("b", "static", "y", lambda t, y, c: False, status="active"),
        ]
        curate_pool(ops, {"a": 0.1, "b": -0.1}, {"a": 0, "b": 0})
        assert ops[0].status == "active"
        assert ops[1].status == "retired"


class TestGAI:
    def test_anchored_rsi(self):
        from arsi.harness.gai import ARSI_GAI, GAIConfig, assert_anchored, rsi_defects

        ok, reason = assert_anchored(ARSI_GAI)
        assert ok is True
        assert ARSI_GAI.is_rsi() is True
        d = rsi_defects(ARSI_GAI)
        # improvement mechanism inside agent is listed, but base stays grounded
        assert "goal_drift_agent_rewrites_standard" not in d.defects

    def test_goal_drift_detected(self):
        from arsi.harness.gai import GAIConfig, assert_anchored, rsi_defects

        bad = GAIConfig(modifier_in_agent=True, base_grounded=True, base_inside_agent=True)
        assert bad.polarity() == "goal_drift"
        ok, _ = assert_anchored(bad)
        assert ok is False
        assert "goal_drift_agent_rewrites_standard" in rsi_defects(bad).defects

    def test_gpi_not_rsi(self):
        from arsi.harness.gai import GAIConfig

        gpi = GAIConfig(modifier_in_agent=False, base_grounded=True, base_inside_agent=False)
        assert gpi.is_gpi() is True
        assert gpi.polarity() == "anchored"


class TestPillars:
    def test_six_pillars_declared(self):
        from arsi.harness.pillars import CHARTER, PILLARS, pillar_summary

        assert len(PILLARS) == 6
        assert CHARTER["role"] == "charter_across_layers"
        assert "L0_constitution" in PILLARS
        assert "L5_epistemology" in PILLARS
        s = pillar_summary()
        assert s["count"] == 6

    def test_pillar_modules_importable(self):
        from arsi.harness.pillars import audit_pillars

        a = audit_pillars()
        # core pillar modules must resolve
        assert a.ok or all("dream_rsi" in m or "portfolio" in m or "world_pool" in m for m in a.missing_modules)


class TestPadfCallGuard:
    def test_secret_redacted(self):
        from arsi.harness.call_guard import guard_call

        r = guard_call("llm", {}, lambda: {"text": "key nvapi-abc123 leaked"})
        assert r.used_fallback is True
        assert r.output.get("redacted") is True

    def test_tool_budget(self):
        from arsi.harness.call_guard import guard_call

        r = guard_call("tool", {"budget_cap": 1.0}, lambda: {"cost": 5.0, "apply": True})
        assert r.output.get("apply") is False
        assert "call_budget" in r.laws or "budget" in str(r.laws)

    def test_ok_passthrough(self):
        from arsi.harness.call_guard import guarded_llm_chat

        r = guarded_llm_chat(lambda p: p + "!", "hi")
        assert r.ok and r.output == "hi!"


class TestPdAnchorHardening:
    def test_soft_to_checkable(self):
        from arsi.harness.anchor_hardening import anchor_set_quality, harden_soft_anchor

        a = harden_soft_anchor("A1", "code crashes with exception when empty", "fail")
        assert a.checkable is True
        assert any(d["detects"] == "crash" for d in a.detectors)
        batch = [a, a, harden_soft_anchor("A3", "wrong meaning in output", "fail"), harden_soft_anchor("A4", "ok", "pass")]
        q = anchor_set_quality(batch)
        assert q["n"] == 4
        assert q["min_viable"] is True

    def test_unmapped_needs_manual(self):
        from arsi.harness.anchor_hardening import harden_soft_anchor

        a = harden_soft_anchor("A2", "vibes are off", "fail")
        assert a.checkable is False


class TestPbStrategyBandit:
    def test_ucb_and_observe(self):
        from arsi.harness.strategy_bandit import StrategyBandit

        b = StrategyBandit(seed=1)
        picks = [b.select() for _ in range(12)]
        assert len(set(picks)) >= 2  # exploration
        for p in picks:
            b.observe(p, 0.5)
        rep = b.report()
        assert rep["total"] == 12
        assert ("conservative" in rep["arms"]) or ("prompt" in rep["arms"])


class TestPfGoalVitals:
    def test_iwm_goal_vitals(self):
        from arsi.iwm import IWM

        i = IWM()
        v = i.observe_goal_vitals(gdi=0.2, quality_gain=0.05, quality_point=0.8)
        assert v["gdi"] == 0.2
        assert v["car"] is not None
        assert v["quality_n"] == 1


class TestPcBoundedContext:
    def test_compress_keeps_recent(self):
        from arsi.harness.bounded_context import ContextBudget, compress_history

        items = [f"old-{i}" for i in range(10)] + ["NEW-A", "NEW-B", "NEW-C"]
        r = compress_history(items, ContextBudget(max_chars=2000, keep_recent=3))
        assert "NEW-A" in r.summary and "NEW-C" in r.summary
        assert r.n_raw == 13
        assert r.n_kept_full == 3
        assert r.overflow_prevented is True

    def test_overflow_prevented(self):
        from arsi.harness.bounded_context import BoundedContextBuffer, ContextBudget

        buf = BoundedContextBuffer(ContextBudget(max_chars=500, keep_recent=2))
        for i in range(30):
            buf.push("x" * 200 + str(i))
        r = buf.render()
        assert r.chars <= 500
        assert len(r.summary) <= 500


class TestPeDetectability:
    def test_order_mechanical_first(self):
        from arsi.harness.detectability import investment_order, recommend_next

        order = investment_order()
        assert order[0] in ("static_syntax", "execution_probe", "mechanical_rule")
        assert order[-1] == "semantic_judge"
        rec = recommend_next({"semantic_judge"})
        assert rec["next"] != "semantic_judge"


class TestPghEveritt:
    def test_no_rewrite_ok(self):
        from arsi.harness.everitt import everitt_guard, guarded_utility_rewrite

        v = everitt_guard(
            utility_rewrite_proposed=False,
            value_anticipates_rewrite=False,
            evaluates_with_current_utility=False,
        )
        assert v.ok is True

    def test_rewrite_blocked_without_anticipation(self):
        from arsi.harness.everitt import everitt_guard, guarded_utility_rewrite

        v = everitt_guard(
            utility_rewrite_proposed=True,
            value_anticipates_rewrite=False,
            evaluates_with_current_utility=True,
        )
        assert v.ok is False
        out, v2 = guarded_utility_rewrite(lambda: "new_util", value_anticipates_rewrite=False)
        assert out is None
        assert v2.ok is False

    def test_rewrite_ok_when_everitt_met(self):
        from arsi.harness.everitt import guarded_utility_rewrite

        out, v = guarded_utility_rewrite(
            lambda: "new_util",
            value_anticipates_rewrite=True,
            evaluates_with_current_utility=True,
        )
        assert out == "new_util"
        assert v.ok is True


class TestPhiIgnition:
    def test_ignited(self):
        from arsi.harness.ignition import ignition_test, run_outer_round

        r = ignition_test(
            incumbent_grade=0.70,
            grade_after_discovered_outer=0.75,
            grade_after_baseline_outer=0.72,
            steps_discovered=50,
            steps_baseline=50,
        )
        assert r.ignited is True
        assert r.outer_gain > 0
        assert r.sample_efficiency_hint is True

    def test_not_ignited_when_degrades(self):
        from arsi.harness.ignition import ignition_test

        r = ignition_test(
            incumbent_grade=0.70,
            grade_after_discovered_outer=0.65,
            grade_after_baseline_outer=0.71,
            steps_discovered=50,
            steps_baseline=40,
        )
        assert r.ignited is False

    def test_outer_round(self):
        from arsi.harness.ignition import run_outer_round

        res = run_outer_round(lambda x: x + 1, 1, lambda x: float(x))
        assert res["accepted"] is True
        assert res["g_after"] == 2.0


class TestPiOvershoot:
    def test_overshoot_flagged(self):
        from arsi.harness.overshoot import detect_overshoot, should_stop_editing

        r = detect_overshoot([0.5, 0.7, 0.8, 0.75, 0.74], patience=2)
        assert r.overshoot is True
        assert r.best_index == 2
        assert r.advice == "rollback_to_best_or_stop"
        stop, advice = should_stop_editing([0.5, 0.7, 0.8, 0.75, 0.74])
        assert stop is True

    def test_no_overshoot_when_improving(self):
        from arsi.harness.overshoot import detect_overshoot

        r = detect_overshoot([0.5, 0.6, 0.7, 0.8])
        assert r.overshoot is False
        assert r.advice == "continue"


class TestEvolver:
    def test_rule_evolve_accepts(self):
        from arsi.harness.evolver import Evolver

        land = {
            "recommendations": [
                {"cluster": "FC-generic_failure-45", "label": "generic_failure", "n": 45, "modules": ["tool_use"]},
                {"cluster": "FC-invalid_or_compile-16", "label": "invalid_or_compile", "n": 16, "modules": ["agent_loop"]},
            ]
        }
        ms = Evolver().propose_from_landscape(land, max_k=3)
        assert len(ms) == 2
        assert all(m.status == "accepted" for m in ms)
        assert ms[0].inverse_op

    def test_evolve_rejects_frozen(self):
        from arsi.harness.evolver import Evolver
        from arsi.harness.manifest import ChangeManifest

        ev = Evolver()
        m = ChangeManifest(symbol="iron_laws.G1", edit_type="control", summary="weaken", diff="x")
        ev._gate(m)
        assert m.status == "rejected"
        assert "frozen" in m.reject_reason

    def test_evolve_rejects_gold(self):
        from arsi.harness.evolver import Evolver
        from arsi.harness.manifest import ChangeManifest

        ev = Evolver()
        m = ChangeManifest(symbol="brief.policy", edit_type="prompt", summary="s", diff="embed gold_answer_key")
        ev._gate(m)
        assert m.status == "rejected"


class TestAppliedManifests:
    def test_code_verifier_compile_gate(self):
        from arsi.sealed_eval.code_verifier import CodeVerifier

        v = CodeVerifier(require_compile=True)
        score, ev = v.verify_python_code(
            "```python\nreturn 1\n```",
            [{"assertion": "True", "description": "t"}],
        )
        assert score == 0.0
        assert ev.get("fail_class") == "compile"

    def test_code_verifier_ok_path(self):
        from arsi.sealed_eval.code_verifier import CodeVerifier

        v = CodeVerifier(require_compile=True)
        code = "def add(a, b):\n    return a + b"
        score, ev = v.verify_python_code(
            code, [{"assertion": "assert add(1, 2) == 3", "description": "t"}]
        )
        assert score == 1.0

    def test_code_verifier_flag_off_skips_compile_gate(self):
        from arsi.sealed_eval.code_verifier import CodeVerifier

        v = CodeVerifier(require_compile=False)
        assert v.require_compile is False

    def test_brief_fail_handling_block(self):
        from arsi.adapters.bidirectional_interface import ARSIBrief

        b = ARSIBrief(
            task_description="t",
            agent_id="hermes",
            recommendations=[],
            relevant_skills=[],
            warnings=[],
            past_lessons=[],
            confidence=0.5,
        )
        text = b.format_for_agent()
        assert "## Fail-handling" in text
        assert "fail_class" in text

    def test_host_strategy_fail_handling(self):
        from arsi.iwm.host_strategy import HostStrategy

        s = HostStrategy(agent_id="hermes")
        block = "\n".join(s.as_structured_block())
        assert "fail_handling" in block


class TestThirdPassInterlock:
    def test_conformance_fallback_rate(self):
        from arsi.harness.conformance import conformance_report

        r = conformance_report([("G5", True), ("G5", True), ("G3", False)], always_fallback_min=0.9)
        assert r.n == 3
        assert r.fallback_rate > 0.5

    def test_domain_guard_and_select(self):
        from arsi.harness.accept import domain_guard, select_among_admissible

        ok, _ = domain_guard(valid_output_rate_before=0.9, valid_output_rate_after=0.89)
        assert ok is True
        bad, reason = domain_guard(valid_output_rate_before=0.9, valid_output_rate_after=0.85)
        assert bad is False and reason == "valid_output_dropped"
        best, s = select_among_admissible(
            [{"admissible": False, "score_hat": 0.99}, {"admissible": True, "score_hat": 0.7}],
            s_star=0.6,
        )
        assert best["score_hat"] == 0.7
        assert s == 0.7

    def test_contractive_and_ceiling(self):
        from arsi.harness.sahoo import capability_ceiling_hit, contractive_regime

        c = contractive_regime(quality_gain_mean=0.1, drift_response=0.05)
        assert c["contractive"] is True
        c2 = contractive_regime(quality_gain_mean=0.1, drift_response=0.5)
        assert c2["contractive"] is False
        assert capability_ceiling_hit([0.5, 0.2, 0.2, 0.1]) is True

    def test_aide85_arms(self):
        from arsi.harness.strategy_bandit import DEFAULT_ARMS, EDIT_TYPE_ARMS

        assert "conservative" in DEFAULT_ARMS
        assert len(DEFAULT_ARMS) == 5
        assert "prompt" in EDIT_TYPE_ARMS


class TestThirdPassFixes:
    def test_shared_calibration(self):
        from arsi.harness.shared_calib import calibrate_shared, small_signal_policy

        cal = calibrate_shared([0.8, 0.84, 0.82], drift_series=[0.1, 0.2, 0.3, 0.4, 0.5], n_anchor_checkable=2)
        assert cal.delta > 0
        assert 0.2 <= cal.gdi_threshold <= 0.8
        pol = small_signal_policy(cal)
        assert "selection_floor" in pol

    def test_dual_sensor_and_goal_drift(self):
        from arsi.harness.interlock import assert_not_goal_drift, dial1_observable, dual_sensor

        r = dual_sensor([0.5, 0.8, 0.7, 0.6], gdi=0.5, gdi_threshold=0.44)
        assert r.goal_drift_suspect is True
        assert r.overshoot is True
        ok, reason = assert_not_goal_drift("sealed", "h1", "h1", gdi=0.6)
        assert ok is False and "goal_drift_suspect" in reason
        ok2, reason2 = assert_not_goal_drift("sealed", "h2", "h1", gdi=0.1)
        assert ok2 is False and "rewritten" in reason2
        d = dial1_observable(["memory_trust"], action="dream")
        assert d["modifier_in_agent"] is True

    def test_reliability_weighted_consensus(self):
        from arsi.harness.drawback import reliability_weighted_consensus

        v = reliability_weighted_consensus(
            [("proven", True, 0.5), ("novice", False, 0.0)],
            kappa=0.5,
        )
        assert v is True
        v2 = reliability_weighted_consensus([("a", None, 0.1)])
        assert v2 is None

    def test_call_guard_audit_contract(self):
        from arsi.harness.call_guard import guard_call

        r = guard_call("brief", {"output_contract": "structured_facts"}, lambda: {"text": "ok"})
        assert r.output["_audit"]["output_contract"] == "structured_facts"


class TestCoEvolution:
    def test_classify_stages(self):
        from arsi.harness.coevolution import classify_coevolution

        assert classify_coevolution(
            n_evolving_units=1, both_units_change=True, mutual_pressure=True
        ).is_coevolution is False
        s1 = classify_coevolution(
            n_evolving_units=2, both_units_change=True, mutual_pressure=True
        )
        assert s1.stage == "agent_agent"
        s2 = classify_coevolution(
            n_evolving_units=2, both_units_change=True, mutual_pressure=True, env_adapts=True
        )
        assert s2.stage == "agent_environment"
        s3 = classify_coevolution(
            n_evolving_units=2, both_units_change=True, mutual_pressure=True,
            env_adapts=True, omega_adapts=True,
        )
        assert s3.stage == "meta_coevolution"

    def test_red_queen_coupling(self):
        from arsi.harness.coevolution import red_queen_pressure

        # A improves while B faces harder tasks → positive corr
        r = red_queen_pressure(
            [0.5, 0.6, 0.7, 0.8],
            [0.5, 0.5, 0.5, 0.5],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        assert r["n"] >= 2
        assert "coupling" in r

    def test_anchored_meta_patch(self):
        from arsi.harness.coevolution import anchored_meta_ok, task_seed_from_failure_cluster

        ok, reason = anchored_meta_ok(True, False, True)
        assert ok is True and reason == "anchored_meta_coevolution"
        bad, why = anchored_meta_ok(True, True, True)
        assert bad is False and "goal_drift" in why
        seeds = task_seed_from_failure_cluster("tool_loop", 2)
        assert len(seeds) == 2
        assert seeds[0]["kind"] == "failure_mode_resample"


class TestFeedbackEvo:
    def test_w1_actionability(self):
        from arsi.harness.feedback_evo import actionability_gate

        g = actionability_gate(0.5, 0.7)
        assert g.ok is True
        bad = actionability_gate(0.5, 0.4)
        assert bad.ok is False
        assert "keep_critic" in bad.reason

    def test_w2_step_outcome(self):
        from arsi.harness.feedback_evo import step_outcome_consistency

        ok, _ = step_outcome_consistency([0.3, 0.4], 0.7)
        assert ok is True
        bad, reason = step_outcome_consistency([0.3, 0.4], 0.9)
        assert bad is False

    def test_w3_relabel(self):
        from arsi.harness.feedback_evo import relabel_on_metric_change

        assert relabel_on_metric_change(["pass"], "m1", "m1") is None
        r = relabel_on_metric_change(["pass", "fail"], "m1", "m2")
        assert len(r) == 2 and r[0].startswith("relabel:")

    def test_w4_hint_fade(self):
        from arsi.harness.feedback_evo import hint_fade

        assert hint_fade(0.2, 1.0) == 1.0
        assert hint_fade(0.95, 1.0) == 0.0
        mid = hint_fade(0.75, 1.0)
        assert 0.0 < mid < 1.0

    def test_w5_metric_update(self):
        from arsi.harness.feedback_evo import metric_update_allowed

        ok, _ = metric_update_allowed([True, True, True, False])
        assert ok is True
        bad, _ = metric_update_allowed([True, False, False, False])
        assert bad is False


class TestDeepRumination:
    def test_epistemic_forbids_positive_predicates(self):
        from arsi.harness.epistemic import (
            Claim,
            admit_claim,
            no_known_defect,
            rewrite_i6_claim,
            verified_contract,
        )

        ok, _ = admit_claim(no_known_defect("harness", ["crash", "leak"]))
        assert ok is True
        bad = Claim(kind="no_known_defect", subject="system", raw="this system is_correct and is_safe")
        ok2, why = admit_claim(bad)
        assert ok2 is False and "forbidden" in why
        v = verified_contract("G3", "pytest_exit_0")
        assert admit_claim(v)[0] is True
        i6 = rewrite_i6_claim(True)
        assert i6.kind == "no_known_defect"
        assert "introspective" in i6.subject or "introspection" in str(i6.raw)

    def test_multiscale_budgets(self):
        from arsi.harness.multiscale import SCALE_BUDGETS, change_allowed, multiscale_contractive

        assert change_allowed("tick", 0.01).allowed is True
        assert change_allowed("tick", 0.5).allowed is False
        assert change_allowed("meta", 0.1).allowed is True
        r = multiscale_contractive(0.1, {"tick": 0.02, "harness_round": 0.5})
        assert r["scales"]["tick"]["contractive"] is True
        assert r["scales"]["harness_round"]["contractive"] is False
        assert r["all_contractive"] is False
        assert "meta" in SCALE_BUDGETS
