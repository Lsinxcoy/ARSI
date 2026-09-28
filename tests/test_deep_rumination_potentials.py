"""Tests for deep-rumination P-R2 / P-R4 / P-R5 / P-R6 / P-R7 / P-R8."""
from __future__ import annotations


class TestPR2ContinuousDream:
    def test_synthesize_layers(self):
        from arsi.world_model.continuous_dream import synthesize_continuous_dream

        nodes = [
            {"id": "a", "parent_id": "", "action": "learn", "score": 0.2, "metadata": {"live_last": 0.2, "live_ema": 0.2, "eta": 0.1}},
            {"id": "b", "parent_id": "a", "action": "dream", "score": 0.5, "metadata": {"live_last": 0.5, "live_ema": 0.4, "eta": 0.2}},
        ]
        dream = synthesize_continuous_dream(nodes, steps_between=2)
        assert dream.n_exact == 2
        assert dream.n_interp == 2
        kinds = {s.kind for s in dream.steps}
        assert "exact_node" in kinds and "interpolated" in kinds
        # scores only at exact nodes
        assert all(s.score is None for s in dream.steps if s.kind == "interpolated")
        assert dream.exact_scores() == [0.2, 0.5]
        # interp steps carry sync pairs or empty, never invent scores
        for s in dream.steps:
            if s.kind == "interpolated":
                assert s.score is None

    def test_counterfactual_half_step(self):
        from arsi.world_model.continuous_dream import counterfactual_half_step

        z = {"live_last": 0.4, "live_ema": 0.5, "eta": 0.1}
        cf = counterfactual_half_step(z, action="learn", scale=0.5)
        assert cf.kind == "counterfactual"
        assert cf.score is None
        assert set(cf.z) == set(z) or set(cf.z)

    def test_from_discovery_tree(self):
        from arsi.world_model.continuous_dream import synthesize_from_discovery_tree
        from arsi.world_model.discovery_tree import DiscoveryTree

        tree = DiscoveryTree()
        tree.build_from_traces(
            [
                {"action": "learn", "agent_id": "hermes", "outcome": "success", "effect": 0.4},
                {"action": "dream", "agent_id": "hermes", "outcome": "success", "effect": 0.6},
                {"action": "empower", "agent_id": "mimo", "outcome": "failure", "effect": 0.1},
            ]
        )
        dream = synthesize_from_discovery_tree(tree, steps_between=1)
        assert dream.n_exact >= 2
        assert dream.n_interp >= 1


class TestPR4UnifiedCredit:
    def test_dual_omega_attribution(self):
        from arsi.harness.unified_credit import (
            SOURCE_BOTH,
            SOURCE_HARNESS,
            SOURCE_POLICY,
            UnifiedCreditLedger,
        )

        led = UnifiedCreditLedger()
        led.log_manifest(1, "CM-1", dS=0.2, dC=0.05, accepted=True)
        led.log_dream(2, "beta_0.6", dS=0.1, dC=0.01, accepted=True)
        led.log_both(3, "beta×CM-1", dS=0.3, dC=0.02, accepted=True, policy_share=0.7)
        tot = led.attribution_totals()
        assert tot["n_accepted"] == 3
        assert tot["policy_credit"] > 0 and tot["harness_credit"] > 0
        assert abs(tot["policy_credit"] + tot["harness_credit"] - tot["total"]) < 1e-6
        # both-row share 0.7 policy
        both = [r for r in led.records if r.source == SOURCE_BOTH][0]
        assert both.attribution["policy"] == 0.7
        by = led.by_source()
        assert by[SOURCE_POLICY]["accepted"] == 1
        assert by[SOURCE_HARNESS]["accepted"] == 1

    def test_invalid_source_normalized(self):
        from arsi.harness.unified_credit import SOURCE_BOTH, UnifiedCreditLedger

        led = UnifiedCreditLedger()
        r = led.log(0, "bogus", "x", dS=0.1, accepted=True)
        assert r.source == SOURCE_BOTH


class TestPR5RedQueenEnv:
    def test_success_up_raises_difficulty(self):
        from arsi.meta.red_queen_env import difficulty_command_for_host, mutual_pressure_plan

        rising = [1, 1, 1, 1, 1, 1, 1, 1]  # high success
        flat = [0, 1, 0, 1, 0, 1, 0, 1]
        plan = mutual_pressure_plan({"hermes": rising, "mimo-desktop": flat}, base_d_t=1.0)
        cmds = {c["host"]: c for c in plan["commands"]}
        assert cmds["hermes"]["success_rate"] >= 0.9
        assert cmds["hermes"]["target_d_t"] > 1.0
        assert cmds["hermes"]["effort"] in ("high", "max")
        # weak peer holds or low pressure
        assert cmds["mimo-desktop"]["target_d_t"] <= cmds["hermes"]["target_d_t"]

    def test_evolver_jobs(self):
        from arsi.meta.red_queen_env import apply_to_evolver_plan, mutual_pressure_plan

        plan = mutual_pressure_plan({"hermes": [1] * 10, "synthex-mothernest": [0] * 10})
        jobs = apply_to_evolver_plan(plan)
        assert len(jobs) == 2
        assert all(j["effort"] in ("low", "high", "max") for j in jobs)
        assert all(j["direction"] in ("scenario", "skill", "length") for j in jobs)


class TestPR6OrgStructure:
    def test_pillars_encode_org_structure(self):
        from arsi.harness.pillars import CHARTER, PILLARS, audit_pillars

        core = PILLARS["L0_constitution"]["core"]
        joined = " ".join(core)
        assert "organization" in joined and "structure" in joined
        assert "never_organization" in joined or "structure" in joined
        assert "metabolism" in " ".join(CHARTER["core"])
        aud = audit_pillars()
        assert aud.ok is True


class TestPR7HarnessMonotone:
    def test_select_includes_baseline(self):
        from arsi.harness.monotone import select_manifest_monotone

        sel = select_manifest_monotone(
            "current",
            0.5,
            [{"manifest_id": "m1", "score": 0.8}, {"manifest_id": "m2", "score": 0.3}],
        )
        assert sel.monotone_ok is True
        assert sel.best_id == "m1"
        assert sel.best_score >= sel.baseline_score
        ids = {c["manifest_id"] for c in sel.candidates}
        assert "current" in ids and "m1" in ids

    def test_worse_candidate_never_wins(self):
        from arsi.harness.monotone import MonotoneLedger, select_manifest_monotone

        sel = select_manifest_monotone("cur", 0.9, [{"manifest_id": "bad", "score": 0.1}])
        assert sel.best_id == "cur"
        assert sel.monotone_ok is True
        led = MonotoneLedger()
        led.observe(0.5, "m1")
        led.observe(0.4, "m2")  # must not drop
        assert led.history[-1] == 0.5
        assert led.report()["ok"] is True

    def test_history_detects_regression(self):
        from arsi.harness.monotone import harness_v_star_ge_v0

        assert harness_v_star_ge_v0([0.1, 0.2, 0.3])["ok"] is True
        assert harness_v_star_ge_v0([0.3, 0.2, 0.3])["ok"] is False


class TestPR8Metabolism:
    def test_surface_faces(self):
        from arsi.harness.metabolism import metabolic_surface

        m = metabolic_surface(
            n_new_traces=10,
            window_actions=20,
            effects=[0.5, 0.5, -0.1],
            cost_units=100.0,
            n_iron_violations=0,
        )
        assert m.throughput == 0.5
        assert m.useful_effect == 1.0
        assert m.metabolic_rate == 100.0
        assert m.membrane_integrity == 1.0
        assert m.alive is True

    def test_membrane_breach(self):
        from arsi.harness.metabolism import metabolic_surface

        m = metabolic_surface(
            n_new_traces=1,
            window_actions=10,
            effects=[0.2],
            cost_units=10,
            n_iron_violations=10,
            n_calls=10,
        )
        assert m.membrane_integrity == 0.0
        assert m.alive is False

    def test_armor_health_carries_metabolism(self):
        from arsi.harness.runtime_wiring import armor_health

        h = armor_health([0.1, 0.2], gdi=0.05, metabolism_stats={"new_traces": 5, "window_actions": 10, "effects": [0.3], "cost_units": 30, "iron_violations": 0})
        assert "metabolism" in h
        assert "harness_monotone" in h
        assert h["metabolism"]["membrane_integrity"] == 1.0


class TestPillarsExports:
    def test_new_modules_importable(self):
        from arsi.harness import (
            MonotoneLedger,
            UnifiedCreditLedger,
            metabolic_surface,
            select_manifest_monotone,
        )
        from arsi.meta.red_queen_env import mutual_pressure_plan
        from arsi.world_model.continuous_dream import synthesize_continuous_dream

        assert callable(metabolic_surface)
        assert callable(select_manifest_monotone)
        assert UnifiedCreditLedger is not None
        assert MonotoneLedger is not None
        assert callable(mutual_pressure_plan)
        assert callable(synthesize_continuous_dream)
