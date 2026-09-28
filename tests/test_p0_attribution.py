"""P0 attribution/stability tests: digester typing, planner ledger, v-field clamp."""
from __future__ import annotations


class TestDigesterTyping:
    def test_params_fail_class_wins(self):
        from arsi.harness.digester import Digester

        d = Digester()
        traces = [
            {
                "outcome": "failure",
                "action": "task:x",
                "agent_id": "synthex-mothernest",
                "params": {"fail_class": "compile_other", "notes": "meh"},
            },
            {
                "outcome": "failure",
                "action": "task:y",
                "agent_id": "hermes",
                "params": {"fail_class": "tool_loop"},
            },
            {
                "outcome": "failure",
                "action": "synthex_gate:cand_p3b",
                "agent_id": "synthex-mothernest",
                "params": {},
            },
        ]
        clusters = d.digest(traces)
        labels = {c.label for c in clusters}
        assert "invalid_or_compile" in labels
        assert "tool_loop" in labels
        # gate evidence must not collapse to generic when pattern hits
        assert "gate_reject" in labels or "eval_mismatch" in labels
        assert "generic_failure" not in labels

    def test_explicit_unknown_not_generic_when_named(self):
        from arsi.harness.digester import _label

        assert _label({"outcome": "failure", "params": {"fail_class": "weird_gate"}}) == "weird_gate"
        assert _label({"outcome": "failure", "params": {"fail_class": "tool_error"}}) == "tool_error"

    def test_label_counts(self):
        from arsi.harness.digester import Digester

        c = Digester.label_counts(
            [
                {"outcome": "failure", "params": {"fail_class": "timeout"}},
                {"outcome": "failure", "params": {"fail_class": "timeout"}},
                {"outcome": "failure", "params": {"fail_class": "premature_complete"}},
            ]
        )
        assert c["timeout_or_budget"] == 2
        assert c["premature_complete"] == 1


class TestPlannerLedger:
    def test_tried_types_counted(self):
        from arsi.harness.digester import FailureCluster
        from arsi.harness.planner import Planner

        clusters = [FailureCluster(cluster_id="FC-tool_loop-3", label="tool_loop", n=3, implicated=["tool_use"])]
        prior = [
            {"edit_type": "prompt", "dimension": "c2", "summary": "a", "evidence": ["tool_loop"]},
            {"edit_type": "prompt", "dimension": "c2", "summary": "b", "evidence": ["tool_loop"]},
            {"edit_type": "config", "dimension": "c6", "summary": "c", "evidence": ["invalid_or_compile"]},
        ]
        land = Planner().landscape(clusters, prior_manifests=prior)
        assert land.tried_edit_types.get("prompt") == 2
        assert land.tried_edit_types.get("config") == 1
        assert "processor" in land.untried_edit_types
        assert "prompt" not in land.untried_edit_types

    def test_prefer_not_stuck_on_prompt(self):
        from arsi.harness.digester import FailureCluster
        from arsi.harness.planner import Planner

        clusters = [FailureCluster(cluster_id="FC-generic_failure-9", label="generic_failure", n=9)]
        prior = [{"edit_type": t, "dimension": "c2", "summary": t, "evidence": ["generic_failure"]} for t in
                 ("prompt", "processor", "tool", "config", "control")]
        land = Planner().landscape(clusters, prior_manifests=prior)
        # all tried → least tried / rotate, not blindly prompt
        assert land.recommendations[0]["rationale"] in ("least_tried_lever_rotate", "untried_lever")


class TestVelocityClamp:
    def test_fit_rejects_explosion(self):
        from arsi.world_model.capability_flow import VelocityField, _finite_clamped, velocity_gt

        assert _finite_clamped(float("inf")) == 0.0
        assert _finite_clamped(float("nan")) == 0.0
        assert _finite_clamped(1e20) == 1e3
        f = VelocityField(n_dims=3)
        for _ in range(20):
            f.fit_sample([1e8, 0.1, 0.2], [1e8, 0.0, 0.0])
        d = f.to_dict()
        assert d["mse_ewma"] < 1e6
        assert all(abs(x) <= 1e3 for x in d["a"] + d["b"])
        v = f.predict_vec([1e20, 0, 0])
        assert all(abs(x) <= 1e3 for x in v)
        gt = velocity_gt([0, 0], [1, 1], 0.0)
        assert all(abs(x) <= 1e3 for x in gt)

    def test_action_residual_clamped(self):
        from arsi.world_model.capability_flow import ActionConditionedVelocityField

        f = ActionConditionedVelocityField(n_dims=2, min_action_samples=1)
        for _ in range(10):
            f.fit_sample([50, 50], [50, 50], action="dream")
        v = f.predict_vec([50, 50], action="dream")
        assert all(abs(x) <= 1e3 for x in v)


class TestNlmClamp:
    def test_nlm_rejects_inf(self):
        from arsi.world_model.nlm_filter import ChannelNLM

        c = ChannelNLM(min_samples=2)
        for x in (1.0, 2.0, float("inf"), float("nan"), 1e20):
            c.observe(x)
        r = c.report()
        assert r["mse_ewma"] == r["mse_ewma"]  # not NaN
        assert abs(r["last"] or 0) <= 1e3


class TestDaemonSingletonHelpers:
    def test_child_respawn_detect(self):
        import os

        from scripts import arsi_daemon as ad

        os.environ["ARSI_DAEMON_SINGLETON"] = "999999"
        try:
            assert ad._is_daemon_child_respawn() is True
        finally:
            os.environ.pop("ARSI_DAEMON_SINGLETON", None)
        assert ad._is_daemon_child_respawn() is False
