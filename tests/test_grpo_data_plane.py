"""GRPO pre-data-plane tests: groups, rewards, holdout split, export-only."""
from __future__ import annotations


class TestGrouping:
    def test_same_brief_same_group(self):
        from arsi.harness.grpo_data import group_key_for

        a = {"task_id": "t1", "action": "x", "params": {"brief_id": "brief_1"}}
        b = {"task_id": "t2", "action": "y", "params": {"brief_id": "brief_1"}}
        assert group_key_for(a) == group_key_for(b)
        c = {"task_id": "t1", "action": "x", "params": {"brief_id": "brief_2"}}
        assert group_key_for(a) != group_key_for(c)

    def test_advantage_ready_needs_variance(self):
        from arsi.harness.grpo_data import build_groups
        from arsi.harness.training_bridge import trace_to_example

        traces = [
            {"task_id": "g", "action": "a", "outcome": "success", "effect": 0.8, "params": {"brief_id": "B"}},
            {"task_id": "g", "action": "a", "outcome": "failure", "effect": 0.1, "params": {"brief_id": "B"}},
        ]
        ex = [trace_to_example(t) for t in traces]
        groups = build_groups(ex, traces)
        assert len(groups) == 1
        assert groups[0].n_members == 2
        assert groups[0].advantage_ready is True


class TestRewards:
    def test_claim_reject_zeroes_reward(self):
        from arsi.harness.grpo_data import compute_reward

        r, d = compute_reward(effect=0.9, success=True, claim_rejected=True)
        assert r <= 0.0
        assert d["penalty_claim"] == 0.5

    def test_fail_class_penalty(self):
        from arsi.harness.grpo_data import compute_reward

        r1, _ = compute_reward(effect=0.5, success=True, fail_class="")
        r2, _ = compute_reward(effect=0.5, success=True, fail_class="gate_reject")
        assert r2 < r1

    def test_table_splits(self):
        from arsi.harness.grpo_data import build_split_from_traces, reward_table

        traces = [
            {"task_id": f"t{i}", "action": "a", "outcome": "success" if i % 2 else "failure", "effect": 0.5 if i % 2 else 0.1}
            for i in range(10)
        ]
        split = build_split_from_traces(traces, private_ratio=0.3)
        rows = reward_table(traces, split=split)
        assert len(rows) == 10
        assert {r.split for r in rows} <= {"selection", "private", "blacklist"}
        assert any(r.split == "selection" for r in rows)
        assert any(r.split == "private" for r in rows)


class TestPolicyMatrix:
    def test_matrix_advantage_ready(self):
        from arsi.harness.grpo_data import policy_matrix_groups

        rows = []
        for wid in ("w1", "w2"):
            for pid, rew in (("beta_0.2", 0.1), ("beta_0.6", 0.5), ("fixed", 0.2)):
                rows.append({"world_id": wid, "policy_id": pid, "reward": rew})
        groups = policy_matrix_groups(rows)
        assert len(groups) == 2
        assert all(g.advantage_ready for g in groups)
        assert groups[0].n_members == 3


class TestExportOnly:
    def test_export_writes_three_artifacts(self, tmp_path):
        from arsi.harness.grpo_data import export_grpo_data_plane

        traces = [
            {
                "task_id": f"task{i}",
                "action": "learn",
                "outcome": "success" if i % 2 == 0 else "failure",
                "effect": 0.7 if i % 2 == 0 else 0.1,
                "agent_id": "hermes",
                "params": {"brief_id": "shared", "fail_class": "" if i % 2 == 0 else "gate_reject"},
            }
            for i in range(8)
        ]
        man = export_grpo_data_plane(traces, tmp_path, harness_variant="default")
        assert man["grpo_live"] is False
        assert man["mode"] == "export_only_no_training"
        assert (tmp_path / "grpo_groups.jsonl").exists()
        assert (tmp_path / "grpo_rewards.jsonl").exists()
        assert (tmp_path / "grpo_split.json").exists()
        assert man["n_exported_rewards"] == 8
        assert man["n_advantage_ready_groups"] >= 1

    def test_blacklist_excluded(self):
        from arsi.harness.grpo_data import reward_table

        traces = [
            {"task_id": "ok", "action": "a", "outcome": "success", "effect": 0.5},
            {"task_id": "bad", "action": "sealed_task", "outcome": "success", "effect": 0.9, "note": "gold_answer"},
        ]
        rows = reward_table(traces)
        assert all(r.task_id != "bad" for r in rows)


class TestToolErrorCM:
    def test_tool_error_playbook_and_template(self):
        from arsi.adapters.host_empower import FAIL_RECOVERY
        from arsi.harness.evolver import _RULE_TEMPLATES, _canon_label

        assert "terminal" in " ".join(FAIL_RECOVERY["tool_error"])
        assert _canon_label("tool_error") == "tool_error"
        tmpl = _RULE_TEMPLATES["tool_error"]
        assert "terminal" in tmpl["diff"]

    def test_applier_writes_skill(self, tmp_path):
        from arsi.empowerment.applier import EmpowermentApplier

        class FakeStore:
            def write_memory(self, rec):
                pass

        ap = EmpowermentApplier(FakeStore())
        ap._skills_dir = tmp_path
        r = ap.apply_change_manifest(
            {
                "manifest_id": "CM-test",
                "module": "tool_use",
                "summary": "Hermes tool_error recovery",
                "diff": "+ on_tool_error: fail_class=tool_error\n",
                "inverse_op": "- drop\n",
            },
            agent_id="hermes",
        )
        assert r["applied"] is True
        assert (tmp_path / "arsi-cmtest" / "SKILL.md").exists()


class TestCompileRetryRing:
    def test_playbook_one_repair_then_abandon(self):
        from arsi.adapters.host_empower import FAIL_RECOVERY

        acts = " ".join(FAIL_RECOVERY["compile_other"])
        assert "candidate_id" in acts
        assert "abandon" in acts
        assert "ONE" in acts or "one" in acts.lower()

    def test_evolver_template_retry_ring(self):
        from arsi.harness.evolver import _RULE_TEMPLATES

        d = _RULE_TEMPLATES["invalid_or_compile"]["diff"]
        assert "candidate_id" in d
        assert "abandon" in d
