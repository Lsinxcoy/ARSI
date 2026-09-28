"""Recuris R0–R2 tests: WM checkers, localization, validation gate."""
from __future__ import annotations


class TestWorkingMemory:
    def test_verbal_done_bounced(self):
        from arsi.iwm.working_memory import (
            STATUS_DONE,
            StateUpdate,
            WorkingState,
            checker_evidence_required,
        )

        v = checker_evidence_required(StateUpdate(goal_id="g0", new_status=STATUS_DONE, claimed=True))
        assert v.accepted is False and "bounce" in v.reason or "evidence" in v.reason

    def test_env_evidence_commits(self):
        from arsi.iwm.working_memory import STATUS_DONE, StateUpdate, WorkingState

        ws = WorkingState.init_from_task("fix parser", ["repro", "patch", "test"])
        row = ws.propose_update(
            StateUpdate(goal_id="g0", new_status=STATUS_DONE, evidence=["tool:pytest_exit_0"])
        )
        assert row["accepted"] is True
        assert ws.unresolved() == ["g1", "g2"]
        row2 = ws.propose_update(StateUpdate(goal_id="g1", new_status=STATUS_DONE))
        assert row2["accepted"] is False

    def test_env_says_done_strict(self):
        from arsi.iwm.working_memory import STATUS_DONE, StateUpdate, checker_env_says_done

        u = StateUpdate(goal_id="g", new_status=STATUS_DONE, evidence=["e1"])
        assert checker_env_says_done(u, ["e1"]).accepted is True
        assert checker_env_says_done(u, ["e2"]).accepted is False


class TestLocalization:
    def test_checker_bounce_path(self):
        from arsi.harness.skill_trace import (
            COMPONENT_CHECKER,
            TraceStep,
            build_gamma,
            component_for_fail_class,
            localize_failure,
        )

        steps = [
            TraceStep(
                w_t={"g0": "pending"},
                skills=["s"],
                action="act",
                observation="ok",
                w_proposed={"g0": "done"},
                checker={"accepted": False, "reason": "truth_bounce_no_env_evidence"},
                w_next={"g0": "pending"},
            )
        ]
        loc = localize_failure(build_gamma("t", steps, 0))
        assert loc.component == COMPONENT_CHECKER
        assert component_for_fail_class("gate_reject") == "checker"

    def test_harness_not_patchable(self):
        from arsi.harness.skill_trace import COMPONENT_HARNESS, TraceStep, build_gamma, localize_failure
        from arsi.harness.skill_patch import propose_patch

        loc = localize_failure(build_gamma("t", [TraceStep(w_t={}, action="a", observation="ok")], 0))
        assert loc.component == COMPONENT_HARNESS
        assert loc.patchable is False
        assert propose_patch(COMPONENT_HARNESS, skill_id="s", skill_title="t", skill_body="b") is None


class TestValidationGate:
    def test_wait_when_dev_small(self):
        from arsi.harness.skill_patch import validation_gate

        g = validation_gate(
            source_before=[0.2] * 4,
            source_after=[0.6] * 4,
            dev_before=[0.5] * 3,
            dev_after=[0.5] * 3,
            min_n=8,
        )
        assert g.accepted is False and "wait" in g.reason

    def test_reject_wide_ci(self):
        from arsi.harness.skill_patch import validation_gate

        # noisy dev around zero with tiny n
        g = validation_gate(
            source_before=[0.2] * 10,
            source_after=[0.8] * 10,
            dev_before=[0.5] * 8,
            dev_after=[0.55, 0.2, 0.9, 0.4, 0.6, 0.5, 0.45, 0.55],
            min_n=8,
        )
        assert g.accepted is False

    def test_accept_stable_dev(self):
        from arsi.harness.skill_patch import validation_gate

        g = validation_gate(
            source_before=[0.2] * 10,
            source_after=[0.8] * 10,
            dev_before=[0.5] * 10,
            dev_after=[0.52] * 10,
            min_n=8,
        )
        assert g.accepted is True
