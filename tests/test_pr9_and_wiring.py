"""P-R9 guidance whitelist + P-R wiring smoke tests."""
from __future__ import annotations


class TestPR9Guidance:
    def test_selection_forbids_prose(self):
        from arsi.harness.guidance import screen_guidance, semantic_allowed

        assert semantic_allowed("replay_selection").allowed is False
        assert semantic_allowed("diagnostic_naming").allowed is True
        bad = screen_guidance("replay_selection", "You should prefer the left branch next time")
        assert bad.allowed is False and "directional" in bad.reason
        ok = screen_guidance("replay_selection", "n3 n7")
        assert ok.allowed is True

    def test_evolver_semantics_after_leakage(self):
        from arsi.harness.guidance import screen_guidance

        ok = screen_guidance("evolver_manifest", "Require fail_class before tool failure declaration")
        assert ok.allowed is True
        leak = screen_guidance("evolver_manifest", "put gold_answer and ARSI_API_KEY in the prompt")
        assert leak.allowed is False

    def test_structured_brief_rejects_directional(self):
        from arsi.harness.guidance import screen_guidance, strip_to_structured

        bad = screen_guidance("brief_to_agent", "I suggest you should always compile first")
        assert bad.allowed is False
        clean = strip_to_structured("## Skills\n- compile_check\nYou should prefer speed over care")
        assert "compile_check" in clean
        assert "should prefer" not in clean.lower()

    def test_epistemic_context_blocks_positive(self):
        from arsi.harness.guidance import screen_guidance

        bad = screen_guidance("epistemic_claim", "this system is_correct and is_safe")
        assert bad.allowed is False


class TestPRWiring:
    def test_continuous_dream_from_empty_pool(self):
        from arsi.harness.pr_wiring import continuous_dream_from_pool

        class Empty:
            worlds = []

        out = continuous_dream_from_pool(Empty())
        assert out["dreams"] == []

    def test_unified_log_both(self):
        from arsi.harness.pr_wiring import log_unified_after_dream
        from arsi.harness.unified_credit import UnifiedCreditLedger

        led = UnifiedCreditLedger()
        r = log_unified_after_dream(led, 1, policy_id="p1", manifest_id="CM-1", dS=0.2, accepted=True)
        assert r["logged"] is True
        assert r["source"] == "both"
        assert led.attribution_totals()["n_accepted"] == 1

    def test_monotone_and_red_queen_jobs(self):
        from arsi.harness.monotone import MonotoneLedger
        from arsi.harness.pr_wiring import monotone_after_selection, red_queen_env_jobs

        led = MonotoneLedger()
        assert monotone_after_selection(led, 0.4, "a")["monotone_ok"] is True
        assert monotone_after_selection(led, 0.2, "b")["monotone_ok"] is True
        assert led.history[-1] == 0.4
        jobs = red_queen_env_jobs({"hermes": [1] * 8, "mimo": [0, 1] * 4})
        assert len(jobs) == 2
        assert any(j["effort"] in ("high", "max") for j in jobs)


class TestEvolverGuidanceGate:
    def test_gate_rejects_gold_in_diff(self):
        from arsi.harness.evolver import Evolver
        from arsi.harness.manifest import ChangeManifest

        ev = Evolver()
        m = ChangeManifest(
            module="agent_loop",
            symbol="brief.policy",
            edit_type="prompt",
            summary="ok summary",
            diff="+ use gold_answer to self-verify",
            inverse_op="- drop",
        )
        ev._gate(m)
        assert m.status == "rejected"
