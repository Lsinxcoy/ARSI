"""P1 tests: compile gate ledger, gate_reject playbook, harness_preflight."""
from __future__ import annotations


class TestCompileGateLedger:
    def test_record_and_effect(self):
        from arsi.harness.compile_gate import (
            compile_gate_effect,
            get_compile_ledger,
            record_compile,
            reset_compile_ledger,
        )

        reset_compile_ledger()
        record_compile(ok=True, source="t")
        record_compile(ok=False, reason="boom", source="t")
        record_compile(ok=False, reason="syn", source="t", syntax=True)
        led = get_compile_ledger()
        assert led.n_pass == 1 and led.n_reject == 2 and led.n_syntax == 1
        assert 0.5 < led.reject_rate < 0.8
        eff = compile_gate_effect(
            {"invalid_or_compile": 6},
            {"invalid_or_compile": 16},
        )
        assert eff["shrank"] is True
        assert eff["delta"] == -10

    def test_code_verifier_feeds_ledger(self):
        from arsi.harness.compile_gate import get_compile_ledger, reset_compile_ledger
        from arsi.sealed_eval.code_verifier import CodeVerifier

        reset_compile_ledger()
        v = CodeVerifier(require_compile=True)
        bad = "def broken(:\n  pass\n"
        score, ev = v.verify_python_code(bad, [])
        assert score == 0.0
        assert ev.get("fail_class") in ("compile", "invalid")
        assert get_compile_ledger().total >= 1


class TestGateRejectPlaybook:
    def test_fail_recovery_has_gate_reject(self):
        from arsi.adapters.host_empower import DEFAULT_FAIL_KEYS, FAIL_RECOVERY, fail_recovery_for

        assert "gate_reject" in FAIL_RECOVERY
        assert "gate_reject" in DEFAULT_FAIL_KEYS
        rec = fail_recovery_for()
        assert any("repair" in a for a in rec["gate_reject"])

    def test_brief_mentions_gate_reject(self):
        from arsi.adapters.bidirectional_interface import ARSIBrief

        b = ARSIBrief(
            task_description="x",
            agent_id="h",
            recommendations=[],
            relevant_skills=[],
            warnings=[],
            past_lessons=[],
        )
        body = b.format_for_agent(full=True)
        assert "gate_reject" in body
        assert "Compile" in body or "compile" in body


class TestHarnessPreflight:
    def test_read_only_preflight(self):
        from arsi.adapters.host_empower import harness_preflight

        class FakeStore:
            def get_recent_traces(self, n=80, agent_id=None):
                return [
                    {
                        "outcome": "failure",
                        "agent_id": "hermes",
                        "action": "synthex_gate:cand_x",
                        "params": {"fail_class": "gate_reject"},
                    },
                    {
                        "outcome": "failure",
                        "agent_id": "hermes",
                        "action": "task",
                        "params": {"fail_class": "compile_other"},
                    },
                ]

        class FakeARSI:
            store = FakeStore()
            world_pool = None
            iwm = None
            _live_capability_scores = []

        out = harness_preflight(FakeARSI(), agent_id="hermes")
        assert out["mode"] == "read_only_no_evolve"
        labels = {c["label"] for c in out["top_clusters"]}
        assert "gate_reject" in labels or "invalid_or_compile" in labels
        assert "gate_reject" in out["recommend_playbook"]

    def test_brief_pull_preflight(self):
        from arsi.adapters.bidirectional_interface import ARSIInterface

        class FakeStore:
            current_generation = 1

            def write_memory(self, rec):
                pass

            def get_recent_traces(self, n=50, agent_id=None):
                return [
                    {
                        "outcome": "failure",
                        "agent_id": "mimo-desktop",
                        "action": "g",
                        "params": {"fail_class": "gate_reject"},
                    }
                ]

        class FakeMne:
            def __init__(self):
                self.store = FakeStore()

            def search_experience(self, **kw):
                return []

            def search_cross_agent(self, **kw):
                return []

        class FakeARSI:
            def __init__(self):
                self.store = FakeStore()
                self.mnemosyne = FakeMne()
                self.iwm = None
                self.world_pool = None
                self.discovery_tree = type("T", (), {"nodes": {}})()
                self._live_capability_scores = []

            def ingest_trace(self, **kw):
                pass

        iface = ARSIInterface(FakeARSI())
        brief = iface.brief("do work", "mimo-desktop", pull=["harness_preflight"])
        assert brief.empower_pack is not None
        assert brief.empower_pack.preflight is not None
        assert brief.empower_pack.preflight["mode"] == "read_only_no_evolve"
        body = brief.format_for_agent(full=True)
        assert "Harness Preflight" in body
