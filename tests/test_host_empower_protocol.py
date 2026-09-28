"""Host protocol empowerment-max upgrade tests (dual boundary)."""
from __future__ import annotations


class TestEmpowermentPack:
    def test_pack_block_has_executable_surfaces(self):
        from arsi.adapters.host_empower import EmpowermentPack, build_empowerment_pack, fail_recovery_for

        pack = build_empowerment_pack(None, agent_id="hermes")
        text = "\n".join(pack.as_structured_block())
        assert "Skill Kit" in text
        assert "Fail Recovery Playbook" in text
        assert "Change-Manifests" in text
        assert "never is_correct" in text
        rec = fail_recovery_for(["tool_loop", "timeout"])
        assert "abort_identical_call" in rec["tool_loop"]
        assert rec["timeout"]

    def test_scaffold_fades_hint(self):
        from arsi.harness.feedback_evo import hint_fade

        assert hint_fade(0.95, 1.0) == 0.0
        assert hint_fade(0.2, 1.0) == 1.0


class TestBriefEmpowerment:
    def test_brief_includes_empower_pack(self):
        from arsi.adapters.bidirectional_interface import ARSIBrief
        from arsi.adapters.host_empower import build_empowerment_pack

        pack = build_empowerment_pack(None, agent_id="mimo-desktop")
        brief = ARSIBrief(
            task_description="fix parser",
            agent_id="mimo-desktop",
            recommendations=["meta only"],
            relevant_skills=["compile-gate"],
            warnings=[],
            past_lessons=[],
            empower_pack=pack,
        )
        body = brief.format_for_agent(full=True)
        assert "Empowerment Pack" in body
        assert "tool_error" in body
        assert "Change-Manifests" in body
        d = brief.to_dict()
        assert d["empower_pack"] is not None


class TestReportClaimGate:
    def test_forbidden_claim_demotes_outcome(self):
        from arsi.adapters.bidirectional_interface import ARSIReport, ARSIInterface

        class FakeStore:
            def __init__(self):
                self.rows = []
                self.current_generation = 1

            def write_memory(self, rec):
                self.rows.append(rec)

            def get_recent_traces(self, n=50, agent_id=None):
                return []

        class FakeMne:
            def __init__(self, store):
                self.store = store

            def search_experience(self, **kw):
                return []

            def search_cross_agent(self, **kw):
                return []

            def record_empowerment(self, *a, **k):
                pass

        class FakeIWM:
            last_gdi = 0.0

        class FakeARSI:
            def __init__(self):
                self.store = FakeStore()
                self.mnemosyne = FakeMne(self.store)
                self.iwm = FakeIWM()
                self.world_pool = None
                self.discovery_tree = type("T", (), {"nodes": {}})()
                self._live_capability_scores = [0.3, 0.4]
                self.traces = []

            def ingest_trace(self, **kw):
                self.traces.append(kw)

        arsi = FakeARSI()
        iface = ARSIInterface(arsi)
        bad = ARSIReport(
            brief_id="b1",
            agent_id="hermes",
            task_description="x",
            outcome="success",
            effect=0.9,
            skills_used=[],
            recommendations_followed=[],
            recommendations_ignored=[],
            claim={"kind": "no_known_defect", "subject": "sys", "raw": "this is_correct and is_safe"},
        )
        out = iface.report(bad)
        assert out["claim_gate"]["accepted"] is False
        assert "forbidden" in out["claim_gate"]["reason"]
        assert arsi.traces[0]["outcome"] == "unknown"
        assert arsi.traces[0]["effect"] <= 0.0

    def test_actionable_fields_land_in_trace(self):
        from arsi.adapters.bidirectional_interface import ARSIReport, ARSIInterface

        class FakeStore:
            current_generation = 1

            def write_memory(self, rec):
                pass

            def get_recent_traces(self, n=50, agent_id=None):
                return []

        class FakeMne:
            def __init__(self, store):
                self.store = store

            def search_experience(self, **kw):
                return []

            def search_cross_agent(self, **kw):
                return []

        class FakeARSI:
            def __init__(self):
                self.store = FakeStore()
                self.mnemosyne = FakeMne(self.store)
                self.iwm = None
                self.world_pool = None
                self.discovery_tree = type("T", (), {"nodes": {}})()
                self._live_capability_scores = []
                self.traces = []

            def ingest_trace(self, **kw):
                self.traces.append(kw)

        arsi = FakeARSI()
        iface = ARSIInterface(arsi)
        r = ARSIReport(
            brief_id="b2",
            agent_id="synthex",
            task_description="emit code",
            outcome="partial",
            effect=0.3,
            skills_used=["compile-gate"],
            recommendations_followed=["compile"],
            recommendations_ignored=[],
            fail_class="compile_other",
            recovery_attempted="run_compile_check",
            recovery_worked=True,
            acceptance_evidence="pytest_exit_0",
            manifest_ids_used=["CM-7b86cd1d1f"],
            compile_checked=True,
            claim={"kind": "verified_contract", "contract_id": "G3", "evidence": "pytest_exit_0"},
            measurements={"tokens": 120},
        )
        out = iface.report(r)
        p = arsi.traces[0]["params"]
        assert p["fail_class"] == "compile_other"
        assert p["recovery_worked"] is True
        assert p["compile_checked"] is True
        assert p["manifest_ids_used"] == ["CM-7b86cd1d1f"]
        assert out["claim_gate"]["accepted"] is True
        assert out["actionable"]["acceptance_evidence"] == "pytest_exit_0"


class TestDualBoundary:
    def test_selection_still_forbids_semantics(self):
        from arsi.harness.guidance import semantic_allowed

        assert semantic_allowed("replay_selection").allowed is False
        assert semantic_allowed("evolver_manifest").allowed is True
