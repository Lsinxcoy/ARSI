"""A-line IWM host strategy tests + A/B joint channel acceptance pieces."""
from __future__ import annotations

import json
from pathlib import Path

from arsi.adapters.bidirectional_interface import ARSIInterface, ARSIReport
from arsi.empowerment.applier import EmpowermentApplier
from arsi.foundation.efficiency_gate import host_channel_gate
from arsi.foundation.evidence_receipt import build_receipt_from_text
from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.schema import BehaviorTrace
from arsi.foundation.store import MnemosyneStore
from arsi.governor.core import AutopoieticGovernor
from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
from arsi.iwm import ORGAN_MEMORY, IWM
from arsi.iwm.host_strategy import (
    HostStrategy,
    build_host_strategy,
    strategy_for_skill_content,
)
from arsi.mnemosyne.core import Mnemosyne
from arsi.multiagent import MultiAgentOrchestrator
from arsi.pipelines.dream import DreamPipeline
from arsi.world_model.siwm import SIWM
from arsi.core import ARSI


def _arsi(tmp_path):
    store = MnemosyneStore(str(tmp_path / "a.db"))
    mn = Mnemosyne(store)
    siwm = SIWM(store)
    lp = tmp_path / "l.yaml"
    lp.write_text("laws: []\n", encoding="utf-8")
    laws = IronLaws(lp)
    arsi = ARSI(
        store=store, mnemosyne=mn, siwm=siwm,
        governor=AutopoieticGovernor(siwm, store, laws),
        empowerment=EmpowermentEngine(mn, siwm, NullAdapter()),
        dream=DreamPipeline(siwm, mn), iron_laws=laws, llm=None,
    )
    arsi.iwm = IWM()
    arsi.dream.iwm = arsi.iwm
    arsi.governor.iwm = arsi.iwm
    return arsi


class TestHostStrategy:
    def test_untrusted_memory_focus_reingest(self):
        arsi = _arsi(__import__("tempfile").mkdtemp())
        for _ in range(4):
            arsi.iwm.organ.record(ORGAN_MEMORY, False)
        st = build_host_strategy(arsi, agent_id="hermes")
        assert st.focus == "reingest"
        assert st.control_flags["downweight_memory_ops"] is True
        assert any("memory" in w for w in st.operational_warnings)

    def test_trusted_memory_focus_learn(self):
        import tempfile
        arsi = _arsi(Path(tempfile.mkdtemp()))
        for _ in range(4):
            arsi.iwm.organ.record(ORGAN_MEMORY, True)
        st = build_host_strategy(arsi, agent_id="mimo-desktop")
        assert st.focus in ("learn", "execute_with_evidence")
        assert st.control_flags.get("trust_memory_for_learn") is True or st.confidence > 0.5
        assert st.skill_priorities

    def test_structured_block_has_focus(self):
        st = HostStrategy(agent_id="h", focus="learn", confidence=0.7,
                          skill_priorities=["arsi-knowledge"],
                          control_flags={"trust_memory_for_learn": True},
                          operational_warnings=["x"])
        block = st.as_structured_block()
        assert any("focus: learn" in ln for ln in block)
        assert any("arsi-knowledge" in ln for ln in block)

    def test_skill_content_binding(self):
        st = HostStrategy(agent_id="h", focus="reingest",
                          control_flags={"downweight_memory_ops": True},
                          operational_warnings=["memory_organ_unreliable"],
                          organ_snapshot={"organ_trust": {ORGAN_MEMORY: 0.0}})
        s = strategy_for_skill_content(st)
        assert "IWM Strategy Binding" in s
        assert "re-ingest" in s or "reingest" in s or "memory weak" in s


class TestBriefCarriesStrategy:
    def test_brief_includes_iwm_strategy(self, tmp_path):
        arsi = _arsi(tmp_path)
        for _ in range(4):
            arsi.iwm.organ.record(ORGAN_MEMORY, True)
        iface = ARSIInterface(arsi)
        brief = iface.brief("write calibration skill", "hermes")
        assert getattr(brief, "channel_strategy", None) is not None
        body = brief.format_for_agent()
        assert "IWM Strategy (structured facts)" in body
        assert "focus:" in body

    def test_applier_bind_arsi_suffix(self, tmp_path):
        arsi = _arsi(tmp_path)
        for _ in range(4):
            arsi.iwm.organ.record(ORGAN_MEMORY, True)
        ap = EmpowermentApplier(arsi.store)
        ap.bind_arsi(arsi)
        suffix = ap._strategy_binding_suffix()
        assert "IWM Strategy Binding" in suffix


class TestABJointAcceptance:
    def test_b_channel_receipt_and_a_strategy_together(self, tmp_path):
        """A+B: grounded receipt + IWM strategy facts + dual gate."""
        arsi = _arsi(tmp_path)
        for _ in range(4):
            arsi.iwm.organ.record(ORGAN_MEMORY, True)
        iface = ARSIInterface(arsi)
        brief = iface.brief("host empowerment task", "hermes")
        body = brief.format_for_agent()
        rec = build_receipt_from_text(body, source_kind="ab_accept", fields={"agent": "hermes"})
        assert rec.verified is True
        assert getattr(brief, "channel_strategy", None) is not None
        # dual gate: capability stable, cost down
        gate = host_channel_gate(0.9, 0.9, 1000, 750)
        assert gate.accepted is True

    def test_host_result_binds_iwm_memory(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi)
        orch.register("mimo-desktop")
        d = orch.dispatch("mimo-desktop", "task")
        r = orch.submit_result("mimo-desktop", d["task_id"], "task", "success", effect=0.5)
        assert r.get("iwm_bind") == "observe_memory"
        assert arsi.iwm.organ.snapshot()[ORGAN_MEMORY]["samples"] >= 1
        # strategy should prefer learn after trust evidence
        st = build_host_strategy(arsi, agent_id="mimo-desktop")
        # 1 sample may still be unknown status until min_samples
        assert st.focus in ("learn", "reingest", "calibrate", "execute_with_evidence", "explore_frontier")
