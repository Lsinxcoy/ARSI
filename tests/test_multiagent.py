"""Multi-agent protocol tests (M1–M4)."""
from __future__ import annotations

from arsi.adapters.bidirectional_interface import ARSIInterface
from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.schema import BehaviorTrace
from arsi.mnemosyne.core import Mnemosyne
from arsi.multiagent import MultiAgentOrchestrator
from arsi.world_model.siwm import SIWM
from arsi.governor.core import AutopoieticGovernor
from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
from arsi.pipelines.dream import DreamPipeline
from arsi.core import ARSI


def _arsi(tmp_path):
    store = MnemosyneStore(str(tmp_path / "m.db"))
    mn = Mnemosyne(store)
    siwm = SIWM(store)
    lp = tmp_path / "laws.yaml"
    lp.write_text("laws: []\n", encoding="utf-8")
    laws = IronLaws(lp)
    arsi = ARSI(
        store=store,
        mnemosyne=mn,
        siwm=siwm,
        governor=AutopoieticGovernor(siwm, store, laws),
        empowerment=EmpowermentEngine(mn, siwm, NullAdapter()),
        dream=DreamPipeline(siwm, mn),
        iron_laws=laws,
        llm=None,
    )
    from arsi.iwm import IWM
    arsi.iwm = IWM()
    arsi.dream.iwm = arsi.iwm
    arsi.governor.iwm = arsi.iwm
    return arsi


class TestMultiAgentProtocol:
    def test_register_dispatch_result(self, tmp_path):
        arsi = _arsi(tmp_path)
        iface = ARSIInterface(arsi)
        orch = MultiAgentOrchestrator(arsi=arsi, interface=iface)
        orch.register("hermes", role="worker", capabilities=["tool", "skill_write"])
        res = orch.dispatch("hermes", "write a calibration skill")
        assert res["dispatched"] is True
        assert res["task_id"]
        out = orch.submit_result(
            "hermes",
            res["task_id"],
            "write a calibration skill",
            outcome="success",
            effect=0.7,
            notes="ok",
        )
        assert out["accepted"] is True
        assert out["agent"]["tasks_completed"] == 1
        # still unmeasured until 3 outcomes (M3)
        assert out["agent"]["organ_status"] == "unmeasured"
        assert out["effect_anchor"]["external"] is True

    def test_organ_measured_after_three(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi, interface=None)
        orch.register("mimo")
        for i in range(3):
            d = orch.dispatch("mimo", f"task {i}")
            orch.submit_result("mimo", d["task_id"], f"task {i}", outcome="success", effect=0.5)
        rec = orch.agents["mimo"]
        assert rec.organ_status == "measured"
        assert rec.success_rate == 1.0

    def test_freeze_blocks_dispatch(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi)
        orch.register("a")
        orch.freeze("budget")
        d = orch.dispatch("a", "x")
        assert d["dispatched"] is False
        assert "frozen" in d["reason"]

    def test_health_unmeasured_list(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi)
        orch.register("h1")
        orch.register("h2")
        h = orch.health()["multi_agent"]
        assert set(h["unmeasured_agents"]) == {"h1", "h2"}
        assert h["agent_count"] == 2

    def test_cycle_dispatches_idle(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi)
        orch.register("w1")
        orch.register("w2")
        c = orch.cycle()
        assert c["status"] == "ok"
        assert len(c["dispatched"]) >= 1
