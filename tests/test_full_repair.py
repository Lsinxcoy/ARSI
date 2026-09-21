"""Full-repair tests: live capability track, D7 isolation, memory bind, scheduler."""
from __future__ import annotations

from pathlib import Path

from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.schema import BehaviorTrace
from arsi.foundation.store import MnemosyneStore
from arsi.governor.core import AutopoieticGovernor
from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
from arsi.iwm import ORGAN_MEMORY, IWM
from arsi.meta.eval_loop import run_eval_loop, detect_live_regression
from arsi.meta.dream_rsi_params import DreamRSIParams
from arsi.meta.live_manifest import ManifestStore
from arsi.mnemosyne.core import Mnemosyne
from arsi.pipelines.dream import DreamPipeline
from arsi.world_model.siwm import SIWM
from arsi.core import ARSI
from arsi.multiagent import MultiAgentOrchestrator
from arsi.governor.operator_scheduler import OperatorScheduler
from arsi.foundation.schema import EmpowermentDimension


def _arsi(tmp_path):
    store = MnemosyneStore(str(tmp_path / "r.db"))
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
    arsi.iwm = IWM()
    arsi.dream.iwm = arsi.iwm
    arsi.governor.iwm = arsi.iwm
    arsi.manifest_store = ManifestStore(tmp_path / "pool")
    arsi.dream_rsi_params = DreamRSIParams()
    return arsi


class TestLiveCapability:
    def test_run_evaluation_has_score_track(self, tmp_path):
        arsi = _arsi(tmp_path)
        for i in range(20):
            arsi.mnemosyne.ingest_trace(BehaviorTrace(
                agent_id="a", action="learn" if i % 2 else "remember",
                outcome="success", effect=0.8,
            ))
        arsi.siwm.train_from_history()
        ev = arsi.run_evaluation()
        assert ev["track"] == "live_capability"
        assert "score" in ev
        assert ev["score"] > 0
        assert "components" in ev
        assert set(ev["components"].keys()) >= {"distill_ratio", "layer1_holdout", "eta_health"}

    def test_run_term_records_live_capability(self, tmp_path):
        arsi = _arsi(tmp_path)
        for i in range(15):
            arsi.mnemosyne.ingest_trace(BehaviorTrace(
                agent_id="a", action="learn", outcome="success", effect=0.7,
            ))
        arsi.siwm.train_from_history()
        out = arsi.run_term(n_steps=2)
        assert len(arsi._live_capability_scores) >= 1
        assert arsi._live_capability_scores[-1] > 0
        assert out.get("evaluation", {}).get("track") == "live_capability"


class TestD7ScoreTracks:
    def test_eval_loop_ignores_pool_scores_for_live(self, tmp_path):
        arsi = _arsi(tmp_path)
        # pool with worlds
        arsi.world_pool.append_from_traces(
            [{"action": "learn", "outcome": "success", "effect": 0.8, "params": {}}] * 5,
            world_id="W1",
        )
        # contaminate old-style term scores with pool negatives
        arsi._term_best_scores.extend([-2.0, -3.7, 0.0])
        # live capability track is healthy
        arsi._live_capability_scores.extend([0.55, 0.56, 0.57, 0.58, 0.59])
        beta_before = arsi.portfolio_policy.beta
        res = run_eval_loop(arsi, params=DreamRSIParams(rollback_window=3, rollback_delta_eps=0.02), report_dir=tmp_path / "ev")
        assert res.notes.get("live_track", {}).get("source") == "live_capability_only"
        assert "pool_track" in res.notes
        # healthy live capability → no rollback
        assert res.rollback_triggered is False
        assert arsi.portfolio_policy.beta == beta_before or not str(arsi.portfolio_policy.name).startswith("rollback")

    def test_eval_loop_rollback_on_live_capability_drop(self, tmp_path):
        arsi = _arsi(tmp_path)
        arsi.world_pool.append_from_traces(
            [{"action": "learn", "outcome": "success", "effect": 0.8, "params": {}}] * 5,
            world_id="W1",
        )
        arsi._live_capability_scores.extend([0.6, 0.6, 0.6, 0.2, 0.2, 0.2])
        res = run_eval_loop(arsi, params=DreamRSIParams(rollback_window=3, rollback_delta_eps=0.02), report_dir=tmp_path / "ev")
        assert res.live_regression is True
        assert res.rollback_triggered is True

    def test_insufficient_live_samples_no_rollback(self, tmp_path):
        arsi = _arsi(tmp_path)
        arsi.world_pool.append_from_traces(
            [{"action": "learn", "outcome": "success", "effect": 0.8, "params": {}}] * 4,
            world_id="W1",
        )
        arsi._live_capability_scores.extend([0.1, 0.1])
        res = run_eval_loop(arsi, params=DreamRSIParams(rollback_window=3), report_dir=tmp_path / "ev")
        assert res.rollback_triggered is False
        assert "insufficient_live_capability" in res.notes.get("regression_reason", "")


class TestHostMemoryBind:
    def test_every_host_result_binds_memory(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi)
        orch.register("h1")
        d = orch.dispatch("h1", "task")
        r = orch.submit_result("h1", d["task_id"], "task", outcome="success", effect=0.5)
        assert r.get("iwm_bind") == "observe_memory"
        assert arsi.iwm.organ.snapshot()[ORGAN_MEMORY]["samples"] >= 1
        # still unmeasured until 3 outcomes
        assert r["agent"]["organ_status"] == "unmeasured"

    def test_measured_after_three(self, tmp_path):
        arsi = _arsi(tmp_path)
        orch = MultiAgentOrchestrator(arsi=arsi)
        orch.register("h2")
        for i in range(3):
            d = orch.dispatch("h2", f"t{i}")
            orch.submit_result("h2", d["task_id"], f"t{i}", outcome="success", effect=0.5)
        assert orch.agents["h2"].organ_status == "measured"
        assert arsi.iwm.organ.trust_weight(ORGAN_MEMORY) > 0


class TestSchedulerUncovered:
    def test_schedule_forces_uncovered_dim(self):
        sch = OperatorScheduler()
        ops = sch.schedule(
            budget=2.0,
            max_operators=2,
            uncovered=["environment", "metacognition"],
        )
        names = [op.dimension for op in ops]
        assert EmpowermentDimension.ENVIRONMENT in names or EmpowermentDimension.METACOGNITION in names

    def test_schedule_without_uncovered_still_works(self):
        sch = OperatorScheduler()
        ops = sch.schedule(budget=3.0, max_operators=2)
        assert 1 <= len(ops) <= 2
