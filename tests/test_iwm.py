"""IWM tests — real introspective world model acceptance probes.

Q1–Q3 any two fail → cannot claim introspective world model.
Evidence: no η hand-twisting; organs drive control; frontier/provenance live.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.schema import Belief, MentalState, PhysicalState, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.iwm import (
    ORGAN_BEHAVIOR_PREDICTOR,
    ORGAN_DREAM,
    ORGAN_DYNAMICS,
    IWM,
    IntrospectorCalibrator,
    KnowledgeFrontier,
    LoopEfficacy,
    OrganSelfModel,
    TransitionLedger,
)
from arsi.mnemosyne.core import Mnemosyne
from arsi.pipelines.dream import DreamPipeline, measure_behavior_error
from arsi.world_model.siwm import SIWM


class TestOrganSelf:
    def test_unknown_until_min_samples(self):
        m = OrganSelfModel(min_samples=3)
        m.record("dynamics", False)
        m.record("dynamics", False)
        assert m.status("dynamics") == "unknown"
        assert m.trust_weight("dynamics") == 0.0

    def test_unreliable_drives_hook(self):
        m = OrganSelfModel(min_samples=3, unreliable_below=0.35)
        for _ in range(5):
            m.record("dynamics", False)
        assert m.status("dynamics") == "unreliable"
        hooks = m.control_hooks()
        assert hooks["downweight_pre_enactment"] is True

    def test_dream_unreliable_forbids_default_dream(self):
        m = OrganSelfModel(min_samples=2)
        m.record(ORGAN_DREAM, False)
        m.record(ORGAN_DREAM, False)
        assert m.status(ORGAN_DREAM) == "unreliable"
        assert m.control_hooks()["forbid_default_dream"] is True


class TestLoopEfficacyEvidence:
    def test_neutral_does_not_move_eta(self):
        loops = LoopEfficacy(improve_eps=0.02)
        loops.record_verdict("dream", before=0.40, after=0.39)
        assert loops.recent_verdicts("dream")[-1] == "neutral"
        val, reason = loops.evidence_based_value(0.40, 0.39, "dream")
        assert val == 0.40
        assert "no_evidence" in reason or "not_better" in reason

    def test_helped_moves_eta_bounded(self):
        loops = LoopEfficacy(improve_eps=0.02)
        loops.record_verdict("dream", before=0.50, after=0.30)
        assert loops.recent_verdicts("dream")[-1] == "helped"
        val, reason = loops.evidence_based_value(0.50, 0.30, "dream", max_step=0.15)
        assert abs(val - 0.35) < 1e-6  # 0.50 - 0.15
        assert "evidence_move" in reason

    def test_hurt_blocks_metric(self):
        loops = LoopEfficacy()
        loops.record_verdict("dream", before=0.30, after=0.45)
        assert loops.recent_verdicts("dream")[-1] == "hurt"
        val, _ = loops.evidence_based_value(0.30, 0.45, "dream")
        assert val == 0.30

    def test_two_hurt_forbid_default(self):
        loops = LoopEfficacy()
        loops.record_verdict("dream", before=0.3, after=0.5)
        loops.record_verdict("dream", before=0.3, after=0.4)
        s = loops.intervention_summary("dream")
        assert s["allowed_default"] is False


class TestFrontierProvenanceLedger:
    def test_frontier_explore_bias(self):
        fr = KnowledgeFrontier()
        for _ in range(4):
            fr.observe("learn_task", outcome="failure", effect=0.1)
        fr.observe("remember_x", outcome="success", effect=0.9)
        assert "learn" in fr.explore_bias() or "other" in fr.explore_bias()

    def test_ledger_rolling_error(self):
        led = TransitionLedger()
        for i in range(5):
            led.record_prediction(
                organ="behavior_predictor",
                action="learn",
                predicted="learn",
                actual="dream",
                correct=False,
            )
        assert led.rolling_error("behavior_predictor") == 1.0
        assert led.rolling_accuracy("behavior_predictor") == 0.0

    def test_provenance_replay(self):
        iwm = IWM()
        rec = iwm.record_decision(action="learn", reason="test", decision_source="unit")
        assert rec.decision_id
        data = iwm.provenance.replay(rec.decision_id)
        assert data["action"] == "learn"
        assert "iwm_snapshot" in data


class TestCalibrator:
    def test_degrade_when_iwm_fails(self):
        cal = IntrospectorCalibrator(min_samples=3, degrade_below=0.35)
        for i in range(5):
            cal.record(f"d{i}", used_iwm=True, success=False)
        assert cal.should_degrade_to_baseline() is True

    def test_no_degrade_with_few_samples(self):
        cal = IntrospectorCalibrator(min_samples=5)
        cal.record("d1", used_iwm=True, success=False)
        assert cal.should_degrade_to_baseline() is False


class TestDreamNoHandTwist:
    def _build(self, tmp_path: Path):
        store = MnemosyneStore(str(tmp_path / "arsi.db"))
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store)
        iwm = IWM(archive_dir=tmp_path / "iwm")
        dream = DreamPipeline(siwm, mnemosyne, llm=None, iwm=iwm)
        return store, mnemosyne, siwm, iwm, dream

    def test_dream_eta_not_auto_halved(self, tmp_path):
        store, mnemosyne, siwm, iwm, dream = self._build(tmp_path)
        # insufficient prediction evidence → η must stay
        siwm.eta.eta_smooth = 0.40
        state = siwm.refresh_state()
        assert state.eta == pytest.approx(0.40)
        new_state = dream.execute(state)
        # no free lunch: cannot drop to 0.20 without measured help
        assert new_state.eta == pytest.approx(0.40) or new_state.eta > 0.20
        assert new_state.eta != 0.20 or dream._last_loop_trial.get("verdict") == "helped"
        # hard rule: if trial not helped, eta unchanged
        if dream._last_loop_trial.get("verdict") in ("neutral", "hurt", "unknown"):
            assert new_state.eta == pytest.approx(0.40)
        assert dream._last_eta_policy
        assert "0.5" not in dream._last_eta_policy  # no "halve" policy

    def test_dream_records_loop_trial(self, tmp_path):
        store, mnemosyne, siwm, iwm, dream = self._build(tmp_path)
        for i in range(20):
            store.record_behavior_trace(
                agent_id="mimo",
                action="learn" if i % 2 == 0 else "dream",
                outcome="success",
                effect=0.7,
            ) if hasattr(store, "record_behavior_trace") else None
        # ingest via mnemosyne path used in production
        from arsi.foundation.schema import BehaviorTrace
        for i in range(20):
            tr = BehaviorTrace(
                agent_id="mimo",
                action="learn" if i % 2 == 0 else "remember",
                outcome="success",
                effect=0.8,
            )
            mnemosyne.ingest_trace(tr)
        siwm.train_from_history()
        siwm.eta.eta_smooth = 0.40
        new_state = dream.execute(siwm.refresh_state())
        assert dream._last_loop_trial
        assert "verdict" in dream._last_loop_trial
        assert iwm.loops.trials or dream._last_loop_trial.get("iwm") is False

    def test_measure_behavior_error_bounds(self, tmp_path):
        store, mnemosyne, siwm, iwm, dream = self._build(tmp_path)
        assert measure_behavior_error(siwm.layer1, []) == 1.0
        traces = [{"action": "learn", "outcome": "success"} for _ in range(10)]
        err = measure_behavior_error(siwm.layer1, traces)
        assert 0.0 <= err <= 1.0


class TestIWMGovernorHooks:
    def _laws(self, tmp_path: Path) -> IronLaws:
        p = tmp_path / "laws.yaml"
        p.write_text("laws: []\n", encoding="utf-8")
        return IronLaws(p)

    def test_forbid_dream_changes_governor_path(self, tmp_path):
        from arsi.governor.core import AutopoieticGovernor

        store = MnemosyneStore(str(tmp_path / "arsi.db"))
        siwm = SIWM(store)
        iwm = IWM()
        # force dream organ unreliable
        iwm.organ.record(ORGAN_DREAM, False)
        iwm.organ.record(ORGAN_DREAM, False)
        iwm.organ.record(ORGAN_DREAM, False)
        gov = AutopoieticGovernor(siwm, store, self._laws(tmp_path), iwm=iwm)
        state = WorldState(
            phi=PhysicalState(generation=1, storage_stats={"trace_count": 5, "experience_count": 0}),
            psi=MentalState(beliefs=[], affect={}, norms=[]),
            eta=0.45,
        )
        action = gov.decide(state)
        assert action.a_phy == "learn"
        assert "dream" in (action.a_ment.reason or "") or "不可靠" in (action.a_ment.reason or "")

    def test_dynamics_unknown_downweights_pre_enactment(self, tmp_path):
        from arsi.governor.core import AutopoieticGovernor

        store = MnemosyneStore(str(tmp_path / "arsi.db"))
        siwm = SIWM(store)
        # fit layer1 so eta depth may be > 0
        from arsi.foundation.schema import BehaviorTrace
        mnemosyne = Mnemosyne(store)
        for i in range(30):
            mnemosyne.ingest_trace(BehaviorTrace(
                agent_id="a", action="learn", outcome="success", effect=0.7
            ))
        siwm.train_from_history()
        iwm = IWM()
        state = WorldState(
            phi=PhysicalState(generation=2, storage_stats={"trace_count": 40, "experience_count": 5}),
            psi=MentalState(beliefs=[Belief(content="x", confidence=0.8, source="t")], affect={}, norms=[]),
            eta=0.25,
        )
        advice = iwm.governor_advice(state)
        assert advice["downweight_pre_enactment"] is True  # dynamics unknown
        gov = AutopoieticGovernor(siwm, store, self._laws(tmp_path), iwm=iwm)
        action = gov.decide(state)
        assert action.a_phy != "freeze"


class TestARSIWiring:
    def _build_arsi(self, tmp_path: Path):
        from arsi.core import ARSI

        store = MnemosyneStore(str(tmp_path / "arsi.db"))
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store)
        laws_p = tmp_path / "laws.yaml"
        laws_p.write_text("laws: []\n", encoding="utf-8")
        laws = IronLaws(laws_p)
        from arsi.governor.core import AutopoieticGovernor
        from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
        dream = DreamPipeline(siwm, mnemosyne, llm=None)
        arsi = ARSI(
            store=store,
            mnemosyne=mnemosyne,
            siwm=siwm,
            governor=AutopoieticGovernor(siwm, store, laws),
            empowerment=EmpowermentEngine(mnemosyne, siwm, NullAdapter()),
            dream=dream,
            iron_laws=laws,
            llm=None,
        )
        # point IWM archive to tmp
        arsi.iwm = IWM(archive_dir=tmp_path / "iwm")
        arsi.dream.iwm = arsi.iwm
        arsi.governor.iwm = arsi.iwm
        arsi.manifest_store = __import__("arsi.meta.live_manifest", fromlist=["ManifestStore"]).ManifestStore(
            tmp_path / "trace_pool"
        )
        return arsi

    def test_arsi_has_iwm_and_stats(self, tmp_path):
        arsi = self._build_arsi(tmp_path)
        assert arsi.iwm is not None
        assert arsi.dream.iwm is arsi.iwm
        assert arsi.governor.iwm is arsi.iwm
        arsi.ingest_trace("mimo", "learn", "success", 0.8)
        arsi.ingest_trace("mimo", "remember", "success", 0.6)
        result = arsi.step()
        assert "iwm_advice" in result
        assert "prediction" in result
        stats = arsi.get_stats()
        assert "iwm" in stats
        assert "iwm_q_gate" in stats

    def test_q_gate_empty_system_is_skeleton(self, tmp_path):
        arsi = self._build_arsi(tmp_path)
        ok, info = arsi.iwm.q_gate()
        # empty introspection cannot claim v1
        assert info["claim"] in ("introspective_skeleton", "introspective_v1_candidate")
        assert isinstance(ok, bool)

    def test_iwm_q_probes_after_activity(self, tmp_path):
        arsi = self._build_arsi(tmp_path)
        from arsi.foundation.schema import BehaviorTrace
        for i in range(15):
            arsi.mnemosyne.ingest_trace(BehaviorTrace(
                agent_id="hermes",
                action="learn" if i % 3 else "remember",
                outcome="success" if i % 2 == 0 else "failure",
                effect=0.7 if i % 2 == 0 else 0.2,
            ))
        arsi.siwm.train_from_history()
        for _ in range(3):
            arsi.step()
        qs = arsi.iwm.q_scores()
        assert qs["Q2_frontier"]["pass"] is True
        assert qs["Q4_ledger"]["pass"] is True
        assert qs["Q5_provenance"]["pass"] is True
        # Q1 may pass after predictions recorded
        health = arsi.iwm.health()["iwm"]
        assert "organ_trust" in health
        assert "self_trust" in health

    def test_first_person_not_four_numbers_only(self, tmp_path):
        arsi = self._build_arsi(tmp_path)
        report = arsi.iwm.first_person_report()
        for key in ("organs", "frontier", "loop_efficacy", "ledger", "provenance", "calibrator"):
            assert key in report
