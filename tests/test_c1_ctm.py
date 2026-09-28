"""C1-1/2/3 — CTM certainty calibration, adaptive think ticks, action-conditioned sync."""
from __future__ import annotations

from arsi.foundation.schema import Belief, MentalState, PhysicalState, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.sync_repr import action_conditioned_sync
from arsi.governor.pre_enactment import PreEnactmentEngine
from arsi.iwm.calibrate import IntrospectorCalibrator
from arsi.world_model.capability_flow import Z_BLOCKS, CapabilityFlowTracker
from arsi.world_model.dynamics import DynamicsModel


def _state(eta=0.2, trace=40, exp=10):
    return WorldState(
        phi=PhysicalState(generation=1, storage_stats={"trace_count": trace, "experience_count": exp}),
        psi=MentalState(beliefs=[Belief(content="x", confidence=0.5, source="t")], affect={}, norms=[]),
        eta=eta,
    )


def _pe(tmp_path, max_ticks=4, trained=False):
    store = MnemosyneStore(str(tmp_path / "pe.db"))
    dyn = DynamicsModel(store, llm=None)
    dyn.transition_model._trained = trained
    return PreEnactmentEngine(dyn, max_think_ticks=max_ticks)


class TestC11Calibration:
    def test_overconfident_errors_lower_trust(self):
        cal = IntrospectorCalibrator(min_samples=5)
        # high confidence, many failures
        for i in range(8):
            cal.record(f"d{i}", used_iwm=True, success=False, confidence=0.95)
        st = cal.self_trust()
        rel = cal.reliability()
        assert rel["n"] == 8
        assert rel["overconfident"] is True
        assert rel["ece"] > 0.2
        assert st < 0.5

    def test_well_calibrated_keeps_rate(self):
        cal = IntrospectorCalibrator(min_samples=5)
        for i in range(10):
            # conf ~ accuracy
            ok = i % 2 == 0
            cal.record(f"d{i}", used_iwm=True, success=ok, confidence=0.5)
        rel = cal.reliability()
        assert rel["n"] == 10
        assert rel["ece"] < 0.2
        assert 0.2 <= cal.self_trust() <= 0.75

    def test_severe_overconf_degrades(self):
        cal = IntrospectorCalibrator(min_samples=5, degrade_below=0.35)
        for i in range(10):
            cal.record(f"d{i}", used_iwm=True, success=False, confidence=0.99)
        assert cal.should_degrade_to_baseline() is True
        assert "reliability" in cal.report()

    def test_advice_includes_calibration(self):
        from arsi.iwm import IWM

        iwm = IWM()
        for i in range(6):
            iwm.apply_outcome(f"x{i}", used_iwm=True, success=False, confidence=0.9)
        adv = iwm.governor_advice()
        assert "calibration" in adv
        assert adv["calibration"].get("n", 0) >= 6


class TestC12AdaptiveThink:
    def test_easy_low_eta_uses_fewer_ticks(self, tmp_path):
        pe = _pe(tmp_path, max_ticks=4, trained=True)
        pe.flow_guide = {"z_now": {"eta": 0.0}, "n_v_updates": 10, "field_mse": 0.0}
        assert pe.plan_think_ticks() <= 2
        assert pe.difficulty() < 0.6

    def test_hard_high_eta_uses_more_ticks(self, tmp_path):
        pe = _pe(tmp_path, max_ticks=4, trained=False)
        pe.flow_guide = {"z_now": {"eta": 0.5}, "n_v_updates": 0, "field_mse": 0.3}
        assert pe.plan_think_ticks() >= 3
        assert pe.difficulty() > 0.7

    def test_selection_certainty_margin(self):
        ev = [
            {"score": 1.0, "confidence": 0.8},
            {"score": 0.1, "confidence": 0.2},
        ]
        assert PreEnactmentEngine.selection_certainty(ev) > 0.7
        ev2 = [
            {"score": 0.5, "confidence": 0.2},
            {"score": 0.5, "confidence": 0.2},
        ]
        assert PreEnactmentEngine.selection_certainty(ev2) < 0.4

    def test_select_best_adaptive_reports_ticks(self, tmp_path):
        pe = _pe(tmp_path, max_ticks=3, trained=True)
        pe.flow_guide = {"z_now": {"eta": 0.1}, "n_v_updates": 5}
        out = pe.select_best_adaptive(_state(), ["learn", "dream", "remember"])
        assert out["action"] in ("learn", "dream", "remember")
        assert 1 <= out["think_ticks"] <= out["planned_think_ticks"] <= 3
        assert "selection_certainty" in out
        assert "early_stop" in out
        assert out["max_think_ticks"] == 3


class TestC13ActionSync:
    def test_unmeasured_small_action(self):
        rows = [{"self_trust": 0.1, "memory_trust": 0.2}, {"self_trust": 0.2, "memory_trust": 0.3}]
        rep = action_conditioned_sync(rows, ["dream", "dream"], {"organ": ("self_trust", "memory_trust")})
        assert rep["by_action"]["dream"]["trusted"] is False
        assert rep["by_action"]["dream"]["note"] == "unmeasured"

    def test_trusted_action_has_blocks(self):
        rows = []
        acts = []
        for i in range(6):
            rows.append({"self_trust": 0.5 + 0.01 * i, "memory_trust": 0.4 + 0.02 * i, "eta": 0.1})
            acts.append("learn")
        for i in range(4):
            rows.append({"self_trust": 0.9 - 0.01 * i, "memory_trust": 0.2, "eta": 0.2})
            acts.append("dream")
        rep = action_conditioned_sync(rows, acts, {"organ": ("self_trust", "memory_trust", "eta")})
        assert rep["by_action"]["learn"]["trusted"] is True
        assert "blocks" in rep["by_action"]["learn"]
        assert rep["by_action"]["dream"]["trusted"] is True

    def test_tracker_sync_has_by_action(self):
        tr = CapabilityFlowTracker()
        for i in range(6):
            tr.observe(
                {
                    "eta": 0.1 + 0.01 * i,
                    "iwm": {"self_trust": 0.5 + 0.01 * i, "memory_trust": 0.4 + 0.01 * i},
                },
                t_wall=100.0 + i * 5.0,
                action="learn" if i < 4 else "dream",
            )
        h = tr.health()
        sync = h.get("sync") or {}
        ba = sync.get("by_action") or {}
        assert "learn" in ba or "by_action" in sync
        # by_action may nest under sync["by_action"] = action_conditioned_sync result
        if "by_action" in ba:
            assert ba["by_action"].get("learn", {}).get("n", 0) >= 3
