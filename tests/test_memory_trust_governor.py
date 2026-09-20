"""Memory-trust → governor advice/control tests."""
from __future__ import annotations

from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.schema import Belief, MentalState, PhysicalState, WorldState
from arsi.foundation.store import MnemosyneStore
from arsi.governor.core import AutopoieticGovernor
from arsi.iwm import ORGAN_MEMORY, IWM
from arsi.world_model.siwm import SIWM


def _state(trace=40, exp=5, eta=0.25):
    return WorldState(
        phi=PhysicalState(generation=2, storage_stats={"trace_count": trace, "experience_count": exp}),
        psi=MentalState(beliefs=[Belief(content="x", confidence=0.7, source="t")], affect={}, norms=[]),
        eta=eta,
    )


def _gov(tmp_path, iwm):
    store = MnemosyneStore(str(tmp_path / "g.db"))
    siwm = SIWM(store)
    lp = tmp_path / "laws.yaml"
    lp.write_text("laws: []\n", encoding="utf-8")
    return AutopoieticGovernor(siwm, store, IronLaws(lp), iwm=iwm), siwm


class TestMemoryTrustAdvice:
    def test_trusted_memory_sets_hooks(self):
        iwm = IWM()
        for _ in range(4):
            iwm.organ.record(ORGAN_MEMORY, True, control_hint="host_ok")
        adv = iwm.governor_advice()
        assert adv["memory_trust"] >= 0.5
        assert adv["memory_status"] == "ok"
        assert adv["trust_memory_for_learn"] is True
        assert adv["prefer_learn"] is True
        assert adv["prefer_learn_reason"] in ("memory_trust", "behavior_predictor")
        assert adv["downweight_memory_ops"] is False

    def test_untrusted_memory_sets_hooks(self):
        iwm = IWM()
        for _ in range(4):
            iwm.organ.record(ORGAN_MEMORY, False, control_hint="write_fail")
        adv = iwm.governor_advice()
        assert adv["memory_trust"] <= 0.0
        assert adv["memory_status"] == "unreliable"
        assert adv["downweight_memory_ops"] is True
        assert adv["prefer_remember_ingest"] is True
        assert adv["prefer_learn"] is True

    def test_governor_prefers_learn_when_memory_trusted(self, tmp_path):
        iwm = IWM()
        for _ in range(4):
            iwm.organ.record(ORGAN_MEMORY, True)
        gov, _ = _gov(tmp_path, iwm)
        action = gov.decide(_state(trace=80, exp=10, eta=0.18))
        assert action.a_phy == "learn"
        hooks = action.params.get("iwm_hooks") or []
        assert "trust_memory_for_learn" in hooks or "prefer_learn" in hooks
        assert "memory" in (action.a_ment.reason or "").lower() or "learn" in action.a_phy

    def test_governor_blocks_evolve_when_memory_untrusted(self, tmp_path):
        iwm = IWM()
        for _ in range(4):
            iwm.organ.record(ORGAN_MEMORY, False)
        gov, _ = _gov(tmp_path, iwm)
        # force learn path after evolve is stripped
        action = gov.decide(_state(trace=30, exp=50, eta=0.18))
        assert action.a_phy != "evolve"
        assert action.a_phy in ("learn", "remember", "maintain")
        hooks = action.params.get("iwm_hooks") or []
        assert "downweight_memory_ops" in hooks or "prefer_remember_ingest" in hooks or "prefer_learn" in hooks
        cands_hooks = gov._last_iwm_advice
        assert cands_hooks.get("downweight_memory_ops") is True

    def test_health_exposes_memory_trust(self):
        iwm = IWM()
        iwm.observe_memory(True, note="ok")
        iwm.observe_memory(True, note="ok")
        iwm.observe_memory(True, note="ok")
        h = iwm.health()["iwm"]
        assert "memory_trust" in h
        assert "trust_memory_for_learn" in h
        assert h["memory_trust"] > 0.5
