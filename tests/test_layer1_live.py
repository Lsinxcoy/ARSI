"""Layer1 holdout + IWM organ binding tests."""
from __future__ import annotations

from arsi.foundation.schema import BehaviorTrace
from arsi.foundation.store import MnemosyneStore
from arsi.iwm import ORGAN_BEHAVIOR_PREDICTOR, IWM
from arsi.mnemosyne.core import Mnemosyne
from arsi.world_model.siwm import BehaviorPredictor, SIWM


def _seed(store, n=40):
    mn = Mnemosyne(store)
    actions = ["learn", "remember", "learn", "maintain"]
    for i in range(n):
        mn.ingest_trace(BehaviorTrace(
            agent_id="mimo",
            action=actions[i % 4],
            outcome="success" if i % 2 == 0 else "partial",
            effect=0.7,
        ))
    return mn


class TestLayer1Holdout:
    def test_holdout_on_deterministic_sequence(self, tmp_path):
        store = MnemosyneStore(str(tmp_path / "s.db"))
        _seed(store, 40)
        siwm = SIWM(store)
        res = siwm.train_from_history()
        assert res["status"] == "trained"
        assert "holdout_accuracy" in res
        assert 0.0 <= res["holdout_accuracy"] <= 1.0
        # deterministic learn/remember cycle should be learnable
        assert res["holdout_accuracy"] >= 0.4

    def test_iwm_organ_bind_holdout(self):
        iwm = IWM()
        iwm.observe_layer1_holdout(0.82, note="holdout_acc=0.820")
        rec = iwm.organ.snapshot()[ORGAN_BEHAVIOR_PREDICTOR]
        assert rec["samples"] >= 1
        assert rec["control_hint"].startswith("holdout")
        assert iwm.last_layer1_holdout == 0.82
        # after min_samples, status should reflect score
        iwm.observe_layer1_holdout(0.9)
        iwm.observe_layer1_holdout(0.85)
        assert iwm.organ.status(ORGAN_BEHAVIOR_PREDICTOR) in ("ok", "unknown", "unreliable")
        assert iwm.organ.reliability(ORGAN_BEHAVIOR_PREDICTOR) > 0.5
        health = iwm.health()["iwm"]
        assert "layer1_holdout" in health
        assert "behavior_predictor_accuracy" in health

    def test_holdout_accuracy_method(self):
        bp = BehaviorPredictor()
        traces = [{"action": "learn", "outcome": "success"} for _ in range(15)]
        traces += [{"action": "dream", "outcome": "success"} for _ in range(5)]
        bp.fit(traces)
        acc = bp.holdout_accuracy(traces[-8:])
        assert 0.0 <= acc <= 1.0
