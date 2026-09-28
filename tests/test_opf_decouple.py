"""OPF decoupling policy tests — organ/pool must not share one cadence."""
from __future__ import annotations


class TestDecouplePolicy:
    def test_policy_blocks_differ(self):
        from arsi.world_model.capability_flow import DECOUPLE_POLICY

        assert DECOUPLE_POLICY["organ"] != DECOUPLE_POLICY["pool"]
        assert DECOUPLE_POLICY["pool"] == "dream_rsi_deploy_only"

    def test_encode_organ_stable_when_only_pool_moves(self):
        from arsi.world_model.capability_flow import encode_dyn_state

        base = {
            "iwm": {"self_trust": 0.4, "memory_trust": 0.8, "behavior_predictor_accuracy": 0.6},
            "eta": 0.1,
            "pool_history": [0.1, 0.2, 0.3],
            "live_history": [0.5],
        }
        z1 = encode_dyn_state(base)
        base2 = dict(base)
        base2["pool_history"] = [0.1, 0.2, 0.9]
        z2 = encode_dyn_state(base2)
        assert z1["self_trust"] == z2["self_trust"]
        assert z1["pool_last"] != z2["pool_last"]

    def test_health_includes_opf(self):
        from arsi.world_model.capability_flow import CapabilityFlowTracker

        tr = CapabilityFlowTracker()
        for i in range(8):
            tr.observe(
                {
                    "iwm": {"self_trust": 0.3 + 0.01 * i},
                    "live_history": [0.5 + 0.01 * i],
                    "pool_history": [-0.3],
                    "eta": 0.1,
                },
                note="test",
                action="learn",
            )
        h = tr.health()
        assert "opf" in h
        assert "decoouple_policy" in h or "decoouple_policy" in h
        assert h["opf"].get("note", "").startswith("statistical_opf") or h["opf"] == {}
