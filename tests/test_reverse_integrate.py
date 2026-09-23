"""ODEWorld reverse integration — z_pre failure reconstruction (P1 reverse)."""
from __future__ import annotations

from arsi.world_model.capability_flow import (
    CapabilityFlowTracker,
    VelocityField,
    Z_KEYS,
    integrate_backward,
    pre_failure_organs,
    reverse_from_failure,
)


class TestIntegrateBackward:
    def test_recovers_higher_pre_when_v_positive(self):
        # If v>0 on memory_trust, going backward lowers z_pre
        field = VelocityField()
        field.b[Z_KEYS.index("memory_trust")] = 0.02  # constant positive v
        z_fail = {k: 0.5 for k in Z_KEYS}
        rev = integrate_backward(z_fail, field, lookback_s=100.0, steps=4)
        assert rev["z_pre"]["memory_trust"] < 0.5
        assert rev["note"] == "euler_reverse_integrate_z_pre"
        assert rev["evidence_status"] == "model_reconstruction"

    def test_recovers_lower_pre_when_v_negative(self):
        field = VelocityField()
        field.b[Z_KEYS.index("memory_trust")] = -0.02
        z_fail = {k: 0.3 for k in Z_KEYS}
        rev = integrate_backward(z_fail, field, lookback_s=100.0, steps=4)
        assert rev["z_pre"]["memory_trust"] > 0.3


class TestPreFailureOrgans:
    def test_detects_degraded_before_failure(self):
        z_pre = {"memory_trust": 0.8, "self_trust": 0.5, "behavior_predictor_trust": 0.6, "eta": 0.1}
        z_fail = {"memory_trust": 0.3, "self_trust": 0.5, "behavior_predictor_trust": 0.2, "eta": 0.1}
        deg = pre_failure_organs(z_pre, z_fail)
        assert "memory_trust" in deg
        assert "behavior_predictor_trust" in deg
        assert "self_trust" not in deg

    def test_none_if_flat(self):
        z = {"memory_trust": 0.5}
        assert pre_failure_organs(z, z) == []


class TestReverseFromFailure:
    def test_package_fields(self):
        field = VelocityField()
        field.b[Z_KEYS.index("memory_trust")] = -0.05
        z_fail = {k: 0.4 for k in Z_KEYS}
        out = reverse_from_failure(z_fail, field, lookback_s=80.0, steps=4)
        assert "z_pre" in out
        assert "pre_failure_organs" in out
        assert out["evidence_status"] == "model_reconstruction"
        # negative v → backward z_pre higher → organ degraded before failure
        assert "memory_trust" in out["pre_failure_organs"]


class TestTrackerReverse:
    def test_reverse_last_failure_marks_obs(self):
        tr = CapabilityFlowTracker()
        tr.observe({"eta": 0.1, "iwm": {"self_trust": 0.8, "memory_trust": 0.8}}, t_wall=1000.0)
        tr.observe({"eta": 0.2, "iwm": {"self_trust": 0.2, "memory_trust": 0.1}}, t_wall=1030.0)
        out = tr.reverse_last_failure()
        assert out["z_pre"]
        assert "pre_failure_organs" in out
        g = tr.flow_guidance()
        assert g.get("pre_failure_organs") is not None
        assert "z_pre" in g


class TestHostStrategyReverse:
    def test_strategy_carries_pre_failure(self):
        from arsi.iwm.host_strategy import build_host_strategy

        class Flow:
            def flow_guidance(self, horizon_s=600.0):
                return {
                    "focus": "reingest",
                    "negative_organs": ["memory_trust"],
                    "pre_failure_organs": ["memory_trust"],
                    "z_pre": {"memory_trust": 0.8, "self_trust": 0.5},
                    "v_hat": {"memory_trust": -0.01},
                    "z_now": {"memory_trust": 0.2},
                    "z_subgoal": {"memory_trust": 0.6},
                }

        class A:
            capability_flow = Flow()
            iwm = None

        st = build_host_strategy(A(), agent_id="hermes")
        assert st.focus == "reingest"
        assert "memory_trust" in st.pre_failure_organs
        assert st.z_pre
        assert st.control_flags.get("pre_failure_reverse") is True
        block = st.as_structured_block()
        assert any("pre_failure_organs" in ln for ln in block)
        assert any("z_pre" in ln for ln in block)
        assert any("model_reconstruction" in w or "unverified" in w for w in st.operational_warnings)
