"""P2-8 action-conditioned velocity field v(z, t; c, a)."""
from __future__ import annotations

from arsi.world_model.capability_flow import (
    ACTION_POOL,
    ActionConditionedVelocityField,
    CapabilityFlowTracker,
    Z_KEYS,
    canon_action,
    integrate,
    integrate_backward,
    reverse_from_failure,
)


class TestCanonAction:
    def test_known(self):
        assert canon_action("learn") == "learn"
        assert canon_action("Dream") == "dream"

    def test_prefix_and_unknown(self):
        assert canon_action("learn_skill") == "learn"
        assert canon_action(None) == "unknown"
        assert canon_action("weird_op") == "unknown"


class TestActionConditionedField:
    def test_fallback_to_global_until_min_samples(self):
        f = ActionConditionedVelocityField(min_action_samples=3)
        z = [0.5] * len(Z_KEYS)
        v = [0.1] * len(Z_KEYS)
        for _ in range(2):
            f.fit_sample(z, v, action="learn")
        # not trusted yet → same as global
        assert f.predict_vec(z, action="learn") == f.predict_vec(z, action=None)

    def test_action_residual_diverges(self):
        f = ActionConditionedVelocityField(min_action_samples=2, residual_lr=0.2)
        z = [0.5] * len(Z_KEYS)
        # learn pushes +0.3, dream pushes -0.3
        for _ in range(8):
            f.fit_sample(z, [0.3] * len(Z_KEYS), action="learn")
            f.fit_sample(z, [-0.3] * len(Z_KEYS), action="dream")
        v_learn = f.predict_vec(z, action="learn")
        v_dream = f.predict_vec(z, action="dream")
        assert f.action_support()["learn"]["trusted"] is True
        assert f.action_support()["dream"]["trusted"] is True
        # residual should separate the two means
        assert sum(v_learn) > sum(v_dream)

    def test_unseen_action_uses_global(self):
        f = ActionConditionedVelocityField(min_action_samples=1)
        z = [0.2] * len(Z_KEYS)
        for _ in range(5):
            f.fit_sample(z, [0.1] * len(Z_KEYS), action="learn")
        assert f.predict_vec(z, action="empower") == f.predict_vec(z, action="unknown")


class TestIntegrateWithAction:
    def test_integrate_records_action(self):
        f = ActionConditionedVelocityField()
        z0 = {k: 0.2 for k in Z_KEYS}
        out = integrate(z0, f, horizon_s=100.0, steps=3, action="dream")
        assert out["action"] == "dream"

    def test_backward_with_action(self):
        f = ActionConditionedVelocityField()
        f.b[Z_KEYS.index("memory_trust")] = -0.05
        z_fail = {k: 0.4 for k in Z_KEYS}
        rev = integrate_backward(z_fail, f, lookback_s=80.0, steps=4, action="learn")
        assert rev["action"] == "learn"
        out = reverse_from_failure(z_fail, f, lookback_s=80.0, action="learn")
        assert out["action"] == "learn"


class TestTrackerActionConditioning:
    def test_observe_tags_action(self):
        tr = CapabilityFlowTracker()
        tr.observe({"eta": 0.1, "iwm": {"memory_trust": 0.8}}, t_wall=100.0, action="learn")
        tr.observe({"eta": 0.2, "iwm": {"memory_trust": 0.2}}, t_wall=130.0, action="dream")
        assert tr._last_action == "dream"
        h = tr.health()
        assert h["last_action"] == "dream"
        assert "action_support" in h

    def test_reverse_uses_action(self):
        tr = CapabilityFlowTracker()
        tr.observe({"eta": 0.1, "iwm": {"memory_trust": 0.8}}, t_wall=100.0, action="learn")
        tr.observe({"eta": 0.3, "iwm": {"memory_trust": 0.1}}, t_wall=160.0, action="repair")
        out = tr.reverse_last_failure(action="repair")
        assert out["action"] == "repair"

    def test_action_pool_complete(self):
        assert "learn" in ACTION_POOL
        assert "unknown" in ACTION_POOL
