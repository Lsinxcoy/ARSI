"""GAI fidelity tests: polarity, delusion box, dual sensor."""
from __future__ import annotations


class TestGAIPolarity:
    def test_arsi_gai_rsi_anchored(self):
        from arsi.harness.gai import ARSI_GAI, assert_anchored

        assert ARSI_GAI.is_rsi() is True
        ok, why = assert_anchored(ARSI_GAI)
        assert ok is True and why == "anchored"

    def test_goal_drift_flagged(self):
        from arsi.harness.gai import GAIConfig, rsi_defects

        cfg = GAIConfig(modifier_in_agent=True, base_grounded=True, base_inside_agent=True)
        assert cfg.polarity() == "goal_drift"
        r = rsi_defects(cfg)
        assert "goal_drift_agent_rewrites_standard" in r.defects


class TestDelusionBox:
    def test_criterion_without_world(self):
        from arsi.harness.gai import delusion_box_check

        ok, why = delusion_box_check(criterion_satisfied=True, grounded_in_world=False)
        assert ok is False and "delusion" in why
        ok2, _ = delusion_box_check(criterion_satisfied=True, grounded_in_world=True)
        assert ok2 is True

    def test_utility_rewrite_needs_everitt(self):
        from arsi.harness.gai import delusion_box_check

        ok, why = delusion_box_check(
            criterion_satisfied=True,
            grounded_in_world=True,
            utility_rewrite=True,
            everitt_ok=False,
        )
        assert ok is False and "everitt" in why


class TestDualSensor:
    def test_gdi_with_stable_anchor_alarms(self):
        from arsi.harness.gai import dual_sensor_goal_drift

        r = dual_sensor_goal_drift(anchor_hash_stable=True, gdi=0.6, steps_past_best=20)
        assert r["goal_drift_suspect"] is True
        assert r["polarity_alarm"] == "goal_drift"
        r2 = dual_sensor_goal_drift(anchor_hash_stable=True, gdi=0.1, steps_past_best=0)
        assert r2["goal_drift_suspect"] is False
