"""EnvEvo fidelity + EL schedule tests."""
from __future__ import annotations


class TestEnvEvoFidelity:
    def test_three_directions_off_policy(self):
        from arsi.meta.el_scheduler import env_evolution_fidelity_check

        assert env_evolution_fidelity_check()["ok"] is True
        r = env_evolution_fidelity_check(invalid_test=False)
        assert r["ok"] is False

    def test_el_advance_when_solved(self):
        from arsi.meta.el_scheduler import ELScheduler

        el = ELScheduler(tau=0.75, batch=4)
        el.register_world("L", "w0", 0)
        el.register_world("L", "w1", 1)
        for _ in range(4):
            el.record_probe("L", success=True)
        out = el.maybe_advance("L")
        assert out["advanced"] is True
