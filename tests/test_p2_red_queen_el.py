"""P2 tests: Red Queen difficulty↑, EL success flag, curriculum hold."""
from __future__ import annotations


class TestELSuccessFlag:
    def test_record_probe_trusts_success(self):
        from arsi.meta.el_scheduler import ELScheduler

        el = ELScheduler(tau=0.5, batch=4)
        el.register_world("lin1", "w1", 0)
        # negative replay score but quality-based success=True must count as pass
        el.record_probe("lin1", success=True, score=-0.34)
        el.record_probe("lin1", success=True, score=-0.34)
        assert el.pass_rate("lin1") == 1.0
        el.record_probe("lin1", success=False, score=-1.0)
        assert el.pass_rate("lin1") == 2 / 3

    def test_curriculum_hold_on_zero_pass(self):
        from arsi.meta.el_scheduler import ELScheduler

        el = ELScheduler(tau=0.75, batch=4)
        el.register_world("lin2", "w2", 0)
        for _ in range(4):
            el.record_probe("lin2", success=False, score=-0.3)
        out = el.maybe_advance("lin2")
        assert out["advanced"] is False
        assert out["reason"] == "curriculum_hold_zero_pass"

    def test_advance_when_pass_high(self):
        from arsi.meta.el_scheduler import ELScheduler

        el = ELScheduler(tau=0.75, batch=4)
        el.register_world("lin3", "w3", 0)
        el.register_world("lin3", "w3b", 1)
        for _ in range(4):
            el.record_probe("lin3", success=True, score=-0.1)
        out = el.maybe_advance("lin3")
        assert out["advanced"] is True


class TestRedQueenHarderChildren:
    def test_verify_rejects_easier_child(self):
        from arsi.world_model.world_evolver import traces_to_world, verify_world

        seed = traces_to_world(
            [
                {"action": f"a{i}", "agent_id": "hermes", "outcome": "success" if i % 3 == 0 else "failure", "effect": 0.5}
                for i in range(20)
            ],
            "seed",
        )
        easy = traces_to_world(
            [{"action": "x", "agent_id": "hermes", "outcome": "success", "effect": 0.9}],
            "easy",
        )
        res = verify_world(easy, seed_world=seed, effort="high")
        # easy child must not fully pass quality under Red Queen rule
        assert res.quality_ok is False or "difficulty" in " ".join(res.reasons)

    def test_evolve_high_does_not_shrink(self):
        from arsi.world_model.world_evolver import WorldEvolver, traces_to_world
        from arsi.meta.env_difficulty import compute_env_difficulty

        seed = traces_to_world(
            [
                {
                    "action": f"act{i%5}",
                    "agent_id": "hermes" if i % 2 == 0 else "mimo",
                    "outcome": "success" if i % 4 == 0 else "failure",
                    "effect": 0.4,
                    "params": {"fail_class": "ok" if i % 4 == 0 else "timeout"},
                }
                for i in range(24)
            ],
            "seed_big",
        )
        ev = WorldEvolver()
        seed_d = compute_env_difficulty(seed)
        accepted = 0
        for _ in range(10):
            r = ev.evolve(seed, direction="length", effort="high")
            if r.accepted:
                accepted += 1
                # Red Queen floor: no catastrophic collapse
                assert r.child_difficulty.get("d_t", 0) >= seed_d.d_t * 0.85 - 1e-6
        assert accepted >= 1


class TestRedQueenWiring:
    def test_mutual_pressure_plan(self):
        from arsi.meta.red_queen_env import mutual_pressure_plan, red_queen_effort_for_pool

        plan = mutual_pressure_plan({"hermes": [1, 1, 1, 1], "mimo-desktop": [0, 0, 1, 0]})
        cmds = {c["host"]: c for c in plan["commands"]}
        assert cmds["hermes"]["effort"] in ("high", "max")
        assert cmds["hermes"]["target_d_t"] > 1.0

        class EmptyPool:
            worlds = []
            _manifest = []

        rq = red_queen_effort_for_pool(EmptyPool())
        assert rq["effort"] in ("low", "high", "max")
        assert rq["note"].startswith("p2_2")
