"""C2 — NLM private filters, kernel v, cognitive map, sync working memory."""
from __future__ import annotations

from arsi.foundation.sync_memory import SyncWorkingMemory
from arsi.world_model.capability_flow import Z_KEYS, CapabilityFlowTracker
from arsi.world_model.cognitive_map import CognitiveMap
from arsi.world_model.kernel_flow import KernelVelocityField
from arsi.world_model.nlm_filter import ChannelNLM, NLMBank


class TestC21NLM:
    def test_private_filter_learns_linear(self):
        ch = ChannelNLM(history_len=4, lr=0.2, min_samples=4)
        # x_t = 0.8 * x_{t-1} pattern-ish ramp
        for i in range(30):
            ch.observe(0.1 * i)
        assert ch.trusted()
        assert ch.n_updates >= 4
        rep = ch.report()
        assert rep["trusted"] is True
        assert "pred" in rep

    def test_bank_untrusted_echoes(self):
        bank = NLMBank(["a", "b"], history_len=4, min_samples=5)
        z1 = bank.observe_z({"a": 1.0, "b": 2.0})
        assert bank.report()["trusted_frac"] == 0.0
        assert z1["a"] == 1.0  # echo when untrusted

    def test_bank_predict_keys(self):
        bank = NLMBank(Z_KEYS, min_samples=3)
        for i in range(10):
            bank.observe_z({k: 0.1 * i for k in Z_KEYS})
        pred = bank.predict_z()
        assert set(pred) == set(Z_KEYS)


class TestC22KernelV:
    def test_learns_constant_velocity(self):
        f = KernelVelocityField(bandwidth=0.5, min_samples=4)
        z = [0.5] * len(Z_KEYS)
        v = [0.1] * len(Z_KEYS)
        for _ in range(10):
            f.fit_sample(z, v, action="learn")
        assert f.trusted()
        pred = f.predict_vec(z, action="learn")
        assert abs(sum(pred) / len(pred) - 0.1) < 0.15

    def test_action_support_fallback(self):
        f = KernelVelocityField(min_samples=4, min_action_samples=3)
        z = [0.2] * len(Z_KEYS)
        for _ in range(2):
            f.fit_sample(z, [0.3] * len(Z_KEYS), action="learn")
        # dream unmeasured → global support
        p_dream = f.predict_vec(z, action="dream")
        p_none = f.predict_vec(z, action=None)
        assert len(p_dream) == len(p_none)


class TestC23CognitiveMap:
    def test_route_uses_historical_delta(self):
        m = CognitiveMap(goal_eps=0.05)
        for _ in range(5):
            m.observe(
                {"memory_trust": 0.2, "self_trust": 0.5},
                {"memory_trust": 0.6, "self_trust": 0.5},
                "learn",
            )
        r = m.route({"memory_trust": 0.2, "self_trust": 0.5}, {"memory_trust": 0.8, "self_trust": 0.5})
        assert r["action_order"][:1] == ["learn"]
        assert r["coverage"] > 0
        assert r["note"] == "cognitive_map_no_external_coords"

    def test_unmeasured_action_slot(self):
        m = CognitiveMap(min_obs=3, goal_eps=0.05)
        r = m.route({"a": 0.0}, {"a": 1.0})
        # keys default may not include 'a' — use known keys
        m2 = CognitiveMap()
        r2 = m2.route({"memory_trust": 0.0}, {"memory_trust": 1.0})
        assert any(s.get("action") is None for s in r2["steps"]) or r2["coverage"] == 0

    def test_report(self):
        m = CognitiveMap()
        m.observe({"eta": 0.1}, {"eta": 0.2}, "dream")
        assert m.report()["edges"] >= 1


class TestC24SyncMemory:
    def test_bind_and_recall(self):
        wm = SyncWorkingMemory(bind_rho=0.0, min_bind_n=2)
        for i in range(4):
            z = {"self_trust": 0.5 + 0.01 * i, "memory_trust": 0.4 + 0.02 * i}
            wm.bind(z, evidence_id=f"ev{i}")
        rec = wm.recall({"self_trust": 0.55, "memory_trust": 0.5})
        assert rec
        assert all("evidence_id" in r for r in rec)

    def test_empty_recall(self):
        wm = SyncWorkingMemory()
        assert wm.recall({}) == []
        assert wm.report()["n_bound"] == 0


class TestTrackerC2:
    def test_health_carries_c2(self):
        tr = CapabilityFlowTracker()
        for i in range(8):
            tr.observe(
                {
                    "eta": 0.1 + 0.01 * i,
                    "iwm": {"self_trust": 0.5 + 0.01 * i, "memory_trust": 0.4 + 0.01 * i},
                },
                t_wall=1000.0 + i * 10.0,
                action="learn",
            )
        h = tr.health()
        assert "nlm" in h
        assert "kernel_v" in h
        assert "cognitive_map" in h
        assert "sync_memory" in h
        g = tr.flow_guidance()
        assert "kernel_v" in g
        assert "cognitive_route" in g
