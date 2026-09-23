"""Capability flow / RankMe / dyn-decoupled D_T — arXiv:2607.27924 PT-Flow adapted."""
from __future__ import annotations

import math
import time

import pytest

from arsi.foundation.rankme import centered_effective_rank, feature_rows_from_dicts
from arsi.meta.env_difficulty import compute_env_difficulty, compute_env_difficulty_dyn
from arsi.world_model.capability_flow import (
    CapabilityFlowTracker,
    VelocityField,
    Z_KEYS,
    encode_dyn_state,
    encode_static_context,
    integrate,
    negative_organs,
    velocity_gt,
)
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import ReplayWorld
from arsi.world_model.world_pool import WorldPool


def _traces(n=12, agents=None, actions=None):
    agents = agents or ["hermes", "mimo"]
    actions = actions or ["learn", "remember", "dream", "empower"]
    out = []
    for i in range(n):
        ok = i % 3 != 0
        out.append({
            "agent_id": agents[i % len(agents)],
            "action": actions[i % len(actions)],
            "outcome": "success" if ok else "failure",
            "effect": 0.6 if ok else 0.2,
            "params": {"fail_class": "ok" if ok else "timeout"},
        })
    return out


def _world(traces, world_id="W"):
    tree = DiscoveryTree()
    tree.build_from_traces(traces)
    return ReplayWorld.from_discovery_tree(tree, world_id=world_id)


class TestRankMe:
    def test_diverse_rows_high_rank(self):
        rows = [
            [math.sin(i * 0.7), math.cos(i * 1.1), float(i % 5) - 2.0, float((i * 3) % 11) / 11.0]
            for i in range(40)
        ]
        r = centered_effective_rank(rows)
        assert r["effective_rank"] > 1.5
        assert r["collapse"] is False
        assert r["n"] == 40

    def test_identical_rows_collapse(self):
        rows = [[1.0, 2.0, 3.0] for _ in range(20)]
        r = centered_effective_rank(rows)
        assert r["collapse"] is True
        assert r["effective_rank"] < 3.0

    def test_empty(self):
        r = centered_effective_rank([])
        assert r["collapse"] is True
        assert r["effective_rank"] == 0.0

    def test_feature_rows(self):
        rows = feature_rows_from_dicts([{"a": 1, "b": 2}, {"a": 0, "b": 1}], ["a", "b"])
        assert rows == [[1.0, 2.0], [0.0, 1.0]]


class TestVelocityAndFlow:
    def test_velocity_gt_uses_wall_clock(self):
        z0 = [0.0, 0.0]
        z1 = [1.0, -2.0]
        v = velocity_gt(z0, z1, dt=2.0)
        assert v[0] == pytest.approx(0.5)
        assert v[1] == pytest.approx(-1.0)

    def test_field_learns_first_order(self):
        field = VelocityField(n_dims=len(Z_KEYS))
        # identity dynamics: v = z
        for i in range(40):
            z = [0.01 * i + 0.1 * k for k in range(len(Z_KEYS))]
            v_gt = list(z)
            field.fit_sample(z, v_gt)
        v_hat = field.predict_vec(z)
        # at least some dims track
        err = sum((a - b) ** 2 for a, b in zip(v_hat, v_gt)) / len(z)
        assert err < 1.0
        assert field.n_updates == 40

    def test_encode_separates_tracks(self):
        stats = {
            "eta": 0.2,
            "beta": 0.6,
            "world_pool_size": 5,
            "iwm": {"self_trust": 0.8, "memory_trust": 0.5, "layer1_holdout": 0.9},
            "env_evolution": {
                "difficulty": {"d_t_mean": 3.0, "d_t_spread": 1.0},
                "el": {"lineages": {"a": {"pass_rate": 0.5}, "b": {"pass_rate": 1.0}}},
            },
            "live_history": [0.5, 0.7],
            "pool_history": [0.1, 0.2],
            "grid_plan": {"branch_count": 3},
            "multi_agent": {"agents": {"h": {"tasks_completed": 3}, "m": {}}},
        }
        z = encode_dyn_state(stats)
        c = encode_static_context(stats)
        assert z["eta"] == 0.2
        assert z["memory_trust"] == 0.5
        assert z["live_last"] == 0.7
        assert z["pool_last"] == 0.2
        assert z["d_t_mean"] == 3.0
        assert z["el_pass_mean"] == 0.75
        assert "beta" not in z
        assert c["beta"] == 0.6
        assert c["pool_size"] == 5.0
        assert set(z.keys()) == set(Z_KEYS)

    def test_negative_organs(self):
        v = {k: 0.0 for k in Z_KEYS}
        v["memory_trust"] = -0.4
        v["self_trust"] = 0.1
        neg = negative_organs(v)
        assert "memory_trust" in neg
        assert "self_trust" not in neg

    def test_integrate_subgoal(self):
        field = VelocityField()
        z0 = {k: 0.1 for k in Z_KEYS}
        # constant positive velocity on self_trust
        field.b[Z_KEYS.index("self_trust")] = 0.01
        out = integrate(z0, field, horizon_s=100.0, steps=5)
        assert out["z_goal"]["self_trust"] > 0.1
        assert out["steps"] == 5

    def test_tracker_two_samples_produces_velocity(self):
        tr = CapabilityFlowTracker()
        base = {
            "eta": 0.1,
            "iwm": {"self_trust": 0.5, "memory_trust": 0.4},
            "live_history": [0.3],
            "pool_history": [0.1],
        }
        o1 = tr.observe(base, t_wall=1000.0, note="t1")
        assert o1.dt == 0.0
        assert o1.rankme
        base2 = {
            "eta": 0.2,
            "iwm": {"self_trust": 0.6, "memory_trust": 0.3},
            "live_history": [0.5],
            "pool_history": [0.2],
        }
        o2 = tr.observe(base2, t_wall=1010.0, note="t2")
        assert o2.dt == pytest.approx(10.0)
        assert o2.v_gt["self_trust"] == pytest.approx(0.01)  # Δ0.1 / 10s
        assert o2.v_gt["eta"] == pytest.approx(0.01)
        assert o2.v_gt["memory_trust"] == pytest.approx(-0.01)
        health = tr.health()
        assert health["n_samples"] == 2
        assert health["n_v_updates"] >= 1
        assert "self_trust" in health["last_z"]
        assert health["static_context"] is not None

    def test_tracker_rejects_too_fast_samples(self):
        tr = CapabilityFlowTracker(min_dt=5.0)
        s = {"eta": 0.0, "iwm": {"self_trust": 0.5}}
        tr.observe(s, t_wall=10.0)
        o = tr.observe(s, t_wall=11.0)  # dt=1 < 5
        assert o.dt == pytest.approx(1.0)
        assert tr.field.n_updates == 0  # no fake velocity

    def test_serialize_roundtrip(self):
        tr = CapabilityFlowTracker()
        tr.observe({"eta": 0.1, "iwm": {"self_trust": 0.4}}, t_wall=1.0)
        tr.observe({"eta": 0.2, "iwm": {"self_trust": 0.5}}, t_wall=20.0)
        data = tr.serialize()
        tr2 = CapabilityFlowTracker.from_serialize(data)
        assert tr2.field.n_updates == tr.field.n_updates
        assert len(tr2._samples) == len(tr._samples)


class TestDynDecoupledDifficulty:
    def test_dyn_difficulty_ignores_agent_name_relabel(self):
        w_a = _world(_traces(10, agents=["hermes", "mimo"]), "A")
        w_b = _world(_traces(10, agents=["EDGE_X1", "batch-9"]), "B")
        da = compute_env_difficulty_dyn(w_a)
        db = compute_env_difficulty_dyn(w_b)
        # dyn structure identical → dyn D_T should match (static rename must not change skill structure)
        assert da.skill_rarity == pytest.approx(db.skill_rarity, abs=0.35)
        assert da.note == "off_policy_env_difficulty_dyn_only"
        # full D_T (with agent novelty) may differ
        fa = compute_env_difficulty(w_a)
        fb = compute_env_difficulty(w_b)
        assert fa.note == "off_policy_env_difficulty"

    def test_pool_splits_static_dyn_and_rankme(self):
        pool = WorldPool(max_worlds=20)
        w = pool.append_from_traces(_traces(12), world_id="T0")
        assert getattr(w, "env_difficulty_dyn", {}).get("d_t", -1) >= 0
        assert w.static_context.get("agents")
        assert w.dyn_features.get("n_nodes", 0) >= 1
        assert w.dyn_row.get("d_t_dyn") is not None
        rk = pool.pool_dyn_rankme()
        assert "effective_rank" in rk
        health = pool.env_evolution_health()
        assert "pool_dyn_rankme" in health


class TestCoreFlowWiring:
    @pytest.fixture
    def arsi(self, tmp_path, monkeypatch):
        import yaml
        from arsi.core import ARSI
        from arsi.foundation import paths as arsi_paths

        arch = tmp_path / "archive"
        arch.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(arsi_paths, "archive_dir", lambda: arch)
        config = {
            "llm": {
                "provider": "openai",
                "model": "test",
                "api_key": "sk-test-disabled",
                "use_proxy": False,
                "fallback_to_heuristic": True,
                "timeout": 1,
            },
            "db_path": ":memory:",
            "iron_laws_path": str(tmp_path / "iron_laws.yaml"),
            "sealed_tasks_path": str(tmp_path / "sealed_tasks.yaml"),
        }
        (tmp_path / "iron_laws.yaml").write_text(
            "laws:\n  - id: G10\n    name: test\n    description: test\n    severity: block\n",
            encoding="utf-8",
        )
        config_path = tmp_path / "arsi.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        instance = ARSI.from_config(config_path)
        if instance.llm:
            instance.llm._client = None
        yield instance
        instance.close()

    def test_harvest_emits_capability_flow(self, arsi):
        for i in range(16):
            arsi.ingest_trace(
                agent_id=["hermes", "mimo"][i % 2],
                action=["learn", "remember", "empower"][i % 3],
                outcome="success",
                effect=0.6,
                params={"fail_class": "ok"},
            )
        harv = arsi.harvest_term_tree()
        assert harv.get("harvested") is True
        cf = harv.get("capability_flow") or {}
        assert "z" in cf or "v_gt" in cf
        assert "env_difficulty_dyn" in harv
        stats = arsi.get_stats()
        assert "capability_flow" in stats
        assert stats["capability_flow"].get("n_samples", 0) >= 1

    def test_dream_cycle_includes_flow(self, arsi):
        for i in range(16):
            arsi.ingest_trace(
                agent_id="hermes",
                action=["learn", "repair"][i % 2],
                outcome="success" if i % 2 == 0 else "failure",
                effect=0.65 if i % 2 == 0 else 0.2,
                params={"fail_class": "ok" if i % 2 == 0 else "timeout"},
            )
        result = arsi.dream_rsi_cycle(skip_llm=True)
        assert result.get("ran") is True
        assert "capability_flow" in result
        ee = result.get("env_evolution") or {}
        assert "pool_dyn_rankme" in ee or ee.get("pool_difficulty") is not None
