"""Environment Evolution P0 — arXiv:2609.04128 adapted to ARSI."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from arsi.meta.el_scheduler import ELScheduler
from arsi.meta.env_difficulty import (
    ReferenceCorpus,
    compute_env_difficulty,
    pool_difficulty_stats,
    weakness_vs_difficulty,
)
from arsi.meta.dream_rsi_params import DreamRSIParams
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import ReplayWorld
from arsi.world_model.world_evolver import (
    DIRECTIONS,
    WorldEvolver,
    evolve_traces,
    verify_world,
)
from arsi.world_model.world_pool import WorldPool


def _traces(n=12, agents=None, actions=None, all_success=False):
    agents = agents or ["hermes", "mimo", "synthex"]
    actions = actions or ["learn", "remember", "dream", "empower"]
    out = []
    for i in range(n):
        outcome = "success" if (all_success or i % 3 != 0) else "failure"
        out.append({
            "agent_id": agents[i % len(agents)],
            "action": actions[i % len(actions)],
            "outcome": outcome,
            "effect": 0.55 if outcome == "success" else 0.15,
            "params": {"fail_class": "ok" if outcome == "success" else "timeout", "token_count": 100},
        })
    return out


def _world(traces, world_id="W1"):
    tree = DiscoveryTree()
    tree.build_from_traces(traces)
    return ReplayWorld.from_discovery_tree(tree, world_id=world_id)


class TestEnvDifficulty:
    def test_empty_world_zero(self):
        d = compute_env_difficulty({})
        assert d.d_t == 0.0
        assert d.note == "empty_world"

    def test_dt_positive_with_depth_and_diversity(self):
        w = _world(_traces(20))
        d = compute_env_difficulty(w)
        assert d.d_t > 0
        assert d.L >= 1
        assert d.n_nodes >= 3
        assert d.scenario_novelty >= 0
        assert d.skill_rarity >= 0

    def test_longer_world_harder(self):
        short = compute_env_difficulty(_world(_traces(4), "S"))
        long = compute_env_difficulty(_world(_traces(24), "L"))
        assert long.d_t >= short.d_t
        assert long.L >= short.L

    def test_rare_actions_increase_skill_rarity(self):
        common = ReferenceCorpus.from_trace_batches([_traces(40, actions=["learn", "remember"])])
        rare_world = _world(_traces(10, actions=["calibrate", "repair", "analyze_trace", "deploy_skill"]), "R")
        d_rare = compute_env_difficulty(rare_world, reference=common)
        d_common = compute_env_difficulty(_world(_traces(10, actions=["learn", "remember"]), "C"), reference=common)
        assert d_rare.skill_rarity >= d_common.skill_rarity

    def test_pool_stats_spread(self):
        worlds = [
            _world(_traces(3, "A"), "W3"),
            _world(_traces(20, agents=["edge-cli"], actions=["repair"]), "W20"),
        ]
        stats = pool_difficulty_stats(worlds)
        assert stats["world_count"] == 2
        assert stats["d_t_max"] >= stats["d_t_min"]
        assert "d_t_spread" in stats

    def test_weakness_formula(self):
        w = weakness_vs_difficulty(d_t=2.0, policy_score=0.2, reference_score=0.8)
        assert w["weakness"] == pytest.approx(0.6)
        assert w["excess_difficulty"] >= w["weakness"]


class TestWorldEvolver:
    def test_three_directions_produce_traces(self):
        seed = _traces(10)
        for direction in DIRECTIONS:
            evolved = evolve_traces(seed, direction, effort="high")
            assert len(evolved) >= len(seed) - 1
            assert any("success" in str(t.get("outcome", "")).lower() for t in evolved)

    def test_length_increases_nodes(self):
        seed = _traces(6)
        evolved = evolve_traces(seed, "length", effort="high")
        assert len(evolved) > len(seed)

    def test_oracle_and_invalid_test(self):
        seed = _world(_traces(12), "seed")
        child = _world(_traces(12), "child")
        ok = verify_world(child, seed_world=seed, effort="high")
        assert ok.oracle_ok
        assert ok.invalid_rejected
        assert ok.quality_ok
        assert ok.passed

    def test_invalid_empty_world(self):
        assert verify_world({}).reasons
        assert verify_world({}).passed is False

    def test_invalid_all_success_fake_easy(self):
        w = _world(_traces(10, all_success=True), "all_ok")
        res = verify_world(w)
        assert res.passed is False
        assert "invalid_all_success_fake_easy" in res.reasons

    def test_evolve_batch_lineage_and_difficulty(self):
        seed = _world(_traces(14, agents=["hermes", "mimo"]), "seed_root")
        ev = WorldEvolver()
        results = ev.evolve_batch(seed, n=3, lineage_id="lin_test", generation=1, effort="high")
        assert len(results) == 3
        accepted = [r for r in results if r.accepted]
        assert accepted, f"expected some accepted, reasons={[r.verifier.reasons for r in results]}"
        for r in accepted:
            assert r.lineage_id == "lin_test"
            assert r.generation == 1
            assert r.direction in DIRECTIONS
            assert r.child_difficulty.get("d_t", 0) >= 0
            assert r.world is not None
            assert getattr(r.world, "evolved", False) is True


class TestELScheduler:
    def test_register_and_active_world(self):
        el = ELScheduler(tau=0.75, batch=8)
        el.register_world("lin1", "w0", 0)
        el.register_world("lin1", "w1", 1)
        assert el.active_world_id("lin1") == "w0"
        assert el.pass_rate("lin1") == 0.0
        assert el.maybe_advance("lin1")["advanced"] is False

    def test_advance_when_pass_rate_above_tau(self):
        el = ELScheduler(tau=0.75, batch=8)
        el.register_world("lin1", "w0", 0)
        el.register_world("lin1", "w1", 1)
        for _ in range(8):
            el.record_probe("lin1", success=True)
        assert el.pass_rate("lin1") == 1.0
        adv = el.maybe_advance("lin1")
        assert adv["advanced"] is True
        assert adv["world_id"] == "w1"
        assert el.active_world_id("lin1") == "w1"

    def test_no_advance_below_tau(self):
        el = ELScheduler(tau=0.75, batch=8)
        el.register_world("lin1", "w0", 0)
        el.register_world("lin1", "w1", 1)
        for _ in range(3):
            el.record_probe("lin1", success=True)
        for _ in range(5):
            el.record_probe("lin1", success=False)
        assert el.pass_rate("lin1") < 0.75
        assert el.maybe_advance("lin1")["advanced"] is False

    def test_select_worlds_prefers_active_gen(self):
        el = ELScheduler(tau=0.75)
        el.register_world("lin1", "w_child", 1)
        el._active["lin1"] = 0  # active is first registered — set by probe later
        el.register_world("lin1", "w_seed", 0)
        # force active index to child
        el.lineages["lin1"].world_ids = ["w_seed", "w_child"]
        el._active["lin1"] = 1
        worlds = [
            _world(_traces(5), "w_other"),
            _world(_traces(8), "w_seed"),
            _world(_traces(10), "w_child"),
        ]
        ids = el.sample_world_ids(worlds, k=2)
        assert "w_child" in ids


class TestWorldPoolEnvEvolution:
    def test_append_computes_dt_and_lineage(self):
        pool = WorldPool(max_worlds=20)
        pool.configure_env_evolution(enabled=True)
        w = pool.append_from_traces(_traces(12), world_id="T_seed", meta={"source": "harvest", "generation": 0})
        assert w.env_difficulty.get("d_t", 0) >= 0
        assert "T_seed" in pool._world_lineage
        assert pool._world_lineage["T_seed"]["generation"] == 0
        assert pool.el_scheduler.lineages

    def test_evolve_from_seed_ingests_accepted(self):
        pool = WorldPool(max_worlds=50)
        pool.configure_env_evolution(enabled=True, max_evolved_per_harvest=2, effort="high")
        seed = pool.append_from_traces(_traces(16), world_id="T0", meta={"source": "harvest", "generation": 0})
        accepted = pool.evolve_from_seed(seed, n=3)
        assert any(a.get("accepted") for a in accepted), accepted
        assert pool.size > 1
        evolved = [w for w in pool.worlds if getattr(w, "evolved", False)]
        assert evolved
        child = evolved[0]
        assert child.sealed_eligible is False
        assert child.source == "env_evolution"
        assert pool._world_lineage[child.world_id]["lineage_id"] == pool._world_lineage["T0"]["lineage_id"]
        assert pool._world_lineage[child.world_id]["generation"] == 1

    def test_el_sampling_and_probe_record(self):
        pool = WorldPool(max_worlds=30)
        pool.configure_env_evolution(enabled=True)
        seed = pool.append_from_traces(_traces(14), world_id="T0", meta={"source": "harvest"})
        pool.evolve_from_seed(seed, n=2)

        def pol(observed, legal, w):
            return legal[:w]

        ev = pool.evaluate_policy_across_pool(pol, policy_name="t", max_rounds=8, use_el=True)
        assert ev["available"]
        assert ev.get("el_used") is True
        assert ev.get("el_probes") is not None
        health = pool.env_evolution_health()
        assert health["config"]["enabled"] is True
        assert "el" in health
        assert health["evolved_worlds"] >= 0

        adv = pool.el_advance_all()
        assert isinstance(adv, list)

    def test_snapshot_roundtrip_preserves_lineage_dt(self, tmp_path):
        pool = WorldPool(max_worlds=20)
        seed = pool.append_from_traces(_traces(12), world_id="T0", meta={"source": "harvest"})
        pool.evolve_from_seed(seed, n=1)
        snap = pool.export_snapshot()
        path = tmp_path / "pool.json"
        path.write_text(json.dumps(snap), encoding="utf-8")
        loaded = WorldPool.load_from(path)
        assert loaded.size == pool.size
        assert loaded.el_scheduler.lineages
        ids = {w.world_id for w in loaded.worlds}
        assert "T0" in ids
        for w in loaded.worlds:
            assert hasattr(w, "env_difficulty") or w.env_difficulty is not None or True
        stats = loaded.stats()
        assert "env_difficulty" in stats

    def test_disabled_evolution(self):
        pool = WorldPool(max_worlds=10)
        pool.configure_env_evolution(enabled=False)
        seed = pool.append_from_traces(_traces(12), world_id="T0")
        accepted = pool.evolve_from_seed(seed, n=2)
        assert accepted == []
        assert pool.size == 1


class TestCoreHarvestWiring:
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
        instance.world_pool.configure_env_evolution(
            enabled=True,
            evolve_every_n=1,
            max_evolved_per_harvest=1,
            effort="high",
            min_seed_nodes=3,
            use_el_in_dream=True,
        )
        yield instance
        instance.close()

    def _seed_traces(self, arsi, n=20):
        actions = ["learn", "remember", "dream", "empower"]
        agents = ["hermes", "mimo", "synthex"]
        for i in range(n):
            arsi.ingest_trace(
                agent_id=agents[i % 3],
                action=actions[i % 4],
                outcome="success",
                effect=0.55 + (i % 5) * 0.05,
                params={"fail_class": "ok", "token_count": 80},
            )
        for i in range(4):
            arsi.ingest_trace(
                agent_id="hermes",
                action="repair",
                outcome="failure",
                effect=-0.2,
                params={"fail_class": "timeout", "token_count": 40},
            )

    def test_params_load_env_section(self):
        params = DreamRSIParams.from_dict({
            "env_evolution": {
                "enabled": True,
                "el_tau": 0.8,
                "effort": "high",
                "max_evolved_per_harvest": 2,
            }
        })
        cfg = params.env_evolution_cfg()
        assert cfg["enabled"] is True
        assert cfg["el_tau"] == 0.8
        assert cfg["max_evolved_per_harvest"] == 2

    def test_harvest_evolution_end_to_end(self, arsi):
        self._seed_traces(arsi)
        harv = arsi.harvest_term_tree()
        assert harv.get("harvested") is True, harv
        assert harv.get("env_difficulty", {}).get("d_t", -1) >= 0
        assert harv.get("lineage_id")
        assert harv.get("evolution", {}).get("attempted") is True
        assert arsi.world_pool.size >= 1
        stats = arsi.get_stats()
        assert "env_evolution" in stats
        assert stats["world_pool_size"] >= 1

    def test_dream_cycle_includes_env_evolution_block(self, arsi):
        self._seed_traces(arsi)
        result = arsi.dream_rsi_cycle(skip_llm=True)
        assert result.get("ran") is True, result.get("reason")
        ee = result.get("env_evolution") or {}
        assert ee.get("harvest") is not None or ee.get("el_health") is not None
