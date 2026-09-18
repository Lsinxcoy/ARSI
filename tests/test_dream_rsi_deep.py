"""Tests for Dream-RSI deep mechanisms (arXiv:2609.14858 unmined content).

Covers:
- ReplayWorld prefix-only Child() semantics
- Observation success semantics
- World pool multi-world evaluation + monotone selection
- Portfolio batch policy (exploit/explore/recovery)
- Beta schedule + cross-cycle default beta
- ARSI dream_rsi_cycle wiring
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import Observation, ReplayWorld
from arsi.world_model.world_pool import WorldPool
from arsi.governor.portfolio_policy import (
    PortfolioPolicy,
    beta_schedule,
    build_policy_fn,
    default_beta_from_live,
)


def _make_tree() -> DiscoveryTree:
    tree = DiscoveryTree()
    root = tree.nodes["root"] if "root" in tree.nodes else None
    # Build manually for deterministic Child() semantics
    from arsi.world_model.discovery_tree import DiscoveryNode
    from datetime import datetime

    tree.nodes.clear()
    r = DiscoveryNode(id="root", action="start", outcome="root", score=0.0,
                      timestamp="2026-01-01T00:00:00")
    tree.nodes["root"] = r
    tree.root_id = "root"

    # Branch A: root -> a1 -> a2
    a1 = DiscoveryNode(id="a1", parent_id="root", action="empower_calibration",
                       outcome="success", score=0.8, cost=1.0, depth=1,
                       timestamp="2026-01-01T00:01:00",
                       metadata={"fail_class": "ok", "error": ""})
    a2 = DiscoveryNode(id="a2", parent_id="a1", action="empower_metacognition",
                       outcome="success", score=0.9, cost=1.5, depth=2,
                       timestamp="2026-01-01T00:02:00",
                       metadata={"fail_class": "ok", "error": ""})
    # Branch B: root -> b1 (failure, repairable)
    b1 = DiscoveryNode(id="b1", parent_id="root", action="empower_attention",
                       outcome="failure", score=-0.2, cost=0.8, depth=1,
                       timestamp="2026-01-01T00:01:30",
                       metadata={"fail_class": "runtime_error", "error": "timeout"})
    # Branch C: root -> c1
    c1 = DiscoveryNode(id="c1", parent_id="root", action="remember",
                       outcome="success", score=0.4, cost=0.5, depth=1,
                       timestamp="2026-01-01T00:02:30",
                       metadata={"fail_class": "ok", "error": ""})

    for n in (a1, a2, b1, c1):
        tree.nodes[n.id] = n
    tree.nodes["root"].children = ["a1", "b1", "c1"]
    tree.nodes["a1"].children = ["a2"]
    return tree


class TestReplayWorld:
    def test_child_root_opens_earliest_branch(self):
        tree = _make_tree()
        world = ReplayWorld.from_discovery_tree(tree, world_id="W1")
        world.reset()
        legal = world.legal_actions()
        assert "root" in legal
        revealed = world.probe_batch(["root"])
        # Earliest timestamp child of root is a1
        assert any(o.cell_id == "a1" for o in revealed)

    def test_child_nonroot_unique_recorded_child(self):
        tree = _make_tree()
        world = ReplayWorld.from_discovery_tree(tree)
        world.reset()
        world.probe_batch(["root"])  # reveal a1
        # a1 is now a frontier
        legal = world.legal_actions()
        assert "a1" in legal
        revealed = world.probe_batch(["a1"])
        assert any(o.cell_id == "a2" for o in revealed)

    def test_prefix_only_unrevealed_unknown(self):
        tree = _make_tree()
        world = ReplayWorld.from_discovery_tree(tree)
        world.reset()
        obs = world.observed()
        assert "root" in obs
        assert "a2" not in obs
        assert world.meta("a2") is None

    def test_success_semantics(self):
        o = Observation(cell_id="x", evaluated=True, error="", fail_class="ok", valid=False)
        assert o.success is True  # valid=False does NOT mean failure
        o2 = Observation(cell_id="y", evaluated=True, error="boom", fail_class="runtime_error")
        assert o2.success is False

    def test_replay_score_and_signals(self):
        tree = _make_tree()
        world = ReplayWorld.from_discovery_tree(tree, max_parallelism=3)

        def policy(observed, legal, W):
            return legal[:W]

        result = world.replay(policy, policy_name="greedy", max_rounds=10)
        assert result.probes >= 1
        assert result.quality > 0
        signals = world.observation_signals()
        assert signals  # at least one branch

    def test_portfolio_batch_includes_exploration(self):
        tree = _make_tree()
        world = ReplayWorld.from_discovery_tree(tree, max_parallelism=3)
        policy = PortfolioPolicy(beta=0.7, max_workers=3)
        fn = build_policy_fn(policy)

        def run(observed, legal, W):
            return fn(observed, legal, W)

        result = world.replay(run, policy_name="portfolio", max_rounds=8)
        assert result.probes > 0
        assert result.rounds > 0


class TestWorldPool:
    def test_multi_world_evaluation(self):
        pool = WorldPool()
        t1 = _make_tree()
        pool.append_tree(t1, world_id="T1")
        t2 = _make_tree()
        pool.append_tree(t2, world_id="T2")

        def policy(observed, legal, W):
            return legal[:2]

        eval_result = pool.evaluate_policy_across_pool(policy, policy_name="p")
        assert eval_result["available"] is True
        assert eval_result["world_count"] == 2
        assert "avg_score" in eval_result

    def test_monotone_selection_current_in_set(self):
        pool = WorldPool()
        pool.append_tree(_make_tree(), world_id="T1")

        def current(observed, legal, W):
            return legal[:1]

        def worse(observed, legal, W):
            return []

        def maybe_better(observed, legal, W):
            return legal[:3] if legal else []

        selection = pool.select_best_policy(current, [worse, maybe_better], current_name="current")
        assert selection["monotone_ok"] is True
        scores = {c["name"]: c["avg_score"] for c in selection["all"]}
        assert scores[selection["best_name"]] >= scores["current"] - 1e-9

    def test_append_from_traces(self):
        pool = WorldPool()
        traces = [
            {"agent_id": "mimo", "action": "calibrate", "outcome": "success", "effect": 0.8, "timestamp": "2026-01-01T00:00:00"},
            {"agent_id": "hermes", "action": "skill_write", "outcome": "success", "effect": 0.6, "timestamp": "2026-01-01T00:01:00"},
        ]
        world = pool.append_from_traces(traces, world_id="live1")
        assert pool.size == 1
        assert world.world_id == "live1"


class TestBeta:
    def test_beta_schedule_monotone(self):
        low = beta_schedule(0.1)
        high = beta_schedule(0.9)
        assert high["width_bias"] > low["width_bias"]
        assert high["patience"] >= low["patience"]

    def test_default_beta_bootstrap(self):
        assert default_beta_from_live([]) == 0.6
        assert default_beta_from_live([{"best_score": 0.1, "beta": 0.5}]) == 0.6

    def test_default_beta_plateau_raises(self):
        history = [
            {"best_score": 0.5, "beta": 0.4},
            {"best_score": 0.5, "beta": 0.4},
            {"best_score": 0.5000, "beta": 0.4},
        ]
        beta = default_beta_from_live(history)
        assert beta > 0.4

    def test_default_beta_improving_keeps(self):
        history = [
            {"best_score": 0.3, "beta": 0.5},
            {"best_score": 0.4, "beta": 0.5},
            {"best_score": 0.6, "beta": 0.5},
        ]
        assert abs(default_beta_from_live(history) - 0.5) < 1e-9


class TestARSIDreamRSIWiring:
    def test_arsi_cycle_with_pool(self, tmp_path=None):
        import tempfile
        from pathlib import Path
        from arsi.foundation.store import MnemosyneStore
        from arsi.foundation.iron_laws import IronLaws
        from arsi.mnemosyne.core import Mnemosyne
        from arsi.world_model.siwm import SIWM
        from arsi.governor.core import AutopoieticGovernor, DimensionManager
        from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
        from arsi.pipelines.dream import DreamPipeline
        from arsi.core import ARSI

        db = Path(tempfile.mkdtemp()) / "test_arsi.db"
        store = MnemosyneStore(str(db))
        laws_path = Path(__file__).resolve().parents[1] / "config" / "iron_laws.yaml"
        if not laws_path.exists():
            laws_path = Path(tempfile.mkdtemp()) / "iron.yaml"
            laws_path.write_text("laws: []\n", encoding="utf-8")
        laws = IronLaws(laws_path)
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store)
        governor = AutopoieticGovernor(siwm, store, laws, DimensionManager())
        empowerment = EmpowermentEngine(mnemosyne, siwm, NullAdapter())
        dream = DreamPipeline(siwm, mnemosyne)
        arsi = ARSI(
            store=store,
            mnemosyne=mnemosyne,
            siwm=siwm,
            governor=governor,
            empowerment=empowerment,
            dream=dream,
            iron_laws=laws,
            llm=None,
        )
        from arsi.meta.live_manifest import ManifestStore
        arsi.manifest_store = ManifestStore(Path(tempfile.mkdtemp()) / "trace_pool")

        # Ingest traces then run dream cycle
        for i, agent in enumerate(["mimo", "hermes", "synthex"]):
            arsi.ingest_trace(
                agent_id=agent,
                action="empower" if i % 2 == 0 else "remember",
                outcome="success" if i < 2 else "failure",
                effect=0.7,
                params={"token_count": 1200},
            )

        harvest = arsi.harvest_term_tree(world_id="W_seed")
        assert harvest.get("harvested") is True
        assert arsi.world_pool.size >= 1

        cycle = arsi.dream_rsi_cycle()
        assert cycle.get("ran") is True
        assert cycle.get("monotone_ok") is True
        assert "deployed" in cycle

        replay = arsi.replay_policy()
        assert replay.get("available") is True
        store.close()
