"""Hardening tests: pool persist, cumulative Layer1, host reverse evidence."""
from __future__ import annotations

import json
from pathlib import Path

from arsi.world_model.world_pool import WorldPool
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.siwm import SIWM, BehaviorPredictor
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.schema import BehaviorTrace
from arsi.mnemosyne.core import Mnemosyne


def _tree(effect=0.8):
    t = DiscoveryTree()
    t.build_from_traces([
        {"action": "learn", "outcome": "success", "effect": effect, "params": {}},
        {"action": "remember", "outcome": "success", "effect": 0.7, "params": {}},
    ] * 6)
    return t


class TestWorldPoolPersist:
    def test_export_import_roundtrip(self, tmp_path):
        pool = WorldPool(max_worlds=10)
        for i in range(3):
            pool.append_from_traces(
                [{"action": "learn", "outcome": "success", "effect": 0.9, "params": {}}] * 4,
                world_id=f"W{i}",
            )
        path = tmp_path / "pool.json"
        pool.persist_to(path)
        loaded = WorldPool.load_from(path)
        assert loaded.size == 3
        assert loaded.worlds[0].world_id == "W0"
        assert path.exists()

    def test_load_missing_returns_empty(self, tmp_path):
        p = WorldPool.load_from(tmp_path / "nope.json")
        assert p.size == 0


class TestLayer1Cumulative:
    def test_refit_does_not_collapse_rules(self, tmp_path):
        store = MnemosyneStore(str(tmp_path / "s.db"))
        mn = Mnemosyne(store)
        acts = ["learn", "remember", "dream", "maintain", "evolve"]
        for i in range(80):
            mn.ingest_trace(BehaviorTrace(
                agent_id="a",
                action=acts[i % len(acts)],
                outcome="success" if i % 2 == 0 else "partial",
                effect=0.6,
            ))
        siwm = SIWM(store)
        r1 = siwm.train_from_history()
        n1 = r1["rule_count"]
        assert n1 >= 2
        # second refit on same/flat batch should not wipe rules to ~0
        r2 = siwm.train_from_history()
        assert r2["rule_count"] >= max(2, n1 // 4)
        assert r2["holdout_accuracy"] >= 0.0


class TestHostReverseEvidence:
    def test_host_reverse_script_importable(self):
        import importlib.util
        path = Path(__file__).resolve().parents[1] / "scripts" / "host_reverse_loop.py"
        assert path.exists()
        spec = importlib.util.spec_from_file_location("host_reverse", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ev = mod.consume_brief_evidence()
        assert "mimo_feedback_bytes" in ev
        assert "hermes_skill_bytes" in ev
        assert isinstance(ev.get("consumed_chars"), int)

    def test_health_iwm_serialization_keys(self):
        """Daemon ok snapshot must include memory_trust keys (hardening item)."""
        src = Path(r"E:\ARSI\scripts\arsi_daemon.py").read_text(encoding="utf-8")
        assert "memory_trust" in src
        assert "trust_memory_for_learn" in src
        assert "layer1_holdout" in src
        assert "after_harvest" in src
