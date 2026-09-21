"""Phase D residual tests — D7 eval loop, hyperparams, CLI helpers."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arsi.meta.dream_rsi_params import DreamRSIParams, load_dream_rsi_params
from arsi.meta.eval_loop import detect_live_regression, fixed_exploration_fn, run_eval_loop
from arsi.meta.live_manifest import ManifestStore
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.world_pool import WorldPool
from arsi.world_model.discovery_tree import DiscoveryNode


def _tree(score, tag):
    t = DiscoveryTree()
    t.nodes.clear()
    t.root_id = "root"
    t.nodes["root"] = DiscoveryNode(id="root", action="start", outcome="root", score=0.0)
    n = DiscoveryNode(
        id=f"{tag}", parent_id="root", action=f"a_{tag}",
        outcome="success", score=score, cost=1.0,
        metadata={"fail_class": "ok", "error": ""},
    )
    t.nodes[n.id] = n
    t.nodes["root"].children = [n.id]
    return t


class TestDreamRSIParams:
    def test_load_defaults_when_missing(self, tmp_path):
        p = load_dream_rsi_params(tmp_path / "nope.yaml")
        assert p.beta1_cost_penalty == 0.1
        assert p.beta_default_uncertain == 0.6
        assert p.official_code_status == "not_released"

    def test_load_from_project_config(self):
        cfg = Path(__file__).resolve().parents[1] / "config" / "dream_rsi_params.yaml"
        if not cfg.exists():
            return
        p = load_dream_rsi_params(cfg)
        assert p.beta_sweep_grid
        assert p.hard_max_branch_count >= 2
        assert p.rollback_window >= 1
        d = p.to_dict()
        assert "official_code_status" in d

    def test_from_dict_overrides(self):
        p = DreamRSIParams.from_dict({
            "replay_objective": {"beta1_cost_penalty": 0.2},
            "beta": {"default_uncertain": 0.4, "sweep_grid": [0.3, 0.7]},
            "eval_loop": {"rollback_window": 2, "rollback_delta_eps": 0.05},
        })
        assert p.beta1_cost_penalty == 0.2
        assert p.beta_default_uncertain == 0.4
        assert p.beta_sweep_grid == [0.3, 0.7]
        assert p.rollback_window == 2


class TestEvalLoop:
    def test_detect_regression_insufficient(self):
        ok, reason = detect_live_regression([0.1, 0.2], window=3)
        assert ok is False
        assert "insufficient" in reason

    def test_detect_regression_window_drop(self):
        scores = [0.5, 0.5, 0.5, 0.45, 0.40, 0.35]
        ok, reason = detect_live_regression(scores, window=3, eps=0.02)
        assert ok is True

    def test_detect_no_regression_when_improving(self):
        scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ok, _ = detect_live_regression(scores, window=3, eps=0.02)
        assert ok is False

    def test_run_eval_loop_and_rollback(self, tmp_path):
        from arsi.foundation.store import MnemosyneStore
        from arsi.foundation.iron_laws import IronLaws
        from arsi.mnemosyne.core import Mnemosyne
        from arsi.world_model.siwm import SIWM
        from arsi.governor.core import AutopoieticGovernor, DimensionManager
        from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
        from arsi.pipelines.dream import DreamPipeline
        from arsi.core import ARSI
        from arsi.governor.portfolio_policy import PortfolioPolicy

        db = Path(tempfile.mkdtemp()) / "t.db"
        store = MnemosyneStore(str(db))
        laws_path = Path(__file__).resolve().parents[1] / "config" / "iron_laws.yaml"
        if not laws_path.exists():
            lp = Path(tempfile.mkdtemp()) / "iron.yaml"
            lp.write_text("laws: []\n", encoding="utf-8")
            laws_path = lp
        laws = IronLaws(laws_path)
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store)
        gov = AutopoieticGovernor(siwm, store, laws, DimensionManager())
        arsi = ARSI(
            store=store, mnemosyne=mnemosyne, siwm=siwm, governor=gov,
            empowerment=EmpowermentEngine(mnemosyne, siwm, NullAdapter()),
            dream=DreamPipeline(siwm, mnemosyne), iron_laws=laws, llm=None,
        )
        arsi.manifest_store = ManifestStore(Path(tempfile.mkdtemp()) / "tp")
        arsi.world_pool = WorldPool()
        arsi.world_pool.append_tree(_tree(0.8, "w1"), world_id="T1")
        arsi.world_pool.append_tree(_tree(0.6, "w2"), world_id="T2")
        arsi.portfolio_policy = PortfolioPolicy(beta=0.6, max_workers=2)
        # simulate live capability regression (D7 track only after full-repair)
        arsi._live_capability_scores = [0.5, 0.5, 0.5, 0.40, 0.35, 0.30]

        result = run_eval_loop(arsi, params=arsi.dream_rsi_params, report_dir=tmp_path / "eval")
        assert result.world_count == 2
        assert "fixed_avg_score" in result.to_dict()
        assert result.live_regression is True
        assert result.rollback_triggered is True
        assert arsi.portfolio_policy.name.startswith("rollback_beta_")
        assert (tmp_path / "eval").exists()
        reports = list((tmp_path / "eval").glob("compare_*.json"))
        assert reports
        store.close()

    def test_fixed_fn_shape(self):
        fn = fixed_exploration_fn(max_workers=2)
        batch = fn({}, ["a", "b", "c", "d"], 4)
        assert len(batch) == 2
        assert batch == ["a", "b"]


class TestARSIDreamCycleIncludesEvalLoop:
    def test_dream_cycle_has_eval_and_params(self):
        import tempfile
        from arsi.foundation.store import MnemosyneStore
        from arsi.foundation.iron_laws import IronLaws
        from arsi.mnemosyne.core import Mnemosyne
        from arsi.world_model.siwm import SIWM
        from arsi.governor.core import AutopoieticGovernor, DimensionManager
        from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
        from arsi.pipelines.dream import DreamPipeline
        from arsi.core import ARSI

        db = Path(tempfile.mkdtemp()) / "t.db"
        store = MnemosyneStore(str(db))
        laws_path = Path(__file__).resolve().parents[1] / "config" / "iron_laws.yaml"
        if not laws_path.exists():
            lp = Path(tempfile.mkdtemp()) / "iron.yaml"
            lp.write_text("laws: []\n", encoding="utf-8")
            laws_path = lp
        laws = IronLaws(laws_path)
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store)
        arsi = ARSI(
            store=store, mnemosyne=mnemosyne, siwm=siwm,
            governor=AutopoieticGovernor(siwm, store, laws, DimensionManager()),
            empowerment=EmpowermentEngine(mnemosyne, siwm, NullAdapter()),
            dream=DreamPipeline(siwm, mnemosyne), iron_laws=laws, llm=None,
        )
        arsi.manifest_store = ManifestStore(Path(tempfile.mkdtemp()) / "tp")
        arsi.world_pool = WorldPool()
        arsi.ingest_trace("mimo", "learn", "success", 0.8)
        arsi.ingest_trace("hermes", "skill", "success", 0.7)
        cycle = arsi.dream_rsi_cycle()
        assert cycle.get("ran") is True
        assert "eval_loop" in cycle
        assert "params" in cycle
        assert cycle["params"]["official_code_status"]
        stats = arsi.get_stats()
        assert "dream_rsi_params" in stats
        assert "world_pool_size" in stats
        store.close()
