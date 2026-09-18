"""Phase D tests — live manifest, grid plan, beta sweep, brief §5.1, ARSI wiring."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arsi.meta.grid_plan import GridPlan, GridPlanningContext, plan_grid
from arsi.meta.live_manifest import LiveCycleManifest, ManifestStore
from arsi.meta.beta_sweep import sweep_beta
from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn, default_beta_from_live
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.world_pool import WorldPool


def _tree(score=0.8, tag="a"):
    from arsi.world_model.discovery_tree import DiscoveryNode
    t = DiscoveryTree()
    t.nodes.clear()
    t.root_id = "root"
    t.nodes["root"] = DiscoveryNode(id="root", action="start", outcome="root", score=0.0)
    n1 = DiscoveryNode(
        id=f"{tag}1", parent_id="root", action=f"act_{tag}",
        outcome="success", score=score, cost=1.0,
        metadata={"fail_class": "ok", "error": ""},
    )
    t.nodes[n1.id] = n1
    t.nodes["root"].children = [n1.id]
    return t


def _pool_with_worlds(n=2):
    pool = WorldPool()
    for i in range(n):
        pool.append_tree(_tree(score=0.5 + i * 0.1, tag=f"w{i}"), world_id=f"T{i+1}")
    return pool


class TestManifestStore:
    def test_append_and_reload(self, tmp_path):
        store = ManifestStore(tmp_path / "trace_pool")
        m1 = store.next_manifest(kind="dream_rsi_cycle", best_score=0.5, beta=0.6)
        store.append(m1)
        m2 = store.next_manifest(kind="run_term", best_score=0.6, beta=0.6)
        store.append(m2)
        assert store.size == 2

        store2 = ManifestStore(tmp_path / "trace_pool")
        assert store2.size == 2
        hist = store2.beta_history(3)
        assert len(hist) == 2
        assert hist[-1]["best_score"] == 0.6

    def test_corrupt_json_skipped(self, tmp_path):
        store = ManifestStore(tmp_path / "tp")
        store.append(store.next_manifest(best_score=0.4))
        bad = tmp_path / "tp" / "iter0099" / "live_cycle_manifest.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{not json", encoding="utf-8")
        store2 = ManifestStore(tmp_path / "tp")
        assert store2.size == 1

    def test_beta_sweep_roundtrip(self, tmp_path):
        store = ManifestStore(tmp_path / "tp")
        m = store.next_manifest(best_score=0.3)
        store.append(m)
        store.save_beta_sweep(m.cycle_id, {"grid": [0.4, 0.6], "selected_default_beta": 0.6})
        data = store.load_beta_sweep(m.cycle_id)
        assert data["selected_default_beta"] == 0.6
        assert len(store.recent_beta_sweeps(3)) == 1


class TestPlanGrid:
    def test_bootstrap_insufficient(self):
        plan = plan_grid(GridPlanningContext(history=[]))
        assert plan.branch_count == 2
        assert plan.refine_count == 3
        assert "bootstrap" in plan.reason

    def test_r1_width_useful_depth_stall(self):
        hist = [
            {"best_score": 0.1, "effective_w": 2, "effective_r": 4, "probe_work": 10},
            {"best_score": 0.12, "effective_w": 2, "effective_r": 4, "probe_work": 10},
            {"best_score": 0.13, "effective_w": 2, "effective_r": 4, "probe_work": 10},
        ]
        plan = plan_grid(GridPlanningContext(history=hist, budget_force=1.0, cost_budget=20.0))
        assert plan.branch_count >= 2
        assert "R1" in plan.reason
        assert plan.refine_count in (2, 3, 4)

    def test_r2_late_gain_deepen(self):
        hist = [
            {"best_score": 0.2, "effective_w": 3, "effective_r": 2, "probe_work": 8},
            {"best_score": 0.21, "effective_w": 3, "effective_r": 2, "probe_work": 8},
            {"best_score": 0.45, "effective_w": 3, "effective_r": 4, "probe_work": 12},
        ]
        plan = plan_grid(GridPlanningContext(history=hist, budget_force=1.0))
        assert plan.reason.startswith("R2") or plan.refine_count >= 4

    def test_budget_clamp(self):
        hist = [
            {"best_score": 0.5, "effective_w": 4, "effective_r": 6, "probe_work": 20},
            {"best_score": 0.51, "effective_w": 4, "effective_r": 6, "probe_work": 20},
            {"best_score": 0.52, "effective_w": 4, "effective_r": 6, "probe_work": 20},
        ]
        plan = plan_grid(GridPlanningContext(history=hist, cost_budget=4.0, budget_force=1.0))
        assert plan.branch_count * plan.refine_count <= 4.0 + 1e-6

    def test_hard_max_respected(self):
        hist = [
            {"best_score": 0.1, "effective_w": 4, "effective_r": 8, "probe_work": 30},
            {"best_score": 0.3, "effective_w": 4, "effective_r": 8, "probe_work": 30},
        ]
        plan = plan_grid(GridPlanningContext(history=hist, hard_max_branch_count=4, hard_max_refine_count=8, budget_force=1.2))
        assert plan.branch_count <= 4
        assert plan.refine_count <= 8


class TestBetaSweep:
    def test_empty_pool(self):
        pool = WorldPool()
        result = sweep_beta(pool, lambda b: build_policy_fn(PortfolioPolicy(beta=b)))
        assert result.reason == "empty_pool"
        assert result.selected_default_beta == 0.6

    def test_sweep_on_pool(self):
        pool = _pool_with_worlds(2)
        result = sweep_beta(
            pool,
            lambda b: build_policy_fn(PortfolioPolicy(beta=b, max_workers=3)),
            grid=[0.2, 0.6, 1.0],
            live_history=[{"best_score": 0.4, "beta": 0.6}, {"best_score": 0.4, "beta": 0.6}],
        )
        assert result.world_count == 2
        assert len(result.points) == 3
        assert 0.0 <= result.selected_default_beta <= 1.0
        d = result.to_dict()
        assert "points" in d and d["reason"]

    def test_default_beta_reads_disk(self, tmp_path, monkeypatch):
        store = ManifestStore(tmp_path / "trace_pool")
        store.append(store.next_manifest(best_score=0.5, beta=0.4))
        store.append(store.next_manifest(best_score=0.5, beta=0.4))
        store.append(store.next_manifest(best_score=0.5000, beta=0.4))
        # Point default_beta_from_live's ManifestStore at tmp by patching class
        import arsi.governor.portfolio_policy as pp
        monkeypatch.setattr(pp, "ManifestStore", lambda *a, **k: ManifestStore(tmp_path / "trace_pool"), raising=False)
        # default_beta_from_live imports ManifestStore inside function from arsi.meta
        import arsi.meta.live_manifest as lm
        monkeypatch.setattr(lm, "ManifestStore", lambda *a, **k: ManifestStore(tmp_path / "trace_pool"))
        beta = default_beta_from_live([])
        assert beta > 0.4  # plateau → raise


class TestBriefSemantics:
    def test_structured_hides_recommendations(self):
        from arsi.adapters.bidirectional_interface import ARSIBrief
        b = ARSIBrief(
            task_description="fix parser",
            agent_id="hermes",
            recommendations=["先跑测试再改代码"],
            relevant_skills=["arsi-calibration"],
            warnings=["budget_high"],
            past_lessons=["[mimo] 之前失败过"],
            confidence=0.8,
            history_simulator={
                "available": True,
                "success_rate": 0.7,
                "sample_size": 10,
                "common_failure_patterns": ["timeout"],
                "score_stats": {"avg": 0.6, "best": 0.9},
            },
            policy="structured",
        )
        text = b.format_for_agent()
        assert "Recommendations" not in text
        assert "Past Lessons" not in text
        assert "arsi-calibration" in text
        assert "success_rate" in text
        assert b.to_dict()["meta_only"]["recommendations"]

    def test_full_policy_keeps_recommendations(self):
        from arsi.adapters.bidirectional_interface import ARSIBrief
        b = ARSIBrief(
            task_description="t",
            agent_id="a",
            recommendations=["do X"],
            relevant_skills=[],
            warnings=[],
            past_lessons=[],
            policy="full",
        )
        text = b.format_for_agent()
        assert "Recommendations" in text
        assert "do X" in text

    def test_query_history_no_best_practice(self):
        from arsi.adapters.bidirectional_interface import ARSIBrief
        b = ARSIBrief(
            task_description="t",
            agent_id="a",
            recommendations=[],
            relevant_skills=[],
            warnings=[],
            past_lessons=[],
            history_simulator={
                "available": True,
                "success_rate": 0.5,
                "sample_size": 4,
                "common_failure_patterns": ["x"],
                "score_stats": {"avg": 0.4},
            },
        )
        q = b.query_history("code")
        assert "best_practice" not in q
        assert "suggested_direction" not in q
        assert q["available"] is True


class TestARSIPhaseDIntegration:
    def _make_arsi(self):
        import tempfile
        from arsi.foundation.store import MnemosyneStore
        from arsi.foundation.iron_laws import IronLaws
        from arsi.mnemosyne.core import Mnemosyne
        from arsi.world_model.siwm import SIWM
        from arsi.governor.core import AutopoieticGovernor, DimensionManager
        from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
        from arsi.pipelines.dream import DreamPipeline
        from arsi.core import ARSI
        from arsi.meta.live_manifest import ManifestStore

        db = Path(tempfile.mkdtemp()) / "t.db"
        store = MnemosyneStore(str(db))
        laws_path = Path(__file__).resolve().parents[1] / "config" / "iron_laws.yaml"
        if not laws_path.exists():
            p = Path(tempfile.mkdtemp()) / "iron.yaml"
            p.write_text("laws: []\n", encoding="utf-8")
            laws_path = p
        laws = IronLaws(laws_path)
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store)
        gov = AutopoieticGovernor(siwm, store, laws, DimensionManager())
        arsi = ARSI(
            store=store,
            mnemosyne=mnemosyne,
            siwm=siwm,
            governor=gov,
            empowerment=EmpowermentEngine(mnemosyne, siwm, NullAdapter()),
            dream=DreamPipeline(siwm, mnemosyne),
            iron_laws=laws,
            llm=None,
        )
        # Isolate manifest store to temp dir
        arsi.manifest_store = ManifestStore(Path(tempfile.mkdtemp()) / "trace_pool")
        arsi.world_pool = WorldPool()
        return arsi, store

    def test_dream_cycle_writes_manifest_and_sweep(self):
        arsi, store = self._make_arsi()
        for i, agent in enumerate(["mimo", "hermes", "synthex"]):
            arsi.ingest_trace(
                agent_id=agent,
                action="empower" if i % 2 == 0 else "remember",
                outcome="success" if i < 2 else "failure",
                effect=0.7,
                params={"token_count": 800},
            )
        cycle = arsi.dream_rsi_cycle()
        assert cycle.get("ran") is True
        assert cycle.get("manifest_cycle", 0) >= 1
        assert "grid_plan" in cycle
        assert "beta_sweep" in cycle
        assert Path(cycle["manifest_path"]).exists()
        assert Path(cycle["beta_sweep_path"]).exists()
        # reload
        hist = arsi.manifest_store.beta_history(5)
        assert hist and "beta" in hist[-1]
        store.close()

    def test_harvest_quality_gate_stats(self):
        arsi, store = self._make_arsi()
        arsi.ingest_trace(agent_id="mimo", action="ok", outcome="success", effect=0.9)
        arsi.ingest_trace(agent_id="bad", action="x", outcome="failure", effect=-0.5)
        # out of range effect should fail gate1 on raw dict path; ingest may clamp
        h = arsi.harvest_term_tree(world_id="W1")
        assert h.get("harvested") is True
        assert "quality_gate" in h
        assert arsi.world_pool.size >= 1
        store.close()

    def test_plan_next_grid_bootstrap_then_history(self):
        arsi, store = self._make_arsi()
        p1 = arsi.plan_next_grid()
        assert p1.branch_count >= 1
        assert "bootstrap" in p1.reason or p1.reason
        # inject history via manifests
        for i in range(3):
            m = arsi.manifest_store.next_manifest(
                best_score=0.2 + i * 0.01,
                beta=0.6,
                effective_grid=None,
            )
            m.effective_grid.branch_count = 2
            m.effective_grid.refine_count = 4
            arsi.manifest_store.append(m)
        arsi._live_cycle_history = arsi.manifest_store.beta_history(5)
        p2 = arsi.plan_next_grid()
        assert isinstance(p2, GridPlan)
        assert p2.branch_count <= 4
        store.close()

    def test_brief_structured_via_interface(self):
        arsi, store = self._make_arsi()
        arsi.discovery_tree.build_from_traces([
            {"agent_id": "mimo", "action": "fix parser", "outcome": "success", "effect": 0.8},
            {"agent_id": "mimo", "action": "fix parser again", "outcome": "failure", "effect": 0.1},
            {"agent_id": "hermes", "action": "parser skill", "outcome": "success", "effect": 0.6},
        ])
        from arsi.adapters.bidirectional_interface import ARSIInterface
        iface = ARSIInterface(arsi, brief_policy="structured")
        brief = iface.brief("parser bug", "hermes")
        text = brief.format_for_agent()
        assert "Recommendations" not in text
        assert brief.policy == "structured"
        store.close()
