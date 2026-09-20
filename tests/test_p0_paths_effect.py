"""P0 paths + effect anchor tests (SYNTHEX S3/S4/S5)."""
from __future__ import annotations

import json
from pathlib import Path

from arsi.foundation.effect_anchor import (
    build_effect_anchor,
    is_external_source,
    record_external_effect,
)
from arsi.foundation.paths import (
    archive_dir,
    append_jsonl,
    config_dir,
    eval_dir,
    identity_report,
    iwm_dir,
    project_root,
    trace_pool_dir,
    write_json_once,
)
from arsi.meta.live_manifest import ManifestStore
from arsi.meta.eval_loop import run_eval_loop
from arsi.foundation.store import MnemosyneStore
from arsi.foundation.verified import VerifiedClaim


class TestPathsAnchor:
    def test_project_root_has_pyproject(self):
        root = project_root()
        assert (root / "pyproject.toml").exists() or (root / "src" / "arsi").exists()
        assert archive_dir() == root / "archive"
        assert trace_pool_dir() == root / "archive" / "trace_pool"
        assert eval_dir() == root / "archive" / "eval"
        assert iwm_dir() == root / "archive" / "iwm"
        assert config_dir() == root / "config"

    def test_manifest_store_default_anchored(self, monkeypatch, tmp_path):
        # default root must be project-anchored, not CWD
        store = ManifestStore()
        assert store.root == project_root() / "archive" / "trace_pool"
        # explicit override still works
        store2 = ManifestStore(tmp_path / "pool")
        assert store2.root == tmp_path / "pool"

    def test_write_json_once_tags_writer(self, tmp_path):
        p = write_json_once(tmp_path / "x.json", {"a": 1}, writer_id="unit_test")
        data = json.loads(p.read_text(encoding="utf-8"))
        assert data["a"] == 1
        assert data["_arsi_write"]["writer_id"] == "unit_test"

    def test_append_jsonl(self, tmp_path):
        p = append_jsonl(tmp_path / "log.jsonl", {"k": 1}, writer_id="unit")
        p = append_jsonl(tmp_path / "log.jsonl", {"k": 2}, writer_id="unit")
        lines = p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["_arsi_write"]["writer_id"] == "unit"

    def test_identity_report(self):
        r = identity_report()
        assert r["project_root"]
        assert r["pyproject_exists"] or r["src_arsi_exists"]


class TestEffectAnchor:
    def test_external_sources(self):
        assert is_external_source("eval_loop")
        assert is_external_source("eval_loop:compare.json")
        assert is_external_source("host_outcome")
        assert not is_external_source("placeholder_mutation")
        assert not is_external_source("")
        assert not is_external_source("self_eval")

    def test_placeholder_not_verified(self):
        a = build_effect_anchor("evolve_attempt", 0.5, source="placeholder_mutation")
        assert a.external is False
        assert a.claim is not None
        assert a.claim.verified is False

    def test_external_with_ref_verified(self, tmp_path):
        class FakeStore:
            def __init__(self):
                self.calls = []
            def record_effect(self, m, e, context=""):
                self.calls.append((m, e, context))

        s = FakeStore()
        a = record_external_effect(
            s, "dream_rsi_eval_loop", 0.09, source="eval_loop", evidence_ref=str(tmp_path / "c.json")
        )
        assert a.external is True
        assert a.claim.verified is True
        assert s.calls and s.calls[0][0] == "dream_rsi_eval_loop"

    def test_external_without_ref_not_verified(self):
        a = build_effect_anchor("m", 0.1, source="eval_loop", evidence_ref="")
        assert a.external is True
        assert a.claim.verified is False


class TestEvalLoopWritesAnchor:
    def test_eval_loop_notes_effect_anchor(self, tmp_path):
        from arsi.governor.portfolio_policy import PortfolioPolicy, build_policy_fn
        from arsi.meta.dream_rsi_params import DreamRSIParams

        class FakePool:
            size = 3
            def evaluate_policy_across_pool(self, fn, policy_name="p", max_rounds=5):
                return {
                    "available": True,
                    "avg_score": 0.2 if "dream" in policy_name else 0.1,
                    "avg_quality": 0.3,
                    "avg_probes": 5,
                    "per_world": [{"probes": 5, "rounds": 2, "score": 0.2}],
                }

        class FakeARSI:
            project_root = project_root()
            world_pool = FakePool()
            portfolio_policy = PortfolioPolicy(beta=0.6)
            _term_best_scores = [0.2, 0.21, 0.22]
            dream_rsi_params = DreamRSIParams()
            store = MnemosyneStore(str(tmp_path / "e.db"))
            manifest_store = ManifestStore(tmp_path / "pool")

        arsi = FakeARSI()
        res = run_eval_loop(arsi, params=arsi.dream_rsi_params, report_dir=tmp_path / "eval")
        assert "effect_anchor" in res.notes
        assert res.notes["effect_anchor"]["external"] is True
        assert Path(res.notes["report_path"]).exists()
        # write tag present
        data = json.loads(Path(res.notes["report_path"]).read_text(encoding="utf-8"))
        assert data.get("_arsi_write", {}).get("writer_id")
