"""World Pool — Dream-RSI history H_t = (T_1, ..., T_t).

Every online term/rollout appends one completed discovery tree as a
replay world. Policy improvement dreams across ALL worlds, not just
the latest one (arXiv:2609.14858 §3).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import ReplayResult, ReplayWorld

logger = logging.getLogger(__name__)


class WorldPool:
    """Growing pool of frozen replay worlds (discovery trees)."""

    def __init__(self, max_worlds: int = 50):
        self.max_worlds = max_worlds
        self.worlds: list[ReplayWorld] = []
        self._manifest: list[dict] = []

    @property
    def size(self) -> int:
        return len(self.worlds)

    def append_tree(
        self,
        tree: DiscoveryTree,
        world_id: Optional[str] = None,
        max_parallelism: int = 3,
        meta: Optional[dict] = None,
    ) -> ReplayWorld:
        """Append a completed discovery tree as a new replay world."""
        wid = world_id or f"T{self.size + 1}"
        world = ReplayWorld.from_discovery_tree(tree, world_id=wid, max_parallelism=max_parallelism)
        self.worlds.append(world)
        entry = {
            "world_id": wid,
            "node_count": len(world._full),
            "baseline_score": world.baseline_score,
            "max_parallelism": world.max_parallelism,
            "meta": meta or {},
        }
        self._manifest.append(entry)
        if len(self.worlds) > self.max_worlds:
            self.worlds = self.worlds[-self.max_worlds :]
            self._manifest = self._manifest[-self.max_worlds :]
        logger.info(f"World pool += {wid} (size={self.size})")
        return world

    def append_from_traces(self, traces: list[dict], world_id: Optional[str] = None, **kwargs) -> ReplayWorld:
        tree = DiscoveryTree()
        tree.build_from_traces(traces)
        return self.append_tree(tree, world_id=world_id, **kwargs)

    def evaluate_policy_across_pool(self, policy_fn, policy_name: str = "policy", max_rounds: int = 12) -> dict:
        """Dream-RSI multi-world evaluation: score policy on EVERY historical world.

        max_rounds default 12 — cost control; worlds still get probe diversity.
        """
        if not self.worlds:
            return {"available": False, "reason": "empty_pool", "avg_score": 0.0, "results": []}

        results: list[ReplayResult] = []
        for world in self.worlds:
            result = world.replay(policy_fn, policy_name=policy_name, max_rounds=max_rounds)
            results.append(result)

        avg_score = sum(r.replay_score for r in results) / len(results)
        avg_quality = sum(r.quality for r in results) / len(results)
        avg_probes = sum(r.probes for r in results) / len(results)
        # pool diversity proxy: unique node counts / fail mixes
        node_counts = [len(w._full) for w in self.worlds]
        return {
            "available": True,
            "policy_name": policy_name,
            "world_count": len(results),
            "avg_score": round(avg_score, 4),
            "avg_quality": round(avg_quality, 4),
            "avg_probes": round(avg_probes, 4),
            "pool_diversity": {
                "node_count_min": min(node_counts) if node_counts else 0,
                "node_count_max": max(node_counts) if node_counts else 0,
                "node_count_mean": round(sum(node_counts) / len(node_counts), 2) if node_counts else 0,
            },
            "track": "pool_replay",
            "max_rounds": max_rounds,
            "per_world": [
                {
                    "world_id": r.world_id,
                    "score": r.replay_score,
                    "quality": r.quality,
                    "probes": r.probes,
                    "rounds": r.rounds,
                    "anchors": r.successful_anchors,
                    "repairables": r.repairables,
                    "score_breakdown": getattr(r, "score_breakdown", {}),
                    "node_count": len(next((w._full for w in self.worlds if w.world_id == r.world_id), {}) or {}),
                }
                for r in results
            ],
            "score_mode": getattr(self.worlds[0], "score_mode", None) if self.worlds else None,
        }

    def select_best_policy(
        self,
        current_policy_fn,
        candidate_policy_fns: list,
        current_name: str = "current",
        candidate_names: Optional[list[str]] = None,
        use_paired_ab: bool = True,
    ) -> dict:
        """S6: paired A/B + effect-size gate when enabled; else max avg_score."""
        if use_paired_ab:
            from arsi.meta.paired_ab import select_policy_paired
            return select_policy_paired(
                self,
                current_policy_fn,
                candidate_policy_fns,
                current_name=current_name,
                candidate_names=candidate_names,
            )

        candidates = [(current_name, current_policy_fn)]
        names = candidate_names or [f"cand_{i+1}" for i in range(len(candidate_policy_fns))]
        for name, fn in zip(names, candidate_policy_fns):
            candidates.append((name, fn))

        scored = []
        for name, fn in candidates:
            eval_result = self.evaluate_policy_across_pool(fn, policy_name=name)
            scored.append((name, fn, eval_result))

        best_name, best_fn, best_eval = max(scored, key=lambda x: x[2].get("avg_score", -1e9))
        return {
            "best_name": best_name,
            "best_fn": best_fn,
            "best_eval": best_eval,
            "all": [
                {"name": n, "avg_score": e.get("avg_score", 0.0), "world_count": e.get("world_count", 0)}
                for n, _, e in scored
            ],
            "monotone_ok": best_eval.get("avg_score", -1e9) >= scored[0][2].get("avg_score", -1e9),
            "gate": {"rule": "max_avg_score_fallback"},
        }

    def stats(self) -> dict:
        return {
            "world_count": self.size,
            "total_nodes": sum(len(w._full) for w in self.worlds),
            "baselines": [w.baseline_score for w in self.worlds],
            "manifest": self._manifest[-10:],
        }

    def save_manifest(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"manifest": self._manifest, "stats": self.stats()}, ensure_ascii=False, indent=2), encoding="utf-8")

    def export_snapshot(self) -> dict:
        """Serialize all worlds for crash-safe pool accumulation."""
        worlds = []
        for w in self.worlds:
            worlds.append({
                "world_id": w.world_id,
                "nodes": w._full,
                "baseline_score": w.baseline_score,
                "max_parallelism": w.max_parallelism,
                "beta1": getattr(w, "beta1", 0.1),
                "beta2": getattr(w, "beta2", 0.05),
                "score_mode": getattr(w, "score_mode", "quality_anchored"),
            })
        return {
            "schema": "arsi.world_pool.v1",
            "max_worlds": self.max_worlds,
            "manifest": self._manifest,
            "worlds": worlds,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "WorldPool":
        pool = cls(max_worlds=int(data.get("max_worlds") or 50))
        pool._manifest = list(data.get("manifest") or [])
        for item in data.get("worlds") or []:
            w = ReplayWorld(
                world_id=item.get("world_id") or f"T{len(pool.worlds)+1}",
                nodes=item.get("nodes") or {},
                baseline_score=float(item.get("baseline_score") or 0.0),
                max_parallelism=int(item.get("max_parallelism") or 3),
                beta1=float(item.get("beta1") or 0.1),
                beta2=float(item.get("beta2") or 0.05),
            )
            if item.get("score_mode"):
                w.score_mode = item["score_mode"]
            pool.worlds.append(w)
        return pool

    def persist_to(self, path: str | Path) -> Path:
        from arsi.foundation.paths import write_json_once, project_root
        p = Path(path)
        if not p.is_absolute():
            p = project_root() / p
        return write_json_once(p, self.export_snapshot(), writer_id="arsi.world_model.world_pool.persist")

    @classmethod
    def load_from(cls, path: str | Path) -> "WorldPool":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls.from_snapshot(data)
        except Exception as e:
            logger.warning(f"WorldPool load failed: {e}")
            return cls()
