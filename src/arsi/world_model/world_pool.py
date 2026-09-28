"""World Pool — Dream-RSI history H_t = (T_1, ..., T_t).

Every online term/rollout appends one completed discovery tree as a
replay world. Policy improvement dreams across ALL worlds, not just
the latest one (arXiv:2609.14858 §3).

Environment Evolution (arXiv:2609.04128):
- Each world carries off-policy D_T difficulty components
- Worlds are tagged into evolution lineages (seed generation 0+)
- WorldEvolver produces children (scenario/skill/length) with Oracle+Invalid
- ELScheduler samples by lineage pass-rate instead of random full-pool
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
    """Growing pool of frozen replay worlds (discovery trees) + env evolution."""

    def __init__(self, max_worlds: int = 50):
        self.max_worlds = max_worlds
        self.worlds: list[ReplayWorld] = []
        self._manifest: list[dict] = []
        # Environment evolution (lazy imports to keep import graph light)
        from arsi.meta.el_scheduler import ELScheduler
        from arsi.meta.env_difficulty import (
            ReferenceCorpus,
            compute_env_difficulty,
            compute_env_difficulty_dyn,
            pool_difficulty_stats,
        )
        from arsi.world_model.world_evolver import WorldEvolver

        self._compute_env_difficulty = compute_env_difficulty
        self._compute_env_difficulty_dyn = compute_env_difficulty_dyn
        self._pool_difficulty_stats = pool_difficulty_stats
        self.el_scheduler = ELScheduler(tau=0.75, batch=8)
        self.evolver = WorldEvolver()
        self.reference: Optional[ReferenceCorpus] = None
        self._world_lineage: dict[str, dict] = {}
        self._evolution_log: list[dict] = []
        self._env_cfg = {
            "enabled": True,
            "el_tau": 0.75,
            "el_batch": 8,
            "effort": "high",
            "evolve_every_n": 1,
            "max_evolved_per_harvest": 1,
            "min_seed_nodes": 3,
            "use_el_in_dream": True,
        }

    @property
    def size(self) -> int:
        return len(self.worlds)

    def configure_env_evolution(self, **cfg) -> dict:
        """Merge env-evolution knobs (from dream_rsi_params / YAML)."""
        for k, v in (cfg or {}).items():
            if v is None:
                continue
            self._env_cfg[k] = v
        try:
            self.el_scheduler.tau = float(self._env_cfg.get("el_tau", self.el_scheduler.tau))
            self.el_scheduler.batch = int(self._env_cfg.get("el_batch", self.el_scheduler.batch))
        except Exception:
            pass
        return dict(self._env_cfg)

    @property
    def env_cfg(self) -> dict:
        return dict(self._env_cfg)

    def difficulty_of(self, world) -> dict:
        try:
            return self._compute_env_difficulty(world, reference=self.reference).to_dict()
        except Exception as e:
            return {"d_t": 0.0, "note": f"error:{e}"}

    def difficulty_dyn_of(self, world) -> dict:
        """D_T on dyn features only (static host identity excluded)."""
        try:
            return self._compute_env_difficulty_dyn(world, reference=self.reference).to_dict()
        except Exception as e:
            return {"d_t": 0.0, "note": f"error:{e}"}

    @staticmethod
    def split_static_dyn(world) -> tuple[dict, dict]:
        """ODEWorld decoupling: static context vs dyn trajectory features."""
        nodes = getattr(world, "_full", {}) or {}
        agents, actions, fails, scores, depths = [], [], [], [], []
        for nid, n in nodes.items():
            if nid == "root" or not isinstance(n, dict):
                continue
            agents.append(str(n.get("agent_id", "unknown")))
            actions.append(str(n.get("action", "unknown")))
            meta = n.get("metadata") or n.get("params") or {}
            fails.append(str(n.get("fail_class") or (meta or {}).get("fail_class") or "ok"))
            scores.append(float(n.get("score") or 0.0))
            depths.append(int(n.get("depth") or 0))
        static = {
            "agents": sorted(set(agents)),
            "agent_count": len(set(agents)),
            "world_id": getattr(world, "world_id", ""),
            "baseline_score": getattr(world, "baseline_score", 0.0),
        }
        dyn = {
            "actions": actions,
            "fail_classes": fails,
            "scores": scores,
            "max_depth": max(depths) if depths else 0,
            "n_nodes": len(actions),
            "unique_actions": sorted(set(actions)),
            "unique_fail_classes": sorted(set(fails)),
        }
        return static, dyn

    def _register_world_lineage(self, world, meta: Optional[dict] = None) -> tuple[str, int]:
        meta = meta or {}
        wid = getattr(world, "world_id", None) or f"T{len(self.worlds)}"
        lin = (
            meta.get("lineage_id")
            or getattr(world, "lineage_id", None)
            or f"lin_{wid}"
        )
        gen = int(meta.get("generation", getattr(world, "generation", 0) or 0) or 0)
        try:
            world.lineage_id = lin
            world.generation = gen
        except Exception:
            pass
        self._world_lineage[wid] = {
            "lineage_id": lin,
            "generation": gen,
            "direction": meta.get("direction") or getattr(world, "direction", "") or "",
            "effort": meta.get("effort") or getattr(world, "effort", "") or "",
            "parent_world_id": meta.get("parent_world_id") or getattr(world, "parent_world_id", "") or "",
            "evolved": bool(meta.get("evolved") or getattr(world, "evolved", False)),
            "source": meta.get("source", "harvest"),
        }
        self.el_scheduler.register_world(lin, wid, gen)
        return lin, gen

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
        self._ingest_world(world, meta=meta or {"source": "harvest", "generation": 0, "evolved": False})
        logger.info(f"World pool += {wid} (size={self.size})")
        return world

    def _ingest_world(self, world: ReplayWorld, meta: Optional[dict] = None) -> ReplayWorld:
        meta = dict(meta or {})
        wid = getattr(world, "world_id", None) or f"T{self.size + 1}"
        world.world_id = wid
        # D_T — third track (policy-independent environment difficulty)
        d = self.difficulty_of(world)
        d_dyn = self.difficulty_dyn_of(world)
        static_ctx, dyn_feats = self.split_static_dyn(world)
        try:
            world.env_difficulty = d
            world.env_difficulty_dyn = d_dyn
            world.static_context = static_ctx
            world.dyn_features = {
                k: v for k, v in dyn_feats.items() if k not in ("actions", "fail_classes", "scores")
            }
            # compact dyn row for RankMe
            world.dyn_row = {
                "d_t_dyn": d_dyn.get("d_t", 0.0),
                "L": d_dyn.get("L", 0.0),
                "skill_rarity": d_dyn.get("skill_rarity", 0.0),
                "scenario_novelty_dyn": d_dyn.get("scenario_novelty", 0.0),
                "n_nodes": d_dyn.get("n_nodes", 0),
                "n_actions": d_dyn.get("n_actions", 0),
            }
        except Exception:
            pass
        self._register_world_lineage(world, meta)
        entry = {
            "world_id": wid,
            "node_count": len(world._full),
            "baseline_score": world.baseline_score,
            "max_parallelism": world.max_parallelism,
            "env_difficulty": {"d_t": d.get("d_t"), "L": d.get("L"), "scenario_novelty": d.get("scenario_novelty"), "skill_rarity": d.get("skill_rarity")},
            "env_difficulty_dyn": {
                "d_t": d_dyn.get("d_t"),
                "L": d_dyn.get("L"),
                "scenario_novelty": d_dyn.get("scenario_novelty"),
                "skill_rarity": d_dyn.get("skill_rarity"),
                "note": d_dyn.get("note"),
            },
            "static_context": static_ctx,
            "dyn_features": world.dyn_features,
            "lineage_id": self._world_lineage.get(wid, {}).get("lineage_id"),
            "generation": self._world_lineage.get(wid, {}).get("generation"),
            "evolved": self._world_lineage.get(wid, {}).get("evolved", False),
            "meta": meta,
        }
        self.worlds.append(world)
        self._manifest.append(entry)
        if len(self.worlds) > self.max_worlds:
            # Prefer dropping non-active lineage parents before active EL targets
            drop_idx = self._pick_drop_index()
            dropped = self.worlds.pop(drop_idx)
            self._manifest.pop(drop_idx) if drop_idx < len(self._manifest) else None
            self._world_lineage.pop(getattr(dropped, "world_id", None), None)
        return world

    def _pick_drop_index(self) -> int:
        active_ids = set()
        for lin in self.el_scheduler.lineages:
            wid = self.el_scheduler.active_world_id(lin)
            if wid:
                active_ids.add(wid)
        for i, w in enumerate(self.worlds):
            if getattr(w, "world_id", None) not in active_ids:
                return i
        return 0

    def append_from_traces(self, traces: list[dict], world_id: Optional[str] = None, **kwargs) -> ReplayWorld:
        tree = DiscoveryTree()
        tree.build_from_traces(traces)
        return self.append_tree(tree, world_id=world_id, **kwargs)

    def evolve_from_seed(
        self,
        seed_world,
        n: int = 1,
        effort: Optional[str] = None,
        directions: Optional[list[str]] = None,
        min_seed_nodes: Optional[int] = None,
    ) -> list[dict]:
        """Off-policy evolve seed world → lineage children; ingest accepted only."""
        cfg = self._env_cfg
        if not cfg.get("enabled", True):
            return []
        effort = effort or str(cfg.get("effort", "high"))
        # P2-2 Red Queen: success↑ → harder effort (unless caller pinned effort)
        try:
            from arsi.meta.red_queen_env import red_queen_effort_for_pool

            rq = red_queen_effort_for_pool(self)
            if effort == str(cfg.get("effort", "high")):
                effort = str(rq.get("effort") or effort)
            self._last_red_queen = rq
        except Exception:
            self._last_red_queen = {}
        min_nodes = int(min_seed_nodes if min_seed_nodes is not None else cfg.get("min_seed_nodes", 3))
        seed_nodes = len(getattr(seed_world, "_full", {}) or {})
        if seed_nodes < min_nodes:
            out = [{"accepted": False, "reason": "seed_too_small", "seed_nodes": seed_nodes, "min_seed_nodes": min_nodes}]
            self._evolution_log.extend(out)
            return out
        try:
            if self.reference is None:
                from arsi.meta.env_difficulty import ReferenceCorpus
                self.reference = ReferenceCorpus.uniform()
                self.evolver.reference = self.reference
        except Exception:
            pass

        seed_id = getattr(seed_world, "world_id", None) or f"seed_{self.size}"
        lin = getattr(seed_world, "lineage_id", None) or f"lin_{seed_id}"
        gen = int(getattr(seed_world, "generation", 0) or 0) + 1
        try:
            seed_world.lineage_id = lin
            if seed_id not in self._world_lineage:
                self._register_world_lineage(seed_world, {"source": "seed", "generation": 0})
        except Exception:
            pass

        results = self.evolver.evolve_batch(
            seed_world,
            n=max(1, int(n)),
            lineage_id=lin,
            generation=gen,
            effort=effort,
            directions=directions,
        )
        accepted = []
        for res in results:
            payload = res.to_dict() if hasattr(res, "to_dict") else dict(res)
            self._evolution_log.append(payload)
            if not getattr(res, "accepted", False):
                continue
            child = getattr(res, "world", None)
            if child is None:
                continue
            # Isolation tag: evolved worlds never feed sealed_tasks
            child.evolved = True
            child.source = "env_evolution"
            child.sealed_eligible = False
            child.lineage_id = lin
            child.generation = gen
            child.direction = getattr(res, "direction", "")
            child.effort = getattr(res, "effort", effort)
            child.parent_world_id = seed_id
            self._ingest_world(
                child,
                meta={
                    "lineage_id": lin,
                    "generation": gen,
                    "direction": child.direction,
                    "effort": child.effort,
                    "parent_world_id": seed_id,
                    "evolved": True,
                    "source": "env_evolution",
                    "seed_id": seed_id,
                    "verifier": getattr(res, "verifier", None).to_dict() if hasattr(getattr(res, "verifier", None), "to_dict") else None,
                },
            )
            accepted.append(payload)
            logger.info(
                f"Env evolution += {child.world_id} lin={lin} g={gen} dir={child.direction} d_t={payload.get('child_difficulty', {}).get('d_t')}"
            )
        return accepted

    def record_policy_probes(self, eval_result: dict) -> list[dict]:
        """Feed pool replay outcomes into ELScheduler (p̂ for each lineage)."""
        out = []
        for row in (eval_result.get("per_world") or []):
            wid = row.get("world_id")
            lin_info = self._world_lineage.get(wid) or {}
            lin = lin_info.get("lineage_id")
            if not lin:
                continue
            score = row.get("score", row.get("replay_score", 0.0)) or 0.0
            quality = row.get("quality", 0.0) or 0.0
            # P2-3: success = quality path solved (replay_score is often negative)
            ok = bool(quality >= 0.3 or score > 0.0)
            self.el_scheduler.record_probe(lin, success=ok, score=None)
            out.append({"world_id": wid, "lineage_id": lin, "score": score, "success": ok})
        return out

    def el_advance_all(self) -> list[dict]:
        """Advance any lineage whose current-gen pass-rate exceeds τ."""
        out = []
        for lin in list(self.el_scheduler.lineages.keys()):
            adv = self.el_scheduler.maybe_advance(lin)
            out.append({"lineage_id": lin, **adv})
        return out

    def evaluate_policy_across_pool(
        self,
        policy_fn,
        policy_name: str = "policy",
        max_rounds: int = 12,
        use_el: bool = False,
        el_k: Optional[int] = None,
    ) -> dict:
        """Dream-RSI multi-world evaluation: score policy on (EL-selected) worlds.

        max_rounds default 12 — cost control; worlds still get probe diversity.
        use_el=True: Evolution-Lineage scheduler picks active-gen worlds first.
        """
        if not self.worlds:
            return {"available": False, "reason": "empty_pool", "avg_score": 0.0, "results": []}

        selected = list(self.worlds)
        el_used = False
        if use_el and self.el_scheduler.lineages:
            k = int(el_k or max(3, len(self.el_scheduler.lineages)))
            picked = self.el_scheduler.select_worlds(self.worlds, k=min(k, len(self.worlds)))
            if picked:
                selected = picked
                el_used = True

        results: list[ReplayResult] = []
        for world in selected:
            result = world.replay(policy_fn, policy_name=policy_name, max_rounds=max_rounds)
            results.append(result)

        avg_score = sum(r.replay_score for r in results) / len(results)
        avg_quality = sum(r.quality for r in results) / len(results)
        avg_probes = sum(r.probes for r in results) / len(results)
        node_counts = [len(w._full) for w in selected]
        per_world = [
            {
                "world_id": r.world_id,
                "score": r.replay_score,
                "quality": r.quality,
                "probes": r.probes,
                "rounds": r.rounds,
                "anchors": r.successful_anchors,
                "repairables": r.repairables,
                "score_breakdown": getattr(r, "score_breakdown", {}),
                "node_count": len(next((w._full for w in selected if w.world_id == r.world_id), {}) or {}),
                "lineage": self._world_lineage.get(r.world_id, {}),
                "env_difficulty": getattr(
                    next((w for w in selected if w.world_id == r.world_id), None),
                    "env_difficulty",
                    {},
                )
                or {},
            }
            for r in results
        ]
        eval_result = {
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
            "el_used": el_used,
            "el_health": self.el_scheduler.health() if el_used or self.el_scheduler.lineages else {},
            "per_world": per_world,
            "score_mode": getattr(selected[0], "score_mode", None) if selected else None,
        }
        if el_used:
            eval_result["el_probes"] = self.record_policy_probes(eval_result)
        return eval_result

    def select_best_policy(
        self,
        current_policy_fn,
        candidate_policy_fns: list,
        current_name: str = "current",
        candidate_names: Optional[list[str]] = None,
        use_paired_ab: bool = True,
        use_el: bool = False,
        same_generation_only: bool = False,
        flow_guidance: Optional[dict] = None,
    ) -> dict:
        """S6: paired A/B + effect-size gate when enabled; else max avg_score.

        P2-9: same_generation_only + difficulty×flow dual gate (promote ≠ D_T↑).
        """
        if use_paired_ab:
            from arsi.meta.paired_ab import select_policy_paired
            return select_policy_paired(
                self,
                current_policy_fn,
                candidate_policy_fns,
                current_name=current_name,
                candidate_names=candidate_names,
                same_generation_only=same_generation_only,
                flow_guidance=flow_guidance,
                apply_difficulty_flow_gate=True,
            )

        candidates = [(current_name, current_policy_fn)]
        names = candidate_names or [f"cand_{i+1}" for i in range(len(candidate_policy_fns))]
        for name, fn in zip(names, candidate_policy_fns):
            candidates.append((name, fn))

        scored = []
        for name, fn in candidates:
            eval_result = self.evaluate_policy_across_pool(fn, policy_name=name, use_el=use_el)
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

    def pool_dyn_rankme(self) -> dict:
        """RankMe on dyn rows of pool worlds — collapse alarm (ODEWorld App.C)."""
        from arsi.foundation.rankme import centered_effective_rank
        keys = ("d_t_dyn", "L", "skill_rarity", "scenario_novelty_dyn", "n_nodes", "n_actions")
        rows = []
        for w in self.worlds:
            row = getattr(w, "dyn_row", None) or {}
            rows.append([float(row.get(k, 0.0) or 0.0) for k in keys])
        return centered_effective_rank(rows)

    def env_evolution_health(self) -> dict:
        worlds = self.worlds
        d_stats = {}
        try:
            d_stats = self._pool_difficulty_stats(worlds, reference=self.reference)
        except Exception as e:
            d_stats = {"error": str(e)}
        evolved_n = sum(1 for w in worlds if getattr(w, "evolved", False) or self._world_lineage.get(getattr(w, "world_id", ""), {}).get("evolved"))
        return {
            "config": self._env_cfg,
            "difficulty": d_stats,
            "pool_dyn_rankme": self.pool_dyn_rankme() if worlds else {},
            "el": self.el_scheduler.health(),
            "evolved_worlds": evolved_n,
            "evolution_accepted": sum(1 for e in self._evolution_log if e.get("accepted")),
            "evolution_rejected": sum(1 for e in self._evolution_log if not e.get("accepted")),
            "lineage_map": {wid: meta for wid, meta in list(self._world_lineage.items())[-20:]},
            "recent_evolutions": self._evolution_log[-8:],
        }

    def stats(self) -> dict:
        d_stats = {}
        try:
            d_stats = self._pool_difficulty_stats(self.worlds, reference=self.reference)
        except Exception as e:
            d_stats = {"error": str(e)}
        return {
            "world_count": self.size,
            "total_nodes": sum(len(w._full) for w in self.worlds),
            "baselines": [w.baseline_score for w in self.worlds],
            "manifest": self._manifest[-10:],
            "env_difficulty": d_stats,
            "el": self.el_scheduler.health() if self.el_scheduler.lineages else {},
            "evolved_count": sum(1 for w in self.worlds if getattr(w, "evolved", False)),
        }

    def save_manifest(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {"manifest": self._manifest, "stats": self.stats(), "env_evolution": self.env_evolution_health()},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def export_snapshot(self) -> dict:
        """Serialize all worlds for crash-safe pool accumulation."""
        worlds = []
        for w in self.worlds:
            lin = self._world_lineage.get(w.world_id, {})
            worlds.append({
                "world_id": w.world_id,
                "nodes": w._full,
                "baseline_score": w.baseline_score,
                "max_parallelism": w.max_parallelism,
                "beta1": getattr(w, "beta1", 0.1),
                "beta2": getattr(w, "beta2", 0.05),
                "score_mode": getattr(w, "score_mode", "quality_anchored"),
                "lineage_id": lin.get("lineage_id") or getattr(w, "lineage_id", ""),
                "generation": lin.get("generation", getattr(w, "generation", 0)),
                "evolved": bool(lin.get("evolved", getattr(w, "evolved", False))),
                "direction": lin.get("direction", getattr(w, "direction", "")),
                "effort": lin.get("effort", getattr(w, "effort", "")),
                "parent_world_id": lin.get("parent_world_id", getattr(w, "parent_world_id", "")),
                "env_difficulty": getattr(w, "env_difficulty", {}) or {},
                "env_difficulty_dyn": getattr(w, "env_difficulty_dyn", {}) or {},
                "static_context": getattr(w, "static_context", {}) or {},
                "dyn_features": getattr(w, "dyn_features", {}) or {},
                "dyn_row": getattr(w, "dyn_row", {}) or {},
                "source": lin.get("source", getattr(w, "source", "harvest")),
            })
        return {
            "schema": "arsi.world_pool.v2",
            "max_worlds": self.max_worlds,
            "manifest": self._manifest,
            "worlds": worlds,
            "el": {
                "tau": self.el_scheduler.tau,
                "batch": self.el_scheduler.batch,
                "advances": self.el_scheduler.advances,
                "lineages": {lin: st.to_dict() for lin, st in self.el_scheduler.lineages.items()},
                "active": dict(self.el_scheduler._active),
            },
            "env_cfg": self._env_cfg,
            "evolution_log": self._evolution_log[-50:],
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "WorldPool":
        pool = cls(max_worlds=int(data.get("max_worlds") or 50))
        pool._manifest = list(data.get("manifest") or [])
        pool._evolution_log = list(data.get("evolution_log") or [])
        if data.get("env_cfg"):
            pool.configure_env_evolution(**data["env_cfg"])
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
            if item.get("env_difficulty"):
                w.env_difficulty = item["env_difficulty"]
            if item.get("env_difficulty_dyn"):
                w.env_difficulty_dyn = item["env_difficulty_dyn"]
            if item.get("static_context"):
                w.static_context = item["static_context"]
            if item.get("dyn_features"):
                w.dyn_features = item["dyn_features"]
            if item.get("dyn_row"):
                w.dyn_row = item["dyn_row"]
            pool.worlds.append(w)
            pool._register_world_lineage(
                w,
                {
                    "lineage_id": item.get("lineage_id") or "",
                    "generation": item.get("generation", 0),
                    "evolved": item.get("evolved", False),
                    "direction": item.get("direction", ""),
                    "effort": item.get("effort", ""),
                    "parent_world_id": item.get("parent_world_id", ""),
                    "source": item.get("source", "harvest"),
                },
            )
        el = data.get("el") or {}
        if el.get("tau") is not None:
            pool.el_scheduler.tau = float(el["tau"])
        if el.get("batch"):
            pool.el_scheduler.batch = int(el["batch"])
        if el.get("advances") is not None:
            pool.el_scheduler.advances = int(el["advances"])
        for lin, st in (el.get("lineages") or {}).items():
            from arsi.meta.el_scheduler import LineageState
            pool.el_scheduler.lineages[lin] = LineageState(
                lineage_id=lin,
                generation=int(st.get("generation", 0)),
                world_ids=list(st.get("world_ids") or []),
                probe_history=list(st.get("probe_history") or []),
                last_advance_ts=st.get("last_advance_ts", ""),
                note=st.get("note", ""),
            )
        for lin, idx in (el.get("active") or {}).items():
            pool.el_scheduler._active[lin] = int(idx)
        return pool

    def persist_to(self, path: str | Path) -> Path:
        from arsi.foundation.paths import write_json_once, project_root
        p = Path(path)
        if not p.is_absolute():
            p = project_root() / p
        data = self.export_snapshot()
        # P: never clobber a larger snapshot with a smaller in-memory pool
        # (short-lived ARSI.from_config processes must not wipe 50-world archives)
        try:
            if p.exists():
                old = json.loads(p.read_text(encoding="utf-8"))
                old_worlds = {w.get("world_id"): w for w in (old.get("worlds") or []) if w.get("world_id")}
                new_ids = {w.get("world_id") for w in data.get("worlds") or []}
                merged = list(data.get("worlds") or [])
                for wid, w in old_worlds.items():
                    if wid not in new_ids:
                        merged.append(w)
                if len(merged) > len(old_worlds):
                    data["worlds"] = merged[: max(self.max_worlds, len(old_worlds))]
                elif len(data.get("worlds") or []) < len(old_worlds) and not getattr(self, "_force_shrink", False):
                    # keep old file; do not shrink
                    logger.warning(
                        "world_pool persist skipped shrink %d→%d",
                        len(old_worlds),
                        len(data.get("worlds") or []),
                    )
                    return p
        except Exception:
            pass
        return write_json_once(p, data, writer_id="arsi.world_model.world_pool.persist")

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
