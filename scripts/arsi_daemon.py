"""ARSI Daemon — 24/7 continuous runner.

Modeled after SYNTHEX's runner.py architecture:
- Tick-based loop with configurable sleep
- Lock file to prevent multiple instances
- Checkpoint for crash recovery
- Health snapshots to JSONL
- Periodic extraction from all three agents
- Periodic dimension analysis + empowerment
- Graceful shutdown on Ctrl-C

Usage:
    python scripts/arsi_daemon.py --tick-sleep 300
    python scripts/arsi_daemon.py --tick-sleep 60 --max-ticks 100
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _pid_still_active(pid: int) -> bool:
    """True only if process exists AND is still running (STILL_ACTIVE)."""
    try:
        import ctypes

        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    except Exception:
        return False


# --- atomic single-instance via O_EXCL lock (when run as script / mp re-exec) ---
if __name__ == "__main__":
    _lock_path = Path(r"E:\ARSI\archive\arsi.lock")
    _lock_path.parent.mkdir(parents=True, exist_ok=True)
    for _attempt in range(2):
        try:
            _fd = os.open(str(_lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(_fd, str(os.getpid()).encode())
            globals()["_ARSISINGLETON_FD"] = _fd
            break
        except FileExistsError:
            try:
                _old = int(_lock_path.read_text().strip() or "0")
            except Exception:
                _old = 0
            if _old and _pid_still_active(_old) and _old != os.getpid():
                sys.exit(0)  # live holder
            try:
                _lock_path.unlink()
            except Exception:
                sys.exit(0)
            if _attempt == 1:
                sys.exit(0)
    os.environ["ARSI_DAEMON_SINGLETON"] = str(os.getpid())

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arsi.core import ARSI
from arsi.adapters.mimo_extractor import MiMoSessionExtractor
from arsi.adapters.mimo_deep_extractor import MiMoDeepExtractor
from arsi.adapters.hermes_adapter import HermesAdapter
from arsi.adapters.hermes_deep_adapter import HermesDeepAdapter
from arsi.adapters.synthex_adapter import SynthexAdapter
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.empowerment.applier import EmpowermentApplier

logger = logging.getLogger("arsi.daemon")


def _other_daemon_pids(exclude: int | None = None) -> list[int]:
    """PIDs of live processes whose cmdline runs arsi_daemon.py (Windows)."""
    import ctypes
    import ctypes.wintypes as wt

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wt.DWORD),
            ("cntUsage", wt.DWORD),
            ("th32ProcessID", wt.DWORD),
            ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
            ("th32ModuleID", wt.DWORD),
            ("cntThreads", wt.DWORD),
            ("th32ParentProcessID", wt.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wt.DWORD),
            ("szExeFile", ctypes.c_char * 260),
        ]

    pids: list[int] = []
    exclude = int(exclude or 0)
    # Prefer Toolhelp snapshot of python.exe + wmic cmdline filter
    try:
        import subprocess

        out = subprocess.check_output(
            ["wmic", "process", "where", "name like 'python%'", "get", "ProcessId,CommandLine", "/format:csv"],
            stderr=subprocess.DEVNULL,
            timeout=8,
            text=True,
            errors="replace",
        )
        for line in out.splitlines():
            if "arsi_daemon" not in line.lower():
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if not parts:
                continue
            try:
                pid = int(parts[-1])
            except Exception:
                continue
            if pid and pid != exclude:
                pids.append(pid)
    except Exception:
        pass
    return sorted(set(pids))


def _is_daemon_child_respawn() -> bool:
    """Detect multiprocessing/subprocess child that re-enters arsi_daemon main."""
    parent = os.environ.get("ARSI_DAEMON_SINGLETON", "")
    if parent and parent.isdigit() and int(parent) != os.getpid():
        # spawned under a live daemon — do not become a second daemon
        return True
    return False

# Paths
ARSI_HOME = Path(__file__).parent.parent
ARCHIVE_DIR = ARSI_HOME / "archive"
LOCK_FILE = ARCHIVE_DIR / "arsi.lock"
CHECKPOINT_FILE = ARCHIVE_DIR / "arsi_checkpoint.json"
SNAPSHOT_FILE = ARCHIVE_DIR / "arsi_health.jsonl"
FEEDBACK_DIR = Path.home() / ".local" / "share" / "mimocode" / "memory" / "arsi_feedback"
FEEDBACK_FILE = FEEDBACK_DIR / "latest_suggestions.md"


class ARSIDaemon:
    """24/7 continuous ARSI runner."""

    def __init__(self, tick_sleep: float = 300.0, checkpoint_path: str = "",
                 snapshot_path: str = "", max_ticks: int = 0):
        self.tick_sleep = tick_sleep
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_FILE
        self.snapshot_path = Path(snapshot_path) if snapshot_path else SNAPSHOT_FILE
        self.max_ticks = max_ticks
        self._running = True
        self._tick_count = 0
        self._start_time = time.time()

        # Adapters
        self.mimo_basic = MiMoSessionExtractor()
        self.mimo_deep = MiMoDeepExtractor()
        self.hermes_basic = HermesAdapter()
        self.hermes_deep = HermesDeepAdapter()
        self.synthex = SynthexAdapter()

        # Source mtimes for change detection
        self._last_mtimes: dict[str, float] = {}

    def run(self) -> None:
        """Main daemon loop."""
        # Setup lock
        self._acquire_lock()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Load checkpoint if exists
        self._load_checkpoint()

        # Build ARSI
        logger.info("Building ARSI...")
        self.arsi = ARSI.from_config("config/arsi.yaml")
        self.applier = EmpowermentApplier(self.arsi.store)
        self.orchestrator = DimensionOrchestrator(self.arsi.store, llm=self.arsi.llm)
        # Register standard hosts so IWM can accumulate organ evidence
        try:
            if getattr(self.arsi, "multi_agent", None) is not None:
                for aid, role, caps in (
                    ("hermes", "tool_runner", ["skills", "tools"]),
                    ("mimo-desktop", "worker", ["sessions", "memory"]),
                    ("synthex-mothernest", "researcher", ["mechanisms", "state"]),
                ):
                    if aid not in self.arsi.multi_agent.agents:
                        self.arsi.multi_agent.register(aid, role=role, capabilities=caps)
                try:
                    from arsi.adapters.bidirectional_interface import ARSIInterface
                    self.arsi.multi_agent.interface = ARSIInterface(self.arsi)
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"host pre-register failed: {e}")

        logger.info(f"ARSI daemon started (tick_sleep={self.tick_sleep}s, max_ticks={self.max_ticks or '∞'})")

        try:
            while self._running:
                self._tick()
                if self.max_ticks and self._tick_count >= self.max_ticks:
                    logger.info(f"Max ticks ({self.max_ticks}) reached, stopping")
                    break
                time.sleep(self.tick_sleep)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt, stopping")
        finally:
            self._shutdown()

    def _tick(self) -> None:
        """Execute one tick cycle."""
        self._tick_count += 1
        tick_start = time.time()
        logger.info(f"── Tick {self._tick_count} ──")

        try:
            # 1. Check for new data
            changes = self._detect_changes()
            if not changes:
                logger.info("  No changes detected, skipping")
                self._write_snapshot({"tick": self._tick_count, "status": "no_changes"})
                return

            logger.info(f"  Changes: {', '.join(changes)}")

            # 2. Extract and ingest traces
            new_traces = self._extract_and_ingest()
            logger.info(f"  Ingested: {new_traces} new traces")
            self._write_progress_snapshot("after_ingest", new_traces=new_traces, changes=changes)

            # 2b. Light harvest every tick — grow WorldPool for paired A/B / I6
            try:
                if new_traces > 0:
                    harv = self.arsi.harvest_term_tree()
                    logger.info(f"  harvest: pool={harv.get('pool_size')} kept={harv.get('traces_kept')} gate={harv.get('quality_gate')}")
                    self._write_progress_snapshot("after_harvest", harvest={
                        "pool_size": harv.get("pool_size"),
                        "traces_kept": harv.get("traces_kept"),
                        "quality_gate": harv.get("quality_gate"),
                    })
            except Exception as e:
                logger.warning(f"harvest failed: {e}")

            # 3. Run ARSI steps (cap per tick to avoid runaway LLM)
            for _ in range(2):
                result = self.arsi.step()
                logger.info(f"  Step: {result['decision']['action']} [{result['decision']['source']}]")
            self._write_progress_snapshot("after_steps")

            # 4. Periodic dimension analysis (every 5 ticks) — LLM-budgeted in gate/dims
            if self._tick_count % 5 == 0:
                self._run_dimension_analysis()
                self._write_progress_snapshot("after_dimensions")

            # 5. Periodic dream (every 10 ticks or when η high)
            if self._tick_count % 10 == 0 or self.arsi.siwm.eta.should_dream():
                self._run_dream()
                self._write_progress_snapshot("after_dream")

            # 5b. Dream-RSI meta cycle (every 8 ticks) — skip LLM revise in daemon
            if self._tick_count % 8 == 0:
                self._run_dream_rsi(skip_llm=True)
                self._write_progress_snapshot("after_dream_rsi")

            # 5c. Multi-agent orchestration pass (if any host agents registered)
            ma_info: dict = {}
            if getattr(self.arsi, "multi_agent", None) is not None:
                try:
                    ma = self.arsi.multi_agent
                    if ma.agents:
                        ma.cycle()
                    ma_info = ma.health().get("multi_agent") or {}
                except Exception as e:
                    logger.warning(f"multi-agent cycle failed: {e}")
                    ma_info = {"error": str(e)}

            # 5d. Real host loop — every 2 ticks or when agents already registered
            host_loop_info = {}
            ma_agents = 0
            if getattr(self.arsi, "multi_agent", None) is not None:
                ma_agents = len(self.arsi.multi_agent.agents or {})
            if self._tick_count % 2 == 0 or ma_agents > 0:
                if changes or ma_agents > 0:
                    try:
                        host_loop_info = self._run_host_loop()
                        logger.info(f"  host_loop: {host_loop_info.get('summary')}")
                    except Exception as e:
                        logger.warning(f"host_loop failed: {e}")
                        host_loop_info = {"error": str(e)}

            # 5e. AEGIS landscape every 3 ticks — digester + planner on live traces
            harness_info = {}
            if self._tick_count % 3 == 0:
                try:
                    harness_info = self._run_harness_landscape()
                    logger.info(
                        f"  harness_landscape: failures={harness_info.get('n_failures')} "
                        f"clusters={len(harness_info.get('clusters') or [])}"
                    )
                    # AEGIS auto-evolve dry-run (propose only — never auto-applies core)
                    if harness_info.get("clusters"):
                        auto = self._run_harness_evolve_dry()
                        harness_info["auto_evolve"] = auto
                        logger.info(
                            f"  harness_evolve_dry: proposed={auto.get('proposed')} "
                            f"accepted={auto.get('accepted')} dry_run={auto.get('dry_run')}"
                        )
                except Exception as e:
                    logger.warning(f"harness landscape failed: {e}")
                    harness_info = {"error": str(e)}

            # 5f. GRPO pre-data-plane export every 6 ticks (export only, no train)
            grpo_info = {}
            if self._tick_count % 6 == 0:
                try:
                    grpo_info = self._run_grpo_export()
                    logger.info(
                        f"  grpo_data_plane: groups={grpo_info.get('n_groups')} "
                        f"ready={grpo_info.get('n_advantage_ready_groups')}"
                    )
                except Exception as e:
                    logger.warning(f"grpo export failed: {e}")
                    grpo_info = {"error": str(e)}

            # 6. Save checkpoint
            self._save_checkpoint()

            # 7. Write health snapshot
            elapsed = time.time() - tick_start
            stats = self.arsi.get_stats()
            iwm_stats = stats.get("iwm") or {}
            iwm_inner = iwm_stats.get("iwm") if isinstance(iwm_stats, dict) else {}
            self._write_snapshot({
                "tick": self._tick_count,
                "status": "ok",
                "elapsed_s": round(elapsed, 2),
                "changes": changes,
                "new_traces": new_traces,
                "eta": stats.get("eta", 0),
                "trace_count": stats.get("trace_count", 0),
                "experience_count": stats.get("experience_count", 0),
                "world_pool_size": stats.get("world_pool_size", 0),
                "manifest_cycles": stats.get("manifest_cycles", 0),
                "env_evolution": stats.get("env_evolution") or {},
                "capability_flow": stats.get("capability_flow") or {},
                "difficulty_flow_gate": (stats.get("last_eval_loop") or {}).get("notes", {}).get("difficulty_flow_gate")
                if isinstance(stats.get("last_eval_loop"), dict)
                else {},
                "beta": stats.get("beta", 0.6),
                "grid_plan": stats.get("grid_plan", {}),
                "iwm": {
                    "self_trust": (iwm_inner or {}).get("self_trust"),
                    "organ_trust": (iwm_inner or {}).get("organ_trust"),
                    "memory_trust": (iwm_inner or {}).get("memory_trust"),
                    "memory_status": (iwm_inner or {}).get("memory_status"),
                    "trust_memory_for_learn": (iwm_inner or {}).get("trust_memory_for_learn"),
                    "downweight_memory_ops": (iwm_inner or {}).get("downweight_memory_ops"),
                    "layer1_holdout": (iwm_inner or {}).get("layer1_holdout"),
                    "behavior_predictor_accuracy": (iwm_inner or {}).get("behavior_predictor_accuracy"),
                    "dream_loop": (iwm_inner or {}).get("dream_loop"),
                    "forbid_default_dream": (iwm_inner or {}).get("dream_loop", {}).get("allowed_default") is False
                    if isinstance(iwm_inner.get("dream_loop"), dict) else None,
                    "unreliable_organs": (iwm_inner or {}).get("unreliable_organs"),
                    "calibration": (iwm_inner or {}).get("calibration")
                    or ((iwm_inner or {}).get("q_gate") or {}).get("scores", {}).get("Q6_calibrate")
                    or {},
                    "degrade_to_baseline": (iwm_inner or {}).get("degrade_to_baseline"),
                    "q_gate": stats.get("iwm_q_gate", {}),
                },
                "verified": self._health_verified(stats),
                "multi_agent": ma_info or (stats.get("multi_agent") or {}),
                "host_loop": host_loop_info if "host_loop_info" in locals() else {},
                "harness_landscape": harness_info if "harness_info" in locals() else {},
                "grpo_data_plane": grpo_info if "grpo_info" in locals() else {},
                "armor": self._armor_health(),
                "layer1": stats.get("layer1"),
                "paths": stats.get("paths"),
            })

        except Exception as e:
            logger.error(f"  Tick failed: {e}", exc_info=True)
            self._write_snapshot({
                "tick": self._tick_count,
                "status": "error",
                "error": str(e),
            })

    def _run_host_loop(self) -> dict:
        """Invoke real host loop in-process via scripts/host_loop helpers."""
        import importlib.util
        hl_path = Path(__file__).parent / "host_loop.py"
        spec = importlib.util.spec_from_file_location("arsi_host_loop", hl_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        interface = None
        try:
            from arsi.adapters.bidirectional_interface import ARSIInterface
            interface = ARSIInterface(self.arsi)
        except Exception:
            pass
        orch = self.arsi.multi_agent
        if orch is not None and interface is not None:
            orch.interface = interface
        report = mod.run_cycle(self.arsi, orch, cycle_id=self._tick_count)
        execs = report.get("executions") or {}
        # Harness variants: bind hosts and observe outcomes (HarnessX §4.5)
        try:
            from arsi.foundation.paths import archive_dir
            from arsi.harness.variants import VariantPool

            vpath = archive_dir() / "harness" / "variants.json"
            pool = VariantPool(max_variants=6)
            if vpath.exists():
                try:
                    import json as _json

                    data = _json.loads(vpath.read_text(encoding="utf-8"))
                    for vid, vd in (data.get("variants") or {}).items():
                        from arsi.harness.variants import HarnessVariant

                        pool.variants[vid] = HarnessVariant(
                            variant_id=vid,
                            label=vd.get("label") or vid,
                            cluster_success=dict(vd.get("cluster_success") or {}),
                            active=bool(vd.get("active", True)),
                            created_at=vd.get("created_at") or "",
                        )
                except Exception:
                    pass
            hosts = list(execs.keys()) or ["hermes", "mimo-desktop", "synthex-mothernest"]
            pool.ensure_hosts(hosts)
            for hid, ex in execs.items():
                pool.observe_host(hid, bool(ex.get("success")))
            vpath.parent.mkdir(parents=True, exist_ok=True)
            _json_dump = pool.report()
            vpath.write_text(
                __import__("json").dumps(_json_dump, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            host_loop_info_variants = {
                "route": {h: pool.route_for_host(h) for h in hosts},
                "active": _json_dump.get("active"),
            }
        except Exception as _e:
            host_loop_info_variants = {"error": str(_e)}
        summary = {
            "ingest": report.get("ingest"),
            "layer1_holdout": (report.get("layer1") or {}).get("holdout_accuracy"),
            "variants": host_loop_info_variants,
            "hosts": {
                k: {"success": v.get("success"), "effect": v.get("effect")}
                for k, v in execs.items()
            },
            "multi_agent_agents": {
                aid: a.get("organ_status")
                for aid, a in ((report.get("multi_agent") or {}).get("multi_agent") or {}).get("agents", {}).items()
            },
        }
        return summary

    def _detect_changes(self) -> list[str]:
        """Detect which data sources have changed."""
        changes = []
        current_mtimes = self._get_source_mtimes()

        for source, mtime in current_mtimes.items():
            old = self._last_mtimes.get(source, 0)
            if mtime > old:
                changes.append(source)

        self._last_mtimes = current_mtimes
        return changes

    def _get_source_mtimes(self) -> dict[str, float]:
        """Get latest modification times for all sources."""
        mtimes = {}

        # MiMo
        mimo_base = self.mimo_basic.memory_base
        if mimo_base and Path(mimo_base).exists():
            latest = 0
            for f in Path(mimo_base).rglob("*.md"):
                latest = max(latest, f.stat().st_mtime)
            mtimes["mimo"] = latest

        # Hermes
        hermes_db = self.hermes_deep.db_path
        if hermes_db.exists():
            mtimes["hermes"] = hermes_db.stat().st_mtime

        # SYNTHEX
        synthex_db = self.synthex.home / "synthex_main.db"
        if synthex_db.exists():
            mtimes["synthex"] = synthex_db.stat().st_mtime
        state_dir = self.synthex.home / "state"
        if state_dir.exists():
            latest = 0
            for f in state_dir.iterdir():
                if f.is_file():
                    latest = max(latest, f.stat().st_mtime)
            mtimes["synthex"] = max(mtimes.get("synthex", 0), latest)

        return mtimes

    def _extract_and_ingest(self) -> int:
        """Extract traces from all sources and ingest into ARSI."""
        count = 0

        # MiMo
        for t in self.mimo_basic.extract_traces():
            self.arsi.ingest_trace("mimo", t["action"], t["outcome"], t.get("effect", 0.5), t.get("params", {}))
            count += 1
        for t in self.mimo_deep.extract_traces():
            self.arsi.ingest_trace("mimo", t["action"], t["outcome"], t.get("effect", 0.5), t.get("params", {}))
            count += 1

        # Hermes
        for t in self.hermes_basic.extract_traces():
            self.arsi.ingest_trace("hermes", t["action"], t["outcome"], t.get("effect", 0.5), t.get("params", {}))
            count += 1
        for t in self.hermes_deep.extract_traces(limit_sessions=50, limit_messages=100):
            self.arsi.ingest_trace("hermes", t["action"], t["outcome"], t.get("effect", 0.5), t.get("params", {}))
            count += 1

        # SYNTHEX
        for t in self.synthex.extract_traces():
            self.arsi.ingest_trace("synthex", t["action"], t["outcome"], t.get("effect", 0.5), t.get("params", {}))
            count += 1

        return count

    def _run_dimension_analysis(self) -> None:
        """Run dimension analysis and apply recommendations."""
        logger.info("  Running dimension analysis...")
        traces = self.arsi.store.get_recent_traces(n=100)
        state = self.arsi.siwm.refresh_state()
        results = self.orchestrator.run_all(traces, state)

        gaps = sum(1 for r in results.values() if r.get("sense", {}).get("gap_detected"))
        logger.info(f"    Gaps detected: {gaps}/6")

        if gaps > 0:
            apply_result = self.applier.apply_all(results)
            logger.info(f"    Applied: {apply_result['applied']} recommendations")

    def _armor_health(self) -> dict:
        """RSI-Armor face: dual sensor / overshoot / conformance / shared calib."""
        try:
            from arsi.harness.runtime_wiring import armor_health
            scores = list(getattr(self.arsi, "_live_capability_scores", []) or [])[-30:]
            gdi = float(getattr(getattr(self.arsi, "iwm", None), "last_gdi", 0.0) or 0.0)
            return armor_health(scores, gdi=gdi)
        except Exception as e:
            return {"error": str(e)}

    def _run_harness_landscape(self) -> dict:
        """AEGIS Digester+Planner over recent production traces."""
        from arsi.foundation.paths import archive_dir, eval_dir, write_json_once
        from arsi.harness.audit import AuditLog
        from arsi.harness.pipeline import load_label_counts, run_landscape, save_label_counts

        traces = self.arsi.store.get_recent_traces(n=400)
        arch = archive_dir()
        audit = AuditLog(arch / "harness")
        label_path = arch / "harness" / "label_counts.json"
        report = run_landscape(
            traces,
            prior_labels=load_label_counts(label_path),
            audit=audit,
        )
        merged = report.pop("_merged_label_counts", None)
        if merged is not None:
            save_label_counts(label_path, merged)
        write_json_once(eval_dir() / "harness_landscape_latest.json", report)
        return report

    def _run_grpo_export(self) -> dict:
        """P1/GRPO: export groups+rewards+split (never trains)."""
        from arsi.foundation.paths import eval_dir
        from arsi.harness.grpo_data import (
            _ChunkPool,
            default_policy_matrix,
            export_grpo_data_plane,
            worlds_from_trace_chunks,
        )

        traces = self.arsi.store.get_recent_traces(n=400)
        pool = getattr(self.arsi, "world_pool", None)
        if pool is None or getattr(pool, "size", 0) < 3:
            pool = _ChunkPool(worlds_from_trace_chunks(traces, n_worlds=8))
        man = export_grpo_data_plane(
            traces,
            eval_dir() / "grpo_data_plane",
            harness_variant="default",
            world_pool=pool,
            policies=default_policy_matrix(5),
            max_worlds=12,
        )
        return {
            k: man.get(k)
            for k in (
                "n_traces",
                "n_exported_rewards",
                "n_groups",
                "n_advantage_ready_groups",
                "n_policy_matrix_rows",
                "split",
                "grpo_live",
            )
        }

    def _run_harness_evolve_dry(self) -> dict:
        """AEGIS Evolver dry-run: propose Change-Manifests, store + audit, never apply."""
        from arsi.foundation.paths import archive_dir, eval_dir, write_json_once
        from arsi.harness.audit import AuditLog
        from arsi.harness.evolver import Evolver
        from arsi.harness.manifest import ManifestStore

        land_path = eval_dir() / "harness_landscape_latest.json"
        if not land_path.exists():
            return {"dry_run": True, "proposed": 0, "accepted": 0, "reason": "no_landscape"}
        try:
            payload = json.loads(land_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {"dry_run": True, "proposed": 0, "accepted": 0, "reason": f"bad_landscape:{e}"}
        landscape = payload.get("landscape") or payload
        arch = archive_dir()
        audit = AuditLog(arch / "harness")
        store = ManifestStore(arch / "harness")
        ev = Evolver()
        proposed = ev.propose_from_landscape(landscape, max_k=3) or []
        accepted = []
        for m in proposed:
            if getattr(m, "status", "") == "accepted":
                store.append(m)
                accepted.append(getattr(m, "manifest_id", "") or "")
                audit.emit(
                    "evolver",
                    "manifest_accepted",
                    ok=True,
                    manifest_id=getattr(m, "manifest_id", ""),
                    dry_run=True,
                )
            else:
                audit.emit(
                    "evolver",
                    "manifest_rejected",
                    ok=False,
                    reason=getattr(m, "reject_reason", ""),
                    dry_run=True,
                )
        out = {
            "dry_run": True,
            "proposed": len(proposed),
            "accepted": len(accepted),
            "accepted_ids": accepted,
            "note": "aegis_auto_evolve_dry_run_never_applies_core",
        }
        write_json_once(eval_dir() / "harness_evolve_dry_latest.json", out)
        return out

    def _run_dream(self) -> None:
        """Run dream cycle."""
        state = self.arsi.siwm.refresh_state()
        if state.eta < 0.1 and self._tick_count % 10 != 0:
            return  # Only dream if η is high or every 10 ticks

        logger.info("  Running dream cycle...")
        eta_before = state.eta
        new_state = self.arsi.dream.execute(state)
        logger.info(f"    η: {eta_before:.4f} → {new_state.eta:.4f}")

    def _run_dream_rsi(self, skip_llm: bool = True) -> None:
        """Phase D: Dream-RSI meta-exploration cycle (world pool + manifest + β sweep)."""
        try:
            logger.info(f"  Running Dream-RSI meta cycle (skip_llm={skip_llm})...")
            result = self.arsi.dream_rsi_cycle(skip_llm=skip_llm)
            if not result.get("ran"):
                logger.info(f"    skipped: {result.get('reason')}")
                return
            logger.info(
                f"    deployed={result.get('deployed')} score={result.get('deployed_score')} "
                f"beta={result.get('beta')} pool={result.get('pool_size')} "
                f"manifest_cycle={result.get('manifest_cycle')}"
            )
            gp = result.get("grid_plan") or {}
            logger.info(f"    grid: W={gp.get('branch_count')} R={gp.get('refine_count')} ({gp.get('reason')})")
        except Exception as e:
            logger.error(f"    Dream-RSI cycle failed: {e}")

    def _write_progress_snapshot(self, phase: str, **extra) -> None:
        """Mid-tick health so operators are not blind for multi-hour ticks."""
        try:
            stats = self.arsi.get_stats()
            iwm_stats = stats.get("iwm") or {}
            iwm_inner = iwm_stats.get("iwm") if isinstance(iwm_stats, dict) else {}
            row = {
                "tick": self._tick_count,
                "status": f"running:{phase}",
                "phase": phase,
                "timestamp": datetime.now().isoformat(),
                "eta": stats.get("eta"),
                "trace_count": stats.get("trace_count"),
                "experience_count": stats.get("experience_count"),
                "world_pool_size": stats.get("world_pool_size"),
                "manifest_cycles": stats.get("manifest_cycles"),
                "beta": stats.get("beta"),
                "iwm": {
                    "organ_trust": (iwm_inner or {}).get("organ_trust"),
                    "memory_trust": (iwm_inner or {}).get("memory_trust"),
                    "layer1_holdout": (iwm_inner or {}).get("layer1_holdout"),
                    "trust_memory_for_learn": (iwm_inner or {}).get("trust_memory_for_learn"),
                    "self_trust": (iwm_inner or {}).get("self_trust"),
                    "calibration": (iwm_inner or {}).get("calibration") or {},
                    "degrade_to_baseline": (iwm_inner or {}).get("degrade_to_baseline"),
                },
                "layer1": stats.get("layer1"),
            }
            row.update(extra or {})
            self._write_snapshot(row)
        except Exception as e:
            logger.warning(f"progress snapshot failed: {e}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint for crash recovery."""
        try:
            stats = self.arsi.get_stats()
            checkpoint = {
                "tick": self._tick_count,
                "timestamp": datetime.now().isoformat(),
                "uptime_s": round(time.time() - self._start_time, 1),
                "stats": {k: v for k, v in stats.items() if k != "iron_laws"},
                "last_mtimes": self._last_mtimes,
            }
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path.write_text(json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")

    def _load_checkpoint(self) -> None:
        """Load checkpoint for crash recovery."""
        if self.checkpoint_path.exists():
            try:
                data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
                self._tick_count = data.get("tick", 0)
                self._last_mtimes = data.get("last_mtimes", {})
                logger.info(f"Resumed from checkpoint: tick={self._tick_count}")
            except Exception as e:
                logger.warning(f"Checkpoint load failed: {e}")

    @staticmethod
    def _health_verified(stats: dict) -> dict:
        """S1: verified claims bind runtime evidence fields + timestamp, not constants."""
        from arsi.foundation.verified import VerifiedClaim
        iwm_ok = bool(stats.get("iwm"))
        layer1 = stats.get("layer1") or {}
        holdout = layer1.get("holdout_accuracy")
        claim = VerifiedClaim(
            claim="daemon_health_iwm_and_layer1",
            verified=bool(iwm_ok and holdout is not None),
            reason="runtime_stats_present" if (iwm_ok and holdout is not None) else "missing_runtime_fields",
            exit_code=0 if (iwm_ok and holdout is not None) else 2,
            command="arsi_daemon._health_verified",
            detail={
                "iwm_in_stats": iwm_ok,
                "layer1_holdout_accuracy": holdout,
                "layer1_live_accuracy": layer1.get("live_accuracy"),
                "world_pool_size": stats.get("world_pool_size"),
                "manifest_cycles": stats.get("manifest_cycles"),
            },
        )
        return claim.to_dict()

    def _write_snapshot(self, data: dict) -> None:
        """Write health snapshot to JSONL."""
        try:
            data["timestamp"] = datetime.now().isoformat()
            self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.snapshot_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Snapshot write failed: {e}")

    def _acquire_lock(self) -> None:
        """Single-instance lock: file + named mutex + live process scan.

        P0-4: a second arsi_daemon (including a child respawn) must exit, not compete.
        """
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # 1) refuse if any OTHER live arsi_daemon is already running
        others = _other_daemon_pids(exclude=os.getpid())
        if others:
            logger.error(f"Another arsi_daemon instance(s) {others} running, exiting")
            sys.exit(1)

        # 2) named mutex — only create if not already held by this process (early __main__)
        try:
            import ctypes

            if globals().get("_ARSISINGLETON_MUTEX"):
                self._mutex = globals()["_ARSISINGLETON_MUTEX"]
            else:
                self._mutex = ctypes.windll.kernel32.CreateMutexW(None, False, "ARSIDaemonSingleton")
                if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
                    logger.error("ARSIDaemonSingleton mutex already held, exiting")
                    sys.exit(1)
        except Exception as e:
            logger.warning(f"mutex acquire failed (continue with lockfile): {e}")

        # 3) lock file is informational only — mutex is source of truth
        #    (OpenProcess on a stale PID can false-positive and block startup)
        LOCK_FILE.write_text(str(os.getpid()))
        os.environ["ARSI_DAEMON_SINGLETON"] = str(os.getpid())
        logger.info(f"Lock acquired: PID {os.getpid()}")

    def _release_lock(self) -> None:
        """Release lock file."""
        try:
            if LOCK_FILE.exists():
                LOCK_FILE.unlink()
        except Exception:
            pass

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Signal {signum} received, stopping...")
        self._running = False

    def _shutdown(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._save_checkpoint()
        self._release_lock()
        if hasattr(self, 'arsi'):
            self.arsi.close()
        logger.info(f"Stopped after {self._tick_count} ticks")


def main():
    if _is_daemon_child_respawn():
        logging.basicConfig(level=logging.WARNING)
        logger.warning("child respawn under ARSI_DAEMON_SINGLETON=%s — exiting", os.environ.get("ARSI_DAEMON_SINGLETON"))
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="ARSI 24/7 Daemon")
    parser.add_argument("--tick-sleep", type=float, default=300.0,
                        help="Seconds between ticks (default: 300 = 5 min)")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT_FILE))
    parser.add_argument("--snapshot", default=str(SNAPSHOT_FILE))
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Max ticks before stopping (0 = infinite)")
    args = parser.parse_args()

    daemon = ARSIDaemon(
        tick_sleep=args.tick_sleep,
        checkpoint_path=args.checkpoint,
        snapshot_path=args.snapshot,
        max_ticks=args.max_ticks,
    )
    daemon.run()


if __name__ == "__main__":
    main()
