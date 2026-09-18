"""Live cycle manifest — Dream-RSI sidecar for meta decisions (Phase D1).

Each online term / dream_rsi_cycle writes one JSON after completion.
plan_grid and default-beta read these files; they are the only trusted
evidence for width/depth and beta (paper Appendix B.2).

Runtime path: E:/ARSI/archive/trace_pool/iter{NNN}/live_cycle_manifest.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SCHEMA = "arsi.live_cycle_manifest.v1"


@dataclass
class GridSnapshot:
    branch_count: int = 2
    refine_count: int = 3
    reason: str = "bootstrap"
    selected_dimensions: list[str] = field(default_factory=list)
    selected_operators: list[str] = field(default_factory=list)
    opened_width: int = 0
    opened_depth: int = 0


@dataclass
class QualityGateStats:
    pass_: int = 0
    warn: int = 0
    fail: int = 0
    not_evaluated: int = 0

    def to_dict(self) -> dict:
        return {
            "pass": self.pass_,
            "warn": self.warn,
            "fail": self.fail,
            "not_evaluated": self.not_evaluated,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QualityGateStats":
        return cls(
            pass_=int(d.get("pass", 0)),
            warn=int(d.get("warn", 0)),
            fail=int(d.get("fail", 0)),
            not_evaluated=int(d.get("not_evaluated", 0)),
        )


@dataclass
class LiveCycleManifest:
    """Metadata about one completed live cycle — not the discovery tree itself."""

    cycle_id: int
    kind: str = "dream_rsi_cycle"  # dream_rsi_cycle | run_term
    timestamp: str = ""
    planned_grid: GridSnapshot = field(default_factory=GridSnapshot)
    effective_grid: GridSnapshot = field(default_factory=GridSnapshot)
    probe_work: int = 0
    decision_rounds: int = 0
    best_score: float = 0.0
    avg_score: float = 0.0
    beta: float = 0.6
    deployed_policy: str = "portfolio"
    pool_size_after: int = 0
    harvested_world_id: str = ""
    quality_gate: QualityGateStats = field(default_factory=QualityGateStats)
    agents: list[str] = field(default_factory=list)
    sealed_delta: Optional[float] = None
    gain_decomposition: dict = field(default_factory=dict)
    notes: str = ""
    schema: str = SCHEMA

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.planned_grid is None:
            self.planned_grid = GridSnapshot()
        if self.effective_grid is None:
            self.effective_grid = GridSnapshot()
        if self.quality_gate is None:
            self.quality_gate = QualityGateStats()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quality_gate"] = self.quality_gate.to_dict()
        d["schema"] = SCHEMA
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LiveCycleManifest":
        pg = d.get("planned_grid") or {}
        eg = d.get("effective_grid") or {}
        return cls(
            cycle_id=int(d.get("cycle_id", 0)),
            kind=str(d.get("kind", "dream_rsi_cycle")),
            timestamp=str(d.get("timestamp", "")),
            planned_grid=GridSnapshot(**{k: pg[k] for k in GridSnapshot.__dataclass_fields__ if k in pg}),
            effective_grid=GridSnapshot(**{k: eg[k] for k in GridSnapshot.__dataclass_fields__ if k in eg}),
            probe_work=int(d.get("probe_work", 0)),
            decision_rounds=int(d.get("decision_rounds", 0)),
            best_score=float(d.get("best_score", 0.0)),
            avg_score=float(d.get("avg_score", 0.0)),
            beta=float(d.get("beta", 0.6)),
            deployed_policy=str(d.get("deployed_policy", "portfolio")),
            pool_size_after=int(d.get("pool_size_after", 0)),
            harvested_world_id=str(d.get("harvested_world_id", "")),
            quality_gate=QualityGateStats.from_dict(d.get("quality_gate") or {}),
            agents=list(d.get("agents") or []),
            sealed_delta=d.get("sealed_delta"),
            gain_decomposition=dict(d.get("gain_decomposition") or {}),
            notes=str(d.get("notes", "")),
        )

    def beta_history_row(self) -> dict:
        return {
            "best_score": self.best_score,
            "beta": self.beta,
            "cycle_id": self.cycle_id,
            "probe_work": self.probe_work,
            "planned_w": self.planned_grid.branch_count,
            "planned_r": self.planned_grid.refine_count,
            "effective_w": self.effective_grid.branch_count,
            "effective_r": self.effective_grid.refine_count,
            "opened_width": self.effective_grid.opened_width,
            "opened_depth": self.effective_grid.opened_depth,
        }


class ManifestStore:
    """Disk-backed store for live cycle manifests."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(__file__).resolve().parents[3] / "archive" / "trace_pool"
        self.root = Path(root)
        self._next_cycle = 1
        self._loaded = False
        self._cache: list[LiveCycleManifest] = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._cache = []
        max_id = 0
        if self.root.exists():
            for p in sorted(self.root.glob("iter*/live_cycle_manifest.json")):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    m = LiveCycleManifest.from_dict(data)
                    self._cache.append(m)
                    max_id = max(max_id, m.cycle_id)
                except Exception as e:
                    logger.warning(f"Skip corrupt manifest {p}: {e}")
        self._cache.sort(key=lambda m: m.cycle_id)
        self._next_cycle = max_id + 1
        self._loaded = True

    @property
    def next_cycle_id(self) -> int:
        self._ensure_loaded()
        return self._next_cycle

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return len(self._cache)

    def append(self, manifest: LiveCycleManifest) -> Path:
        self._ensure_loaded()
        if manifest.cycle_id <= 0:
            manifest.cycle_id = self._next_cycle
        self._next_cycle = max(self._next_cycle, manifest.cycle_id + 1)
        iter_dir = self.root / f"iter{manifest.cycle_id:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        path = iter_dir / "live_cycle_manifest.json"
        path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        # refresh cache entry
        self._cache = [m for m in self._cache if m.cycle_id != manifest.cycle_id]
        self._cache.append(manifest)
        self._cache.sort(key=lambda m: m.cycle_id)
        logger.info(f"Manifest written: {path}")
        return path

    def recent(self, n: int = 3) -> list[LiveCycleManifest]:
        self._ensure_loaded()
        return self._cache[-n:] if n > 0 else list(self._cache)

    def all(self) -> list[LiveCycleManifest]:
        self._ensure_loaded()
        return list(self._cache)

    def beta_history(self, n: int = 5) -> list[dict]:
        return [m.beta_history_row() for m in self.recent(n)]

    def save_beta_sweep(self, cycle_id: int, sweep: dict) -> Path:
        self._ensure_loaded()
        iter_dir = self.root / f"iter{cycle_id:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        path = iter_dir / "beta_sweep.json"
        path.write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load_beta_sweep(self, cycle_id: int) -> Optional[dict]:
        path = self.root / f"iter{cycle_id:04d}" / "beta_sweep.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Corrupt beta_sweep {path}: {e}")
            return None

    def recent_beta_sweeps(self, n: int = 3) -> list[dict]:
        out = []
        for m in self.recent(n):
            s = self.load_beta_sweep(m.cycle_id)
            if s:
                s = dict(s)
                s["cycle_id"] = m.cycle_id
                out.append(s)
        return out

    def next_manifest(self, **kwargs: Any) -> LiveCycleManifest:
        self._ensure_loaded()
        kwargs.setdefault("cycle_id", self._next_cycle)
        return LiveCycleManifest(**kwargs)
