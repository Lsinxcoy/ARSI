"""Core data models for ARSI.

All data structures use Pydantic for type safety and serialization.
Based on: ARSI Whitepaper v0.8 §5 (Formal Definitions)
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────

class MemoryZone(str, Enum):
    SELF = "self"
    EXPERIENCE = "experience"
    PROXY = "proxy"


class EmpowermentDimension(str, Enum):
    SKILL = "skill"
    HARNESS = "harness"
    KNOWLEDGE = "knowledge"
    DECOMPOSITION = "decomposition"
    METACOGNITION = "metacognition"
    ATTENTION = "attention"
    ENVIRONMENT = "environment"
    SOCIAL_GRAPH = "social_graph"
    CALIBRATION = "calibration"


class LifecycleStage(str, Enum):
    INFANCY = "infancy"
    GROWTH = "growth"
    MATURITY = "maturity"
    DECLINE = "decline"


class VerificationStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    FAILURE = "failure"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    QUARANTINED = "quarantined"


# ── Physical State (Φ) ─────────────────────────────────────────────

class PhysicalState(BaseModel):
    """Φ — physical state snapshot."""
    mechanisms: dict[str, float] = Field(default_factory=dict)
    pipeline_health: dict[str, float] = Field(default_factory=dict)
    consumption_rate: dict[str, float] = Field(default_factory=dict)
    resource_usage: dict[str, float] = Field(default_factory=dict)
    storage_stats: dict[str, int] = Field(default_factory=dict)
    steps_since_change: int = 0
    generation: int = 0


# ── Mental State (Ψ) ───────────────────────────────────────────────

class Belief(BaseModel):
    content: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None


class Goal(BaseModel):
    description: str
    priority: float = Field(ge=0.0, le=1.0, default=0.5)
    dimension: EmpowermentDimension = EmpowermentDimension.SKILL
    status: str = "active"


class IntentRecord(BaseModel):
    """Governor's coupled action — mental part."""
    reason: str = ""
    belief_ref: list[str] = Field(default_factory=list)
    expected_effect: str = ""
    risk_assessment: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class MentalState(BaseModel):
    """Ψ — mental state (system's self-model)."""
    beliefs: list[Belief] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    intents: list[IntentRecord] = Field(default_factory=list)
    affect: dict[str, float] = Field(default_factory=dict)
    norms: list[str] = Field(default_factory=list)


# ── World State ────────────────────────────────────────────────────

class WorldState(BaseModel):
    """S_t = (Φ_t, Ψ_t) — joint world state."""
    phi: PhysicalState = Field(default_factory=PhysicalState)
    psi: MentalState = Field(default_factory=MentalState)
    eta: float = Field(ge=0.0, le=1.0, default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Memory ─────────────────────────────────────────────────────────

class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    zone: MemoryZone = MemoryZone.PROXY
    content: str = ""
    tags: list[str] = Field(default_factory=list)
    agent_id: Optional[str] = None
    generation: int = 0
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    utility_score: float = 0.0
    freshness: float = Field(ge=0.0, le=1.0, default=1.0)
    created_at: datetime = Field(default_factory=datetime.now)
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE


class CausalLink(BaseModel):
    """Causal chain for experience provenance."""
    experience_id: str
    source_traces: list[str] = Field(default_factory=list)
    distilled_at: datetime = Field(default_factory=datetime.now)
    verified_by: Optional[str] = None
    applied_to: list[dict] = Field(default_factory=list)
    freshness_eta: float = Field(ge=0.0, le=1.0, default=1.0)


# ── Behavior Trace ─────────────────────────────────────────────────

class BehaviorTrace(BaseModel):
    """A single behavioral observation from a host agent or ARSI itself."""
    id: str = Field(default_factory=lambda: uuid4().hex)
    agent_id: str = ""
    state_before: WorldState = Field(default_factory=WorldState)
    action: str = ""
    action_params: dict = Field(default_factory=dict)
    state_after: WorldState = Field(default_factory=WorldState)
    outcome: str = ""
    effect: float = 0.0
    generation: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Coupled Action ─────────────────────────────────────────────────

class CoupledAction(BaseModel):
    """a = (a_phy, a_ment) — Governor's output."""
    a_phy: str = ""
    a_ment: IntentRecord = Field(default_factory=IntentRecord)
    target_agent: Optional[str] = None
    params: dict = Field(default_factory=dict)

    @classmethod
    def freeze(cls, reason: str) -> "CoupledAction":
        return cls(
            a_phy="freeze",
            a_ment=IntentRecord(reason=reason, risk_assessment="iron_law_triggered"),
        )


# ── Empowerment ────────────────────────────────────────────────────

class EmpowermentOp(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    dimension: EmpowermentDimension = EmpowermentDimension.SKILL
    target_agent: str = ""
    content: dict = Field(default_factory=dict)
    causal_chain: str = ""
    pre_snapshot: dict = Field(default_factory=dict)
    post_snapshot: Optional[dict] = None
    verification: VerificationStatus = VerificationStatus.UNKNOWN
    verification_evidence: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    rolled_back: bool = False


# ── Evaluation ─────────────────────────────────────────────────────

class TaskResult(BaseModel):
    task_id: str
    declared_confidence: float = 0.0
    actual_score: float = 0.0
    self_model_error: float = 0.0
    status: VerificationStatus = VerificationStatus.UNKNOWN


class BaselineReport(BaseModel):
    capability: float = 0.0
    self_model_accuracy: float = 0.0
    per_task: list[TaskResult] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class TermReport(BaseModel):
    term_id: str
    capability_delta: float = 0.0
    self_model_delta: float = 0.0
    per_task_delta: list[dict] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
