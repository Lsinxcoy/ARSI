"""Configuration loader for ARSI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    provider: str = "openai"  # openai | anthropic | custom
    model: str = "gpt-4o-mini"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.3
    timeout: int = 60
    use_proxy: bool = False
    proxy_url: str = "http://127.0.0.1:7890"
    fallback_to_heuristic: bool = True
    extra_headers: dict = Field(default_factory=dict)


class EtaConfig(BaseModel):
    alpha: float = 0.1
    theta_low: float = 0.15
    theta_high: float = 0.40


class DimensionConfig(BaseModel):
    max_per_term: int = 2
    min_data_sufficiency: float = 0.3


class QualityGateConfig(BaseModel):
    world_min_verdict: str = "WARN"
    semantic: str = "auto"


class GridPlanConfig(BaseModel):
    hard_max_branch_count: int = 4
    hard_max_refine_count: int = 8
    bootstrap_branch_count: int = 2
    bootstrap_refine_count: int = 3


class BetaSweepConfig(BaseModel):
    grid: list[float] = Field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    parallel_lambda: float = 0.05


class BriefConfig(BaseModel):
    policy: str = "structured"


class ARSIConfig(BaseModel):
    """Top-level ARSI configuration."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    eta: EtaConfig = Field(default_factory=EtaConfig)
    dimensions: DimensionConfig = Field(default_factory=DimensionConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    grid_plan: GridPlanConfig = Field(default_factory=GridPlanConfig)
    beta_sweep: BetaSweepConfig = Field(default_factory=BetaSweepConfig)
    brief: BriefConfig = Field(default_factory=BriefConfig)
    db_path: str = "arsi.db"
    iron_laws_path: str = "config/iron_laws.yaml"
    sealed_tasks_path: str = "config/sealed_tasks.yaml"
    log_level: str = "INFO"

    model_config = {"extra": "allow"}


def load_config(path: str | Path) -> ARSIConfig:
    path = Path(path)
    if not path.exists():
        return ARSIConfig()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return ARSIConfig(**data)
