"""Harness RSI substrate — ModularRSI × HarnessX (arXiv:2609.14857 / 2606.14249).

Nine-dimensional harness map · Change-Manifest · AEGIS (Digester/Planner/gates)
· variant pool. Diagnostic/evolution substrate only — iron laws stay frozen.
"""
from arsi.harness.taxonomy import (
    DIMENSIONS,
    FROZEN_DIMS,
    MODULES,
    aegis_fidelity_check,
    contrastive_batches,
    dimension_of_symbol,
    is_edit_allowed,
    module_scope,
    module_scope_ok,
)
from arsi.harness.manifest import ChangeManifest, ManifestStore
from arsi.harness.audit import AuditLog
from arsi.harness.digester import Digester, FailureCluster
from arsi.harness.planner import AdaptationLandscape, Planner
from arsi.harness.gates import GateResult, regression_check, seesaw_check, critic_check
from arsi.harness.variants import VariantPool, HarnessVariant
from arsi.harness.evolver import Evolver
from arsi.harness.pipeline import run_landscape
from arsi.harness.training_bridge import TrainingExample, export_training_set, trace_to_example
from arsi.harness.edit_budget import edit_budget, stall_flag, unexercised_components
from arsi.harness.noise_floor import calibrate_delta, passes_floor
from arsi.harness.credit import CreditLedger, EditRecord
from arsi.harness.accept import admit, structural_novelty
from arsi.harness.prune import exploration_directive, prune_targets
from arsi.harness.contracts import (
    DEFAULT_CONTRACTS,
    ContractResult,
    LocalContract,
    apply_contracts,
    guarded,
    well_formedness,
)
from arsi.harness.sahoo import (
    WEIGHTS_STATUS,
    capability_alignment_ratio,
    car_frontier,
    decide_stop,
    goal_drift_index,
    regression_risk,
)
from arsi.harness.holdout import (
    accept_rewrite,
    first_order_vs_second_order,
    outer_loop_select,
    private_grade,
    split_tasks,
)
from arsi.harness.hack_kpi import lineage_hack_trend, reward_hack_rate
from arsi.harness.minimality import (
    dual_regression,
    minimal_edit_score,
    prefer_minimal,
    self_harness_round,
)
from arsi.harness.drawback import (
    DrawbackOp,
    MetricExpression,
    MetricVerdict,
    birth_gate,
    curate_pool,
    grader_fidelity_check,
    is_vacuous_metric,
    leave_one_out_marginal,
    metric_fitness,
    recall_weighted_adreement,
    validity_gate,
)
from arsi.harness.gai import (
    ARSI_GAI,
    GAIConfig,
    RSIDefectReport,
    assert_anchored,
    rsi_defects,
)
from arsi.harness.call_guard import (
    guard_call,
    guarded_brief,
    guarded_llm_chat,
    guarded_tool,
)
from arsi.harness.anchor_hardening import (
    HardenedAnchor,
    anchor_set_quality,
    harden_batch,
    harden_soft_anchor,
)
from arsi.harness.strategy_bandit import DEFAULT_ARMS, ArmStats, StrategyBandit
from arsi.harness.pillars import CHARTER, PILLARS, audit_pillars, pillar_summary
from arsi.harness.bounded_context import BoundedContextBuffer, ContextBudget, compress_history
from arsi.harness.detectability import investment_order, recommend_next
from arsi.harness.everitt import EverittVerdict, everitt_guard, guarded_utility_rewrite
from arsi.harness.ignition import IgnitionResult, ignition_test, run_outer_round
from arsi.harness.overshoot import OvershootReport, detect_overshoot, should_stop_editing
from arsi.harness.unified_credit import (
    SOURCE_BOTH,
    SOURCE_HARNESS,
    SOURCE_POLICY,
    UnifiedCreditLedger,
    UnifiedEditRecord,
    unify_from_pair,
)
from arsi.harness.monotone import (
    MonotoneLedger,
    MonotoneSelection,
    extend_admitted_history,
    harness_v_star_ge_v0,
    select_manifest_monotone,
)
from arsi.harness.metabolism import (
    MetabolicSurface,
    metabolic_surface,
    metabolism_from_stats,
)
from arsi.harness.skill_patch import (
    GateVerdict,
    SkillPatch,
    propose_patch,
    validation_gate,
)
from arsi.harness.skill_trace import (
    COMPONENT_EXPERIENTIAL,
    COMPONENT_HARNESS,
    TraceStep,
    build_gamma,
    component_for_fail_class,
    localize_failure,
)
from arsi.harness.coevolution import (
    STAGE1_AGENT,
    STAGE2_ENV,
    STAGE3_META,
    anchored_meta_ok,
    classify_coevolution,
    coevo_fidelity_check,
    red_queen_pressure,
)
from arsi.harness.guidance import (
    GUIDANCE_MATRIX,
    GuidanceVerdict,
    guidance_policy_summary,
    screen_guidance,
    semantic_allowed,
    strip_to_structured,
)
from arsi.harness.pr_wiring import (
    armor_health_full,
    continuous_dream_from_pool,
    log_unified_after_dream,
    monotone_after_selection,
    red_queen_env_jobs,
)
from arsi.harness.compile_gate import (
    CompileGateLedger,
    compile_gate_effect,
    get_compile_ledger,
    record_compile,
)
from arsi.harness.rrsi import (
    BETA0,
    BETA1,
    COMPONENT_VOCAB,
    RRSIRoundVerdict,
    cost_rule_ok,
    delta_from_h0,
    rrsi_round,
    three_track_eval,
)
from arsi.harness.grpo_data import (
    GRPO_LIVE,
    GroupRow,
    RewardRow,
    build_groups,
    build_split_from_traces,
    compute_reward,
    export_grpo_data_plane,
    group_key_for,
    reward_table,
)

# ensure_hosts / route_for_host / observe_host live on VariantPool

__all__ = [
    "DIMENSIONS",
    "FROZEN_DIMS",
    "MODULES",
    "aegis_fidelity_check",
    "contrastive_batches",
    "module_scope_ok",
    "dimension_of_symbol",
    "is_edit_allowed",
    "module_scope",
    "ChangeManifest",
    "ManifestStore",
    "AuditLog",
    "Digester",
    "FailureCluster",
    "AdaptationLandscape",
    "Planner",
    "GateResult",
    "regression_check",
    "seesaw_check",
    "critic_check",
    "VariantPool",
    "HarnessVariant",
    "Evolver",
    "run_landscape",
    "TrainingExample",
    "export_training_set",
    "trace_to_example",
    "edit_budget",
    "stall_flag",
    "unexercised_components",
    "calibrate_delta",
    "passes_floor",
    "CreditLedger",
    "EditRecord",
    "admit",
    "structural_novelty",
    "prune_targets",
    "exploration_directive",
    "DEFAULT_CONTRACTS",
    "ContractResult",
    "LocalContract",
    "apply_contracts",
    "guarded",
    "well_formedness",
    "goal_drift_index",
    "regression_risk",
    "capability_alignment_ratio",
    "decide_stop",
    "car_frontier",
    "WEIGHTS_STATUS",
    "split_tasks",
    "private_grade",
    "accept_rewrite",
    "first_order_vs_second_order",
    "outer_loop_select",
    "reward_hack_rate",
    "lineage_hack_trend",
    "dual_regression",
    "minimal_edit_score",
    "prefer_minimal",
    "self_harness_round",
    "DrawbackOp",
    "MetricExpression",
    "MetricVerdict",
    "birth_gate",
    "curate_pool",
    "is_vacuous_metric",
    "leave_one_out_marginal",
    "metric_fitness",
    "recall_weighted_adreement",
    "validity_gate",
    "grader_fidelity_check",
    "GAIConfig",
    "ARSI_GAI",
    "RSIDefectReport",
    "assert_anchored",
    "rsi_defects",
    "guard_call",
    "guarded_llm_chat",
    "guarded_tool",
    "guarded_brief",
    "harden_soft_anchor",
    "harden_batch",
    "anchor_set_quality",
    "HardenedAnchor",
    "StrategyBandit",
    "ArmStats",
    "DEFAULT_ARMS",
    "CHARTER",
    "PILLARS",
    "audit_pillars",
    "pillar_summary",
    "compress_history",
    "BoundedContextBuffer",
    "ContextBudget",
    "investment_order",
    "recommend_next",
    "everitt_guard",
    "guarded_utility_rewrite",
    "EverittVerdict",
    "ignition_test",
    "run_outer_round",
    "IgnitionResult",
    "detect_overshoot",
    "should_stop_editing",
    "OvershootReport",
    "SOURCE_POLICY",
    "SOURCE_HARNESS",
    "SOURCE_BOTH",
    "UnifiedCreditLedger",
    "UnifiedEditRecord",
    "unify_from_pair",
    "MonotoneLedger",
    "MonotoneSelection",
    "select_manifest_monotone",
    "harness_v_star_ge_v0",
    "extend_admitted_history",
    "MetabolicSurface",
    "metabolic_surface",
    "metabolism_from_stats",
    "STAGE1_AGENT",
    "STAGE2_ENV",
    "STAGE3_META",
    "anchored_meta_ok",
    "classify_coevolution",
    "coevo_fidelity_check",
    "red_queen_pressure",
    "GUIDANCE_MATRIX",
    "GuidanceVerdict",
    "semantic_allowed",
    "screen_guidance",
    "strip_to_structured",
    "guidance_policy_summary",
    "continuous_dream_from_pool",
    "log_unified_after_dream",
    "monotone_after_selection",
    "red_queen_env_jobs",
    "armor_health_full",
    "CompileGateLedger",
    "compile_gate_effect",
    "get_compile_ledger",
    "record_compile",
    "GRPO_LIVE",
    "GroupRow",
    "RewardRow",
    "build_groups",
    "build_split_from_traces",
    "compute_reward",
    "export_grpo_data_plane",
    "group_key_for",
    "reward_table",
    "BETA0",
    "BETA1",
    "COMPONENT_VOCAB",
    "RRSIRoundVerdict",
    "cost_rule_ok",
    "delta_from_h0",
    "rrsi_round",
    "three_track_eval",
    "GateVerdict",
    "SkillPatch",
    "propose_patch",
    "validation_gate",
    "TraceStep",
    "build_gamma",
    "localize_failure",
    "component_for_fail_class",
    "COMPONENT_EXPERIENTIAL",
    "COMPONENT_HARNESS",
]
