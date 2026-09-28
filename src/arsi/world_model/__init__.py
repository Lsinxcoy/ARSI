"""ARSI World Model."""
from arsi.world_model.siwm import SIWM, EtaTracker, MindZero, BehaviorPredictor
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import ReplayWorld, Observation, ReplayResult
from arsi.world_model.world_pool import WorldPool
from arsi.world_model.world_evolver import WorldEvolver, EvolutionResult, verify_world, evolve_traces
from arsi.world_model.capability_flow import (
    ACTION_POOL,
    ActionConditionedVelocityField,
    CapabilityFlowTracker,
    VelocityField,
    Z_BLOCKS,
    Z_KEYS,
    STATIC_KEYS,
    canon_action,
    flow_fidelity_check,
    encode_dyn_state,
    encode_static_context,
    integrate,
    integrate_backward,
    pre_failure_organs,
    reverse_from_failure,
    velocity_gt,
)
from arsi.world_model.cognitive_map import CognitiveMap
from arsi.world_model.kernel_flow import KernelVelocityField
from arsi.world_model.nlm_filter import ChannelNLM, NLMBank
from arsi.world_model.dream_rsi_deep import (
    ActionBatch,
    AdaptiveBehaviorController,
    DeepReplaySimulator,
    HistorySimulator,
    dream_selection_fidelity_check,
)
from arsi.world_model.continuous_dream import (
    ContinuousDream,
    DreamStep,
    counterfactual_half_step,
    interpolate_segment,
    synthesize_continuous_dream,
    synthesize_from_discovery_tree,
)
from arsi.world_model.opf_discipline import (
    OPFDisciplineReport,
    cross_block_orthogonality,
    factor_activity,
    intervention_response,
    opf_discipline,
)

# Re-export IWM (introspective world model) for convenience
from arsi.iwm import IWM

__all__ = [
    "SIWM",
    "EtaTracker",
    "MindZero",
    "BehaviorPredictor",
    "DiscoveryTree",
    "ReplayWorld",
    "Observation",
    "ReplayResult",
    "WorldPool",
    "WorldEvolver",
    "EvolutionResult",
    "verify_world",
    "evolve_traces",
    "CapabilityFlowTracker",
    "VelocityField",
    "encode_dyn_state",
    "encode_static_context",
    "ACTION_POOL",
    "ActionConditionedVelocityField",
    "canon_action",
    "Z_KEYS",
    "Z_BLOCKS",
    "STATIC_KEYS",
    "flow_fidelity_check",
    "integrate",
    "integrate_backward",
    "pre_failure_organs",
    "reverse_from_failure",
    "velocity_gt",
    "ChannelNLM",
    "NLMBank",
    "KernelVelocityField",
    "CognitiveMap",
    "ActionBatch",
    "AdaptiveBehaviorController",
    "DeepReplaySimulator",
    "HistorySimulator",
    "dream_selection_fidelity_check",
    "ContinuousDream",
    "DreamStep",
    "counterfactual_half_step",
    "interpolate_segment",
    "synthesize_continuous_dream",
    "synthesize_from_discovery_tree",
    "OPFDisciplineReport",
    "cross_block_orthogonality",
    "factor_activity",
    "intervention_response",
    "opf_discipline",
    "IWM",
]
