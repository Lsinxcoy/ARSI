"""ARSI World Model."""
from arsi.world_model.siwm import SIWM, EtaTracker, MindZero, BehaviorPredictor
from arsi.world_model.discovery_tree import DiscoveryTree
from arsi.world_model.replay_world import ReplayWorld, Observation, ReplayResult
from arsi.world_model.world_pool import WorldPool
from arsi.world_model.world_evolver import WorldEvolver, EvolutionResult, verify_world, evolve_traces
from arsi.world_model.capability_flow import (
    CapabilityFlowTracker,
    VelocityField,
    encode_dyn_state,
    encode_static_context,
    integrate,
    integrate_backward,
    pre_failure_organs,
    reverse_from_failure,
    velocity_gt,
)
from arsi.world_model.dream_rsi_deep import (
    ActionBatch,
    AdaptiveBehaviorController,
    DeepReplaySimulator,
    HistorySimulator,
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
    "integrate",
    "integrate_backward",
    "pre_failure_organs",
    "reverse_from_failure",
    "velocity_gt",
    "ActionBatch",
    "AdaptiveBehaviorController",
    "DeepReplaySimulator",
    "HistorySimulator",
    "IWM",
]
