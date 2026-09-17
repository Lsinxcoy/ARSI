"""ARSI Core — the main system orchestrator.

Assembles all modules into a runnable autopoietic system:
  Mnemosyne (memory) + SIWM (world model) + Governor (decisions)
  + Empowerment (agent improvement) + Dream (self-refresh)
  + Sealed Evaluator (honest measurement) + LLM (intelligence)

Usage:
    from arsi.core import ARSI
    arsi = ARSI.from_config("config/arsi.yaml")
    arsi.ingest_trace(agent_id="a1", action="learn", outcome="success", effect=0.8)
    action = arsi.step()  # one full Governor decision + execution cycle
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from arsi.foundation.config import ARSIConfig, load_config
from arsi.foundation.iron_laws import IronLaws
from arsi.foundation.llm import LLMClient, LLMConfig
from arsi.foundation.schema import (
    BehaviorTrace,
    CoupledAction,
    EmpowermentOp,
    VerificationStatus,
    WorldState,
)
from arsi.foundation.store import MnemosyneStore
from arsi.mnemosyne.core import Mnemosyne
from arsi.mnemosyne.memory_proxy import MemoryProxy, NullNativeMemory
from arsi.world_model.siwm import SIWM
from arsi.governor.core import AutopoieticGovernor, DimensionManager
from arsi.governor.pre_enactment import PreEnactmentEngine
from arsi.empowerment.engine import EmpowermentEngine, NullAdapter
from arsi.empowerment.dimensions import DimensionOrchestrator
from arsi.empowerment.lifecycle_integration import DimensionLifecycleIntegrator
from arsi.pipelines.dream import DreamPipeline
from arsi.llm_brain import LLMPoweredGovernor, LLMPoweredMindZero, LLMPoweredDiagnosis, LLMPoweredDream
from arsi.world_model.dynamics import DynamicsModel
from arsi.world_model.counterfactual import CounterfactualSimulator
from arsi.sealed_eval.gain_decomposition import GainDecomposer
from arsi.foundation.cost_ledger import CostLedger

logger = logging.getLogger(__name__)


class ARSI:
    """The complete ARSI system.

    One object that owns everything: memory, world model, governor,
    empowerment, dream pipeline, sealed evaluation, and LLM.
    """

    def __init__(
        self,
        store: MnemosyneStore,
        mnemosyne: Mnemosyne,
        siwm: SIWM,
        governor: AutopoieticGovernor,
        empowerment: EmpowermentEngine,
        dream: DreamPipeline,
        iron_laws: IronLaws,
        llm: Optional[LLMClient] = None,
    ):
        self.store = store
        self.mnemosyne = mnemosyne
        self.siwm = siwm
        self.governor = governor
        self.empowerment = empowerment
        self.dream = dream
        self.iron_laws = iron_laws
        self.llm = llm
        self._step_count = 0
        self._term_count = 0
        self._memory_proxies: dict[str, MemoryProxy] = {}

        # LLM brain (wraps heuristic modules with LLM intelligence)
        self.llm_governor = LLMPoweredGovernor(llm=llm, heuristic_governor=governor)
        self.llm_mindzero = LLMPoweredMindZero(llm=llm, heuristic_mindzero=siwm.mindzero)
        self.llm_diagnosis = LLMPoweredDiagnosis(llm=llm)
        self.llm_dream = LLMPoweredDream(llm=llm)

        # SIWM Layer 2 + Pre-enactment
        self.dynamics = DynamicsModel(store, llm=llm)
        self.pre_enactment = PreEnactmentEngine(self.dynamics)
        self._dynamics_trained = False

        # SIWM Layer 3: Counterfactual simulator
        self.counterfactual = CounterfactualSimulator(self.dynamics, llm=llm)

        # N1: Gain decomposition
        self.gain_decomposer = GainDecomposer(store, llm=llm)

        # N2: Cost ledger
        self.cost_ledger = CostLedger()

        # N3: Dimension lifecycle integration
        self.dim_orchestrator = DimensionOrchestrator(store, llm=llm)
        self.dim_lifecycle = DimensionLifecycleIntegrator(
            store, self.dim_orchestrator, governor.dims, llm=llm
        )

        # Enable semantic edge discovery
        self.mnemosyne.edge_discovery.enable_semantic()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "ARSI":
        """Build ARSI from a YAML config file."""
        config_path = Path(config_path)
        config = load_config(config_path)
        config_dir = config_path.parent

        # Store
        store = MnemosyneStore(config.db_path)

        # Iron laws
        laws_path = config_dir / Path(config.iron_laws_path).name
        iron_laws = IronLaws(laws_path)

        # LLM
        llm = cls._build_llm(config)

        # Core modules
        mnemosyne = Mnemosyne(store)
        siwm = SIWM(store, config.eta)
        dim_mgr = DimensionManager(
            max_per_term=config.dimensions.max_per_term,
            min_data_sufficiency=config.dimensions.min_data_sufficiency,
        )
        governor = AutopoieticGovernor(siwm, store, iron_laws, dim_mgr)
        empowerment = EmpowermentEngine(mnemosyne, siwm, NullAdapter())
        dream = DreamPipeline(siwm, mnemosyne, llm=llm)

        logger.info(f"ARSI initialized (db={config.db_path}, llm={'yes' if llm and llm.available else 'no'})")

        return cls(
            store=store,
            mnemosyne=mnemosyne,
            siwm=siwm,
            governor=governor,
            empowerment=empowerment,
            dream=dream,
            iron_laws=iron_laws,
            llm=llm,
        )

    @staticmethod
    def _build_llm(config: ARSIConfig) -> Optional[LLMClient]:
        """Build LLM client from config."""
        api_key = os.environ.get("ARSI_API_KEY", "")
        if not api_key and config.llm.api_key:
            api_key = config.llm.api_key

        llm_config = LLMConfig(
            provider=config.llm.provider,
            model=config.llm.model,
            api_key=api_key,
            api_base=config.llm.api_base or "",
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
            timeout=config.llm.timeout,
            use_proxy=config.llm.use_proxy,
            proxy_url=config.llm.proxy_url,
            extra_headers=config.llm.extra_headers or {},
            fallback_to_heuristic=config.llm.fallback_to_heuristic,
        )
        return LLMClient(llm_config)

    # ── Trace Ingestion ─────────────────────────────────────────

    def ingest_trace(
        self,
        agent_id: str,
        action: str,
        outcome: str,
        effect: float = 0.0,
        params: Optional[dict] = None,
    ) -> BehaviorTrace:
        """Ingest a behavior trace from a host agent."""
        trace = BehaviorTrace(
            agent_id=agent_id,
            action=action,
            action_params=params or {},
            outcome=outcome,
            effect=effect,
            generation=self.store.current_generation,
        )
        self.mnemosyne.ingest_trace(trace)
        return trace

    # ── Memory Proxy ────────────────────────────────────────────

    def get_memory_proxy(self, agent_id: str, native=None) -> MemoryProxy:
        """Get or create a memory proxy for an agent."""
        if agent_id not in self._memory_proxies:
            self._memory_proxies[agent_id] = MemoryProxy(
                self.mnemosyne, native or NullNativeMemory(), agent_id
            )
        return self._memory_proxies[agent_id]

    # ── Core Step ───────────────────────────────────────────────

    def step(self) -> dict:
        """Execute one full ARSI cycle.

        Decision priority:
          1. Pre-enactment (Layer 2 dynamics prediction)
          2. LLM (if pre-enactment confidence too low)
          3. Heuristic (fallback)
        """
        self._step_count += 1
        result = {"step": self._step_count, "actions": [], "decision_source": "heuristic"}

        # 0. Train dynamics model periodically
        if self._step_count % 10 == 1 or not self._dynamics_trained:
            train_result = self.dynamics.train()
            self._dynamics_trained = train_result["statistical"]["trained"]
            result["dynamics_trained"] = self._dynamics_trained

        # 1. Refresh world state
        state = self.siwm.refresh_state()
        result["state"] = {
            "generation": state.phi.generation,
            "eta": round(state.eta, 4),
            "belief_count": len(state.psi.beliefs),
            "storage": state.phi.storage_stats,
        }

        # 2. Three-layer decision
        candidates = ["dream", "learn", "evolve", "maintain", "remember"]

        # Layer A: Pre-enactment (predict consequences)
        pre_result = self.pre_enactment.select_best(state, candidates)
        if pre_result["confidence"] > 0.1:
            decision = pre_result
            result["decision_source"] = "pre_enactment"
        else:
            # Layer B: LLM decision
            llm_calls_before = self.llm_governor._llm_calls
            llm_decision = self.llm_governor.decide(state, candidates)
            decision = llm_decision
            result["decision_source"] = "llm" if self.llm_governor._llm_calls > llm_calls_before else "heuristic"
            # Track LLM cost
            if self.llm_governor._llm_calls > llm_calls_before:
                self.cost_ledger.record_llm_call(input_tokens=400, output_tokens=150)

        result["decision"] = {
            "action": decision["action"],
            "reason": decision.get("reason", ""),
            "risk": decision.get("risk", "low"),
            "source": result["decision_source"],
            "confidence": decision.get("confidence", 0.0),
        }

        # 3. Execute the action
        exec_result = self._execute_action_str(decision["action"], state)
        result["actions"].append(exec_result)

        # 4. Update η (predict what we did)
        self.siwm.predict_and_update(decision["action"])

        # 5. Governor metabolism (distill policy)
        if self._step_count % 5 == 0:
            policy = self.governor.distill_policy()
            result["policy_distilled"] = bool(policy)

        # 6. Check if dream is needed
        if self.siwm.eta.should_dream():
            dream_result = self.dream.execute(self.siwm.refresh_state())
            result["actions"].append({
                "type": "dream",
                "eta_before": state.eta,
                "eta_after": dream_result.eta,
            })

        return result

    def _execute_action_str(self, action_name: str, state: WorldState) -> dict:
        """Execute an action by name."""
        a = action_name

        if a == "dream":
            new_state = self.dream.execute(state)
            return {"type": "dream", "eta_after": new_state.eta}

        elif a == "learn":
            return self._execute_learn(state)

        elif a == "evolve":
            return self._execute_evolve(state)

        elif a == "maintain":
            return self._execute_maintain()

        elif a == "remember":
            return {"type": "remember", "note": "waiting for new traces"}

        elif a == "freeze":
            return {"type": "freeze", "reason": action.a_ment.reason}

        else:
            return {"type": a, "note": "unknown action"}

    def _execute_learn(self, state: WorldState) -> dict:
        """Learn: distill PROXY traces into EXPERIENCE."""
        proxy_count = state.phi.storage_stats.get("proxy_count", 0)
        exp_count = state.phi.storage_stats.get("experience_count", 0)

        if proxy_count <= exp_count:
            return {"type": "learn", "note": "no new traces to distill"}

        # Distill high-importance traces
        proxy_traces = self.store.search_memories(
            zone=__import__("arsi.foundation.schema", fromlist=["MemoryZone"]).MemoryZone.PROXY,
            limit=10,
        )
        distilled = 0
        for t in proxy_traces:
            if t.importance >= 0.5 and t.status.value == "active":
                self.mnemosyne._distill(t)
                distilled += 1

        return {"type": "learn", "distilled": distilled}

    def _execute_evolve(self, state: WorldState) -> dict:
        """Evolve: attempt mechanism mutation (placeholder)."""
        exp_count = state.phi.storage_stats.get("experience_count", 0)
        if exp_count < 5:
            return {"type": "evolve", "note": f"insufficient experience ({exp_count}/5)"}

        # Record an evolution attempt
        self.store.record_effect("evolve_attempt", 0.5, "placeholder_mutation")
        return {"type": "evolve", "note": "mutation attempted", "effect": 0.5}

    def _execute_maintain(self) -> dict:
        """Maintain: memory consolidation + decay."""
        result = self.mnemosyne.consolidate()
        return {"type": "maintain", **result}

    # ── Empowerment ─────────────────────────────────────────────

    def empower_agent(self, agent_id: str) -> dict:
        """Empower a host agent with LLM-powered diagnosis."""
        state = self.siwm.refresh_state()
        traces = self.store.get_recent_traces(n=20, agent_id=agent_id)

        # LLM diagnosis
        diagnosis = self.llm_diagnosis.diagnose(traces, state)

        # Execute empowerment
        op = self.empowerment.empower(agent_id, state)

        return {
            "empowerment_id": op.id,
            "agent_id": agent_id,
            "diagnosis": diagnosis,
            "verification": op.verification.value,
            "evidence": op.verification_evidence,
        }

    # ── Sealed Evaluation ───────────────────────────────────────

    def run_evaluation(self, task_set_path: Optional[str] = None) -> dict:
        """Run sealed evaluation (requires agent adapter)."""
        # Simplified: report current stats as a proxy for evaluation
        stats = self.get_stats()
        return {
            "capability_proxy": stats["experience_count"] / max(stats["trace_count"], 1),
            "self_model_eta": self.siwm.eta.value,
            "generation": stats["generation"],
        }

    # ── Stats ───────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get comprehensive system stats."""
        m_stats = self.mnemosyne.get_stats()
        g_stats = self.governor.stats
        e_stats = self.empowerment.stats
        d_stats = self.dream.stats
        pe_stats = self.pre_enactment.stats
        dyn_stats = self.dynamics.stats

        return {
            **m_stats,
            "step_count": self._step_count,
            "term_count": self._term_count,
            "eta": round(self.siwm.eta.value, 4),
            "governor_decisions": g_stats.get("decision_count", 0),
            "empowerments": e_stats.get("empowerment_count", 0),
            "dreams": d_stats.get("dream_count", 0),
            "llm_available": self.llm.available if self.llm else False,
            "llm_governor_calls": self.llm_governor._llm_calls,
            "llm_mindzero_calls": self.llm_mindzero._llm_calls,
            "llm_diagnosis_calls": self.llm_diagnosis._llm_calls,
            "pre_enactment_count": pe_stats.get("pre_enactment_count", 0),
            "dynamics_trained": dyn_stats.get("transition_model", {}).get("trained", False),
            "dynamics_actions": dyn_stats.get("transition_model", {}).get("action_count", 0),
            "cost_tokens": self.cost_ledger.snapshot()["tokens"]["total"],
            "cost_llm_calls": self.cost_ledger.snapshot()["tokens"]["llm_calls"],
            "cost_verifier_queries": self.cost_ledger.snapshot()["verifier"]["queries"],
            "iron_laws": self.iron_laws.law_ids,
        }

    # ── Lifecycle ───────────────────────────────────────────────

    def run_term(self, n_steps: int = 10) -> dict:
        """Run a full improvement term with gain decomposition and cost tracking."""
        self._term_count += 1
        term_id = f"term_{self._term_count}"

        # Reset cost ledger for this term
        self.cost_ledger.reset()

        # Record before-evaluation
        before_eval = self.run_evaluation()
        traces_before = len(self.store.get_recent_traces(n=1000))

        results = []
        for i in range(n_steps):
            result = self.step()
            results.append(result)
            # Track LLM costs (approximate)
            if result.get("decision_source") == "llm":
                self.cost_ledger.record_llm_call(input_tokens=500, output_tokens=200)
            logger.info(f"  Step {i+1}/{n_steps}: {result['decision']['action']}")

        # End-of-term evaluation
        after_eval = self.run_evaluation()
        self.cost_ledger.record_verifier_query(1)

        # Gain decomposition
        term_traces = self.store.get_recent_traces(n=100)
        decomposition = self.gain_decomposer.decompose(before_eval, after_eval, term_traces)
        self.gain_decomposer.record(decomposition, term_id)

        # Cost-benefit
        cost_benefit = self.cost_ledger.cost_benefit(decomposition.get("total", 0))

        # Dimension lifecycle integration
        state = self.siwm.refresh_state()
        lifecycle_result = self.dim_lifecycle.run_and_integrate(term_traces, state)

        # Record term report
        self.mnemosyne.write_self_record("term_report", {
            "term_id": term_id,
            "steps": n_steps,
            "final_stats": self.get_stats(),
            "evaluation": after_eval,
            "gain_decomposition": decomposition,
            "cost_benefit": cost_benefit,
            "lifecycle_recommendation": lifecycle_result.get("recommendation", {}),
        })

        return {
            "term_id": term_id,
            "steps": results,
            "evaluation": after_eval,
            "gain_decomposition": decomposition,
            "cost_benefit": cost_benefit,
            "lifecycle": lifecycle_result.get("lifecycle_updates", {}),
            "final_stats": self.get_stats(),
        }

    def close(self) -> None:
        """Clean shutdown."""
        self.store.close()
