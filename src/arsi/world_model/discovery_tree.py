"""Discovery Tree — structured tree from behavior traces.

Inspired by Dream-RSI (arXiv:2609.14858):
- Each node = one attempt with parent, children, outcome, score
- Tree structure enables replay simulation
- Alternative policies can navigate the recorded tree

In ARSI context:
- Nodes = behavior traces organized by causal/temporal structure
- Tree enables Governor to replay "what if" scenarios
- Discovery history becomes a reusable simulator
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class DiscoveryNode:
    """One node in the discovery tree = one attempt."""
    id: str = field(default_factory=lambda: uuid4().hex[:12])
    parent_id: Optional[str] = None
    action: str = ""
    agent_id: str = ""
    outcome: str = ""
    score: float = 0.0
    cost: float = 0.0  # execution cost (tokens, time, etc.)
    parallel_group: int = 0  # which parallel batch this belonged to
    depth: int = 0
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)
    children: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "action": self.action,
            "agent_id": self.agent_id,
            "outcome": self.outcome,
            "score": self.score,
            "cost": self.cost,
            "parallel_group": self.parallel_group,
            "depth": self.depth,
            "timestamp": self.timestamp,
            "children": self.children,
        }


class DiscoveryTree:
    """Structured discovery tree from behavior traces.

    Converts flat trace lists into a navigable tree structure
    that enables replay simulation (Dream-RSI style).
    """

    def __init__(self):
        self.nodes: dict[str, DiscoveryNode] = {}
        self.root_id: str = ""
        self._build_count = 0

    def build_from_traces(self, traces: list[dict]) -> dict:
        """Build discovery tree from behavior traces.

        Groups traces by agent and temporal proximity into branches.
        """
        self.nodes.clear()
        self._build_count += 1

        # Create root
        root = DiscoveryNode(
            id="root",
            action="start",
            outcome="root",
            score=0.0,
            timestamp=datetime.now().isoformat(),
        )
        self.nodes["root"] = root
        self.root_id = "root"

        # Group traces by agent
        agent_traces: dict[str, list[dict]] = {}
        for t in traces:
            agent = t.get("agent_id", t.get("params", {}).get("source", "unknown"))
            agent_traces.setdefault(agent, []).append(t)

        # Build branches per agent
        for agent, agent_list in agent_traces.items():
            self._build_branch(agent, agent_list)

        # Compute scores
        self._compute_scores()

        stats = self.get_stats()
        logger.info(f"Discovery tree built: {stats['node_count']} nodes, {stats['max_depth']} depth")
        return stats

    def _build_branch(self, agent: str, traces: list[dict]) -> None:
        """Build a branch for one agent's traces."""
        prev_node_id = self.root_id

        for i, trace in enumerate(traces):
            # Quality signal must be readable — near-zero scores let β1·N
            # dominate replay selection (runtime wall at ~-3.673).
            effect = float(trace.get("effect", 0.5) or 0.5)
            outcome = str(trace.get("outcome", "unknown") or "unknown").lower()
            if "success" in outcome:
                score = 0.2 + max(0.0, min(1.0, effect)) * 0.8
            elif "fail" in outcome:
                score = -abs(effect) if effect else -0.2
            elif "partial" in outcome or "recorded" in outcome:
                score = max(0.05, effect * 0.5)
            else:
                score = effect * 0.5 if effect else 0.0

            node = DiscoveryNode(
                parent_id=prev_node_id,
                action=trace.get("action", "unknown"),
                agent_id=agent,
                outcome=trace.get("outcome", outcome),
                score=score,
                cost=trace.get("params", {}).get("token_count", 0) / 1000.0,
                depth=i + 1,
                timestamp=trace.get("timestamp", datetime.now().isoformat()),
                metadata=trace.get("params", {}),
            )
            self.nodes[node.id] = node

            # Link to parent
            if prev_node_id in self.nodes:
                self.nodes[prev_node_id].children.append(node.id)

            prev_node_id = node.id

    def _compute_scores(self) -> None:
        """Compute aggregate scores for internal nodes."""
        for node_id in reversed(list(self.nodes.keys())):
            node = self.nodes[node_id]
            if node.children:
                child_scores = [self.nodes[cid].score for cid in node.children if cid in self.nodes]
                if child_scores:
                    # Internal node score = max of children (best discovery)
                    node.score = max(child_scores)

    def get_leaves(self) -> list[DiscoveryNode]:
        """Get all leaf nodes (eligible for continued exploration)."""
        return [n for n in self.nodes.values() if not n.children]

    def get_best_path(self) -> list[DiscoveryNode]:
        """Get the path from root to best-scoring leaf."""
        leaves = self.get_leaves()
        if not leaves:
            return [self.nodes[self.root_id]]

        best_leaf = max(leaves, key=lambda n: n.score)

        # Trace back to root
        path = []
        current = best_leaf
        while current:
            path.append(current)
            if current.parent_id and current.parent_id in self.nodes:
                current = self.nodes[current.parent_id]
            else:
                break
        path.reverse()
        return path

    def get_branch(self, node_id: str) -> list[DiscoveryNode]:
        """Get all descendants of a node."""
        result = []
        stack = [node_id]
        while stack:
            nid = stack.pop()
            if nid in self.nodes:
                node = self.nodes[nid]
                result.append(node)
                stack.extend(node.children)
        return result

    def replay_alternative(
        self,
        policy_fn,
        max_rounds: int = 10,
    ) -> dict:
        """Replay an alternative policy through the recorded tree.

        This is the Dream-RSI core mechanism:
        - Policy selects which nodes to explore
        - Replay returns recorded outcomes (no re-execution)
        - Score = quality - cost + parallelism

        Args:
            policy_fn: callable(observed_tree) -> list of node_ids to explore
            max_rounds: max decision rounds

        Returns:
            Replay result with score, path, cost
        """
        observed: set[str] = {self.root_id}
        total_cost = 0.0
        total_rounds = 0
        explored_nodes: list[str] = []

        for round_num in range(max_rounds):
            # Get eligible nodes (root + leaves of observed subtree)
            eligible = self._get_eligible(observed)
            if not eligible:
                break

            # Policy selects batch
            batch = policy_fn(list(eligible), observed)
            if not batch:
                break

            # Replay: reveal recorded children
            for nid in batch:
                if nid not in self.nodes:
                    continue
                node = self.nodes[nid]
                # Reveal children
                for child_id in node.children:
                    if child_id not in observed and child_id in self.nodes:
                        observed.add(child_id)
                        child = self.nodes[child_id]
                        total_cost += child.cost
                        explored_nodes.append(child_id)

            total_rounds += 1

        # Compute replay score
        observed_nodes = [self.nodes[nid] for nid in observed if nid in self.nodes]
        best_score = max((n.score for n in observed_nodes), default=0.0)
        num_explored = len(explored_nodes)

        # Dream-RSI replay objective: quality - β₁·cost + β₂·parallelism
        beta1 = 0.1  # cost penalty
        beta2 = 0.05  # parallelism bonus
        parallelism = num_explored / max(total_rounds, 1)
        replay_score = best_score - beta1 * total_cost + beta2 * parallelism

        return {
            "replay_score": round(replay_score, 4),
            "best_discovery": round(best_score, 4),
            "total_cost": round(total_cost, 4),
            "num_explored": num_explored,
            "rounds": total_rounds,
            "parallelism": round(parallelism, 4),
            "explored_nodes": explored_nodes,
        }

    def _get_eligible(self, observed: set[str]) -> list[str]:
        """Get eligible nodes for exploration (root + observed leaves)."""
        eligible = []
        for nid in observed:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            # Root is always eligible
            if nid == self.root_id:
                eligible.append(nid)
            # Observed leaves with unexplored children
            elif node.children:
                unexplored = [c for c in node.children if c not in observed]
                if unexplored:
                    eligible.append(nid)
            # Observed leaves with no children (terminal)
            # Not eligible for further exploration
        return eligible

    def get_stats(self) -> dict:
        """Get tree statistics."""
        depths = [n.depth for n in self.nodes.values()]
        agents = set(n.agent_id for n in self.nodes.values() if n.agent_id)
        outcomes = {}
        for n in self.nodes.values():
            if n.outcome:
                outcomes[n.outcome] = outcomes.get(n.outcome, 0) + 1

        return {
            "node_count": len(self.nodes),
            "max_depth": max(depths) if depths else 0,
            "leaf_count": len(self.get_leaves()),
            "agents": list(agents),
            "outcomes": outcomes,
            "build_count": self._build_count,
        }

    def to_dict(self) -> dict:
        """Serialize tree for persistence."""
        return {
            "root_id": self.root_id,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "stats": self.get_stats(),
        }
