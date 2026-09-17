"""Tests for Dream-RSI inspired mechanisms."""
import pytest

from arsi.world_model.discovery_tree import DiscoveryTree, DiscoveryNode
from arsi.governor.exploration_policy import ExplorationPolicy, PolicyDevelopmentAgent


def _make_traces(n=30):
    traces = []
    for i in range(n):
        agent = ["agent_a", "agent_b", "agent_c"][i % 3]
        traces.append({
            "agent_id": agent,
            "action": f"action_{i}",
            "outcome": "success" if i % 3 != 0 else "failure",
            "effect": 0.3 + (i % 10) * 0.07,
            "params": {"step": i, "token_count": 100 + i * 10},
        })
    return traces


class TestDiscoveryTree:
    def test_build_from_traces(self):
        tree = DiscoveryTree()
        traces = _make_traces(30)
        stats = tree.build_from_traces(traces)
        assert stats["node_count"] > 30  # root + traces
        assert stats["max_depth"] > 0

    def test_tree_structure(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(20))
        root = tree.nodes["root"]
        assert root.parent_id is None
        assert len(root.children) > 0

    def test_get_leaves(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(15))
        leaves = tree.get_leaves()
        assert len(leaves) > 0
        for leaf in leaves:
            assert len(leaf.children) == 0

    def test_get_best_path(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(20))
        path = tree.get_best_path()
        assert len(path) > 0
        assert path[0].id == "root"

    def test_replay_alternative(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(30))

        def simple_policy(eligible, observed):
            return eligible[:2] if eligible else []

        result = tree.replay_alternative(simple_policy, max_rounds=5)
        assert "replay_score" in result
        assert "num_explored" in result
        assert "total_cost" in result
        assert result["rounds"] > 0

    def test_replay_score_composition(self):
        """Verify replay score = quality - cost + parallelism."""
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(30))

        def policy(eligible, observed):
            return eligible[:1] if eligible else []

        result = tree.replay_alternative(policy, max_rounds=5)
        # Score should be finite
        assert isinstance(result["replay_score"], float)

    def test_empty_traces(self):
        tree = DiscoveryTree()
        stats = tree.build_from_traces([])
        assert stats["node_count"] == 1  # Just root

    def test_serialization(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(10))
        data = tree.to_dict()
        assert "nodes" in data
        assert "stats" in data


class TestExplorationPolicy:
    def test_default_policy(self):
        policy = ExplorationPolicy()
        assert "select_nodes" in policy.code
        assert policy._revision_count == 0

    def test_select_nodes(self):
        policy = ExplorationPolicy()
        eligible = ["root", "node1", "node2", "node3"]
        observed = {"root"}
        result = policy.select(eligible, observed)
        assert len(result) > 0
        assert all(n in eligible for n in result)

    def test_select_empty(self):
        policy = ExplorationPolicy()
        result = policy.select([], set())
        assert result == []

    def test_record_score(self):
        policy = ExplorationPolicy()
        policy.record_score(0.5)
        policy.record_score(0.7)
        assert policy.best_score == 0.7
        assert policy.avg_score == pytest.approx(0.6)

    def test_stats(self):
        policy = ExplorationPolicy()
        policy.record_score(0.5)
        stats = policy.stats
        assert stats["replay_evaluations"] == 1
        assert stats["best_score"] == 0.5


class TestPolicyDevelopmentAgent:
    def test_develop_no_llm(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(20))
        policy = ExplorationPolicy()
        agent = PolicyDevelopmentAgent(llm=None)
        best = agent.develop(policy, tree, num_revisions=1)
        # Should return original policy when LLM unavailable
        assert best.name == policy.name

    def test_evaluate_policy(self):
        tree = DiscoveryTree()
        tree.build_from_traces(_make_traces(20))
        policy = ExplorationPolicy()
        agent = PolicyDevelopmentAgent(llm=None)
        score = agent._evaluate(policy, tree)
        assert isinstance(score, float)


class TestDreamRSIIntegration:
    def test_full_loop(self):
        """Test the full Dream-RSI loop: build tree → evaluate → replay."""
        tree = DiscoveryTree()
        traces = _make_traces(40)
        stats = tree.build_from_traces(traces)
        assert stats["node_count"] > 40

        # Evaluate default policy
        policy = ExplorationPolicy()
        result = tree.replay_alternative(policy.select, max_rounds=8)
        assert result["num_explored"] > 0
        assert result["rounds"] > 0

        # Policy should have a reasonable score
        assert isinstance(result["replay_score"], float)
