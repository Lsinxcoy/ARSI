"""Programmable Exploration Policy — LLM-revisable policy code.

Inspired by Dream-RSI (arXiv:2609.14858):
- Exploration policy is executable code, not fixed rules
- Policy-development agent (LLM) revises the code based on replay feedback
- Each revision produces a new policy version
- Best version is selected by replay score

In ARSI context:
- Governor's decision logic becomes programmable
- LLM proposes policy revisions
- Replay simulator evaluates revisions offline
- Best policy is deployed
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Callable, Optional

from arsi.foundation.llm import LLMClient
from arsi.world_model.discovery_tree import DiscoveryTree

logger = logging.getLogger(__name__)


def _parse_llm_json(content: str) -> Optional[dict]:
    c = content.strip()
    if c.startswith("```python"):
        c = c[9:]
    elif c.startswith("```"):
        c = c[3:]
    if c.endswith("```"):
        c = c[:-3]
    return c.strip()


class ExplorationPolicy:
    """A programmable exploration policy.

    The policy is a callable that takes observed tree state
    and returns which nodes to explore next.
    """

    def __init__(self, name: str = "default", code: str = ""):
        self.name = name
        self.code = code or self._default_code()
        self._compiled = None
        self._revision_count = 0
        self._replay_scores: list[float] = []

    def _default_code(self) -> str:
        """Default exploration policy: breadth-first with score threshold."""
        return '''
def select_nodes(eligible, observed, tree_stats):
    """Select which nodes to explore next.

    Args:
        eligible: list of eligible node IDs
        observed: set of already-observed node IDs
        tree_stats: dict with tree statistics

    Returns:
        list of node IDs to explore (batch)
    """
    # Default: explore up to 3 nodes per round
    # Prefer nodes with higher scores (exploitation)
    # But always include root (exploration)
    batch = []
    max_batch = 3

    # Always consider root for new branches
    if "root" in eligible:
        batch.append("root")

    # Add other eligible nodes up to max_batch
    for nid in eligible:
        if nid != "root" and len(batch) < max_batch:
            batch.append(nid)

    return batch[:max_batch]
'''

    def select(self, eligible: list[str], observed: set[str], tree_stats: dict = None) -> list[str]:
        """Execute the policy to select nodes."""
        if self._compiled is None:
            self._compile()

        try:
            result = self._compiled(eligible, observed, tree_stats or {})
            if isinstance(result, list):
                return [str(r) for r in result if str(r) in eligible]
        except Exception as e:
            logger.warning(f"Policy execution failed: {e}")

        # Fallback: explore first eligible
        return eligible[:1] if eligible else []

    def _compile(self) -> None:
        """Compile the policy code into a callable."""
        try:
            local_ns = {}
            exec(self.code, {"__builtins__": {"max": max, "min": min, "len": len, "sorted": sorted, "range": range, "enumerate": enumerate, "zip": zip, "list": list, "set": set, "dict": dict}}, local_ns)
            if "select_nodes" in local_ns:
                self._compiled = local_ns["select_nodes"]
            else:
                logger.warning("Policy code missing select_nodes function")
                self._compiled = self._fallback_select
        except Exception as e:
            logger.warning(f"Policy compile failed: {e}")
            self._compiled = self._fallback_select

    @staticmethod
    def _fallback_select(eligible, observed, tree_stats):
        return eligible[:3]

    def revise(self, feedback: str, llm: Optional[LLMClient] = None) -> bool:
        """Revise policy code based on replay feedback.

        Uses LLM to generate improved policy code.
        """
        if not llm or not llm.available:
            logger.warning("LLM unavailable for policy revision")
            return False

        prompt = f"""你是一个探索策略开发专家。根据回放反馈改进探索策略代码。

当前策略代码：
```python
{self.code}
```

回放反馈：
{feedback}

要求：
1. 保持 select_nodes(eligible, observed, tree_stats) 函数签名
2. 根据反馈调整探索逻辑（如：更积极/更保守、偏好特定类型节点等）
3. 代码必须是有效的 Python

输出改进后的完整策略代码（只输出代码，不要其他文字）。"""

        response = llm.chat(
            prompt,
            system="你是探索策略代码生成器。只输出 Python 代码。",
            max_tokens=1500,
        )

        if not response.success:
            return False

        new_code = _parse_llm_json(response.content)
        if not new_code or "def select_nodes" not in new_code:
            logger.warning("LLM returned invalid policy code")
            return False

        self.code = new_code
        self._compiled = None  # Force recompile
        self._revision_count += 1
        logger.info(f"Policy revised (revision {self._revision_count})")
        return True

    def record_score(self, score: float) -> None:
        """Record a replay score for this policy."""
        self._replay_scores.append(score)

    @property
    def best_score(self) -> float:
        return max(self._replay_scores) if self._replay_scores else 0.0

    @property
    def avg_score(self) -> float:
        return sum(self._replay_scores) / len(self._replay_scores) if self._replay_scores else 0.0

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "revision_count": self._revision_count,
            "replay_evaluations": len(self._replay_scores),
            "best_score": round(self.best_score, 4),
            "avg_score": round(self.avg_score, 4),
            "code_length": len(self.code),
        }


class PolicyDevelopmentAgent:
    """LLM agent that revises exploration policies based on replay feedback.

    Dream-RSI's policy-development agent:
    - Examines replay trajectories and scores
    - Identifies successful decisions and recurring failures
    - Revises executable policy code
    - Evaluates revised policy on same replay worlds
    """

    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm
        self._revision_count = 0

    def develop(
        self,
        current_policy: ExplorationPolicy,
        tree: DiscoveryTree,
        num_revisions: int = 3,
    ) -> ExplorationPolicy:
        """Develop improved policy through iterative revision.

        Args:
            current_policy: starting policy
            tree: discovery tree for replay
            num_revisions: number of revision iterations

        Returns:
            Best policy found
        """
        best_policy = current_policy
        best_score = -float("inf")

        # Evaluate current policy
        current_score = self._evaluate(current_policy, tree)
        current_policy.record_score(current_score)
        best_score = current_score

        for revision in range(num_revisions):
            # Generate feedback from replay
            feedback = self._generate_feedback(current_policy, tree)

            # Revise policy
            new_policy = ExplorationPolicy(
                name=f"{current_policy.name}_r{revision + 1}",
                code=current_policy.code,
            )
            if not new_policy.revise(feedback, self.llm):
                logger.warning(f"Revision {revision + 1} failed, keeping current policy")
                continue

            # Evaluate revised policy
            new_score = self._evaluate(new_policy, tree)
            new_policy.record_score(new_score)

            logger.info(f"Revision {revision + 1}: score {new_score:.4f} (best: {best_score:.4f})")

            # Select best
            if new_score > best_score:
                best_policy = new_policy
                best_score = new_score

        self._revision_count += num_revisions
        return best_policy

    def _evaluate(self, policy: ExplorationPolicy, tree: DiscoveryTree) -> float:
        """Evaluate policy via replay."""
        result = tree.replay_alternative(policy.select, max_rounds=10)
        return result["replay_score"]

    def _generate_feedback(self, policy: ExplorationPolicy, tree: DiscoveryTree) -> str:
        """Generate feedback for policy revision."""
        # Run replay and analyze
        result = tree.replay_alternative(policy.select, max_rounds=10)
        stats = tree.get_stats()

        feedback = f"""回放结果分析：
- 回放分数: {result['replay_score']:.4f}
- 最佳发现: {result['best_discovery']:.4f}
- 探索节点数: {result['num_explored']}
- 执行成本: {result['total_cost']:.4f}
- 轮次数: {result['rounds']}
- 并行度: {result['parallelism']:.4f}

树统计：
- 总节点: {stats['node_count']}
- 最大深度: {stats['max_depth']}
- 叶节点: {stats['leaf_count']}
- 结果分布: {stats['outcomes']}

改进建议方向：
1. 如果探索节点数太少 → 更积极地探索新分支
2. 如果执行成本太高 → 更保守，优先高分节点
3. 如果并行度低 → 每轮选择更多节点
4. 如果最佳发现低 → 更注重利用（exploitation）而非探索（exploration）"""

        return feedback
