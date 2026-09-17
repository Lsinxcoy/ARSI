"""LLM Agent Adapter — simulates a host agent executing sealed evaluation tasks.

Uses the LLM to "execute" tasks (answer questions, write code, compute values)
and returns structured results for the sealed evaluator to score.

This is the bridge between ARSI's sealed evaluator and an actual agent.
In production, this would be replaced by a real agent adapter (Claude Code, etc.).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from arsi.foundation.llm import LLMClient

logger = logging.getLogger(__name__)


class LLMAgentAdapter:
    """Agent adapter that uses LLM to execute tasks.

    Implements the AgentAdapter protocol expected by SealedEvaluator.
    """

    def __init__(self, llm: LLMClient, agent_id: str = "llm-agent"):
        self.llm = llm
        self.agent_id = agent_id
        self._execution_count = 0
        self._history: list[dict] = []

    def execute(self, prompt: str, timeout: int = 60) -> str:
        """Execute a task by asking the LLM to solve it."""
        self._execution_count += 1

        if not self.llm or not self.llm.available:
            return "[agent unavailable]"

        # Craft system prompt based on task type
        system = self._build_system_prompt(prompt)

        response = self.llm.chat(prompt, system=system, max_tokens=2000)

        result = response.content if response.success else f"[error: {response.error}]"

        self._history.append({
            "prompt": prompt[:100],
            "result": result[:200],
            "success": response.success,
        })

        return result

    def declare_capability(self, task_id: str) -> float:
        """Agent declares its confidence for a task (0.0 to 1.0).

        This is used for self-model accuracy measurement.
        """
        if not self.llm or not self.llm.available:
            return 0.0

        prompt = f"""你正在被评估。任务 ID: {task_id}

不要执行任务，只评估你对完成这个任务的置信度。

输出一个 0.0 到 1.0 的数字，表示你认为自己能正确完成此任务的概率。
只输出数字，不要其他文字。"""

        response = self.llm.chat(prompt, system="你是一个诚实的自我评估器。只输出一个数字。", max_tokens=10)

        if response.success:
            try:
                # Extract number from response
                numbers = re.findall(r'[\d.]+', response.content)
                if numbers:
                    conf = float(numbers[0])
                    return max(0.0, min(1.0, conf))
            except (ValueError, IndexError):
                pass

        return 0.5  # Default when uncertain

    def _build_system_prompt(self, task_prompt: str) -> str:
        """Build appropriate system prompt based on task content."""
        prompt_lower = task_prompt.lower()

        if "修复" in task_prompt or "fix" in prompt_lower or "bug" in prompt_lower:
            return "你是一个代码修复专家。给出修复后的完整代码。"
        elif "解释" in task_prompt or "什么是" in task_prompt or "why" in prompt_lower:
            return "你是一个知识问答专家。给出准确、简洁的解释。"
        elif "计算" in task_prompt or "compute" in prompt_lower or "mean" in prompt_lower:
            return "你是一个计算专家。给出精确的数值结果，用 key=value 格式输出。"
        else:
            return "你是一个通用问题解决专家。给出准确的答案。"

    @property
    def stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "execution_count": self._execution_count,
            "history_size": len(self._history),
        }
