"""Memory Proxy — intercepts agent memory calls and routes through Mnemosyne.

Agent thinks it's using its native memory API.
Actually all reads/writes go through ARSI.
Degrades to native memory when ARSI is unavailable.

Based on: ARSI Whitepaper v0.8 §8.8 (Proxy Mode)
"""
from __future__ import annotations

import logging
from typing import Optional, Protocol

from arsi.foundation.schema import MemoryRecord, MemoryZone
from arsi.mnemosyne.core import Mnemosyne

logger = logging.getLogger(__name__)


class NativeMemory(Protocol):
    """Interface for a host agent's native memory system."""

    def remember(self, content: str, tags: list[str] | None = None) -> None: ...
    def recall(self, query: str, k: int = 5) -> list[str]: ...


class NullNativeMemory:
    """Fallback when agent has no native memory."""

    def remember(self, content: str, tags: list[str] | None = None) -> None:
        pass

    def recall(self, query: str, k: int = 5) -> list[str]:
        return []


class MemoryProxy:
    """Memory proxy takeover layer.

    Invariants:
    - Agent is unaware of the proxy (same API)
    - ARSI failure → automatic degradation to native memory
    - Write path: PROXY zone + native backup
    - Read path: EXPERIENCE → PROXY → native (merged)
    """

    def __init__(self, mnemosyne: Mnemosyne, native: NativeMemory, agent_id: str):
        self.mnemosyne = mnemosyne
        self.native = native or NullNativeMemory()
        self.agent_id = agent_id
        self.degraded = False
        self._write_count = 0
        self._read_count = 0

    def remember(self, content: str, tags: list[str] | None = None) -> None:
        """Write path: PROXY zone + native backup."""
        try:
            record = MemoryRecord(
                zone=MemoryZone.PROXY,
                content=content,
                tags=tags or [],
                agent_id=self.agent_id,
                generation=self.mnemosyne.current_generation,
            )
            self.mnemosyne.store.write_memory(record)
            self._write_count += 1

            # Also write to native as backup
            self.native.remember(content, tags)

        except Exception as e:
            logger.warning(f"Memory proxy write failed, degrading: {e}")
            self.degraded = True
            self.native.remember(content, tags)

    def recall(self, query: str, k: int = 5) -> list[str]:
        """Read path: EXPERIENCE → PROXY → native (merged)."""
        if self.degraded:
            return self.native.recall(query, k)

        try:
            # 1. EXPERIENCE zone (highest quality, cross-agent)
            exp_results = self.mnemosyne.store.search_memories(
                zone=MemoryZone.EXPERIENCE, limit=k
            )

            # 2. PROXY zone (this agent's raw traces)
            proxy_results = self.mnemosyne.store.search_memories(
                zone=MemoryZone.PROXY, agent_id=self.agent_id, limit=k
            )

            # 3. Native memory (fallback)
            native_results = self.native.recall(query, k)

            # 4. Merge and rank
            merged = self._merge(exp_results, proxy_results, native_results)
            self._read_count += 1
            return merged[:k]

        except Exception as e:
            logger.warning(f"Memory proxy read failed, degrading: {e}")
            self.degraded = True
            return self.native.recall(query, k)

    def _merge(
        self,
        exp: list[MemoryRecord],
        proxy: list[MemoryRecord],
        native: list[str],
    ) -> list[str]:
        """Merge results: EXPERIENCE first, then PROXY, then native."""
        seen = set()
        result = []

        for r in exp:
            if r.content not in seen:
                result.append(r.content)
                seen.add(r.content)

        for r in proxy:
            if r.content not in seen:
                result.append(r.content)
                seen.add(r.content)

        for n in native:
            if n not in seen:
                result.append(n)
                seen.add(n)

        return result

    @property
    def stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "degraded": self.degraded,
            "write_count": self._write_count,
            "read_count": self._read_count,
        }
