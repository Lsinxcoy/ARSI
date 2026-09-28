"""P-c Bounded context compression (AIDE² AIDE_85 second-pass).

Unbounded history concat → prompt overflow death. Compress to bounded summary;
context management is a first-class evolvable component (c3).
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence


@dataclass
class ContextBudget:
    max_chars: int = 4000
    keep_recent: int = 3
    note: str = "aide85_bounded_history"


@dataclass
class CompressedContext:
    summary: str
    n_raw: int
    n_kept_full: int
    chars: int
    dropped: int
    overflow_prevented: bool = False


def compress_history(
    items: Sequence[str],
    budget: Optional[ContextBudget] = None,
) -> CompressedContext:
    """Keep recent items in full; older ones fold into a one-line digest."""
    b = budget or ContextBudget()
    items = [str(x or "") for x in (items or [])]
    n = len(items)
    recent = items[-b.keep_recent :] if b.keep_recent > 0 else []
    older = items[: max(0, n - b.keep_recent)]
    digest_bits = []
    for i, s in enumerate(older):
        digest_bits.append(f"[{i}] {s[:80].replace(chr(10), ' ')}")
    digest = " | ".join(digest_bits)[-800:] if digest_bits else ""
    parts = []
    if digest:
        parts.append(f"## Prior digest\n{digest}")
    for j, s in enumerate(recent):
        parts.append(f"## Recent {j+1}\n{s}")
    out = "\n\n".join(parts)
    dropped = 0
    if len(out) > b.max_chars:
        # shrink digest first, then oldest recent
        overflow = len(out) - b.max_chars
        if digest:
            digest = digest[max(0, overflow) :]
            parts = []
            if digest:
                parts.append(f"## Prior digest\n{digest}")
            for j, s in enumerate(recent):
                parts.append(f"## Recent {j+1}\n{s}")
            out = "\n\n".join(parts)
        while len(out) > b.max_chars and recent:
            recent = recent[1:]
            dropped += 1
            parts = ([f"## Prior digest\n{digest}"] if digest else []) + [
                f"## Recent {j+1}\n{s}" for j, s in enumerate(recent)
            ]
            out = "\n\n".join(parts)
    return CompressedContext(
        summary=out[: b.max_chars],
        n_raw=n,
        n_kept_full=len(recent),
        chars=min(len(out), b.max_chars),
        dropped=dropped,
        overflow_prevented=len(out) > b.max_chars or n > b.keep_recent,
    )


class BoundedContextBuffer:
    """Live buffer that never grows past max_chars (AIDE₀ death mode)."""

    def __init__(self, budget: Optional[ContextBudget] = None):
        self.budget = budget or ContextBudget()
        self.raw: deque[str] = deque(maxlen=200)

    def push(self, item: str) -> None:
        self.raw.append(str(item or ""))

    def render(self) -> CompressedContext:
        return compress_history(list(self.raw), self.budget)

    def report(self) -> dict:
        r = self.render()
        return asdict(r) | {"max_chars": self.budget.max_chars}
