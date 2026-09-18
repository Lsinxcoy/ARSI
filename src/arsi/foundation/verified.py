"""Evidence-bound verified claims (SYNTHEX P0 / S1).

Rule: never write verified=True without a real test exit code + timestamp.
Constants and name-prefixes are not evidence.
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class VerifiedClaim:
    claim: str = ""
    verified: bool = False
    reason: str = ""
    exit_code: Optional[int] = None
    command: str = ""
    timestamp: str = ""
    detail: dict = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.detail is None:
            self.detail = {}

    def to_dict(self) -> dict:
        return asdict(self)


def unverified(claim: str, reason: str) -> VerifiedClaim:
    return VerifiedClaim(claim=claim, verified=False, reason=reason or "no_evidence")


def verified_from_pytest(
    claim: str,
    project_root: str | Path,
    pytest_args: Optional[list[str]] = None,
    timeout: int = 180,
) -> VerifiedClaim:
    """Run pytest; verified=True only if process exit code == 0."""
    root = Path(project_root)
    args = pytest_args or ["-q", "--tb=line"]
    cmd = [sys.executable, "-m", "pytest", *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        )
    except Exception as e:
        return VerifiedClaim(claim=claim, verified=False, reason=f"pytest_run_failed:{e}", command=" ".join(cmd))

    ok = proc.returncode == 0
    tail = (proc.stdout or "")[-500:]
    return VerifiedClaim(
        claim=claim,
        verified=ok,
        reason="pytest_exit_0" if ok else f"pytest_exit_{proc.returncode}",
        exit_code=proc.returncode,
        command=" ".join(cmd),
        detail={"stdout_tail": tail, "stderr_tail": (proc.stderr or "")[-300:]},
    )


def verified_from_exit_code(claim: str, exit_code: int, command: str = "") -> VerifiedClaim:
    ok = int(exit_code) == 0
    return VerifiedClaim(
        claim=claim,
        verified=ok,
        reason="exit_0" if ok else f"exit_{exit_code}",
        exit_code=int(exit_code),
        command=command,
    )
