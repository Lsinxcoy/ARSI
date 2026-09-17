"""Iron Laws — real runtime enforcement.

Implements actual checks for G5 (objective utility anchor),
G6 (circuit breaker), and G10 (sealed evaluation protection).

These are not placeholders — they enforce real constraints.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any, Optional

from arsi.foundation.iron_laws import IronLawViolation

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Tripped, blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


class G5ObjectiveUtility:
    """G5: Objective utility anchor.

    - Effect values must be in [-1, +1]
    - Mechanisms cannot self-evaluate (effect cannot come from the mechanism itself)
    - Effect must have external evidence
    """

    def __init__(self):
        self._violation_count = 0

    def check_effect_range(self, effect: float, source: str = "") -> None:
        """Verify effect is in valid range."""
        if effect < -1.0 or effect > 1.0:
            self._violation_count += 1
            raise IronLawViolation(
                f"G5: effect {effect} out of range [-1, +1] (source={source})"
            )

    def check_no_self_eval(self, mechanism_name: str, effect_source: str) -> None:
        """Verify effect doesn't come from the mechanism itself."""
        if mechanism_name and effect_source and mechanism_name in effect_source:
            self._violation_count += 1
            raise IronLawViolation(
                f"G5: mechanism '{mechanism_name}' cannot self-evaluate "
                f"(effect_source={effect_source})"
            )

    def validate_effect(self, mechanism: str, effect: float, evidence: str = "") -> bool:
        """Full validation: range + no self-eval + evidence required."""
        self.check_effect_range(effect, mechanism)
        self.check_no_self_eval(mechanism, evidence)

        if not evidence:
            logger.warning(f"G5: effect for '{mechanism}' has no evidence (warn only)")
            return False

        return True

    @property
    def violation_count(self) -> int:
        return self._violation_count


class G6CircuitBreaker:
    """G6: Circuit breaker.

    CLOSED → OPEN when failure threshold exceeded
    OPEN → HALF_OPEN after cooldown
    HALF_OPEN → CLOSED on success, → OPEN on failure
    """

    def __init__(self, failure_threshold: int = 5, cooldown_seconds: float = 30.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._trip_count = 0

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info("G6: Circuit breaker CLOSED (recovery successful)")

    def record_failure(self) -> None:
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._trip_count += 1
            logger.warning("G6: Circuit breaker OPEN (recovery failed)")
        elif self.state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self._trip_count += 1
            logger.warning(
                f"G6: Circuit breaker OPEN (threshold {self.failure_threshold} exceeded)"
            )

    def allow_call(self) -> bool:
        """Check if a call should be allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self.cooldown_seconds:
                self.state = CircuitState.HALF_OPEN
                logger.info("G6: Circuit breaker HALF_OPEN (testing recovery)")
                return True
            return False

        # HALF_OPEN: allow one test call
        return True

    def reset(self) -> None:
        """Manual reset (human override)."""
        self.state = CircuitState.CLOSED
        self._failure_count = 0
        logger.info("G6: Circuit breaker manually reset")

    @property
    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "failure_count": self._failure_count,
            "trip_count": self._trip_count,
            "threshold": self.failure_threshold,
        }


class G10SealedProtection:
    """G10: Sealed evaluation protection.

    - Sealed task content must not be readable by ARSI mechanisms
    - Scoring rules must not be modifiable
    - Access log is append-only
    - Release gate requires sealed evaluation pass
    """

    def __init__(self):
        self._access_log: list[dict] = []
        self._protected_paths: set[str] = set()
        self._violation_count = 0

    def register_protected_path(self, path: str) -> None:
        """Register a path as sealed-protected."""
        self._protected_paths.add(path)

    def check_access(self, path: str, operation: str, caller: str = "") -> bool:
        """Check if access to a protected path is allowed.

        Returns True if access is allowed (reading from sealed evaluator itself).
        Raises IronLawViolation if ARSI mechanism tries to access.
        """
        is_protected = any(path.startswith(p) for p in self._protected_paths)

        if not is_protected:
            return True

        # Sealed evaluator itself can access
        if caller == "sealed_evaluator":
            self._log_access(path, operation, caller, allowed=True)
            return True

        # ARSI mechanisms cannot access protected paths
        self._violation_count += 1
        self._log_access(path, operation, caller, allowed=False)
        raise IronLawViolation(
            f"G10: access denied to sealed path '{path}' "
            f"(operation={operation}, caller={caller})"
        )

    def check_release_gate(self, eval_passed: bool, eval_score: float,
                           min_score: float = 0.0) -> None:
        """Check if release gate allows promotion."""
        if not eval_passed:
            raise IronLawViolation(
                f"G10: release gate blocked — sealed evaluation not passed"
            )
        if eval_score < min_score:
            raise IronLawViolation(
                f"G10: release gate blocked — score {eval_score} < minimum {min_score}"
            )

    def _log_access(self, path: str, operation: str, caller: str, allowed: bool) -> None:
        """Append-only access log."""
        self._access_log.append({
            "timestamp": time.time(),
            "path": path,
            "operation": operation,
            "caller": caller,
            "allowed": allowed,
        })

    @property
    def access_log(self) -> list[dict]:
        return list(self._access_log)

    @property
    def violation_count(self) -> int:
        return self._violation_count


class EnforcedIronLaws:
    """Combined iron law enforcement for G5, G6, G10.

    Replaces placeholder checks with real runtime enforcement.
    """

    def __init__(self, sealed_task_path: str = "config/sealed_tasks.yaml"):
        self.g5 = G5ObjectiveUtility()
        self.g6 = G6CircuitBreaker()
        self.g10 = G10SealedProtection()

        # Register sealed paths
        self.g10.register_protected_path(sealed_task_path)
        self.g10.register_protected_path("config/iron_laws.yaml")

    def validate_effect(self, mechanism: str, effect: float, evidence: str = "") -> bool:
        """G5: Validate an effect value."""
        return self.g5.validate_effect(mechanism, effect, evidence)

    def check_call_allowed(self) -> bool:
        """G6: Check if a call should be allowed."""
        return self.g6.allow_call()

    def record_call_result(self, success: bool) -> None:
        """G6: Record call result."""
        if success:
            self.g6.record_success()
        else:
            self.g6.record_failure()

    def check_sealed_access(self, path: str, operation: str, caller: str = "") -> bool:
        """G10: Check sealed path access."""
        return self.g10.check_access(path, operation, caller)

    def check_release(self, eval_passed: bool, eval_score: float, min_score: float = 0.0) -> None:
        """G10: Check release gate."""
        self.g10.check_release_gate(eval_passed, eval_score, min_score)

    def get_violation_summary(self) -> dict:
        """Get summary of all violations."""
        return {
            "g5_violations": self.g5.violation_count,
            "g6_state": self.g6.stats,
            "g10_violations": self.g10.violation_count,
            "g10_access_log_size": len(self.g10.access_log),
        }
