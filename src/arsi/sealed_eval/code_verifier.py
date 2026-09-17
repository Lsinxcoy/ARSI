"""Real code execution verifier for sealed evaluation.

Replaces simplified keyword checking with actual code execution.
Uses subprocess isolation for safety.

Based on: ARSI Whitepaper §9.1 (Sealed Evaluation)
"""
from __future__ import annotations

import ast
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CodeVerifier:
    """Executes code and verifies correctness via assertions.

    Safety:
    - Runs in subprocess with timeout
    - No network access assumed
    - Temp directory isolation
    """

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def verify_python_code(self, code: str, test_cases: list[dict]) -> tuple[float, dict]:
        """Execute Python code and run test cases.

        Args:
            code: Python code to test
            test_cases: list of {assertion: str, description: str}

        Returns:
            (score, evidence) where score is 0.0 to 1.0
        """
        if not code or not code.strip():
            return 0.0, {"reason": "empty_code"}

        # Extract Python code from markdown fences
        extracted = self._extract_code(code)
        if not extracted:
            return 0.0, {"reason": "no_python_code_found"}

        # Check syntax
        try:
            ast.parse(extracted)
        except SyntaxError as e:
            return 0.0, {"reason": f"syntax_error: {e}"}

        # Run test cases
        passed = 0
        total = len(test_cases)
        results = []

        for tc in test_cases:
            assertion = tc.get("assertion", "")
            description = tc.get("description", "")
            if not assertion:
                continue

            success, output = self._run_assertion(extracted, assertion)
            results.append({
                "assertion": assertion,
                "description": description,
                "passed": success,
                "output": output[:200] if output else "",
            })
            if success:
                passed += 1

        score = passed / max(total, 1)
        evidence = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "results": results,
        }
        return score, evidence

    def verify_value_extraction(self, output: str, expected: dict, tolerance: float = 0.1) -> tuple[float, dict]:
        """Verify numeric values extracted from output.

        Args:
            output: agent's output text
            expected: dict of {key: expected_value}
            tolerance: acceptable error

        Returns:
            (score, evidence)
        """
        if not expected:
            return 0.0, {"reason": "no_expected_values"}

        matches = 0
        details = {}

        for key, expected_val in expected.items():
            # Try multiple patterns to extract value
            actual = self._extract_value(output, key)
            if actual is not None:
                is_match = abs(actual - expected_val) <= tolerance
                details[key] = {
                    "expected": expected_val,
                    "actual": actual,
                    "match": is_match,
                    "error": abs(actual - expected_val),
                }
                if is_match:
                    matches += 1
            else:
                details[key] = {
                    "expected": expected_val,
                    "actual": None,
                    "match": False,
                    "error": None,
                }

        score = matches / len(expected)
        return score, {"value_details": details, "matches": matches, "total": len(expected)}

    def verify_keyword_match(self, output: str, keywords: list[str]) -> tuple[float, dict]:
        """Verify keywords present in output.

        Enhanced: also checks for semantic variants.
        """
        if not keywords:
            return 0.0, {"reason": "no_keywords"}

        found = []
        missing = []
        for kw in keywords:
            if kw.lower() in output.lower():
                found.append(kw)
            else:
                # Check for common variants
                variants = self._get_variants(kw)
                if any(v.lower() in output.lower() for v in variants):
                    found.append(kw)
                else:
                    missing.append(kw)

        score = len(found) / len(keywords)
        return score, {
            "keywords_found": found,
            "keywords_missing": missing,
            "total": len(keywords),
        }

    def _extract_code(self, text: str) -> str:
        """Extract Python code from text (handles markdown fences)."""
        # Try markdown code blocks
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'```py\n(.*?)```',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()

        # If no fences, check if text itself looks like Python
        if 'def ' in text or 'class ' in text or 'import ' in text:
            return text.strip()

        return ""

    def _run_assertion(self, code: str, assertion: str) -> tuple[bool, str]:
        """Run code + assertion in a subprocess."""
        script = f"""{code}

# Assertion
try:
    {assertion}
    print("PASS")
except AssertionError as e:
    print(f"FAIL: {{e}}")
except Exception as e:
    print(f"ERROR: {{type(e).__name__}}: {{e}}")
"""

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(script)
                tmp_path = f.name

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=self.timeout,
            )

            output = result.stdout.strip()
            passed = "PASS" in output
            return passed, output

        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        except Exception as e:
            return False, f"EXEC_ERROR: {e}"
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

    def _extract_value(self, text: str, key: str) -> Optional[float]:
        """Extract numeric value for a key from text."""
        patterns = [
            rf'{key}\s*[=:]\s*([\d.]+)',
            rf'{key}\s+is\s+([\d.]+)',
            rf'{key}[:\s]+([\d.]+)',
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    def _get_variants(self, keyword: str) -> list[str]:
        """Get common variants of a keyword for fuzzy matching."""
        variants = {
            "一致性": ["consistency", "consistent"],
            "可用性": ["availability", "available"],
            "分区容错": ["partition", "tolerance", "partition tolerance"],
            "pivot": ["pivot", "枢轴"],
            "递归": ["recursion", "recursive", "递归"],
            "最坏": ["worst", "最坏"],
        }
        return variants.get(keyword, [keyword])
