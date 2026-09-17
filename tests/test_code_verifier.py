"""Tests for real code execution verifier."""
import pytest

from arsi.sealed_eval.code_verifier import CodeVerifier


@pytest.fixture
def verifier():
    return CodeVerifier(timeout=10)


class TestCodeVerifier:
    def test_extract_code_from_markdown(self, verifier):
        text = "Here's the fix:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
        code = verifier._extract_code(text)
        assert "def add" in code
        assert "return a + b" in code

    def test_extract_code_plain(self, verifier):
        text = "def add(a, b):\n    return a + b"
        code = verifier._extract_code(text)
        assert "def add" in code

    def test_extract_code_none(self, verifier):
        code = verifier._extract_code("just some text")
        assert code == ""

    def test_verify_correct_code(self, verifier):
        code = "def add(a, b):\n    return a + b"
        test_cases = [
            {"assertion": "assert add(1, 2) == 3", "description": "1+2=3"},
            {"assertion": "assert add(0, 0) == 0", "description": "0+0=0"},
        ]
        score, evidence = verifier.verify_python_code(code, test_cases)
        assert score == 1.0
        assert evidence["passed"] == 2

    def test_verify_wrong_code(self, verifier):
        code = "def add(a, b):\n    return a - b"  # Wrong: subtract instead of add
        test_cases = [
            {"assertion": "assert add(1, 2) == 3", "description": "1+2=3"},
        ]
        score, evidence = verifier.verify_python_code(code, test_cases)
        assert score == 0.0

    def test_verify_syntax_error(self, verifier):
        code = "def add(a, b)\n    return a + b"  # Missing colon
        test_cases = [{"assertion": "assert add(1,2)==3", "description": "test"}]
        score, evidence = verifier.verify_python_code(code, test_cases)
        assert score == 0.0
        assert "syntax_error" in evidence.get("reason", "")

    def test_verify_empty_code(self, verifier):
        score, evidence = verifier.verify_python_code("", [])
        assert score == 0.0

    def test_verify_value_extraction(self, verifier):
        output = "mean=6.0, std=2.83, median=6.0"
        expected = {"mean": 6.0, "std": 2.83, "median": 6.0}
        score, evidence = verifier.verify_value_extraction(output, expected, tolerance=0.1)
        assert score == 1.0

    def test_verify_value_wrong(self, verifier):
        output = "mean=10.0"
        expected = {"mean": 6.0}
        score, evidence = verifier.verify_value_extraction(output, expected, tolerance=0.1)
        assert score == 0.0

    def test_verify_keyword_match(self, verifier):
        output = "快速排序的最坏时间复杂度是 O(n²)，因为 pivot 选择不当导致递归深度过大"
        keywords = ["pivot", "递归", "最坏"]
        score, evidence = verifier.verify_keyword_match(output, keywords)
        assert score == 1.0

    def test_verify_keyword_with_variants(self, verifier):
        output = "CAP theorem: consistency, availability, partition tolerance"
        keywords = ["一致性", "可用性", "分区容错"]
        score, evidence = verifier.verify_keyword_match(output, keywords)
        # Should find via English variants
        assert score > 0.5

    def test_verify_off_by_one_fix(self, verifier):
        """Test the actual code_fix_001 task."""
        code = """def sum_first_n(n):
    total = 0
    for i in range(1, n + 1):
        total += i
    return total"""
        test_cases = [
            {"assertion": "assert sum_first_n(3) == 6", "description": "1+2+3=6"},
            {"assertion": "assert sum_first_n(5) == 15", "description": "1+2+3+4+5=15"},
        ]
        score, evidence = verifier.verify_python_code(code, test_cases)
        assert score == 1.0

    def test_verify_average_fix(self, verifier):
        """Test the actual code_fix_002 task."""
        code = """def average(numbers):
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)"""
        test_cases = [
            {"assertion": "assert average([]) == 0.0", "description": "empty list"},
            {"assertion": "assert average([1, 2, 3]) == 2.0", "description": "1,2,3 avg"},
        ]
        score, evidence = verifier.verify_python_code(code, test_cases)
        assert score == 1.0
