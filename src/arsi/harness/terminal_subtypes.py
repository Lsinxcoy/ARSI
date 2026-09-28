"""Terminal failure subtypes from captured stderr/exit (Recuris-style fine typing)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence

SUBTYPE_TIMEOUT = "timeout"
SUBTYPE_PERMISSION = "permission"
SUBTYPE_NOT_FOUND = "not_found"
SUBTYPE_RESOURCE = "resource_limit"
SUBTYPE_SYNTAX = "syntax"
SUBTYPE_OTHER = "other"
SUBTYPE_NOISY_OK = "noisy_ok"
SUBTYPE_SCRIPT = "script_error"
SUBTYPE_PATH_ESCAPE = "path_escape"
SUBTYPE_TYPE = "type_error"

_PATTERNS = [
    (SUBTYPE_PATH_ESCAPE, re.compile(r"unicodeescape|\\\\UXXXX|truncated \\\\U|unicode error", re.I)),
    (SUBTYPE_TYPE, re.compile(r"attributeerror|typeerror|keyerror|indexerror|nameerror", re.I)),
    (SUBTYPE_SCRIPT, re.compile(r"file \"<stdin>\"|file \"<string>\"|most recent call last|traceback", re.I)),
    (SUBTYPE_TIMEOUT, re.compile(r"timed?\s*out|timeout|deadline exceeded", re.I)),
    (SUBTYPE_PERMISSION, re.compile(r"permission denied|access is denied|eacces|eperm|forbidden", re.I)),
    (SUBTYPE_NOT_FOUND, re.compile(r"not found|no such file|command not found|unknown method|enoent", re.I)),
    (SUBTYPE_RESOURCE, re.compile(r"no space|out of memory|resource.?limit|too many|quota", re.I)),
    (SUBTYPE_SYNTAX, re.compile(r"syntaxerror|syntax error|invalid syntax|parse error", re.I)),
]


def classify_terminal_failure(
    *,
    stderr_tail: str = "",
    stdout_tail: str = "",
    exit_code: Optional[int] = None,
    tool_name: str = "",
) -> str:
    blob = f"{stderr_tail} {stdout_tail} {tool_name}".strip()
    for name, pat in _PATTERNS:
        if pat.search(blob):
            return name
    # P: exit 0 + no real error tokens = not a failure (historical false labels)
    low = blob.lower()
    real_err = any(
        w in low
        for w in (
            "traceback",
            "exception",
            "denied",
            "failed",
            "timed out",
            "not found",
            "syntax",
        )
    )
    try:
        ec = int(exit_code) if exit_code is not None else None
    except Exception:
        ec = None
    if not real_err and (ec is None or ec == 0):
        return SUBTYPE_NOISY_OK
    if ec is not None:
        try:
            if ec < 0:
                return SUBTYPE_TIMEOUT
            if ec == 126:
                return SUBTYPE_PERMISSION
            if ec == 127:
                return SUBTYPE_NOT_FOUND
            if ec == 137:
                return SUBTYPE_RESOURCE
        except Exception:
            pass
    return SUBTYPE_OTHER


def subtype_histogram(traces: Sequence[dict]) -> dict:
    """Count terminal-fail subtypes over decoded traces (action_params/params)."""
    hist: Counter = Counter()
    n = 0
    for t in traces or []:
        if not isinstance(t, dict):
            continue
        p = t.get("params") or t.get("action_params") or {}
        if not isinstance(p, dict):
            continue
        act = str(t.get("action") or "")
        fc = str(p.get("fail_class") or "")
        is_term = "terminal" in act or fc == "tool_error_terminal"
        if not is_term:
            continue
        if "fail" not in str(t.get("outcome") or "").lower() and fc != "tool_error_terminal":
            continue
        st = classify_terminal_failure(
            stderr_tail=str(p.get("stderr_tail") or ""),
            stdout_tail=str(p.get("stdout_tail") or ""),
            exit_code=p.get("exit_code"),
            tool_name=str(p.get("tool_name") or act),
        )
        if st == SUBTYPE_NOISY_OK:
            continue
        hist[st] += 1
        n += 1
    return {
        "n_terminal_fails": n,
        "histogram": dict(hist),
        "top": hist.most_common(6),
        "note": "terminal_subtype_from_capture",
    }


def digester_label_for_subtype(subtype: str) -> str:
    m = {
        SUBTYPE_TIMEOUT: "timeout_or_budget",
        SUBTYPE_PERMISSION: "tool_error_terminal",
        SUBTYPE_NOT_FOUND: "tool_error_terminal",
        SUBTYPE_RESOURCE: "resource_limit",
        SUBTYPE_SYNTAX: "invalid_or_compile",
    }
    return m.get(subtype or "", "tool_error_terminal")
