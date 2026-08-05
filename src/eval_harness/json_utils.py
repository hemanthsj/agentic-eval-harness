"""Robust JSON extraction from LLM output.

Real models sometimes wrap JSON in markdown fences or add a sentence before the
payload. This strips fences and, failing that, slices out the first balanced
JSON object/array so the caller's ``json.loads`` has a clean string to parse.
"""

from __future__ import annotations

import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> str:
    """Return the most likely JSON substring from ``text``."""
    text = (text or "").strip()

    # 1. Prefer a fenced ```json ... ``` block if present.
    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()

    # 2. Otherwise slice from the first opening bracket to its matching close.
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    start = min(starts)
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Unbalanced — return the tail from the first bracket and let json.loads error.
    return text[start:]
