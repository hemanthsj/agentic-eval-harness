"""Report rendering: JSON on disk and readable console output.

The console view surfaces each message's overall verdict and the failing
criteria with their reasoning traces — the "reasoning traces" the README
promises — using simple ✓/✗ symbols.
"""

from __future__ import annotations

import json
import os

from .models import EvalReport
from .rubric import Rubric

_PASS = "✓"  # ✓
_FAIL = "✗"  # ✗


def write_json(report: EvalReport, path: str) -> None:
    """Write an ``EvalReport`` to ``path`` as pretty JSON."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _symbol(verdict: str) -> str:
    return _PASS if verdict == "pass" else (_FAIL if verdict == "fail" else "-")


def format_console(report: EvalReport, rubric: Rubric) -> str:
    """Render an ``EvalReport`` as readable console text with reasoning traces."""
    titles = {c.id: c.title for c in rubric.criteria}
    lines: list[str] = []
    lines.append(f"Eval report — rubric '{report.rubric_id}' "
                 f"(provider={report.provider}, model={report.model})")
    lines.append("=" * 72)

    for r in report.results:
        overall_sym = _PASS if r.overall == "pass" else _FAIL
        lines.append(f"{overall_sym} [{r.overall.upper()}] {r.message_id}")
        for v in r.verdicts:
            title = titles.get(v.criterion_id, v.criterion_id)
            lines.append(f"    {_symbol(v.verdict)} {v.criterion_id} ({title}): {v.reasoning}")
        lines.append("")

    s = report.summary()
    lines.append("-" * 72)
    lines.append(f"Summary: {s['pass']} pass / {s['fail']} fail (of {s['total']} messages)")
    return "\n".join(lines)
