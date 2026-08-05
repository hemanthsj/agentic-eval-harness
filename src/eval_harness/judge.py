"""The LLM-judge.

With the mock provider the judge delegates to the deterministic ``mock_judge``
heuristics. With a real provider it builds a strict-reviewer prompt, calls the
model, parses the JSON verdicts, and computes the overall pass/fail (a fail if
any *critical* criterion fails).
"""

from __future__ import annotations

import json
from typing import Optional

from .json_utils import extract_json
from .llm import LLMProvider, MockProvider, mock_judge
from .models import CriterionVerdict, JudgeResult, Message
from .rubric import Criterion, Rubric

_JUDGE_SYSTEM = (
    "You are a strict, fair compliance reviewer for outbound marketing/servicing "
    "messages (SMS and email). You evaluate a single message against a fixed set "
    "of criteria and output STRICT JSON only — no prose, no markdown. For each "
    "criterion, emit a verdict of 'pass', 'fail', or 'na' with a short reasoning "
    "trace. Be precise and cite the specific text that drove your verdict."
)


def _valid_verdict(value: str) -> str:
    value = (value or "").strip().lower()
    return value if value in ("pass", "fail", "na") else "na"


class Judge:
    """Evaluates messages against a rubric using the configured provider."""

    def __init__(self, provider: LLMProvider, rubric: Rubric) -> None:
        self.provider = provider
        self.rubric = rubric

    def judge(self, message: Message) -> JudgeResult:
        if isinstance(self.provider, MockProvider):
            return mock_judge(message, self.rubric)
        return self._judge_real(message)

    def judge_all(self, messages: list[Message]) -> list[JudgeResult]:
        return [self.judge(m) for m in messages]

    # -- real-LLM path ------------------------------------------------------
    def _judge_real(self, message: Message) -> JudgeResult:
        criteria = self.rubric.applicable(message.channel)
        prompt = self._build_prompt(message, criteria)
        raw = self.provider.complete(_JUDGE_SYSTEM, prompt, temperature=0.0, max_tokens=1536)
        return self._parse(message, criteria, raw)

    def _build_prompt(self, message: Message, criteria: list[Criterion]) -> str:
        crit_lines = "\n".join(
            f"- {c.id} ({c.severity}): {c.title} — {c.description}" for c in criteria
        )
        return (
            f"MESSAGE\n"
            f"  channel: {message.channel}\n"
            f"  jurisdiction: {message.jurisdiction}\n"
            f"  campaign_type: {message.campaign_type}\n"
            f"  sender: {message.sender}\n"
            f"  subject: {message.subject or ''}\n"
            f"  body: {message.body}\n\n"
            f"CRITERIA\n{crit_lines}\n\n"
            "Return STRICT JSON with this shape:\n"
            '{"verdicts": [{"criterion_id": "<id>", "verdict": "pass|fail|na", '
            '"reasoning": "<short>"}], "notes": "<optional>"}\n'
            "Include exactly one entry per criterion listed above."
        )

    def _parse(self, message: Message, criteria: list[Criterion], raw: str) -> JudgeResult:
        data = json.loads(extract_json(raw))
        by_id = {v.get("criterion_id"): v for v in data.get("verdicts", [])}
        severity_by_id = {c.id: c.severity for c in criteria}

        verdicts: list[CriterionVerdict] = []
        for c in criteria:
            entry = by_id.get(c.id, {})
            verdicts.append(
                CriterionVerdict(
                    criterion_id=c.id,
                    verdict=_valid_verdict(entry.get("verdict", "na")),
                    reasoning=str(entry.get("reasoning", "")).strip(),
                )
            )

        critical_fail = any(
            v.verdict == "fail" and severity_by_id.get(v.criterion_id) == "critical"
            for v in verdicts
        )
        overall = "fail" if critical_fail else "pass"
        return JudgeResult(
            message_id=message.id,
            overall=overall,
            verdicts=verdicts,
            notes=str(data.get("notes", "")).strip(),
        )
