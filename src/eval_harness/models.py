"""Core data models for the eval harness.

Everything here is a plain dataclass with explicit ``to_dict`` / ``from_dict``
helpers so the whole object graph round-trips cleanly through JSON — no pydantic,
no magic. That keeps the reports human-readable and the tests trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------
# We keep channels as simple string constants rather than an Enum so they
# serialize to JSON as-is and read naturally in YAML rubrics and JSONL data.
SMS = "sms"
EMAIL = "email"
CHANNELS = frozenset({SMS, EMAIL})

Channel = Literal["sms", "email"]
Verdict = Literal["pass", "fail", "na"]
Overall = Literal["pass", "fail"]


@dataclass
class Message:
    """A single synthetic outbound comms message under evaluation."""

    id: str
    channel: str
    jurisdiction: str
    campaign_type: str
    sender: str
    body: str
    subject: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "channel": self.channel,
            "jurisdiction": self.jurisdiction,
            "campaign_type": self.campaign_type,
            "sender": self.sender,
            "body": self.body,
            "subject": self.subject,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            id=data["id"],
            channel=data["channel"],
            jurisdiction=data["jurisdiction"],
            campaign_type=data["campaign_type"],
            sender=data["sender"],
            body=data["body"],
            subject=data.get("subject"),
        )


@dataclass
class CriterionVerdict:
    """The judge's verdict on one rubric criterion, with a reasoning trace."""

    criterion_id: str
    verdict: Verdict
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "criterion_id": self.criterion_id,
            "verdict": self.verdict,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CriterionVerdict":
        return cls(
            criterion_id=data["criterion_id"],
            verdict=data["verdict"],
            reasoning=data["reasoning"],
        )


@dataclass
class JudgeResult:
    """The judge's full assessment of one message."""

    message_id: str
    overall: Overall
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    notes: str = ""

    def verdict_for(self, criterion_id: str) -> Optional[str]:
        """Return the verdict string for ``criterion_id``, or ``None`` if the
        criterion was not evaluated (e.g. it does not apply to this channel)."""
        for v in self.verdicts:
            if v.criterion_id == criterion_id:
                return v.verdict
        return None

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "overall": self.overall,
            "verdicts": [v.to_dict() for v in self.verdicts],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JudgeResult":
        return cls(
            message_id=data["message_id"],
            overall=data["overall"],
            verdicts=[CriterionVerdict.from_dict(v) for v in data.get("verdicts", [])],
            notes=data.get("notes", ""),
        )


@dataclass
class LabeledMessage:
    """A message paired with ground-truth per-criterion verdicts.

    ``expected`` maps ``criterion_id`` -> ``"pass" | "fail" | "na"``. This is the
    human-authored answer key we score the judge against.
    """

    message: Message
    expected: dict[str, str]

    def to_dict(self) -> dict:
        return {"message": self.message.to_dict(), "expected": dict(self.expected)}

    @classmethod
    def from_dict(cls, data: dict) -> "LabeledMessage":
        return cls(
            message=Message.from_dict(data["message"]),
            expected=dict(data["expected"]),
        )


@dataclass
class EvalReport:
    """The output of judging a batch of messages against a rubric."""

    results: list[JudgeResult]
    rubric_id: str
    provider: str
    model: str

    def summary(self) -> dict:
        """Return counts of overall pass/fail across all judged messages."""
        passed = sum(1 for r in self.results if r.overall == "pass")
        failed = sum(1 for r in self.results if r.overall == "fail")
        return {"pass": passed, "fail": failed, "total": len(self.results)}

    def to_dict(self) -> dict:
        return {
            "rubric_id": self.rubric_id,
            "provider": self.provider,
            "model": self.model,
            "summary": self.summary(),
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EvalReport":
        return cls(
            results=[JudgeResult.from_dict(r) for r in data.get("results", [])],
            rubric_id=data["rubric_id"],
            provider=data["provider"],
            model=data["model"],
        )
