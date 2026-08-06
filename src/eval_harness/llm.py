"""LLM providers and the deterministic mock heuristics.

The harness talks to two kinds of provider:

* ``MockProvider`` — fully offline and deterministic. It does NOT pretend to be
  an LLM; instead the generator and judge detect it via ``isinstance`` and call
  the rule-based helpers ``mock_generate`` / ``mock_judge`` defined here. This
  keeps the ``LLMProvider`` protocol clean (a real LLM only ever needs
  ``complete``) while giving tests a network-free, reproducible path.
* ``AnthropicProvider`` — a thin wrapper over the real Anthropic API. The
  ``anthropic`` dependency is imported lazily so the mock path works without it.
"""

from __future__ import annotations

import os
import re
from typing import Optional, Protocol, runtime_checkable

from .models import (
    EMAIL,
    CriterionVerdict,
    JudgeResult,
    Message,
)
from .rubric import Rubric


# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a (system, prompt) pair into text."""

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        ...


# ---------------------------------------------------------------------------
# Deterministic mock heuristics
# ---------------------------------------------------------------------------
# Phrases that read as deceptive/exaggerated claims or manufactured urgency.
_MISLEADING_PHRASES = (
    "guaranteed",
    "risk-free",
    "act now",
    "100% free",
    "winner",
    "claim your prize",
)

# Phrases that signal an established relationship / prior consent.
_CONSENT_PHRASES = ("you signed up", "as a customer", "your policy")

# A loose "postal address looks present" detector for CAN-SPAM style disclosure.
_ADDRESS_RE = re.compile(
    r"(\bP\.?\s*O\.?\s*Box\b)"
    r"|(\b\d{1,6}\s+[A-Za-z0-9.\s]{2,40}?\b"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Suite|Ste|Lane|Ln|Drive|Dr)\b)",
    re.IGNORECASE,
)


def _sender_tokens(sender: str) -> list[str]:
    """Significant (>= 3 char, alphabetic) tokens of a sender name."""
    return [t for t in re.findall(r"[A-Za-z]+", sender) if len(t) >= 3]


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)


def _v(criterion_id: str, verdict: str, reasoning: str) -> CriterionVerdict:
    return CriterionVerdict(criterion_id=criterion_id, verdict=verdict, reasoning=reasoning)


def mock_judge(message: Message, rubric: Rubric) -> JudgeResult:
    """Rule-based stand-in for an LLM judge.

    Each criterion is decided by a small, human-readable heuristic. Overall is a
    fail if any *critical* criterion fails, else a pass. Only criteria applicable
    to the message's channel are evaluated.
    """
    body = message.body or ""
    subject = message.subject or ""
    haystack = f"{subject}\n{body}".lower()

    verdicts: list[CriterionVerdict] = []
    severity_by_id = {c.id: c.severity for c in rubric.criteria}

    for criterion in rubric.applicable(message.channel):
        cid = criterion.id

        if cid == "opt_out_present":
            if "stop" in body.lower() or "unsubscribe" in haystack:
                verdicts.append(_v(cid, "pass", "Found an opt-out cue (STOP / unsubscribe)."))
            else:
                verdicts.append(_v(cid, "fail", "No opt-out mechanism (STOP / unsubscribe) found."))

        elif cid == "sender_identified":
            tokens = _sender_tokens(message.sender)
            if tokens and any(t.lower() in haystack for t in tokens):
                verdicts.append(_v(cid, "pass", f"Sender name '{message.sender}' appears in the message."))
            else:
                verdicts.append(_v(cid, "fail", f"Sender name '{message.sender}' not found in body/subject."))

        elif cid == "no_misleading_claims":
            hit = next((p for p in _MISLEADING_PHRASES if p in haystack), None)
            if hit:
                verdicts.append(_v(cid, "fail", f"Contains suspect claim/urgency phrase: '{hit}'."))
            else:
                verdicts.append(_v(cid, "pass", "No deceptive/exaggerated claim phrases detected."))

        elif cid == "consent_language":
            hit = next((p for p in _CONSENT_PHRASES if p in haystack), None)
            if hit:
                verdicts.append(_v(cid, "pass", f"References prior consent/relationship: '{hit}'."))
            else:
                verdicts.append(_v(cid, "na", "No explicit consent/relationship language to assess."))

        elif cid == "jurisdiction_disclosure":
            # Only meaningful for email (rubric restricts applicability).
            if _ADDRESS_RE.search(body):
                verdicts.append(_v(cid, "pass", "A physical postal address appears to be present."))
            else:
                verdicts.append(_v(cid, "fail", "No physical postal address / required disclosure found."))

        elif cid == "reading_level_ok":
            if "!!!" in body or _caps_ratio(body) > 0.5:
                verdicts.append(_v(cid, "fail", "Manipulative formatting (ALL CAPS or '!!!') detected."))
            else:
                verdicts.append(_v(cid, "pass", "Language is plain and non-manipulative."))

        else:
            # Unknown criterion: abstain rather than guess.
            verdicts.append(_v(cid, "na", "No heuristic defined for this criterion."))

    critical_fail = any(
        v.verdict == "fail" and severity_by_id.get(v.criterion_id) == "critical"
        for v in verdicts
    )
    overall = "fail" if critical_fail else "pass"
    return JudgeResult(message_id=message.id, overall=overall, verdicts=verdicts, notes="mock-judge")


# ---------------------------------------------------------------------------
# Deterministic mock generator library
# ---------------------------------------------------------------------------
# A hand-written library spanning compliant and non-compliant examples across
# channels, jurisdictions, and campaign types (insurance-comms flavored).
_TEMPLATES: list[dict] = [
    {
        "channel": "sms",
        "jurisdiction": "US-CA",
        "campaign_type": "policy_renewal",
        "sender": "Acme Insurance",
        "subject": None,
        "body": "Acme Insurance: your auto policy renews on 05/01. Questions? Call us. Reply STOP to unsubscribe.",
    },
    {
        "channel": "sms",
        "jurisdiction": "US-NY",
        "campaign_type": "payment_due",
        "sender": "Beacon Mutual",
        "subject": None,
        "body": "Beacon Mutual: your premium payment is due 04/15. Log in to pay. Reply STOP to opt out.",
    },
    {
        "channel": "sms",
        "jurisdiction": "US-TX",
        "campaign_type": "marketing_promo",
        "sender": "QuickCover",
        "subject": None,
        "body": "ACT NOW! QuickCover gives you GUARANTEED low rates!!! Claim your prize today!",
    },
    {
        "channel": "sms",
        "jurisdiction": "US-FL",
        "campaign_type": "marketing_promo",
        "sender": "Harbor Life",
        "subject": None,
        "body": "Save on life insurance with Harbor Life. Get a quote in minutes. Reply STOP to unsubscribe.",
    },
    {
        "channel": "sms",
        "jurisdiction": "US-WA",
        "campaign_type": "claims_update",
        "sender": "Summit Claims",
        "subject": None,
        "body": "Summit Claims: your claim #4821 has been approved. We'll email details shortly.",
    },
    {
        "channel": "email",
        "jurisdiction": "US-CA",
        "campaign_type": "policy_renewal",
        "sender": "Acme Insurance",
        "subject": "Your Acme policy renews soon",
        "body": (
            "Hi there,\n\nAs a customer, your homeowners policy with Acme Insurance renews on 06/01. "
            "No action needed to keep coverage.\n\nAcme Insurance, 123 Market Street, Suite 400, "
            "San Francisco, CA.\nUnsubscribe anytime."
        ),
    },
    {
        "channel": "email",
        "jurisdiction": "US-NY",
        "campaign_type": "payment_due",
        "sender": "Beacon Mutual",
        "subject": "Payment reminder",
        "body": (
            "You signed up for autopay, but your card on file expired. Please update it to avoid a "
            "lapse in your Beacon Mutual policy.\n\nBeacon Mutual, PO Box 22, Albany, NY.\n"
            "To unsubscribe, click here."
        ),
    },
    {
        "channel": "email",
        "jurisdiction": "US-TX",
        "campaign_type": "marketing_promo",
        "sender": "QuickCover",
        "subject": "You are a WINNER",
        "body": (
            "Congratulations, you are a WINNER! Claim your prize with a risk-free QuickCover policy. "
            "Guaranteed acceptance, act now!!!"
        ),
    },
    {
        "channel": "email",
        "jurisdiction": "US-FL",
        "campaign_type": "marketing_promo",
        "sender": "Harbor Life",
        "subject": "A quick note from Harbor Life",
        "body": (
            "Thinking about life insurance? Harbor Life makes it simple. Reply to chat with an agent.\n\n"
            "Harbor Life, 55 Ocean Ave, Miami, FL.\nUnsubscribe."
        ),
    },
    {
        "channel": "email",
        "jurisdiction": "US-WA",
        "campaign_type": "claims_update",
        "sender": "Summit Claims",
        "subject": "Update on your claim",
        "body": (
            "Your policy claim is being processed and we will follow up within 3 business days.\n\n"
            "Summit Claims. Unsubscribe from these updates."
        ),
    },
]


def mock_generate(n: int, seed: Optional[int] = None) -> list[Message]:
    """Deterministically produce ``n`` messages by cycling the template library.

    ``seed`` only rotates the starting offset, keeping output fully reproducible.
    """
    if n <= 0:
        return []
    start = (seed or 0) % len(_TEMPLATES)
    messages: list[Message] = []
    for i in range(n):
        tmpl = _TEMPLATES[(start + i) % len(_TEMPLATES)]
        messages.append(
            Message(
                id=f"gen-{i + 1:04d}",
                channel=tmpl["channel"],
                jurisdiction=tmpl["jurisdiction"],
                campaign_type=tmpl["campaign_type"],
                sender=tmpl["sender"],
                body=tmpl["body"],
                subject=tmpl["subject"],
            )
        )
    return messages


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class MockProvider:
    """Deterministic, offline provider.

    ``complete`` is intentionally a minimal stub: the generator and judge detect
    a ``MockProvider`` via ``isinstance`` and route to the rule-based helpers
    instead of calling this. It exists only to satisfy the ``LLMProvider``
    protocol.
    """

    name = "mock"
    model = "mock-deterministic"

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        # The mock never needs to synthesize free text; return an empty JSON
        # object so any accidental real-path caller degrades gracefully.
        return "{}"


class AnthropicProvider:
    """Thin wrapper over the real Anthropic Messages API.

    The ``anthropic`` package is imported lazily inside ``__init__`` so that the
    mock path (and the test suite) never requires the optional dependency.
    """

    name = "anthropic"

    def __init__(self, model: Optional[str] = None) -> None:
        try:
            import anthropic  # noqa: F401  (lazy import; optional dependency)
        except ImportError as exc:  # pragma: no cover - exercised only with real provider
            raise RuntimeError(
                "The 'anthropic' package is required for EVAL_PROVIDER=anthropic. "
                "Install it with: pip install 'agentic-eval-harness[anthropic]'"
            ) from exc

        self._anthropic = anthropic
        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL", "claude-sonnet-5")

    def complete(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate all text blocks in the response.
        parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
        return "".join(parts)


def get_provider(name: Optional[str] = None) -> LLMProvider:
    """Return a provider by name, defaulting to the ``EVAL_PROVIDER`` env var.

    Defaults to the offline ``MockProvider`` when nothing is configured.
    """
    resolved = (name or os.environ.get("EVAL_PROVIDER", "mock")).strip().lower()
    if resolved == "anthropic":
        return AnthropicProvider()
    if resolved == "mock":
        return MockProvider()
    raise ValueError(f"Unknown provider '{resolved}'. Use 'mock' or 'anthropic'.")


def provider_label(provider: LLMProvider) -> tuple[str, str]:
    """Return a ``(name, model)`` label pair for a provider, for reports."""
    name = getattr(provider, "name", provider.__class__.__name__)
    model = getattr(provider, "model", "unknown")
    return name, model
