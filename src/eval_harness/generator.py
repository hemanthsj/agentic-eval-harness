"""Synthetic message generation.

With the mock provider this is a deterministic library of hand-written
insurance-comms templates. With a real provider it prompts the LLM to emit a
JSON array of messages and parses the result robustly.
"""

from __future__ import annotations

import json
from typing import Optional

from .json_utils import extract_json
from .llm import LLMProvider, MockProvider, mock_generate
from .models import CHANNELS, Message

_GEN_SYSTEM = (
    "You generate synthetic outbound comms messages (SMS and email) for testing "
    "a compliance eval harness. Produce a realistic mix of compliant and "
    "non-compliant examples across jurisdictions and campaign types."
)

_GEN_PROMPT_TEMPLATE = (
    "Generate exactly {n} synthetic insurance-comms messages.\n"
    "Return STRICT JSON only: an array of objects, each with keys "
    '"channel" ("sms" or "email"), "jurisdiction", "campaign_type", "sender", '
    '"body", and "subject" (null for sms). Do not wrap in markdown.'
)


class MessageGenerator:
    """Generates synthetic messages via the configured provider."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def generate(self, n: int, seed: Optional[int] = None) -> list[Message]:
        if isinstance(self.provider, MockProvider):
            return mock_generate(n, seed=seed)
        return self._generate_real(n)

    def _generate_real(self, n: int) -> list[Message]:
        raw = self.provider.complete(
            _GEN_SYSTEM,
            _GEN_PROMPT_TEMPLATE.format(n=n),
            temperature=0.7,
            max_tokens=4096,
        )
        payload = extract_json(raw)
        items = json.loads(payload)
        if not isinstance(items, list):
            raise ValueError("Generator expected a JSON array of messages.")

        messages: list[Message] = []
        for i, item in enumerate(items[:n]):
            channel = str(item.get("channel", "email")).lower()
            if channel not in CHANNELS:
                channel = "email"
            messages.append(
                Message(
                    id=f"gen-{i + 1:04d}",
                    channel=channel,
                    jurisdiction=str(item.get("jurisdiction", "unknown")),
                    campaign_type=str(item.get("campaign_type", "unknown")),
                    sender=str(item.get("sender", "Unknown Sender")),
                    body=str(item.get("body", "")),
                    subject=item.get("subject"),
                )
            )
        return messages
