"""Agentic eval harness for comms compliance.

An LLM-judge harness that (1) generates synthetic outbound comms messages
(SMS/email), (2) evaluates each against a compliance rubric using an LLM-judge
that emits per-criterion pass/fail with reasoning traces, and — the key
differentiator — (3) validates the judge itself against a labeled ground-truth
dataset, reporting judge accuracy/precision/recall per criterion.

The harness runs fully offline via a deterministic mock provider (so tests need
no network or API key), with a real Anthropic provider available as an extra.
"""

__version__ = "0.1.0"
