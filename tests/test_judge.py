"""Tests for the mock judge heuristics."""

import os

from eval_harness.judge import Judge
from eval_harness.llm import MockProvider
from eval_harness.models import Message
from eval_harness.rubric import Rubric

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUBRIC = os.path.join(_REPO_ROOT, "rubrics", "comms_compliance.yaml")


def _judge():
    return Judge(MockProvider(), Rubric.load(_RUBRIC))


def test_missing_opt_out_fails_critically():
    msg = Message(
        id="m1",
        channel="sms",
        jurisdiction="US-WA",
        campaign_type="claims_update",
        sender="Summit Claims",
        body="Summit Claims: your claim has been approved. Details to follow.",
    )
    result = _judge().judge(msg)
    assert result.verdict_for("opt_out_present") == "fail"
    assert result.overall == "fail"  # opt_out_present is critical


def test_compliant_message_passes():
    msg = Message(
        id="m2",
        channel="sms",
        jurisdiction="US-CA",
        campaign_type="policy_renewal",
        sender="Acme Insurance",
        body="Acme Insurance: your policy renews 05/01. Reply STOP to unsubscribe.",
    )
    result = _judge().judge(msg)
    assert result.verdict_for("opt_out_present") == "pass"
    assert result.verdict_for("sender_identified") == "pass"
    assert result.overall == "pass"


def test_misleading_claim_fails_critically():
    msg = Message(
        id="m3",
        channel="sms",
        jurisdiction="US-TX",
        campaign_type="marketing_promo",
        sender="QuickCover",
        body="QuickCover GUARANTEED low rates! Reply STOP to unsubscribe.",
    )
    result = _judge().judge(msg)
    assert result.verdict_for("no_misleading_claims") == "fail"
    assert result.overall == "fail"  # no_misleading_claims is critical
