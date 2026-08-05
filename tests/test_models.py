"""Round-trip serialization tests for the core data models."""

from eval_harness.models import (
    CriterionVerdict,
    EvalReport,
    JudgeResult,
    Message,
)


def test_message_round_trip():
    m = Message(
        id="m1",
        channel="email",
        jurisdiction="US-CA",
        campaign_type="policy_renewal",
        sender="Acme Insurance",
        body="Hello, reply STOP to unsubscribe.",
        subject="Hi",
    )
    assert Message.from_dict(m.to_dict()) == m


def test_message_round_trip_no_subject():
    m = Message(
        id="m2",
        channel="sms",
        jurisdiction="US-TX",
        campaign_type="payment_due",
        sender="Beacon Mutual",
        body="Payment due. Reply STOP.",
    )
    restored = Message.from_dict(m.to_dict())
    assert restored == m
    assert restored.subject is None


def test_judge_result_round_trip_and_verdict_for():
    jr = JudgeResult(
        message_id="m1",
        overall="fail",
        verdicts=[
            CriterionVerdict("opt_out_present", "fail", "no opt-out"),
            CriterionVerdict("sender_identified", "pass", "brand present"),
        ],
        notes="test",
    )
    restored = JudgeResult.from_dict(jr.to_dict())
    assert restored == jr
    assert restored.verdict_for("opt_out_present") == "fail"
    assert restored.verdict_for("sender_identified") == "pass"
    assert restored.verdict_for("nonexistent") is None


def test_eval_report_round_trip_and_summary():
    report = EvalReport(
        results=[
            JudgeResult("m1", "pass", []),
            JudgeResult("m2", "fail", []),
            JudgeResult("m3", "pass", []),
        ],
        rubric_id="comms_compliance_v0",
        provider="mock",
        model="mock-deterministic",
    )
    assert report.summary() == {"pass": 2, "fail": 1, "total": 3}

    restored = EvalReport.from_dict(report.to_dict())
    assert restored.rubric_id == report.rubric_id
    assert restored.provider == report.provider
    assert restored.model == report.model
    assert [r.message_id for r in restored.results] == ["m1", "m2", "m3"]
    assert restored.summary() == report.summary()
