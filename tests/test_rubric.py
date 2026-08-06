"""Tests for rubric loading and channel applicability filtering."""

import os

from eval_harness.rubric import Rubric

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUBRIC = os.path.join(_REPO_ROOT, "rubrics", "comms_compliance.yaml")


def test_load_rubric():
    rubric = Rubric.load(_RUBRIC)
    assert rubric.id == "comms_compliance_v0"
    assert len(rubric.criteria) == 6
    ids = {c.id for c in rubric.criteria}
    assert "opt_out_present" in ids
    assert "jurisdiction_disclosure" in ids


def test_applicable_filters_by_channel():
    rubric = Rubric.load(_RUBRIC)

    email_ids = {c.id for c in rubric.applicable("email")}
    sms_ids = {c.id for c in rubric.applicable("sms")}

    # jurisdiction_disclosure is email-only.
    assert "jurisdiction_disclosure" in email_ids
    assert "jurisdiction_disclosure" not in sms_ids

    # opt_out_present applies to both channels.
    assert "opt_out_present" in email_ids
    assert "opt_out_present" in sms_ids

    # Email has all six; SMS has five (all but jurisdiction_disclosure).
    assert len(email_ids) == 6
    assert len(sms_ids) == 5
