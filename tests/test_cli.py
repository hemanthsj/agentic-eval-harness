"""End-to-end CLI test — the one-command demo runs offline and reports."""

import io
import os
from contextlib import redirect_stdout

from eval_harness.cli import main

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_run_command_exits_zero_and_prints_scorecard(monkeypatch):
    monkeypatch.setenv("EVAL_PROVIDER", "mock")
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(["run"])
    output = buf.getvalue()

    assert code == 0
    assert "Judge scorecard" in output
    assert "Provider: mock" in output
    assert "Overall accuracy" in output


def test_generate_and_judge_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("EVAL_PROVIDER", "mock")
    messages_path = tmp_path / "messages.jsonl"
    report_path = tmp_path / "report.json"

    with redirect_stdout(io.StringIO()):
        assert main(["generate", "--n", "4", "--out", str(messages_path)]) == 0
        assert main(["judge", "--messages", str(messages_path), "--out", str(report_path)]) == 0

    assert messages_path.exists()
    assert report_path.exists()
