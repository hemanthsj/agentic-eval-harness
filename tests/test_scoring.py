"""Tests for scoring the judge against the labeled ground-truth dataset."""

import os

from eval_harness.judge import Judge
from eval_harness.llm import MockProvider
from eval_harness.rubric import Rubric
from eval_harness.scoring import load_labeled, score_judge

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUBRIC = os.path.join(_REPO_ROOT, "rubrics", "comms_compliance.yaml")
_LABELS = os.path.join(_REPO_ROOT, "data", "labeled_messages.jsonl")


def test_mock_judge_scores_high_against_ground_truth():
    rubric = Rubric.load(_RUBRIC)
    labels = load_labeled(_LABELS)
    assert len(labels) >= 10  # a meaningful dataset

    judge = Judge(MockProvider(), rubric)
    results = judge.judge_all([lm.message for lm in labels])
    scores = score_judge(results, labels)

    assert scores["overall_accuracy"] >= 0.8


def test_score_dict_shape():
    rubric = Rubric.load(_RUBRIC)
    labels = load_labeled(_LABELS)
    results = Judge(MockProvider(), rubric).judge_all([lm.message for lm in labels])
    scores = score_judge(results, labels)

    assert set(scores.keys()) == {"overall_accuracy", "n_judgments", "criteria"}
    assert scores["n_judgments"] > 0

    # Every per-criterion entry carries the full metric set.
    for cid, metrics in scores["criteria"].items():
        assert set(metrics.keys()) == {
            "n", "accuracy", "precision", "recall", "f1", "tp", "fp", "tn", "fn",
        }
