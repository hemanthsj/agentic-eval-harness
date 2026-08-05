"""Scoring the judge against ground truth — the credibility centerpiece.

Given the judge's results and a labeled dataset, we treat "fail" as the positive
class (i.e. detecting a compliance violation) and compute per-criterion accuracy,
precision, recall, F1, and a tp/fp/tn/fn confusion count — plus overall accuracy
across all criterion judgments. "na" is treated as "not a violation": it is
excluded from the fail-class precision/recall numerator, but still counts toward
accuracy via exact match.
"""

from __future__ import annotations

import json

from .models import LabeledMessage, JudgeResult


def load_labeled(path: str) -> list[LabeledMessage]:
    """Load labeled messages from a JSONL file (one object per line)."""
    labeled: list[LabeledMessage] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            labeled.append(LabeledMessage.from_dict(json.loads(line)))
    return labeled


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_judge(results: list[JudgeResult], labels: list[LabeledMessage]) -> dict:
    """Score judge ``results`` against ground-truth ``labels``.

    Returns a structured dict with overall accuracy and per-criterion metrics.
    """
    by_message = {r.message_id: r for r in results}

    # Per-criterion running tallies.
    crit: dict[str, dict] = {}

    def bucket(cid: str) -> dict:
        return crit.setdefault(
            cid, {"n": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        )

    total = 0
    total_correct = 0

    for label in labels:
        result = by_message.get(label.message.id)
        if result is None:
            continue
        for cid, expected in label.expected.items():
            predicted = result.verdict_for(cid)
            if predicted is None:
                # Judge did not evaluate this criterion (e.g. not applicable);
                # skip rather than penalize.
                continue

            b = bucket(cid)
            b["n"] += 1
            total += 1
            correct = predicted == expected
            b["correct"] += int(correct)
            total_correct += int(correct)

            # Confusion counts with "fail" as the positive (violation) class.
            exp_pos = expected == "fail"
            pred_pos = predicted == "fail"
            if exp_pos and pred_pos:
                b["tp"] += 1
            elif not exp_pos and pred_pos:
                b["fp"] += 1
            elif not exp_pos and not pred_pos:
                b["tn"] += 1
            else:  # exp_pos and not pred_pos
                b["fn"] += 1

    criteria_out: dict[str, dict] = {}
    for cid, b in sorted(crit.items()):
        tp, fp, fn = b["tp"], b["fp"], b["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        criteria_out[cid] = {
            "n": b["n"],
            "accuracy": (b["correct"] / b["n"]) if b["n"] else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "tp": tp,
            "fp": fp,
            "tn": b["tn"],
            "fn": fn,
        }

    return {
        "overall_accuracy": (total_correct / total) if total else 0.0,
        "n_judgments": total,
        "criteria": criteria_out,
    }


def format_scorecard(scores: dict) -> str:
    """Render a readable Markdown-ish table of judge metrics."""
    lines: list[str] = []
    lines.append("Judge scorecard (vs ground truth)")
    lines.append("=" * 72)
    lines.append(
        f"Overall accuracy: {scores['overall_accuracy']:.3f}  "
        f"({scores['n_judgments']} judgments)"
    )
    lines.append("")
    header = f"{'criterion':<26}{'n':>4}{'acc':>7}{'prec':>7}{'rec':>7}{'f1':>7}"
    lines.append(header)
    lines.append("-" * len(header))
    for cid, m in scores["criteria"].items():
        lines.append(
            f"{cid:<26}{m['n']:>4}{m['accuracy']:>7.2f}"
            f"{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}"
        )
    lines.append("")
    lines.append("Positive class = 'fail' (detecting a compliance violation).")
    return "\n".join(lines)
