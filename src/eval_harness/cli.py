"""Command-line interface for the eval harness.

Subcommands:
  generate  — generate synthetic messages to JSONL
  judge     — judge messages against a rubric, write an EvalReport, print summary
  score     — score a judge report against a labeled dataset
  run       — end-to-end demo on the built-in labeled dataset (one-command demo)

The provider is chosen via the EVAL_PROVIDER environment variable (default
"mock"), and the provider/model in use is printed on every run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from .generator import MessageGenerator
from .judge import Judge
from .llm import get_provider, provider_label
from .models import EvalReport, Message
from .report import format_console, write_json
from .rubric import Rubric
from .scoring import format_scorecard, load_labeled, score_judge

# Resolve bundled defaults relative to the repo root (works for editable installs).
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
_DEFAULT_RUBRIC = os.path.join(_REPO_ROOT, "rubrics", "comms_compliance.yaml")
_DEFAULT_LABELS = os.path.join(_REPO_ROOT, "data", "labeled_messages.jsonl")


def _load_messages(path: str) -> list[Message]:
    messages: list[Message] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            messages.append(Message.from_dict(json.loads(line)))
    return messages


def _write_messages(messages: list[Message], path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for m in messages:
            fh.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")


def _print_provider_banner(name: str, model: str) -> None:
    print(f"Provider: {name}  (model: {model})")
    print()


def cmd_generate(args: argparse.Namespace) -> int:
    provider = get_provider()
    name, model = provider_label(provider)
    _print_provider_banner(name, model)
    messages = MessageGenerator(provider).generate(args.n, seed=args.seed)
    _write_messages(messages, args.out)
    print(f"Generated {len(messages)} messages -> {args.out}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    provider = get_provider()
    name, model = provider_label(provider)
    _print_provider_banner(name, model)

    rubric = Rubric.load(args.rubric)
    messages = _load_messages(args.messages)
    results = Judge(provider, rubric).judge_all(messages)
    report = EvalReport(results=results, rubric_id=rubric.id, provider=name, model=model)

    print(format_console(report, rubric))
    if args.out:
        write_json(report, args.out)
        print(f"\nWrote report -> {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    with open(args.report, "r", encoding="utf-8") as fh:
        report = EvalReport.from_dict(json.load(fh))
    labels = load_labeled(args.labels)
    scores = score_judge(report.results, labels)
    print(format_scorecard(scores))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """End-to-end demo: judge the built-in labeled dataset, then score the judge."""
    provider = get_provider()
    name, model = provider_label(provider)
    _print_provider_banner(name, model)

    rubric = Rubric.load(args.rubric)
    labels = load_labeled(args.labels)
    messages = [lm.message for lm in labels]

    results = Judge(provider, rubric).judge_all(messages)
    report = EvalReport(results=results, rubric_id=rubric.id, provider=name, model=model)

    print(format_console(report, rubric))
    print()
    scores = score_judge(results, labels)
    print(format_scorecard(scores))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eval-harness",
        description="LLM-judge harness for comms compliance (and validating the judge).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate synthetic messages to JSONL.")
    p_gen.add_argument("--n", type=int, default=10, help="Number of messages to generate.")
    p_gen.add_argument("--out", required=True, help="Output JSONL path.")
    p_gen.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    p_gen.set_defaults(func=cmd_generate)

    p_judge = sub.add_parser("judge", help="Judge messages against a rubric.")
    p_judge.add_argument("--messages", required=True, help="Input messages JSONL.")
    p_judge.add_argument("--rubric", default=_DEFAULT_RUBRIC, help="Rubric YAML path.")
    p_judge.add_argument("--out", default=None, help="Optional EvalReport JSON output path.")
    p_judge.set_defaults(func=cmd_judge)

    p_score = sub.add_parser("score", help="Score a judge report vs ground truth.")
    p_score.add_argument("--report", required=True, help="EvalReport JSON path.")
    p_score.add_argument("--labels", default=_DEFAULT_LABELS, help="Labeled JSONL path.")
    p_score.set_defaults(func=cmd_score)

    p_run = sub.add_parser("run", help="End-to-end demo on the built-in labeled dataset.")
    p_run.add_argument("--rubric", default=_DEFAULT_RUBRIC, help="Rubric YAML path.")
    p_run.add_argument("--labels", default=_DEFAULT_LABELS, help="Labeled JSONL path.")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
