# agentic-eval-harness

Build a system that generates synthetic outbound messages, runs them through an LLM-judge against a rubric (disclosures, opt-out language, jurisdiction rules), and reports pass/fail with reasoning traces — then **validates the judge itself** against a labeled ground-truth dataset.

An LLM-judge for comms-compliance is only as trustworthy as the judge. This harness treats the judge as a system under test: it scores the judge's per-criterion verdicts against a human-labeled answer key and reports accuracy, precision, and recall. It runs fully offline through a deterministic mock provider (no API key, no network) and swaps in a real Anthropic judge with one environment variable.

## Why

Evals are the hard part. Once you have a rubric and an LLM-judge, the tempting move is to trust its verdicts and ship. But a judge you don't trust is worse than no judge — it launders bad decisions behind a veneer of rigor. The differentiator here is the third stage: we hold the judge accountable to labeled ground truth and surface exactly where it over- or under-flags. "How I built an LLM-judge I actually trust" is the whole point.

## Architecture

```
  Generator ──► Messages ──► Judge (+ Rubric) ──► Results ──► Scorecard
 (synthetic     (SMS/email)   per-criterion       (verdicts   (vs labeled
  library or                  pass/fail with       + reasoning  ground truth:
  real LLM)                   reasoning traces)     traces)      acc/prec/recall)
                                    ▲                                ▲
                                    │                                │
                              Rubric (YAML)                 Labeled dataset (JSONL)
```

- **Generator** produces synthetic SMS/email messages (a deterministic library under the mock provider; JSON-emitting prompts under a real LLM).
- **Judge** evaluates each message against the **Rubric**, emitting a `pass` / `fail` / `na` verdict per criterion with a short reasoning trace, and an overall verdict (fail if any *critical* criterion fails).
- **Scorecard** compares the judge's verdicts against a labeled ground-truth dataset and reports per-criterion metrics.

## Quickstart

```bash
pip install -e .
eval-harness run          # fully offline with the mock provider — no API key needed
```

Example output (abridged):

```
Provider: mock  (model: mock-deterministic)

✗ [FAIL] lbl-0003
    ✗ opt_out_present: No opt-out mechanism (STOP / unsubscribe) found.
    ✓ sender_identified: Sender name 'QuickCover' appears in the message.
    ✗ no_misleading_claims: Contains suspect claim/urgency phrase: 'guaranteed'.
    ✗ reading_level_ok: Manipulative formatting (ALL CAPS or '!!!') detected.

Judge scorecard (vs ground truth)
========================================================================
Overall accuracy: 0.970  (66 judgments)

criterion                    n    acc   prec    rec     f1
----------------------------------------------------------
no_misleading_claims        12   0.92   0.67   1.00   0.80
opt_out_present             12   1.00   1.00   1.00   1.00
...
Positive class = 'fail' (detecting a compliance violation).
```

Other subcommands:

```bash
eval-harness generate --n 10 --out messages.jsonl
eval-harness judge --messages messages.jsonl --out report.json
eval-harness score --report report.json
```

## Using the real Anthropic judge

Set the provider and key, then run any command exactly as above:

```bash
export EVAL_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export EVAL_JUDGE_MODEL=claude-sonnet-5   # optional; this is the default
pip install -e '.[anthropic]'
eval-harness run
```

The real judge builds a strict-reviewer prompt embedding the message and applicable criteria, requests STRICT JSON, and parses per-criterion verdicts (with robust markdown-fence stripping). Everything else — the rubric, the scorecard, the reasoning traces — is identical to the mock path. See `.env.example` for configuration.

## The rubric

Rubrics are YAML (`rubrics/comms_compliance.yaml`). Each criterion has an `id`, `title`, `description`, `severity` (`critical` / `major` / `minor`), and an optional `applies_to` list of channels (empty = all). The judge only evaluates criteria applicable to a message's channel, and the overall verdict fails if any *critical* criterion fails.

The v0 rubric (`comms_compliance_v0`) ships six criteria: opt-out present (critical), sender identified (major), no misleading claims (critical), consent language (major), jurisdiction disclosure (major, email-only), and plain reading level (minor).

**Adding a criterion:** append an entry to the YAML. If you use the mock judge, add a matching heuristic branch in `mock_judge` (`src/eval_harness/llm.py`); the real LLM judge needs no code change — it reads the criterion straight from the rubric.

## Validating the judge

This is the credibility centerpiece. `data/labeled_messages.jsonl` pairs each message with human-authored ground-truth verdicts. `score_judge` treats `fail` (a detected violation) as the positive class and, per criterion, computes accuracy, precision, recall, F1, and a tp/fp/tn/fn confusion count — plus overall accuracy across every criterion judgment. `na` is treated as "not a violation": excluded from the fail-class precision/recall numerator but still scored for exact-match accuracy.

The bundled dataset includes deliberately tricky cases (e.g. a benign "act now" in a servicing email, and an implied-but-unnamed sender) where the heuristic judge disagrees with ground truth — so the scorecard shows realistic sub-1.0 precision rather than a suspiciously perfect judge.

## Roadmap

- **v0 (this):** synthetic generation, rubric-driven LLM-judge with reasoning traces, judge validation against labeled ground truth, offline mock + real Anthropic providers.
- **Next:** judge calibration (confidence-aware verdicts), inter-judge agreement across models, more jurisdictions and channel types, richer rubric versioning, and an HTML report.

## Disclaimer

This is a heuristic compliance *aid* for engineering and evaluation workflows. It is **NOT legal advice** and does not constitute a complete compliance program. The rubric criteria and mock heuristics are illustrative. Consult qualified counsel for actual compliance requirements.
