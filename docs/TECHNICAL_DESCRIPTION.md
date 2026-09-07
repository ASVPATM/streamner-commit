# Technical details

Testing is paused after v10. The [README](../README.md#technical-details) gives
the short overview; the [testing history](TESTING_HISTORY.md) explains v1–v10.

## What the code does

1. Append exact text chunks to the pinned GLiNER StreamingSpan model.
2. Save the scores the model generates or revises.
3. Replay those scores through commitment policies, without rerunning the model.
4. Compare committed entities with annotations and measure how much context was needed.

Once committed, a prediction cannot be extended, relabeled, or retracted. The model's
internal predictions can still change. Deployable policies never see future text or annotations.

## Current baseline and historical policies

The retained reference is the two-update buffer at threshold 0.95. It holds the
first eligible prediction at the visible text edge for two arriving observations,
then releases current candidates through the same overlap resolver. Extensions
do not restart the deadline. Explicit-close output is reported separately.
Two observations are not necessarily two model words.

V9 selected threshold 0.90 as a candidate; v10 compared it with 0.95 on two further
development blocks. It improved masking F2 in both, but one block lost precision
and strict F1. It has not replaced the reference. The detector weights are unchanged.

Boundary-readiness checks were diagnostic. V8 compared fixed extra waiting with
small, parent-separated learned assessors; all learned branches fell back to the
buffer under their calibration rules. V9's learned admission filters did not
establish an advantage over the simple 0.90 candidate.

**EMA** smooths confidence only when the model produces a fresh score:

`smoothed = alpha × new_score + (1 − alpha) × previous_smoothed`

It becomes eligible to commit above a threshold. A larger alpha reacts faster.
The first score initializes the average, so EMA does not necessarily wait for repeated evidence.

**StabilityGate** requires sufficient confidence and score observations, limited recent
score movement, separation between competing labels, and no sufficiently better visible
longer span. Every enabled check must pass. Stricter settings can miss more entities.
Both policies use the same overlap resolver and irreversible-commitment rules.

Other implemented comparisons: fixed threshold, fixed lag, snapshot patience, and
rescore patience. The future-aware oracle is diagnostic only.

V2 compared fixed thresholds, existing policies and a conditional gate on two
development folds. The conditional gate did not establish an advantage over the
simpler threshold. EMA and StabilityGate remain historical comparisons.

## Running the public code

Python 3.12 is required. The locked main environment targets Apple Silicon:

```bash
uv sync --locked --dev
MLX_ENABLE_TF32=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run pytest -q
```

Checkpoint-backed tests require separately obtained assets and explicit opt-in flags.
GLiNER/PyTorch belong in a separate reference environment. Linux can replay existing
traces in a separately prepared environment; the main MLX dependency set is Mac-specific.

Entry points are in `scripts/`: `run_trace_generation.py`, `run_policy_sweep.py`,
`run_benchmark.py`, `make_figures.py`, and `make_tables.py`. Use `--help` first.
The full sweep can be expensive. It supports bounded workers, progress, and checkpoints;
the regular benchmark is not checkpointed.

The later v1–v10 studies used separate frozen diagnostic bundles. Their runner
loads one example at a time, reports progress, checks input/code identities and
saves checkpoints. Those snapshots and their private inputs are not all included
in the public repository; the commands above do not reproduce the latest study.

## Current limits

The [latest result](../results/replication/2026-09-06/README.md) is v10: 96 reused
development sentences, two fixed policies, chunk 1. The larger
[v6 run](../results/overnight/2026-09-06/README.md) covered 1,200 reused test sentences
and four chunk sizes. The original grid search remains incomplete; these later
tests do not complete it or establish the best possible settings.

Some delay averages count only correct commitments. Compare recall, all-gold
deadline coverage and per-task results alongside them. Source-input chunks and
model-word coordinates differ, and replay time is not inference latency.

The v3 boundary conflict audit replayed one unchanged baseline and recorded which
overlapping spans prevented acceptance. V4 tested one/two-update buffering on
those same examples. V5 froze the two-update buffer and checked different
diagnostic parents. These findings led to the current reference; they are not
three independent confirmation samples. See the testing history for later reuse.

This is a commitment evaluator, not an end-to-end redaction service. Releasing
raw text before a later correct detection can still expose PII. No production
privacy guarantee or superiority over the upstream model has been established.

The model, dataset, and metric settings are in `configs/`; code is under
`src/streamner_commit/`; tests are under `tests/`. No corpus text, weights, or raw traces
are bundled. Historical pilot producer IDs are retained in the result metadata; they
are not commits in the clean public history.

[Credits](../README.md#credits) · [Data, dependencies and other references](THIRD_PARTY.md)
