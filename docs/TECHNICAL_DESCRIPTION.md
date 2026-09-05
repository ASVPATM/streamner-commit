# Technical details

## What the code does

1. Append exact text chunks to the pinned GLiNER StreamingSpan model.
2. Save the scores the model generates or revises.
3. Replay those scores through commitment policies, without rerunning the model.
4. Compare committed entities with annotations and measure how much context was needed.

Once committed, a prediction cannot be extended, relabeled, or retracted. The model's
internal predictions can still change. Deployable policies never see future text or annotations.

## Two policies

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

## Running the code

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

## Current limits

The [latest pilot](../results/pilot/2026-09-05/README.md) covers chunk 1 only.
The grid search and ablations are incomplete. Delay averages count only correct
commitments, so compare recall and per-task results alongside delay.

A useful next step is a bounded, subset-aware replay runner that constructs one example's
observations at a time. That runner is not implemented yet. Use fixed parent-disjoint
development folds for tuning; a revised method needs fresh confirmation examples.

The model, dataset, and metric settings are in `configs/`; code is under
`src/streamner_commit/`; tests are under `tests/`. No corpus text, weights, or raw traces
are bundled. Historical pilot producer IDs are retained in the result metadata; they
are not commits in the clean public history.
