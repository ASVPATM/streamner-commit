# Pilot results — 2026-09-05

400 development examples; 1,200 held-out examples; four datasets; chunk 1.
The interrupted development search completed 383/1,396 configurations, including
22/540 full StabilityGate settings and no ablations. Fourteen policy/mode choices
were then evaluated without retuning on test.

## Development-selected matched-quality comparison

| Policy | Precision | Recall | F1 | Mean delay (words) |
| --- | ---: | ---: | ---: | ---: |
| EMA | 0.779 | 0.645 | 0.705 | 6.54 |
| StabilityGate | 0.557 | 0.372 | 0.446 | 2.38 |

“Matched quality” names the development selection rule; it does not mean equal test
precision or recall. All policies and both selection modes are in `held_out_metrics.csv`.

EMA had lower average delay on three of four datasets. StabilityGate's overall delay
advantage came from Privy, where its recall was substantially lower. Delay counts only
correct commitments, so this is not evidence that the same entities were committed faster.

## Files and limits

- `held_out_metrics.csv`: per-task and pooled metrics for all selected policies.
- `coverage.csv`: completed versus planned search.
- `bootstrap.csv`: 12 paired comparisons, with only 50 resamples; exploratory intervals.
- `frozen_dev_pilot.json`: exact development-selected settings.
- `source_benchmark_manifest.json` and `provenance.json`: original identities and checksums.

This was a partial run, not a full validation. Other chunk sizes and ablations are missing.
The traces came from two reviewed historical producer commits, retained in the metadata;
the original same-commit gate was not passed. Corpus text, raw IDs, traces, and weights
are not included. The older files under `results/analysis/` are smoke-workflow outputs.
