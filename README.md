# StreamNER-Commit

A work-in-progress tool for deciding when to lock an entity prediction as text arrives.
The goal is to avoid committing too early, without waiting longer than necessary.

It compares simple rules, including EMA score smoothing, with a multi-check policy called
StabilityGate. Model inference uses MLX on Apple Silicon; saved-score replay can run on Linux.

## Latest test results

A partial run evaluated 14 policy/mode choices on 1,200 held-out examples. In the
development-selected matched-quality comparison, EMA scored **0.705 F1** and
StabilityGate **0.446 F1**. StabilityGate's lower overall average delay came with
much lower recall. Only **22 of 540** full StabilityGate settings were searched.

[Results](results/pilot/2026-09-05/README.md) ·
[Technical details](docs/TECHNICAL_DESCRIPTION.md) ·
[Data and dependencies](docs/THIRD_PARTY.md)
