# StreamNER-Commit

A work-in-progress tool for deciding when to lock an entity prediction as text arrives.
The goal is to avoid committing too early, without waiting longer than necessary.

It compares simple rules, including EMA score smoothing, with a multi-check policy called
StabilityGate. Model inference uses MLX on Apple Silicon; saved-score replay can run on Linux.

## Latest test results

**2026-09-05 pilot:** 1,200 held-out examples, four datasets, one word per update.
Settings were selected on development data; the mode names do not imply equal test quality or delay.

### Matched-quality selection

| Policy | Precision ↑ | Recall ↑ | F1 ↑ | Delay (words) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Fixed threshold | 0.518 | 0.670 | 0.584 | 6.27 |
| Fixed lag | 0.774 | 0.645 | 0.704 | 6.53 |
| Snapshot patience | 0.774 | 0.645 | 0.704 | 6.53 |
| Rescore patience | 0.498 | 0.377 | 0.429 | 2.32 |
| EMA | 0.779 | 0.645 | 0.705 | 6.54 |
| StabilityGate | 0.557 | 0.372 | 0.446 | 2.38 |
| Oracle (future-aware) | 0.604 | 0.729 | 0.661 | 5.92 |

### Matched-latency selection

| Policy | Precision ↑ | Recall ↑ | F1 ↑ | Delay (words) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Fixed threshold | 0.518 | 0.670 | 0.584 | 6.27 |
| Fixed lag | 0.774 | 0.645 | 0.704 | 6.53 |
| Snapshot patience | 0.774 | 0.645 | 0.704 | 6.53 |
| Rescore patience | 0.774 | 0.645 | 0.704 | 6.53 |
| EMA | 0.780 | 0.640 | 0.703 | 6.56 |
| StabilityGate | 0.726 | 0.353 | 0.475 | 2.57 |
| Oracle (future-aware) | 0.604 | 0.729 | 0.661 | 5.92 |

Delay averages **correct commitments only**; lower delay can accompany missed entities.
The oracle uses future information and is not deployable. Only **22/540** full StabilityGate
settings were searched, with no ablations—these are partial results.

[Results](results/pilot/2026-09-05/README.md) ·
[Technical details](docs/TECHNICAL_DESCRIPTION.md) ·
[Data and dependencies](docs/THIRD_PARTY.md)
