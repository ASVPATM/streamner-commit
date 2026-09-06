# StreamNER-Commit

A work-in-progress tool for deciding when to lock entity predictions as text arrives.
The goal is fewer premature commitments without unnecessary waiting. Model inference
uses MLX on Apple Silicon; saved-score replay also runs on Linux.

## Latest test results

**2026-09-06: small development tests**, four datasets, one word per update.
The two-update buffer was tested on 80 examples, then frozen and checked on
40 different diagnostic parents. These are not new held-out benchmark results.

| Sample | Policy / output | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| 80 examples | Threshold 0.95 | 82.22% | 64.91% | 72.55% |
| 80 examples | Buffer, online only | 90.54% | 58.77% | 71.28% |
| 80 examples | Buffer, after close | 89.89% | 70.18% | 78.82% |
| 40 new diagnostic parents | Threshold 0.95 | 77.46% | 71.43% | 74.32% |
| 40 new diagnostic parents | Buffer, online only | 80.33% | 63.64% | 71.01% |
| 40 new diagnostic parents | Buffer, after close | 82.86% | 75.32% | 78.91% |

Close means an explicit end-of-message signal. Online recall fell; close recovered
pending entities. All net strict gains in the 40-example check came from AI4Privacy.
The samples are small and differ from the older 1,200-example pilot.

[Latest numbers and limits](results/development/2026-09-06/README.md) ·
[Older test pilot](results/pilot/2026-09-05/README.md) ·
[Technical details](docs/TECHNICAL_DESCRIPTION.md) · [Data](docs/THIRD_PARTY.md)
