# StreamNER-Commit

A work-in-progress tool for deciding when to lock entity predictions as text arrives.
The goal is fewer premature commitments without unnecessary waiting. Model inference
uses MLX on Apple Silicon; saved-score replay also runs on Linux.

## Latest test results

**Overnight v6 · published 2026-09-06:** 1,200 reused test sentences, 300 per dataset;
all 14,400 policy replays completed across chunks 1/2/4/8. Below: primary chunk 1,
using the [GLiNER model card's evaluation columns](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0#evaluation).

| Scope / evaluation | Precision\* | Recall\* | Masking F1 | Masking F2 | FPR | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English avg · HF card† | 87.36% | 91.55% | 89.18% | 90.53% | 3.01% | — |
| English avg · local full text† | 86.14% | 91.19% | 88.38% | 89.99% | 3.19% | 70.68% |
| English avg · EMA live‡ | 95.00% | 72.98% | 82.29% | 76.40% | 1.08% | 72.92% |
| English avg · buffer live‡ | 93.82% | 61.14% | 73.82% | 65.62% | 0.97% | 68.08% |
| English avg · buffer after close‡ | 94.57% | 74.99% | 83.43% | 78.11% | 1.16% | 76.45% |

\* Precision/recall measure **masked characters**, not exact entities. English
averages weight the four datasets equally; they are not pooled scores.

† HF: threshold 0.5, bfloat16. Local full text: threshold 0.5, MLX float32.
Same model/dataset revisions, different samples and inference conditions:
the closest reference, **not a matched benchmark reproduction**.

‡ Same-sample local policy comparison at threshold 0.95. After close includes an
explicit end-of-message flush, not live detection. The buffer trades live recall
for waiting; its better after-close strict F1 is not an upstream benchmark win.

[Per-dataset comparisons, all chunks and counts](results/overnight/2026-09-06/README.md) ·
[Earlier small tests](results/development/2026-09-06/README.md) ·
[Technical details](docs/TECHNICAL_DESCRIPTION.md) · [Data](docs/THIRD_PARTY.md)
