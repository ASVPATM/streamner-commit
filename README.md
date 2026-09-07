# StreamNER-Commit

A work-in-progress tool for deciding when to lock entity predictions as text arrives.
The goal is fewer premature commitments without unnecessary waiting. Model inference
uses MLX on Apple Silicon; saved-score replay also runs on Linux.

## Latest test results

**V10 · 2026-09-06:** 96 reused development sentences, 24 per dataset; two fixed
48-parent blocks, one-word chunks, 192 completed policy replays. No model training.

### Main result — 0.90 buffer candidate, live

| Scope | Precision* | Recall* | Masking F1 | Masking F2 | FPR ↓ | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English average | 92.00% | 74.33% | 81.43% | 76.88% | 1.05% | 70.06% |

### Same-sample reference and after-close results

| Scope / output | Precision* | Recall* | Masking F1 | Masking F2 | FPR ↓ | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.95 reference · live | 92.02% | 63.27% | 73.87% | 66.97% | 0.79% | 68.24% |
| 0.95 reference · after close | 93.12% | 72.82% | 81.04% | 75.77% | 0.79% | 74.89% |
| 0.90 candidate · after close | 92.23% | 83.88% | 87.12% | 84.98% | 1.21% | 75.27% |

Both use the same two-update buffer. Live masking F2 improved in both blocks,
but block B lost precision and strict F1. **0.95 remains the reference; 0.90 is
not automatically promoted.** After close includes end-of-message flushing,
not live detection. [Per-dataset tables, costs and provenance](results/replication/2026-09-06/README.md).

### GLiNER comparison — earlier model-level reference

Full-text output from **v6**, not the v10 sample or streaming buffer. Columns follow
the [GLiNER model card](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0#evaluation).

| English average / evaluation | Precision* | Recall* | Masking F1 | Masking F2 | FPR ↓ | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GLiNER reported† | 87.36% | 91.55% | 89.18% | 90.53% | 3.01% | — |
| Local full text† | 86.14% | 91.19% | 88.38% | 89.99% | 3.19% | 70.68% |

\* Precision/recall measure **masked characters**; strict NER requires exact entities.
English averages weight datasets equally. FPR measures mistaken masking of non-PII;
lower is better. **F2 is lower than F1 because recall is lower than precision**:
weighting recall more heavily penalizes missed PII, rather than boosting the score.

† Both full-text cutoffs are 0.5; HF uses bfloat16, local uses MLX float32.
Same model/dataset revisions, different samples and inference conditions:
**not a matched benchmark reproduction**. Our 0.90/0.95 are commitment-policy
choices, not guarantees of accuracy. [Testing and threshold explanation](results/overnight/2026-09-06/README.md#why-f2-is-lower-and-the-thresholds-differ).

[Overnight v6: 1,200 sentences, four chunks](results/overnight/2026-09-06/README.md) ·
[Earlier small tests](results/development/2026-09-06/README.md) ·
[Technical details](docs/TECHNICAL_DESCRIPTION.md) · [Data](docs/THIRD_PARTY.md)
