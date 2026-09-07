# StreamNER-Commit

A work-in-progress tool for deciding when to lock entity predictions as text arrives.
The goal is fewer premature commitments without unnecessary waiting. Model inference
uses MLX on Apple Silicon; saved-score replay also runs on Linux.

Testing is paused after v10. The 0.95 buffer remains the reference and 0.90 the
leading candidate. [How I tested v1–v10 and why I paused](docs/TESTING_HISTORY.md).

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
[Earlier small tests](results/development/2026-09-06/README.md)

## Technical details

`Text chunks → GLiNER / MLX → saved scores → commitment policy → evaluation`

- **Detector:** the upstream GLiNER StreamingSpan PII checkpoint, with a Qwen3-0.6B
  backbone. Its weights stayed unchanged throughout these tests.
- **Buffer:** when an eligible prediction reaches the visible text edge, keep it
  provisional for two arriving updates. Then resolve candidates using their current
  scores. Extensions do not restart the wait; committed predictions cannot change.
  Two updates are not necessarily two model words.
- **Earlier policies:** EMA smooths fresh confidence scores. StabilityGate adds
  checks for repeated scores, stability and competing predictions. Neither is the
  current reference. V8/v9 also tested small learned policy heads, not detector
  fine-tuning.
- **Testing:** replay saved scores without rerunning GLiNER for each setting.
  Later runs load one example at a time, report
  progress and save checkpoints. Replay time is not live inference latency.

This evaluates prediction commitment, not a complete redaction service. Text already
sent downstream cannot be made private by a later detection. Raw text, traces and
private study bundles are not published; result folders contain reviewed aggregates.

[Setup and implementation notes](docs/TECHNICAL_DESCRIPTION.md) ·
[Testing history and limits](docs/TESTING_HISTORY.md)

## Credits

Built by [ASVPATM](https://github.com/ASVPATM), with OpenAI Codex assistance for code,
tests and documentation. Model work comes from the [GLiNER authors](https://github.com/urchade/GLiNER)
and [Knowledgator / Wordcab](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0).
Apple's [MLX](https://github.com/ml-explore/mlx) provides the inference runtime.
Evaluation uses [PIIMB](https://huggingface.co/datasets/piimb/pii-masking-benchmark),
including work from AI4Privacy, Gretel, NVIDIA and Privy contributors.
[Dependency, data and reference credits](docs/THIRD_PARTY.md).
