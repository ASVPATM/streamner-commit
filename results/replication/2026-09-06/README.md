# V10 candidate replication results

Published 2026-09-06. Frozen two-update buffer at **0.90 vs the unchanged 0.95
reference**. All 192 policy/example pairs completed on 96 development sentences
from 96 parents: two predefined 48-parent blocks, 12 parents per dataset per block.
There are 118 annotated entities. Primary chunk: one arriving word.

The candidate was selected in v9; there was no fitting or tuning between these
blocks. All v9 parents and their siblings, and all 64 reserved parents and their
siblings, were excluded. These examples were previously used in v8 development:
this is **additional development replication, not an untouched final test**.

## English average — same examples, separate output phases

| Scope / output | Precision* | Recall* | Masking F1 | Masking F2 | FPR ↓ | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Live · 0.95 reference | 92.02% | 63.27% | 73.87% | 66.97% | 0.79% | 68.24% |
| Live · 0.90 candidate | 92.00% | 74.33% | 81.43% | 76.88% | 1.05% | 70.06% |
| After close · 0.95 reference | 93.12% | 72.82% | 81.04% | 75.77% | 0.79% | 74.89% |
| After close · 0.90 candidate | 92.23% | 83.88% | 87.12% | 84.98% | 1.21% | 75.27% |

\* Precision/recall are label-agnostic masked-character metrics. Each task pools
its counts; the English average gives each of the four task metrics equal weight,
including F-scores. Strict NER requires exact boundaries and type. FPR is the
fraction of non-PII characters masked incorrectly; lower is better. Live excludes
explicit end-of-message flushing. After-close results include it.

The combined live recall change is +11.06 pp and masking F2 change
is +9.90 pp. These are same-sample comparisons, **not improvements
measured against the larger overnight sample or GLiNER model-card population**.
Model weights, scores and the two-update buffer mechanism were unchanged.

## Replication consistency and costs

| Block | Live masking F2, 0.95 → 0.90 | Mask precision, 0.95 → 0.90 | Strict F1 change |
| --- | ---: | ---: | ---: |
| A | 68.67% → 80.80% | 93.33% → 96.43% | +4.80 pp |
| B | 63.08% → 71.39% | 89.71% → 85.55% | -1.82 pp |

Masking F2 improved in seven of eight dataset/block cells. Nemotron in B declined
0.62 points and has just five annotated entities in that block. Combined masking
precision can conceal local regressions: block B lost precision and strict F1.

The 512-resample paired-parent descriptive interval for the combined macro-F2
difference is +3.71 to +14.94 points. Removing the largest-gain parent for sensitivity
still leaves +8.26 points. The primary result retains every parent; these checks
are not a promotion rule or evidence from a fresh benchmark.

## Per-dataset comparisons

Each task has 24 sentences, with the same sentences used for both controls.

| Scope / output | Precision* | Recall* | Masking F1 | Masking F2 | FPR ↓ | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ai4privacy-en · live · 0.95 | 100.00% | 58.50% | 73.82% | 63.80% | 0.00% | 55.70% |
| ai4privacy-en · live · 0.90 | 100.00% | 66.36% | 79.78% | 71.14% | 0.00% | 57.14% |
| ai4privacy-en · after close · 0.95 | 100.00% | 64.67% | 78.55% | 69.59% | 0.00% | 60.98% |
| ai4privacy-en · after close · 0.90 | 100.00% | 72.52% | 84.07% | 76.74% | 0.00% | 62.07% |
| gretel · live · 0.95 | 100.00% | 48.59% | 65.40% | 54.16% | 0.00% | 64.86% |
| gretel · live · 0.90 | 93.24% | 60.50% | 73.38% | 65.07% | 1.33% | 61.90% |
| gretel · after close · 0.95 | 100.00% | 66.46% | 79.85% | 71.24% | 0.00% | 73.17% |
| gretel · after close · 0.90 | 94.70% | 78.37% | 85.76% | 81.17% | 1.33% | 69.57% |
| nemotron-pii · live · 0.95 | 68.09% | 60.38% | 64.00% | 61.78% | 3.18% | 63.16% |
| nemotron-pii · live · 0.90 | 74.77% | 75.47% | 75.12% | 75.33% | 2.86% | 70.00% |
| nemotron-pii · after close · 0.95 | 72.48% | 74.53% | 73.49% | 74.11% | 3.18% | 76.19% |
| nemotron-pii · after close · 0.90 | 74.22% | 89.62% | 81.20% | 86.05% | 3.50% | 78.26% |
| privy · live · 0.95 | 100.00% | 85.63% | 92.26% | 88.16% | 0.00% | 89.23% |
| privy · live · 0.90 | 100.00% | 95.00% | 97.44% | 95.96% | 0.00% | 91.18% |
| privy · after close · 0.95 | 100.00% | 85.63% | 92.26% | 88.16% | 0.00% | 89.23% |
| privy · after close · 0.90 | 100.00% | 95.00% | 97.44% | 95.96% | 0.00% | 91.18% |

## What changed in the actual output

- Live PII coverage: 806 → 932 of 1,280 annotated characters. That is 126 gained,
  none previously covered lost, spread across 12 parents; the largest supplies
  18.25% of the gain.
- Live non-PII masking: 30 → 41 characters, from 17 newly masked and six avoided.
  Three added non-gold first-name predictions account for the 17 newly masked
  characters. After close, non-PII masking is 30 → 47.
- Live exact entity TP/FP/FN: 69/13/49 → 75/21/43. All 69 previously correct entities
  remain; six correct and nine wrong identities are added, one wrong is removed.
  Six added exact-entity errors nevertheless mask 66 PII characters and no non-PII.
- For the same 69 correct entities, mean added-word delay changes by +0.029 words:
  64 unchanged, three slower, two earlier. Correct-only means are 0.54 → 0.67 words
  on different cohorts. These are not wall-clock delays.
- All-gold strict recall within two arriving updates: 69/118 → 74/118.
  Updates need not equal model words; missed entities stay in this denominator.

## Remaining gaps and interpretation

The candidate still misses 348 PII characters live and 243 after close. Of the
243, 217 have an exact gold proposal whose best saved score is below 0.90; 12
belong to an above-width-cap address and 14 to word-unaligned gold boundaries.
This retrospective diagnosis does **not** mean lowering a cutoff will safely
recover 217 characters. Missing proposals, confidence and false masking need
separate treatment; more waiting is not a demonstrated solution.

F2 weights recall more heavily than F1. When recall is lower than precision,
that weighting penalizes missed PII more—it does not award a higher score.
A 0.90 or 0.95 admission score is not a guarantee of 90% or 95% empirical accuracy.

**Decision:** retain 0.90 as the leading simple masking candidate and 0.95 as the
reference. No baseline promotion, reserved confirmation or model training occurred.
Acceptable false-masking/delay costs must be specified before promotion.

## Verification and provenance

The private return checksums, exact frozen membership and predecessor identities
passed verification. All 192 policy records, 197 commitment/source-score joins,
118 structural coverage records, 72 aggregate metric rows and 30 comparisons were
checked; all 96 reference records retained baseline parity. This was a read-only
source audit, not a second independent policy rerun.

Worker time was 355.29 seconds (5m55s), peak RSS 1,396.24 MiB, one half-duty worker;
setup, preflight and report assembly are additional. No hardware safety claim
or runtime guarantee for a different sample follows from this measurement.

[metrics.json](metrics.json) contains aggregate counts, all blocks, paired costs,
delay and gap summaries. [provenance.json](provenance.json) records model/dataset,
protocol and consumer hashes. This is an **aggregate-only publication**, not a
self-contained public reproduction package: the frozen consumer snapshot, raw
traces, source text, membership and detailed private return remain private.

[Overnight v6 and model-card comparison](../../overnight/2026-09-06/README.md) ·
[Earlier small buffer results](../../development/2026-09-06/README.md)
