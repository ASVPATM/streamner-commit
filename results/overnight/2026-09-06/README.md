# Overnight v6 results

Published 2026-09-06. All **14,400 policy/example/chunk pairs** completed:
1,200 test sentences from 1,160 parents, 300 sentences per dataset, at chunks
1/2/4/8. These are the same test examples used in the older pilot, not a fresh
held-out evaluation. The overnight run replays saved model scores; it does not
fine-tune GLiNER or rerun model inference.

## Reading the comparison

The tables follow the [GLiNER model card](https://huggingface.co/knowledgator/gliner-stream-pii-v1.0#evaluation).
Precision, recall, F1, F2 and FPR below are **label-agnostic character masking**
metrics. Strict NER F1 additionally requires the exact entity boundaries and type.
Task scores use summed counts; English averages are the unweighted means of the
four task scores, including the individual F-scores—not F-scores recalculated
from average precision/recall. These match the aggregation conventions described
by [PIIMB](https://huggingface.co/datasets/piimb/pii-masking-benchmark#metrics).

† **Metric-compatible reference, not an exact reproduction.** The retrieved model
card and local checkpoint are pinned to model
revision `e871777dc4b3b688747a0433fff8d94a36fcc7b0` and dataset revision
`4a13e9ffe6fd0d275efbde8afd4d8d8f1ffc2133`, sentences subset. The model card reports PIIMB 0.3.0,
threshold 0.5 and bfloat16, evaluated 2026-07-24. Our saved full-sentence reference
uses threshold 0.5 and MLX float32 on a 300-per-task sample. The card's evaluation
section does not identify the inference strategy/chunk schedule; identical
end-to-end conditions are not established. No claim of outperforming the upstream
model follows from these tables. Multilingual tasks were not run locally.

‡ **Controlled local comparisons.** Buffer, EMA and threshold replay the same
examples and frozen scores. All use admission threshold 0.95; EMA has alpha 0.8.
The buffer waits two arriving updates at the first eligible text-edge start.
Hard commitments cannot be revised. Its explicit-close output includes pending
releases at end of message and is not interchangeable with live output.
Final incremental snapshots at threshold 0.5 are also retained in
[metrics.json](metrics.json); those can revise earlier predictions.

## Closest model-level reference

Saved cold full-sentence output, not the commitment policy. Values in percent;
— means the model card did not report that aggregate. HF numbers are transcribed
as published, not recalculated from their rounded task rows.
[Versioned source values](hf_reference.json).

| Scope / evaluation | Precision | Recall | Masking F1 | Masking F2 | FPR | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English average · HF† | 87.36% | 91.55% | 89.18% | 90.53% | 3.01% | — |
| English average · local full text† | 86.14% | 91.19% | 88.38% | 89.99% | 3.19% | 70.68% |
| ai4privacy-en · HF† | 94.69% | 95.16% | 94.92% | 95.06% | 1.52% | 67.99% |
| ai4privacy-en · local full text† | 94.91% | 95.06% | 94.98% | 95.03% | 1.50% | 66.51% |
| gretel · HF† | 86.95% | 95.58% | 91.06% | 93.72% | 5.20% | 67.38% |
| gretel · local full text† | 85.18% | 96.17% | 90.34% | 93.75% | 5.56% | 67.93% |
| nemotron-pii · HF† | 72.75% | 88.07% | 79.68% | 84.51% | 4.98% | 68.93% |
| nemotron-pii · local full text† | 71.25% | 86.11% | 77.98% | 82.66% | 5.24% | 67.15% |
| privy · HF† | 95.07% | 87.39% | 91.07% | 88.83% | 0.33% | 81.68% |
| privy · local full text† | 93.21% | 87.43% | 90.23% | 88.53% | 0.47% | 81.13% |

## Current buffer: same columns, different operating point

Primary chunk 1. Both rows per task use exactly the same 300 sentences.
These threshold-0.95 outcomes should not be ranked directly against the upstream
threshold-0.5 table; inspect the precision/recall tradeoff and close dependence.

| Scope / output | Precision | Recall | Masking F1 | Masking F2 | FPR | Strict NER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| English average · live‡ | 93.82% | 61.14% | 73.82% | 65.62% | 0.97% | 68.08% |
| English average · after close‡ | 94.57% | 74.99% | 83.43% | 78.11% | 1.16% | 76.45% |
| ai4privacy-en · live‡ | 98.75% | 57.23% | 72.47% | 62.49% | 0.21% | 60.40% |
| ai4privacy-en · after close‡ | 98.96% | 68.70% | 81.10% | 73.18% | 0.21% | 66.35% |
| gretel · live‡ | 88.50% | 61.53% | 72.59% | 65.53% | 2.66% | 61.90% |
| gretel · after close‡ | 89.22% | 81.30% | 85.07% | 82.77% | 3.27% | 72.93% |
| nemotron-pii · live‡ | 88.93% | 52.05% | 65.67% | 56.76% | 0.98% | 69.09% |
| nemotron-pii · after close‡ | 90.99% | 75.60% | 82.59% | 78.25% | 1.13% | 85.27% |
| privy · live‡ | 99.09% | 73.75% | 84.56% | 77.73% | 0.05% | 80.93% |
| privy · after close‡ | 99.10% | 74.34% | 84.95% | 78.25% | 0.05% | 81.24% |

## Matched local policy comparisons across all chunks

**Pooled micro scores**, to match the original 1,200-example pilot's reporting.
This is why the buffer's chunk-1 after-close strict F1 here is **74.47%**, while
the equally weighted four-task average above is **76.45%**. No underlying counts
changed. All three policies use threshold 0.95. The threshold and EMA have no
close flush, so their after-close scores equal their live scores.

| Chunk / policy | TP | FP | FN | Strict precision | Strict recall | Strict F1 | Masking F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 · Threshold | 1035 | 302 | 569 | 77.41% | 64.53% | 70.38% | 75.73% |
| 1 · EMA | 1034 | 294 | 570 | 77.86% | 64.46% | 70.53% | 75.63% |
| 1 · Buffer live | 932 | 190 | 672 | 83.07% | 58.10% | 68.38% | 67.10% |
| 1 · Buffer after close | 1088 | 230 | 516 | 82.55% | 67.83% | 74.47% | 77.58% |
| 2 · Threshold | 1050 | 270 | 554 | 79.55% | 65.46% | 71.82% | 76.01% |
| 2 · EMA | 1048 | 264 | 556 | 79.88% | 65.34% | 71.88% | 75.88% |
| 2 · Buffer live | 923 | 185 | 681 | 83.30% | 57.54% | 68.07% | 66.19% |
| 2 · Buffer after close | 1084 | 221 | 520 | 83.07% | 67.58% | 74.53% | 77.26% |
| 4 · Threshold | 1068 | 238 | 536 | 81.78% | 66.58% | 73.40% | 76.71% |
| 4 · EMA | 1067 | 232 | 537 | 82.14% | 66.52% | 73.51% | 76.58% |
| 4 · Buffer live | 929 | 180 | 675 | 83.77% | 57.92% | 68.49% | 66.59% |
| 4 · Buffer after close | 1085 | 211 | 519 | 83.72% | 67.64% | 74.83% | 77.31% |
| 8 · Threshold | 1074 | 213 | 530 | 83.45% | 66.96% | 74.30% | 76.45% |
| 8 · EMA | 1074 | 209 | 530 | 83.71% | 66.96% | 74.40% | 76.43% |
| 8 · Buffer live | 928 | 179 | 676 | 83.83% | 57.86% | 68.46% | 66.48% |
| 8 · Buffer after close | 1080 | 204 | 524 | 84.11% | 67.33% | 74.79% | 76.71% |

At chunk 1, buffer after-close strict F1 is +3.94 percentage points versus the
original EMA on the **same examples**. It gains 59 correct entities and loses 5;
20 wrong entities are added and 84 removed. The descriptive paired
task-stratified parent-bootstrap interval for the F1 difference is +2.91 to +5.07
points (2,000 resamples). Reused test data and prior design choices limit that
interval; it is not fresh confirmation.

Live buffer F1 is lower than EMA (68.38% versus 70.53%). Closing the buffer adds
156 correct and 40 wrong entities. Of 516 remaining after-close misses, 474 never
reach 0.95 in the saved scores. Extra waiting alone cannot recover those misses.

## Waiting

Chunk 1; delay summaries include **live correct entities only**, not all gold.

| Policy | Live correct | Mean context words | Mean extra words | P95 extra words | Correct added at close | Wrong added at close |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| threshold_095 | 1035 | 6.53 | 0.06 | 0.00 | 0 | 0 |
| ema_published | 1034 | 6.54 | 0.06 | 0.00 | 0 | 0 |
| buffer_2 | 932 | 8.54 | 1.01 | 3.00 | 156 | 40 |

Context words count from the entity's end in the model-word coordinate system.
Extra words count from the first prefix where the whole gold entity was visible.
Neither is wall-clock latency. Input chunk units and model words can differ.
The two-update buffering deadline does not cap total gold-to-detection word delay.
The extra delay among 884 entities correctly committed live by both buffer and
EMA averages 0.65 model words; 145 EMA-live-correct entities instead wait until
close under the buffer. Close-signal wall-clock delay is unmeasured.

## Provenance and limits

The returned archive's checksums and protocol identity were verified. All 4,800
example/chunk records were checked against frozen membership; their TP/FP identity
sets and counts reconstruct every one of the 120 published policy aggregate rows.
Published EMA parity passed. Five whitespace-ended gold annotations remain in
strict and masking denominators; unavailable word endpoints are explicit.

[metrics.json](metrics.json) contains all per-task/pooled counts, unweighted task
averages, both cached model references, aggregate waiting/paired comparisons,
policy specifications and provenance hashes. Rates are fractions there
(0.90 = 90%). Masking F2 and FPR are calculated from the saved character counts;
no additional replay was needed.

The historical trace producer and later diagnostic consumer differ, so the
original same-commit reproducibility gate remains unpassed. No raw text, example
or parent IDs, case traces, private reports or machine paths are published.
These results do not establish deployment safety or a replacement for the upstream
benchmark. [Data attribution and terms](../../../docs/THIRD_PARTY.md).

[Earlier small development tests](../../development/2026-09-06/README.md) ·
[Older test pilot](../../pilot/2026-09-05/README.md)
