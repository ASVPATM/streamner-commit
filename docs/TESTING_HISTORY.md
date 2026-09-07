# Testing history: v1–v10

These are experiment numbers, not ten model releases. GLiNER's detector weights
never changed. Most changes affected which predictions were accepted and when.

## How the tests worked

1. Generate and save streaming model scores on Apple Silicon with MLX.
2. Pick fixed examples from four English PIIMB tasks: AI4Privacy, Gretel,
   Nemotron-PII and Privy. Keep sentences from the same source document, or
   *parent*, together when separating fitting and evaluation groups.
3. Replay identical scores through each policy. Later runs use
   one-example loading, resource limits, progress reports and resumable checkpoints.
   V8/v9 fit small policy heads; neither retrains GLiNER.
4. Check completion, input/code identities and saved counts before interpreting
   results. Compare errors, masking and delay on the same examples. Check that
   unchanged reference policies still reproduce their earlier output.

The original broad sweep was interrupted because of its cost. Its partial results
remain historical evidence, not a completed search for the best settings.

## What each version tested

| Run / sample | Test | Finding |
| --- | --- | --- |
| **V1 · 80 dev sentences** | Twelve fixed controls and checks explaining missed entities. | Requiring a second fresh score rejected many candidates that were never rescored. EMA matched the simple threshold on this sample. |
| **V2 · 80 other dev parents** | Eight controls on two fixed folds, including a conditional gate. | The added gate did not establish an advantage over a simpler threshold. Premature boundaries needed a closer look. |
| **V3 · same 80 as v2** | Audit overlap conflicts without changing the threshold policy. | All eight blocked gold entities were incomplete when an earlier prediction was committed. This was an explanation, not an accuracy improvement. |
| **V4 · same 80 as v3** | Compare threshold-only with one- and two-update provisional buffers. | A short buffer could reconsider boundaries before commitment. Message-close recovery needed its own score. |
| **V5 · 40 other dev parents** | Freeze the two-update buffer and check it against the threshold. | After-close strict F1 rose from 74.32% to 78.91%; live F1 was 71.01%. Net gains came from AI4Privacy. |
| **V6 · 1,200 reused test sentences** | Overnight replay: three policies, chunk sizes 1/2/4/8, 14,400 comparisons. | After-close boundary gains held on this larger sample, but live recall remained weak. The README's full-text GLiNER comparison comes from this run. |
| **V7 · 80 dev sentences** | Test earlier release and coordination between overlapping pending predictions, separately and together. | Earlier release lost accuracy. Coordination did not activate on this sample, so it had no demonstrated benefit. |
| **V8 · 333 dev sentences, 311 parents** | Four chunk sizes; compare extra waiting and small learned readiness heads on parent-separated folds. | All learned branches fell back to the buffer under their calibration rules. Uniform extra waiting hurt live recall. |
| **V9 · 96 dev parents** | Compare admission cutoffs and learned score/context filters. | The 0.90 buffer was the strongest simple candidate. 0.85 added little recall; 0.50 caused more false masking. Context helped against raw 0.50, but did not establish a replacement for 0.90. |
| **V10 · 96 other dev parents** | Freeze 0.90 versus 0.95 on two predefined blocks; no between-block tuning. | Live masking F2 improved in both blocks. Block B lost precision and strict F1, so 0.90 remains a candidate rather than an automatic default. |

“Other parents” is relative to the stated predecessor, not every earlier test.
V8 reused development examples; v9/v10 drew from that development pool. V10 excluded
v9 parents and the reserved confirmation group. V6 reused the original pilot's
test sample. None of this is a fresh, full-benchmark validation of the final candidate.

## Reading the results

- **Strict NER:** the predicted boundaries and entity type must match exactly.
- **Masking:** count covered PII characters, regardless of the predicted type.
  A boundary error can still mask some PII. F2 puts more weight on missed PII than F1.
- **FPR:** the fraction of non-PII characters masked by mistake. Lower is better.
- **Live / after close:** output while text arrives versus output after releasing
  pending predictions at the end. After-close recovery is not timely live masking.
- **Delay:** model words or arriving updates, not seconds. Correct-only averages
  leave out missed entities, so read them alongside recall and deadline coverage.

The README's English averages weight the four datasets equally. Some older tables
pool counts instead. Different samples, output phases and averaging methods cannot
be treated as a v1–v10 score progression. Longer replay does not train the detector.

## Why I paused

The original wide sweeps cost too much time and CPU for my setup. Bounded replay
made later tests manageable, but another large sweep would mostly measure more
settings without addressing the remaining detection errors.

The 0.90 buffer improved coverage, with some extra false masking. It still missed
243 PII characters after close in v10. Of those, 217 had an exact candidate whose
best saved score stayed below 0.90. More waiting cannot raise those saved scores,
and a lower cutoff can also admit wrong predictions.

I am keeping 0.95 as the reference, 0.90 as the candidate, and the earlier findings.
Before another run, I want a specific detection change to test and a clear limit
on false masking and delay. No final candidate promotion or production release
has been made.

## Published records

- [Original pilot](../results/pilot/2026-09-05/README.md)
- [V4/v5 buffer results](../results/development/2026-09-06/README.md)
- [V6 overnight results and model-card comparison](../results/overnight/2026-09-06/README.md)
- [V10 replication, paired costs and provenance](../results/replication/2026-09-06/README.md)

These links contain reviewed aggregate results. V1–v3 and v7–v9 above summarize
retained private reviews; their detailed returns are not public. The frozen study
code, raw traces and membership needed to reproduce every run are not all shipped
in this repository. A public clone alone does not reproduce v1–v10.

[Credits and acknowledgements](../README.md#credits) · [Data and dependency terms](THIRD_PARTY.md)
