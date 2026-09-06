# Small development results — 2026-09-06

The unchanged threshold-0.95 policy is compared with a short provisional buffer:
when an eligible candidate reaches the visible text edge, wait at most two input
updates at that start before resolving current candidates. Hard commitments are
not revised. An explicit message close can release remaining pending candidates.

V4 used 80 previously analyzed development examples (20 per dataset). V5 froze
the two-update setting and used 40 different diagnostic parents (10 per dataset),
excluding all 160 earlier diagnostic parents. Those parents may have appeared in
the original development sweep. V4 also tested a one-update buffer; its complete
aggregate results are retained in [metrics.json](metrics.json).

## Confirmation counts

| Output | Correct entities (TP) | Wrong entities (FP) | Missed entities (FN) | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Threshold 0.95 | 55 | 16 | 22 | 77.46% | 71.43% | 74.32% |
| Two-update buffer, online | 49 | 12 | 28 | 80.33% | 63.64% | 71.01% |
| Two-update buffer, after close | 58 | 12 | 19 | 82.86% | 75.32% | 78.91% |

After close: +5.39 percentage points precision, +3.90 recall, +4.59 F1 versus
the same-sample threshold baseline. All 55 baseline-correct entities were retained.
Close added nine correct entities and no wrong entities to the buffer's online
output. Character masking recall increased from 74.59% to 77.49% after close.

| Dataset | Gold entities | Baseline F1 | Buffer F1 after close |
| --- | ---: | ---: | ---: |
| AI4Privacy EN | 42 | 74.07% | 82.50% |
| Gretel | 13 | 56.00% | 56.00% |
| Nemotron PII | 4 | 80.00% | 80.00% |
| Privy | 18 | 87.50% | 87.50% |

All net strict gains came from AI4Privacy; support in the other datasets is sparse.
The exploratory paired 200-resample interval for the F1 difference was
+1.16 to +8.48 percentage points. This small-sample interval is not deployment validation.

## Waiting and limits

Among 48 entities correctly committed online by both policies, 33 were unchanged
and 15 waited two additional updates. The mean added model-word delay was 0.98
on that jointly correct subset only. Seven baseline-correct entities instead
waited until close; their close-signal wall-clock delay is **unmeasured**. A
two-update wait is not necessarily two model words.

V4 finished in 256 seconds and V5 in 97 seconds, at roughly half-duty on one Linux
worker (646/310 MiB peak RSS). These are replay times, not inference latency.
The historical trace producer and newer diagnostic consumer differ: the original
same-commit reproducibility gate is still not passed. Aggregate metrics and
provenance hashes are published; raw text, IDs, case traces and private bundles are not.

The [older 1,200-example pilot](../../pilot/2026-09-05/README.md) followed an
interrupted development sweep. Its EMA F1 of 70.53% used a different sample and
online output: subtracting it from these numbers would not measure improvement.
