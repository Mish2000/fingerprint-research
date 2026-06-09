# harris nist_sd300b Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.018072 | 0.14% | 0.00% | 99.86% | 1/699/0/700 | 0.5034 | 0.4936 |
| test | 1.00% | 0.010370 | 0.57% | 0.29% | 99.43% | 4/696/2/698 | 0.5034 | 0.4936 |
| val | 0.50% | 0.018072 | 0.00% | 0.43% | 100.00% | 0/700/3/697 | 0.5241 | 0.4786 |
| val | 1.00% | 0.010370 | 0.14% | 1.00% | 99.86% | 1/699/7/693 | 0.5241 | 0.4786 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.015044 | 0.00% | 0.29% | 2 | 698 | 0 | 700 |
| 0.10% | 0.015044 | 0.00% | 0.29% | 2 | 698 | 0 | 700 |
| 0.25% | 0.015044 | 0.00% | 0.29% | 2 | 698 | 0 | 700 |
| 0.50% | 0.008636 | 0.43% | 0.71% | 5 | 695 | 3 | 697 |
| 1.00% | 0.008249 | 0.86% | 0.86% | 6 | 694 | 6 | 694 |
| 2.00% | 0.007847 | 1.57% | 1.14% | 8 | 692 | 11 | 689 |
| 5.00% | 0.005674 | 4.86% | 4.57% | 32 | 668 | 34 | 666 |
| 10.00% | 0.004794 | 9.86% | 8.43% | 59 | 641 | 69 | 631 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.018072 | 700 | 1 | 699 | 0.14% | 99.86% |
| test | 1.00% | 0.010370 | 700 | 4 | 696 | 0.57% | 99.43% |
| val | 0.50% | 0.018072 | 700 | 0 | 700 | 0.00% | 100.00% |
| val | 1.00% | 0.010370 | 700 | 1 | 699 | 0.14% | 99.86% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.018072 | 0.43% | 3 | 700 | 0 | 700 | 0.00% | 100.00% |
| test | 1.00% | 0.010370 | 1.00% | 7 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 0.50% | 0.018072 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 0.010370 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.018072 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 0.010370 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 561.351 ms/pair, score CSV p50 561.203 ms, p95 1169.786 ms.
- val: reported avg 572.347 ms/pair, score CSV p50 572.669 ms, p95 1188.509 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
