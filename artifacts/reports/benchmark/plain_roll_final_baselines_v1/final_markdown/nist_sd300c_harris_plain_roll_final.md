# harris nist_sd300c Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.015260 | 0.14% | 0.14% | 99.86% | 1/699/1/699 | 0.4960 | 0.5071 |
| test | 1.00% | 0.009778 | 0.43% | 0.43% | 99.57% | 3/697/3/697 | 0.4960 | 0.5071 |
| val | 0.50% | 0.015260 | 0.00% | 0.43% | 100.00% | 0/700/3/697 | 0.5437 | 0.4550 |
| val | 1.00% | 0.009778 | 0.43% | 1.00% | 99.57% | 3/697/7/693 | 0.5437 | 0.4550 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.018332 | 0.00% | 0.00% | 0 | 700 | 0 | 700 |
| 0.10% | 0.018332 | 0.00% | 0.00% | 0 | 700 | 0 | 700 |
| 0.25% | 0.012038 | 0.14% | 0.43% | 3 | 697 | 1 | 699 |
| 0.50% | 0.008815 | 0.43% | 0.86% | 6 | 694 | 3 | 697 |
| 1.00% | 0.007442 | 1.00% | 1.43% | 10 | 690 | 7 | 693 |
| 2.00% | 0.006897 | 2.00% | 2.71% | 19 | 681 | 14 | 686 |
| 5.00% | 0.005902 | 4.71% | 5.14% | 36 | 664 | 33 | 667 |
| 10.00% | 0.004798 | 9.86% | 10.00% | 70 | 630 | 69 | 631 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.015260 | 700 | 1 | 699 | 0.14% | 99.86% |
| test | 1.00% | 0.009778 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 0.50% | 0.015260 | 700 | 0 | 700 | 0.00% | 100.00% |
| val | 1.00% | 0.009778 | 700 | 3 | 697 | 0.43% | 99.57% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.015260 | 0.43% | 3 | 700 | 1 | 699 | 0.14% | 99.86% |
| test | 1.00% | 0.009778 | 1.00% | 7 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 0.50% | 0.015260 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 0.009778 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.015260 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 0.009778 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 576.126 ms/pair, score CSV p50 576.232 ms, p95 1188.818 ms.
- val: reported avg 601.825 ms/pair, score CSV p50 604.137 ms, p95 1243.558 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
