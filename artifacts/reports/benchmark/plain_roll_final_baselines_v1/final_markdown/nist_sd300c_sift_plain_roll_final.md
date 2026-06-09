# sift nist_sd300c Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.010667 | 22.29% | 0.14% | 77.71% | 156/544/1/699 | 0.7912 | 0.2936 |
| test | 1.00% | 0.008667 | 29.57% | 0.86% | 70.43% | 207/493/6/694 | 0.7912 | 0.2936 |
| val | 0.50% | 0.010667 | 22.29% | 0.43% | 77.71% | 156/544/3/697 | 0.7895 | 0.2850 |
| val | 1.00% | 0.008667 | 27.00% | 1.00% | 73.00% | 189/511/7/693 | 0.7895 | 0.2850 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.013333 | 0.00% | 17.86% | 125 | 575 | 0 | 700 |
| 0.10% | 0.013333 | 0.00% | 17.86% | 125 | 575 | 0 | 700 |
| 0.25% | 0.010660 | 0.14% | 22.57% | 158 | 542 | 1 | 699 |
| 0.50% | 0.009327 | 0.43% | 27.29% | 191 | 509 | 3 | 697 |
| 1.00% | 0.007995 | 1.00% | 32.43% | 227 | 473 | 7 | 693 |
| 2.00% | 0.006667 | 1.71% | 38.14% | 267 | 433 | 12 | 688 |
| 5.00% | 0.004664 | 4.57% | 49.86% | 349 | 351 | 32 | 668 |
| 10.00% | 0.003997 | 8.00% | 56.71% | 397 | 303 | 56 | 644 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.010667 | 700 | 156 | 544 | 22.29% | 77.71% |
| test | 1.00% | 0.008667 | 700 | 207 | 493 | 29.57% | 70.43% |
| val | 0.50% | 0.010667 | 700 | 156 | 544 | 22.29% | 77.71% |
| val | 1.00% | 0.008667 | 700 | 189 | 511 | 27.00% | 73.00% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.010667 | 0.43% | 3 | 700 | 1 | 699 | 0.14% | 99.86% |
| test | 1.00% | 0.008667 | 1.00% | 7 | 700 | 6 | 694 | 0.86% | 99.14% |
| val | 0.50% | 0.010667 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 0.008667 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.010667 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 0.008667 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 86.194 ms/pair, score CSV p50 93.535 ms, p95 167.254 ms.
- val: reported avg 90.544 ms/pair, score CSV p50 98.187 ms, p95 173.885 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
