# sift_plain_roll_v2 nist_sd300b Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 8.124565 | 45.71% | 0.43% | 54.29% | 320/380/3/697 | 0.7882 | 0.2957 |
| test | 1.00% | 6.654213 | 50.00% | 1.57% | 50.00% | 350/350/11/689 | 0.7882 | 0.2957 |
| val | 0.50% | 8.124565 | 41.00% | 0.43% | 59.00% | 287/413/3/697 | 0.8006 | 0.2729 |
| val | 1.00% | 6.654213 | 45.86% | 1.00% | 54.14% | 321/379/7/693 | 0.8006 | 0.2729 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 11.922592 | 0.00% | 38.29% | 268 | 432 | 0 | 700 |
| 0.10% | 11.922592 | 0.00% | 38.29% | 268 | 432 | 0 | 700 |
| 0.25% | 9.055633 | 0.14% | 43.14% | 302 | 398 | 1 | 699 |
| 0.50% | 8.015417 | 0.43% | 46.43% | 325 | 375 | 3 | 697 |
| 1.00% | 6.976152 | 1.00% | 49.71% | 348 | 352 | 7 | 693 |
| 2.00% | 6.047334 | 2.00% | 51.43% | 360 | 340 | 14 | 686 |
| 5.00% | 4.835804 | 5.00% | 55.43% | 388 | 312 | 35 | 665 |
| 10.00% | 4.093485 | 9.29% | 59.29% | 415 | 285 | 65 | 635 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 8.124565 | 700 | 320 | 380 | 45.71% | 54.29% |
| test | 1.00% | 6.654213 | 700 | 350 | 350 | 50.00% | 50.00% |
| val | 0.50% | 8.124565 | 700 | 287 | 413 | 41.00% | 59.00% |
| val | 1.00% | 6.654213 | 700 | 321 | 379 | 45.86% | 54.14% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 8.124565 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| test | 1.00% | 6.654213 | 1.00% | 7 | 700 | 11 | 689 | 1.57% | 98.43% |
| val | 0.50% | 8.124565 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 6.654213 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 8.124565 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 6.654213 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 121.382 ms/pair, score CSV p50 119.932 ms, p95 223.259 ms.
- val: reported avg 117.260 ms/pair, score CSV p50 117.287 ms, p95 216.963 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
