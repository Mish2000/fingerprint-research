# minutiae nist_sd300c Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.254902 | 0.86% | 0.71% | 99.14% | 6/694/5/695 | 0.5285 | 0.4764 |
| test | 1.00% | 0.235294 | 1.43% | 1.00% | 98.57% | 10/690/7/693 | 0.5285 | 0.4764 |
| val | 0.50% | 0.254902 | 0.29% | 0.43% | 99.71% | 2/698/3/697 | 0.5132 | 0.4900 |
| val | 1.00% | 0.235294 | 1.14% | 1.00% | 98.86% | 8/692/7/693 | 0.5132 | 0.4900 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.294118 | 0.00% | 0.43% | 3 | 697 | 0 | 700 |
| 0.10% | 0.294118 | 0.00% | 0.43% | 3 | 697 | 0 | 700 |
| 0.25% | 0.279152 | 0.14% | 0.57% | 4 | 696 | 1 | 699 |
| 0.50% | 0.279152 | 0.14% | 0.57% | 4 | 696 | 1 | 699 |
| 1.00% | 0.232912 | 1.00% | 1.57% | 11 | 689 | 7 | 693 |
| 2.00% | 0.216506 | 2.00% | 1.86% | 13 | 687 | 14 | 686 |
| 5.00% | 0.191176 | 5.00% | 6.57% | 46 | 654 | 35 | 665 |
| 10.00% | 0.176649 | 7.86% | 11.29% | 79 | 621 | 55 | 645 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.254902 | 700 | 6 | 694 | 0.86% | 99.14% |
| test | 1.00% | 0.235294 | 700 | 10 | 690 | 1.43% | 98.57% |
| val | 0.50% | 0.254902 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 1.00% | 0.235294 | 700 | 8 | 692 | 1.14% | 98.86% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.254902 | 0.43% | 3 | 700 | 5 | 695 | 0.71% | 99.29% |
| test | 1.00% | 0.235294 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |
| val | 0.50% | 0.254902 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 0.235294 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.254902 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 0.235294 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 1160.768 ms/pair, score CSV p50 1123.717 ms, p95 2026.503 ms.
- val: reported avg 1160.781 ms/pair, score CSV p50 1140.335 ms, p95 2032.592 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
