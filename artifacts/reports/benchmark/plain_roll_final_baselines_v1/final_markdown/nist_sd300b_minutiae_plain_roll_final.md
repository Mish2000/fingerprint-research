# minutiae nist_sd300b Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.288462 | 0.00% | 0.00% | 100.00% | 0/700/0/700 | 0.5230 | 0.4807 |
| test | 1.00% | 0.243243 | 0.86% | 0.00% | 99.14% | 6/694/0/700 | 0.5230 | 0.4807 |
| val | 0.50% | 0.288462 | 0.14% | 0.29% | 99.86% | 1/699/2/698 | 0.5011 | 0.4979 |
| val | 1.00% | 0.243243 | 0.71% | 1.00% | 99.29% | 5/695/7/693 | 0.5011 | 0.4979 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| 0.10% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| 0.25% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| 0.50% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| 1.00% | 0.235294 | 0.57% | 1.29% | 9 | 691 | 4 | 696 |
| 2.00% | 0.220752 | 1.43% | 1.86% | 13 | 687 | 10 | 690 |
| 5.00% | 0.202133 | 4.14% | 3.43% | 24 | 676 | 29 | 671 |
| 10.00% | 0.176877 | 9.86% | 9.86% | 69 | 631 | 69 | 631 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.288462 | 700 | 0 | 700 | 0.00% | 100.00% |
| test | 1.00% | 0.243243 | 700 | 6 | 694 | 0.86% | 99.14% |
| val | 0.50% | 0.288462 | 700 | 1 | 699 | 0.14% | 99.86% |
| val | 1.00% | 0.243243 | 700 | 5 | 695 | 0.71% | 99.29% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.288462 | 0.29% | 2 | 700 | 0 | 700 | 0.00% | 100.00% |
| test | 1.00% | 0.243243 | 1.00% | 7 | 700 | 0 | 700 | 0.00% | 100.00% |
| val | 0.50% | 0.288462 | 0.29% | 2 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 1.00% | 0.243243 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.288462 from VAL negatives, VAL FAR 0.29%.
- Target FAR 1.00%: threshold 0.243243 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 1135.213 ms/pair, score CSV p50 1104.145 ms, p95 1962.214 ms.
- val: reported avg 1123.656 ms/pair, score CSV p50 1107.283 ms, p95 1963.871 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
