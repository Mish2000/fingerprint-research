# sourceafis_open nist_sd300c Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 21.468048 | 74.00% | 0.43% | 26.00% | 518/182/3/697 | 0.8815 | 0.1750 |
| test | 1.00% | 15.858243 | 77.57% | 1.14% | 22.43% | 543/157/8/692 | 0.8815 | 0.1750 |
| val | 0.50% | 21.468048 | 74.00% | 0.43% | 26.00% | 518/182/3/697 | 0.8962 | 0.1464 |
| val | 1.00% | 15.858243 | 76.57% | 1.00% | 23.43% | 536/164/7/693 | 0.8962 | 0.1464 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 32.534031 | 0.00% | 69.71% | 488 | 212 | 0 | 700 |
| 0.10% | 32.534031 | 0.00% | 69.71% | 488 | 212 | 0 | 700 |
| 0.25% | 29.127558 | 0.14% | 71.43% | 500 | 200 | 1 | 699 |
| 0.50% | 20.239358 | 0.43% | 75.29% | 527 | 173 | 3 | 697 |
| 1.00% | 16.346105 | 1.00% | 77.43% | 542 | 158 | 7 | 693 |
| 2.00% | 12.493146 | 2.00% | 78.43% | 549 | 151 | 14 | 686 |
| 5.00% | 8.576031 | 5.00% | 80.29% | 562 | 138 | 35 | 665 |
| 10.00% | 5.223237 | 9.57% | 81.29% | 569 | 131 | 67 | 633 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 21.468048 | 700 | 518 | 182 | 74.00% | 26.00% |
| test | 1.00% | 15.858243 | 700 | 543 | 157 | 77.57% | 22.43% |
| val | 0.50% | 21.468048 | 700 | 518 | 182 | 74.00% | 26.00% |
| val | 1.00% | 15.858243 | 700 | 536 | 164 | 76.57% | 23.43% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 21.468048 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| test | 1.00% | 15.858243 | 1.00% | 7 | 700 | 8 | 692 | 1.14% | 98.86% |
| val | 0.50% | 21.468048 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 15.858243 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 21.468048 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 15.858243 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 338.395 ms/pair, score CSV p50 396.687 ms, p95 726.070 ms.
- val: reported avg 335.377 ms/pair, score CSV p50 410.626 ms, p95 724.103 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
