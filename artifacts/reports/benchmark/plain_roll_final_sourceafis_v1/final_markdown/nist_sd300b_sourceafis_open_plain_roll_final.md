# sourceafis_open nist_sd300b Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 17.393218 | 76.00% | 0.43% | 24.00% | 532/168/3/697 | 0.8902 | 0.1700 |
| test | 1.00% | 14.855737 | 77.14% | 0.71% | 22.86% | 540/160/5/695 | 0.8902 | 0.1700 |
| val | 0.50% | 17.393218 | 76.86% | 0.43% | 23.14% | 538/162/3/697 | 0.8899 | 0.1586 |
| val | 1.00% | 14.855737 | 78.14% | 1.00% | 21.86% | 547/153/7/693 | 0.8899 | 0.1586 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 30.373939 | 0.00% | 70.71% | 495 | 205 | 0 | 700 |
| 0.10% | 30.373939 | 0.00% | 70.71% | 495 | 205 | 0 | 700 |
| 0.25% | 27.216095 | 0.14% | 71.86% | 503 | 197 | 1 | 699 |
| 0.50% | 17.159973 | 0.43% | 76.29% | 534 | 166 | 3 | 697 |
| 1.00% | 13.073319 | 1.00% | 78.00% | 546 | 154 | 7 | 693 |
| 2.00% | 10.613628 | 1.86% | 78.57% | 550 | 150 | 13 | 687 |
| 5.00% | 7.529935 | 4.57% | 79.57% | 557 | 143 | 32 | 668 |
| 10.00% | 4.509821 | 9.57% | 81.43% | 570 | 130 | 67 | 633 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 17.393218 | 700 | 532 | 168 | 76.00% | 24.00% |
| test | 1.00% | 14.855737 | 700 | 540 | 160 | 77.14% | 22.86% |
| val | 0.50% | 17.393218 | 700 | 538 | 162 | 76.86% | 23.14% |
| val | 1.00% | 14.855737 | 700 | 547 | 153 | 78.14% | 21.86% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 17.393218 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| test | 1.00% | 14.855737 | 1.00% | 7 | 700 | 5 | 695 | 0.71% | 99.29% |
| val | 0.50% | 17.393218 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 14.855737 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 17.393218 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 14.855737 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 163.056 ms/pair, score CSV p50 179.593 ms, p95 370.177 ms.
- val: reported avg 159.735 ms/pair, score CSV p50 179.173 ms, p95 355.235 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
