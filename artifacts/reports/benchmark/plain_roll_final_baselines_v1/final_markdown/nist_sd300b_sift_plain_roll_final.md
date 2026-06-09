# sift nist_sd300b Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.010667 | 22.71% | 0.57% | 77.29% | 159/541/4/696 | 0.8049 | 0.2757 |
| test | 1.00% | 0.008667 | 28.00% | 0.71% | 72.00% | 196/504/5/695 | 0.8049 | 0.2757 |
| val | 0.50% | 0.010667 | 22.29% | 0.43% | 77.71% | 156/544/3/697 | 0.7941 | 0.2814 |
| val | 1.00% | 0.008667 | 28.00% | 1.00% | 72.00% | 196/504/7/693 | 0.7941 | 0.2814 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.018000 | 0.00% | 11.71% | 82 | 618 | 0 | 700 |
| 0.10% | 0.018000 | 0.00% | 11.71% | 82 | 618 | 0 | 700 |
| 0.25% | 0.014667 | 0.14% | 16.29% | 114 | 586 | 1 | 699 |
| 0.50% | 0.011992 | 0.43% | 20.71% | 145 | 555 | 3 | 697 |
| 1.00% | 0.007995 | 0.86% | 31.43% | 220 | 480 | 6 | 694 |
| 2.00% | 0.005476 | 1.71% | 40.71% | 285 | 415 | 12 | 688 |
| 5.00% | 0.004664 | 3.86% | 50.14% | 351 | 349 | 27 | 673 |
| 10.00% | 0.003757 | 6.86% | 57.57% | 403 | 297 | 48 | 652 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.010667 | 700 | 159 | 541 | 22.71% | 77.29% |
| test | 1.00% | 0.008667 | 700 | 196 | 504 | 28.00% | 72.00% |
| val | 0.50% | 0.010667 | 700 | 156 | 544 | 22.29% | 77.71% |
| val | 1.00% | 0.008667 | 700 | 196 | 504 | 28.00% | 72.00% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.010667 | 0.43% | 3 | 700 | 4 | 696 | 0.57% | 99.43% |
| test | 1.00% | 0.008667 | 1.00% | 7 | 700 | 5 | 695 | 0.71% | 99.29% |
| val | 0.50% | 0.010667 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 0.008667 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.010667 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 0.008667 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 61.307 ms/pair, score CSV p50 59.348 ms, p95 115.306 ms.
- val: reported avg 59.990 ms/pair, score CSV p50 59.509 ms, p95 110.881 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
