# classic_v2 nist_sd300b Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.008667 | 0.14% | 0.00% | 99.86% | 1/699/0/700 | 0.5048 | 0.4971 |
| test | 1.00% | 0.006667 | 0.43% | 0.29% | 99.57% | 3/697/2/698 | 0.5048 | 0.4971 |
| val | 0.50% | 0.008667 | 0.29% | 0.14% | 99.71% | 2/698/1/699 | 0.5175 | 0.4857 |
| val | 1.00% | 0.006667 | 0.57% | 0.57% | 99.43% | 4/696/4/696 | 0.5175 | 0.4857 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.014000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 0.10% | 0.014000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 0.25% | 0.014000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 0.50% | 0.004667 | 0.43% | 1.29% | 9 | 691 | 3 | 697 |
| 1.00% | 0.004667 | 0.43% | 1.29% | 9 | 691 | 3 | 697 |
| 2.00% | 0.004000 | 1.57% | 2.14% | 15 | 685 | 11 | 689 |
| 5.00% | 0.003333 | 3.14% | 3.29% | 23 | 677 | 22 | 678 |
| 10.00% | 0.002667 | 7.14% | 7.29% | 51 | 649 | 50 | 650 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.008667 | 700 | 1 | 699 | 0.14% | 99.86% |
| test | 1.00% | 0.006667 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 0.50% | 0.008667 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 1.00% | 0.006667 | 700 | 4 | 696 | 0.57% | 99.43% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.008667 | 0.14% | 1 | 700 | 0 | 700 | 0.00% | 100.00% |
| test | 1.00% | 0.006667 | 0.57% | 4 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 0.50% | 0.008667 | 0.14% | 1 | 700 | 1 | 699 | 0.14% | 99.86% |
| val | 1.00% | 0.006667 | 0.57% | 4 | 700 | 4 | 696 | 0.57% | 99.43% |

## Thresholds

- Target FAR 0.50%: threshold 0.008667 from VAL negatives, VAL FAR 0.14%.
- Target FAR 1.00%: threshold 0.006667 from VAL negatives, VAL FAR 0.57%.

## Latency

- test: reported avg 29.042 ms/pair, score CSV p50 29.373 ms, p95 59.502 ms.
- val: reported avg 29.049 ms/pair, score CSV p50 29.028 ms, p95 60.734 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
