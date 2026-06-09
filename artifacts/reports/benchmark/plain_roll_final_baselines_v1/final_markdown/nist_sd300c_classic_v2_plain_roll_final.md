# classic_v2 nist_sd300c Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.005333 | 0.29% | 0.57% | 99.71% | 2/698/4/696 | 0.4776 | 0.5250 |
| test | 1.00% | 0.004667 | 0.57% | 1.00% | 99.43% | 4/696/7/693 | 0.4776 | 0.5250 |
| val | 0.50% | 0.005333 | 0.29% | 0.29% | 99.71% | 2/698/2/698 | 0.5071 | 0.4950 |
| val | 1.00% | 0.004667 | 0.71% | 1.00% | 99.29% | 5/695/7/693 | 0.5071 | 0.4950 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 0.10% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 0.25% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 0.50% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| 1.00% | 0.004667 | 1.00% | 0.57% | 4 | 696 | 7 | 693 |
| 2.00% | 0.004000 | 1.71% | 1.43% | 10 | 690 | 12 | 688 |
| 5.00% | 0.003333 | 3.43% | 3.14% | 22 | 678 | 24 | 676 |
| 10.00% | 0.002667 | 7.71% | 7.29% | 51 | 649 | 54 | 646 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.005333 | 700 | 2 | 698 | 0.29% | 99.71% |
| test | 1.00% | 0.004667 | 700 | 4 | 696 | 0.57% | 99.43% |
| val | 0.50% | 0.005333 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 1.00% | 0.004667 | 700 | 5 | 695 | 0.71% | 99.29% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 0.005333 | 0.29% | 2 | 700 | 4 | 696 | 0.57% | 99.43% |
| test | 1.00% | 0.004667 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |
| val | 0.50% | 0.005333 | 0.29% | 2 | 700 | 2 | 698 | 0.29% | 99.71% |
| val | 1.00% | 0.004667 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 0.005333 from VAL negatives, VAL FAR 0.29%.
- Target FAR 1.00%: threshold 0.004667 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 51.954 ms/pair, score CSV p50 53.358 ms, p95 106.678 ms.
- val: reported avg 52.046 ms/pair, score CSV p50 60.857 ms, p95 108.979 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
