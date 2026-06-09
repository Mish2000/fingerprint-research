# sift_plain_roll_v2 nist_sd300c Plain/Roll Final

Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.

## Fixed operating points

| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 10.257961 | 39.57% | 0.14% | 60.43% | 277/423/1/699 | 0.7859 | 0.2879 |
| test | 1.00% | 7.892201 | 43.14% | 0.43% | 56.86% | 302/398/3/697 | 0.7859 | 0.2879 |
| val | 0.50% | 10.257961 | 35.00% | 0.43% | 65.00% | 245/455/3/697 | 0.7764 | 0.3050 |
| val | 1.00% | 7.892201 | 40.86% | 1.00% | 59.14% | 286/414/7/693 | 0.7764 | 0.3050 |

## TAR vs FAR Distribution

| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00% | 16.888704 | 0.00% | 31.29% | 219 | 481 | 0 | 700 |
| 0.10% | 16.888704 | 0.00% | 31.29% | 219 | 481 | 0 | 700 |
| 0.25% | 9.121438 | 0.14% | 41.43% | 290 | 410 | 1 | 699 |
| 0.50% | 7.534219 | 0.43% | 43.71% | 306 | 394 | 3 | 697 |
| 1.00% | 6.770636 | 1.00% | 45.86% | 321 | 379 | 7 | 693 |
| 2.00% | 5.746851 | 2.00% | 48.71% | 341 | 359 | 14 | 686 |
| 5.00% | 4.634746 | 5.00% | 53.43% | 374 | 326 | 35 | 665 |
| 10.00% | 3.846939 | 9.57% | 58.43% | 409 | 291 | 67 | 633 |

## Positive-only verification evidence

| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 10.257961 | 700 | 277 | 423 | 39.57% | 60.43% |
| test | 1.00% | 7.892201 | 700 | 302 | 398 | 43.14% | 56.86% |
| val | 0.50% | 10.257961 | 700 | 245 | 455 | 35.00% | 65.00% |
| val | 1.00% | 7.892201 | 700 | 286 | 414 | 40.86% | 59.14% |

## Negative-only impostor evidence

| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.50% | 10.257961 | 0.43% | 3 | 700 | 1 | 699 | 0.14% | 99.86% |
| test | 1.00% | 7.892201 | 1.00% | 7 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 0.50% | 10.257961 | 0.43% | 3 | 700 | 3 | 697 | 0.43% | 99.57% |
| val | 1.00% | 7.892201 | 1.00% | 7 | 700 | 7 | 693 | 1.00% | 99.00% |

## Thresholds

- Target FAR 0.50%: threshold 10.257961 from VAL negatives, VAL FAR 0.43%.
- Target FAR 1.00%: threshold 7.892201 from VAL negatives, VAL FAR 1.00%.

## Latency

- test: reported avg 155.324 ms/pair, score CSV p50 158.460 ms, p95 306.743 ms.
- val: reported avg 149.027 ms/pair, score CSV p50 154.918 ms, p95 280.647 ms.

## Pair audit summary

| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
