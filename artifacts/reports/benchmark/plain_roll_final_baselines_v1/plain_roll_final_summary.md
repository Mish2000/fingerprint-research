# Plain/Roll Final Benchmark

Created: `2026-06-09T11:35:15Z`
Total runtime: `16.12s`

## Protocol

- Datasets: NIST SD300B and NIST SD300C unless overridden.
- Splits: VAL and TEST.
- Pair filter: one plain capture and one rolled capture.
- Labels: positive pairs must share subject, negative pairs must use different subjects.
- Finger protocol: selected pairs preserve `frgp` or `finger_id` as `finger_position`.
- Thresholds: calibrated on VAL negative scores only and applied unchanged to VAL and TEST.

Although scoring may be executed on one selected-pair CSV for reproducibility, positive and negative outcomes are audited and reported separately. TAR/FRR are computed only from positive pairs, and FAR/TNR are computed only from negative pairs.

## Expert TAR/FAR Distribution Summary

- Fixed operating points show selected calibrated thresholds from VAL negatives applied unchanged to VAL and TEST.
- The threshold sweep shows the full behavior across candidate thresholds from each score CSV.
- TAR/FRR are computed only from positive pairs.
- FAR/TNR are computed only from negative pairs.
- FA means negative pairs incorrectly accepted as matches.
- TR means negative pairs correctly rejected.
- TAR/FAR distribution rows maximize TAR within each FAR ceiling; tied TAR rows use the highest threshold as the more conservative operating point.

| method | dataset | split | FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classic_v2 | nist_sd300b | test | 0.00% | 0.014000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300b | test | 0.10% | 0.014000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300b | test | 0.25% | 0.014000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300b | test | 0.50% | 0.004667 | 0.43% | 1.29% | 9 | 691 | 3 | 697 |
| classic_v2 | nist_sd300b | test | 1.00% | 0.004667 | 0.43% | 1.29% | 9 | 691 | 3 | 697 |
| classic_v2 | nist_sd300b | test | 2.00% | 0.004000 | 1.57% | 2.14% | 15 | 685 | 11 | 689 |
| classic_v2 | nist_sd300b | test | 5.00% | 0.003333 | 3.14% | 3.29% | 23 | 677 | 22 | 678 |
| classic_v2 | nist_sd300b | test | 10.00% | 0.002667 | 7.14% | 7.29% | 51 | 649 | 50 | 650 |
| classic_v2 | nist_sd300c | test | 0.00% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300c | test | 0.10% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300c | test | 0.25% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300c | test | 0.50% | 0.010000 | 0.00% | 0.14% | 1 | 699 | 0 | 700 |
| classic_v2 | nist_sd300c | test | 1.00% | 0.004667 | 1.00% | 0.57% | 4 | 696 | 7 | 693 |
| classic_v2 | nist_sd300c | test | 2.00% | 0.004000 | 1.71% | 1.43% | 10 | 690 | 12 | 688 |
| classic_v2 | nist_sd300c | test | 5.00% | 0.003333 | 3.43% | 3.14% | 22 | 678 | 24 | 676 |
| classic_v2 | nist_sd300c | test | 10.00% | 0.002667 | 7.71% | 7.29% | 51 | 649 | 54 | 646 |
| harris | nist_sd300b | test | 0.00% | 0.015044 | 0.00% | 0.29% | 2 | 698 | 0 | 700 |
| harris | nist_sd300b | test | 0.10% | 0.015044 | 0.00% | 0.29% | 2 | 698 | 0 | 700 |
| harris | nist_sd300b | test | 0.25% | 0.015044 | 0.00% | 0.29% | 2 | 698 | 0 | 700 |
| harris | nist_sd300b | test | 0.50% | 0.008636 | 0.43% | 0.71% | 5 | 695 | 3 | 697 |
| harris | nist_sd300b | test | 1.00% | 0.008249 | 0.86% | 0.86% | 6 | 694 | 6 | 694 |
| harris | nist_sd300b | test | 2.00% | 0.007847 | 1.57% | 1.14% | 8 | 692 | 11 | 689 |
| harris | nist_sd300b | test | 5.00% | 0.005674 | 4.86% | 4.57% | 32 | 668 | 34 | 666 |
| harris | nist_sd300b | test | 10.00% | 0.004794 | 9.86% | 8.43% | 59 | 641 | 69 | 631 |
| harris | nist_sd300c | test | 0.00% | 0.018332 | 0.00% | 0.00% | 0 | 700 | 0 | 700 |
| harris | nist_sd300c | test | 0.10% | 0.018332 | 0.00% | 0.00% | 0 | 700 | 0 | 700 |
| harris | nist_sd300c | test | 0.25% | 0.012038 | 0.14% | 0.43% | 3 | 697 | 1 | 699 |
| harris | nist_sd300c | test | 0.50% | 0.008815 | 0.43% | 0.86% | 6 | 694 | 3 | 697 |
| harris | nist_sd300c | test | 1.00% | 0.007442 | 1.00% | 1.43% | 10 | 690 | 7 | 693 |
| harris | nist_sd300c | test | 2.00% | 0.006897 | 2.00% | 2.71% | 19 | 681 | 14 | 686 |
| harris | nist_sd300c | test | 5.00% | 0.005902 | 4.71% | 5.14% | 36 | 664 | 33 | 667 |
| harris | nist_sd300c | test | 10.00% | 0.004798 | 9.86% | 10.00% | 70 | 630 | 69 | 631 |
| minutiae | nist_sd300b | test | 0.00% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| minutiae | nist_sd300b | test | 0.10% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| minutiae | nist_sd300b | test | 0.25% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| minutiae | nist_sd300b | test | 0.50% | 0.238806 | 0.00% | 1.00% | 7 | 693 | 0 | 700 |
| minutiae | nist_sd300b | test | 1.00% | 0.235294 | 0.57% | 1.29% | 9 | 691 | 4 | 696 |
| minutiae | nist_sd300b | test | 2.00% | 0.220752 | 1.43% | 1.86% | 13 | 687 | 10 | 690 |
| minutiae | nist_sd300b | test | 5.00% | 0.202133 | 4.14% | 3.43% | 24 | 676 | 29 | 671 |
| minutiae | nist_sd300b | test | 10.00% | 0.176877 | 9.86% | 9.86% | 69 | 631 | 69 | 631 |
| minutiae | nist_sd300c | test | 0.00% | 0.294118 | 0.00% | 0.43% | 3 | 697 | 0 | 700 |
| minutiae | nist_sd300c | test | 0.10% | 0.294118 | 0.00% | 0.43% | 3 | 697 | 0 | 700 |
| minutiae | nist_sd300c | test | 0.25% | 0.279152 | 0.14% | 0.57% | 4 | 696 | 1 | 699 |
| minutiae | nist_sd300c | test | 0.50% | 0.279152 | 0.14% | 0.57% | 4 | 696 | 1 | 699 |
| minutiae | nist_sd300c | test | 1.00% | 0.232912 | 1.00% | 1.57% | 11 | 689 | 7 | 693 |
| minutiae | nist_sd300c | test | 2.00% | 0.216506 | 2.00% | 1.86% | 13 | 687 | 14 | 686 |
| minutiae | nist_sd300c | test | 5.00% | 0.191176 | 5.00% | 6.57% | 46 | 654 | 35 | 665 |
| minutiae | nist_sd300c | test | 10.00% | 0.176649 | 7.86% | 11.29% | 79 | 621 | 55 | 645 |
| sift | nist_sd300b | test | 0.00% | 0.018000 | 0.00% | 11.71% | 82 | 618 | 0 | 700 |
| sift | nist_sd300b | test | 0.10% | 0.018000 | 0.00% | 11.71% | 82 | 618 | 0 | 700 |
| sift | nist_sd300b | test | 0.25% | 0.014667 | 0.14% | 16.29% | 114 | 586 | 1 | 699 |
| sift | nist_sd300b | test | 0.50% | 0.011992 | 0.43% | 20.71% | 145 | 555 | 3 | 697 |
| sift | nist_sd300b | test | 1.00% | 0.007995 | 0.86% | 31.43% | 220 | 480 | 6 | 694 |
| sift | nist_sd300b | test | 2.00% | 0.005476 | 1.71% | 40.71% | 285 | 415 | 12 | 688 |
| sift | nist_sd300b | test | 5.00% | 0.004664 | 3.86% | 50.14% | 351 | 349 | 27 | 673 |
| sift | nist_sd300b | test | 10.00% | 0.003757 | 6.86% | 57.57% | 403 | 297 | 48 | 652 |
| sift | nist_sd300c | test | 0.00% | 0.013333 | 0.00% | 17.86% | 125 | 575 | 0 | 700 |
| sift | nist_sd300c | test | 0.10% | 0.013333 | 0.00% | 17.86% | 125 | 575 | 0 | 700 |
| sift | nist_sd300c | test | 0.25% | 0.010660 | 0.14% | 22.57% | 158 | 542 | 1 | 699 |
| sift | nist_sd300c | test | 0.50% | 0.009327 | 0.43% | 27.29% | 191 | 509 | 3 | 697 |
| sift | nist_sd300c | test | 1.00% | 0.007995 | 1.00% | 32.43% | 227 | 473 | 7 | 693 |
| sift | nist_sd300c | test | 2.00% | 0.006667 | 1.71% | 38.14% | 267 | 433 | 12 | 688 |
| sift | nist_sd300c | test | 5.00% | 0.004664 | 4.57% | 49.86% | 349 | 351 | 32 | 668 |
| sift | nist_sd300c | test | 10.00% | 0.003997 | 8.00% | 56.71% | 397 | 303 | 56 | 644 |
| sift_plain_roll_v2 | nist_sd300b | test | 0.00% | 11.922592 | 0.00% | 38.29% | 268 | 432 | 0 | 700 |
| sift_plain_roll_v2 | nist_sd300b | test | 0.10% | 11.922592 | 0.00% | 38.29% | 268 | 432 | 0 | 700 |
| sift_plain_roll_v2 | nist_sd300b | test | 0.25% | 9.055633 | 0.14% | 43.14% | 302 | 398 | 1 | 699 |
| sift_plain_roll_v2 | nist_sd300b | test | 0.50% | 8.015417 | 0.43% | 46.43% | 325 | 375 | 3 | 697 |
| sift_plain_roll_v2 | nist_sd300b | test | 1.00% | 6.976152 | 1.00% | 49.71% | 348 | 352 | 7 | 693 |
| sift_plain_roll_v2 | nist_sd300b | test | 2.00% | 6.047334 | 2.00% | 51.43% | 360 | 340 | 14 | 686 |
| sift_plain_roll_v2 | nist_sd300b | test | 5.00% | 4.835804 | 5.00% | 55.43% | 388 | 312 | 35 | 665 |
| sift_plain_roll_v2 | nist_sd300b | test | 10.00% | 4.093485 | 9.29% | 59.29% | 415 | 285 | 65 | 635 |
| sift_plain_roll_v2 | nist_sd300c | test | 0.00% | 16.888704 | 0.00% | 31.29% | 219 | 481 | 0 | 700 |
| sift_plain_roll_v2 | nist_sd300c | test | 0.10% | 16.888704 | 0.00% | 31.29% | 219 | 481 | 0 | 700 |
| sift_plain_roll_v2 | nist_sd300c | test | 0.25% | 9.121438 | 0.14% | 41.43% | 290 | 410 | 1 | 699 |
| sift_plain_roll_v2 | nist_sd300c | test | 0.50% | 7.534219 | 0.43% | 43.71% | 306 | 394 | 3 | 697 |
| sift_plain_roll_v2 | nist_sd300c | test | 1.00% | 6.770636 | 1.00% | 45.86% | 321 | 379 | 7 | 693 |
| sift_plain_roll_v2 | nist_sd300c | test | 2.00% | 5.746851 | 2.00% | 48.71% | 341 | 359 | 14 | 686 |
| sift_plain_roll_v2 | nist_sd300c | test | 5.00% | 4.634746 | 5.00% | 53.43% | 374 | 326 | 35 | 665 |
| sift_plain_roll_v2 | nist_sd300c | test | 10.00% | 3.846939 | 9.57% | 58.43% | 409 | 291 | 67 | 633 |

## TEST Operating Points

| method | dataset | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER | avg ms/pair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| classic_v2 | nist_sd300b | 0.50% | 0.008667 | 0.14% | 0.00% | 99.86% | 1/699/0/700 | 0.5048 | 0.4971 | 29.042 |
| classic_v2 | nist_sd300b | 1.00% | 0.006667 | 0.43% | 0.29% | 99.57% | 3/697/2/698 | 0.5048 | 0.4971 | 29.042 |
| classic_v2 | nist_sd300c | 0.50% | 0.005333 | 0.29% | 0.57% | 99.71% | 2/698/4/696 | 0.4776 | 0.5250 | 51.954 |
| classic_v2 | nist_sd300c | 1.00% | 0.004667 | 0.57% | 1.00% | 99.43% | 4/696/7/693 | 0.4776 | 0.5250 | 51.954 |
| harris | nist_sd300b | 0.50% | 0.018072 | 0.14% | 0.00% | 99.86% | 1/699/0/700 | 0.5034 | 0.4936 | 561.351 |
| harris | nist_sd300b | 1.00% | 0.010370 | 0.57% | 0.29% | 99.43% | 4/696/2/698 | 0.5034 | 0.4936 | 561.351 |
| harris | nist_sd300c | 0.50% | 0.015260 | 0.14% | 0.14% | 99.86% | 1/699/1/699 | 0.4960 | 0.5071 | 576.126 |
| harris | nist_sd300c | 1.00% | 0.009778 | 0.43% | 0.43% | 99.57% | 3/697/3/697 | 0.4960 | 0.5071 | 576.126 |
| minutiae | nist_sd300b | 0.50% | 0.288462 | 0.00% | 0.00% | 100.00% | 0/700/0/700 | 0.5230 | 0.4807 | 1135.213 |
| minutiae | nist_sd300b | 1.00% | 0.243243 | 0.86% | 0.00% | 99.14% | 6/694/0/700 | 0.5230 | 0.4807 | 1135.213 |
| minutiae | nist_sd300c | 0.50% | 0.254902 | 0.86% | 0.71% | 99.14% | 6/694/5/695 | 0.5285 | 0.4764 | 1160.768 |
| minutiae | nist_sd300c | 1.00% | 0.235294 | 1.43% | 1.00% | 98.57% | 10/690/7/693 | 0.5285 | 0.4764 | 1160.768 |
| sift | nist_sd300b | 0.50% | 0.010667 | 22.71% | 0.57% | 77.29% | 159/541/4/696 | 0.8049 | 0.2757 | 61.307 |
| sift | nist_sd300b | 1.00% | 0.008667 | 28.00% | 0.71% | 72.00% | 196/504/5/695 | 0.8049 | 0.2757 | 61.307 |
| sift | nist_sd300c | 0.50% | 0.010667 | 22.29% | 0.14% | 77.71% | 156/544/1/699 | 0.7912 | 0.2936 | 86.194 |
| sift | nist_sd300c | 1.00% | 0.008667 | 29.57% | 0.86% | 70.43% | 207/493/6/694 | 0.7912 | 0.2936 | 86.194 |
| sift_plain_roll_v2 | nist_sd300b | 0.50% | 8.124565 | 45.71% | 0.43% | 54.29% | 320/380/3/697 | 0.7882 | 0.2957 | 121.382 |
| sift_plain_roll_v2 | nist_sd300b | 1.00% | 6.654213 | 50.00% | 1.57% | 50.00% | 350/350/11/689 | 0.7882 | 0.2957 | 121.382 |
| sift_plain_roll_v2 | nist_sd300c | 0.50% | 10.257961 | 39.57% | 0.14% | 60.43% | 277/423/1/699 | 0.7859 | 0.2879 | 155.324 |
| sift_plain_roll_v2 | nist_sd300c | 1.00% | 7.892201 | 43.14% | 0.43% | 56.86% | 302/398/3/697 | 0.7859 | 0.2879 | 155.324 |

## VAL Calibration

| method | dataset | target FAR | threshold | VAL FAR | false accepts / negatives | enough negatives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| classic_v2 | nist_sd300b | 0.50% | 0.008667 | 0.14% | 1/700 | True |
| classic_v2 | nist_sd300b | 1.00% | 0.006667 | 0.57% | 4/700 | True |
| classic_v2 | nist_sd300c | 0.50% | 0.005333 | 0.29% | 2/700 | True |
| classic_v2 | nist_sd300c | 1.00% | 0.004667 | 1.00% | 7/700 | True |
| harris | nist_sd300b | 0.50% | 0.018072 | 0.43% | 3/700 | True |
| harris | nist_sd300b | 1.00% | 0.010370 | 1.00% | 7/700 | True |
| harris | nist_sd300c | 0.50% | 0.015260 | 0.43% | 3/700 | True |
| harris | nist_sd300c | 1.00% | 0.009778 | 1.00% | 7/700 | True |
| minutiae | nist_sd300b | 0.50% | 0.288462 | 0.29% | 2/700 | True |
| minutiae | nist_sd300b | 1.00% | 0.243243 | 1.00% | 7/700 | True |
| minutiae | nist_sd300c | 0.50% | 0.254902 | 0.43% | 3/700 | True |
| minutiae | nist_sd300c | 1.00% | 0.235294 | 1.00% | 7/700 | True |
| sift | nist_sd300b | 0.50% | 0.010667 | 0.43% | 3/700 | True |
| sift | nist_sd300b | 1.00% | 0.008667 | 1.00% | 7/700 | True |
| sift | nist_sd300c | 0.50% | 0.010667 | 0.43% | 3/700 | True |
| sift | nist_sd300c | 1.00% | 0.008667 | 1.00% | 7/700 | True |
| sift_plain_roll_v2 | nist_sd300b | 0.50% | 8.124565 | 0.43% | 3/700 | True |
| sift_plain_roll_v2 | nist_sd300b | 1.00% | 6.654213 | 1.00% | 7/700 | True |
| sift_plain_roll_v2 | nist_sd300c | 0.50% | 10.257961 | 0.43% | 3/700 | True |
| sift_plain_roll_v2 | nist_sd300c | 1.00% | 7.892201 | 1.00% | 7/700 | True |

## Latency

| method | dataset | split | N | reported avg ms | score CSV p50 ms | score CSV p95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| classic_v2 | nist_sd300b | test | 1400 | 29.042 | 29.373 | 59.502 |
| classic_v2 | nist_sd300b | val | 1400 | 29.049 | 29.028 | 60.734 |
| classic_v2 | nist_sd300c | test | 1400 | 51.954 | 53.358 | 106.678 |
| classic_v2 | nist_sd300c | val | 1400 | 52.046 | 60.857 | 108.979 |
| harris | nist_sd300b | test | 1400 | 561.351 | 561.203 | 1169.786 |
| harris | nist_sd300b | val | 1400 | 572.347 | 572.669 | 1188.509 |
| harris | nist_sd300c | test | 1400 | 576.126 | 576.232 | 1188.818 |
| harris | nist_sd300c | val | 1400 | 601.825 | 604.137 | 1243.558 |
| minutiae | nist_sd300b | test | 1400 | 1135.213 | 1104.145 | 1962.214 |
| minutiae | nist_sd300b | val | 1400 | 1123.656 | 1107.283 | 1963.871 |
| minutiae | nist_sd300c | test | 1400 | 1160.768 | 1123.717 | 2026.503 |
| minutiae | nist_sd300c | val | 1400 | 1160.781 | 1140.335 | 2032.592 |
| sift | nist_sd300b | test | 1400 | 61.307 | 59.348 | 115.306 |
| sift | nist_sd300b | val | 1400 | 59.990 | 59.509 | 110.881 |
| sift | nist_sd300c | test | 1400 | 86.194 | 93.535 | 167.254 |
| sift | nist_sd300c | val | 1400 | 90.544 | 98.187 | 173.885 |
| sift_plain_roll_v2 | nist_sd300b | test | 1400 | 121.382 | 119.932 | 223.259 |
| sift_plain_roll_v2 | nist_sd300b | val | 1400 | 117.260 | 117.287 | 216.963 |
| sift_plain_roll_v2 | nist_sd300c | test | 1400 | 155.324 | 158.460 | 306.743 |
| sift_plain_roll_v2 | nist_sd300c | val | 1400 | 149.027 | 154.918 | 280.647 |

## Pair Audit

| dataset | split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nist_sd300b | test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| nist_sd300b | val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| nist_sd300c | test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| nist_sd300c | val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |

## Selected Pair Sets

- nist_sd300b test: 1400 pairs (700 positive, 700 negative), source `C:\fingerprint-research\data\manifests\nist_sd300b\pairs_test.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300b_test.csv`
- nist_sd300b val: 1400 pairs (700 positive, 700 negative), source `C:\fingerprint-research\data\manifests\nist_sd300b\pairs_val.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300b_val.csv`
- nist_sd300c test: 1400 pairs (700 positive, 700 negative), source `C:\fingerprint-research\data\manifests\nist_sd300c\pairs_test.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300c_test.csv`
- nist_sd300c val: 1400 pairs (700 positive, 700 negative), source `C:\fingerprint-research\data\manifests\nist_sd300c\pairs_val.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300c_val.csv`

## Artifacts

- failures: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_failures.csv`
- latency_summary: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_latency_summary.csv`
- manifest: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_manifest.json`
- markdown_nist_sd300b_classic_v2: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300b_classic_v2_plain_roll_final.md`
- markdown_nist_sd300b_harris: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300b_harris_plain_roll_final.md`
- markdown_nist_sd300b_minutiae: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300b_minutiae_plain_roll_final.md`
- markdown_nist_sd300b_sift: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300b_sift_plain_roll_final.md`
- markdown_nist_sd300b_sift_plain_roll_v2: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300b_sift_plain_roll_v2_plain_roll_final.md`
- markdown_nist_sd300c_classic_v2: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300c_classic_v2_plain_roll_final.md`
- markdown_nist_sd300c_harris: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300c_harris_plain_roll_final.md`
- markdown_nist_sd300c_minutiae: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300c_minutiae_plain_roll_final.md`
- markdown_nist_sd300c_sift: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300c_sift_plain_roll_final.md`
- markdown_nist_sd300c_sift_plain_roll_v2: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\final_markdown\nist_sd300c_sift_plain_roll_v2_plain_roll_final.md`
- metrics: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_metrics.csv`
- negative_only_metrics: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_negative_only_metrics.csv`
- pair_audit_json_nist_sd300b_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300b_test.json`
- pair_audit_json_nist_sd300b_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300b_val.json`
- pair_audit_json_nist_sd300c_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300c_test.json`
- pair_audit_json_nist_sd300c_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300c_val.json`
- pair_audit_markdown_nist_sd300b_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300b_test.md`
- pair_audit_markdown_nist_sd300b_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300b_val.md`
- pair_audit_markdown_nist_sd300c_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300c_test.md`
- pair_audit_markdown_nist_sd300c_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_nist_sd300c_val.md`
- pair_audit_summary: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\pair_audit\pair_audit_summary.md`
- positive_only_metrics: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_positive_only_metrics.csv`
- selected_pairs_nist_sd300b_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300b_test.csv`
- selected_pairs_nist_sd300b_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300b_val.csv`
- selected_pairs_nist_sd300c_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300c_test.csv`
- selected_pairs_nist_sd300c_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\selected_pairs\pairs_nist_sd300c_val.csv`
- summary: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_summary.md`
- tar_far_distribution: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_tar_far_distribution.csv`
- threshold_sweep: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_threshold_sweep.csv`
- thresholds: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_baselines_v1\plain_roll_final_thresholds.csv`

