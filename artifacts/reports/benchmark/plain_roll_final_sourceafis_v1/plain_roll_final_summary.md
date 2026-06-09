# Plain/Roll Final Benchmark

Created: `2026-06-09T13:13:21Z`
Total runtime: `1470.35s`

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
| sourceafis_open | nist_sd300b | test | 0.00% | 30.373939 | 0.00% | 70.71% | 495 | 205 | 0 | 700 |
| sourceafis_open | nist_sd300b | test | 0.10% | 30.373939 | 0.00% | 70.71% | 495 | 205 | 0 | 700 |
| sourceafis_open | nist_sd300b | test | 0.25% | 27.216095 | 0.14% | 71.86% | 503 | 197 | 1 | 699 |
| sourceafis_open | nist_sd300b | test | 0.50% | 17.159973 | 0.43% | 76.29% | 534 | 166 | 3 | 697 |
| sourceafis_open | nist_sd300b | test | 1.00% | 13.073319 | 1.00% | 78.00% | 546 | 154 | 7 | 693 |
| sourceafis_open | nist_sd300b | test | 2.00% | 10.613628 | 1.86% | 78.57% | 550 | 150 | 13 | 687 |
| sourceafis_open | nist_sd300b | test | 5.00% | 7.529935 | 4.57% | 79.57% | 557 | 143 | 32 | 668 |
| sourceafis_open | nist_sd300b | test | 10.00% | 4.509821 | 9.57% | 81.43% | 570 | 130 | 67 | 633 |
| sourceafis_open | nist_sd300c | test | 0.00% | 32.534031 | 0.00% | 69.71% | 488 | 212 | 0 | 700 |
| sourceafis_open | nist_sd300c | test | 0.10% | 32.534031 | 0.00% | 69.71% | 488 | 212 | 0 | 700 |
| sourceafis_open | nist_sd300c | test | 0.25% | 29.127558 | 0.14% | 71.43% | 500 | 200 | 1 | 699 |
| sourceafis_open | nist_sd300c | test | 0.50% | 20.239358 | 0.43% | 75.29% | 527 | 173 | 3 | 697 |
| sourceafis_open | nist_sd300c | test | 1.00% | 16.346105 | 1.00% | 77.43% | 542 | 158 | 7 | 693 |
| sourceafis_open | nist_sd300c | test | 2.00% | 12.493146 | 2.00% | 78.43% | 549 | 151 | 14 | 686 |
| sourceafis_open | nist_sd300c | test | 5.00% | 8.576031 | 5.00% | 80.29% | 562 | 138 | 35 | 665 |
| sourceafis_open | nist_sd300c | test | 10.00% | 5.223237 | 9.57% | 81.29% | 569 | 131 | 67 | 633 |

## TEST Operating Points

| method | dataset | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER | avg ms/pair |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sourceafis_open | nist_sd300b | 0.50% | 17.393218 | 76.00% | 0.43% | 24.00% | 532/168/3/697 | 0.8902 | 0.1700 | 163.056 |
| sourceafis_open | nist_sd300b | 1.00% | 14.855737 | 77.14% | 0.71% | 22.86% | 540/160/5/695 | 0.8902 | 0.1700 | 163.056 |
| sourceafis_open | nist_sd300c | 0.50% | 21.468048 | 74.00% | 0.43% | 26.00% | 518/182/3/697 | 0.8815 | 0.1750 | 338.395 |
| sourceafis_open | nist_sd300c | 1.00% | 15.858243 | 77.57% | 1.14% | 22.43% | 543/157/8/692 | 0.8815 | 0.1750 | 338.395 |

## VAL Calibration

| method | dataset | target FAR | threshold | VAL FAR | false accepts / negatives | enough negatives |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| sourceafis_open | nist_sd300b | 0.50% | 17.393218 | 0.43% | 3/700 | True |
| sourceafis_open | nist_sd300b | 1.00% | 14.855737 | 1.00% | 7/700 | True |
| sourceafis_open | nist_sd300c | 0.50% | 21.468048 | 0.43% | 3/700 | True |
| sourceafis_open | nist_sd300c | 1.00% | 15.858243 | 1.00% | 7/700 | True |

## Latency

| method | dataset | split | N | reported avg ms | score CSV p50 ms | score CSV p95 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sourceafis_open | nist_sd300b | test | 1400 | 163.056 | 179.593 | 370.177 |
| sourceafis_open | nist_sd300b | val | 1400 | 159.735 | 179.173 | 355.235 |
| sourceafis_open | nist_sd300c | test | 1400 | 338.395 | 396.687 | 726.070 |
| sourceafis_open | nist_sd300c | val | 1400 | 335.377 | 410.626 | 724.103 |

## Pair Audit

| dataset | split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nist_sd300b | val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| nist_sd300b | test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| nist_sd300c | val | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |
| nist_sd300c | test | True | 1400 | 700 | 700 | 0 | 0 | 0 | 0 | 0 | 0 |

## Selected Pair Sets

- nist_sd300b test: 1400 pairs ( positive,  negative), source `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300b_test.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300b_test.csv`
- nist_sd300b val: 1400 pairs ( positive,  negative), source `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300b_val.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300b_val.csv`
- nist_sd300c test: 1400 pairs ( positive,  negative), source `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300c_test.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300c_test.csv`
- nist_sd300c val: 1400 pairs ( positive,  negative), source `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300c_val.csv`, selected `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300c_val.csv`

## Artifacts

- failures: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_failures.csv`
- latency_summary: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_latency_summary.csv`
- manifest: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_manifest.json`
- markdown_nist_sd300b_sourceafis_open: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\final_markdown\nist_sd300b_sourceafis_open_plain_roll_final.md`
- markdown_nist_sd300c_sourceafis_open: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\final_markdown\nist_sd300c_sourceafis_open_plain_roll_final.md`
- metrics: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_metrics.csv`
- negative_only_metrics: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_negative_only_metrics.csv`
- pair_audit_json_nist_sd300b_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300b_test.json`
- pair_audit_json_nist_sd300b_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300b_val.json`
- pair_audit_json_nist_sd300c_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300c_test.json`
- pair_audit_json_nist_sd300c_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300c_val.json`
- pair_audit_markdown_nist_sd300b_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300b_test.md`
- pair_audit_markdown_nist_sd300b_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300b_val.md`
- pair_audit_markdown_nist_sd300c_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300c_test.md`
- pair_audit_markdown_nist_sd300c_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\pair_audit\pair_audit_nist_sd300c_val.md`
- positive_only_metrics: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_positive_only_metrics.csv`
- raw_sourceafis_manifest: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\run_meta\sourceafis_raw\sourceafis_plain_roll_manifest.json`
- selected_pairs_nist_sd300b_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300b_test.csv`
- selected_pairs_nist_sd300b_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300b_val.csv`
- selected_pairs_nist_sd300c_test: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300c_test.csv`
- selected_pairs_nist_sd300c_val: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\selected_pairs\pairs_nist_sd300c_val.csv`
- summary: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_summary.md`
- tar_far_distribution: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_tar_far_distribution.csv`
- threshold_sweep: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_threshold_sweep.csv`
- thresholds: `C:\fingerprint-research\artifacts\reports\benchmark\plain_roll_final_sourceafis_v1\plain_roll_final_thresholds.csv`
