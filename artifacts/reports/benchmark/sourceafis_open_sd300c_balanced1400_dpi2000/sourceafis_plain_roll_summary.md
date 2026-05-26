# SourceAFIS Plain/Roll Benchmark

Provider: `sourceafis_open` through the fingerprint_engine abstraction and the SourceAFIS HTTP sidecar.

Score semantics: SourceAFIS raw similarity score; higher means a stronger match. Scores are not normalized. Thresholds are calibrated on VAL only and then applied unchanged to TEST.

Total runtime: 2353.65 s
Template extraction failures: 0
Scoring failures: 0
Extraction timeout failures: 0
Transport failures: 0
Extraction invalid image failures: 0

## Runtime Settings

- Request/read timeout: 60.0 s
- Extract timeout: 240.0 s
- Verify timeout: 120.0 s
- Connect timeout: 5.0 s
- Max retries: 1
- Retry backoff: 1.00 s
- Sample strategy: `balanced_spread`
- Sample seed: 13
- Sidecar warmup: ok in 264.68 ms

## DPI Handling

- DPI strategy: `explicit`
- image_dpi argument: 2000
- Inferred DPI counts: none
- Images with unknown DPI: 0

## Dataset Protocols

| dataset | split | compatible | pairs | positives | negatives | reason | pairs CSV |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| nist_sd300c | val | True | 1400 | 700 | 700 | positive/negative same-finger plain-vs-roll pairs | `C:\fingerprint-research\data\manifests\nist_sd300c\pairs_val.csv` |
| nist_sd300c | test | True | 1400 | 700 | 700 | positive/negative same-finger plain-vs-roll pairs | `C:\fingerprint-research\data\manifests\nist_sd300c\pairs_test.csv` |

## TEST Operating Points

| dataset | target FAR | threshold | VAL FAR | TEST TAR | TEST FAR | TEST FRR | TA | FR | FA | TR | negatives enough |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nist_sd300c | 0.01% | 42.343268 | 0.00% | 64.29% | 0.00% | 35.71% | 450 | 250 | 0 | 700 | False |
| nist_sd300c | 0.10% | 42.343268 | 0.00% | 64.29% | 0.00% | 35.71% | 450 | 250 | 0 | 700 | False |
| nist_sd300c | 0.50% | 20.060420 | 0.43% | 75.29% | 0.57% | 24.71% | 527 | 173 | 4 | 696 | True |
| nist_sd300c | 1.00% | 14.483464 | 1.00% | 78.00% | 1.29% | 22.00% | 546 | 154 | 9 | 691 | True |

## AUC And EER

| dataset | split | AUC | EER | EER threshold | scored pairs | unscored pairs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| nist_sd300c | test | 0.8815 | 17.43% | 3.309443 | 1400 | 0 |
| nist_sd300c | val | 0.8962 | 15.14% | 3.030303 | 1400 | 0 |

## Calibration

| dataset | target FAR | threshold | VAL calibration FAR | false accepts / negatives | minimum negatives for target | enough negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nist_sd300c | 0.01% | 42.343268 | 0.00% | 0/700 | 10000 | False |
| nist_sd300c | 0.10% | 42.343268 | 0.00% | 0/700 | 1000 | False |
| nist_sd300c | 0.50% | 20.060420 | 0.43% | 3/700 | 200 | True |
| nist_sd300c | 1.00% | 14.483464 | 1.00% | 7/700 | 100 | True |

## Latency

| dataset | split | operation | status | count | p50 ms | p95 ms | mean ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| nist_sd300c | test | template_cache_lookup | ok | 1382 | 0.000 | 0.000 | 0.000 |
| nist_sd300c | test | template_extraction | ok | 1418 | 545.014 | 805.020 | 548.248 |
| nist_sd300c | test | verification | ok | 1400 | 249.619 | 262.657 | 249.966 |
| nist_sd300c | val | template_cache_lookup | ok | 1392 | 0.000 | 0.000 | 0.000 |
| nist_sd300c | val | template_extraction | ok | 1408 | 589.639 | 825.803 | 570.228 |
| nist_sd300c | val | verification | ok | 1400 | 265.446 | 279.565 | 265.390 |

## Output Schema

- `sourceafis_plain_roll_scores_val.csv`: dataset, split, pair_id, label, is_positive, subject_a, subject_b, finger_position, path_a, path_b, dpi_a, dpi_b, raw_score, score_semantics, higher_is_more_similar, provider_id, provider_version, template_format, template_version, extraction_cache_hit_a, extraction_cache_hit_b, extraction_latency_ms_a, extraction_latency_ms_b, extraction_retry_count_a, extraction_retry_count_b, verification_latency_ms, verification_wall_latency_ms, verification_retry_count, normalized_score_returned, warnings, error
- `sourceafis_plain_roll_scores_test.csv`: dataset, split, pair_id, label, is_positive, subject_a, subject_b, finger_position, path_a, path_b, dpi_a, dpi_b, raw_score, score_semantics, higher_is_more_similar, provider_id, provider_version, template_format, template_version, extraction_cache_hit_a, extraction_cache_hit_b, extraction_latency_ms_a, extraction_latency_ms_b, extraction_retry_count_a, extraction_retry_count_b, verification_latency_ms, verification_wall_latency_ms, verification_retry_count, normalized_score_returned, warnings, error
- `sourceafis_plain_roll_thresholds.csv`: dataset, target_far, threshold, calibration_split, calibration_negative_count, calibration_positive_count, calibration_false_accepts, calibration_far, enough_negatives_for_target, minimum_negatives_for_target, selection_rule, higher_is_more_similar
- `sourceafis_plain_roll_metrics.csv`: dataset, split, target_far, threshold, threshold_split, threshold_val_far, threshold_val_false_accepts, tar, far, frr, ta, fr, fa, tr, n_positive, n_negative, n_scored, n_unscored, auc, eer, eer_threshold, enough_negatives_for_target, minimum_negatives_for_target, score_count, score_min, score_p05, score_p25, score_median, score_mean, score_p75, score_p95, score_max, positive_score_mean, negative_score_mean
- `sourceafis_plain_roll_latency_summary.csv`: dataset, split, operation, status, count, cache_hits, cache_misses, min_ms, p50_ms, mean_ms, p95_ms, max_ms, total_ms
- `sourceafis_plain_roll_failures.csv`: dataset, split, pair_id, operation, path, subject_a, subject_b, finger_position, retry_count, cached_failure, failure_category, error_type, error_message
- `sourceafis_vs_sift_v2_comparison.csv`: optional comparison against existing SIFT v2 evidence.
- `sourceafis_vs_sift_v2_comparison.md`: optional comparison against existing SIFT v2 evidence.

## Artifacts

- scores_val: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_scores_val.csv`
- scores_test: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_scores_test.csv`
- thresholds: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_thresholds.csv`
- metrics: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_metrics.csv`
- latency_summary: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_latency_summary.csv`
- failures: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_failures.csv`
- summary: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_summary.md`
- manifest: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_plain_roll_manifest.json`
- comparison_csv: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_vs_sift_v2_comparison.csv`
- comparison_markdown: `C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_open_sd300c_balanced1400_dpi2000\sourceafis_vs_sift_v2_comparison.md`
