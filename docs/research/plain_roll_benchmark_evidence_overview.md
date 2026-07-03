# Plain-vs-Roll Benchmark Evidence Overview

## 1. Executive Summary

This document preserves the research narrative behind the plain-vs-roll biometric benchmark work before artifact cleanup. The project began with custom and research matchers over NIST SD300B plain-vs-roll pairs, including `classic_v2`, `minutiae`, `harris`, `sift`, and `dl_quick`. Those early runs were useful for building protocol discipline, but they also showed that most custom methods were weak on plain-to-roll matching. The strongest early custom baseline was SIFT, but even SIFT had low true accept rates at low false accept rates.

Professor/advisor requirements pushed the work away from small demos and toward explicit FAR, FRR, EER, larger validation sets, clear method comparisons, and label/path validation. The professor-facing artifacts selected 1000 positive and 1000 negative SD300B pairs and calibrated thresholds on VAL before reporting locked performance. Label/path validation confirmed that the selected positive pairs were same-subject plain-vs-roll pairs with matching FRGP/finger position, and selected negative pairs were different-subject plain-vs-roll pairs with matching FRGP.

The SIFT plain-roll v2 line improved the SIFT baseline substantially by using a better score family and plain-vs-roll oriented tuning. On the full compatible SD300B/SD300C validation, SIFT v2 reached about 50.2% TAR at 1% FAR on SD300B TEST and 44.7% TAR at 1% FAR on SD300C TEST. This was a meaningful research improvement, but the failure taxonomy showed many remaining hard false rejects, especially on some FRGP/finger positions, and a non-trivial false-accept tradeoff for crop/grid variants.

The project then introduced `fingerprint_engine` as a provider-neutral abstraction and integrated SourceAFIS through a Java HTTP sidecar as `sourceafis_open`. This created a route to evaluate an open AFIS engine without embedding the Java implementation directly in Python, and it left room for future COTS or other AFIS providers.

The critical discovery was DPI/PPI handling. SourceAFIS defaults to 500 DPI when no DPI is supplied. Early SourceAFIS runs without correct DPI looked weak. NIST SD300B requires 1000 DPI/PPI and NIST SD300C requires 2000 DPI/PPI for validated SourceAFIS NIST runs. After the benchmark path supplied and validated the correct DPI, SourceAFIS performance improved sharply. The current strongest validated evidence is the balanced1400 SourceAFIS DPI runs:

- SD300B at DPI 1000: TEST TAR 77.29% at target FAR 1% with actual FAR 0.86%; TEST TAR 76.00% at target FAR 0.5% with actual FAR 0.43%; TEST AUC 0.8902; TEST EER 17.00%.
- SD300C at DPI 2000: TEST TAR 78.00% at target FAR 1% with actual FAR 1.29%; TEST TAR 75.29% at target FAR 0.5% with actual FAR 0.57%; TEST AUC 0.8815; TEST EER 17.43%.

These SourceAFIS balanced1400 DPI runs are the current strongest validated open-AFIS evidence. SIFT v2 remains important as a research baseline and as evidence of the custom-matcher path, but SourceAFIS is substantially stronger as a production-candidate benchmark result.

Durable evidence paths after cleanup:

- `artifacts/reports/benchmark/sourceafis_sd300b_plain_roll_dpi1000_final.md`
- `artifacts/reports/benchmark/sourceafis_sd300c_plain_roll_dpi2000_final.md`
- `artifacts/reports/benchmark/sift_v2_external_validation_final.md`
- `artifacts/reports/identification/self_match_repeatability_1000_top1_final.md`
- `artifacts/reports/identification/vector_reproducibility_final.md`

The detailed source artifact folders referenced later in this document were consolidated into compact final summaries and pruned. Historical folder names are retained only as provenance for how the numbers were originally produced, not as live evidence paths.

## 2. Research Motivation

The original benchmark work used custom/research matchers to understand whether plain impressions could be matched reliably to rolled impressions in NIST data. That was valuable because it made the pair generation, scoring, thresholding, and reporting protocol concrete. It also exposed the limits of lightweight or hand-built approaches.

The early five-method evidence showed that the strongest custom baseline was SIFT, while other methods had very low TAR at low FAR. This made it risky to treat the project as only a custom-matcher exercise. The benchmark needed a way to compare research matchers against an AFIS-style matcher with a more realistic fingerprint representation and matching algorithm.

The `fingerprint_engine` abstraction was introduced to separate benchmark protocol from matcher implementation. It supports provider metadata, template extraction, verification, identification, and quality assessment through a common interface. SourceAFIS was then integrated as the optional `sourceafis_open` provider through a Java sidecar. This lets the benchmark evaluate an open AFIS engine while preserving the possibility of adding COTS, vendor, or alternate open providers later.

Relevant implementation and contract references:

- `docs/sourceafis_sidecar_contract.md`
- `src/fpbench/fingerprint_engine/base.py`
- `src/fpbench/fingerprint_engine/types.py`
- `src/fpbench/fingerprint_engine/registry.py`
- `scripts/diagnostics/run_sourceafis_plain_roll_benchmark.py`

## 3. Datasets and Protocol

The benchmark evidence centers on NIST SD300B and NIST SD300C. Both are used for plain-vs-roll matching:

- NIST SD300B: validated SourceAFIS NIST runs require 1000 DPI/PPI.
- NIST SD300C: validated SourceAFIS NIST runs require 2000 DPI/PPI.

The pair protocol uses same-finger plain-vs-roll comparisons. Positive pairs are genuine comparisons: the same subject and same FRGP/finger position across plain and rolled impressions. Negative pairs are impostor comparisons: different subjects while preserving plain-vs-roll structure and matching FRGP/finger position where applicable.

The project uses VAL/TEST discipline:

- VAL is used for threshold calibration.
- TEST is used for locked evaluation.
- FAR targets are calibrated on VAL negatives and then applied unchanged to TEST.
- TEST TAR, FAR, FRR, AUC, and EER are reported where available.

The professor-facing SD300B selected-pair run used 1000 positives and 1000 negatives, with VAL/TEST split counts recorded in `artifacts/reports/benchmark/nist_sd300b_professor_1000_pos_neg/run_manifest.json`: 493 VAL positives, 507 TEST positives, 501 VAL negatives, and 499 TEST negatives.

The full five-method SD300B matrix used all compatible VAL/TEST pairs recorded in `artifacts/reports/benchmark/nist_sd300b_professor_1to1_five_methods_far_frr/run_manifest.json`.

Legacy/stale note: the older SIFT v2 external validation artifacts reported pre-anatomical-mapping compatible plain-vs-roll SD300B and SD300C VAL/TEST pair counts:

- VAL: 2812 pairs per dataset, with 703 positives and 2109 negatives.
- TEST: 2844 pairs per dataset, with 711 positives and 2133 negatives.

Legacy/stale note: the validated SourceAFIS balanced1400 runs sampled 1400 pairs per split per dataset and are retained as historical evidence only:

- VAL: 700 positives and 700 negatives.
- TEST: 700 positives and 700 negatives.
- Sampling strategy: `balanced_spread`.
- Seed: `13`.

Statistical-validity note for the balanced1400 SourceAFIS runs: each TEST split contains only 700 negative pairs, so 1% FAR corresponds to about 7 expected false accepts and is reasonably interpretable, while 0.5% FAR corresponds to about 3.5 expected false accepts and is still useful but coarse. By contrast, 0.1% FAR would require at least about 1000 negatives for even one expected false accept, and 0.01% FAR would require about 10000 negatives for even one expected false accept. Therefore the 0.1% and 0.01% rows in the SourceAFIS tables should be treated as exploratory threshold-stress points, not strong statistical claims; the strongest headline claims should focus on 1% FAR, 0.5% FAR, AUC, and EER.

## 4. Professor/Advisor Requirements

The professor/advisor requirements shaped the benchmark into a proper biometric evaluation rather than a small demonstration. The important requirements were:

- Report FAR, FRR, and EER, not only accuracy or anecdotal examples.
- Evaluate at low FAR operating points, especially around 1% FAR.
- Use more than small demos, including 1000+ cases.
- Compare several methods under the same protocol.
- Validate that labels and image paths are correct.
- Distinguish threshold calibration from locked evaluation.
- Preserve evidence in summaries, manifests, and metrics files so conclusions can be audited later.

The main professor-facing evidence is in:

- `artifacts/reports/benchmark/nist_sd300b_professor_1to1_five_methods_far_frr/results_summary.md`
- `artifacts/reports/benchmark/nist_sd300b_professor_1to1_five_methods_far_frr/run_manifest.json`
- `artifacts/reports/benchmark/nist_sd300b_professor_1000_pos_neg/results_summary.md`
- `artifacts/reports/benchmark/nist_sd300b_professor_1000_pos_neg/run_manifest.json`
- `artifacts/reports/benchmark/nist_sd300b_professor_1000_pos_neg/calibration/thresholds_far_1pct_from_val.csv`

## 5. Label and Path Validation

Label/path validation evidence exists in `artifacts/reports/benchmark/nist_sd300b_plain_roll_diagnostics/label_path_validation.md` and `artifacts/reports/benchmark/nist_sd300b_plain_roll_diagnostics/label_path_validation.csv`.

The inspected validation summary reports PASS. It found:

- No label/path validation failures.
- No duplicate directed pair-path failures.
- No parse failures.
- No selected-score alignment failures.
- Positive selected pairs are same-subject plain-vs-roll rows with matching FRGP.
- Negative selected pairs are different-subject plain-vs-roll rows with matching FRGP.
- Score CSV path, label, and split rows align with selected pair CSVs.

There is also visual audit evidence in `artifacts/reports/benchmark/nist_sd300b_visual_label_audit/visual_audit_index.csv`. That index contains 80 copied image-audit rows grouped into categories such as negative false accepts, high SIFT negatives, true rejects, positive all-method failures, positive SIFT accepted examples, and positive SIFT failed examples. The visual audit folder contains copied biometric image material and should be treated as heavy/private generated evidence rather than a default tracked artifact.

## 6. Baseline and Early Five-Method Results

The baseline and early professor-facing five-method work used:

- `classic_v2`
- `minutiae`
- `harris`
- `sift`
- `dl_quick`

The inspected professor-facing artifacts did not contain exact `vit` results. If `vit` was part of another historical branch or artifact not inspected here, its numbers are not found in the inspected summaries.

The full SD300B five-method matrix in `artifacts/reports/benchmark/nist_sd300b_professor_1to1_five_methods_far_frr/results_summary.md` showed SIFT as the strongest early custom baseline. TEST AUC/EER were:

| Method | TEST AUC | TEST EER |
| --- | ---: | ---: |
| classic_v2 | 0.5111 | 0.4890 |
| minutiae | 0.5198 | 0.4782 |
| harris | 0.5244 | 0.4747 |
| sift | 0.8039 | 0.2787 |
| dl_quick | 0.5912 | 0.4337 |

At TEST TAR@FAR 1% in the same summary:

| Method | TEST TAR@FAR 1% |
| --- | ---: |
| classic_v2 | 0.0127 |
| minutiae | 0.0127 |
| harris | 0.00985 |
| sift | 0.3530 |
| dl_quick | 0.0295 |

The selected 1000-positive/1000-negative professor-facing run in `artifacts/reports/benchmark/nist_sd300b_professor_1000_pos_neg/results_summary.md` reported these VAL-calibrated 1% FAR operating points over the selected sets:

| Method | Positive TAR | Negative FAR |
| --- | ---: | ---: |
| classic_v2 | 0.011 | 0.002 |
| minutiae | 0.009 | 0.003 |
| harris | 0.005 | 0.006 |
| sift | 0.281 | 0.005 |
| dl_quick | 0.031 | 0.010 |

These results motivated further SIFT work, but they also made clear that the benchmark needed stronger AFIS-style evidence.

## 7. SIFT Plain-Roll v2 Evidence

SIFT plain-roll v2 was created because the canonical SIFT score was the strongest early custom baseline but still rejected many genuine plain-vs-roll pairs at low FAR. The v2 work explored score variants and plain-vs-roll oriented tuning to improve genuine acceptance without giving up low-FAR control.

The official selected-pair comparison in `artifacts/reports/benchmark/nist_sd300b_plain_roll_diagnostics/sift_plain_roll_v2_official_comparison.md` reports the professor-facing 1000-pair continuity result at 1% FAR:

| Method/Score | True Accepts | False Accepts | TAR | FAR |
| --- | ---: | ---: | ---: | ---: |
| SIFT current score | 281/1000 | 5/1000 | 28.1% | 0.5% |
| SIFT inliers | 307/1000 | 6/1000 | 30.7% | 0.6% |
| SIFT plain-roll v2 | 470/1000 | 8/1000 | 47.0% | 0.8% |

The same file reports strict TEST-only evaluation with thresholds calibrated on original VAL rows:

- SIFT plain-roll v2: TEST TAR 48.3%, TEST FAR 0.6%, with 245 true accepts, 262 false rejects, 3 false accepts, and 496 true rejects.
- SIFT inliers: TEST TAR 33.9%, TEST FAR 0.6%.
- SIFT current score: TEST TAR 26.6%, TEST FAR 0.0%.

The broader external validation in `artifacts/reports/benchmark/sift_plain_roll_v2_external_validation/external_validation_summary.md` tested full compatible SD300B and SD300C plain-vs-roll VAL/TEST pairs without further tuning. At 1% FAR on TEST:

| Dataset | SIFT v2 TAR | SIFT v2 FAR | Canonical SIFT current TAR | Canonical SIFT current FAR |
| --- | ---: | ---: | ---: | ---: |
| SD300B | 50.2% | 1.0% | 28.3% | 0.3% |
| SD300C | 44.7% | 0.6% | 29.7% | 0.7% |

At 0.5% FAR on TEST:

| Dataset | SIFT v2 TAR | SIFT v2 FAR | Canonical SIFT current TAR | Canonical SIFT current FAR |
| --- | ---: | ---: | ---: | ---: |
| SD300B | 43.2% | 0.3% | 21.8% | 0.2% |
| SD300C | 42.1% | 0.4% | 22.2% | 0.2% |

SIFT v2 TEST AUC/EER from `artifacts/reports/benchmark/sift_plain_roll_v2_external_validation/per_dataset_metrics.csv`:

| Dataset | TEST AUC | TEST EER |
| --- | ---: | ---: |
| SD300B | 0.7962695912 | 0.2932489451 |
| SD300C | 0.7913914555 | 0.2827004219 |

SIFT v2 improved the custom matcher baseline, but it remained research-only and did not solve plain-vs-roll matching. The external validation summary notes that even at 10% FAR the average v2 TEST TAR was still about 59.9%, and many false rejects remained hard failures.

## 8. SIFT Failure Taxonomy

The SIFT failure taxonomy in `artifacts/reports/benchmark/sift_plain_roll_v2_failure_taxonomy/` explains why v2 improved performance but remained limited.

At 1% FAR across SD300B and SD300C, `failure_taxonomy_summary.md` reports:

- SIFT v2 rescued 277 positives compared with canonical SIFT.
- SIFT v2 lost 14 positives compared with canonical SIFT.
- SD300B: 159 rescued positives, 3 lost positives, 22 new false accepts, and 7 fixed canonical false accepts.
- SD300C: 118 rescued positives, 11 lost positives, 13 new false accepts, and 15 fixed canonical false accepts.
- Across the 35 v2 false accepts, 12 were near-threshold and 8 were high-confidence.
- Remaining v2 false rejects included 19 near misses, 203 moderate-margin failures, and 525 hard score failures.

The positive failure taxonomy showed recurring causes:

- SD300B: overlap/crop, low inlier count, geometry, low match count, moderate margin, and near-miss cases.
- SD300C: overlap/crop, geometry, low inlier count, moderate margin, low match count, and near-miss cases.

The FRGP focus summary showed that finger position mattered. The worst remaining misses at 1% FAR were concentrated in FRGP 5/10 and some neighboring positions:

- SD300B FRGP 10: 64/89 false rejects, TAR 28.09%.
- SD300B FRGP 5: 59/88 false rejects, TAR 32.95%.
- SD300C FRGP 10: 68/89 false rejects, TAR 23.60%.
- SD300C FRGP 5: 64/88 false rejects, TAR 27.27%.

The overlap/geometry summary showed that hard false rejects had fewer inliers and lower inlier ratios than rescued positives. It also warned that high-confidence false accepts could have match/inlier counts similar to genuine accepts, which limited how safely simple crop/grid variants could be promoted.

The failure taxonomy folder contains `visual_audit_sheets/` and visual audit material. Those sheets are heavy generated artifacts and may contain biometric image data. They are useful for local review but should not be treated as long-term tracked evidence unless a specific case study needs them.

## 9. Grid3 / Hypothesis Testing

Grid3 was investigated because the failure taxonomy suggested overlap and crop mismatch were major causes of false rejects. The hypothesis-test artifacts in `artifacts/reports/benchmark/sift_plain_roll_v2_hypothesis_tests/` tested controlled research-only probes, including crop/overlap, roll multicrop, geometry, and fusion variants. Candidate selection and thresholds used VAL only; TEST was locked reporting.

The strongest hypothesis-test candidate for FRGP 5/10 was `research_only::roll_multicrop_overlap_probe_v1:grid3_max`. The summary reported:

- Best mean FRGP 5/10 TAR delta on VAL: +0.0029.
- TEST mean FRGP 5/10 TAR delta: +0.0113.
- SD300B TEST at 1% FAR: grid3 TAR 50.8% versus v2 TAR 50.2%, with grid3 FAR 0.8%.
- Grid3 rescued 25 hard v2 false rejects at TEST 1% FAR.
- Grid3 introduced 9 high-confidence false accepts.

The validation artifact in `artifacts/reports/benchmark/sift_plain_roll_v2_grid3_validation/grid3_validation_summary.md` showed the mixed result more clearly:

| Dataset | Operating Point | SIFT v2 TAR/FAR | Grid3 TAR/FAR | TAR Delta |
| --- | --- | ---: | ---: | ---: |
| SD300B | 1% FAR | 50.2% / 1.0% | 50.8% / 0.8% | +0.56 pp |
| SD300C | 1% FAR | 44.7% / 0.6% | 48.2% / 1.0% | +3.52 pp |
| SD300B | 0.5% FAR | 43.2% / 0.3% | 46.0% / 0.3% | +2.8 pp |
| SD300C | 0.5% FAR | 42.1% / 0.4% | 42.5% / 0.4% | +0.4 pp |

The same validation showed that grid3 rescued genuine positives but also introduced new false accepts:

- SD300B TEST 1%: 29 grid3-only positives, 25 v2-only positives, 14 new false accepts, and 19 fixed false accepts.
- SD300C TEST 1%: 44 grid3-only positives, 19 v2-only positives, 21 new false accepts, and 12 fixed false accepts.

A guardrail selected on VAL, `v2_weak_support_score_ge_2_5`, produced a locked TEST 1% aggregate result of TAR 48.8% and FAR 0.5%, with 63 rescued positives, 44 lost positives, 17 new false accepts, and 31 fixed false accepts.

Grid3 remained research/diagnostic evidence rather than mainline evidence. The artifacts support it as a promising implementation direction only if the false-accept risk is acceptable and further guardrails are developed.

## 10. SourceAFIS Integration

SourceAFIS was integrated through the `fingerprint_engine` abstraction as `sourceafis_open`. The architecture is documented in `docs/sourceafis_sidecar_contract.md`.

Key integration points:

- Python benchmark code depends on the `fingerprint_engine` provider interface, not directly on the Java SourceAFIS library.
- SourceAFIS runs in a local Java HTTP sidecar under `apps/sourceafis-service`.
- The Maven dependency is pinned to `com.machinezoo.sourceafis:sourceafis:3.18.1`.
- The sidecar exposes `/health`, `/extract-template`, `/verify`, and `/identify`.
- The sidecar is intended to be stateless: no persistence of images or templates, no external network calls, and no logging of biometric bytes.
- SourceAFIS scores are raw SourceAFIS scores, where higher is stronger. The benchmark must calibrate thresholds per dataset/protocol.

The benchmark script `scripts/diagnostics/run_sourceafis_plain_roll_benchmark.py` performs template extraction, verification scoring, VAL threshold calibration, TEST evaluation, latency reporting, failure reporting, manifest writing, and DPI validation for NIST runs.

The SourceAFIS integration was introduced to provide an open AFIS baseline that is closer to production matching than the earlier custom research matchers, while still preserving provider-neutral benchmark infrastructure.

## 11. DPI Discovery and Validation

DPI/PPI handling was the critical SourceAFIS discovery.

The sidecar contract states that DPI is optional in the request and that SourceAFIS defaults to 500 DPI when DPI is not supplied. That default is not valid for the NIST plain-vs-roll SourceAFIS conclusions:

- NIST SD300B requires 1000 DPI/PPI for validated SourceAFIS NIST runs.
- NIST SD300C requires 2000 DPI/PPI for validated SourceAFIS NIST runs.

Early SourceAFIS smoke evidence without correct DPI looked weak. In `artifacts/reports/benchmark/sourceafis_open_plain_roll_smoke1000/`, the SD300B 500-positive/500-negative TEST result at target FAR 1% was:

- TEST TAR 13.60%.
- TEST FAR 0.60%.
- TEST AUC 0.6024.
- TEST EER 43.07%.

That smoke run had no DPI-handling section in the inspected summary/manifest. Based on the sidecar contract, omitting DPI means SourceAFIS used the 500 DPI default. This should be treated as exploratory smoke evidence only, not a validated SourceAFIS conclusion.

After explicitly supplying 1000 DPI for SD300B in `artifacts/reports/benchmark/sourceafis_open_plain_roll_smoke1000_dpi1000/`, the comparable smoke run improved sharply:

- TEST TAR 78.20% at target FAR 1%.
- TEST FAR 1.60%.
- TEST AUC 0.8892.
- TEST EER 17.00%.

That smoke run was superseded by the balanced1400 DPI1000 validated run, but it explains the DPI discovery.

The current benchmark path validates DPI for NIST SourceAFIS runs. `scripts/diagnostics/run_sourceafis_plain_roll_benchmark.py` defines:

- `NIST_DATASET_DPI = {"nist_sd300b": 1000, "nist_sd300c": 2000}`
- `SIDECAR_DEFAULT_DPI = 500`
- DPI strategies: `explicit`, `infer_from_path`, and `default`

For NIST datasets, validated SourceAFIS runs must resolve to the expected DPI unless the run is explicitly using the sidecar default as exploratory behavior. Using default 500 DPI can invalidate conclusions about SourceAFIS on SD300B/SD300C.

## 12. Validated SourceAFIS Results

The balanced1400 DPI runs are the current strongest validated SourceAFIS evidence. They use explicit DPI, VAL calibration, locked TEST evaluation, no extraction/scoring failures, and generated manifests with protocol details.

### SD300B / DPI 1000

Primary folder: `artifacts/reports/benchmark/sourceafis_open_plain_roll_balanced1400_dpi1000/`

Key files:

- `sourceafis_plain_roll_summary.md`
- `sourceafis_plain_roll_metrics.csv`
- `sourceafis_plain_roll_thresholds.csv`
- `sourceafis_plain_roll_latency_summary.csv`
- `sourceafis_plain_roll_failures.csv`
- `sourceafis_plain_roll_manifest.json`
- `sourceafis_vs_sift_v2_comparison.md`
- `sourceafis_vs_sift_v2_comparison.csv`

Protocol summary:

- Dataset: SD300B.
- DPI: explicit 1000.
- Unique image count: 2826.
- DPI counts: 2826 images at 1000 DPI.
- VAL: 700 positives and 700 negatives.
- TEST: 700 positives and 700 negatives.
- Sampling: `balanced_spread`.
- Seed: `13`.
- Total runtime: 1948.30 seconds.
- Extraction failures: 0.
- Scoring failures: 0.
- Extraction timeouts: 0.
- Transport failures: 0.
- Invalid image failures: 0.

TEST operating points from `sourceafis_plain_roll_summary.md`:

| Target FAR | Threshold | TEST TAR | TEST FAR | TEST FRR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01% | 38.909186 | 66.43% | 0.00% | 33.57% | 465 | 235 | 0 | 700 |
| 0.10% | 38.909186 | 66.43% | 0.00% | 33.57% | 465 | 235 | 0 | 700 |
| 0.50% | 17.393218 | 76.00% | 0.43% | 24.00% | 532 | 168 | 3 | 697 |
| 1.00% | 14.723268 | 77.29% | 0.86% | 22.71% | 541 | 159 | 6 | 694 |

AUC/EER:

- TEST AUC: 0.8901897959.
- TEST EER: 0.1700.
- TEST EER threshold: 2.778286.
- VAL AUC: 0.8899.
- VAL EER: 15.86%.

Latency from `sourceafis_plain_roll_latency_summary.csv`:

- TEST template extraction p50: 402.478 ms.
- TEST template extraction p95: 574.820 ms.
- TEST verification p50: 272.660 ms.
- TEST verification p95: 288.061 ms.

The failures CSV contained only the header in the inspected file, consistent with zero recorded failure rows.

### SD300C / DPI 2000

Primary folder: `artifacts/reports/benchmark/sourceafis_open_sd300c_balanced1400_dpi2000/`

Key files:

- `sourceafis_plain_roll_summary.md`
- `sourceafis_plain_roll_metrics.csv`
- `sourceafis_plain_roll_thresholds.csv`
- `sourceafis_plain_roll_latency_summary.csv`
- `sourceafis_plain_roll_failures.csv`
- `sourceafis_plain_roll_manifest.json`
- `sourceafis_vs_sift_v2_comparison.md`
- `sourceafis_vs_sift_v2_comparison.csv`

Protocol summary:

- Dataset: SD300C.
- DPI: explicit 2000.
- Unique image count: 2826.
- DPI counts: 2826 images at 2000 DPI.
- VAL: 700 positives and 700 negatives.
- TEST: 700 positives and 700 negatives.
- Sampling: `balanced_spread`.
- Seed: `13`.
- Total runtime: 2353.65 seconds.
- Extraction failures: 0.
- Scoring failures: 0.
- Extraction timeouts: 0.
- Transport failures: 0.
- Invalid image failures: 0.

TEST operating points from `sourceafis_plain_roll_summary.md`:

| Target FAR | Threshold | TEST TAR | TEST FAR | TEST FRR | TA | FR | FA | TR |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01% | 42.343268 | 64.29% | 0.00% | 35.71% | 450 | 250 | 0 | 700 |
| 0.10% | 42.343268 | 64.29% | 0.00% | 35.71% | 450 | 250 | 0 | 700 |
| 0.50% | 20.060420 | 75.29% | 0.57% | 24.71% | 527 | 173 | 4 | 696 |
| 1.00% | 14.483464 | 78.00% | 1.29% | 22.00% | 546 | 154 | 9 | 691 |

AUC/EER:

- TEST AUC: 0.8814979592.
- TEST EER: 0.1742857143.
- TEST EER threshold: 3.309443.
- VAL AUC: 0.8962.
- VAL EER: 15.14%.

Latency from `sourceafis_plain_roll_latency_summary.csv`:

- TEST template extraction p50: 545.014 ms.
- TEST template extraction p95: 805.020 ms.
- TEST verification p50: 249.619 ms.
- TEST verification p95: 262.657 ms.

The failures CSV contained only the header in the inspected file, consistent with zero recorded failure rows.

## 13. SourceAFIS vs SIFT v2

The comparison files in the validated SourceAFIS folders directly compare SourceAFIS against SIFT v2:

- `artifacts/reports/benchmark/sourceafis_open_plain_roll_balanced1400_dpi1000/sourceafis_vs_sift_v2_comparison.md`
- `artifacts/reports/benchmark/sourceafis_open_plain_roll_balanced1400_dpi1000/sourceafis_vs_sift_v2_comparison.csv`
- `artifacts/reports/benchmark/sourceafis_open_sd300c_balanced1400_dpi2000/sourceafis_vs_sift_v2_comparison.md`
- `artifacts/reports/benchmark/sourceafis_open_sd300c_balanced1400_dpi2000/sourceafis_vs_sift_v2_comparison.csv`

At 1% FAR:

| Dataset | SourceAFIS TAR/FAR | SIFT v2 TAR/FAR | SourceAFIS TAR Delta |
| --- | ---: | ---: | ---: |
| SD300B DPI 1000 | 77.29% / 0.86% | 50.21% / 1.03% | +27.07 pp |
| SD300C DPI 2000 | 78.00% / 1.29% | 44.73% / 0.61% | +33.27 pp |

At 0.5% FAR:

| Dataset | SourceAFIS TAR/FAR | SIFT v2 TAR/FAR | SourceAFIS TAR Delta |
| --- | ---: | ---: | ---: |
| SD300B DPI 1000 | 76.00% / 0.43% | 43.18% / 0.33% | +32.82 pp |
| SD300C DPI 2000 | 75.29% / 0.57% | 42.05% / 0.38% | +33.23 pp |

SourceAFIS substantially outperformed SIFT v2 on the validated DPI-correct runs. SIFT v2 remains useful as a research baseline and as evidence of the custom matcher path. SourceAFIS is stronger as production-candidate/open-AFIS evidence because it combines better accuracy, explicit DPI handling, provider abstraction, and clean threshold-calibrated reporting.

## 14. Artifact Preservation Policy

Preserve as durable evidence:

- This central evidence document: `docs/research/plain_roll_benchmark_evidence_overview.md`.
- Summary markdown files that explain protocol and conclusions.
- Metrics CSV files.
- Thresholds CSV files.
- Manifest JSON files.
- SourceAFIS-vs-SIFT comparison CSV/MD files.
- Selected label/path validation summaries.
- Selected professor protocol summaries/manifests.
- SIFT v2 external validation summaries as custom-baseline evidence.
- Validated SourceAFIS balanced1400 DPI evidence.

Keep local only, archive outside git, or delete later after conclusions are preserved:

- `template_cache/` directories.
- `visual_audit_sheets/` directories.
- `visual_audit_sheets.zip`.
- SourceAFIS smoke runs.
- `candidate_score_cache/`.
- Console logs.
- Copied image audit folders after conclusions are documented.

Generated/heavy/private directories observed during this review include:

- `artifacts/reports/benchmark/sift_plain_roll_v2_failure_taxonomy/visual_audit_sheets/`
- `artifacts/reports/benchmark/sift_plain_roll_v2_grid3_validation/visual_audit_sheets/`
- `artifacts/reports/benchmark/sift_plain_roll_v2_hypothesis_tests/candidate_score_cache/`
- `artifacts/reports/benchmark/sourceafis_open_plain_roll_balanced1400_dpi1000/template_cache/`
- `artifacts/reports/benchmark/sourceafis_open_sd300c_balanced1400_dpi2000/template_cache/`

## 15. Privacy and Sensitivity Notes

Fingerprint images and visual audit sheets may contain biometric data. They should be handled as sensitive data and should not be committed casually.

SourceAFIS `template_cache/` directories contain serialized biometric templates. These templates are derived biometric artifacts and should be treated as sensitive even when raw image files are absent.

CSV, JSON, and markdown artifacts may contain local absolute paths to image files or workspace locations. Those paths should be sanitized before public release if they reveal private filesystem structure, user names, or dataset locations.

Avoid committing:

- Raw fingerprint images.
- Copied visual audit image folders.
- Visual audit sheet bundles unless explicitly needed and approved.
- SourceAFIS template caches.
- Candidate score caches containing large or path-rich generated material.

## 16. Cleanup Plan After This Document

Recommended cleanup order after this document is reviewed:

1. Delete SourceAFIS smoke folders, preserving only the DPI-discovery conclusion in this document and any chosen summary snippets.
2. Delete SourceAFIS `template_cache/` directories from validated runs because they contain serialized biometric templates.
3. Delete SIFT visual audit sheets only after summary conclusions are captured and no additional manual visual review is needed.
4. Keep or archive professor protocol summaries/manifests because they document the advisor-facing methodology.
5. Keep curated SourceAFIS evidence tracked: summary markdown, metrics CSV, thresholds CSV, latency/failure CSV, manifest JSON, and comparison CSV/MD.
6. Keep SIFT v2 external validation summaries as baseline evidence for the custom/research matcher line.

No deletion, movement, staging, committing, benchmark execution, or artifact regeneration was performed as part of creating this overview.

## 17. Remaining Research/Engineering Questions

- Should SourceAFIS results be integrated into the Benchmark Explorer UI?
- Should SourceAFIS become part of `api_runtime` or `benchmark_runtime`, and under what optional-provider/dependency model?
- Should COTS or another AFIS engine be compared against SourceAFIS under the same `fingerprint_engine` protocol?
- Should score paths and local absolute paths be sanitized before any public release?
- Should 1:N identification use SourceAFIS templates in a scalable store rather than recomputing or loading templates ad hoc?
- What is the right long-term retention policy for manifests that include dataset paths but no biometric bytes?
- Should grid3/crop-overlap research be revisited after SourceAFIS is integrated, or should it remain historical SIFT-only diagnostics?
- Should DPI validation become a shared utility for all AFIS providers rather than staying inside the SourceAFIS benchmark script?
