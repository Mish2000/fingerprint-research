# SourceAFIS SD300B Plain/Roll DPI 1000 Final Summary

Source folder: `artifacts/reports/benchmark/sourceafis_open_plain_roll_balanced1400_dpi1000`

## Scope

- Dataset/protocol: `nist_sd300b` same-finger plain-vs-roll verification pairs, sampled with `balanced_spread`, seed `13`.
- DPI: explicit `image_dpi=1000`; 2,826 unique images reported at 1000 DPI; unknown DPI count `0`.
- Sample counts: VAL `1,400` pairs (`700` positive, `700` negative); TEST `1,400` pairs (`700` positive, `700` negative).
- Provider: `sourceafis_open` / SourceAFIS `3.18.1` through the HTTP sidecar.
- Score semantics: SourceAFIS raw similarity score; higher score means more similar; scores are not normalized.

## Calibration

Thresholds were selected on VAL only and applied unchanged to TEST. The selection rule was the lowest VAL raw-score threshold with VAL FAR less than or equal to the target FAR.

| target FAR | threshold | VAL calibration FAR | VAL false accepts / negatives | enough negatives |
|---:|---:|---:|---:|---|
| 1.00% | 14.72326764987426 | 1.00% | 7/700 | true |
| 0.50% | 17.393218350729448 | 0.43% | 3/700 | true |

## TEST Results

| target FAR | TEST TAR | TEST FAR | TEST FRR | TA | FR | FA | TR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00% | 77.29% | 0.86% | 22.71% | 541 | 159 | 6 | 694 |
| 0.50% | 76.00% | 0.43% | 24.00% | 532 | 168 | 3 | 697 |

- TEST AUC: `0.8902`.
- TEST EER: `17.00%` at threshold `2.778286`.
- Scored TEST pairs: `1,400`; unscored TEST pairs: `0`.
- Failure count: `0` template extraction, scoring, timeout, transport, or invalid-image failures.

## Latency

TEST latency summary:

| operation | count | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|
| template extraction | 1,418 | 402.478 | 574.820 | 417.430 |
| verification | 1,400 | 272.660 | 288.061 | 272.902 |
| template cache lookup | 1,382 | 0.000 | 0.000 | 0.000 |

## Comparison Against SIFT v2

Compared with `artifacts/reports/benchmark/sift_plain_roll_v2_external_validation/per_dataset_metrics.csv`:

| target FAR | SourceAFIS TAR / FAR | SIFT v2 TAR / FAR | SourceAFIS TAR delta | SourceAFIS AUC / EER | SIFT v2 AUC / EER |
|---:|---:|---:|---:|---:|---:|
| 1.00% | 77.29% / 0.86% | 50.21% / 1.03% | +27.07 pp | 0.8902 / 17.00% | 0.7963 / 29.32% |
| 0.50% | 76.00% / 0.43% | 43.18% / 0.33% | +32.82 pp | 0.8902 / 17.00% | 0.7963 / 29.32% |

## Privacy And Reproducibility

- Privacy note: do not retain `template_cache/`. The manifest states the cache contains SourceAFIS template bytes; the final summary does not expose those bytes.
- Manifest: `sourceafis_plain_roll_manifest.json`, created `2026-05-25T20:16:41Z`, git commit `6d4efecdd84da1bdb1cb385fe284039e1423a2ad`, clean at run time.
- Regeneration/source command: no command string is recorded in the manifest; regeneration context is preserved through the manifest schema, runtime settings, provider settings, dataset protocol, target FARs, and output schema.

Final recommendation: keep this as the current validated SourceAFIS SD300B plain/roll evidence at explicit 1000 DPI.
