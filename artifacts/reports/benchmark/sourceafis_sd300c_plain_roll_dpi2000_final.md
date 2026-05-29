# SourceAFIS SD300C Plain/Roll DPI 2000 Final Summary

Source folder: `artifacts/reports/benchmark/sourceafis_open_sd300c_balanced1400_dpi2000`

## Scope

- Dataset/protocol: `nist_sd300c` same-finger plain-vs-roll verification pairs, sampled with `balanced_spread`, seed `13`.
- DPI: explicit `image_dpi=2000`; 2,826 unique images reported at 2000 DPI; unknown DPI count `0`.
- Sample counts: VAL `1,400` pairs (`700` positive, `700` negative); TEST `1,400` pairs (`700` positive, `700` negative).
- Provider: `sourceafis_open` / SourceAFIS `3.18.1` through the HTTP sidecar.
- Score semantics: SourceAFIS raw similarity score; higher score means more similar; scores are not normalized.

## Calibration

Thresholds were selected on VAL only and applied unchanged to TEST. The selection rule was the lowest VAL raw-score threshold with VAL FAR less than or equal to the target FAR.

| target FAR | threshold | VAL calibration FAR | VAL false accepts / negatives | enough negatives |
|---:|---:|---:|---:|---|
| 1.00% | 14.483463789540309 | 1.00% | 7/700 | true |
| 0.50% | 20.06041975470194 | 0.43% | 3/700 | true |

## TEST Results

| target FAR | TEST TAR | TEST FAR | TEST FRR | TA | FR | FA | TR |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.00% | 78.00% | 1.29% | 22.00% | 546 | 154 | 9 | 691 |
| 0.50% | 75.29% | 0.57% | 24.71% | 527 | 173 | 4 | 696 |

- TEST AUC: `0.8815`.
- TEST EER: `17.43%` at threshold `3.309443`.
- Scored TEST pairs: `1,400`; unscored TEST pairs: `0`.
- Failure count: `0` template extraction, scoring, timeout, transport, or invalid-image failures.

## Latency

TEST latency summary:

| operation | count | p50 ms | p95 ms | mean ms |
|---|---:|---:|---:|---:|
| template extraction | 1,418 | 545.014 | 805.020 | 548.248 |
| verification | 1,400 | 249.619 | 262.657 | 249.966 |
| template cache lookup | 1,382 | 0.000 | 0.000 | 0.000 |

## Comparison Against SIFT v2

Compared with `artifacts/reports/benchmark/sift_plain_roll_v2_external_validation/per_dataset_metrics.csv`:

| target FAR | SourceAFIS TAR / FAR | SIFT v2 TAR / FAR | SourceAFIS TAR delta | SourceAFIS AUC / EER | SIFT v2 AUC / EER |
|---:|---:|---:|---:|---:|---:|
| 1.00% | 78.00% / 1.29% | 44.73% / 0.61% | +33.27 pp | 0.8815 / 17.43% | 0.7914 / 28.27% |
| 0.50% | 75.29% / 0.57% | 42.05% / 0.38% | +33.23 pp | 0.8815 / 17.43% | 0.7914 / 28.27% |

## Privacy And Reproducibility

- Privacy note: do not retain `template_cache/`. The manifest states the cache contains SourceAFIS template bytes; the final summary does not expose those bytes.
- Manifest: `sourceafis_plain_roll_manifest.json`, created `2026-05-25T21:17:06Z`, git commit `6d4efecdd84da1bdb1cb385fe284039e1423a2ad`, clean at run time.
- Regeneration/source command: no command string is recorded in the manifest; regeneration context is preserved through the manifest schema, runtime settings, provider settings, dataset protocol, target FARs, and output schema.

Final recommendation: keep this as the current validated SourceAFIS SD300C plain/roll evidence at explicit 2000 DPI.
