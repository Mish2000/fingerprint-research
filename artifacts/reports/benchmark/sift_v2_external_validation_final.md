# SIFT v2 External Validation Final Summary

Source folder: `artifacts/reports/benchmark/sift_plain_roll_v2_external_validation`

## Scope

- Datasets: `nist_sd300b` and `nist_sd300c`.
- Protocol: full compatible NIST plain-vs-roll VAL/TEST pairs; validation-only run with no tuning.
- Pair counts per dataset: VAL `2,812` pairs (`703` positive, `2,109` negative); TEST `2,844` pairs (`711` positive, `2,133` negative).
- Methods compared: canonical `sift` with `current_score`, canonical `sift` with `inliers`, and custom `sift_plain_roll_v2` with `official_score`.
- Manifest status: `research_only_not_canonical_not_default_not_showcase`.

## TEST Operating Points

Primary comparison against canonical SIFT uses the `current_score` variant.

| dataset | target FAR | canonical SIFT TAR / FAR | SIFT v2 TAR / FAR | TAR delta |
|---|---:|---:|---:|---:|
| SD300B | 1.00% | 28.27% / 0.33% | 50.21% / 1.03% | +21.94 pp |
| SD300B | 0.50% | 21.80% / 0.19% | 43.18% / 0.33% | +21.38 pp |
| SD300C | 1.00% | 29.68% / 0.70% | 44.73% / 0.61% | +15.05 pp |
| SD300C | 0.50% | 22.22% / 0.23% | 42.05% / 0.38% | +19.83 pp |

Canonical SIFT `inliers` also remains in the source metrics. At 1.00% FAR, it reached `31.65%` TAR on SD300B TEST and `32.49%` TAR on SD300C TEST, still below SIFT v2.

## AUC And EER

| dataset | method | variant | TEST AUC | TEST EER |
|---|---|---|---:|---:|
| SD300B | sift | current_score | 0.8039 | 27.87% |
| SD300B | sift | inliers | 0.8046 | 27.57% |
| SD300B | sift_plain_roll_v2 | official_score | 0.7963 | 29.32% |
| SD300C | sift | current_score | 0.7976 | 28.55% |
| SD300C | sift | inliers | 0.7977 | 28.57% |
| SD300C | sift_plain_roll_v2 | official_score | 0.7914 | 28.27% |

The operating-point TAR improvement is real on both datasets, but the AUC/EER profile shows this is not a production-candidate verifier by itself.

## Use And Recommendation

SIFT v2 remains useful as a custom research baseline because it demonstrates a repeatable plain-vs-roll improvement over canonical SIFT at low-FAR operating points on both SD300B and SD300C. It is especially useful for explaining the limits of hand-built classical matching and for comparing future custom matchers against a documented validation run.

It is superseded by SourceAFIS for production-candidate evidence: SourceAFIS reaches about `77-78%` TEST TAR at the 1.00% FAR target on the balanced SourceAFIS SD300B/SD300C runs, versus `45-50%` for SIFT v2 on the full external validation protocol.

Source commands: `run_manifest.json` records eight successful `evaluate.py` invocations, covering canonical SIFT and SIFT v2 for VAL/TEST on SD300B and SD300C.

Final recommendation: keep this as baseline evidence only, not as canonical/default/showcase or production-candidate evidence.
