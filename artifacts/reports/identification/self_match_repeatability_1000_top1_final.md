# Self-Match Repeatability 1000 Top-1 Final Summary

Source folder: `artifacts/reports/identification/self_match_repeatability_1000_top1`

## Scope

- Purpose: 1:N self-match repeatability experiment, not 1:1 verification.
- Dataset: `nist_sd300b`.
- Sample count: `1,000` selected images, `1,000` enrolled records, `1,000` queries per method.
- Seed: `1337`.
- Methods: `classic_gftt_orb`, `minutiae`, `harris`, `sift`, `dl`.
- Shortlist size: `25`.
- Rerank policy: `top1`, meaning only the retrieval top-1 candidate is pairwise-reranked; `24` candidates are skipped on average.
- Errors: `0` enrollment errors and `0` query errors for every method.

## Top-1 Results

| method | retrieval top-1 self-match | final top-1 self-match | self in shortlist | mean retrieval rank | mean final rank | exact vector score |
|---|---:|---:|---:|---:|---:|---:|
| classic_gftt_orb | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 100.00% |
| minutiae | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 100.00% |
| harris | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 100.00% |
| sift | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 100.00% |
| dl | 100.00% | 100.00% | 100.00% | 1.00 | 1.00 | 100.00% |

The legacy `top1_is_self` field and the explicit `retrieval_top1_is_self` / `final_top1_is_self` fields all agree at `1,000/1,000` for every method.

## Latency

| method | mean total query ms | p95 total query ms | mean probe embed ms | mean shortlist scan ms | mean rerank ms |
|---|---:|---:|---:|---:|---:|
| classic_gftt_orb | 1,510.4 | 1,652.5 | 31.6 | 90.9 | 93.0 |
| minutiae | 4,640.4 | 5,524.4 | 937.5 | 110.8 | 2,415.2 |
| harris | 3,395.3 | 3,801.8 | 682.4 | 107.5 | 1,459.8 |
| sift | 1,524.5 | 1,679.9 | 92.7 | 87.1 | 208.9 |
| dl | 1,327.9 | 1,497.2 | 30.5 | 91.5 | 89.6 |

## DB Layout

The run used PostgreSQL with dual biometric and identity databases. Credentials are intentionally omitted here.

- Layout version: `v4_dual_database_identity_profile_split`.
- Table prefix: `self_match_exp_20260515_200150_`.
- Vector storage mode: `method_generic_pgvector_table_with_legacy_compat`.
- Method-generic vectors supported: `true`.
- Key tables: `method_retrieval_vectors`, `feature_vectors`, `raw_fingerprints`, `person_directory`, and `identity_map`.

Final recommendation: keep this as advisor-facing identification repeatability evidence.
