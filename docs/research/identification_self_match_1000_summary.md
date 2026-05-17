# 1:N Identification Self-Match Repeatability Summary

## Experiment Goal

This experiment measured 1:N identification repeatability under a self-match protocol. A sampled fingerprint image was enrolled as its own experiment identity and then queried with the same image. The expected outcome for each query was that the enrolled copy of the same image should rank first.

This is an identification sanity/repeatability experiment. It is not a 1:1 verification benchmark and does not estimate FAR, FRR, EER, or AUC.

## Run Configuration

- Dataset: `nist_sd300b`
- Selected cases: 1000
- Enrolled identities/images: 1000
- Query cases: 1000 per method
- Methods compared: `classic_gftt_orb`, `minutiae`, `harris`, `sift`, `dl`
- Shortlist size: 25
- Rerank policy: `top1`
- Output directory: `artifacts/reports/identification/self_match_repeatability_1000_top1`
- Table prefix: `self_match_exp_20260515_200150_`

## Database Isolation

The run used a timestamped table prefix, `self_match_exp_20260515_200150_`, so experiment tables were isolated from production/runtime tables and from earlier experiment outputs. The biometric data tables and identity mapping table were namespaced by this prefix:

- `biometric_db.self_match_exp_20260515_200150_raw_fingerprints`
- `biometric_db.self_match_exp_20260515_200150_feature_vectors`
- `biometric_db.self_match_exp_20260515_200150_method_retrieval_vectors`
- `biometric_db.self_match_exp_20260515_200150_person_directory`
- `identity_db.self_match_exp_20260515_200150_identity_map`

This clean prefix gives the experiment a separate table set while preserving the dual-database identity/profile split. Because the prefix was unique for this run, no existing benchmark or runtime tables had to be reused.

## Rerank Policy

`rerank_policy=top1` means the vector retrieval stage first creates a shortlist of up to 25 candidates, but only the retrieval top-1 candidate is passed through the method's pairwise reranker. This keeps the run advisor-facing and tractable while still checking whether the final returned Top-1 identity remains the self-match. It is intentionally much cheaper than full-shortlist reranking.

## Results

| Method | Selected | Enrolled | Queries | Enroll Errors | Query Errors | Retrieval Top-1 Self-Match | Final Top-1 Self-Match | Self In Shortlist | Mean Final Rank | Mean Total Query ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `classic_gftt_orb` | 1000 | 1000 | 1000 | 0 | 0 | 1000 / 100.00% | 1000 / 100.00% | 1000 / 100.00% | 1.00 | 1510.4 |
| `minutiae` | 1000 | 1000 | 1000 | 0 | 0 | 1000 / 100.00% | 1000 / 100.00% | 1000 / 100.00% | 1.00 | 4640.4 |
| `harris` | 1000 | 1000 | 1000 | 0 | 0 | 1000 / 100.00% | 1000 / 100.00% | 1000 / 100.00% | 1.00 | 3395.3 |
| `sift` | 1000 | 1000 | 1000 | 0 | 0 | 1000 / 100.00% | 1000 / 100.00% | 1000 / 100.00% | 1.00 | 1524.5 |
| `dl` | 1000 | 1000 | 1000 | 0 | 0 | 1000 / 100.00% | 1000 / 100.00% | 1000 / 100.00% | 1.00 | 1327.9 |

## Conclusion

All 1000 self-queries were identified as Top-1 by all five methods. The CSV shows `final_top1_self_match_count=1000` and `final_top1_self_match_rate=1` for `classic_gftt_orb`, `minutiae`, `harris`, `sift`, and `dl`.

The same result also holds at retrieval time: each method had `retrieval_top1_self_match_count=1000`, `retrieval_top1_self_match_rate=1`, and `self_in_shortlist_count=1000`.

## Limitations

This protocol only tests whether each method can re-identify an enrolled image when the exact same image is queried against a 1000-entry gallery. It does not test genuine-vs-impostor decision behavior, cross-impression robustness, or calibrated operating thresholds.

Therefore, these results must not be reported as FAR, FRR, EER, AUC, or verification accuracy. They establish 1:N self-match repeatability, not 1:1 verification performance.

## Next Experiment

The next required experiment is a 1:1 verification benchmark for the professor-facing methods:

- `classic_v2`
- `minutiae`
- `harris`
- `sift`
- `dl_quick`

That benchmark should run on the canonical pair files and report FAR, FRR, EER, and AUC from genuine and impostor pair scores.
