# Deep Fusion v2 statistical replay proof

This diagnostic proves the following claim:

> If a manual weighting exactly reproduces the trained statistical fusion model, including preprocessing, imputation, scaling, every learned feature coefficient, and the intercept, then the manually computed score is identical to `model.predict_proba`.

It also shows why a single manual weight per method group is not generally equivalent to the trained statistical model: the learned logistic-regression coefficients vary by individual transformed feature and may have different signs inside the same group.

## Script

```text
scripts/diagnostics/prove_deep_fusion_v2_statistical_weight_replay.py
```

## Recommended command for the official Fusion v2 model

```powershell
cd C:\fingerprint-research

python scripts\diagnostics\prove_deep_fusion_v2_statistical_weight_replay.py `
  --repo-root C:\fingerprint-research `
  --method sourceafis_sift_quality_deep_fusion_v2 `
  --benchmark-dir C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_sift_quality_deep_fusion_v2_full_pairs `
  --model-dir C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_sift_quality_deep_fusion_v2_full_pairs\model\sourceafis_sift_quality_deep_fusion_v2 `
  --outdir C:\fingerprint-research\artifacts\reports\diagnostics\deep_fusion_v2_statistical_replay `
  --datasets nist_sd300b,nist_sd300c `
  --splits val,test `
  --target-fars 0.005,0.01
```

## Outputs

The script writes:

```text
artifacts/reports/diagnostics/deep_fusion_v2_statistical_replay/
  replay_equivalence_summary.md
  replay_score_equivalence.csv
  replay_decision_equivalence.csv
  replay_score_differences.csv
  learned_feature_coefficients.csv
  coefficient_group_summary.csv
  replay_manifest.json
```

## Expected evidence

A successful replay should show approximately:

```text
max_abs_diff_manual_vs_sklearn <= 1e-12
decision_mismatches_manual_vs_sklearn = 0
```

This proves that manually replaying the complete statistical weighting is equivalent to the trained model.

The coefficient files then support the second conclusion: a group-level weighting scheme is more constrained than the trained model, because the statistical model learns separate coefficients for individual features after preprocessing.
