# SourceAFIS + SIFT + Quality + Deep Fusion v2

This package promotes the successful score-only Fusion v2 prototype into an official, protocol-compliant benchmark pipeline.

## Method

`sourceafis_sift_quality_deep_fusion_v2` combines:

- SourceAFIS raw similarity score and PPI metadata
- SIFT plain-roll v2 score and geometry features (`inliers`, `matches`, `k1`, `k2`)
- Deterministic image-quality features for both images and pairwise deltas
- Deep pair reranker features (`deep_score`, `deep_logit`)
- Dataset/finger/frgp categorical calibration features

The model is a `LogisticRegression(class_weight="balanced")` inside a sklearn pipeline with median imputation, standard scaling for numeric features, and one-hot encoding for categorical features.

## Protocol

The benchmark follows the project protocol:

1. Fit the fusion model on TRAIN only.
2. Select thresholds only from VAL negative scores.
3. Apply frozen thresholds to TEST.
4. Report TAR/FAR/FRR/AUC/EER and confusion counts.
5. Do not use TEST for fitting, threshold selection, hyperparameter tuning, or checkpoint selection.

## Main script

```powershell
python pipelines\benchmark\train_run_sourceafis_sift_quality_deep_fusion_v2.py `
  --repo-root C:\fingerprint-research `
  --outdir C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_sift_quality_deep_fusion_v2_full_pairs `
  --datasets nist_sd300b,nist_sd300c `
  --splits val,test `
  --target-fars 0.005,0.01 `
  --run-ablation `
  --save-training-table
```

For a fast debug run without image-quality extraction:

```powershell
python pipelines\benchmark\train_run_sourceafis_sift_quality_deep_fusion_v2.py `
  --repo-root C:\fingerprint-research `
  --outdir C:\fingerprint-research\artifacts\reports\benchmark\sourceafis_sift_quality_deep_fusion_v2_debug_no_quality `
  --variants sourceafis_sift_deep_logit `
  --no-quality
```

The `--no-quality` flag is for debugging only; it is not the official Fusion v2 result.

## Outputs

```text
artifacts/reports/benchmark/sourceafis_sift_quality_deep_fusion_v2_full_pairs/
  plain_roll_final_metrics.csv
  plain_roll_final_thresholds.csv
  plain_roll_final_statistical_comparison.csv
  plain_roll_final_summary.md
  plain_roll_final_manifest.json
  ablation_metrics.csv
  ablation_summary.md
  model/
  scores/
```

## Failure taxonomy

Current canonical diagnostics are rebuilt from the statistical Fusion v2 score files:

```powershell
python scripts\diagnostics\build_current_fusion_v2_diagnostics.py `
  --repo-root C:\fingerprint-research
```

This writes:

```text
artifacts/reports/diagnostics/sourceafis_sift_quality_deep_fusion_v2_current_failure_taxonomy/
artifacts/reports/diagnostics/true_accept_failures_across_methods_current/
```

The older v1-vs-v2 taxonomy output is quarantined under `artifacts/reports/diagnostics/legacy_stale_20260629/`.
