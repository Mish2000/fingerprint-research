# SourceAFIS + SIFT + Quality Fusion v1

## Goal

`sourceafis_sift_quality_fusion_v1` is a trainable plain-vs-roll benchmark method that combines SourceAFIS scores, SIFT Plain/Roll v2 scores, optional regular SIFT scores when present, lightweight image quality features, and dataset metadata.

## Inputs

- Full train pairs from `data/manifests/nist_sd300b/pairs_train.csv`
- Full train pairs from `data/manifests/nist_sd300c/pairs_train.csv`
- Train score CSVs in `artifacts/reports/benchmark/plain_roll_train_scores_v1/`
- Full canonical val/test pairs from `data/manifests/nist_sd300b/pairs_val.csv` and `data/manifests/nist_sd300b/pairs_test.csv`
- Full canonical val/test pairs from `data/manifests/nist_sd300c/pairs_val.csv` and `data/manifests/nist_sd300c/pairs_test.csv`

Expected train score files:

- `scores_nist_sd300b_sourceafis_open_train.csv`
- `scores_nist_sd300b_sift_plain_roll_v2_train.csv`
- `scores_nist_sd300c_sourceafis_open_train.csv`
- `scores_nist_sd300c_sift_plain_roll_v2_train.csv`

## Protocol

The fusion model is fit on train rows only. Validation rows are used only for threshold calibration/model selection, and test rows are used only for final evaluation.

The benchmark runner freezes the trained model, computes fusion probabilities for val/test pairs, selects operating thresholds from validation negative fusion scores only, then applies the frozen thresholds unchanged to test.

The default benchmark uses the full canonical SD300 anatomical pair CSVs. Run-local `selected_pairs` files are materialized as output/staging only and are fingerprinted in the benchmark manifest.

## Output Artifacts

Default output directory:

`artifacts/reports/benchmark/plain_roll_final_fusion_v1/`

Primary files:

- `plain_roll_final_metrics.csv`
- `plain_roll_final_thresholds.csv`
- `plain_roll_final_threshold_sweep.csv`
- `plain_roll_final_tar_far_distribution.csv`
- `plain_roll_final_summary.md`
- `plain_roll_final_manifest.json`
- `plain_roll_final_failures.csv`
- `scores/scores_<dataset>_sourceafis_sift_quality_fusion_v1_<split>.csv`
- `selected_pairs/pairs_<dataset>_<split>.csv`
- `model/fusion_model.joblib`
- `model/feature_schema.json`
- `model/training_manifest.json`

Phase 1.5 ablation output directory:

`artifacts/reports/benchmark/plain_roll_fusion_ablation_v1/`

Primary files:

- `ablation_metrics.csv`
- `ablation_thresholds.csv`
- `ablation_summary.md`
- `ablation_manifest.json`
- `model/<variant>/fusion_model.joblib`

Full-pairs robustness output directory:

`artifacts/reports/benchmark/plain_roll_final_fusion_v1_full_pairs/`

Additional Phase 1.5 full-pairs file:

- `plain_roll_final_statistical_comparison.csv`
- `plain_roll_final_statistical_comparison.md`

## PowerShell Commands

Generate missing train scores when needed:

```powershell
python pipelines/benchmark/train_sourceafis_sift_quality_fusion.py --generate_missing_scores --train_score_dir "artifacts/reports/benchmark/plain_roll_train_scores_v1"
```

Train the fusion model from existing train scores:

```powershell
python pipelines/benchmark/train_sourceafis_sift_quality_fusion.py --train_score_dir "artifacts/reports/benchmark/plain_roll_train_scores_v1" --outdir "artifacts/reports/benchmark/plain_roll_final_fusion_v1"
```

Run the full canonical-pairs benchmark:

```powershell
python pipelines/benchmark/run_sourceafis_sift_quality_fusion_benchmark.py --pair_scope full --model_dir "artifacts/reports/benchmark/plain_roll_final_fusion_v1/model" --outdir "artifacts/reports/benchmark/plain_roll_final_fusion_v1"
```

Run the Phase 1.5 full canonical-pairs ablation report:

```powershell
python pipelines/benchmark/run_sourceafis_sift_quality_fusion_phase15.py --mode ablation --save_training_features
```

Generate full val/test SourceAFIS scores for the robustness benchmark:

```powershell
python scripts/diagnostics/run_sourceafis_plain_roll_benchmark.py --datasets "nist_sd300b,nist_sd300c" --splits "val,test" --outdir "artifacts/reports/benchmark/plain_roll_full_scores_v1/sourceafis" --limit_per_split 0
```

Generate full val/test SIFT Plain/Roll v2 scores for the robustness benchmark:

```powershell
foreach ($dataset in @("nist_sd300b", "nist_sd300c")) {
  foreach ($split in @("val", "test")) {
    python pipelines/benchmark/evaluate.py --method sift_plain_roll_v2 --dataset $dataset --split $split --data_dir "data/manifests/$dataset" --limit 0 --out_scores "artifacts/reports/benchmark/plain_roll_full_scores_v1/sift/scores_${dataset}_sift_plain_roll_v2_${split}.csv" --out_roc "artifacts/reports/benchmark/plain_roll_full_scores_v1/sift/roc_${dataset}_sift_plain_roll_v2_${split}.png" --out_run_meta "artifacts/reports/benchmark/plain_roll_full_scores_v1/sift/run_${dataset}_sift_plain_roll_v2_${split}.meta.json" --summary_csv "artifacts/reports/benchmark/plain_roll_full_scores_v1/sift/evaluate_results_summary.csv"
  }
}
```

Run the full val/test robustness benchmark:

```powershell
python pipelines/benchmark/run_sourceafis_sift_quality_fusion_benchmark.py --pair_scope full --model_dir "artifacts/reports/benchmark/plain_roll_final_fusion_v1/model" --sourceafis_score_dir "artifacts/reports/benchmark/plain_roll_full_scores_v1/sourceafis" --sift_plain_roll_score_dir "artifacts/reports/benchmark/plain_roll_full_scores_v1/sift" --outdir "artifacts/reports/benchmark/plain_roll_final_fusion_v1_full_pairs"
```

Or run full-pairs validation through the Phase 1.5 wrapper, which can generate missing full SourceAFIS/SIFT scores and attaches pair traceability to full SIFT score CSVs:

```powershell
$env:SOURCEAFIS_ENABLED = "true"
$env:SOURCEAFIS_SERVICE_URL = "http://127.0.0.1:8765"
python pipelines/benchmark/run_sourceafis_sift_quality_fusion_phase15.py --mode full --generate_missing_scores --sourceafis_template_cache_dir "artifacts/reports/benchmark/plain_roll_train_scores_v1/sourceafis_template_cache"
```

The fusion score CSVs now mark latency as run-level average only: `pair_total_ms` is left empty, `run_level_avg_pair_ms` carries the batch average, and `pair_total_ms_semantics` states that true per-pair fusion latency was not measured.

Run relevant tests:

```powershell
pytest tests/test_universal_quality_features.py tests/test_sourceafis_sift_quality_fusion_features.py tests/test_sourceafis_sift_quality_fusion_no_test_leakage.py tests/test_sourceafis_sift_quality_fusion_benchmark.py
```
