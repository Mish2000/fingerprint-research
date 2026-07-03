# Phase 2B — `deep_pair_reranker_v1`

This implementation adds a first deep-learning prototype for the plain-vs-roll fingerprint research track.

## Scope

The model is intentionally a measurable prototype, not yet the final cross-attention matcher:

```text
image_a -> shared encoder -> emb_a
image_b -> shared encoder -> emb_b
[emb_a, emb_b, abs_diff, product] -> MLP -> logit -> sigmoid -> deep_pair_score
```

The score is compatible with the existing plain/roll benchmark stack:

- threshold calibration from VAL negative scores only;
- TEST is used only for final scoring/evaluation;
- score CSVs include `label` and `score` plus traceability columns;
- run metadata follows `v2_benchmark_run_meta` and `v2_benchmark_eval_config`.

## Added files

```text
src/fpbench/deep/
  __init__.py
  image_io.py
  transforms.py
  pair_dataset.py
  samplers.py
  models.py
  losses.py
  train_utils.py
  inference.py
  metrics.py

pipelines/training/train_deep_pair_reranker.py
pipelines/benchmark/run_deep_pair_reranker_benchmark.py

tests/test_deep_pair_dataset.py
tests/test_deep_pair_transforms.py
tests/test_deep_pair_reranker_smoke.py
tests/test_deep_pair_reranker_score_schema.py
tests/test_deep_pair_reranker_run_meta_schema.py
```

## Local smoke tests

```powershell
python -m pytest `
  tests/test_deep_pair_dataset.py `
  tests/test_deep_pair_transforms.py `
  tests/test_deep_pair_reranker_smoke.py `
  tests/test_deep_pair_reranker_score_schema.py `
  tests/test_deep_pair_reranker_run_meta_schema.py -q
```

## Debug training run

```powershell
python pipelines/training/train_deep_pair_reranker.py `
  --repo-root C:\fingerprint-research `
  --epochs 1 `
  --batch-size 8 `
  --input-size 128 `
  --embedding-dim 64 `
  --hidden-dim 128 `
  --limit-train-pairs-per-dataset 64 `
  --limit-val-pairs-per-dataset 32 `
  --debug
```

## Full training run — local GPU

```powershell
python pipelines/training/train_deep_pair_reranker.py `
  --repo-root C:\fingerprint-research `
  --epochs 8 `
  --batch-size 32 `
  --input-size 384 `
  --channels 1 `
  --backbone small_cnn `
  --embedding-dim 256 `
  --hidden-dim 512 `
  --device cuda `
  --amp
```

## Full training run — Kaggle/Linux

Upload/copy the repo and raw data so that the relative suffix under `data/raw/...` is preserved. Then use `--data-root` to remap the Windows paths stored in the pair CSVs.

```bash
python pipelines/training/train_deep_pair_reranker.py \
  --repo-root /kaggle/working/fingerprint-research \
  --data-root /kaggle/input/fingerprint-research-data \
  --epochs 8 \
  --batch-size 32 \
  --input-size 384 \
  --channels 1 \
  --backbone small_cnn \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --device cuda \
  --amp \
  --num-workers 2
```

## Selected-pairs benchmark

```powershell
python pipelines/benchmark/run_deep_pair_reranker_benchmark.py `
  --repo-root C:\fingerprint-research `
  --checkpoint artifacts/checkpoints/deep_pair_reranker_v1/best.pt `
  --outdir artifacts/reports/benchmark/deep_pair_reranker_v1 `
  --device cuda `
  --amp
```

## Full-pairs benchmark

```powershell
python pipelines/benchmark/run_deep_pair_reranker_benchmark.py `
  --repo-root C:\fingerprint-research `
  --checkpoint artifacts/checkpoints/deep_pair_reranker_v1/best.pt `
  --outdir artifacts/reports/benchmark/deep_pair_reranker_v1_full_pairs `
  --full-pairs `
  --device cuda `
  --amp
```

## Expected outputs

Training:

```text
artifacts/checkpoints/deep_pair_reranker_v1/
  best.pt
  last.pt
  config.json
  training_manifest.json
```

Benchmark:

```text
artifacts/reports/benchmark/deep_pair_reranker_v1/
  plain_roll_final_metrics.csv
  plain_roll_final_thresholds.csv
  plain_roll_final_summary.md
  plain_roll_final_failures.csv
  scores/scores_<dataset>_deep_pair_reranker_v1_<split>.csv
  run_meta/run_<dataset>_deep_pair_reranker_v1_<split>.meta.json
```

## Next step after this prototype

If `deep_pair_reranker_v1` shows signal, add `sourceafis_sift_quality_deep_fusion_v2` using:

```text
sourceafis_score
sift_plain_roll_v2_score
sift geometry features
quality features
deep_pair_score
```

Keep the fusion model simple and explainable first, e.g. logistic regression or a small LightGBM model.
