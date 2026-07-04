from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.fpbench.universal.deep_fusion_v2 import (
    GROUP_WEIGHTED_METHOD,
    parse_group_weights,
    run_variants,
)


def _write_pair_bundle(root: Path, dataset: str, split: str, base: pd.DataFrame) -> None:
    bundle_dir = root / "data/manifests" / dataset
    bundle_dir.mkdir(parents=True, exist_ok=True)
    pair_columns = [
        "dataset",
        "split",
        "pair_id",
        "label",
        "subject_a",
        "subject_b",
        "frgp",
        "path_a",
        "path_b",
    ]
    base[pair_columns].to_csv(bundle_dir / f"pairs_{split}.csv", index=False)


def _write_scores(root: Path, dataset: str, split: str, labels: list[int]) -> None:
    n = len(labels)
    pair_ids = [str(i) for i in range(n)]
    base = pd.DataFrame({
        "dataset": [dataset] * n,
        "split": [split] * n,
        "pair_id": pair_ids,
        "label": labels,
        "path_a": [f"a_{i}.png" for i in range(n)],
        "path_b": [f"b_{i}.png" for i in range(n)],
        "subject_a": [str(i) for i in range(n)],
        "subject_b": [str(i if y else i + 1000) for i, y in enumerate(labels)],
        "finger_position": ["7"] * n,
        "frgp": ["7"] * n,
    })
    _write_pair_bundle(root, dataset, split, base)

    source = base.copy()
    source["raw_score"] = [0.05 if y == 0 else 0.95 for y in labels]
    source["dpi_a"] = 1000
    source["dpi_b"] = 1000
    sift = base.copy()
    sift["score"] = [0.1 if y == 0 else 0.9 for y in labels]
    sift["inliers"] = [2 if y == 0 else 20 for y in labels]
    sift["matches"] = [10 if y == 0 else 30 for y in labels]
    sift["k1"] = 100
    sift["k2"] = 100
    deep = base.copy()
    deep["score"] = [0.05 if y == 0 else 0.95 for y in labels]
    deep["logit"] = [-3 if y == 0 else 3 for y in labels]

    if split == "train":
        train_dir = root / "artifacts/reports/benchmark/plain_roll_train_scores_v2_anatomical_full_pairs/scores"
        train_dir.mkdir(parents=True, exist_ok=True)
        source.to_csv(train_dir / f"scores_{dataset}_sourceafis_open_train.csv", index=False)
        sift.to_csv(train_dir / f"scores_{dataset}_sift_plain_roll_v2_train.csv", index=False)
        sift.to_csv(train_dir / f"scores_{dataset}_sift_train.csv", index=False)
        deep.to_csv(train_dir / f"scores_{dataset}_deep_pair_reranker_fast_ddp_train.csv", index=False)
    else:
        src_dir = root / "artifacts/reports/benchmark/plain_roll_final_sourceafis_v2_anatomical_full_pairs/scores"
        sift_dir = root / "artifacts/reports/benchmark/plain_roll_final_baselines_v2_anatomical_full_pairs"
        deep_dir = root / "artifacts/reports/benchmark/deep_pair_reranker_fast_ddp_full_pairs/scores"
        src_dir.mkdir(parents=True, exist_ok=True)
        sift_dir.mkdir(parents=True, exist_ok=True)
        deep_dir.mkdir(parents=True, exist_ok=True)
        source.to_csv(src_dir / f"scores_{dataset}_sourceafis_open_{split}.csv", index=False)
        sift.to_csv(sift_dir / f"scores_{dataset}_sift_plain_roll_v2_{split}.csv", index=False)
        deep.to_csv(deep_dir / f"scores_{dataset}_deep_pair_reranker_fast_ddp_{split}.csv", index=False)


def test_parse_group_weights_normalizes_percent_values():
    weights = parse_group_weights("sourceafis=45,sift=15,deep=30,quality=10")
    assert round(sum(weights.values()), 12) == 1.0
    assert weights["sourceafis"] == 0.45
    assert weights["sift"] == 0.15
    assert weights["deep"] == 0.30
    assert weights["quality"] == 0.10


def test_group_weighted_variant_writes_scores_and_manifest(tmp_path: Path):
    root = tmp_path / "repo"
    labels_train = [0, 0, 0, 0, 1, 1, 1, 1]
    labels_eval = [0, 0, 0, 1, 1, 1]
    _write_scores(root, "nist_sd300b", "train", labels_train)
    _write_scores(root, "nist_sd300b", "val", labels_eval)
    _write_scores(root, "nist_sd300b", "test", labels_eval)
    outdir = root / "artifacts/reports/benchmark/sourceafis_sift_quality_deep_group_weighted_fusion_v2_full_pairs"

    results = run_variants(
        repo_root=root,
        outdir=outdir,
        datasets=("nist_sd300b",),
        splits=("val", "test"),
        target_fars=(0.01,),
        variants=(GROUP_WEIGHTED_METHOD,),
        include_quality_override=False,
        group_weights=parse_group_weights("sourceafis=45,sift=15,deep=30,quality=10"),
    )

    assert not results["metrics"].empty
    assert set(results["metrics"]["method"]) == {GROUP_WEIGHTED_METHOD}
    assert (outdir / "model/group_weights.json").exists()
    assert (outdir / "model" / GROUP_WEIGHTED_METHOD / "feature_manifest.json").exists()
    assert (outdir / "scores/ablation" / f"scores_nist_sd300b_{GROUP_WEIGHTED_METHOD}_test.csv").exists()
