from __future__ import annotations

import pandas as pd
import pytest

from src.fpbench.universal.deep_fusion_v2 import (
    DeepFusionV2Error,
    assert_matches_pair_bundle,
    merge_tables,
    prepare_deep,
    prepare_sift,
    prepare_sourceafis,
)


def _base_tables():
    source_raw = pd.DataFrame({
        "dataset": ["nist_sd300b", "nist_sd300b"],
        "split": ["train", "train"],
        "pair_id": ["p0", "p1"],
        "label": [0, 1],
        "raw_score": [1.0, 10.0],
        "dpi_a": [1000, 1000],
        "dpi_b": [1000, 1000],
    })
    sift_raw = pd.DataFrame({
        "dataset": ["nist_sd300b", "nist_sd300b"],
        "split": ["train", "train"],
        "pair_id": ["p0", "p1"],
        "label": [0, 1],
        "path_a": ["a0.png", "a1.png"],
        "path_b": ["b0.png", "b1.png"],
        "subject_a": ["1", "2"],
        "subject_b": ["3", "2"],
        "finger_position": ["7", "7"],
        "frgp": ["7", "7"],
        "score": [2.0, 9.0],
        "inliers": [3, 20],
        "matches": [10, 30],
        "k1": [100, 100],
        "k2": [100, 100],
    })
    deep_raw = pd.DataFrame({
        "dataset": ["nist_sd300b", "nist_sd300b"],
        "split": ["train", "train"],
        "pair_id": ["p0", "p1"],
        "label": [0, 1],
        "score": [0.1, 0.9],
        "logit": [-2.0, 2.0],
    })
    return source_raw, sift_raw, deep_raw


def test_feature_join_preserves_rows_and_labels():
    source_raw, sift_raw, deep_raw = _base_tables()
    source = prepare_sourceafis(source_raw, dataset="nist_sd300b", split="train")
    sift = prepare_sift(sift_raw, dataset="nist_sd300b", split="train")
    deep = prepare_deep(deep_raw, dataset="nist_sd300b", split="train")
    merged = merge_tables(source, sift, deep)
    assert len(merged) == 2
    assert merged["label"].tolist() == [0, 1]
    assert {"sourceafis_score", "sift_score", "deep_score", "deep_logit"}.issubset(merged.columns)


def test_feature_join_detects_label_mismatch():
    source_raw, sift_raw, deep_raw = _base_tables()
    deep_raw.loc[1, "label"] = 0
    with pytest.raises(DeepFusionV2Error, match="deep_label mismatches"):
        merge_tables(
            prepare_sourceafis(source_raw, dataset="nist_sd300b", split="train"),
            prepare_sift(sift_raw, dataset="nist_sd300b", split="train"),
            prepare_deep(deep_raw, dataset="nist_sd300b", split="train"),
        )


def test_pair_bundle_guard_detects_missing_canonical_pair(tmp_path):
    source_raw, sift_raw, deep_raw = _base_tables()
    bundle_dir = tmp_path / "data" / "manifests" / "nist_sd300b"
    bundle_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "pair_id": "p0",
                "label": 0,
                "split": "train",
                "subject_a": "1",
                "subject_b": "3",
                "frgp": "7",
                "path_a": "a0.png",
                "path_b": "b0.png",
            },
            {
                "pair_id": "missing",
                "label": 1,
                "split": "train",
                "subject_a": "2",
                "subject_b": "2",
                "frgp": "7",
                "path_a": "a2.png",
                "path_b": "b2.png",
            },
        ]
    ).to_csv(bundle_dir / "pairs_train.csv", index=False)
    merged = merge_tables(
        prepare_sourceafis(source_raw, dataset="nist_sd300b", split="train"),
        prepare_sift(sift_raw, dataset="nist_sd300b", split="train"),
        prepare_deep(deep_raw, dataset="nist_sd300b", split="train"),
    )

    with pytest.raises(DeepFusionV2Error, match="do not match canonical pair bundle"):
        assert_matches_pair_bundle(merged, repo_root=tmp_path, dataset="nist_sd300b", split="train")
