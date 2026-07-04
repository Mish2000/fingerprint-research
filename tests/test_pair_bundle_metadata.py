from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from pipelines.ingest.pair_bundle_utils import (
    CANONICAL_PAIR_COLUMNS,
    CANONICAL_PAIR_SCHEMA_VERSION,
    PAIR_BUILD_META_SCHEMA_VERSION,
    SPLIT_SUBJECTS_SCHEMA_VERSION,
    build_pairs_split_build_meta,
    build_split_subjects_metadata,
    canonicalize_pairs_df,
    validate_canonical_pairs_df,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
)
from src.fpbench.universal.pair_bundle_metadata import build_pair_bundle_metadata, file_sha256


def test_canonicalize_pairs_promotes_finger_id_and_adds_columns():
    df = pd.DataFrame([
        {
            "path_a": "a.png",
            "path_b": "b.png",
            "label": 1,
            "subject_a": 10,
            "subject_b": 10,
            "raw_frgp": 11,
            "finger_id": 2,
        }
    ])
    out = canonicalize_pairs_df(df, split="train")
    assert list(out.columns) == CANONICAL_PAIR_COLUMNS
    assert out.loc[0, "frgp"] == 2
    assert out.loc[0, "split"] == "train"


def test_metadata_builders_include_schema_versions_and_pair_columns():
    split_meta = build_split_subjects_metadata(
        splits={"train": [1], "val": [2], "test": [3]},
        seed=42,
        neg_per_pos=3,
        impostors_per_pos=3,
        same_finger_policy=True,
        negative_pair_policy="same_finger_other_subject_same_split",
        positive_pair_policy="same_subject_same_finger_plain_to_roll",
        finger_col="frgp",
        resolved_data_dir=Path("/tmp/x"),
        manifest_path=Path("/tmp/x/manifest.csv"),
    )
    assert split_meta["schema_version"] == SPLIT_SUBJECTS_SCHEMA_VERSION
    assert split_meta["pair_schema_version"] == CANONICAL_PAIR_SCHEMA_VERSION
    assert split_meta["pair_columns"] == CANONICAL_PAIR_COLUMNS

    build_meta = build_pairs_split_build_meta(
        dataset="nist_sd300b",
        seed=42,
        neg_per_pos=3,
        impostors_per_pos=3,
        finger_col="frgp",
        positive_pair_policy="same_subject_same_finger_plain_to_roll",
        negative_pair_policy="same_finger_other_subject_same_split",
    )
    assert build_meta["schema_version"] == PAIR_BUILD_META_SCHEMA_VERSION
    assert build_meta["pair_schema_version"] == CANONICAL_PAIR_SCHEMA_VERSION
    assert build_meta["pair_columns"] == CANONICAL_PAIR_COLUMNS


def test_validate_canonical_pairs_df_rejects_subject_label_mismatch():
    df = pd.DataFrame([
        {
            "pair_id": 0,
            "label": 1,
            "split": "val",
            "subject_a": 10,
            "subject_b": 11,
            "frgp": 2,
            "path_a": "a.png",
            "path_b": "b.png",
        }
    ])
    with pytest.raises(ValueError, match="positive rows with subject_a != subject_b"):
        validate_canonical_pairs_df(df, context="bad pairs", expected_split="val")


def test_validate_canonical_pairs_df_rejects_missing_canonical_column():
    df = pd.DataFrame([
        {
            "pair_id": 0,
            "label": 1,
            "split": "train",
            "subject_a": 10,
            "subject_b": 10,
            "frgp": 2,
            "path_a": "a.png",
        }
    ])
    with pytest.raises(ValueError, match="missing canonical pair columns"):
        validate_canonical_pairs_df(df, context="incomplete pairs", expected_split="train")


def test_validate_split_subjects_metadata_rejects_missing_pair_schema_fields():
    payload = {
        "schema_version": SPLIT_SUBJECTS_SCHEMA_VERSION,
        "seed": 42,
        "neg_per_pos": 3,
        "impostors_per_pos": 3,
        "same_finger_policy": True,
        "negative_pair_policy": "same_finger_other_subject_same_split",
        "positive_pair_policy": "same_subject_same_finger_plain_to_roll",
        "finger_col": "frgp",
        "splits": {"train": [1], "val": [2], "test": [3]},
    }
    with pytest.raises(ValueError, match="missing required fields"):
        validate_split_subjects_metadata(payload)


def test_validate_pairs_split_build_meta_rejects_wrong_pair_schema_version():
    payload = build_pairs_split_build_meta(
        dataset="nist_sd300b",
        seed=42,
        neg_per_pos=3,
        impostors_per_pos=3,
        finger_col="frgp",
        positive_pair_policy="same_subject_same_finger_plain_to_roll",
        negative_pair_policy="same_finger_other_subject_same_split",
    )
    payload["pair_schema_version"] = "old_schema"
    with pytest.raises(ValueError, match="pair_schema_version"):
        validate_pairs_split_build_meta(payload)


def test_build_pair_bundle_metadata_fingerprints_sources(tmp_path: Path):
    base = tmp_path / "data" / "manifests" / "nist_sd300b"
    pairs_dir = base / "pairs"
    pairs_dir.mkdir(parents=True)
    manifest = pd.DataFrame(
        [
            {"dataset": "nist_sd300b", "capture": "plain", "raw_frgp": 11, "frgp": 1, "path": "a.png"},
            {"dataset": "nist_sd300b", "capture": "roll", "raw_frgp": 1, "frgp": 1, "path": "b.png"},
        ]
    )
    manifest.to_csv(base / "manifest.csv", index=False)
    (pairs_dir / "split_subjects.json").write_text('{"splits":{"train":[1],"val":[2],"test":[3]}}', encoding="utf-8")
    pairs = pd.DataFrame(
        [
            {
                "pair_id": 0,
                "label": 1,
                "split": "val",
                "subject_a": 1,
                "subject_b": 1,
                "frgp": 1,
                "path_a": "a.png",
                "path_b": "b.png",
            },
            {
                "pair_id": 1,
                "label": 0,
                "split": "val",
                "subject_a": 1,
                "subject_b": 2,
                "frgp": 6,
                "path_a": "c.png",
                "path_b": "d.png",
            },
        ]
    )
    pair_path = base / "pairs_val.csv"
    pairs.to_csv(pair_path, index=False)

    metadata = build_pair_bundle_metadata(
        dataset="nist_sd300b",
        split="val",
        pair_source_path=pair_path,
        repo_root=tmp_path,
    )

    assert metadata["dataset_id"] == "nist_sd300b"
    assert metadata["pair_source_sha256"] == file_sha256(pair_path)
    assert metadata["manifest_source_sha256"] == file_sha256(base / "manifest.csv")
    assert metadata["split_subjects_sha256"] == file_sha256(pairs_dir / "split_subjects.json")
    assert metadata["pair_count"] == 2
    assert metadata["positive_count"] == 1
    assert metadata["negative_count"] == 1
    assert metadata["frgp_counts"] == {"1": 1, "6": 1}
    assert metadata["sd300_frgp_semantics"] == "anatomical"
    assert metadata["sd300_raw_frgp_available"] is True
