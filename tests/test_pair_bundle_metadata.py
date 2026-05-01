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


def test_canonicalize_pairs_promotes_finger_id_and_adds_columns():
    df = pd.DataFrame([
        {
            "path_a": "a.png",
            "path_b": "b.png",
            "label": 1,
            "subject_a": 10,
            "subject_b": 10,
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
