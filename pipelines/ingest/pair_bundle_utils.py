from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

CANONICAL_PAIR_SCHEMA_VERSION = "v2_canonical_pair_csv"
SPLIT_SUBJECTS_SCHEMA_VERSION = "v2_split_subjects"
PAIR_BUILD_META_SCHEMA_VERSION = "v2_canonical_pair_bundle"

CANONICAL_PAIR_COLUMNS = [
    "pair_id",
    "label",
    "split",
    "subject_a",
    "subject_b",
    "frgp",
    "path_a",
    "path_b",
]

REQUIRED_SPLIT_SUBJECTS_FIELDS = [
    "schema_version",
    "seed",
    "neg_per_pos",
    "impostors_per_pos",
    "same_finger_policy",
    "negative_pair_policy",
    "positive_pair_policy",
    "finger_col",
    "pair_schema_version",
    "pair_columns",
    "splits",
]

REQUIRED_PAIR_BUILD_META_FIELDS = [
    "dataset",
    "seed",
    "neg_per_pos",
    "impostors_per_pos",
    "finger_col",
    "positive_pair_policy",
    "negative_pair_policy",
    "schema_version",
    "pair_schema_version",
    "pair_columns",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def _require_fields(payload: Mapping[str, Any], required_fields: Sequence[str], *, context: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")


def canonicalize_pairs_df(df: pd.DataFrame, *, split: Optional[str] = None) -> pd.DataFrame:
    out = df.copy()

    if "frgp" not in out.columns and "finger_id" in out.columns:
        out = out.rename(columns={"finger_id": "frgp"})

    if split is not None:
        out["split"] = str(split)
    elif "split" not in out.columns:
        raise ValueError("pairs dataframe must include a 'split' column or be canonicalized with split=...")

    for col in ["subject_a", "subject_b", "frgp", "label", "path_a", "path_b"]:
        if col not in out.columns:
            raise ValueError(f"pairs dataframe missing required column: {col}")

    out["split"] = out["split"].astype(str)
    out["label"] = out["label"].astype(int)
    out["subject_a"] = out["subject_a"].astype(int)
    out["subject_b"] = out["subject_b"].astype(int)
    out["frgp"] = out["frgp"].astype(int)
    out["path_a"] = out["path_a"].astype(str)
    out["path_b"] = out["path_b"].astype(str)

    sort_cols = ["label", "subject_a", "subject_b", "frgp", "path_a", "path_b"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    if "pair_id" in out.columns:
        out = out.drop(columns=["pair_id"])
    out.insert(0, "pair_id", range(len(out)))
    return out[CANONICAL_PAIR_COLUMNS].copy()


def validate_canonical_pairs_df(
    df: pd.DataFrame,
    *,
    context: str = "pairs dataframe",
    expected_split: Optional[str] = None,
    require_exact_columns: bool = True,
    require_non_empty: bool = False,
) -> pd.DataFrame:
    missing = [col for col in CANONICAL_PAIR_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"{context} missing canonical pair columns: {missing}")

    if require_exact_columns:
        extra = [col for col in df.columns if col not in CANONICAL_PAIR_COLUMNS]
        if extra:
            raise ValueError(f"{context} contains non-canonical extra columns: {extra}")
        if list(df.columns) != CANONICAL_PAIR_COLUMNS:
            raise ValueError(
                f"{context} must use canonical pair column order {CANONICAL_PAIR_COLUMNS}; found {list(df.columns)}"
            )

    out = df.loc[:, CANONICAL_PAIR_COLUMNS].copy()

    if require_non_empty and out.empty:
        raise ValueError(f"{context} must not be empty")

    if out[CANONICAL_PAIR_COLUMNS].isnull().any().any():
        null_counts = out[CANONICAL_PAIR_COLUMNS].isnull().sum()
        bad = {k: int(v) for k, v in null_counts.items() if int(v) > 0}
        raise ValueError(f"{context} contains nulls in canonical fields: {bad}")

    out["pair_id"] = out["pair_id"].astype(int)
    out["label"] = out["label"].astype(int)
    out["split"] = out["split"].astype(str)
    out["subject_a"] = out["subject_a"].astype(int)
    out["subject_b"] = out["subject_b"].astype(int)
    out["frgp"] = out["frgp"].astype(int)
    out["path_a"] = out["path_a"].astype(str)
    out["path_b"] = out["path_b"].astype(str)

    if set(out["label"].unique()) - {0, 1}:
        raise ValueError(f"{context} label column must contain only 0/1; found {sorted(out['label'].unique().tolist())}")

    expected_pair_ids = list(range(len(out)))
    actual_pair_ids = out["pair_id"].tolist()
    if actual_pair_ids != expected_pair_ids:
        raise ValueError(
            f"{context} pair_id must be contiguous and zero-based after canonicalization; "
            f"found first values {actual_pair_ids[:5]}"
        )

    if out.duplicated(subset=["pair_id"]).any():
        raise ValueError(f"{context} contains duplicate pair_id values")

    if expected_split is not None:
        split_values = sorted(out["split"].unique().tolist())
        if split_values != [str(expected_split)]:
            raise ValueError(
                f"{context} must contain only split={expected_split!r}; found split values {split_values}"
            )

    pos_bad = int(((out["label"] == 1) & (out["subject_a"] != out["subject_b"])).sum())
    if pos_bad:
        raise ValueError(f"{context} has {pos_bad} positive rows with subject_a != subject_b")

    neg_bad = int(((out["label"] == 0) & (out["subject_a"] == out["subject_b"])).sum())
    if neg_bad:
        raise ValueError(f"{context} has {neg_bad} negative rows with subject_a == subject_b")

    same_path_rows = int((out["path_a"] == out["path_b"]).sum())
    if same_path_rows:
        raise ValueError(f"{context} has {same_path_rows} rows where path_a == path_b")

    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, ensure_ascii=False), encoding="utf-8")


def build_split_subjects_metadata(
    *,
    splits: Mapping[str, Iterable[int]],
    seed: int,
    neg_per_pos: int,
    impostors_per_pos: Optional[int] = None,
    same_finger_policy: Any,
    negative_pair_policy: str,
    positive_pair_policy: str,
    finger_col: str,
    resolved_data_dir: Optional[Path | str] = None,
    manifest_path: Optional[Path | str] = None,
    max_pos_per_subject: Optional[int] = None,
    max_pos_per_finger: Optional[int] = None,
    pair_mode: Optional[str] = None,
    schema_version: str = SPLIT_SUBJECTS_SCHEMA_VERSION,
    pair_schema_version: str = CANONICAL_PAIR_SCHEMA_VERSION,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": str(schema_version),
        "seed": int(seed),
        "neg_per_pos": int(neg_per_pos),
        "impostors_per_pos": int(impostors_per_pos if impostors_per_pos is not None else neg_per_pos),
        "same_finger_policy": same_finger_policy,
        "negative_pair_policy": str(negative_pair_policy),
        "positive_pair_policy": str(positive_pair_policy),
        "finger_col": str(finger_col),
        "pair_schema_version": str(pair_schema_version),
        "pair_columns": list(CANONICAL_PAIR_COLUMNS),
        "splits": {k: sorted(int(x) for x in v) for k, v in splits.items()},
    }
    if resolved_data_dir is not None:
        payload["resolved_data_dir"] = str(resolved_data_dir)
    if manifest_path is not None:
        payload["manifest_path"] = str(manifest_path)
    if max_pos_per_subject is not None:
        payload["max_pos_per_subject"] = int(max_pos_per_subject)
    if max_pos_per_finger is not None:
        payload["max_pos_per_finger"] = int(max_pos_per_finger)
    if pair_mode is not None:
        payload["pair_mode"] = str(pair_mode)
    validate_split_subjects_metadata(payload, context="split_subjects metadata")
    return payload


def validate_split_subjects_metadata(payload: Mapping[str, Any], *, context: str = "split_subjects metadata") -> None:
    _require_fields(payload, REQUIRED_SPLIT_SUBJECTS_FIELDS, context=context)

    if str(payload.get("schema_version")) != SPLIT_SUBJECTS_SCHEMA_VERSION:
        raise ValueError(
            f"{context} schema_version must be {SPLIT_SUBJECTS_SCHEMA_VERSION!r}; "
            f"found {payload.get('schema_version')!r}"
        )
    if str(payload.get("pair_schema_version")) != CANONICAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            f"{context} pair_schema_version must be {CANONICAL_PAIR_SCHEMA_VERSION!r}; "
            f"found {payload.get('pair_schema_version')!r}"
        )
    if list(payload.get("pair_columns", [])) != CANONICAL_PAIR_COLUMNS:
        raise ValueError(
            f"{context} pair_columns must be {CANONICAL_PAIR_COLUMNS}; "
            f"found {payload.get('pair_columns')!r}"
        )

    splits = payload.get("splits")
    if not isinstance(splits, Mapping) or not splits:
        raise ValueError(f"{context} must include a non-empty splits mapping")

    split_keys = sorted(str(k) for k in splits.keys())
    if split_keys != ["test", "train", "val"]:
        raise ValueError(f"{context} splits must contain train/val/test; found {split_keys}")

    seen_subjects: set[int] = set()
    for split_name, subject_ids in splits.items():
        if not isinstance(subject_ids, Iterable):
            raise ValueError(f"{context} split {split_name!r} must be iterable")
        normalized = [int(x) for x in subject_ids]
        if normalized != sorted(normalized):
            raise ValueError(f"{context} split {split_name!r} must be sorted ascending")
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{context} split {split_name!r} contains duplicate subject IDs")
        overlap = seen_subjects.intersection(normalized)
        if overlap:
            overlap_preview = sorted(overlap)[:5]
            raise ValueError(
                f"{context} subject IDs overlap across splits; first duplicates: {overlap_preview}"
            )
        seen_subjects.update(normalized)


def build_pairs_split_build_meta(
    *,
    dataset: str,
    seed: int,
    neg_per_pos: int,
    impostors_per_pos: Optional[int],
    finger_col: str,
    positive_pair_policy: str,
    negative_pair_policy: str,
    schema_version: str = PAIR_BUILD_META_SCHEMA_VERSION,
    pair_schema_version: str = CANONICAL_PAIR_SCHEMA_VERSION,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "dataset": str(dataset),
        "seed": int(seed),
        "neg_per_pos": int(neg_per_pos),
        "impostors_per_pos": int(impostors_per_pos if impostors_per_pos is not None else neg_per_pos),
        "finger_col": str(finger_col),
        "positive_pair_policy": str(positive_pair_policy),
        "negative_pair_policy": str(negative_pair_policy),
        "schema_version": str(schema_version),
        "pair_schema_version": str(pair_schema_version),
        "pair_columns": list(CANONICAL_PAIR_COLUMNS),
    }
    if extra:
        payload.update({str(k): _jsonable(v) for k, v in extra.items()})
    validate_pairs_split_build_meta(payload, context="pairs_split_build metadata")
    return payload


def validate_pairs_split_build_meta(payload: Mapping[str, Any], *, context: str = "pairs_split_build metadata") -> None:
    _require_fields(payload, REQUIRED_PAIR_BUILD_META_FIELDS, context=context)

    if str(payload.get("schema_version")) != PAIR_BUILD_META_SCHEMA_VERSION:
        raise ValueError(
            f"{context} schema_version must be {PAIR_BUILD_META_SCHEMA_VERSION!r}; "
            f"found {payload.get('schema_version')!r}"
        )
    if str(payload.get("pair_schema_version")) != CANONICAL_PAIR_SCHEMA_VERSION:
        raise ValueError(
            f"{context} pair_schema_version must be {CANONICAL_PAIR_SCHEMA_VERSION!r}; "
            f"found {payload.get('pair_schema_version')!r}"
        )
    if list(payload.get("pair_columns", [])) != CANONICAL_PAIR_COLUMNS:
        raise ValueError(
            f"{context} pair_columns must be {CANONICAL_PAIR_COLUMNS}; "
            f"found {payload.get('pair_columns')!r}"
        )
