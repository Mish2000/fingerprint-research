from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality import IMAGE_QUALITY_FEATURES, extract_image_quality, parse_file_uri


METHOD_NAME = "sourceafis_sift_quality_fusion_v1"
SOURCEAFIS_METHOD = "sourceafis_open"
SIFT_PLAIN_ROLL_METHOD = "sift_plain_roll_v2"

PAIR_KEY_COLUMNS = ["dataset", "split", "pair_id"]
PAIR_CONTEXT_COLUMNS = [
    "dataset",
    "split",
    "pair_id",
    "label",
    "subject_a",
    "subject_b",
    "finger_position",
    "frgp",
    "path_a",
    "path_b",
]
DEFAULT_CATEGORICAL_COLUMNS = ["dataset", "finger_position", "frgp"]
SCORE_EXTRA_COLUMNS = ["inliers", "matches", "k1", "k2", "dpi_a", "dpi_b", "pair_total_ms"]
SCORE_CONTEXT_COLUMNS = ["subject_a", "subject_b", "finger_position", "frgp", "path_a", "path_b"]
QUALITY_DELTA_BASES = [
    "width",
    "height",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "contrast_proxy",
    "foreground_ratio",
    "sharpness_laplacian_var",
    "edge_density",
]


class FeatureJoinError(ValueError):
    """Raised when pair and score tables cannot be joined one-to-one."""


@dataclass(frozen=True)
class PairScoreSpec:
    dataset: str
    split: str
    pairs_csv: Path
    sourceafis_scores_csv: Path
    sift_plain_roll_scores_csv: Path
    sift_scores_csv: Path | None = None


def _string_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _finger_column(df: pd.DataFrame) -> str | None:
    for column in ("finger_position", "frgp", "finger_id", "finger"):
        if column in df.columns:
            return column
    return None


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], *, table_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise FeatureJoinError(f"{table_name} is missing required columns: {missing}")


def normalize_pair_frame(
    pairs: pd.DataFrame,
    *,
    dataset: str,
    split: str,
) -> pd.DataFrame:
    _ensure_columns(
        pairs,
        ["pair_id", "label", "subject_a", "subject_b", "path_a", "path_b"],
        table_name="pairs",
    )
    finger_col = _finger_column(pairs)
    if finger_col is None:
        raise FeatureJoinError("pairs is missing finger_position/frgp/finger_id.")

    out = pairs.copy()
    out["dataset"] = dataset if "dataset" not in out.columns else out["dataset"].fillna(dataset)
    out["split"] = split if "split" not in out.columns else out["split"].fillna(split)
    out["dataset"] = _string_series(out["dataset"])
    out["split"] = _string_series(out["split"]).str.lower()
    out["pair_id"] = _string_series(out["pair_id"])
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    out["subject_a"] = _string_series(out["subject_a"])
    out["subject_b"] = _string_series(out["subject_b"])
    out["finger_position"] = _string_series(out[finger_col])
    out["frgp"] = out["finger_position"]
    out["path_a"] = _string_series(out["path_a"])
    out["path_b"] = _string_series(out["path_b"])

    out = out[(out["dataset"] == dataset) & (out["split"] == split.lower())].copy()
    duplicates = out.duplicated(PAIR_KEY_COLUMNS, keep=False)
    if bool(duplicates.any()):
        examples = out.loc[duplicates, PAIR_KEY_COLUMNS].head(5).to_dict("records")
        raise FeatureJoinError(f"pairs contains duplicate pair keys: {examples}")
    return out[PAIR_CONTEXT_COLUMNS].reset_index(drop=True)


def load_pair_frame(path: str | Path, *, dataset: str, split: str, repo_root: Path | None = None) -> pd.DataFrame:
    resolved = parse_file_uri(path, repo_root=repo_root)
    return normalize_pair_frame(pd.read_csv(resolved), dataset=dataset, split=split)


def _first_existing_score_column(df: pd.DataFrame) -> str:
    for column in ("score", "raw_score", "similarity", "match_score"):
        if column in df.columns:
            return column
    raise FeatureJoinError("score table is missing a score/raw_score column.")


def _normalize_score_frame(
    scores: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    score_name: str,
    table_name: str,
) -> pd.DataFrame:
    _ensure_columns(scores, ["pair_id", "label"], table_name=table_name)
    score_column = _first_existing_score_column(scores)

    out = scores.copy()
    out["dataset"] = dataset if "dataset" not in out.columns else out["dataset"].fillna(dataset)
    out["split"] = split if "split" not in out.columns else out["split"].fillna(split)
    out["dataset"] = _string_series(out["dataset"])
    out["split"] = _string_series(out["split"]).str.lower()
    out["pair_id"] = _string_series(out["pair_id"])
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    out = out[(out["dataset"] == dataset) & (out["split"] == split.lower())].copy()

    duplicates = out.duplicated(PAIR_KEY_COLUMNS, keep=False)
    if bool(duplicates.any()):
        examples = out.loc[duplicates, PAIR_KEY_COLUMNS].head(5).to_dict("records")
        raise FeatureJoinError(f"{table_name} contains duplicate pair keys: {examples}")

    columns = [*PAIR_KEY_COLUMNS, "label", score_column]
    for column in SCORE_CONTEXT_COLUMNS:
        if column in out.columns and column not in columns:
            columns.append(column)
    for column in SCORE_EXTRA_COLUMNS:
        if column in out.columns and column not in columns:
            columns.append(column)
    renamed = out[columns].rename(
        columns={
            "label": f"{score_name}_label",
            score_column: score_name,
        }
    )
    for column in list(renamed.columns):
        if column in PAIR_KEY_COLUMNS or column in {score_name, f"{score_name}_label"}:
            continue
        renamed = renamed.rename(columns={column: f"{score_name}_{column}"})
    return renamed.reset_index(drop=True)


def load_score_frame(
    path: str | Path,
    *,
    dataset: str,
    split: str,
    score_name: str,
    repo_root: Path | None = None,
) -> pd.DataFrame:
    resolved = parse_file_uri(path, repo_root=repo_root)
    return _normalize_score_frame(
        pd.read_csv(resolved),
        dataset=dataset,
        split=split,
        score_name=score_name,
        table_name=str(resolved),
    )


def _merge_score(
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    score_name: str,
) -> pd.DataFrame:
    if len(scores) != len(pairs):
        raise FeatureJoinError(
            f"{score_name} row count does not match pair bundle: scores={len(scores)} pairs={len(pairs)}"
        )
    key_check = pairs[PAIR_KEY_COLUMNS].merge(
        scores[PAIR_KEY_COLUMNS],
        on=PAIR_KEY_COLUMNS,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    missing = int((key_check["_merge"] == "left_only").sum())
    extra = int((key_check["_merge"] == "right_only").sum())
    if missing or extra:
        examples = key_check.loc[key_check["_merge"] != "both", PAIR_KEY_COLUMNS + ["_merge"]].head(5).to_dict("records")
        raise FeatureJoinError(
            f"{score_name} pair key set does not match pair bundle: missing={missing} extra={extra} examples={examples}"
        )

    merged = pairs.merge(scores, on=PAIR_KEY_COLUMNS, how="left", validate="one_to_one")
    if merged[score_name].isna().any():
        missing = merged.loc[merged[score_name].isna(), PAIR_KEY_COLUMNS].head(5).to_dict("records")
        raise FeatureJoinError(f"Missing {score_name} rows for pair keys: {missing}")

    label_column = f"{score_name}_label"
    label_mismatch = merged[label_column].notna() & (merged[label_column].astype(int) != merged["label"].astype(int))
    if bool(label_mismatch.any()):
        examples = merged.loc[label_mismatch, PAIR_KEY_COLUMNS + ["label", label_column]].head(5).to_dict("records")
        raise FeatureJoinError(f"{score_name} labels do not match pair labels: {examples}")

    drop_columns = [label_column]
    for column in SCORE_CONTEXT_COLUMNS:
        score_column = f"{score_name}_{column}"
        if score_column not in merged.columns or column not in merged.columns:
            continue
        left = merged[column].fillna("").astype(str).str.strip()
        right = merged[score_column].fillna("").astype(str).str.strip()
        mismatch = right.ne("") & left.ne(right)
        if bool(mismatch.any()):
            examples = merged.loc[mismatch, PAIR_KEY_COLUMNS + [column, score_column]].head(5).to_dict("records")
            raise FeatureJoinError(f"{score_name} {column} values do not match pair bundle: {examples}")
        drop_columns.append(score_column)
    return merged.drop(columns=drop_columns)


def infer_ppi_from_path(path: Any) -> float:
    parts = [part for part in str(path).replace("\\", "/").split("/") if part]
    for index, part in enumerate(parts[:-1]):
        if part.lower() == "images":
            try:
                value = float(parts[index + 1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value) and 100 <= value <= 4000:
                return value
    return float("nan")


def _first_numeric(row: pd.Series, columns: Iterable[str]) -> float:
    for column in columns:
        if column not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return float("nan")


def add_ppi_features(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    ppi_a: list[float] = []
    ppi_b: list[float] = []
    for _, row in out.iterrows():
        left = _first_numeric(row, ["sourceafis_score_dpi_a", "dpi_a"])
        right = _first_numeric(row, ["sourceafis_score_dpi_b", "dpi_b"])
        if not math.isfinite(left):
            left = infer_ppi_from_path(row.get("path_a", ""))
        if not math.isfinite(right):
            right = infer_ppi_from_path(row.get("path_b", ""))
        ppi_a.append(left)
        ppi_b.append(right)
    out["ppi_a"] = ppi_a
    out["ppi_b"] = ppi_b
    out["ppi"] = pd.DataFrame({"a": ppi_a, "b": ppi_b}).mean(axis=1, skipna=True)
    return out


def add_quality_features(table: pd.DataFrame, *, repo_root: Path | None = None) -> pd.DataFrame:
    out = table.copy()
    path_values = [str(path) for path in pd.concat([out["path_a"], out["path_b"]], ignore_index=True)]
    unique_paths = list(dict.fromkeys(path_values))

    def load_quality(path_text: str) -> tuple[str, dict[str, float]]:
        return path_text, extract_image_quality(path_text, repo_root=repo_root)

    if len(unique_paths) >= 256:
        with ThreadPoolExecutor(max_workers=8) as executor:
            quality_by_path = dict(executor.map(load_quality, unique_paths))
    else:
        quality_by_path = dict(load_quality(path) for path in unique_paths)

    a_features = [
        {f"a_{key}": value for key, value in quality_by_path[str(path)].items()}
        for path in out["path_a"]
    ]
    b_features = [
        {f"b_{key}": value for key, value in quality_by_path[str(path)].items()}
        for path in out["path_b"]
    ]

    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(a_features), pd.DataFrame(b_features)], axis=1)
    for name in QUALITY_DELTA_BASES:
        out[f"pair_{name}_abs_delta"] = (
            pd.to_numeric(out[f"a_{name}"], errors="coerce") - pd.to_numeric(out[f"b_{name}"], errors="coerce")
        ).abs()
    return out


def build_feature_table(
    *,
    dataset: str,
    split: str,
    pairs_csv: str | Path,
    sourceafis_scores_csv: str | Path,
    sift_plain_roll_scores_csv: str | Path,
    sift_scores_csv: str | Path | None = None,
    repo_root: Path | None = None,
    include_quality: bool = True,
) -> pd.DataFrame:
    pairs = load_pair_frame(pairs_csv, dataset=dataset, split=split, repo_root=repo_root)
    sourceafis = load_score_frame(
        sourceafis_scores_csv,
        dataset=dataset,
        split=split,
        score_name="sourceafis_score",
        repo_root=repo_root,
    )
    sift_plain_roll = load_score_frame(
        sift_plain_roll_scores_csv,
        dataset=dataset,
        split=split,
        score_name="sift_plain_roll_v2_score",
        repo_root=repo_root,
    )

    table = _merge_score(pairs, sourceafis, score_name="sourceafis_score")
    table = _merge_score(table, sift_plain_roll, score_name="sift_plain_roll_v2_score")
    if sift_scores_csv is not None:
        sift = load_score_frame(
            sift_scores_csv,
            dataset=dataset,
            split=split,
            score_name="sift_score",
            repo_root=repo_root,
        )
        table = _merge_score(table, sift, score_name="sift_score")

    table = add_ppi_features(table)
    if include_quality:
        table = add_quality_features(table, repo_root=repo_root)
    table.insert(0, "method", METHOD_NAME)
    return table.reset_index(drop=True)


def build_feature_tables(
    specs: Iterable[PairScoreSpec],
    *,
    repo_root: Path | None = None,
    include_quality: bool = True,
) -> pd.DataFrame:
    frames = [
        build_feature_table(
            dataset=spec.dataset,
            split=spec.split,
            pairs_csv=spec.pairs_csv,
            sourceafis_scores_csv=spec.sourceafis_scores_csv,
            sift_plain_roll_scores_csv=spec.sift_plain_roll_scores_csv,
            sift_scores_csv=spec.sift_scores_csv,
            repo_root=repo_root,
            include_quality=include_quality,
        )
        for spec in specs
    ]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _is_numeric_feature_name(column: str) -> bool:
    if column in {
        "sourceafis_score",
        "sift_plain_roll_v2_score",
        "sift_score",
        "ppi",
        "ppi_a",
        "ppi_b",
    }:
        return True
    if column.startswith(("a_", "b_")):
        return True
    if column.startswith("pair_") and column.endswith("_abs_delta"):
        return True
    if re.match(r"^(sourceafis_score|sift_plain_roll_v2_score|sift_score)_(inliers|matches|k1|k2)$", column):
        return True
    return False


def default_numeric_feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in table.columns if _is_numeric_feature_name(str(column))]


def default_categorical_feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in DEFAULT_CATEGORICAL_COLUMNS if column in table.columns]
