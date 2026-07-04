from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .pair_bundle_metadata import build_pair_bundle_metadata

try:  # quality is optional for fast unit tests, but required for the official variant.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - exercised only on minimal environments.
    cv2 = None

METHOD_NAME = "sourceafis_sift_quality_deep_fusion_v2"
GROUP_WEIGHTED_METHOD = "sourceafis_sift_quality_deep_group_weighted_fusion_v2"
METHOD_PROTO = "sourceafis_sift_deep_score_fusion_v2_proto"
DEEP_METHOD = "deep_pair_reranker_fast_ddp"
SOURCEAFIS_METHOD = "sourceafis_open"
SIFT_METHOD = "sift_plain_roll_v2"

DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_TARGET_FARS = (0.005, 0.01)
PAIR_KEY_COLUMNS = ["dataset", "split", "pair_id"]
CONTEXT_COLUMNS = [
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
CATEGORICAL_FEATURES = ["dataset", "finger_position", "frgp"]
SOURCE_FEATURES = ["sourceafis_score", "source_dpi_a", "source_dpi_b"]
SIFT_SCORE_FEATURES = ["sift_score"]
SIFT_GEOMETRY_FEATURES = ["sift_inliers", "sift_matches", "sift_k1", "sift_k2"]
DEEP_SCORE_FEATURES = ["deep_score"]
DEEP_LOGIT_FEATURES = ["deep_logit"]
QUALITY_BASE_FEATURES = [
    "image_read_ok",
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
QUALITY_FEATURES = (
    [f"a_{name}" for name in QUALITY_BASE_FEATURES]
    + [f"b_{name}" for name in QUALITY_BASE_FEATURES]
    + [f"pair_{name}_abs_delta" for name in QUALITY_DELTA_BASES]
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    description: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = tuple(CATEGORICAL_FEATURES)
    include_quality: bool = False
    is_official: bool = False
    group_weights: Mapping[str, float] | None = None
    group_weight_mode: str | None = None
    group_weight_metric: str | None = None


VARIANTS: dict[str, VariantSpec] = {
    "sourceafis_only_calibrated": VariantSpec(
        name="sourceafis_only_calibrated",
        description="SourceAFIS score calibrated with train-only logistic regression.",
        numeric_features=tuple(SOURCE_FEATURES),
    ),
    "sourceafis_sift_score": VariantSpec(
        name="sourceafis_sift_score",
        description="SourceAFIS plus SIFT plain-roll score.",
        numeric_features=tuple(SOURCE_FEATURES + SIFT_SCORE_FEATURES),
    ),
    "sourceafis_sift_geometry": VariantSpec(
        name="sourceafis_sift_geometry",
        description="SourceAFIS plus SIFT score and geometry counts.",
        numeric_features=tuple(SOURCE_FEATURES + SIFT_SCORE_FEATURES + SIFT_GEOMETRY_FEATURES),
    ),
    "sourceafis_sift_quality": VariantSpec(
        name="sourceafis_sift_quality",
        description="SourceAFIS plus SIFT geometry and deterministic image-quality features.",
        numeric_features=tuple(SOURCE_FEATURES + SIFT_SCORE_FEATURES + SIFT_GEOMETRY_FEATURES + QUALITY_FEATURES),
        include_quality=True,
    ),
    "sourceafis_sift_deep_score": VariantSpec(
        name="sourceafis_sift_deep_score",
        description="SourceAFIS plus SIFT geometry and deep probability score.",
        numeric_features=tuple(SOURCE_FEATURES + SIFT_SCORE_FEATURES + SIFT_GEOMETRY_FEATURES + DEEP_SCORE_FEATURES),
    ),
    "sourceafis_sift_deep_logit": VariantSpec(
        name="sourceafis_sift_deep_logit",
        description="SourceAFIS plus SIFT geometry and deep probability/logit features.",
        numeric_features=tuple(
            SOURCE_FEATURES + SIFT_SCORE_FEATURES + SIFT_GEOMETRY_FEATURES + DEEP_SCORE_FEATURES + DEEP_LOGIT_FEATURES
        ),
    ),
    METHOD_NAME: VariantSpec(
        name=METHOD_NAME,
        description="Official Fusion v2: SourceAFIS + SIFT geometry + image quality + deep pair reranker score/logit.",
        numeric_features=tuple(
            SOURCE_FEATURES
            + SIFT_SCORE_FEATURES
            + SIFT_GEOMETRY_FEATURES
            + QUALITY_FEATURES
            + DEEP_SCORE_FEATURES
            + DEEP_LOGIT_FEATURES
        ),
        include_quality=True,
        is_official=True,
    ),
    GROUP_WEIGHTED_METHOD: VariantSpec(
        name=GROUP_WEIGHTED_METHOD,
        description=(
            "Fusion v2 group-weighted experiment: normalized features are multiplied by method-level "
            "weights for SourceAFIS, SIFT, deep CNN, and quality before logistic regression."
        ),
        numeric_features=tuple(
            SOURCE_FEATURES
            + SIFT_SCORE_FEATURES
            + SIFT_GEOMETRY_FEATURES
            + QUALITY_FEATURES
            + DEEP_SCORE_FEATURES
            + DEEP_LOGIT_FEATURES
        ),
        include_quality=True,
    ),
}


class DeepFusionV2Error(ValueError):
    pass


GROUP_FEATURE_MAP: dict[str, tuple[str, ...]] = {
    "sourceafis": tuple(SOURCE_FEATURES),
    "sift": tuple(SIFT_SCORE_FEATURES + SIFT_GEOMETRY_FEATURES),
    "deep": tuple(DEEP_SCORE_FEATURES + DEEP_LOGIT_FEATURES),
    "quality": tuple(QUALITY_FEATURES),
}
GROUP_WEIGHT_KEYS = tuple(GROUP_FEATURE_MAP.keys())
DEFAULT_GROUP_WEIGHTS = {
    "sourceafis": 0.45,
    "sift": 0.15,
    "deep": 0.30,
    "quality": 0.10,
}


class FeatureGroupWeightScaler(BaseEstimator, TransformerMixin):
    """Apply method-level weights after numeric feature normalization.

    The ColumnTransformer emits normalized numeric features first and one-hot
    categorical features afterwards. This transformer multiplies only the
    numeric prefix by group weights and leaves metadata/categorical columns at
    weight 1.0.
    """

    def __init__(self, numeric_weights: Iterable[float]):
        self.numeric_weights = tuple(float(value) for value in numeric_weights)

    def fit(self, X: Any, y: Any = None):  # noqa: N803 - sklearn API
        return self

    def transform(self, X: Any) -> np.ndarray:  # noqa: N803 - sklearn API
        arr = np.asarray(X, dtype=float).copy()
        if arr.ndim != 2:
            raise DeepFusionV2Error(f"Expected a 2D feature matrix, got shape={arr.shape}")
        n = min(len(self.numeric_weights), arr.shape[1])
        if n:
            arr[:, :n] *= np.asarray(self.numeric_weights[:n], dtype=float)
        return arr


def normalize_group_weights(weights: Mapping[str, float] | None) -> dict[str, float]:
    """Return positive group weights normalized to sum to 1.0."""

    if weights is None:
        weights = DEFAULT_GROUP_WEIGHTS
    unknown = sorted(set(weights) - set(GROUP_WEIGHT_KEYS))
    if unknown:
        raise DeepFusionV2Error(f"Unknown group weight key(s): {unknown}. Expected: {list(GROUP_WEIGHT_KEYS)}")
    values = {key: float(weights.get(key, 0.0)) for key in GROUP_WEIGHT_KEYS}
    negative = {key: value for key, value in values.items() if value < 0}
    if negative:
        raise DeepFusionV2Error(f"Group weights must be non-negative. Got: {negative}")
    total = float(sum(values.values()))
    if not math.isfinite(total) or total <= 0:
        raise DeepFusionV2Error("Group weights must have a positive finite sum.")
    return {key: float(value / total) for key, value in values.items()}


def parse_group_weights(value: str | Mapping[str, float] | None) -> dict[str, float]:
    if value is None or value == "":
        return normalize_group_weights(DEFAULT_GROUP_WEIGHTS)
    if isinstance(value, Mapping):
        return normalize_group_weights(value)
    parsed: dict[str, float] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise DeepFusionV2Error(f"Invalid group weight item {item!r}; expected name=value")
        key, raw = item.split("=", 1)
        key = key.strip().lower()
        parsed[key] = float(raw.strip())
    return normalize_group_weights(parsed)


def feature_group_for(column: str) -> str | None:
    for group, columns in GROUP_FEATURE_MAP.items():
        if column in columns:
            return group
    return None


def numeric_weight_vector(numeric_features: Iterable[str], weights: Mapping[str, float] | None) -> tuple[float, ...]:
    normalized = normalize_group_weights(weights)
    vector: list[float] = []
    for column in numeric_features:
        group = feature_group_for(str(column))
        vector.append(float(normalized[group]) if group is not None else 1.0)
    return tuple(vector)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_csv_list(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def parse_float_list(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    return tuple(float(item) for item in value)


def read_csv(path: str | Path, *, label: str) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    df = pd.read_csv(resolved)
    if df.empty:
        raise DeepFusionV2Error(f"{label} is empty: {resolved}")
    return df


def sourceafis_train_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root
        / "artifacts/reports/benchmark/plain_roll_train_scores_v2_anatomical_full_pairs/scores"
        / f"scores_{dataset}_sourceafis_open_train.csv"
    )


def sift_train_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root
        / "artifacts/reports/benchmark/plain_roll_train_scores_v2_anatomical_full_pairs/scores"
        / f"scores_{dataset}_sift_plain_roll_v2_train.csv"
    )


def deep_train_path(repo_root: Path, dataset: str) -> Path:
    candidate = (
        repo_root
        / "artifacts/reports/benchmark/plain_roll_train_scores_v2_anatomical_full_pairs/scores"
        / f"scores_{dataset}_{DEEP_METHOD}_train.csv"
    )
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Missing deep train scores for {dataset}: {candidate}")


def sourceafis_eval_path(repo_root: Path, dataset: str, split: str) -> Path:
    return (
        repo_root
        / "artifacts/reports/benchmark/plain_roll_final_sourceafis_v2_anatomical_full_pairs/scores"
        / f"scores_{dataset}_sourceafis_open_{split}.csv"
    )


def sift_eval_path(repo_root: Path, dataset: str, split: str) -> Path:
    return (
        repo_root
        / "artifacts/reports/benchmark/plain_roll_final_baselines_v2_anatomical_full_pairs"
        / f"scores_{dataset}_{SIFT_METHOD}_{split}.csv"
    )


def deep_eval_path(repo_root: Path, dataset: str, split: str) -> Path:
    return repo_root / "artifacts/reports/benchmark/deep_pair_reranker_fast_ddp_full_pairs/scores" / f"scores_{dataset}_{DEEP_METHOD}_{split}.csv"


def pair_bundle_path(repo_root: Path, dataset: str, split: str) -> Path | None:
    candidates = [
        repo_root / "data" / "manifests" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "manifests" / dataset / "pairs" / f"pairs_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def pair_bundle_metadata(repo_root: Path, dataset: str, split: str) -> dict[str, Any] | None:
    path = pair_bundle_path(repo_root, dataset, split)
    if path is None:
        return None
    return build_pair_bundle_metadata(dataset=dataset, split=split, pair_source_path=path, repo_root=repo_root)


def assert_matches_pair_bundle(table: pd.DataFrame, *, repo_root: Path, dataset: str, split: str) -> None:
    path = pair_bundle_path(repo_root, dataset, split)
    if path is None:
        return
    pairs = pd.read_csv(path)
    key = PAIR_KEY_COLUMNS
    expected = pairs.copy()
    expected["dataset"] = dataset if "dataset" not in expected.columns else expected["dataset"].fillna(dataset)
    expected["split"] = split if "split" not in expected.columns else expected["split"].fillna(split)
    expected["dataset"] = expected["dataset"].astype(str).str.strip()
    expected["split"] = expected["split"].astype(str).str.strip().str.lower()
    expected["pair_id"] = expected["pair_id"].astype(str).str.strip()
    actual = table.copy()
    actual["dataset"] = actual["dataset"].astype(str).str.strip()
    actual["split"] = actual["split"].astype(str).str.strip().str.lower()
    actual["pair_id"] = actual["pair_id"].astype(str).str.strip()
    merged = expected[key + ["label", "frgp"]].merge(
        actual[key + ["label", "frgp"]],
        on=key,
        how="outer",
        suffixes=("_pair_bundle", "_scores"),
        indicator=True,
    )
    missing = int((merged["_merge"] == "left_only").sum())
    extra = int((merged["_merge"] == "right_only").sum())
    if missing or extra:
        raise DeepFusionV2Error(
            f"{dataset}/{split} score tables do not match canonical pair bundle {path}: "
            f"missing={missing}, extra={extra}"
        )
    both = merged[merged["_merge"] == "both"].copy()
    label_mismatch = int(
        (pd.to_numeric(both["label_pair_bundle"], errors="coerce").astype(int) != pd.to_numeric(both["label_scores"], errors="coerce").astype(int)).sum()
    )
    frgp_mismatch = int(
        (pd.to_numeric(both["frgp_pair_bundle"], errors="coerce").astype(int) != pd.to_numeric(both["frgp_scores"], errors="coerce").astype(int)).sum()
    )
    if label_mismatch or frgp_mismatch:
        raise DeepFusionV2Error(
            f"{dataset}/{split} score tables disagree with canonical pair bundle {path}: "
            f"label={label_mismatch}, frgp={frgp_mismatch}"
        )


def normalize_key_columns(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    if "split" not in out.columns:
        out["split"] = split
    if "pair_id" not in out.columns:
        raise DeepFusionV2Error(f"table is missing pair_id. Columns={list(out.columns)}")
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out = out[(out["dataset"] == dataset) & (out["split"] == split.lower())].copy()
    dup = out.duplicated(PAIR_KEY_COLUMNS, keep=False)
    if bool(dup.any()):
        examples = out.loc[dup, PAIR_KEY_COLUMNS].head(5).to_dict("records")
        raise DeepFusionV2Error(f"Duplicate pair keys in {dataset}/{split}: {examples}")
    return out


def score_column(df: pd.DataFrame) -> str:
    for column in ("score", "raw_score", "similarity", "match_score", "probability"):
        if column in df.columns:
            return column
    raise DeepFusionV2Error(f"Could not find score column. Columns={list(df.columns)}")


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str], *, table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise DeepFusionV2Error(f"{table_name} missing required columns: {missing}; found={list(df.columns)}")


def prepare_sourceafis(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = normalize_key_columns(df, dataset=dataset, split=split)
    _ensure_columns(out, ["label"], table_name="sourceafis scores")
    sc = score_column(out)
    if "score" not in out.columns and "raw_score" in out.columns:
        out["score"] = out["raw_score"]
        sc = "score"
    cols = ["dataset", "split", "pair_id", "label", sc]
    for extra in ("raw_score", "dpi_a", "dpi_b"):
        if extra in out.columns and extra not in cols:
            cols.append(extra)
    renamed = out[cols].rename(
        columns={
            "label": "source_label",
            sc: "sourceafis_score",
            "raw_score": "sourceafis_raw_score",
            "dpi_a": "source_dpi_a",
            "dpi_b": "source_dpi_b",
        }
    )
    return renamed.reset_index(drop=True)


def prepare_sift(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = normalize_key_columns(df, dataset=dataset, split=split)
    required = ["label", "path_a", "path_b", "subject_a", "subject_b"]
    _ensure_columns(out, required, table_name="sift scores")
    if "finger_position" not in out.columns:
        if "frgp" in out.columns:
            out["finger_position"] = out["frgp"]
        else:
            raise DeepFusionV2Error("SIFT table is missing finger_position/frgp.")
    if "frgp" not in out.columns:
        out["frgp"] = out["finger_position"]
    sc = score_column(out)
    cols = [
        "dataset", "split", "pair_id", "label",
        "path_a", "path_b", "subject_a", "subject_b", "finger_position", "frgp",
        sc,
    ]
    for extra in ("inliers", "matches", "k1", "k2"):
        if extra in out.columns and extra not in cols:
            cols.append(extra)
    renamed = out[cols].rename(
        columns={
            "label": "sift_label",
            sc: "sift_score",
            "inliers": "sift_inliers",
            "matches": "sift_matches",
            "k1": "sift_k1",
            "k2": "sift_k2",
        }
    )
    return renamed.reset_index(drop=True)


def prepare_deep(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = normalize_key_columns(df, dataset=dataset, split=split)
    _ensure_columns(out, ["label"], table_name="deep scores")
    sc = score_column(out)
    cols = ["dataset", "split", "pair_id", "label", sc]
    for extra in ("logit", "probability", "input_size"):
        if extra in out.columns and extra not in cols:
            cols.append(extra)
    renamed = out[cols].rename(
        columns={
            "label": "deep_label",
            sc: "deep_score",
            "logit": "deep_logit",
            "probability": "deep_probability",
        }
    )
    return renamed.reset_index(drop=True)


def merge_tables(source: pd.DataFrame, sift: pd.DataFrame, deep: pd.DataFrame) -> pd.DataFrame:
    key = PAIR_KEY_COLUMNS
    merged = sift.merge(source, on=key, how="inner", validate="one_to_one")
    merged = merged.merge(deep, on=key, how="inner", validate="one_to_one")
    if len(merged) != len(sift):
        raise DeepFusionV2Error(f"Merge lost rows: sift={len(sift)} merged={len(merged)}")

    merged["label"] = pd.to_numeric(merged["sift_label"], errors="raise").astype(int)
    for label_col in ("source_label", "deep_label"):
        other = pd.to_numeric(merged[label_col], errors="raise").astype(int)
        mismatch = other != merged["label"]
        if bool(mismatch.any()):
            examples = merged.loc[mismatch, key + ["label", label_col]].head(5).to_dict("records")
            raise DeepFusionV2Error(f"{label_col} mismatches label. Examples: {examples}")

    merged["subject_a"] = merged["subject_a"].astype(str).str.strip()
    merged["subject_b"] = merged["subject_b"].astype(str).str.strip()
    merged["finger_position"] = merged["finger_position"].astype(str).str.strip()
    merged["frgp"] = merged["frgp"].astype(str).str.strip()
    for column in SOURCE_FEATURES + SIFT_SCORE_FEATURES + SIFT_GEOMETRY_FEATURES + DEEP_SCORE_FEATURES + DEEP_LOGIT_FEATURES:
        if column not in merged.columns:
            merged[column] = np.nan
    for column in CATEGORICAL_FEATURES:
        if column not in merged.columns:
            merged[column] = "__missing__"
        values = merged[column].fillna("__missing__").astype(str).str.strip()
        merged[column] = values.mask(values == "", "__missing__")
    return merged.reset_index(drop=True)


def resolve_image_path(value: Any, *, repo_root: Path | None = None) -> Path:
    raw = str(value).strip()
    direct = Path(raw)
    if direct.exists():
        return direct.resolve()
    normalized = raw.replace("\\", "/")
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.extend([repo_root / normalized, repo_root / "data" / normalized, repo_root / "data" / "raw" / normalized])
        lower = normalized.lower()
        marker = "data/raw/"
        if marker in lower:
            idx = lower.index(marker)
            rel_after_raw = normalized[idx + len(marker):].lstrip("/")
            candidates.append(repo_root / "data" / "raw" / rel_after_raw)
            candidates.append(repo_root / (normalized[idx:].lstrip("/")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Return the original path for a clean image_read_ok=0 rather than failing the whole run.
    return direct


def _empty_quality() -> dict[str, float]:
    return {
        "image_read_ok": 0.0,
        "width": float("nan"),
        "height": float("nan"),
        "aspect_ratio": float("nan"),
        "mean_intensity": float("nan"),
        "std_intensity": float("nan"),
        "contrast_proxy": float("nan"),
        "foreground_ratio": float("nan"),
        "sharpness_laplacian_var": float("nan"),
        "edge_density": float("nan"),
    }


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except Exception:
        return float("nan")
    return number if math.isfinite(number) else float("nan")


@lru_cache(maxsize=65536)
def _extract_quality_cached(path_text: str) -> tuple[tuple[str, float], ...]:
    if cv2 is None:
        return tuple(_empty_quality().items())
    img = cv2.imread(path_text, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return tuple(_empty_quality().items())
    height, width = img.shape[:2]
    img_f = img.astype(np.float32)
    mean = float(np.mean(img_f))
    std = float(np.std(img_f))
    try:
        p05, p95 = np.percentile(img_f, [5, 95])
    except Exception:
        p05, p95 = float("nan"), float("nan")
    try:
        threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_fraction = float(np.mean(img < threshold))
        light_fraction = float(np.mean(img >= threshold))
        foreground_ratio = min(max(dark_fraction, 0.0), 1.0)
        if foreground_ratio > 0.95:
            foreground_ratio = min(max(light_fraction, 0.0), 1.0)
    except Exception:
        foreground_ratio = float("nan")
    try:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = float(laplian_var if False else laplacian.var())
    except Exception:
        sharpness = float("nan")
    try:
        edges = cv2.Canny(img, 50, 150)
        edge_density = float(np.mean(edges > 0))
    except Exception:
        edge_density = float("nan")
    quality = {
        "image_read_ok": 1.0,
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height) if height else float("nan"),
        "mean_intensity": mean,
        "std_intensity": std,
        "contrast_proxy": float((p95 - p05) / 255.0) if math.isfinite(float(p95)) and math.isfinite(float(p05)) else float("nan"),
        "foreground_ratio": foreground_ratio,
        "sharpness_laplacian_var": sharpness,
        "edge_density": edge_density,
    }
    return tuple((name, _finite_or_nan(quality[name])) for name in QUALITY_BASE_FEATURES)


def extract_image_quality(path: Any, *, repo_root: Path | None = None) -> dict[str, float]:
    resolved = resolve_image_path(path, repo_root=repo_root)
    return dict(_extract_quality_cached(str(resolved)))


def add_quality_features(table: pd.DataFrame, *, repo_root: Path | None = None) -> pd.DataFrame:
    out = table.copy()
    path_values = list(dict.fromkeys([*out["path_a"].astype(str).tolist(), *out["path_b"].astype(str).tolist()]))
    quality_by_path = {path: extract_image_quality(path, repo_root=repo_root) for path in path_values}
    a_quality = [{f"a_{k}": v for k, v in quality_by_path[str(path)].items()} for path in out["path_a"].astype(str)]
    b_quality = [{f"b_{k}": v for k, v in quality_by_path[str(path)].items()} for path in out["path_b"].astype(str)]
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(a_quality), pd.DataFrame(b_quality)], axis=1)
    for name in QUALITY_DELTA_BASES:
        out[f"pair_{name}_abs_delta"] = (
            pd.to_numeric(out[f"a_{name}"], errors="coerce") - pd.to_numeric(out[f"b_{name}"], errors="coerce")
        ).abs()
    return out


def load_train_dataset(repo_root: Path, dataset: str) -> pd.DataFrame:
    source = prepare_sourceafis(read_csv(sourceafis_train_path(repo_root, dataset), label=f"{dataset} SourceAFIS train"), dataset=dataset, split="train")
    sift = prepare_sift(read_csv(sift_train_path(repo_root, dataset), label=f"{dataset} SIFT train"), dataset=dataset, split="train")
    deep = prepare_deep(read_csv(deep_train_path(repo_root, dataset), label=f"{dataset} deep train"), dataset=dataset, split="train")
    merged = merge_tables(source=source, sift=sift, deep=deep)
    assert_matches_pair_bundle(merged, repo_root=repo_root, dataset=dataset, split="train")
    return merged


def load_eval_dataset(repo_root: Path, dataset: str, split: str, source_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source_key = f"{dataset}:{split}"
    if source_key not in source_cache:
        source_cache[source_key] = read_csv(sourceafis_eval_path(repo_root, dataset, split), label=f"{dataset} SourceAFIS {split}")
    source = prepare_sourceafis(source_cache[source_key], dataset=dataset, split=split)
    sift = prepare_sift(read_csv(sift_eval_path(repo_root, dataset, split), label=f"{dataset} SIFT {split}"), dataset=dataset, split=split)
    deep = prepare_deep(read_csv(deep_eval_path(repo_root, dataset, split), label=f"{dataset} deep {split}"), dataset=dataset, split=split)
    merged = merge_tables(source=source, sift=sift, deep=deep)
    assert_matches_pair_bundle(merged, repo_root=repo_root, dataset=dataset, split=split)
    return merged


def ensure_quality_if_needed(tables: Mapping[tuple[str, str], pd.DataFrame], *, repo_root: Path, enabled: bool) -> dict[tuple[str, str], pd.DataFrame]:
    if not enabled:
        return dict(tables)
    return {key: add_quality_features(table, repo_root=repo_root) for key, table in tables.items()}


def assert_train_only(table: pd.DataFrame) -> None:
    if "split" not in table.columns:
        raise DeepFusionV2Error("Feature table is missing split column.")
    splits = sorted(set(table["split"].astype(str).str.lower()))
    disallowed = [split for split in splits if split != "train"]
    if disallowed:
        raise DeepFusionV2Error(f"Fusion fitting may use train rows only; found non-train splits: {disallowed}")


def median_imputer_preserving_empty_features() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - sklearn < 1.2 compatibility.
        return SimpleImputer(strategy="median")


def build_model(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    *,
    group_weights: Mapping[str, float] | None = None,
) -> Pipeline:
    numeric = list(numeric_features)
    categorical = list(categorical_features)
    numeric_pipe = Pipeline(steps=[("imputer", median_imputer_preserving_empty_features()), ("scaler", StandardScaler())])
    # sklearn >=1.2 uses sparse_output; older versions use sparse. Use a small compatibility shim.
    try:
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover
        one_hot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipe, numeric), ("categorical", one_hot, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps: list[tuple[str, Any]] = [("features", preprocessor)]
    if group_weights is not None:
        steps.append(("group_weights", FeatureGroupWeightScaler(numeric_weight_vector(numeric, group_weights))))
    steps.append(
        (
            "logistic_regression",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=13, solver="lbfgs"),
        )
    )
    return Pipeline(steps=steps)


def _model_columns(spec: VariantSpec) -> list[str]:
    return list(spec.numeric_features) + list(spec.categorical_features)


def fit_variant_model(train: pd.DataFrame, spec: VariantSpec) -> Pipeline:
    assert_train_only(train)
    labels = pd.to_numeric(train["label"], errors="raise").astype(int)
    if sorted(set(labels.tolist())) != [0, 1]:
        raise DeepFusionV2Error("Training requires both labels 0 and 1.")
    features = train.copy()
    for col in spec.numeric_features:
        if col not in features.columns:
            features[col] = np.nan
    for col in spec.categorical_features:
        if col not in features.columns:
            features[col] = "__missing__"
    model = build_model(spec.numeric_features, spec.categorical_features, group_weights=spec.group_weights)
    model.fit(features[_model_columns(spec)], labels.to_numpy(dtype=int))
    return model


def predict_variant_scores(model: Pipeline, table: pd.DataFrame, spec: VariantSpec) -> np.ndarray:
    features = table.copy()
    for col in spec.numeric_features:
        if col not in features.columns:
            features[col] = np.nan
    for col in spec.categorical_features:
        if col not in features.columns:
            features[col] = "__missing__"
    classifier = model.named_steps.get("logistic_regression")
    classes = list(getattr(classifier, "classes_", []))
    if 1 not in classes:
        raise DeepFusionV2Error("Logistic regression was not fit with positive class label 1.")
    positive_idx = classes.index(1)
    return model.predict_proba(features[_model_columns(spec)])[:, positive_idx].astype(float)


def _labels_from_table(table: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(table["label"], errors="raise").astype(int).to_numpy()


def compute_auto_group_weights(
    *,
    train: pd.DataFrame,
    eval_tables: Mapping[tuple[str, str], pd.DataFrame],
    metric: str = "auc",
    target_far: float = 0.01,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Estimate method-level weights from VAL only, never from TEST.

    Each method group is first trained as an isolated train-only logistic model,
    using only that group's numeric features. Its VAL performance is then used as
    the relative group strength and normalized to sum to 1.0.
    """

    metric_name = str(metric).strip().lower()
    if metric_name not in {"auc", "tar_at_far", "eer_complement"}:
        raise DeepFusionV2Error("auto group-weight metric must be one of: auc, tar_at_far, eer_complement")
    val_tables = {key: table for key, table in eval_tables.items() if str(key[1]).lower() == "val"}
    if not val_tables:
        raise DeepFusionV2Error("Automatic group weights require VAL tables; no val split was loaded.")

    raw_scores: dict[str, float] = {}
    diagnostics: dict[str, Any] = {
        "mode": "auto_val",
        "metric": metric_name,
        "target_far": float(target_far),
        "test_used_for_weight_selection": False,
        "groups": {},
    }
    for group, columns in GROUP_FEATURE_MAP.items():
        if not columns:
            raw_scores[group] = 0.0
            continue
        model = build_model(columns, tuple())
        model.fit(train[list(columns)], _labels_from_table(train))
        labels_parts: list[np.ndarray] = []
        score_parts: list[np.ndarray] = []
        for (_, split), table in sorted(val_tables.items()):
            if str(split).lower() != "val":
                continue
            cur = table.copy()
            for column in columns:
                if column not in cur.columns:
                    cur[column] = np.nan
            classifier = model.named_steps.get("logistic_regression")
            classes = list(getattr(classifier, "classes_", []))
            positive_idx = classes.index(1)
            labels_parts.append(_labels_from_table(cur))
            score_parts.append(model.predict_proba(cur[list(columns)])[:, positive_idx].astype(float))
        labels = np.concatenate(labels_parts) if labels_parts else np.array([], dtype=int)
        scores = np.concatenate(score_parts) if score_parts else np.array([], dtype=float)
        finite = np.isfinite(scores)
        labels = labels[finite]
        scores = scores[finite]
        if labels.size == 0 or sorted(set(labels.tolist())) != [0, 1]:
            value = 0.0
        elif metric_name == "auc":
            value = float(roc_auc_score(labels, scores))
        elif metric_name == "eer_complement":
            value = float(1.0 - eer_from_scores(labels, scores))
        else:
            threshold, _, _ = select_threshold_from_val_negatives(labels, scores, float(target_far))
            cm = confusion_metrics(labels, scores, threshold)
            value = float(cm["TAR"])
        if not math.isfinite(value):
            value = 0.0
        raw_scores[group] = max(0.0, value)
        diagnostics["groups"][group] = {
            "features": list(columns),
            "raw_metric_value": float(raw_scores[group]),
            "val_pairs": int(labels.size),
            "val_positives": int(np.sum(labels == 1)) if labels.size else 0,
            "val_negatives": int(np.sum(labels == 0)) if labels.size else 0,
        }

    if sum(raw_scores.values()) <= 0:
        weights = normalize_group_weights(DEFAULT_GROUP_WEIGHTS)
        diagnostics["fallback"] = "default_group_weights"
    else:
        weights = normalize_group_weights(raw_scores)
    diagnostics["weights"] = weights
    diagnostics["weights_percent"] = {key: float(100.0 * value) for key, value in weights.items()}
    return weights, diagnostics


def select_threshold_from_val_negatives(labels: np.ndarray, scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    labels = labels.astype(int)
    negatives = scores[labels == 0]
    negatives = negatives[np.isfinite(negatives)]
    if negatives.size == 0:
        return float("nan"), 0, float("nan")
    for threshold in sorted(float(value) for value in np.unique(negatives)):
        false_accepts = int(np.sum(negatives >= threshold))
        actual_far = float(false_accepts / negatives.size)
        if actual_far <= float(target_far) + 1e-15:
            return float(threshold), false_accepts, actual_far
    return float(np.nextafter(np.max(negatives), math.inf)), 0, 0.0


def eer_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels.astype(int), scores.astype(float))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def confusion_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = labels.astype(int)
    scores = scores.astype(float)
    pred = scores >= float(threshold)
    pos = labels == 1
    neg = labels == 0
    ta = int(np.sum(pred & pos))
    fr = int(np.sum((~pred) & pos))
    fa = int(np.sum(pred & neg))
    tr = int(np.sum((~pred) & neg))
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    return {
        "n_pairs": int(labels.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "TA": ta,
        "FR": fr,
        "FA": fa,
        "TR": tr,
        "TAR": float(ta / n_pos) if n_pos else float("nan"),
        "FAR": float(fa / n_neg) if n_neg else float("nan"),
        "FRR": float(fr / n_pos) if n_pos else float("nan"),
        "auc": float(roc_auc_score(labels, scores)) if n_pos and n_neg else float("nan"),
        "eer": eer_from_scores(labels, scores) if n_pos and n_neg else float("nan"),
    }


def score_output_table(table: pd.DataFrame, scores: np.ndarray, *, method: str, model_path: str) -> pd.DataFrame:
    keep = [
        "dataset", "split", "pair_id", "label", "subject_a", "subject_b", "finger_position", "frgp", "path_a", "path_b",
        "sourceafis_score", "sift_score", "sift_inliers", "sift_matches", "deep_score", "deep_logit",
    ]
    for col in keep:
        if col not in table.columns:
            table[col] = np.nan
    out = table[keep].copy()
    out.insert(0, "method", method)
    out["score"] = scores.astype(float)
    out["score_semantics"] = "logistic_regression_positive_class_probability"
    out["higher_is_more_similar"] = True
    out["model_path"] = model_path
    return out


def save_model_bundle(model: Pipeline, spec: VariantSpec, model_dir: Path, *, train_rows: int, train_label_counts: Mapping[str, int]) -> dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fusion_model.joblib"
    manifest_path = model_dir / "feature_manifest.json"
    joblib.dump(model, model_path)
    manifest = {
        "schema_version": "sourceafis_sift_quality_deep_fusion_v2_model_bundle_v1",
        "method": spec.name,
        "description": spec.description,
        "created_at": utc_now(),
        "fit_split": "train",
        "train_rows": int(train_rows),
        "train_label_counts": dict(train_label_counts),
        "numeric_features": list(spec.numeric_features),
        "categorical_features": list(spec.categorical_features),
        "include_quality": bool(spec.include_quality),
        "group_weights": dict(spec.group_weights or {}),
        "group_weights_percent": {key: 100.0 * float(value) for key, value in dict(spec.group_weights or {}).items()},
        "group_weight_mode": spec.group_weight_mode,
        "group_weight_metric": spec.group_weight_metric,
        "test_used_for_training": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "manifest_path": str(manifest_path)}


def render_summary(
    *,
    title: str,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    manifest: Mapping[str, Any],
    comparison: pd.DataFrame | None = None,
) -> str:
    text = f"# {title}\n\n"
    text += f"Created: `{manifest.get('created_at', '')}`\n\n"
    text += "Protocol: fit on TRAIN only; select thresholds from VAL negatives only; apply frozen thresholds to TEST.\n\n"
    text += "## Metrics\n\n"
    text += metrics.to_markdown(index=False)
    text += "\n\n## Thresholds\n\n"
    text += thresholds.to_markdown(index=False)
    if comparison is not None and not comparison.empty:
        text += "\n\n## Comparison vs Fusion v1\n\n"
        text += comparison.to_markdown(index=False)
    text += "\n"
    return text


def build_comparison_to_baseline(repo_root: Path, metrics: pd.DataFrame, *, baseline_method: str = "sourceafis_sift_quality_fusion_v1") -> pd.DataFrame:
    baseline_path = (
        repo_root
        / "artifacts/reports/benchmark/plain_roll_final_fusion_v1_v2_anatomical_full_pairs/plain_roll_final_metrics.csv"
    )
    if not baseline_path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.rename(columns={c: c.upper() for c in baseline.columns if c.upper() in {"TAR", "FAR", "TA", "FA"}})
    base = baseline[baseline["split"].astype(str).str.lower() == "test"].copy()
    current = metrics[metrics["split"].astype(str).str.lower() == "test"].copy()
    rows = []
    for _, row in current.iterrows():
        match = base[
            (base["dataset"].astype(str) == str(row["dataset"]))
            & (pd.to_numeric(base["target_far"], errors="coerce").round(12) == round(float(row["target_far"]), 12))
        ]
        if match.empty:
            continue
        b = match.iloc[0]
        rows.append(
            {
                "dataset": row["dataset"],
                "target_far": float(row["target_far"]),
                "baseline_method": baseline_method,
                "baseline_TAR": float(b["TAR"]),
                "current_method": row["method"],
                "current_TAR": float(row["TAR"]),
                "delta_TAR_pp": 100.0 * (float(row["TAR"]) - float(b["TAR"])),
                "baseline_FAR": float(b["FAR"]),
                "current_FAR": float(row["FAR"]),
                "delta_FA": int(row["FA"]) - int(b["FA"]),
                "delta_TA": int(row["TA"]) - int(b["TA"]),
            }
        )
    return pd.DataFrame(rows)


def run_variants(
    *,
    repo_root: Path,
    outdir: Path,
    datasets: Iterable[str],
    splits: Iterable[str],
    target_fars: Iterable[float],
    variants: Iterable[str],
    include_quality_override: bool | None = None,
    save_training_table: bool = False,
    group_weights: Mapping[str, float] | None = None,
    auto_group_weights: bool = False,
    group_weight_metric: str = "auc",
    group_weight_target_far: float = 0.01,
) -> dict[str, pd.DataFrame]:
    datasets = tuple(datasets)
    splits = tuple(splits)
    target_fars = tuple(float(x) for x in target_fars)
    variant_names = tuple(variants)
    specs = [VARIANTS[name] for name in variant_names]
    need_group_weights = GROUP_WEIGHTED_METHOD in variant_names
    need_quality = any(spec.include_quality for spec in specs)
    if include_quality_override is not None:
        need_quality = bool(include_quality_override)

    outdir.mkdir(parents=True, exist_ok=True)
    scores_dir = outdir / "scores"
    model_root = outdir / "model"
    scores_dir.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    train_frames = []
    eval_tables: dict[tuple[str, str], pd.DataFrame] = {}
    pair_bundles: list[dict[str, Any]] = []
    source_cache: dict[str, pd.DataFrame] = {}
    print("[load] train/eval base tables")
    for dataset in datasets:
        train_df = load_train_dataset(repo_root, dataset)
        train_frames.append(train_df)
        train_bundle = pair_bundle_metadata(repo_root, dataset, "train")
        if train_bundle is not None:
            pair_bundles.append(train_bundle)
        print(f"  {dataset}/train rows={len(train_df)} labels={train_df['label'].value_counts().sort_index().to_dict()}")
        for split in splits:
            eval_tables[(dataset, split)] = load_eval_dataset(repo_root, dataset, split, source_cache)
            eval_bundle = pair_bundle_metadata(repo_root, dataset, split)
            if eval_bundle is not None:
                pair_bundles.append(eval_bundle)
            print(f"  {dataset}/{split} rows={len(eval_tables[(dataset, split)])}")
    train = pd.concat(train_frames, ignore_index=True, sort=False)

    if need_quality:
        print("[quality] extracting deterministic image-quality features")
        train = add_quality_features(train, repo_root=repo_root)
        eval_tables = ensure_quality_if_needed(eval_tables, repo_root=repo_root, enabled=True)

    group_weight_diagnostics: dict[str, Any] | None = None
    if need_group_weights:
        if auto_group_weights:
            print(f"[group-weights] estimating automatic weights from VAL using metric={group_weight_metric}")
            resolved_group_weights, group_weight_diagnostics = compute_auto_group_weights(
                train=train,
                eval_tables=eval_tables,
                metric=group_weight_metric,
                target_far=float(group_weight_target_far),
            )
            group_weight_mode = "auto_val"
        else:
            resolved_group_weights = normalize_group_weights(group_weights)
            group_weight_mode = "manual" if group_weights is not None else "default_manual"
            group_weight_diagnostics = {
                "mode": group_weight_mode,
                "metric": None,
                "target_far": None,
                "test_used_for_weight_selection": False,
                "weights": resolved_group_weights,
                "weights_percent": {key: 100.0 * value for key, value in resolved_group_weights.items()},
            }
        specs = [
            replace(
                spec,
                group_weights=resolved_group_weights,
                group_weight_mode=group_weight_mode,
                group_weight_metric=str(group_weight_metric) if auto_group_weights else None,
            )
            if spec.name == GROUP_WEIGHTED_METHOD
            else spec
            for spec in specs
        ]
        (model_root / "group_weights.json").write_text(
            json.dumps(group_weight_diagnostics, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"[group-weights] {group_weight_diagnostics['weights_percent']}")

    all_metrics: list[pd.DataFrame] = []
    all_thresholds: list[pd.DataFrame] = []
    official_metrics: pd.DataFrame | None = None
    official_thresholds: pd.DataFrame | None = None

    train_counts = {str(k): int(v) for k, v in train["label"].value_counts().sort_index().to_dict().items()}
    if save_training_table:
        train.to_csv(model_root / "training_feature_table.csv", index=False)

    for spec in specs:
        print(f"[fit] {spec.name}")
        model = fit_variant_model(train, spec)
        bundle = save_model_bundle(model, spec, model_root / spec.name, train_rows=len(train), train_label_counts=train_counts)
        variant_score_frames: list[pd.DataFrame] = []
        score_lookup: dict[tuple[str, str], str] = {}

        for dataset in datasets:
            for split in splits:
                table = eval_tables[(dataset, split)]
                scores = predict_variant_scores(model, table, spec)
                out_scores = score_output_table(table.copy(), scores, method=spec.name, model_path=bundle["model_path"])
                subdir = scores_dir if spec.is_official else scores_dir / "ablation"
                subdir.mkdir(parents=True, exist_ok=True)
                out_csv = subdir / f"scores_{dataset}_{spec.name}_{split}.csv"
                out_scores.to_csv(out_csv, index=False)
                score_lookup[(dataset, split)] = str(out_csv)
                variant_score_frames.append(out_scores)
                print(f"  [score] {dataset}/{split}: {out_csv} rows={len(out_scores)}")

        all_scores = pd.concat(variant_score_frames, ignore_index=True, sort=False)
        thresholds_rows = []
        metrics_rows = []
        for dataset in datasets:
            val = all_scores[(all_scores["dataset"] == dataset) & (all_scores["split"] == "val")].copy()
            if val.empty:
                raise DeepFusionV2Error(f"Cannot select threshold for {dataset}: VAL scores missing")
            val_labels = pd.to_numeric(val["label"], errors="raise").astype(int).to_numpy()
            val_scores = pd.to_numeric(val["score"], errors="raise").astype(float).to_numpy()
            for target_far in target_fars:
                threshold, cal_fa, cal_far = select_threshold_from_val_negatives(val_labels, val_scores, target_far)
                thresholds_rows.append(
                    {
                        "method": spec.name,
                        "dataset": dataset,
                        "target_far": float(target_far),
                        "threshold": float(threshold),
                        "calibration_split": "val",
                        "calibration_negatives": int(np.sum(val_labels == 0)),
                        "calibration_positives": int(np.sum(val_labels == 1)),
                        "calibration_false_accepts": int(cal_fa),
                        "calibration_far": float(cal_far),
                        "selection_rule": "lowest VAL negative-score threshold with VAL FAR <= target",
                        "higher_is_more_similar": True,
                        "scores_csv": score_lookup.get((dataset, "val"), ""),
                    }
                )
                for split in splits:
                    cur = all_scores[(all_scores["dataset"] == dataset) & (all_scores["split"] == split)].copy()
                    labels = pd.to_numeric(cur["label"], errors="raise").astype(int).to_numpy()
                    values = pd.to_numeric(cur["score"], errors="raise").astype(float).to_numpy()
                    metrics_rows.append(
                        {
                            "method": spec.name,
                            "dataset": dataset,
                            "split": split,
                            "target_far": float(target_far),
                            "threshold": float(threshold),
                            **confusion_metrics(labels, values, threshold),
                            "scores_csv": score_lookup[(dataset, split)],
                            "model_path": bundle["model_path"],
                        }
                    )
        mdf = pd.DataFrame(metrics_rows)
        tdf = pd.DataFrame(thresholds_rows)
        all_metrics.append(mdf)
        all_thresholds.append(tdf)
        if spec.is_official:
            official_metrics = mdf.copy()
            official_thresholds = tdf.copy()

    metrics_all = pd.concat(all_metrics, ignore_index=True, sort=False)
    thresholds_all = pd.concat(all_thresholds, ignore_index=True, sort=False)
    official_metrics = official_metrics if official_metrics is not None else metrics_all[metrics_all["method"] == variant_names[-1]].copy()
    official_thresholds = official_thresholds if official_thresholds is not None else thresholds_all[thresholds_all["method"] == variant_names[-1]].copy()

    metrics_path = outdir / "plain_roll_final_metrics.csv"
    thresholds_path = outdir / "plain_roll_final_thresholds.csv"
    manifest_path = outdir / "plain_roll_final_manifest.json"
    comparison_path = outdir / "plain_roll_final_statistical_comparison.csv"
    summary_path = outdir / "plain_roll_final_summary.md"
    ablation_metrics_path = outdir / "ablation_metrics.csv"
    ablation_summary_path = outdir / "ablation_summary.md"

    official_metrics.to_csv(metrics_path, index=False)
    official_thresholds.to_csv(thresholds_path, index=False)
    metrics_all.to_csv(ablation_metrics_path, index=False)
    comparison = build_comparison_to_baseline(repo_root, official_metrics)
    comparison.to_csv(comparison_path, index=False)
    manifest = {
        "schema_version": "sourceafis_sift_quality_deep_fusion_v2_benchmark_v1",
        "method": str(official_metrics["method"].iloc[0]) if not official_metrics.empty and "method" in official_metrics.columns else METHOD_NAME,
        "created_at": utc_now(),
        "repo_root": str(repo_root),
        "datasets": list(datasets),
        "splits": list(splits),
        "target_fars": list(target_fars),
        "fit_splits": ["train"],
        "threshold_calibration_split": "val",
        "test_used_for_training": False,
        "pair_bundles": pair_bundles,
        "variants": [spec.name for spec in specs],
        "official_numeric_features": list(specs[-1].numeric_features),
        "official_categorical_features": list(specs[-1].categorical_features),
        "include_quality": bool(need_quality),
        "group_weight_diagnostics": group_weight_diagnostics,
        "metrics_csv": str(metrics_path),
        "thresholds_csv": str(thresholds_path),
        "comparison_csv": str(comparison_path),
        "ablation_metrics_csv": str(ablation_metrics_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(
        render_summary(title=f"{METHOD_NAME} benchmark", metrics=official_metrics, thresholds=official_thresholds, manifest=manifest, comparison=comparison),
        encoding="utf-8",
    )
    ablation_summary_path.write_text(
        render_summary(title="Fusion v2 ablation benchmark", metrics=metrics_all, thresholds=thresholds_all, manifest=manifest),
        encoding="utf-8",
    )
    return {
        "metrics": official_metrics,
        "thresholds": official_thresholds,
        "comparison": comparison,
        "ablation_metrics": metrics_all,
        "ablation_thresholds": thresholds_all,
    }
