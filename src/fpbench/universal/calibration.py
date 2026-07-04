from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fusion_features import METHOD_NAME, default_categorical_feature_columns, default_numeric_feature_columns


MODEL_SCHEMA_VERSION = "sourceafis_sift_quality_fusion_model_v1"
DEFAULT_TARGET_FARS = (0.005, 0.01)
MISSING_CATEGORY = "__missing__"

THRESHOLD_COLUMNS = [
    "method",
    "dataset",
    "target_far",
    "threshold",
    "calibration_split",
    "calibration_negative_count",
    "calibration_positive_count",
    "calibration_false_accepts",
    "calibration_far",
    "enough_negatives_for_target",
    "minimum_negatives_for_target",
    "selection_rule",
    "higher_is_more_similar",
    "scores_csv",
]


class FusionCalibrationError(ValueError):
    """Raised for fusion model training or calibration protocol violations."""


@dataclass(frozen=True)
class ThresholdSelection:
    target_far: float
    threshold: float
    false_accepts: int
    actual_far: float
    negative_count: int
    positive_count: int
    enough_negatives_for_target: bool
    minimum_negatives_for_target: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _minimum_negatives_for_target(target_far: float) -> int:
    if target_far <= 0:
        return -1
    return int(math.ceil(1.0 / float(target_far)))


def _finite_scores(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def select_threshold_from_negative_scores(
    negative_scores: Any,
    *,
    target_far: float,
    positive_count: int = 0,
) -> ThresholdSelection:
    """Select a threshold using only validation negative scores."""

    target = float(target_far)
    if target < 0:
        raise FusionCalibrationError("target_far must be non-negative.")

    negatives = _finite_scores(negative_scores)
    negative_count = int(negatives.size)
    minimum_negatives = _minimum_negatives_for_target(target)
    enough_negatives = bool(negative_count >= minimum_negatives) if minimum_negatives > 0 else False

    if negative_count == 0:
        return ThresholdSelection(
            target_far=target,
            threshold=float("nan"),
            false_accepts=0,
            actual_far=float("nan"),
            negative_count=0,
            positive_count=int(positive_count),
            enough_negatives_for_target=False,
            minimum_negatives_for_target=minimum_negatives,
        )

    for threshold in sorted(float(value) for value in np.unique(negatives)):
        false_accepts = int(np.sum(negatives >= threshold))
        actual_far = float(false_accepts / negative_count)
        if actual_far <= target + 1e-15:
            return ThresholdSelection(
                target_far=target,
                threshold=float(threshold),
                false_accepts=false_accepts,
                actual_far=actual_far,
                negative_count=negative_count,
                positive_count=int(positive_count),
                enough_negatives_for_target=enough_negatives,
                minimum_negatives_for_target=minimum_negatives,
            )

    return ThresholdSelection(
        target_far=target,
        threshold=float(math.nextafter(float(np.max(negatives)), math.inf)),
        false_accepts=0,
        actual_far=0.0,
        negative_count=negative_count,
        positive_count=int(positive_count),
        enough_negatives_for_target=enough_negatives,
        minimum_negatives_for_target=minimum_negatives,
    )


def build_threshold_table_from_scores(
    scores: pd.DataFrame,
    target_fars: Iterable[float] = DEFAULT_TARGET_FARS,
    *,
    method: str = METHOD_NAME,
    scores_csv_lookup: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    required = {"dataset", "split", "label", "score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise FusionCalibrationError(f"scores table is missing required columns: {missing}")

    rows: list[dict[str, Any]] = []
    for dataset, dataset_scores in scores.groupby("dataset", sort=True):
        val = dataset_scores[dataset_scores["split"].astype(str).str.lower() == "val"].copy()
        labels = pd.to_numeric(val["label"], errors="coerce").fillna(-1).astype(int)
        score_values = pd.to_numeric(val["score"], errors="coerce")
        negatives = score_values[labels == 0]
        positive_count = int((labels == 1).sum())
        for target_far in target_fars:
            selection = select_threshold_from_negative_scores(
                negatives,
                target_far=float(target_far),
                positive_count=positive_count,
            )
            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "target_far": float(target_far),
                    "threshold": float(selection.threshold),
                    "calibration_split": "val",
                    "calibration_negative_count": int(selection.negative_count),
                    "calibration_positive_count": int(selection.positive_count),
                    "calibration_false_accepts": int(selection.false_accepts),
                    "calibration_far": float(selection.actual_far),
                    "enough_negatives_for_target": bool(selection.enough_negatives_for_target),
                    "minimum_negatives_for_target": int(selection.minimum_negatives_for_target),
                    "selection_rule": "lowest VAL negative-score threshold with VAL FAR <= target",
                    "higher_is_more_similar": True,
                    "scores_csv": (scores_csv_lookup or {}).get((str(dataset), "val"), ""),
                }
            )
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def assert_train_only(table: pd.DataFrame) -> None:
    if "split" not in table.columns:
        raise FusionCalibrationError("feature table is missing split column.")
    splits = sorted(set(table["split"].astype(str).str.lower()))
    disallowed = [split for split in splits if split != "train"]
    if disallowed:
        raise FusionCalibrationError(
            f"Fusion model fitting may use train rows only; found non-train split(s): {disallowed}"
        )


def _category_values(series: pd.Series) -> list[str]:
    values = series.fillna(MISSING_CATEGORY).astype(str).str.strip()
    values = values.mask(values == "", MISSING_CATEGORY)
    return sorted(set(values.tolist()))


def fit_feature_schema(
    table: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Any]:
    numeric = numeric_features or default_numeric_feature_columns(table)
    categorical = categorical_features or default_categorical_feature_columns(table)
    numeric = [column for column in numeric if column in table.columns]
    categorical = [column for column in categorical if column in table.columns]

    fill_values: dict[str, float] = {}
    for column in numeric:
        values = pd.to_numeric(table[column], errors="coerce")
        finite = values[np.isfinite(values)]
        fill_values[column] = float(finite.median()) if len(finite) else 0.0

    levels = {column: _category_values(table[column]) for column in categorical}
    model_features = list(numeric)
    for column in categorical:
        model_features.extend(f"cat__{column}={value}" for value in levels[column])

    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "categorical_levels": levels,
        "fill_values": fill_values,
        "model_features": model_features,
    }


def transform_features(table: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for column in schema.get("numeric_features", []):
        fill_value = float(schema.get("fill_values", {}).get(column, 0.0))
        if column in table.columns:
            values = pd.to_numeric(table[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill_value)
        else:
            values = pd.Series(fill_value, index=table.index, dtype=float)
        frames.append(pd.DataFrame({column: values.astype(float)}))

    for column in schema.get("categorical_features", []):
        if column in table.columns:
            values = table[column].fillna(MISSING_CATEGORY).astype(str).str.strip()
            values = values.mask(values == "", MISSING_CATEGORY)
        else:
            values = pd.Series(MISSING_CATEGORY, index=table.index)
        for level in schema.get("categorical_levels", {}).get(column, []):
            frames.append(pd.DataFrame({f"cat__{column}={level}": (values == str(level)).astype(float)}))

    if frames:
        matrix = pd.concat(frames, axis=1)
    else:
        matrix = pd.DataFrame(index=table.index)
    for column in schema.get("model_features", []):
        if column not in matrix.columns:
            matrix[column] = 0.0
    return matrix[schema.get("model_features", [])].astype(float)


def fit_fusion_model(
    table: pd.DataFrame,
    *,
    random_state: int = 13,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[Pipeline, dict[str, Any]]:
    assert_train_only(table)
    if "label" not in table.columns:
        raise FusionCalibrationError("feature table is missing label column.")
    labels = pd.to_numeric(table["label"], errors="coerce").fillna(-1).astype(int)
    if sorted(set(labels.tolist())) != [0, 1]:
        raise FusionCalibrationError("Fusion model training requires both label classes 0 and 1.")

    schema = fit_feature_schema(
        table,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
    matrix = transform_features(table, schema)
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(matrix, labels.to_numpy(dtype=int))
    return model, schema


def predict_fusion_scores(model: Pipeline, schema: dict[str, Any], table: pd.DataFrame) -> np.ndarray:
    matrix = transform_features(table, schema)
    classifier = model.named_steps.get("logistic_regression")
    if classifier is None or not hasattr(model, "predict_proba"):
        raise FusionCalibrationError("Loaded fusion model does not expose predict_proba.")
    classes = list(getattr(classifier, "classes_", []))
    if 1 not in classes:
        raise FusionCalibrationError("Loaded fusion model was not fit with positive class label 1.")
    positive_index = classes.index(1)
    return model.predict_proba(matrix)[:, positive_index].astype(float)


def save_model_bundle(
    *,
    model: Pipeline,
    schema: dict[str, Any],
    model_dir: str | Path,
    training_manifest: dict[str, Any],
) -> dict[str, Path]:
    output = Path(model_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "fusion_model.joblib"
    schema_path = output / "feature_schema.json"
    manifest_path = output / "training_manifest.json"
    joblib.dump(model, model_path)
    schema_path.write_text(json.dumps(schema, indent=2, ensure_ascii=True), encoding="utf-8")
    manifest = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "created_at": _utc_now(),
        **training_manifest,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "model": model_path,
        "feature_schema": schema_path,
        "training_manifest": manifest_path,
    }


def load_model_bundle(model_dir: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    root = Path(model_dir)
    model = joblib.load(root / "fusion_model.joblib")
    schema = json.loads((root / "feature_schema.json").read_text(encoding="utf-8"))
    if schema.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise FusionCalibrationError(
            f"Unexpected feature schema version: {schema.get('schema_version')!r}"
        )
    return model, schema
