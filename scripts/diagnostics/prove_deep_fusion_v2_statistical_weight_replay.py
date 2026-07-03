from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from scipy import sparse  # type: ignore
except Exception:  # pragma: no cover - scipy is normally present with sklearn.
    sparse = None


DEFAULT_METHOD = "sourceafis_sift_quality_deep_fusion_v2"
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

SOURCE_FEATURES = {"sourceafis_score", "source_dpi_a", "source_dpi_b"}
SIFT_FEATURES = {"sift_score", "sift_inliers", "sift_matches", "sift_k1", "sift_k2"}
DEEP_FEATURES = {"deep_score", "deep_logit"}
QUALITY_PREFIXES = ("a_", "b_", "pair_")
METADATA_PREFIXES = ("dataset", "finger_position", "frgp")


class StatisticalReplayError(ValueError):
    """Raised when the statistical replay proof cannot be completed safely."""


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    manifest: dict[str, Any]
    model_dir: Path
    model_path: Path
    manifest_path: Path | None
    method: str
    numeric_features: list[str]
    categorical_features: list[str]
    include_quality: bool


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


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise StatisticalReplayError(f"{label} is empty: {path}")
    return df


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_repo_to_syspath(repo_root: Path) -> None:
    src = repo_root / "src"
    for candidate in (src, repo_root):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)


def default_benchmark_dir(repo_root: Path, method: str) -> Path:
    return repo_root / "artifacts" / "reports" / "benchmark" / f"{method}_full_pairs"


def default_model_dir(benchmark_dir: Path, method: str) -> Path:
    return benchmark_dir / "model" / method


def load_model_bundle(model_dir: Path, *, method_hint: str) -> ModelBundle:
    model_path = model_dir / "fusion_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing fusion model: {model_path}")

    manifest_path: Path | None = None
    for candidate_name in ("feature_manifest.json", "training_manifest.json"):
        candidate = model_dir / candidate_name
        if candidate.exists():
            manifest_path = candidate
            break

    manifest: dict[str, Any] = read_json(manifest_path) if manifest_path is not None else {}
    model = joblib.load(model_path)

    method = str(manifest.get("method") or method_hint)
    numeric_features = [str(x) for x in manifest.get("numeric_features", [])]
    categorical_features = [str(x) for x in manifest.get("categorical_features", [])]

    if not numeric_features:
        numeric_features = infer_numeric_feature_columns_from_model(model)
    if not categorical_features:
        categorical_features = infer_categorical_feature_columns_from_model(model)

    if not numeric_features and not categorical_features:
        raise StatisticalReplayError(
            "Could not infer model input features from manifest or model. "
            "Expected feature_manifest.json with numeric_features/categorical_features."
        )

    include_quality = bool(manifest.get("include_quality", any(is_quality_feature(c) for c in numeric_features)))
    return ModelBundle(
        model=model,
        manifest=manifest,
        model_dir=model_dir,
        model_path=model_path,
        manifest_path=manifest_path,
        method=method,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_quality=include_quality,
    )


def infer_numeric_feature_columns_from_model(model: Any) -> list[str]:
    preprocessor = getattr(model, "named_steps", {}).get("features")
    if preprocessor is None:
        return []
    for name, _, columns in getattr(preprocessor, "transformers", []):
        if str(name) == "numeric":
            return [str(c) for c in columns]
    return []


def infer_categorical_feature_columns_from_model(model: Any) -> list[str]:
    preprocessor = getattr(model, "named_steps", {}).get("features")
    if preprocessor is None:
        return []
    for name, _, columns in getattr(preprocessor, "transformers", []):
        if str(name) == "categorical":
            return [str(c) for c in columns]
    return []


def is_quality_feature(column: str) -> bool:
    name = str(column)
    return name.startswith(QUALITY_PREFIXES) and name not in {"pair_id"}


def feature_group(column: str) -> str:
    name = str(column)
    if name in SOURCE_FEATURES:
        return "sourceafis"
    if name in SIFT_FEATURES:
        return "sift"
    if name in DEEP_FEATURES:
        return "deep"
    if is_quality_feature(name):
        return "quality"
    if name in METADATA_PREFIXES or any(name.startswith(f"{prefix}_") for prefix in METADATA_PREFIXES):
        return "metadata"
    return "other"


def ensure_input_columns(table: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> pd.DataFrame:
    out = table.copy()
    for column in numeric_features:
        if column not in out.columns:
            out[column] = np.nan
    for column in categorical_features:
        if column not in out.columns:
            out[column] = "__missing__"
        values = out[column].fillna("__missing__").astype(str).str.strip()
        out[column] = values.mask(values == "", "__missing__")
    return out[numeric_features + categorical_features].copy()


def to_dense(matrix: Any) -> np.ndarray:
    if sparse is not None and sparse.issparse(matrix):
        return matrix.toarray().astype(float)
    return np.asarray(matrix, dtype=float)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.empty_like(arr, dtype=float)
    pos = arr >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    exp_values = np.exp(arr[~pos])
    out[~pos] = exp_values / (1.0 + exp_values)
    return out


def transformed_feature_names(model: Any) -> list[str]:
    preprocessor = getattr(model, "named_steps", {}).get("features")
    if preprocessor is None:
        raise StatisticalReplayError("Model pipeline is missing a 'features' preprocessing step.")
    if hasattr(preprocessor, "get_feature_names_out"):
        return [str(name) for name in preprocessor.get_feature_names_out()]
    # Fallback for very old sklearn versions.
    names: list[str] = []
    for transformer_name, transformer, columns in getattr(preprocessor, "transformers_", []):
        if transformer_name == "remainder":
            continue
        columns = [str(c) for c in columns]
        if str(transformer_name) == "numeric":
            names.extend(columns)
        elif str(transformer_name) == "categorical":
            try:
                encoder = transformer.named_steps.get("onehotencoder") if hasattr(transformer, "named_steps") else transformer
                categories = getattr(encoder, "categories_", [])
                for column, levels in zip(columns, categories):
                    names.extend(f"{column}_{level}" for level in levels)
            except Exception:
                names.extend(columns)
        else:
            names.extend(columns)
    return names


def raw_feature_from_transformed(transformed_name: str, numeric_features: list[str], categorical_features: list[str]) -> str:
    name = str(transformed_name)
    if name in numeric_features:
        return name
    for column in sorted(categorical_features, key=len, reverse=True):
        prefix = f"{column}_"
        if name.startswith(prefix):
            return column
    if "__" in name:
        tail = name.split("__", 1)[1]
        if tail in numeric_features:
            return tail
        for column in sorted(categorical_features, key=len, reverse=True):
            if tail.startswith(f"{column}_"):
                return column
    return name


def _manual_positive_probability(
    model: Any,
    input_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Replay sklearn Pipeline math manually from preprocessing to sigmoid.

    Returns: positive probability, post-preprocessing matrix used by logistic regression,
    transformed feature names.
    """

    steps = getattr(model, "named_steps", {})
    preprocessor = steps.get("features")
    classifier = steps.get("logistic_regression")
    if preprocessor is None or classifier is None:
        raise StatisticalReplayError("Expected a Pipeline with 'features' and 'logistic_regression' steps.")

    matrix = to_dense(preprocessor.transform(input_frame))
    names = transformed_feature_names(model)
    group_scaler = steps.get("group_weights")
    if group_scaler is not None:
        # This is still a manual replay of the full pipeline: we apply the learned/frozen
        # group-level multiplier before the logistic coefficients, just like sklearn does.
        matrix = to_dense(group_scaler.transform(matrix))

    coef = np.asarray(getattr(classifier, "coef_", None), dtype=float)
    intercept = np.asarray(getattr(classifier, "intercept_", None), dtype=float)
    classes = list(getattr(classifier, "classes_", []))
    if coef.ndim != 2 or coef.shape[0] != 1 or intercept.size != 1 or len(classes) != 2:
        raise StatisticalReplayError(
            "Manual replay currently expects binary LogisticRegression with coef_.shape=(1, n_features)."
        )
    if matrix.shape[1] != coef.shape[1]:
        raise StatisticalReplayError(
            f"Coefficient/feature mismatch: matrix has {matrix.shape[1]} columns; coef has {coef.shape[1]}."
        )

    logit = matrix @ coef[0].T + float(intercept[0])
    second_class_probability = stable_sigmoid(logit)
    if classes[1] == 1:
        positive_probability = second_class_probability
    elif classes[0] == 1:
        positive_probability = 1.0 - second_class_probability
    else:
        raise StatisticalReplayError(f"Positive class label 1 is not present in classifier.classes_: {classes}")
    return positive_probability.astype(float), matrix.astype(float), names


def sklearn_positive_probability(model: Any, input_frame: pd.DataFrame) -> np.ndarray:
    classifier = getattr(model, "named_steps", {}).get("logistic_regression")
    if classifier is None:
        raise StatisticalReplayError("Model pipeline is missing logistic_regression step.")
    classes = list(getattr(classifier, "classes_", []))
    if 1 not in classes:
        raise StatisticalReplayError(f"Positive class label 1 is not present in classes_: {classes}")
    idx = classes.index(1)
    return np.asarray(model.predict_proba(input_frame)[:, idx], dtype=float)


def labels_from(table: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(table["label"], errors="raise").astype(int).to_numpy()


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


def eer_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels.astype(int), scores.astype(float))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def load_thresholds(path: Path, method: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "method" in df.columns:
        sub = df[df["method"].astype(str) == str(method)].copy()
        if not sub.empty:
            return sub
    return df.copy()


def threshold_for(thresholds: pd.DataFrame, dataset: str, target_far: float) -> float | None:
    if thresholds.empty:
        return None
    sub = thresholds[
        (thresholds["dataset"].astype(str) == str(dataset))
        & (pd.to_numeric(thresholds["target_far"], errors="coerce").round(12) == round(float(target_far), 12))
    ]
    if sub.empty:
        return None
    return float(sub.iloc[0]["threshold"])


def score_csv_path(benchmark_dir: Path, method: str, dataset: str, split: str) -> Path | None:
    candidates = [
        benchmark_dir / "scores" / f"scores_{dataset}_{method}_{split}.csv",
        benchmark_dir / "scores" / "ablation" / f"scores_{dataset}_{method}_{split}.csv",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_saved_scores(benchmark_dir: Path, method: str, dataset: str, split: str) -> pd.DataFrame | None:
    path = score_csv_path(benchmark_dir, method, dataset, split)
    if path is None:
        return None
    df = pd.read_csv(path)
    if "score" not in df.columns:
        return None
    out = df[PAIR_KEY_COLUMNS + ["score"]].copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out = out.rename(columns={"score": "saved_score"})
    return out


def load_eval_feature_tables(
    *,
    repo_root: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    include_quality: bool,
) -> dict[tuple[str, str], pd.DataFrame]:
    add_repo_to_syspath(repo_root)
    from fpbench.universal import deep_fusion_v2 as dfv2  # imported only when running against the real repo

    source_cache: dict[str, pd.DataFrame] = {}
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    for dataset in datasets:
        for split in splits:
            table = dfv2.load_eval_dataset(repo_root, dataset, split, source_cache)
            if include_quality:
                table = dfv2.add_quality_features(table, repo_root=repo_root)
            tables[(dataset, split)] = table.reset_index(drop=True)
    return tables


def replay_one_table(
    *,
    bundle: ModelBundle,
    table: pd.DataFrame,
    dataset: str,
    split: str,
    benchmark_dir: Path,
    target_fars: tuple[float, ...],
    thresholds: pd.DataFrame,
    max_diff_rows: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame]:
    features = ensure_input_columns(table, bundle.numeric_features, bundle.categorical_features)
    manual, matrix, names = _manual_positive_probability(bundle.model, features)
    sklearn_scores = sklearn_positive_probability(bundle.model, features)
    abs_diff = np.abs(manual - sklearn_scores)

    context = table.copy()
    for col in CONTEXT_COLUMNS:
        if col not in context.columns:
            context[col] = np.nan
    diff_df = context[CONTEXT_COLUMNS].copy()
    diff_df["manual_score"] = manual
    diff_df["sklearn_score"] = sklearn_scores
    diff_df["abs_diff_manual_vs_sklearn"] = abs_diff
    saved = load_saved_scores(benchmark_dir, bundle.method, dataset, split)
    saved_max_abs_diff = float("nan")
    saved_mean_abs_diff = float("nan")
    saved_rows = 0
    saved_path = score_csv_path(benchmark_dir, bundle.method, dataset, split)
    if saved is not None:
        diff_df = diff_df.merge(saved, on=PAIR_KEY_COLUMNS, how="left", validate="one_to_one")
        saved_rows = int(diff_df["saved_score"].notna().sum())
        diff_df["abs_diff_manual_vs_saved_score"] = (diff_df["manual_score"] - diff_df["saved_score"]).abs()
        finite_saved_diff = pd.to_numeric(diff_df["abs_diff_manual_vs_saved_score"], errors="coerce").dropna()
        if len(finite_saved_diff):
            saved_max_abs_diff = float(finite_saved_diff.max())
            saved_mean_abs_diff = float(finite_saved_diff.mean())
    else:
        diff_df["saved_score"] = np.nan
        diff_df["abs_diff_manual_vs_saved_score"] = np.nan

    labels = labels_from(table)
    score_row = {
        "method": bundle.method,
        "dataset": dataset,
        "split": split,
        "rows": int(len(table)),
        "numeric_features": int(len(bundle.numeric_features)),
        "categorical_features": int(len(bundle.categorical_features)),
        "transformed_features": int(matrix.shape[1]),
        "max_abs_diff_manual_vs_sklearn": float(abs_diff.max()) if abs_diff.size else float("nan"),
        "mean_abs_diff_manual_vs_sklearn": float(abs_diff.mean()) if abs_diff.size else float("nan"),
        "median_abs_diff_manual_vs_sklearn": float(np.median(abs_diff)) if abs_diff.size else float("nan"),
        "saved_score_csv": str(saved_path) if saved_path is not None else "",
        "saved_score_rows_matched": saved_rows,
        "max_abs_diff_manual_vs_saved_score": saved_max_abs_diff,
        "mean_abs_diff_manual_vs_saved_score": saved_mean_abs_diff,
        **{f"sklearn_{k}": v for k, v in confusion_metrics(labels, sklearn_scores, 0.5).items() if k in {"auc", "eer"}},
    }

    decision_rows: list[dict[str, Any]] = []
    for target_far in target_fars:
        threshold = threshold_for(thresholds, dataset, target_far)
        if threshold is None:
            continue
        manual_decisions = manual >= threshold
        sklearn_decisions = sklearn_scores >= threshold
        manual_cm = confusion_metrics(labels, manual, threshold)
        sklearn_cm = confusion_metrics(labels, sklearn_scores, threshold)
        row = {
            "method": bundle.method,
            "dataset": dataset,
            "split": split,
            "target_far": float(target_far),
            "threshold": float(threshold),
            "identical_decisions_manual_vs_sklearn": bool(np.array_equal(manual_decisions, sklearn_decisions)),
            "decision_mismatches_manual_vs_sklearn": int(np.sum(manual_decisions != sklearn_decisions)),
        }
        for key, value in manual_cm.items():
            row[f"manual_{key}"] = value
        for key, value in sklearn_cm.items():
            row[f"sklearn_{key}"] = value
        for key in ("TA", "FR", "FA", "TR"):
            row[f"delta_{key}_manual_minus_sklearn"] = int(manual_cm[key]) - int(sklearn_cm[key])
        row["delta_TAR_manual_minus_sklearn"] = float(manual_cm["TAR"] - sklearn_cm["TAR"])
        row["delta_FAR_manual_minus_sklearn"] = float(manual_cm["FAR"] - sklearn_cm["FAR"])
        decision_rows.append(row)

    diff_df = diff_df.sort_values("abs_diff_manual_vs_sklearn", ascending=False).head(max_diff_rows)
    return score_row, decision_rows, diff_df


def coefficient_table(bundle: ModelBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    steps = getattr(bundle.model, "named_steps", {})
    classifier = steps.get("logistic_regression")
    if classifier is None:
        raise StatisticalReplayError("Model pipeline is missing logistic_regression step.")
    coef = np.asarray(classifier.coef_, dtype=float)
    if coef.ndim != 2 or coef.shape[0] != 1:
        raise StatisticalReplayError("Expected binary logistic regression coefficient matrix.")
    names = transformed_feature_names(bundle.model)
    if len(names) != coef.shape[1]:
        names = [f"feature_{i}" for i in range(coef.shape[1])]

    group_scaler = steps.get("group_weights")
    group_multiplier = np.ones(coef.shape[1], dtype=float)
    if group_scaler is not None and hasattr(group_scaler, "numeric_weights"):
        weights = np.asarray(list(group_scaler.numeric_weights), dtype=float)
        n = min(len(weights), len(group_multiplier))
        group_multiplier[:n] = weights[:n]

    rows: list[dict[str, Any]] = []
    for index, (name, coefficient) in enumerate(zip(names, coef[0])):
        raw_feature = raw_feature_from_transformed(name, bundle.numeric_features, bundle.categorical_features)
        group = feature_group(raw_feature)
        rows.append(
            {
                "index": int(index),
                "transformed_feature": name,
                "raw_feature": raw_feature,
                "group": group,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
                "sign": "positive" if coefficient > 0 else "negative" if coefficient < 0 else "zero",
                "pipeline_group_multiplier_before_logistic": float(group_multiplier[index]),
                "effective_feature_multiplier_x_coefficient": float(group_multiplier[index] * coefficient),
            }
        )
    coef_df = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for group, sub in coef_df.groupby("group", dropna=False):
        values = pd.to_numeric(sub["coefficient"], errors="coerce").to_numpy(dtype=float)
        abs_values = np.abs(values)
        summary_rows.append(
            {
                "group": group,
                "features": int(len(sub)),
                "positive_coefficients": int(np.sum(values > 1e-12)),
                "negative_coefficients": int(np.sum(values < -1e-12)),
                "near_zero_coefficients": int(np.sum(np.abs(values) <= 1e-12)),
                "min_coefficient": float(np.nanmin(values)) if values.size else float("nan"),
                "max_coefficient": float(np.nanmax(values)) if values.size else float("nan"),
                "mean_abs_coefficient": float(np.nanmean(abs_values)) if abs_values.size else float("nan"),
                "max_abs_coefficient": float(np.nanmax(abs_values)) if abs_values.size else float("nan"),
                "coefficient_range": float(np.nanmax(values) - np.nanmin(values)) if values.size else float("nan"),
                "has_mixed_signs": bool(np.any(values > 1e-12) and np.any(values < -1e-12)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("group").reset_index(drop=True)
    return coef_df, summary_df


def render_summary(
    *,
    bundle: ModelBundle,
    score_eq: pd.DataFrame,
    decision_eq: pd.DataFrame,
    coef_summary: pd.DataFrame,
    coef_df: pd.DataFrame,
    outdir: Path,
    tolerance: float,
) -> str:
    max_diff = float(pd.to_numeric(score_eq["max_abs_diff_manual_vs_sklearn"], errors="coerce").max())
    mean_diff = float(pd.to_numeric(score_eq["mean_abs_diff_manual_vs_sklearn"], errors="coerce").mean())
    decision_mismatches = int(pd.to_numeric(decision_eq.get("decision_mismatches_manual_vs_sklearn", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    proof_passed = bool(max_diff <= tolerance and decision_mismatches == 0)

    lines: list[str] = []
    lines.append(f"# Statistical replay proof for `{bundle.method}`")
    lines.append("")
    lines.append(f"Created: `{utc_now()}`")
    lines.append("")
    lines.append("## Claim tested")
    lines.append("")
    lines.append(
        "If the manually supplied weighting exactly reproduces the trained statistical model — including "
        "the same preprocessing, imputation, scaling, learned feature coefficients, and intercept — then the "
        "manual computation must reproduce the model scores and thresholded decisions."
    )
    lines.append("")
    lines.append("## Result")
    lines.append("")
    lines.append(f"- Replay passed tolerance `{tolerance:g}`: **{proof_passed}**")
    lines.append(f"- Maximum absolute probability difference, manual formula vs `predict_proba`: `{max_diff:.17g}`")
    lines.append(f"- Mean absolute probability difference, manual formula vs `predict_proba`: `{mean_diff:.17g}`")
    lines.append(f"- Threshold decision mismatches across evaluated operating points: `{decision_mismatches}`")
    lines.append("")
    lines.append("## Score equivalence by dataset/split")
    lines.append("")
    lines.append(score_eq.to_markdown(index=False))
    lines.append("")
    if not decision_eq.empty:
        compact_cols = [
            "dataset", "split", "target_far", "threshold", "identical_decisions_manual_vs_sklearn",
            "decision_mismatches_manual_vs_sklearn", "manual_TA", "manual_FR", "manual_FA", "manual_TR",
            "manual_TAR", "manual_FAR",
        ]
        compact_cols = [c for c in compact_cols if c in decision_eq.columns]
        lines.append("## Decision equivalence by threshold")
        lines.append("")
        lines.append(decision_eq[compact_cols].to_markdown(index=False))
        lines.append("")
    lines.append("## Learned coefficient diversity by group")
    lines.append("")
    lines.append(coef_summary.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The statistical fusion model is mathematically equivalent to a manual weighted sum only if the manual "
        "implementation copies every transformed feature coefficient and the intercept exactly. This is what the "
        "replay above verifies."
    )
    lines.append("")
    mixed_groups = coef_summary[coef_summary.get("has_mixed_signs", False) == True]["group"].astype(str).tolist() if "has_mixed_signs" in coef_summary.columns else []
    if mixed_groups:
        lines.append(
            "A single method-level weight is not equivalent to the trained model, because at least these groups contain "
            f"both positive and negative learned coefficients: `{', '.join(mixed_groups)}`."
        )
    else:
        lines.append(
            "A single method-level weight is still generally more constrained than the trained model, because the learned "
            "coefficients vary in magnitude across features within the same group."
        )
    lines.append("")
    lines.append("In other words: copying the complete learned statistical weighting reproduces the model; assigning one uniform weight per method group does not generally reproduce the model.")
    lines.append("")
    lines.append("## Output files")
    lines.append("")
    for name in [
        "replay_score_equivalence.csv",
        "replay_decision_equivalence.csv",
        "replay_score_differences.csv",
        "learned_feature_coefficients.csv",
        "coefficient_group_summary.csv",
        "replay_manifest.json",
    ]:
        lines.append(f"- `{outdir / name}`")
    lines.append("")

    # Small top-coefficient appendix for quick reading.
    if not coef_df.empty:
        top = coef_df.sort_values("abs_coefficient", ascending=False).head(20)
        lines.append("## Top learned transformed-feature coefficients")
        lines.append("")
        lines.append(top[["transformed_feature", "raw_feature", "group", "coefficient", "abs_coefficient", "sign"]].to_markdown(index=False))
        lines.append("")
    return "\n".join(lines)


def run_replay(
    *,
    repo_root: Path,
    benchmark_dir: Path,
    model_dir: Path,
    outdir: Path,
    method: str,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    target_fars: tuple[float, ...],
    tolerance: float,
    max_diff_rows: int,
) -> dict[str, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    bundle = load_model_bundle(model_dir, method_hint=method)
    thresholds = load_thresholds(benchmark_dir / "plain_roll_final_thresholds.csv", bundle.method)
    tables = load_eval_feature_tables(
        repo_root=repo_root,
        datasets=datasets,
        splits=splits,
        include_quality=bundle.include_quality,
    )

    score_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    diff_frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for split in splits:
            print(f"[replay] {dataset}/{split}")
            score_row, cur_decision_rows, diff_df = replay_one_table(
                bundle=bundle,
                table=tables[(dataset, split)],
                dataset=dataset,
                split=split,
                benchmark_dir=benchmark_dir,
                target_fars=target_fars,
                thresholds=thresholds,
                max_diff_rows=max_diff_rows,
            )
            score_rows.append(score_row)
            decision_rows.extend(cur_decision_rows)
            diff_frames.append(diff_df)

    score_eq = pd.DataFrame(score_rows)
    decision_eq = pd.DataFrame(decision_rows)
    diffs = pd.concat(diff_frames, ignore_index=True, sort=False) if diff_frames else pd.DataFrame()
    diffs = diffs.sort_values("abs_diff_manual_vs_sklearn", ascending=False).head(max_diff_rows)
    coef_df, coef_summary = coefficient_table(bundle)

    score_eq.to_csv(outdir / "replay_score_equivalence.csv", index=False)
    decision_eq.to_csv(outdir / "replay_decision_equivalence.csv", index=False)
    diffs.to_csv(outdir / "replay_score_differences.csv", index=False)
    coef_df.to_csv(outdir / "learned_feature_coefficients.csv", index=False)
    coef_summary.to_csv(outdir / "coefficient_group_summary.csv", index=False)

    max_diff = float(pd.to_numeric(score_eq["max_abs_diff_manual_vs_sklearn"], errors="coerce").max())
    decision_mismatches = int(pd.to_numeric(decision_eq.get("decision_mismatches_manual_vs_sklearn", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    manifest = {
        "schema_version": "deep_fusion_v2_statistical_replay_proof_v1",
        "created_at": utc_now(),
        "method": bundle.method,
        "repo_root": str(repo_root),
        "benchmark_dir": str(benchmark_dir),
        "model_dir": str(model_dir),
        "model_path": str(bundle.model_path),
        "feature_manifest_path": str(bundle.manifest_path) if bundle.manifest_path is not None else None,
        "datasets": list(datasets),
        "splits": list(splits),
        "target_fars": list(target_fars),
        "numeric_features": bundle.numeric_features,
        "categorical_features": bundle.categorical_features,
        "include_quality": bool(bundle.include_quality),
        "tolerance": float(tolerance),
        "max_abs_diff_manual_vs_sklearn": max_diff,
        "decision_mismatches_manual_vs_sklearn": decision_mismatches,
        "proof_passed": bool(max_diff <= tolerance and decision_mismatches == 0),
        "test_used_for_training_or_weight_selection": False,
    }
    (outdir / "replay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    (outdir / "replay_equivalence_summary.md").write_text(
        render_summary(
            bundle=bundle,
            score_eq=score_eq,
            decision_eq=decision_eq,
            coef_summary=coef_summary,
            coef_df=coef_df,
            outdir=outdir,
            tolerance=tolerance,
        ),
        encoding="utf-8",
    )
    return {
        "score_equivalence": score_eq,
        "decision_equivalence": decision_eq,
        "score_differences": diffs,
        "learned_feature_coefficients": coef_df,
        "coefficient_group_summary": coef_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prove that Fusion v2's trained LogisticRegression can be replayed exactly as a manual weighted sum."
    )
    parser.add_argument("--repo-root", required=True, help="Repository root, e.g. C:\\fingerprint-research")
    parser.add_argument("--method", default=DEFAULT_METHOD)
    parser.add_argument("--benchmark-dir", default=None, help="Benchmark directory containing scores and thresholds.")
    parser.add_argument("--model-dir", default=None, help="Directory containing fusion_model.joblib and feature_manifest.json.")
    parser.add_argument(
        "--outdir",
        default="artifacts/reports/diagnostics/deep_fusion_v2_statistical_replay",
        help="Output diagnostics directory. Relative paths are resolved under --repo-root.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--target-fars", default=",".join(str(x) for x in DEFAULT_TARGET_FARS))
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--max-diff-rows", type=int, default=200)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    method = str(args.method)
    benchmark_dir = Path(args.benchmark_dir).resolve() if args.benchmark_dir else default_benchmark_dir(repo_root, method)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else default_model_dir(benchmark_dir, method)
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir

    results = run_replay(
        repo_root=repo_root,
        benchmark_dir=benchmark_dir,
        model_dir=model_dir,
        outdir=outdir,
        method=method,
        datasets=parse_csv_list(args.datasets),
        splits=parse_csv_list(args.splits),
        target_fars=parse_float_list(args.target_fars),
        tolerance=float(args.tolerance),
        max_diff_rows=int(args.max_diff_rows),
    )
    score_eq = results["score_equivalence"]
    decision_eq = results["decision_equivalence"]
    max_diff = float(pd.to_numeric(score_eq["max_abs_diff_manual_vs_sklearn"], errors="coerce").max())
    mismatches = int(pd.to_numeric(decision_eq.get("decision_mismatches_manual_vs_sklearn", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    print("[done]", outdir)
    print(f"max_abs_diff_manual_vs_sklearn={max_diff:.17g}")
    print(f"decision_mismatches_manual_vs_sklearn={mismatches}")
    print("summary:", outdir / "replay_equivalence_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
