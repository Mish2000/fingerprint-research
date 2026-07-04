from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


METHOD_NAME = "sourceafis_sift_deep_score_fusion_v2_proto"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_TARGET_FARS = (0.005, 0.01)

NUMERIC_FEATURES = [
    "sourceafis_score",
    "sift_score",
    "sift_inliers",
    "sift_matches",
    "sift_k1",
    "sift_k2",
    "source_dpi_a",
    "source_dpi_b",
    "deep_score",
    "deep_logit",
]

CATEGORICAL_FEATURES = [
    "dataset",
    "finger_position",
    "frgp",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in str(value).split(",") if item.strip())


def read_csv(path: Path, *, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def sourceafis_train_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "plain_roll_train_scores_v1"
        / f"scores_{dataset}_sourceafis_open_train.csv"
    )


def sift_train_path(repo_root: Path, dataset: str) -> Path:
    return (
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "plain_roll_train_scores_v1"
        / f"scores_{dataset}_sift_plain_roll_v2_train.csv"
    )


def deep_train_path(repo_root: Path, dataset: str) -> Path:
    candidates = [
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "plain_roll_train_scores_v1"
        / f"scores_{dataset}_deep_pair_reranker_fast_ddp_train.csv",
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "deep_pair_reranker_fast_ddp_train_scores"
        / "scores"
        / f"scores_{dataset}_deep_pair_reranker_fast_ddp_train.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing deep train scores for "
        f"{dataset}. Tried: {[str(path) for path in candidates]}"
    )


def sourceafis_eval_path(repo_root: Path, split: str) -> Path:
    return (
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "plain_roll_full_scores_v1"
        / "sourceafis"
        / f"sourceafis_plain_roll_scores_{split}.csv"
    )


def sift_eval_path(repo_root: Path, dataset: str, split: str) -> Path:
    return (
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "plain_roll_full_scores_v1"
        / "sift"
        / f"scores_{dataset}_sift_plain_roll_v2_{split}.csv"
    )


def deep_eval_path(repo_root: Path, dataset: str, split: str) -> Path:
    return (
        repo_root
        / "artifacts"
        / "reports"
        / "benchmark"
        / "deep_pair_reranker_fast_ddp_full_pairs"
        / "scores"
        / f"scores_{dataset}_deep_pair_reranker_fast_ddp_{split}.csv"
    )


def normalize_key_columns(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset" not in out.columns:
        out["dataset"] = dataset
    if "split" not in out.columns:
        out["split"] = split
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out = out[(out["dataset"] == dataset) & (out["split"] == split)].copy()
    return out


def score_col(df: pd.DataFrame) -> str:
    for col in ("score", "raw_score", "similarity", "match_score", "probability"):
        if col in df.columns:
            return col
    raise ValueError(f"Could not find score column in columns={list(df.columns)}")


def prepare_sourceafis(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = normalize_key_columns(df, dataset=dataset, split=split)
    if "score" not in out.columns and "raw_score" in out.columns:
        out["score"] = out["raw_score"]
    cols = ["dataset", "split", "pair_id", "label", score_col(out)]
    for extra in ("raw_score", "dpi_a", "dpi_b"):
        if extra in out.columns and extra not in cols:
            cols.append(extra)
    out = out[cols].rename(
        columns={
            "label": "source_label",
            score_col(out): "sourceafis_score",
            "raw_score": "sourceafis_raw_score",
            "dpi_a": "source_dpi_a",
            "dpi_b": "source_dpi_b",
        }
    )
    return out


def prepare_sift(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = normalize_key_columns(df, dataset=dataset, split=split)
    required = ["dataset", "split", "pair_id", "label", "path_a", "path_b", "subject_a", "subject_b"]
    for col in required:
        if col not in out.columns:
            raise ValueError(f"SIFT table missing required column {col!r}. Columns={list(out.columns)}")
    if "finger_position" not in out.columns:
        if "frgp" in out.columns:
            out["finger_position"] = out["frgp"]
        else:
            raise ValueError("SIFT table missing finger_position/frgp.")
    if "frgp" not in out.columns:
        out["frgp"] = out["finger_position"]
    cols = [
        "dataset", "split", "pair_id", "label",
        "path_a", "path_b", "subject_a", "subject_b", "finger_position", "frgp",
        score_col(out),
    ]
    for extra in ("inliers", "matches", "k1", "k2"):
        if extra in out.columns:
            cols.append(extra)
    out = out[cols].rename(
        columns={
            "label": "sift_label",
            score_col(out): "sift_score",
            "inliers": "sift_inliers",
            "matches": "sift_matches",
            "k1": "sift_k1",
            "k2": "sift_k2",
        }
    )
    return out


def prepare_deep(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = normalize_key_columns(df, dataset=dataset, split=split)
    cols = ["dataset", "split", "pair_id", "label", score_col(out)]
    for extra in ("logit", "probability", "frgp"):
        if extra in out.columns:
            cols.append(extra)
    out = out[cols].rename(
        columns={
            "label": "deep_label",
            score_col(out): "deep_score",
            "logit": "deep_logit",
            "probability": "deep_probability",
        }
    )
    return out


def merge_tables(source: pd.DataFrame, sift: pd.DataFrame, deep: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "split", "pair_id"]
    merged = sift.merge(source, on=key, how="inner", validate="one_to_one")
    merged = merged.merge(deep, on=key, how="inner", validate="one_to_one")

    if len(merged) != len(sift):
        raise ValueError(f"Merge lost rows: sift={len(sift)} merged={len(merged)}")

    merged["label"] = pd.to_numeric(merged["sift_label"], errors="raise").astype(int)
    for label_col in ("source_label", "deep_label"):
        other = pd.to_numeric(merged[label_col], errors="raise").astype(int)
        mismatches = int((other != merged["label"]).sum())
        if mismatches:
            examples = merged.loc[other != merged["label"], key + ["label", label_col]].head(5).to_dict("records")
            raise ValueError(f"{label_col} mismatches label in {mismatches} rows. Examples: {examples}")

    for col in NUMERIC_FEATURES:
        if col not in merged.columns:
            merged[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in merged.columns:
            merged[col] = "__missing__"
        merged[col] = merged[col].fillna("__missing__").astype(str).str.strip()
        merged[col] = merged[col].mask(merged[col] == "", "__missing__")

    return merged


def load_train_dataset(repo_root: Path, dataset: str) -> pd.DataFrame:
    source = prepare_sourceafis(read_csv(sourceafis_train_path(repo_root, dataset), label=f"{dataset} source train"), dataset=dataset, split="train")
    sift = prepare_sift(read_csv(sift_train_path(repo_root, dataset), label=f"{dataset} sift train"), dataset=dataset, split="train")
    deep = prepare_deep(read_csv(deep_train_path(repo_root, dataset), label=f"{dataset} deep train"), dataset=dataset, split="train")
    return merge_tables(source=source, sift=sift, deep=deep)


def load_eval_dataset(repo_root: Path, dataset: str, split: str, source_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if split not in source_cache:
        source_cache[split] = read_csv(sourceafis_eval_path(repo_root, split), label=f"sourceafis {split}")
    source_all = source_cache[split]
    source = prepare_sourceafis(source_all, dataset=dataset, split=split)
    sift = prepare_sift(read_csv(sift_eval_path(repo_root, dataset, split), label=f"{dataset} sift {split}"), dataset=dataset, split=split)
    deep = prepare_deep(read_csv(deep_eval_path(repo_root, dataset, split), label=f"{dataset} deep {split}"), dataset=dataset, split=split)
    return merge_tables(source=source, sift=sift, deep=deep)


def build_model() -> Pipeline:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore")
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("features", preprocessor),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=13,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def select_threshold_from_val_negatives(labels: np.ndarray, scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    negatives = scores[labels == 0]
    negatives = negatives[np.isfinite(negatives)]
    if negatives.size == 0:
        return float("nan"), 0, float("nan")

    for threshold in sorted(float(value) for value in np.unique(negatives)):
        false_accepts = int(np.sum(negatives >= threshold))
        actual_far = float(false_accepts / negatives.size)
        if actual_far <= float(target_far) + 1e-15:
            return float(threshold), false_accepts, actual_far

    threshold = float(np.nextafter(np.max(negatives), math.inf))
    return threshold, 0, 0.0


def eer_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _thresholds = roc_curve(labels.astype(int), scores.astype(float))
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def confusion_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = labels.astype(int)
    predictions = scores >= float(threshold)
    positives = labels == 1
    negatives = labels == 0
    ta = int(np.sum(predictions & positives))
    fr = int(np.sum((~predictions) & positives))
    fa = int(np.sum(predictions & negatives))
    tr = int(np.sum((~predictions) & negatives))
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
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


def score_table(model: Pipeline, table: pd.DataFrame, *, model_path: str = "") -> pd.DataFrame:
    scores = model.predict_proba(table[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1].astype(float)
    out = table[
        [
            "dataset", "split", "pair_id", "label", "subject_a", "subject_b",
            "finger_position", "frgp", "path_a", "path_b",
            "sourceafis_score", "sift_score", "deep_score", "deep_logit",
            "sift_inliers", "sift_matches", "sift_k1", "sift_k2",
        ]
    ].copy()
    out.insert(0, "method", METHOD_NAME)
    out["score"] = scores
    out["score_semantics"] = "logistic_regression_positive_class_probability"
    out["higher_is_more_similar"] = True
    out["model_path"] = model_path
    return out


def write_markdown_summary(path: Path, *, metrics: pd.DataFrame, thresholds: pd.DataFrame, manifest: dict[str, Any]) -> None:
    text = "# sourceafis_sift_deep_score_fusion_v2_proto benchmark\n\n"
    text += "This is a train-only fitted score/geometry/deep fusion prototype. It does not include image quality features.\n\n"
    text += f"Created: `{manifest['created_at']}`\n\n"
    text += "## Metrics\n\n"
    text += metrics.to_markdown(index=False)
    text += "\n\n## Thresholds\n\n"
    text += thresholds.to_markdown(index=False)
    text += "\n"
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--outdir",
        default="artifacts/reports/benchmark/sourceafis_sift_deep_score_fusion_v2_proto_full_pairs",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--target-fars", default=",".join(str(x) for x in DEFAULT_TARGET_FARS))
    parser.add_argument("--save-training-table", action="store_true")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    scores_dir = outdir / "scores"
    model_dir = outdir / "model"
    scores_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    datasets = parse_csv_list(args.datasets)
    splits = parse_csv_list(args.splits)
    target_fars = parse_float_list(args.target_fars)

    print("[load] training tables")
    train_tables = []
    for dataset in datasets:
        df = load_train_dataset(repo_root, dataset)
        print(f"  {dataset}/train rows={len(df)} labels={df['label'].value_counts().sort_index().to_dict()}")
        train_tables.append(df)
    train = pd.concat(train_tables, ignore_index=True, sort=False)

    print("[fit] logistic regression")
    model = build_model()
    model.fit(train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train["label"].astype(int).to_numpy())

    model_path = model_dir / "fusion_model.joblib"
    joblib.dump(model, model_path)

    if args.save_training_table:
        train.to_csv(model_dir / "training_feature_table.csv", index=False)

    source_cache: dict[str, pd.DataFrame] = {}
    score_lookup: dict[tuple[str, str], Path] = {}
    eval_tables: dict[tuple[str, str], pd.DataFrame] = {}
    all_score_frames = []

    for dataset in datasets:
        for split in splits:
            print(f"[score] {dataset}/{split}")
            table = load_eval_dataset(repo_root, dataset, split, source_cache)
            eval_tables[(dataset, split)] = table
            scores = score_table(model, table, model_path=str(model_path))
            out_csv = scores_dir / f"scores_{dataset}_{METHOD_NAME}_{split}.csv"
            scores.to_csv(out_csv, index=False)
            score_lookup[(dataset, split)] = out_csv
            all_score_frames.append(scores)
            print(f"  wrote {out_csv} rows={len(scores)}")

    all_scores = pd.concat(all_score_frames, ignore_index=True, sort=False)

    thresholds_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    for dataset in datasets:
        val_scores = all_scores[(all_scores["dataset"] == dataset) & (all_scores["split"] == "val")].copy()
        if val_scores.empty:
            raise ValueError(f"Cannot select thresholds for {dataset}: missing val scores.")
        val_labels = pd.to_numeric(val_scores["label"], errors="raise").astype(int).to_numpy()
        val_values = pd.to_numeric(val_scores["score"], errors="raise").astype(float).to_numpy()

        for target_far in target_fars:
            threshold, calibration_fa, calibration_far = select_threshold_from_val_negatives(
                val_labels,
                val_values,
                target_far,
            )
            thresholds_rows.append(
                {
                    "method": METHOD_NAME,
                    "dataset": dataset,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "calibration_split": "val",
                    "calibration_negatives": int(np.sum(val_labels == 0)),
                    "calibration_positives": int(np.sum(val_labels == 1)),
                    "calibration_false_accepts": int(calibration_fa),
                    "calibration_far": float(calibration_far),
                    "selection_rule": "lowest VAL negative-score threshold with VAL FAR <= target",
                    "higher_is_more_similar": True,
                    "scores_csv": str(score_lookup.get((dataset, "val"), "")),
                }
            )

            for split in splits:
                split_scores = all_scores[(all_scores["dataset"] == dataset) & (all_scores["split"] == split)].copy()
                labels = pd.to_numeric(split_scores["label"], errors="raise").astype(int).to_numpy()
                values = pd.to_numeric(split_scores["score"], errors="raise").astype(float).to_numpy()
                row = {
                    "method": METHOD_NAME,
                    "dataset": dataset,
                    "split": split,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    **confusion_metrics(labels, values, threshold),
                    "scores_csv": str(score_lookup[(dataset, split)]),
                    "model_path": str(model_path),
                }
                metrics_rows.append(row)

    thresholds = pd.DataFrame(thresholds_rows)
    metrics = pd.DataFrame(metrics_rows)

    metrics_path = outdir / "plain_roll_final_metrics.csv"
    thresholds_path = outdir / "plain_roll_final_thresholds.csv"
    summary_path = outdir / "plain_roll_final_summary.md"
    manifest_path = outdir / "plain_roll_final_manifest.json"

    metrics.to_csv(metrics_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)

    manifest = {
        "method": METHOD_NAME,
        "repo_root": str(repo_root),
        "datasets": list(datasets),
        "splits": list(splits),
        "target_fars": list(target_fars),
        "fit_splits": ["train"],
        "threshold_calibration_split": "val",
        "test_used_for_training": False,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_path": str(model_path),
        "metrics_csv": str(metrics_path),
        "thresholds_csv": str(thresholds_path),
        "scores_dir": str(scores_dir),
        "created_at": utc_now(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown_summary(summary_path, metrics=metrics, thresholds=thresholds, manifest=manifest)

    print("[done]")
    print(metrics)
    print("metrics:", metrics_path)
    print("thresholds:", thresholds_path)
    print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
