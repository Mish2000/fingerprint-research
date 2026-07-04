from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark import run_plain_roll_final_benchmark as final
from pipelines.benchmark import run_sourceafis_sift_quality_fusion_benchmark as fusion_benchmark
from pipelines.benchmark import train_sourceafis_sift_quality_fusion as fusion_train
from scripts.diagnostics import run_sourceafis_plain_roll_benchmark as sourceafis
from src.fpbench.universal import calibration
from src.fpbench.universal.fusion_features import (
    METHOD_NAME,
    build_feature_table,
    build_feature_tables,
    default_categorical_feature_columns,
    default_numeric_feature_columns,
)


PHASE15_SCHEMA_VERSION = "sourceafis_sift_quality_fusion_phase15_validation_v1"
DEFAULT_ABLATION_OUTDIR = REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_fusion_ablation_v1"
DEFAULT_FULL_PAIRS_OUTDIR = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_final_fusion_v1_full_pairs"
)
DEFAULT_FULL_SCORE_DIR = REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_full_scores_v1"
DEFAULT_FULL_SOURCEAFIS_SCORE_DIR = DEFAULT_FULL_SCORE_DIR / "sourceafis"
DEFAULT_FULL_SIFT_SCORE_DIR = DEFAULT_FULL_SCORE_DIR / "sift"
DEFAULT_SELECTED_SOURCEAFIS_SCORE_DIR = fusion_benchmark.DEFAULT_SOURCEAFIS_SCORE_DIR
DEFAULT_SELECTED_SIFT_SCORE_DIR = fusion_benchmark.DEFAULT_SIFT_PLAIN_ROLL_SCORE_DIR
DEFAULT_TARGET_FARS = calibration.DEFAULT_TARGET_FARS
DEFAULT_DATASETS = fusion_benchmark.DEFAULT_DATASETS
DEFAULT_SPLITS = fusion_benchmark.DEFAULT_SPLITS


class Phase15ValidationError(RuntimeError):
    """Raised when Phase 1.5 validation inputs or scoring fail."""


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str


ABLATION_VARIANTS = (
    AblationVariant(
        "sourceafis_only_calibrated",
        "Logistic calibration over SourceAFIS score only.",
    ),
    AblationVariant(
        "sourceafis_plus_sift_score",
        "SourceAFIS score plus the scalar SIFT Plain/Roll v2 score.",
    ),
    AblationVariant(
        "sourceafis_plus_sift_geometry",
        "SourceAFIS score plus SIFT score and SIFT geometry/match-count columns.",
    ),
    AblationVariant(
        "sourceafis_plus_sift_quality_full",
        "Full SourceAFIS + SIFT + quality feature set used by sourceafis_sift_quality_fusion_v1.",
    ),
    AblationVariant(
        "quality_only_control",
        "Image quality and PPI features only; no SourceAFIS or SIFT score inputs.",
    ),
)

COMPARISON_COLUMNS = [
    "dataset",
    "split",
    "target_far",
    "fusion_method",
    "baseline_method",
    "fusion_threshold",
    "sourceafis_threshold",
    "fusion_threshold_val_far",
    "sourceafis_threshold_val_far",
    "n_pairs",
    "n_positive",
    "n_negative",
    "fusion_ta",
    "fusion_fr",
    "fusion_fa",
    "fusion_tr",
    "fusion_tar",
    "fusion_far",
    "sourceafis_ta",
    "sourceafis_fr",
    "sourceafis_fa",
    "sourceafis_tr",
    "sourceafis_tar",
    "sourceafis_far",
    "rescued_positives",
    "lost_positives",
    "fixed_false_accepts",
    "new_false_accepts",
    "positive_discordant_count",
    "negative_discordant_count",
    "paired_discordant_count",
    "fusion_only_accepts",
    "sourceafis_only_accepts",
    "both_accept",
    "both_reject",
    "fusion_scores_csv",
    "sourceafis_scores_csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_file_uri(raw: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    return fusion_benchmark.parse_file_uri(raw, repo_root=repo_root)


def _parse_csv_arg(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    return tuple(item.strip() for item in raw_items if item.strip())


def _git_info(repo_root: Path) -> dict[str, Any]:
    def run_git(args: list[str]) -> tuple[int, str]:
        proc = subprocess.run(["git", *args], cwd=str(repo_root), capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()

    commit_rc, commit = run_git(["rev-parse", "HEAD"])
    branch_rc, branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status_rc, status = run_git(["status", "--porcelain"])
    return {
        "commit": commit if commit_rc == 0 else "",
        "branch": branch if branch_rc == 0 else "",
        "is_dirty": bool(status.strip()) if status_rc == 0 else None,
        "status_porcelain": status,
    }


def _existing_columns(table: pd.DataFrame, names: Iterable[str]) -> list[str]:
    return [name for name in names if name in table.columns]


def _sift_geometry_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in (
            "sift_plain_roll_v2_score_inliers",
            "sift_plain_roll_v2_score_matches",
            "sift_plain_roll_v2_score_k1",
            "sift_plain_roll_v2_score_k2",
        )
        if column in table.columns
    ]


def _quality_numeric_columns(table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in table.columns:
        text = str(column)
        if text in {"ppi", "ppi_a", "ppi_b"}:
            columns.append(text)
        elif text.startswith(("a_", "b_")):
            columns.append(text)
        elif text.startswith("pair_") and text.endswith("_abs_delta"):
            columns.append(text)
    return columns


def feature_columns_for_variant(variant_name: str, table: pd.DataFrame) -> tuple[list[str], list[str]]:
    if variant_name == "sourceafis_only_calibrated":
        return _existing_columns(table, ["sourceafis_score"]), []
    if variant_name == "sourceafis_plus_sift_score":
        return _existing_columns(table, ["sourceafis_score", "sift_plain_roll_v2_score"]), []
    if variant_name == "sourceafis_plus_sift_geometry":
        return _existing_columns(table, ["sourceafis_score", "sift_plain_roll_v2_score"]) + _sift_geometry_columns(table), []
    if variant_name == "sourceafis_plus_sift_quality_full":
        return default_numeric_feature_columns(table), default_categorical_feature_columns(table)
    if variant_name == "quality_only_control":
        return _quality_numeric_columns(table), []
    raise Phase15ValidationError(f"Unknown ablation variant: {variant_name!r}")


def _variant_needs_quality(variant_name: str) -> bool:
    return variant_name in {"sourceafis_plus_sift_quality_full", "quality_only_control"}


def _variant_by_name(names: tuple[str, ...] | None = None) -> tuple[AblationVariant, ...]:
    if not names:
        return ABLATION_VARIANTS
    lookup = {variant.name: variant for variant in ABLATION_VARIANTS}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise Phase15ValidationError(f"Unknown ablation variant(s): {missing}")
    return tuple(lookup[name] for name in names)


def _run_subprocess(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(cwd)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise Phase15ValidationError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _build_train_features(
    *,
    datasets: tuple[str, ...],
    train_score_dir: Path,
    sift_score_dir: Path | None,
    repo_root: Path,
    generate_missing_scores: bool,
    generate_sourceafis_scores: bool,
    generate_sift_scores: bool,
    sourceafis_template_cache_dir: Path | None,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
    include_quality: bool,
) -> pd.DataFrame:
    fusion_train.ensure_train_scores(
        datasets=datasets,
        score_dir=train_score_dir,
        repo_root=repo_root,
        generate_missing_scores=bool(generate_missing_scores),
        generate_sourceafis_scores=bool(generate_sourceafis_scores),
        generate_sift_scores=bool(generate_sift_scores),
        sourceafis_template_cache_dir=sourceafis_template_cache_dir,
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
        max_retries=int(max_retries),
        retry_backoff_seconds=float(retry_backoff_seconds),
        dpi_strategy=dpi_strategy,
        image_dpi=image_dpi,
    )
    specs = fusion_train.build_training_specs(
        datasets=datasets,
        train_score_dir=train_score_dir,
        sift_score_dir=sift_score_dir,
        repo_root=repo_root,
    )
    table = build_feature_tables(specs, repo_root=repo_root, include_quality=bool(include_quality))
    calibration.assert_train_only(table)
    return table


def _load_or_build_train_features(
    *,
    cache_csv: Path,
    datasets: tuple[str, ...],
    train_score_dir: Path,
    sift_score_dir: Path | None,
    repo_root: Path,
    generate_missing_scores: bool,
    generate_sourceafis_scores: bool,
    generate_sift_scores: bool,
    sourceafis_template_cache_dir: Path | None,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
    include_quality: bool,
) -> pd.DataFrame:
    if cache_csv.exists() and cache_csv.stat().st_size > 0:
        table = pd.read_csv(cache_csv)
        calibration.assert_train_only(table)
        return table
    table = _build_train_features(
        datasets=datasets,
        train_score_dir=train_score_dir,
        sift_score_dir=sift_score_dir,
        repo_root=repo_root,
        generate_missing_scores=generate_missing_scores,
        generate_sourceafis_scores=generate_sourceafis_scores,
        generate_sift_scores=generate_sift_scores,
        sourceafis_template_cache_dir=sourceafis_template_cache_dir,
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        dpi_strategy=dpi_strategy,
        image_dpi=image_dpi,
        include_quality=include_quality,
    )
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cache_csv, index=False)
    return table


def train_ablation_models(
    *,
    feature_table: pd.DataFrame,
    variants: tuple[AblationVariant, ...],
    outdir: Path,
    repo_root: Path,
    train_score_dir: Path,
    random_state: int,
) -> dict[str, tuple[Any, dict[str, Any], Path]]:
    model_root = outdir / "model"
    model_root.mkdir(parents=True, exist_ok=True)
    label_counts = feature_table["label"].value_counts().sort_index().to_dict()
    dataset_counts = feature_table.groupby("dataset").size().astype(int).to_dict()

    trained: dict[str, tuple[Any, dict[str, Any], Path]] = {}
    for variant in variants:
        numeric_features, categorical_features = feature_columns_for_variant(variant.name, feature_table)
        if not numeric_features and not categorical_features:
            raise Phase15ValidationError(f"Ablation variant {variant.name!r} has no usable feature columns.")
        model, schema = calibration.fit_fusion_model(
            feature_table,
            random_state=int(random_state),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        schema["feature_variant"] = variant.name
        schema["feature_variant_description"] = variant.description
        variant_model_dir = model_root / variant.name
        manifest = {
            "repo_root": str(repo_root),
            "git": _git_info(repo_root),
            "python": sys.version,
            "platform": platform.platform(),
            "phase15_schema_version": PHASE15_SCHEMA_VERSION,
            "variant": {
                "name": variant.name,
                "description": variant.description,
                "numeric_features": numeric_features,
                "categorical_features": categorical_features,
            },
            "protocol": {
                "fit_splits": ["train"],
                "val_used_for": "threshold calibration only",
                "test_used_for": "final evaluation only",
                "no_test_leakage": True,
            },
            "train_score_dir": str(train_score_dir),
            "training_rows": int(len(feature_table)),
            "label_counts": {str(key): int(value) for key, value in label_counts.items()},
            "dataset_counts": {str(key): int(value) for key, value in dataset_counts.items()},
            "feature_count": int(len(schema.get("model_features", []))),
            "numeric_feature_count": int(len(schema.get("numeric_features", []))),
            "categorical_feature_count": int(len(schema.get("categorical_features", []))),
            "model": {
                "type": "sklearn.pipeline.Pipeline",
                "classifier": "LogisticRegression",
                "class_weight": "balanced",
                "scaler": "StandardScaler",
                "random_state": int(random_state),
            },
            "created_at": _utc_now(),
        }
        calibration.save_model_bundle(
            model=model,
            schema=schema,
            model_dir=variant_model_dir,
            training_manifest=manifest,
        )
        trained[variant.name] = (model, schema, variant_model_dir)
    return trained


def _materialize_eval_features(
    *,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    outdir: Path,
    pair_scope: str,
    selected_pairs_dir: Path | None,
    sourceafis_score_dir: Path,
    sift_plain_roll_score_dir: Path,
    sift_score_dir: Path | None,
    repo_root: Path,
    limit_per_split: int,
) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[tuple[str, str], Path], list[dict[str, Any]], dict[tuple[str, str], dict[str, Path | None]]]:
    feature_tables: dict[tuple[str, str], pd.DataFrame] = {}
    selected_paths: dict[tuple[str, str], Path] = {}
    statuses: list[dict[str, Any]] = []
    score_sources: dict[tuple[str, str], dict[str, Path | None]] = {}

    for dataset in datasets:
        for split in splits:
            pairs_csv, status = fusion_benchmark._materialize_pairs(
                dataset=dataset,
                split=split,
                pair_scope=pair_scope,
                selected_pairs_dir=selected_pairs_dir,
                outdir=outdir,
                repo_root=repo_root,
                limit_per_split=int(limit_per_split),
            )
            selected_paths[(dataset, split)] = pairs_csv
            statuses.append(status)
            sourceafis_scores = fusion_benchmark._resolve_score_path(
                sourceafis_score_dir, dataset, "sourceafis_open", split
            )
            sift_plain_roll_scores = fusion_benchmark._resolve_score_path(
                sift_plain_roll_score_dir, dataset, "sift_plain_roll_v2", split
            )
            sift_scores = fusion_benchmark._optional_score_path(sift_score_dir, dataset, "sift", split)
            score_sources[(dataset, split)] = {
                "sourceafis": sourceafis_scores,
                "sift_plain_roll_v2": sift_plain_roll_scores,
                "sift": sift_scores,
            }
            feature_tables[(dataset, split)] = build_feature_table(
                dataset=dataset,
                split=split,
                pairs_csv=pairs_csv,
                sourceafis_scores_csv=sourceafis_scores,
                sift_plain_roll_scores_csv=sift_plain_roll_scores,
                sift_scores_csv=sift_scores,
                repo_root=repo_root,
            )
    return feature_tables, selected_paths, statuses, score_sources


def _write_variant_score_run(
    *,
    variant: AblationVariant,
    dataset: str,
    split: str,
    feature_table: pd.DataFrame,
    pairs_csv: Path,
    model: Any,
    schema: dict[str, Any],
    outdir: Path,
) -> tuple[final.ScoreRun, pd.DataFrame]:
    start = time.perf_counter()
    scores = calibration.predict_fusion_scores(model, schema, feature_table)
    elapsed_ms = float((time.perf_counter() - start) * 1000.0)
    avg_ms = float(elapsed_ms / max(int(len(feature_table)), 1))

    score_dir = outdir / "scores"
    run_meta_dir = outdir / "run_meta"
    roc_dir = outdir / "roc"
    score_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    roc_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = score_dir / f"scores_{dataset}_{variant.name}_{split}.csv"
    run_meta_json = run_meta_dir / f"run_{dataset}_{variant.name}_{split}.meta.json"
    method_meta_json = scores_csv.with_suffix(".meta.json")

    output = feature_table.copy()
    output["method"] = variant.name
    output["score"] = scores
    output["score_semantics"] = "logistic_regression_positive_class_probability"
    output["higher_is_more_similar"] = True
    output["run_level_avg_pair_ms"] = avg_ms
    output["pair_total_ms"] = float("nan")
    output["pair_total_ms_semantics"] = "not_measured_per_pair; run_level_avg_pair_ms is batch prediction average"

    score_columns = [
        "method",
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
        "score",
        "score_semantics",
        "higher_is_more_similar",
        "sourceafis_score",
        "sift_plain_roll_v2_score",
        "sift_plain_roll_v2_score_inliers",
        "sift_plain_roll_v2_score_matches",
        "sift_plain_roll_v2_score_k1",
        "sift_plain_roll_v2_score_k2",
        "sift_score",
        "ppi",
        "ppi_a",
        "ppi_b",
        "run_level_avg_pair_ms",
        "pair_total_ms",
        "pair_total_ms_semantics",
    ]
    for column in score_columns:
        if column not in output.columns:
            output[column] = ""
    output[score_columns].to_csv(scores_csv, index=False)

    method_meta = {
        "schema_version": "sourceafis_sift_quality_fusion_ablation_score_meta_v1",
        "method": variant.name,
        "base_method": METHOD_NAME,
        "feature_variant": variant.name,
        "avg_ms_pair": avg_ms,
        "p50_ms_pair": None,
        "p95_ms_pair": None,
        "total_ms": elapsed_ms,
        "score_semantics": "logistic_regression_positive_class_probability",
        "latency_scope": "batch_prediction_run_level_average_only",
        "pair_total_ms_semantics": "not_measured_per_pair",
    }
    method_meta_json.write_text(json.dumps(method_meta, indent=2, ensure_ascii=True), encoding="utf-8")
    run_meta = {
        "schema_version": "sourceafis_sift_quality_fusion_ablation_run_meta_v1",
        "method": variant.name,
        "base_method": METHOD_NAME,
        "created_at": _utc_now(),
        "method_meta_json": str(method_meta_json),
        "timing": {
            "avg_ms_pair_reported": avg_ms,
            "avg_ms_pair_wall": avg_ms,
            "total_ms": elapsed_ms,
            "n_pairs": int(len(output)),
            "latency_scope": "batch_prediction_run_level_average_only",
            "pair_total_ms_semantics": "not_measured_per_pair",
        },
        "row": {
            "avg_ms_pair_reported": avg_ms,
            "avg_ms_pair_wall": avg_ms,
            "meta_json": str(method_meta_json),
        },
    }
    run_meta_json.write_text(json.dumps(run_meta, indent=2, ensure_ascii=True), encoding="utf-8")

    run = final.ScoreRun(
        method=variant.name,
        dataset=dataset,
        split=split,
        selected_pairs_csv=pairs_csv,
        scores_csv=scores_csv,
        roc_png=roc_dir / f"roc_{dataset}_{variant.name}_{split}.png",
        run_meta_json=run_meta_json,
        command=[],
        elapsed_seconds=float(elapsed_ms / 1000.0),
        reused_existing_scores=False,
    )
    score_frame = output[["method", "dataset", "split", "pair_id", "label", "score"]].copy()
    return run, score_frame


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{100.0 * number:.2f}%"


def render_ablation_summary(
    *,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    variants: tuple[AblationVariant, ...],
    output_paths: dict[str, Path],
    total_runtime_s: float,
) -> str:
    lines = [
        "# SourceAFIS/SIFT/Quality Fusion Ablation v1",
        "",
        f"Created: `{_utc_now()}`",
        f"Total runtime: `{total_runtime_s:.2f}s`",
        "",
        "## Protocol",
        "",
        "- Models are fit on train rows only.",
        "- Thresholds are selected from VAL negative scores only.",
        "- Selected VAL/TEST pairs are evaluated with frozen train-only models and frozen VAL thresholds.",
        "- Latency in this ablation is batch prediction timing only; score CSVs do not contain true per-pair latency.",
        "",
        "## Variants",
        "",
        "| variant | description |",
        "| --- | --- |",
    ]
    for variant in variants:
        lines.append(f"| {variant.name} | {variant.description} |")

    lines.extend(
        [
            "",
            "## TEST Operating Points",
            "",
            "| variant | dataset | target FAR | threshold | TAR | FAR | TA/FR/FA/TR | AUC | EER |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    test_metrics = metrics[metrics["split"].astype(str).str.lower() == "test"].copy() if not metrics.empty else metrics
    for _, row in test_metrics.sort_values(["method", "dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['dataset']} | {_fmt_pct(row['target_far'])} | "
            f"{_fmt_float(row['threshold'], 6)} | {_fmt_pct(row['tar'])} | {_fmt_pct(row['far'])} | "
            f"{int(row['ta'])}/{int(row['fr'])}/{int(row['fa'])}/{int(row['tr'])} | "
            f"{_fmt_float(row['auc'])} | {_fmt_float(row['eer'])} |"
        )

    lines.extend(
        [
            "",
            "## VAL Thresholds",
            "",
            "| variant | dataset | target FAR | threshold | VAL FAR | false accepts / negatives |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in thresholds.sort_values(["method", "dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['dataset']} | {_fmt_pct(row['target_far'])} | "
            f"{_fmt_float(row['threshold'], 6)} | {_fmt_pct(row['calibration_far'])} | "
            f"{int(row['calibration_false_accepts'])}/{int(row['calibration_negative_count'])} |"
        )

    lines.extend(["", "## Artifacts", ""])
    for key, path in sorted(output_paths.items()):
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def run_ablation_validation(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    variants: tuple[str, ...] | None = None,
    train_score_dir: Path = fusion_train.DEFAULT_TRAIN_SCORE_DIR,
    outdir: Path = DEFAULT_ABLATION_OUTDIR,
    pair_scope: str = "full",
    selected_pairs_dir: Path | None = fusion_benchmark.DEFAULT_SELECTED_PAIRS_DIR,
    sourceafis_score_dir: Path = DEFAULT_SELECTED_SOURCEAFIS_SCORE_DIR,
    sift_plain_roll_score_dir: Path = DEFAULT_SELECTED_SIFT_SCORE_DIR,
    sift_score_dir: Path | None = None,
    target_fars: tuple[float, ...] = DEFAULT_TARGET_FARS,
    limit_per_split: int = 0,
    repo_root: Path = REPO_ROOT,
    generate_missing_scores: bool = False,
    generate_sourceafis_scores: bool = True,
    generate_sift_scores: bool = True,
    sourceafis_template_cache_dir: Path | None = None,
    request_timeout_seconds: float | None = None,
    extract_timeout_seconds: float | None = None,
    verify_timeout_seconds: float | None = None,
    max_retries: int = sourceafis.DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = sourceafis.DEFAULT_RETRY_BACKOFF_SECONDS,
    dpi_strategy: str = sourceafis.DEFAULT_DPI_STRATEGY,
    image_dpi: int | None = None,
    random_state: int = 13,
    save_training_features: bool = False,
) -> dict[str, Path]:
    if "val" not in tuple(split.lower() for split in splits):
        raise Phase15ValidationError("Ablation validation requires val split for threshold calibration.")
    start = time.perf_counter()
    output = parse_file_uri(outdir, repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=True)
    variant_specs = _variant_by_name(variants)
    train_scores = parse_file_uri(train_score_dir, repo_root=repo_root)
    selected_dir = parse_file_uri(selected_pairs_dir, repo_root=repo_root) if selected_pairs_dir is not None else None
    sourceafis_dir = parse_file_uri(sourceafis_score_dir, repo_root=repo_root)
    sift_plain_roll_dir = parse_file_uri(sift_plain_roll_score_dir, repo_root=repo_root)
    sift_dir = parse_file_uri(sift_score_dir, repo_root=repo_root) if sift_score_dir is not None else None

    score_only_cache = output / "training_feature_table_score_only.csv"
    quality_cache = output / "training_feature_table_quality.csv"
    score_only_variants = tuple(variant for variant in variant_specs if not _variant_needs_quality(variant.name))
    quality_variants = tuple(variant for variant in variant_specs if _variant_needs_quality(variant.name))

    trained: dict[str, tuple[Any, dict[str, Any], Path]] = {}
    if score_only_variants:
        score_train_features = _load_or_build_train_features(
            cache_csv=score_only_cache,
            datasets=datasets,
            train_score_dir=train_scores,
            sift_score_dir=sift_dir,
            repo_root=repo_root,
            generate_missing_scores=bool(generate_missing_scores),
            generate_sourceafis_scores=bool(generate_sourceafis_scores),
            generate_sift_scores=bool(generate_sift_scores),
            sourceafis_template_cache_dir=sourceafis_template_cache_dir,
            request_timeout_seconds=request_timeout_seconds,
            extract_timeout_seconds=extract_timeout_seconds,
            verify_timeout_seconds=verify_timeout_seconds,
            max_retries=int(max_retries),
            retry_backoff_seconds=float(retry_backoff_seconds),
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
            include_quality=False,
        )
        trained.update(
            train_ablation_models(
                feature_table=score_train_features,
                variants=score_only_variants,
                outdir=output,
                repo_root=repo_root,
                train_score_dir=train_scores,
                random_state=int(random_state),
            )
        )
    if quality_variants:
        quality_train_features = _load_or_build_train_features(
            cache_csv=quality_cache,
            datasets=datasets,
            train_score_dir=train_scores,
            sift_score_dir=sift_dir,
            repo_root=repo_root,
            generate_missing_scores=bool(generate_missing_scores),
            generate_sourceafis_scores=bool(generate_sourceafis_scores),
            generate_sift_scores=bool(generate_sift_scores),
            sourceafis_template_cache_dir=sourceafis_template_cache_dir,
            request_timeout_seconds=request_timeout_seconds,
            extract_timeout_seconds=extract_timeout_seconds,
            verify_timeout_seconds=verify_timeout_seconds,
            max_retries=int(max_retries),
            retry_backoff_seconds=float(retry_backoff_seconds),
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
            include_quality=True,
        )
        trained.update(
            train_ablation_models(
                feature_table=quality_train_features,
                variants=quality_variants,
                outdir=output,
                repo_root=repo_root,
                train_score_dir=train_scores,
                random_state=int(random_state),
            )
        )

    feature_tables, selected_paths, dataset_statuses, score_sources = _materialize_eval_features(
        datasets=datasets,
        splits=splits,
        outdir=output,
        pair_scope=pair_scope,
        selected_pairs_dir=selected_dir,
        sourceafis_score_dir=sourceafis_dir,
        sift_plain_roll_score_dir=sift_plain_roll_dir,
        sift_score_dir=sift_dir,
        repo_root=repo_root,
        limit_per_split=int(limit_per_split),
    )
    pair_audit_dir = output / "pair_audit"
    pair_audit_reports = final.write_pair_audits(
        selected_pairs=selected_paths,
        pair_audit_out=pair_audit_dir,
        repo_root=repo_root,
    )
    final.write_pair_audit_summary_markdown(pair_audit_reports, pair_audit_dir / "pair_audit_summary.md")

    score_runs: list[final.ScoreRun] = []
    for variant in variant_specs:
        model, schema, _model_dir = trained[variant.name]
        for dataset in datasets:
            for split in splits:
                run, _score_frame = _write_variant_score_run(
                    variant=variant,
                    dataset=dataset,
                    split=split,
                    feature_table=feature_tables[(dataset, split)],
                    pairs_csv=selected_paths[(dataset, split)],
                    model=model,
                    schema=schema,
                    outdir=output,
                )
                score_runs.append(run)

    latency = final.build_latency_rows(score_runs)
    thresholds = final.build_threshold_table(score_runs, tuple(float(x) for x in target_fars))
    metrics = final.build_metrics_table(score_runs, thresholds, latency)

    paths: dict[str, Path] = {
        "metrics": output / "ablation_metrics.csv",
        "thresholds": output / "ablation_thresholds.csv",
        "summary": output / "ablation_summary.md",
        "latency_summary": output / "ablation_latency_summary.csv",
        "manifest": output / "ablation_manifest.json",
        "score_only_training_features": score_only_cache,
        "quality_training_features": quality_cache,
    }
    if not save_training_features:
        paths = {key: value for key, value in paths.items() if not key.endswith("_training_features")}
    for (dataset, split), path in selected_paths.items():
        paths[f"selected_pairs_{dataset}_{split}"] = path
    for variant in variant_specs:
        paths[f"model_{variant.name}"] = trained[variant.name][2] / "fusion_model.joblib"

    final._write_csv(paths["metrics"], metrics, final.METRICS_COLUMNS)
    final._write_csv(paths["thresholds"], thresholds, final.THRESHOLD_COLUMNS)
    final._write_csv(paths["latency_summary"], latency, final.LATENCY_COLUMNS)

    total_runtime_s = time.perf_counter() - start
    paths["summary"].write_text(
        render_ablation_summary(
            metrics=metrics,
            thresholds=thresholds,
            variants=variant_specs,
            output_paths=paths,
            total_runtime_s=total_runtime_s,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": PHASE15_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "git": _git_info(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "protocol": {
            "fit_splits": ["train"],
            "threshold_split": "val",
            "threshold_selection": "VAL negative scores only",
            "test_used_for": "final evaluation only",
            "no_test_leakage": True,
        },
        "datasets": list(datasets),
        "splits": list(splits),
        "target_fars": [float(x) for x in target_fars],
        "variants": [variant.__dict__ for variant in variant_specs],
        "train_score_dir": str(train_scores),
        "pair_scope": pair_scope,
        "selected_pairs_dir": str(selected_dir) if selected_dir is not None else "",
        "sourceafis_score_dir": str(sourceafis_dir),
        "sift_plain_roll_score_dir": str(sift_plain_roll_dir),
        "score_sources": {
            f"{dataset}_{split}": {
                key: str(value) if value is not None else ""
                for key, value in sources.items()
            }
            for (dataset, split), sources in score_sources.items()
        },
        "selected_pair_sets": dataset_statuses,
        "outputs": {key: str(value) for key, value in paths.items()},
        "total_runtime_s": float(total_runtime_s),
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return paths


def _missing_score_keys(
    *,
    score_dir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    method: str,
) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []
    for dataset in datasets:
        for split in splits:
            try:
                path = fusion_benchmark._resolve_score_path(score_dir, dataset, method, split)
            except fusion_benchmark.FusionBenchmarkError:
                missing.append((dataset, split))
                continue
            if not _score_csv_has_pair_traceability(path):
                missing.append((dataset, split))
    return missing


def _score_csv_has_pair_traceability(path: Path) -> bool:
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return {"pair_id", "label"} <= columns


def _repair_sift_traceability_if_possible(
    *,
    missing_keys: list[tuple[str, str]],
    score_dir: Path,
    repo_root: Path,
) -> None:
    for dataset, split in missing_keys:
        try:
            scores_csv = fusion_benchmark._resolve_score_path(score_dir, dataset, "sift_plain_roll_v2", split)
        except fusion_benchmark.FusionBenchmarkError:
            continue
        if _score_csv_has_pair_traceability(scores_csv):
            continue
        pairs_csv = fusion_benchmark._pairs_path(dataset, split, repo_root=repo_root)
        fusion_train._attach_pair_traceability(
            dataset=dataset,
            pairs_csv=pairs_csv,
            scores_csv=scores_csv,
        )


def _full_score_generation_help(sourceafis_dir: Path, sift_dir: Path) -> str:
    return (
        "Missing full-pairs score CSVs. Generate them with this runner using --mode full --generate_missing_scores, "
        "or manually run the SourceAFIS/SIFT full-score commands documented in "
        "docs/research/sourceafis_sift_quality_fusion_v1.md.\n"
        f"Expected SourceAFIS dir: {sourceafis_dir}\n"
        f"Expected SIFT dir: {sift_dir}"
    )


def ensure_full_pair_scores(
    *,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    sourceafis_score_dir: Path,
    sift_plain_roll_score_dir: Path,
    repo_root: Path,
    generate_missing_scores: bool,
    generate_sourceafis_scores: bool,
    generate_sift_scores: bool,
    sourceafis_template_cache_dir: Path | None,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
) -> None:
    sourceafis_score_dir.mkdir(parents=True, exist_ok=True)
    sift_plain_roll_score_dir.mkdir(parents=True, exist_ok=True)
    missing_sourceafis = _missing_score_keys(
        score_dir=sourceafis_score_dir,
        datasets=datasets,
        splits=splits,
        method="sourceafis_open",
    )
    missing_sift = _missing_score_keys(
        score_dir=sift_plain_roll_score_dir,
        datasets=datasets,
        splits=splits,
        method="sift_plain_roll_v2",
    )
    if missing_sift:
        _repair_sift_traceability_if_possible(
            missing_keys=missing_sift,
            score_dir=sift_plain_roll_score_dir,
            repo_root=repo_root,
        )
        missing_sift = _missing_score_keys(
            score_dir=sift_plain_roll_score_dir,
            datasets=datasets,
            splits=splits,
            method="sift_plain_roll_v2",
        )
    if not missing_sourceafis and not missing_sift:
        return
    if not generate_missing_scores:
        missing_text = "\n".join(
            [f"  sourceafis_open {dataset} {split}" for dataset, split in missing_sourceafis]
            + [f"  sift_plain_roll_v2 {dataset} {split}" for dataset, split in missing_sift]
        )
        raise Phase15ValidationError(f"{_full_score_generation_help(sourceafis_score_dir, sift_plain_roll_score_dir)}\nMissing:\n{missing_text}")

    if missing_sourceafis:
        if not generate_sourceafis_scores:
            raise Phase15ValidationError(f"Missing SourceAFIS full-pairs scores and SourceAFIS generation is disabled: {missing_sourceafis}")
        sourceafis.run_benchmark(
            datasets=datasets,
            splits=splits,
            outdir=sourceafis_score_dir,
            target_fars=DEFAULT_TARGET_FARS,
            limit_per_split=0,
            repo_root=repo_root,
            template_cache_dir=sourceafis_template_cache_dir,
            request_timeout_seconds=request_timeout_seconds,
            extract_timeout_seconds=extract_timeout_seconds,
            verify_timeout_seconds=verify_timeout_seconds,
            max_retries=int(max_retries),
            retry_backoff_seconds=float(retry_backoff_seconds),
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
        )

    if missing_sift:
        if not generate_sift_scores:
            raise Phase15ValidationError(f"Missing SIFT full-pairs scores and SIFT generation is disabled: {missing_sift}")
        for dataset, split in missing_sift:
            scores_csv = sift_plain_roll_score_dir / f"scores_{dataset}_sift_plain_roll_v2_{split}.csv"
            roc_png = sift_plain_roll_score_dir / f"roc_{dataset}_sift_plain_roll_v2_{split}.png"
            run_meta = sift_plain_roll_score_dir / f"run_{dataset}_sift_plain_roll_v2_{split}.meta.json"
            summary_csv = sift_plain_roll_score_dir / "evaluate_results_summary.csv"
            cmd = [
                sys.executable,
                str(repo_root / "pipelines" / "benchmark" / "evaluate.py"),
                "--method",
                "sift_plain_roll_v2",
                "--dataset",
                dataset,
                "--split",
                split,
                "--data_dir",
                str(repo_root / "data" / "manifests" / dataset),
                "--limit",
                "0",
                "--out_scores",
                str(scores_csv),
                "--out_roc",
                str(roc_png),
                "--out_run_meta",
                str(run_meta),
                "--summary_csv",
                str(summary_csv),
            ]
            _run_subprocess(cmd, cwd=repo_root)
            fusion_train._attach_pair_traceability(
                dataset=dataset,
                pairs_csv=fusion_benchmark._pairs_path(dataset, split, repo_root=repo_root),
                scores_csv=scores_csv,
            )

    still_missing_sourceafis = _missing_score_keys(
        score_dir=sourceafis_score_dir,
        datasets=datasets,
        splits=splits,
        method="sourceafis_open",
    )
    still_missing_sift = _missing_score_keys(
        score_dir=sift_plain_roll_score_dir,
        datasets=datasets,
        splits=splits,
        method="sift_plain_roll_v2",
    )
    if still_missing_sourceafis or still_missing_sift:
        raise Phase15ValidationError(
            "Full score generation finished with missing files: "
            f"sourceafis={still_missing_sourceafis} sift={still_missing_sift}"
        )


def _first_score_column(df: pd.DataFrame) -> str:
    for column in ("score", "raw_score", "similarity", "match_score"):
        if column in df.columns:
            return column
    raise Phase15ValidationError("Score CSV is missing score/raw_score column.")


def _load_scores_for_dataset(path: Path, *, dataset: str, split: str, score_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if "dataset" not in raw.columns:
        raw["dataset"] = dataset
    if "split" not in raw.columns:
        raw["split"] = split
    raw["dataset"] = raw["dataset"].astype(str).str.strip()
    raw["split"] = raw["split"].astype(str).str.strip().str.lower()
    raw["pair_id"] = raw["pair_id"].astype(str).str.strip()
    raw["label"] = pd.to_numeric(raw["label"], errors="coerce").fillna(-1).astype(int)
    score_column = _first_score_column(raw)
    filtered = raw[(raw["dataset"] == dataset) & (raw["split"] == split.lower())].copy()
    if filtered.empty:
        raise Phase15ValidationError(f"No rows for dataset={dataset!r} split={split!r} in {path}")
    return filtered[["dataset", "split", "pair_id", "label", score_column]].rename(columns={score_column: score_name})


def _threshold_from_val_negatives(scores: pd.DataFrame, *, target_far: float, score_column: str) -> calibration.ThresholdSelection:
    labels = pd.to_numeric(scores["label"], errors="coerce").fillna(-1).astype(int)
    values = pd.to_numeric(scores[score_column], errors="coerce")
    return calibration.select_threshold_from_negative_scores(
        values[labels == 0],
        target_far=float(target_far),
        positive_count=int((labels == 1).sum()),
    )


def write_sourceafis_statistical_comparison(
    *,
    outdir: Path,
    datasets: tuple[str, ...],
    target_fars: tuple[float, ...],
    sourceafis_score_dir: Path,
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        fusion_val_path = outdir / "scores" / f"scores_{dataset}_{METHOD_NAME}_val.csv"
        fusion_test_path = outdir / "scores" / f"scores_{dataset}_{METHOD_NAME}_test.csv"
        if not fusion_val_path.exists() or not fusion_test_path.exists():
            raise Phase15ValidationError(f"Missing fusion full-pairs score CSVs for {dataset} in {outdir / 'scores'}")
        source_val_path = fusion_benchmark._resolve_score_path(sourceafis_score_dir, dataset, "sourceafis_open", "val")
        source_test_path = fusion_benchmark._resolve_score_path(sourceafis_score_dir, dataset, "sourceafis_open", "test")

        fusion_val = _load_scores_for_dataset(fusion_val_path, dataset=dataset, split="val", score_name="fusion_score")
        fusion_test = _load_scores_for_dataset(fusion_test_path, dataset=dataset, split="test", score_name="fusion_score")
        source_val = _load_scores_for_dataset(source_val_path, dataset=dataset, split="val", score_name="sourceafis_score")
        source_test = _load_scores_for_dataset(source_test_path, dataset=dataset, split="test", score_name="sourceafis_score")

        test = fusion_test.merge(
            source_test,
            on=["dataset", "split", "pair_id"],
            how="inner",
            suffixes=("_fusion", "_sourceafis"),
            validate="one_to_one",
        )
        if len(test) != len(fusion_test):
            raise Phase15ValidationError(
                f"SourceAFIS and fusion TEST rows do not cover the same pairs for {dataset}: "
                f"fusion={len(fusion_test)} joined={len(test)}"
            )
        if not (test["label_fusion"].astype(int) == test["label_sourceafis"].astype(int)).all():
            raise Phase15ValidationError(f"SourceAFIS and fusion TEST labels disagree for {dataset}.")
        labels = test["label_fusion"].astype(int).to_numpy()
        fusion_scores = pd.to_numeric(test["fusion_score"], errors="coerce").to_numpy(dtype=float)
        source_scores = pd.to_numeric(test["sourceafis_score"], errors="coerce").to_numpy(dtype=float)
        positives = labels == 1
        negatives = labels == 0

        for target_far in target_fars:
            fusion_selection = _threshold_from_val_negatives(
                fusion_val,
                target_far=float(target_far),
                score_column="fusion_score",
            )
            source_selection = _threshold_from_val_negatives(
                source_val,
                target_far=float(target_far),
                score_column="sourceafis_score",
            )
            fusion_accept = fusion_scores >= float(fusion_selection.threshold)
            source_accept = source_scores >= float(source_selection.threshold)
            fusion_counts = final.compute_confusion(labels, fusion_scores, float(fusion_selection.threshold))
            source_counts = final.compute_confusion(labels, source_scores, float(source_selection.threshold))
            fusion_only = fusion_accept & ~source_accept
            source_only = source_accept & ~fusion_accept
            both_accept = fusion_accept & source_accept
            both_reject = ~fusion_accept & ~source_accept
            rows.append(
                {
                    "dataset": dataset,
                    "split": "test",
                    "target_far": float(target_far),
                    "fusion_method": METHOD_NAME,
                    "baseline_method": "sourceafis_open",
                    "fusion_threshold": float(fusion_selection.threshold),
                    "sourceafis_threshold": float(source_selection.threshold),
                    "fusion_threshold_val_far": float(fusion_selection.actual_far),
                    "sourceafis_threshold_val_far": float(source_selection.actual_far),
                    "n_pairs": int(len(test)),
                    "n_positive": int(np.sum(positives)),
                    "n_negative": int(np.sum(negatives)),
                    "fusion_ta": int(fusion_counts["ta"]),
                    "fusion_fr": int(fusion_counts["fr"]),
                    "fusion_fa": int(fusion_counts["fa"]),
                    "fusion_tr": int(fusion_counts["tr"]),
                    "fusion_tar": float(fusion_counts["tar"]),
                    "fusion_far": float(fusion_counts["far"]),
                    "sourceafis_ta": int(source_counts["ta"]),
                    "sourceafis_fr": int(source_counts["fr"]),
                    "sourceafis_fa": int(source_counts["fa"]),
                    "sourceafis_tr": int(source_counts["tr"]),
                    "sourceafis_tar": float(source_counts["tar"]),
                    "sourceafis_far": float(source_counts["far"]),
                    "rescued_positives": int(np.sum(fusion_only & positives)),
                    "lost_positives": int(np.sum(source_only & positives)),
                    "fixed_false_accepts": int(np.sum(source_only & negatives)),
                    "new_false_accepts": int(np.sum(fusion_only & negatives)),
                    "positive_discordant_count": int(np.sum((fusion_accept != source_accept) & positives)),
                    "negative_discordant_count": int(np.sum((fusion_accept != source_accept) & negatives)),
                    "paired_discordant_count": int(np.sum(fusion_accept != source_accept)),
                    "fusion_only_accepts": int(np.sum(fusion_only)),
                    "sourceafis_only_accepts": int(np.sum(source_only)),
                    "both_accept": int(np.sum(both_accept)),
                    "both_reject": int(np.sum(both_reject)),
                    "fusion_scores_csv": str(fusion_test_path),
                    "sourceafis_scores_csv": str(source_test_path),
                }
            )

    comparison = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    csv_path = outdir / "plain_roll_final_statistical_comparison.csv"
    md_path = outdir / "plain_roll_final_statistical_comparison.md"
    final._write_csv(csv_path, comparison, COMPARISON_COLUMNS)

    lines = [
        "# Fusion vs SourceAFIS Paired Statistical Comparison",
        "",
        "Thresholds for both methods are selected independently from VAL negatives at the same target FAR, then applied to the same TEST pairs.",
        "",
        "| dataset | target FAR | rescued positives | lost positives | fixed false accepts | new false accepts | positive discordants | negative discordants |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comparison.sort_values(["dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {_fmt_pct(row['target_far'])} | {int(row['rescued_positives'])} | "
            f"{int(row['lost_positives'])} | {int(row['fixed_false_accepts'])} | "
            f"{int(row['new_false_accepts'])} | {int(row['positive_discordant_count'])} | "
            f"{int(row['negative_discordant_count'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"statistical_comparison": csv_path, "statistical_comparison_markdown": md_path}


def run_full_pairs_validation(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    outdir: Path = DEFAULT_FULL_PAIRS_OUTDIR,
    model_dir: Path = fusion_benchmark.DEFAULT_MODEL_DIR,
    sourceafis_score_dir: Path = DEFAULT_FULL_SOURCEAFIS_SCORE_DIR,
    sift_plain_roll_score_dir: Path = DEFAULT_FULL_SIFT_SCORE_DIR,
    sift_score_dir: Path | None = None,
    target_fars: tuple[float, ...] = DEFAULT_TARGET_FARS,
    repo_root: Path = REPO_ROOT,
    generate_missing_scores: bool = False,
    generate_sourceafis_scores: bool = True,
    generate_sift_scores: bool = True,
    sourceafis_template_cache_dir: Path | None = None,
    request_timeout_seconds: float | None = None,
    extract_timeout_seconds: float | None = None,
    verify_timeout_seconds: float | None = None,
    max_retries: int = sourceafis.DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = sourceafis.DEFAULT_RETRY_BACKOFF_SECONDS,
    dpi_strategy: str = sourceafis.DEFAULT_DPI_STRATEGY,
    image_dpi: int | None = None,
) -> dict[str, Path]:
    output = parse_file_uri(outdir, repo_root=repo_root)
    model_root = parse_file_uri(model_dir, repo_root=repo_root)
    sourceafis_dir = parse_file_uri(sourceafis_score_dir, repo_root=repo_root)
    sift_plain_roll_dir = parse_file_uri(sift_plain_roll_score_dir, repo_root=repo_root)
    sift_dir = parse_file_uri(sift_score_dir, repo_root=repo_root) if sift_score_dir is not None else None

    ensure_full_pair_scores(
        datasets=datasets,
        splits=splits,
        sourceafis_score_dir=sourceafis_dir,
        sift_plain_roll_score_dir=sift_plain_roll_dir,
        repo_root=repo_root,
        generate_missing_scores=bool(generate_missing_scores),
        generate_sourceafis_scores=bool(generate_sourceafis_scores),
        generate_sift_scores=bool(generate_sift_scores),
        sourceafis_template_cache_dir=sourceafis_template_cache_dir,
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
        max_retries=int(max_retries),
        retry_backoff_seconds=float(retry_backoff_seconds),
        dpi_strategy=dpi_strategy,
        image_dpi=image_dpi,
    )
    paths = fusion_benchmark.run_benchmark(
        datasets=datasets,
        splits=splits,
        outdir=output,
        model_dir=model_root,
        pair_scope="full",
        sourceafis_score_dir=sourceafis_dir,
        sift_plain_roll_score_dir=sift_plain_roll_dir,
        sift_score_dir=sift_dir,
        target_fars=tuple(float(x) for x in target_fars),
        limit_per_split=0,
        repo_root=repo_root,
    )
    comparison_paths = write_sourceafis_statistical_comparison(
        outdir=output,
        datasets=datasets,
        target_fars=tuple(float(x) for x in target_fars),
        sourceafis_score_dir=sourceafis_dir,
    )
    paths.update(comparison_paths)
    summary_path = paths.get("summary")
    if summary_path is not None and Path(summary_path).exists():
        comparison_md = comparison_paths["statistical_comparison_markdown"].read_text(encoding="utf-8")
        with Path(summary_path).open("a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(comparison_md)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 1.5 validation for sourceafis_sift_quality_fusion_v1."
    )
    parser.add_argument("--mode", choices=("ablation", "full", "all"), default="all")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--variants", default="", help="Optional comma-separated subset of ablation variants.")
    parser.add_argument("--train_score_dir", default=str(fusion_train.DEFAULT_TRAIN_SCORE_DIR))
    parser.add_argument("--ablation_outdir", default=str(DEFAULT_ABLATION_OUTDIR))
    parser.add_argument("--full_outdir", default=str(DEFAULT_FULL_PAIRS_OUTDIR))
    parser.add_argument("--model_dir", default=str(fusion_benchmark.DEFAULT_MODEL_DIR))
    parser.add_argument("--pair_scope", choices=("selected", "full"), default="full")
    parser.add_argument(
        "--selected_pairs_dir",
        default="",
        help="Optional non-artifact exact-pairs directory, only valid with --pair_scope selected.",
    )
    parser.add_argument("--selected_sourceafis_score_dir", default=str(DEFAULT_SELECTED_SOURCEAFIS_SCORE_DIR))
    parser.add_argument("--selected_sift_plain_roll_score_dir", default=str(DEFAULT_SELECTED_SIFT_SCORE_DIR))
    parser.add_argument("--full_sourceafis_score_dir", default=str(DEFAULT_FULL_SOURCEAFIS_SCORE_DIR))
    parser.add_argument("--full_sift_plain_roll_score_dir", default=str(DEFAULT_FULL_SIFT_SCORE_DIR))
    parser.add_argument("--sift_score_dir", default="")
    parser.add_argument("--target_far", type=float, nargs="*", default=list(DEFAULT_TARGET_FARS))
    parser.add_argument("--limit_per_split", type=int, default=0)
    parser.add_argument("--generate_missing_scores", action="store_true")
    parser.add_argument("--no_generate_sourceafis", action="store_true")
    parser.add_argument("--no_generate_sift", action="store_true")
    parser.add_argument("--sourceafis_template_cache_dir", default="")
    parser.add_argument("--request_timeout_seconds", type=float, default=None)
    parser.add_argument("--extract_timeout_seconds", type=float, default=None)
    parser.add_argument("--verify_timeout_seconds", type=float, default=None)
    parser.add_argument("--max_retries", type=int, default=sourceafis.DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_backoff_seconds", type=float, default=sourceafis.DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--dpi_strategy", choices=sourceafis.DPI_STRATEGIES, default=sourceafis.DEFAULT_DPI_STRATEGY)
    parser.add_argument("--image_dpi", type=int, default=None)
    parser.add_argument("--random_state", type=int, default=13)
    parser.add_argument("--save_training_features", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = _parse_csv_arg(args.datasets)
    splits = tuple(split.lower() for split in _parse_csv_arg(args.splits))
    target_fars = tuple(float(item) for item in args.target_far)
    variant_names = _parse_csv_arg(args.variants) if str(args.variants).strip() else None
    sift_score_dir = parse_file_uri(args.sift_score_dir) if str(args.sift_score_dir).strip() else None
    template_cache = (
        parse_file_uri(args.sourceafis_template_cache_dir)
        if str(args.sourceafis_template_cache_dir).strip()
        else None
    )
    try:
        all_paths: dict[str, Path] = {}
        ablation_paths: dict[str, Path] = {}
        if args.mode in {"ablation", "all"}:
            ablation_paths = run_ablation_validation(
                datasets=datasets,
                splits=splits,
                variants=variant_names,
                train_score_dir=parse_file_uri(args.train_score_dir),
                outdir=parse_file_uri(args.ablation_outdir),
                pair_scope=str(args.pair_scope),
                selected_pairs_dir=parse_file_uri(args.selected_pairs_dir)
                if str(args.selected_pairs_dir).strip()
                else None,
                sourceafis_score_dir=parse_file_uri(args.selected_sourceafis_score_dir),
                sift_plain_roll_score_dir=parse_file_uri(args.selected_sift_plain_roll_score_dir),
                sift_score_dir=sift_score_dir,
                target_fars=target_fars,
                limit_per_split=int(args.limit_per_split),
                repo_root=REPO_ROOT,
                generate_missing_scores=bool(args.generate_missing_scores),
                generate_sourceafis_scores=not bool(args.no_generate_sourceafis),
                generate_sift_scores=not bool(args.no_generate_sift),
                sourceafis_template_cache_dir=template_cache,
                request_timeout_seconds=args.request_timeout_seconds,
                extract_timeout_seconds=args.extract_timeout_seconds,
                verify_timeout_seconds=args.verify_timeout_seconds,
                max_retries=int(args.max_retries),
                retry_backoff_seconds=float(args.retry_backoff_seconds),
                dpi_strategy=str(args.dpi_strategy),
                image_dpi=args.image_dpi,
                random_state=int(args.random_state),
                save_training_features=bool(args.save_training_features),
            )
            all_paths.update({f"ablation_{key}": value for key, value in ablation_paths.items()})
        if args.mode in {"full", "all"}:
            requested_model_dir = parse_file_uri(args.model_dir)
            if (
                args.mode == "all"
                and str(parse_file_uri(args.model_dir)) == str(parse_file_uri(str(fusion_benchmark.DEFAULT_MODEL_DIR)))
                and "model_sourceafis_plus_sift_quality_full" in ablation_paths
            ):
                requested_model_dir = Path(ablation_paths["model_sourceafis_plus_sift_quality_full"]).parent
            full_paths = run_full_pairs_validation(
                datasets=datasets,
                splits=splits,
                outdir=parse_file_uri(args.full_outdir),
                model_dir=requested_model_dir,
                sourceafis_score_dir=parse_file_uri(args.full_sourceafis_score_dir),
                sift_plain_roll_score_dir=parse_file_uri(args.full_sift_plain_roll_score_dir),
                sift_score_dir=sift_score_dir,
                target_fars=target_fars,
                repo_root=REPO_ROOT,
                generate_missing_scores=bool(args.generate_missing_scores),
                generate_sourceafis_scores=not bool(args.no_generate_sourceafis),
                generate_sift_scores=not bool(args.no_generate_sift),
                sourceafis_template_cache_dir=template_cache,
                request_timeout_seconds=args.request_timeout_seconds,
                extract_timeout_seconds=args.extract_timeout_seconds,
                verify_timeout_seconds=args.verify_timeout_seconds,
                max_retries=int(args.max_retries),
                retry_backoff_seconds=float(args.retry_backoff_seconds),
                dpi_strategy=str(args.dpi_strategy),
                image_dpi=args.image_dpi,
            )
            all_paths.update({f"full_{key}": value for key, value in full_paths.items()})
    except (
        Phase15ValidationError,
        calibration.FusionCalibrationError,
        fusion_benchmark.FusionBenchmarkError,
        fusion_train.FusionTrainingError,
        sourceafis.SourceAfisBenchmarkError,
    ) as exc:
        print(f"Phase 1.5 validation failed: {exc}", file=sys.stderr)
        return 2

    print("Wrote Phase 1.5 validation artifacts:")
    for path in all_paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
