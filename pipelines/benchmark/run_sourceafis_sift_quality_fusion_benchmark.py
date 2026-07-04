from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark import run_plain_roll_final_benchmark as final
from src.fpbench.universal import calibration
from src.fpbench.universal.fusion_features import METHOD_NAME, PairScoreSpec, build_feature_table
from src.fpbench.universal.pair_bundle_metadata import (
    SD300_DATASETS,
    build_pair_bundle_metadata,
    file_sha256,
    is_artifact_selected_pairs_path,
)


OUTPUT_SCHEMA_VERSION = "plain_roll_final_fusion_benchmark_v1"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_TARGET_FARS = calibration.DEFAULT_TARGET_FARS
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_final_fusion_v1_v2_anatomical_full_pairs"
)
DEFAULT_MODEL_DIR = DEFAULT_OUTDIR / "model"
DEFAULT_SELECTED_PAIRS_DIR: Path | None = None
DEFAULT_SOURCEAFIS_SCORE_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_final_sourceafis_v2_anatomical_full_pairs"
    / "scores"
)
DEFAULT_SIFT_PLAIN_ROLL_SCORE_DIR = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_final_baselines_v2_anatomical_full_pairs"
)
SOURCEAFIS_COMPARISON_COLUMNS = [
    "dataset",
    "split",
    "target_far",
    "sourceafis_tar",
    "sourceafis_far",
    "fusion_v1_tar",
    "fusion_v1_far",
    "delta_tar_pp",
    "delta_far_pp",
    "sourceafis_threshold",
    "fusion_v1_threshold",
    "sourceafis_metrics_csv",
    "fusion_metrics_csv",
]


class FusionBenchmarkError(RuntimeError):
    """Raised when the fusion benchmark cannot be produced."""


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_file_uri(raw: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _pairs_path(dataset: str, split: str, *, repo_root: Path = REPO_ROOT) -> Path:
    candidates = [
        repo_root / "data" / "manifests" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "manifests" / dataset / "pairs" / f"pairs_{split}.csv",
        repo_root / "data" / "processed" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "processed" / dataset / "pairs" / f"pairs_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FusionBenchmarkError(f"Could not locate pairs_{split}.csv for dataset={dataset!r}.")


def _score_path_candidates(score_dir: Path, dataset: str, method: str, split: str) -> list[Path]:
    name = f"scores_{dataset}_{method}_{split}.csv"
    method_name = f"scores_{method}_{split}.csv"
    candidates = [
        score_dir / name,
        score_dir / "scores" / name,
        score_dir / method_name,
        score_dir / "scores" / method_name,
    ]
    if method == "sourceafis_open":
        raw_name = f"sourceafis_plain_roll_scores_{split}.csv"
        candidates.extend([score_dir / raw_name, score_dir / "scores" / raw_name])
    return candidates


def _resolve_score_path(score_dir: Path, dataset: str, method: str, split: str) -> Path:
    candidates = _score_path_candidates(score_dir, dataset, method, split)
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    raise FusionBenchmarkError(
        f"Missing score CSV for dataset={dataset!r} method={method!r} split={split!r}. "
        f"Checked: {[str(path) for path in candidates]}"
    )


def _optional_score_path(score_dir: Path | None, dataset: str, method: str, split: str) -> Path | None:
    if score_dir is None:
        return None
    for candidate in _score_path_candidates(score_dir, dataset, method, split):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _score_meta_candidates(score_csv: Path) -> list[Path]:
    return [score_csv.with_suffix(".meta.json"), Path(str(score_csv) + ".meta.json")]


def _resolve_score_meta(score_csv: Path) -> Path | None:
    for candidate in _score_meta_candidates(score_csv):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _metadata_value(meta: dict[str, Any], key: str) -> Any:
    if key in meta:
        return meta.get(key)
    nested = meta.get("pair_bundle_metadata")
    if isinstance(nested, dict):
        return nested.get(key)
    return None


def _source_score_hash_record(score_csv: Path) -> dict[str, Any]:
    meta_path = _resolve_score_meta(score_csv)
    return {
        "score_csv": str(score_csv),
        "score_csv_sha256": file_sha256(score_csv),
        "meta_json": str(meta_path) if meta_path is not None else "",
        "meta_json_sha256": file_sha256(meta_path) if meta_path is not None else "",
    }


def _validate_source_score_metadata(
    *,
    dataset: str,
    split: str,
    method: str,
    pairs_csv: Path,
    score_csv: Path,
    repo_root: Path,
) -> dict[str, Any]:
    meta_path = _resolve_score_meta(score_csv)
    if meta_path is None:
        if dataset in SD300_DATASETS:
            raise FusionBenchmarkError(f"Missing required score meta JSON for {dataset}/{split}/{method}: {score_csv}")
        return {
            "dataset": dataset,
            "split": split,
            "method": method,
            "score_csv": str(score_csv),
            "meta_json": "",
            "status": "meta_missing_non_sd300",
        }

    meta = _read_json(meta_path)
    pair_bundle = build_pair_bundle_metadata(
        dataset=dataset,
        split=split,
        pair_source_path=pairs_csv,
        repo_root=repo_root,
    )
    expected = {
        "method": method,
        "dataset_id": dataset,
        "split": split,
        "pair_source_sha256": pair_bundle["pair_source_sha256"],
        "manifest_source_sha256": pair_bundle["manifest_source_sha256"],
        "split_subjects_sha256": pair_bundle["split_subjects_sha256"],
        "sd300_frgp_semantics": "anatomical" if dataset in SD300_DATASETS else "dataset_native",
        "run_pair_bundle_version": pair_bundle["run_pair_bundle_version"],
    }
    mismatches: dict[str, dict[str, str]] = {}
    for key, expected_value in expected.items():
        actual = _metadata_value(meta, key)
        if str(actual) != str(expected_value):
            mismatches[key] = {"expected": str(expected_value), "actual": str(actual)}
    if mismatches:
        raise FusionBenchmarkError(
            f"Score metadata mismatch for {dataset}/{split}/{method}: {mismatches}. meta={meta_path}"
        )

    return {
        "dataset": dataset,
        "split": split,
        "method": method,
        "score_csv": str(score_csv),
        "meta_json": str(meta_path),
        "status": "pass",
        **_source_score_hash_record(score_csv),
        "pair_source_sha256": pair_bundle["pair_source_sha256"],
        "manifest_source_sha256": pair_bundle["manifest_source_sha256"],
        "split_subjects_sha256": pair_bundle["split_subjects_sha256"],
    }


def _sourceafis_metrics_csv_for_score_dir(sourceafis_score_dir: Path) -> Path:
    root = sourceafis_score_dir.parent if sourceafis_score_dir.name.lower() == "scores" else sourceafis_score_dir
    return root / "plain_roll_final_metrics.csv"


def _limit_pairs_for_smoke(pairs: pd.DataFrame, limit: int) -> pd.DataFrame:
    if int(limit) <= 0 or len(pairs) <= int(limit):
        return pairs.copy()
    labels = pd.to_numeric(pairs.get("label"), errors="coerce").fillna(-1).astype(int)
    positive_target = int(math.ceil(int(limit) / 2))
    negative_target = int(math.floor(int(limit) / 2))
    positives = pairs[labels == 1].head(positive_target)
    negatives = pairs[labels == 0].head(negative_target)
    limited = pd.concat([positives, negatives], axis=0).sort_index()
    if len(limited) < int(limit):
        limited = pd.concat([limited, pairs.drop(index=limited.index, errors="ignore").head(int(limit) - len(limited))])
    return limited.head(int(limit)).reset_index(drop=True)


def _materialize_pairs(
    *,
    dataset: str,
    split: str,
    pair_scope: str,
    selected_pairs_dir: Path | None,
    outdir: Path,
    repo_root: Path,
    limit_per_split: int,
) -> tuple[Path, dict[str, Any]]:
    if pair_scope == "selected":
        if selected_pairs_dir is None:
            raise FusionBenchmarkError("--selected_pairs_dir is required when --pair_scope selected is used.")
        if is_artifact_selected_pairs_path(selected_pairs_dir, repo_root=repo_root):
            raise FusionBenchmarkError(
                "Refusing to use artifacts/reports/**/selected_pairs as fusion benchmark input. "
                "Use --pair_scope full for canonical data/manifests pairs."
            )
        source = selected_pairs_dir / f"pairs_{dataset}_{split}.csv"
        if not source.exists():
            raise FusionBenchmarkError(f"Missing selected pairs CSV: {source}")
    elif pair_scope == "full":
        source = _pairs_path(dataset, split, repo_root=repo_root)
    else:
        raise FusionBenchmarkError(f"Unsupported pair_scope: {pair_scope!r}")

    target = outdir / "selected_pairs" / f"pairs_{dataset}_{split}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    source_rows = pd.read_csv(source)
    rows = _limit_pairs_for_smoke(source_rows, int(limit_per_split))
    if int(limit_per_split) <= 0:
        shutil.copy2(source, target)
    else:
        rows.to_csv(target, index=False)

    labels = pd.to_numeric(rows.get("label"), errors="coerce").fillna(-1).astype(int)
    status = {
        "dataset": dataset,
        "split": split,
        "compatible": bool(len(rows)),
        "reason": f"{pair_scope} pair scope" + (" smoke subset" if int(limit_per_split) > 0 else ""),
        "pairs_csv": str(source),
        "selected_pairs_csv": str(target),
        "n_pairs": int(len(rows)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "source_is_selected_pairs": pair_scope == "selected",
        "pair_scope": pair_scope,
        "limit_per_split": int(limit_per_split),
    }
    status.update(
        build_pair_bundle_metadata(
            dataset=dataset,
            split=split,
            pair_source_path=source,
            repo_root=repo_root,
        )
    )
    status["materialized_pairs_path"] = str(target)
    status["materialized_pairs_sha256"] = file_sha256(target)
    return target, status


def _write_method_meta(
    path: Path,
    *,
    scores: pd.DataFrame,
    scores_csv: Path,
    total_ms: float,
    pair_bundle: dict[str, Any],
) -> None:
    pair_times = pd.to_numeric(scores.get("pair_total_ms"), errors="coerce").dropna()
    run_level_avg_ms = float(total_ms / max(int(len(scores)), 1))
    payload = {
        "schema_version": "sourceafis_sift_quality_fusion_score_meta_v1",
        "method": METHOD_NAME,
        "avg_ms_pair": float(pair_times.mean()) if len(pair_times) else run_level_avg_ms,
        "p50_ms_pair": float(pair_times.median()) if len(pair_times) else None,
        "p95_ms_pair": float(pair_times.quantile(0.95)) if len(pair_times) else None,
        "total_ms": float(total_ms),
        "score_semantics": "logistic_regression_positive_class_probability",
        "latency_scope": "run_level_average_only",
        "pair_total_ms_semantics": "not_measured_per_pair",
        **pair_bundle,
        "scores_csv": str(scores_csv),
        "score_count": int(len(scores)),
        "pair_bundle_metadata": pair_bundle,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_run_meta(
    path: Path,
    *,
    scores: pd.DataFrame,
    method_meta_json: Path,
    total_ms: float,
    pair_bundle: dict[str, Any],
) -> None:
    n_pairs = max(int(len(scores)), 1)
    avg_ms = float(total_ms / n_pairs)
    payload = {
        "schema_version": "sourceafis_sift_quality_fusion_run_meta_v1",
        "method": METHOD_NAME,
        "created_at": _utc_now(),
        "method_meta_json": str(method_meta_json),
        "timing": {
            "avg_ms_pair_reported": avg_ms,
            "avg_ms_pair_wall": avg_ms,
            "total_ms": float(total_ms),
            "n_pairs": int(len(scores)),
            "latency_scope": "run_level_average_only",
            "pair_total_ms_semantics": "not_measured_per_pair",
        },
        "row": {
            "avg_ms_pair_reported": avg_ms,
            "avg_ms_pair_wall": avg_ms,
            "meta_json": str(method_meta_json),
        },
        **pair_bundle,
        "pair_bundle": pair_bundle,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _score_split(
    *,
    dataset: str,
    split: str,
    pairs_csv: Path,
    sourceafis_scores_csv: Path,
    sift_plain_roll_scores_csv: Path,
    sift_scores_csv: Path | None,
    model: Any,
    schema: dict[str, Any],
    outdir: Path,
    repo_root: Path,
) -> tuple[final.ScoreRun, pd.DataFrame]:
    start = time.perf_counter()
    feature_table = build_feature_table(
        dataset=dataset,
        split=split,
        pairs_csv=pairs_csv,
        sourceafis_scores_csv=sourceafis_scores_csv,
        sift_plain_roll_scores_csv=sift_plain_roll_scores_csv,
        sift_scores_csv=sift_scores_csv,
        repo_root=repo_root,
    )
    fusion_scores = calibration.predict_fusion_scores(model, schema, feature_table)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    per_pair_ms = float(elapsed_ms / max(int(len(feature_table)), 1))

    score_dir = outdir / "scores"
    run_meta_dir = outdir / "run_meta"
    roc_dir = outdir / "roc"
    score_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    roc_dir.mkdir(parents=True, exist_ok=True)

    scores_csv = score_dir / f"scores_{dataset}_{METHOD_NAME}_{split}.csv"
    run_meta_json = run_meta_dir / f"run_{dataset}_{METHOD_NAME}_{split}.meta.json"
    method_meta_json = scores_csv.with_suffix(".meta.json")
    pair_bundle = build_pair_bundle_metadata(
        dataset=dataset,
        split=split,
        pair_source_path=pairs_csv,
        repo_root=repo_root,
    )

    base_columns = [
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
        "sourceafis_score",
        "sift_plain_roll_v2_score",
        "sift_score",
        "ppi",
        "ppi_a",
        "ppi_b",
    ]
    output = feature_table.copy()
    output["score"] = fusion_scores
    output["score_semantics"] = "logistic_regression_positive_class_probability"
    output["higher_is_more_similar"] = True
    output["run_level_avg_pair_ms"] = per_pair_ms
    output["pair_total_ms"] = float("nan")
    output["pair_total_ms_semantics"] = "not_measured_per_pair; run_level_avg_pair_ms repeats batch average"
    for column in (
        "pair_source_sha256",
        "manifest_source_sha256",
        "split_subjects_sha256",
        "run_pair_bundle_version",
        "sd300_frgp_semantics",
        "sd300_raw_frgp_available",
    ):
        output[column] = pair_bundle.get(column, "")
    for column in base_columns:
        if column not in output.columns:
            output[column] = ""
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
        "sift_score",
        "ppi",
        "ppi_a",
        "ppi_b",
        "run_level_avg_pair_ms",
        "pair_total_ms",
        "pair_total_ms_semantics",
        "pair_source_sha256",
        "manifest_source_sha256",
        "split_subjects_sha256",
        "run_pair_bundle_version",
        "sd300_frgp_semantics",
        "sd300_raw_frgp_available",
    ]
    output[score_columns].to_csv(scores_csv, index=False)
    _write_method_meta(
        method_meta_json,
        scores=output,
        scores_csv=scores_csv,
        total_ms=elapsed_ms,
        pair_bundle=pair_bundle,
    )
    _write_run_meta(
        run_meta_json,
        scores=output,
        method_meta_json=method_meta_json,
        total_ms=elapsed_ms,
        pair_bundle=pair_bundle,
    )

    run = final.ScoreRun(
        method=METHOD_NAME,
        dataset=dataset,
        split=split,
        selected_pairs_csv=pairs_csv,
        scores_csv=scores_csv,
        roc_png=roc_dir / f"roc_{dataset}_{METHOD_NAME}_{split}.png",
        run_meta_json=run_meta_json,
        command=[],
        elapsed_seconds=float(elapsed_ms / 1000.0),
        reused_existing_scores=False,
    )
    score_frame = output[["method", "dataset", "split", "pair_id", "label", "score"]].copy()
    return run, score_frame


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


def _write_manifest(
    path: Path,
    *,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    target_fars: tuple[float, ...],
    dataset_statuses: list[dict[str, Any]],
    pair_audit_reports: list[final.PairAuditReport],
    score_runs: list[final.ScoreRun],
    failures: pd.DataFrame,
    output_paths: dict[str, Path],
    total_runtime_s: float,
    repo_root: Path,
    pair_scope: str,
    model_dir: Path,
    sourceafis_score_dir: Path,
    sift_plain_roll_score_dir: Path,
    sift_score_dir: Path | None,
    source_score_file_sha256s: dict[str, Any],
    source_score_alignment_checks: list[dict[str, Any]],
    sourceafis_metrics_csv: Path,
) -> None:
    final.write_manifest(
        path,
        datasets=datasets,
        methods=(METHOD_NAME,),
        splits=splits,
        target_fars=target_fars,
        dataset_statuses=dataset_statuses,
        pair_audit_reports=pair_audit_reports,
        score_runs=score_runs,
        failures=failures,
        output_paths=output_paths,
        total_runtime_s=total_runtime_s,
        repo_root=repo_root,
        limit_per_split=0,
        sample_strategy=f"{pair_scope}_pairs_exact",
        sample_seed=0,
        select_pairs_only=False,
        strict_pair_audit=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = OUTPUT_SCHEMA_VERSION
    payload["git"] = _git_info(repo_root)
    payload["python"] = sys.version
    payload["platform"] = platform.platform()
    payload["fusion_method"] = {
        "method": METHOD_NAME,
        "pair_scope": pair_scope,
        "model_dir": str(model_dir),
        "model_artifacts": {
            "fusion_model": str(model_dir / "fusion_model.joblib"),
            "feature_schema": str(model_dir / "feature_schema.json"),
            "training_manifest": str(model_dir / "training_manifest.json"),
        },
        "score_sources": {
            "sourceafis_score_dir": str(sourceafis_score_dir),
            "sift_plain_roll_v2_score_dir": str(sift_plain_roll_score_dir),
            "sift_score_dir": str(sift_score_dir) if sift_score_dir is not None else "",
        },
        "training_protocol": "model fit on train only",
        "threshold_protocol": "thresholds selected from VAL negative fusion scores only",
        "test_protocol": "frozen model and frozen VAL thresholds are applied to TEST",
        "source_score_file_sha256s": source_score_file_sha256s,
    }
    payload["run_pair_bundle_version"] = "sd300_anatomical_full_pairs_v2" if set(datasets) & SD300_DATASETS else ""
    payload["sd300_frgp_semantics"] = "anatomical" if set(datasets) & SD300_DATASETS else "dataset_native"
    payload["sd300_raw_frgp_available"] = bool(set(datasets) & SD300_DATASETS)
    payload["trained_on_splits"] = ["train"]
    payload["thresholds_selected_on"] = "val"
    payload["test_used_for_training"] = False
    payload["legacy_scores_used"] = False
    payload["artifact_selected_pairs_used_as_input"] = False
    pair_source_sha256s: dict[str, dict[str, str]] = {dataset: {} for dataset in datasets}
    manifest_source_sha256s: dict[str, str] = {}
    split_subjects_sha256s: dict[str, str] = {}
    for status in dataset_statuses:
        dataset = str(status.get("dataset", ""))
        split = str(status.get("split", ""))
        if dataset:
            pair_source_sha256s.setdefault(dataset, {})[split] = str(status.get("pair_source_sha256", ""))
            manifest_source_sha256s.setdefault(dataset, str(status.get("manifest_source_sha256", "")))
            split_subjects_sha256s.setdefault(dataset, str(status.get("split_subjects_sha256", "")))
    payload["pair_source_sha256s"] = pair_source_sha256s
    payload["manifest_source_sha256s"] = manifest_source_sha256s
    payload["split_subjects_sha256s"] = split_subjects_sha256s
    payload["source_score_file_sha256s"] = source_score_file_sha256s
    payload["source_score_alignment_checks"] = source_score_alignment_checks
    payload["sourceafis_v2_comparison"] = {
        "sourceafis_metrics_csv": str(sourceafis_metrics_csv),
        "comparison_csv": str(output_paths.get("sourceafis_comparison", "")),
        "comparison_markdown": str(output_paths.get("sourceafis_comparison_markdown", "")),
        "sourceafis_score_dir": str(sourceafis_score_dir),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _summary_prefix(*, pair_scope: str, model_dir: Path) -> str:
    return (
        f"# {METHOD_NAME}\n\n"
        f"Pair scope: `{pair_scope}`. Model artifacts: `{model_dir}`.\n\n"
        "The fusion score is the positive-class probability from the frozen train-only logistic model. "
        "Operating thresholds are selected from validation negatives only and applied unchanged to test.\n\n"
        "Latency is reported as a run-level average only for this fusion runner; score CSV rows do not contain "
        "true per-pair fusion latency measurements.\n\n"
    )


def _write_sourceafis_comparison(
    *,
    csv_path: Path,
    markdown_path: Path,
    fusion_metrics: pd.DataFrame,
    sourceafis_metrics_csv: Path,
    fusion_metrics_csv: Path,
    require: bool,
) -> bool:
    if not sourceafis_metrics_csv.exists():
        if require:
            raise FusionBenchmarkError(f"Missing SourceAFIS v2 metrics CSV for comparison: {sourceafis_metrics_csv}")
        return False

    source = pd.read_csv(sourceafis_metrics_csv)
    required_columns = {"method", "dataset", "split", "target_far", "tar", "far", "threshold"}
    missing = sorted(required_columns - set(source.columns))
    if missing:
        raise FusionBenchmarkError(f"SourceAFIS metrics CSV is missing columns {missing}: {sourceafis_metrics_csv}")

    source_rows = source[source["method"].astype(str) == "sourceafis_open"][
        ["dataset", "split", "target_far", "tar", "far", "threshold"]
    ].copy()
    fusion_rows = fusion_metrics[fusion_metrics["method"].astype(str) == METHOD_NAME][
        ["dataset", "split", "target_far", "tar", "far", "threshold"]
    ].copy()
    for frame in (source_rows, fusion_rows):
        frame["target_far_key"] = pd.to_numeric(frame["target_far"], errors="coerce").round(12)
        frame["dataset"] = frame["dataset"].astype(str)
        frame["split"] = frame["split"].astype(str)

    source_rows = source_rows.rename(
        columns={
            "tar": "sourceafis_tar",
            "far": "sourceafis_far",
            "threshold": "sourceafis_threshold",
        }
    )
    fusion_rows = fusion_rows.rename(
        columns={
            "tar": "fusion_v1_tar",
            "far": "fusion_v1_far",
            "threshold": "fusion_v1_threshold",
        }
    )
    merged = fusion_rows.merge(
        source_rows,
        on=["dataset", "split", "target_far_key"],
        how="inner",
        suffixes=("_fusion", "_sourceafis"),
        validate="one_to_one",
    )
    if merged.empty and require:
        raise FusionBenchmarkError(f"No matching SourceAFIS v2 metric rows found in {sourceafis_metrics_csv}")

    rows = pd.DataFrame(
        {
            "dataset": merged["dataset"],
            "split": merged["split"],
            "target_far": merged["target_far_fusion"],
            "sourceafis_tar": merged["sourceafis_tar"],
            "sourceafis_far": merged["sourceafis_far"],
            "fusion_v1_tar": merged["fusion_v1_tar"],
            "fusion_v1_far": merged["fusion_v1_far"],
            "delta_tar_pp": (merged["fusion_v1_tar"] - merged["sourceafis_tar"]) * 100.0,
            "delta_far_pp": (merged["fusion_v1_far"] - merged["sourceafis_far"]) * 100.0,
            "sourceafis_threshold": merged["sourceafis_threshold"],
            "fusion_v1_threshold": merged["fusion_v1_threshold"],
            "sourceafis_metrics_csv": str(sourceafis_metrics_csv),
            "fusion_metrics_csv": str(fusion_metrics_csv),
        }
    )
    rows = rows.sort_values(["dataset", "split", "target_far"], kind="stable")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(csv_path, index=False, columns=SOURCEAFIS_COMPARISON_COLUMNS)

    lines = [
        "# SourceAFIS v2 Comparison",
        "",
        "Positive delta TAR means Fusion v1 accepted more genuine pairs at that operating point.",
        "",
        "| dataset | split | target_far | SourceAFIS TAR | SourceAFIS FAR | Fusion v1 TAR | Fusion v1 FAR | delta TAR pp | delta FAR pp |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            "| {dataset} | {split} | {target_far:.3%} | {sourceafis_tar:.6f} | {sourceafis_far:.6f} | "
            "{fusion_tar:.6f} | {fusion_far:.6f} | {delta_tar:.3f} | {delta_far:.3f} |".format(
                dataset=row.dataset,
                split=row.split,
                target_far=float(row.target_far),
                sourceafis_tar=float(row.sourceafis_tar),
                sourceafis_far=float(row.sourceafis_far),
                fusion_tar=float(row.fusion_v1_tar),
                fusion_far=float(row.fusion_v1_far),
                delta_tar=float(row.delta_tar_pp),
                delta_far=float(row.delta_far_pp),
            )
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def run_benchmark(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    outdir: Path = DEFAULT_OUTDIR,
    model_dir: Path = DEFAULT_MODEL_DIR,
    pair_scope: str = "full",
    selected_pairs_dir: Path | None = DEFAULT_SELECTED_PAIRS_DIR,
    sourceafis_score_dir: Path = DEFAULT_SOURCEAFIS_SCORE_DIR,
    sift_plain_roll_score_dir: Path = DEFAULT_SIFT_PLAIN_ROLL_SCORE_DIR,
    sift_score_dir: Path | None = None,
    target_fars: tuple[float, ...] = DEFAULT_TARGET_FARS,
    limit_per_split: int = 0,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    if "val" not in tuple(split.lower() for split in splits):
        raise FusionBenchmarkError("Fusion benchmark requires val split for threshold calibration.")
    start = time.perf_counter()
    output = parse_file_uri(outdir, repo_root=repo_root)
    model_root = parse_file_uri(model_dir, repo_root=repo_root)
    selected_dir = parse_file_uri(selected_pairs_dir, repo_root=repo_root) if selected_pairs_dir is not None else None
    sourceafis_dir = parse_file_uri(sourceafis_score_dir, repo_root=repo_root)
    sift_plain_roll_dir = parse_file_uri(sift_plain_roll_score_dir, repo_root=repo_root)
    sift_dir = parse_file_uri(sift_score_dir, repo_root=repo_root) if sift_score_dir is not None else None
    output.mkdir(parents=True, exist_ok=True)

    model, schema = calibration.load_model_bundle(model_root)
    selected_paths: dict[tuple[str, str], Path] = {}
    dataset_statuses: list[dict[str, Any]] = []
    for dataset in datasets:
        for split in splits:
            pairs_csv, status = _materialize_pairs(
                dataset=dataset,
                split=split,
                pair_scope=pair_scope,
                selected_pairs_dir=selected_dir,
                outdir=output,
                repo_root=repo_root,
                limit_per_split=int(limit_per_split),
            )
            selected_paths[(dataset, split)] = pairs_csv
            dataset_statuses.append(status)

    pair_audit_dir = output / "pair_audit"
    pair_audit_reports = final.write_pair_audits(
        selected_pairs=selected_paths,
        pair_audit_out=pair_audit_dir,
        repo_root=repo_root,
    )
    final.write_pair_audit_summary_markdown(pair_audit_reports, pair_audit_dir / "pair_audit_summary.md")

    score_runs: list[final.ScoreRun] = []
    score_frames: list[pd.DataFrame] = []
    score_csv_lookup: dict[tuple[str, str], str] = {}
    source_score_alignment_checks: list[dict[str, Any]] = []
    source_score_file_sha256s: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for dataset in datasets:
        for split in splits:
            sourceafis_scores = _resolve_score_path(sourceafis_dir, dataset, "sourceafis_open", split)
            sift_plain_roll_scores = _resolve_score_path(sift_plain_roll_dir, dataset, "sift_plain_roll_v2", split)
            sift_scores = _optional_score_path(sift_dir, dataset, "sift", split)
            source_specs = {
                "sourceafis_open": sourceafis_scores,
                "sift_plain_roll_v2": sift_plain_roll_scores,
            }
            if sift_scores is not None:
                source_specs["sift"] = sift_scores
            for method, score_csv in source_specs.items():
                check = _validate_source_score_metadata(
                    dataset=dataset,
                    split=split,
                    method=method,
                    pairs_csv=selected_paths[(dataset, split)],
                    score_csv=score_csv,
                    repo_root=repo_root,
                )
                source_score_alignment_checks.append(check)
                source_score_file_sha256s.setdefault(dataset, {}).setdefault(split, {})[method] = _source_score_hash_record(
                    score_csv
                )
            run, score_frame = _score_split(
                dataset=dataset,
                split=split,
                pairs_csv=selected_paths[(dataset, split)],
                sourceafis_scores_csv=sourceafis_scores,
                sift_plain_roll_scores_csv=sift_plain_roll_scores,
                sift_scores_csv=sift_scores,
                model=model,
                schema=schema,
                outdir=output,
                repo_root=repo_root,
            )
            score_runs.append(run)
            score_frames.append(score_frame)
            score_csv_lookup[(dataset, split)] = str(run.scores_csv)

    all_scores = pd.concat(score_frames, ignore_index=True, sort=False)
    thresholds = calibration.build_threshold_table_from_scores(
        all_scores,
        target_fars=target_fars,
        method=METHOD_NAME,
        scores_csv_lookup=score_csv_lookup,
    )
    latency = final.build_latency_rows(score_runs)
    metrics = final.build_metrics_table(score_runs, thresholds, latency)
    positive_only_metrics = final.build_positive_only_metrics_table(metrics)
    negative_only_metrics = final.build_negative_only_metrics_table(metrics)
    threshold_sweep = final.build_threshold_sweep_table(score_runs, latency)
    tar_far_distribution = final.build_tar_far_distribution_table(threshold_sweep)
    failures = pd.DataFrame(columns=final.FAILURE_COLUMNS)

    paths: dict[str, Path] = {
        "thresholds": output / "plain_roll_final_thresholds.csv",
        "metrics": output / "plain_roll_final_metrics.csv",
        "positive_only_metrics": output / "plain_roll_final_positive_only_metrics.csv",
        "negative_only_metrics": output / "plain_roll_final_negative_only_metrics.csv",
        "threshold_sweep": output / "plain_roll_final_threshold_sweep.csv",
        "tar_far_distribution": output / "plain_roll_final_tar_far_distribution.csv",
        "latency_summary": output / "plain_roll_final_latency_summary.csv",
        "failures": output / "plain_roll_final_failures.csv",
        "sourceafis_comparison": output / "plain_roll_final_sourceafis_comparison.csv",
        "sourceafis_comparison_markdown": output / "plain_roll_final_sourceafis_comparison.md",
        "summary": output / "plain_roll_final_summary.md",
        "manifest": output / "plain_roll_final_manifest.json",
        "model": model_root / "fusion_model.joblib",
        "feature_schema": model_root / "feature_schema.json",
        "training_manifest": model_root / "training_manifest.json",
    }
    for (dataset, split), path in selected_paths.items():
        paths[f"selected_pairs_{dataset}_{split}"] = path
    for report in pair_audit_reports:
        paths[f"pair_audit_json_{report.dataset}_{report.split}"] = report.json_path
        paths[f"pair_audit_markdown_{report.dataset}_{report.split}"] = report.markdown_path

    final._write_csv(paths["thresholds"], thresholds, final.THRESHOLD_COLUMNS)
    final._write_csv(paths["metrics"], metrics, final.METRICS_COLUMNS)
    final._write_csv(paths["positive_only_metrics"], positive_only_metrics, final.POSITIVE_ONLY_METRICS_COLUMNS)
    final._write_csv(paths["negative_only_metrics"], negative_only_metrics, final.NEGATIVE_ONLY_METRICS_COLUMNS)
    final._write_csv(paths["threshold_sweep"], threshold_sweep, final.THRESHOLD_SWEEP_COLUMNS)
    final._write_csv(paths["tar_far_distribution"], tar_far_distribution, final.TAR_FAR_DISTRIBUTION_COLUMNS)
    final._write_csv(paths["latency_summary"], latency, final.LATENCY_COLUMNS)
    final._write_csv(paths["failures"], failures, final.FAILURE_COLUMNS)
    sourceafis_metrics_csv = _sourceafis_metrics_csv_for_score_dir(sourceafis_dir)
    comparison_written = _write_sourceafis_comparison(
        csv_path=paths["sourceafis_comparison"],
        markdown_path=paths["sourceafis_comparison_markdown"],
        fusion_metrics=metrics,
        sourceafis_metrics_csv=sourceafis_metrics_csv,
        fusion_metrics_csv=paths["metrics"],
        require=bool(set(datasets) & SD300_DATASETS),
    )
    if not comparison_written:
        paths.pop("sourceafis_comparison", None)
        paths.pop("sourceafis_comparison_markdown", None)

    markdown_dir = output / "final_markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        md_path = markdown_dir / f"{dataset}_{METHOD_NAME}_plain_roll_final.md"
        md_path.write_text(
            final.render_method_dataset_markdown(
                method=METHOD_NAME,
                dataset=dataset,
                metrics=metrics,
                thresholds=thresholds,
                tar_far_distribution=tar_far_distribution,
                latency=latency,
                pair_audit_reports=pair_audit_reports,
            ),
            encoding="utf-8",
        )
        paths[f"markdown_{dataset}_{METHOD_NAME}"] = md_path

    total_runtime_s = time.perf_counter() - start
    comparison_summary = ""
    comparison_markdown = paths.get("sourceafis_comparison_markdown")
    if comparison_markdown is not None and comparison_markdown.exists():
        comparison_summary = "\n" + comparison_markdown.read_text(encoding="utf-8")
    paths["summary"].write_text(
        _summary_prefix(pair_scope=pair_scope, model_dir=model_root)
        + final.render_combined_markdown(
            metrics=metrics,
            thresholds=thresholds,
            tar_far_distribution=tar_far_distribution,
            latency=latency,
            failures=failures,
            dataset_statuses=dataset_statuses,
            pair_audit_reports=pair_audit_reports,
            output_paths=paths,
            total_runtime_s=total_runtime_s,
        )
        + comparison_summary,
        encoding="utf-8",
    )
    _write_manifest(
        paths["manifest"],
        datasets=datasets,
        splits=splits,
        target_fars=target_fars,
        dataset_statuses=dataset_statuses,
        pair_audit_reports=pair_audit_reports,
        score_runs=score_runs,
        failures=failures,
        output_paths=paths,
        total_runtime_s=total_runtime_s,
        repo_root=repo_root,
        pair_scope=pair_scope,
        model_dir=model_root,
        sourceafis_score_dir=sourceafis_dir,
        sift_plain_roll_score_dir=sift_plain_roll_dir,
        sift_score_dir=sift_dir,
        source_score_file_sha256s=source_score_file_sha256s,
        source_score_alignment_checks=source_score_alignment_checks,
        sourceafis_metrics_csv=sourceafis_metrics_csv,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sourceafis_sift_quality_fusion_v1 on canonical full val/test pairs by default."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--pair_scope", choices=("selected", "full"), default="full")
    parser.add_argument(
        "--selected_pairs_dir",
        default="",
        help="Optional non-artifact exact-pairs directory, only valid with --pair_scope selected.",
    )
    parser.add_argument("--sourceafis_score_dir", default=str(DEFAULT_SOURCEAFIS_SCORE_DIR))
    parser.add_argument("--sift_plain_roll_score_dir", default=str(DEFAULT_SIFT_PLAIN_ROLL_SCORE_DIR))
    parser.add_argument(
        "--sift_score_dir",
        default="",
        help="Optional directory containing scores_<dataset>_sift_<split>.csv files.",
    )
    parser.add_argument("--target_far", type=float, nargs="*", default=list(DEFAULT_TARGET_FARS))
    parser.add_argument(
        "--limit_per_split",
        type=int,
        default=0,
        help="Optional smoke-test cap. Default 0 evaluates the exact selected/full pair files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_benchmark(
            datasets=_parse_csv_arg(args.datasets),
            splits=tuple(split.lower() for split in _parse_csv_arg(args.splits)),
            outdir=parse_file_uri(args.outdir),
            model_dir=parse_file_uri(args.model_dir),
            pair_scope=str(args.pair_scope),
            selected_pairs_dir=parse_file_uri(args.selected_pairs_dir)
            if str(args.selected_pairs_dir).strip()
            else None,
            sourceafis_score_dir=parse_file_uri(args.sourceafis_score_dir),
            sift_plain_roll_score_dir=parse_file_uri(args.sift_plain_roll_score_dir),
            sift_score_dir=parse_file_uri(args.sift_score_dir) if str(args.sift_score_dir).strip() else None,
            target_fars=tuple(float(item) for item in args.target_far),
            limit_per_split=int(args.limit_per_split),
            repo_root=REPO_ROOT,
        )
    except (FusionBenchmarkError, calibration.FusionCalibrationError) as exc:
        print(f"Fusion benchmark failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {METHOD_NAME} benchmark artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
