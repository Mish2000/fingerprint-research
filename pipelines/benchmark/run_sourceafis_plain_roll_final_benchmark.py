from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark import run_plain_roll_final_benchmark as final
from scripts.diagnostics import run_sourceafis_plain_roll_benchmark as sourceafis
from src.fpbench.fingerprint_engine.base import FingerprintEngine
from src.fpbench.universal.pair_bundle_metadata import build_pair_bundle_metadata, is_artifact_selected_pairs_path


PROVIDER_ID = "sourceafis_open"
OUTPUT_SCHEMA_VERSION = "plain_roll_final_sourceafis_benchmark_v1"
DEFAULT_SELECTED_PAIRS_DIR: Path | None = None
DEFAULT_PAIR_AUDIT_DIR: Path | None = None
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_final_sourceafis_v1"
)
DEFAULT_RAW_SOURCEAFIS_OUTDIR = DEFAULT_OUTDIR / "run_meta" / "sourceafis_raw"
DEFAULT_DATASETS = final.DEFAULT_DATASETS
DEFAULT_SPLITS = final.DEFAULT_SPLITS
DEFAULT_TARGET_FARS = final.DEFAULT_TARGET_FARS


class SourceAfisFinalBenchmarkError(RuntimeError):
    """Raised when the SourceAFIS final evidence bundle cannot be produced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_csv_arg(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _sourceafis_score_path(raw_outdir: Path, split: str) -> Path:
    return raw_outdir / f"sourceafis_plain_roll_scores_{split}.csv"


def _sourceafis_outputs_complete(raw_outdir: Path, splits: tuple[str, ...]) -> bool:
    required = [
        _sourceafis_score_path(raw_outdir, split)
        for split in splits
    ]
    required.extend(
        [
            raw_outdir / "sourceafis_plain_roll_thresholds.csv",
            raw_outdir / "sourceafis_plain_roll_metrics.csv",
            raw_outdir / "sourceafis_plain_roll_latency_summary.csv",
            raw_outdir / "sourceafis_plain_roll_failures.csv",
            raw_outdir / "sourceafis_plain_roll_manifest.json",
        ]
    )
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _load_sourceafis_scores(raw_outdir: Path, splits: tuple[str, ...]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = _sourceafis_score_path(raw_outdir, split)
        if not path.exists():
            raise SourceAfisFinalBenchmarkError(f"Missing SourceAFIS raw score CSV: {path}")
        frame = pd.read_csv(path)
        if "raw_score" not in frame.columns:
            raise SourceAfisFinalBenchmarkError(f"SourceAFIS raw score CSV lacks raw_score column: {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_selected_pairs(
    *,
    selected_pairs_dir: Path | None,
    outdir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    repo_root: Path,
) -> tuple[dict[tuple[str, str], Path], dict[tuple[str, str], Path]]:
    target_dir = outdir / "selected_pairs"
    selected_paths: dict[tuple[str, str], Path] = {}
    source_paths: dict[tuple[str, str], Path] = {}
    if selected_pairs_dir is not None and is_artifact_selected_pairs_path(selected_pairs_dir, repo_root=repo_root):
        raise SourceAfisFinalBenchmarkError(
            "Refusing to use artifacts/reports/**/selected_pairs as SourceAFIS final input. "
            "Omit --selected_pairs_dir to use canonical data/manifests pairs."
        )
    for dataset in datasets:
        for split in splits:
            if selected_pairs_dir is None:
                src = final._pairs_path(dataset, split, repo_root=repo_root)
                if src is None:
                    raise SourceAfisFinalBenchmarkError(
                        f"Missing canonical pair CSV for dataset={dataset!r} split={split!r}"
                    )
            else:
                src = selected_pairs_dir / f"pairs_{dataset}_{split}.csv"
            if not src.exists():
                raise SourceAfisFinalBenchmarkError(f"Missing pair CSV: {src}")
            dst = target_dir / f"pairs_{dataset}_{split}.csv"
            _copy_file(src, dst)
            selected_paths[(dataset, split)] = dst
            source_paths[(dataset, split)] = src
    return selected_paths, source_paths


def _copy_pair_audits(
    *,
    pair_audit_dir: Path,
    outdir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
) -> None:
    if not pair_audit_dir.exists():
        return
    target_dir = outdir / "pair_audit"
    _copy_file(pair_audit_dir / "pair_audit_summary.md", target_dir / "pair_audit_summary.md")
    for dataset in datasets:
        for split in splits:
            stem = f"pair_audit_{dataset}_{split}"
            _copy_file(pair_audit_dir / f"{stem}.json", target_dir / f"{stem}.json")
            _copy_file(pair_audit_dir / f"{stem}.md", target_dir / f"{stem}.md")


def _finger_column(df: pd.DataFrame) -> str:
    for column in ("finger_position", "frgp", "finger_id"):
        if column in df.columns:
            return column
    raise SourceAfisFinalBenchmarkError("Selected pairs are missing finger_position/frgp/finger_id.")


def _normalized_pair_keys(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    finger_col = _finger_column(df)
    required = ["pair_id", "label", "subject_a", "subject_b", "path_a", "path_b"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise SourceAfisFinalBenchmarkError(f"Pair evidence is missing required columns: {missing}")
    out = pd.DataFrame(
        {
            "dataset": dataset,
            "split": split,
            "pair_id": df["pair_id"].astype(str),
            "label": pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int).astype(str),
            "subject_a": df["subject_a"].astype(str),
            "subject_b": df["subject_b"].astype(str),
            "finger_position": df[finger_col].astype(str),
            "path_a": df["path_a"].astype(str),
            "path_b": df["path_b"].astype(str),
        }
    )
    return out.reset_index(drop=True)


def _pair_key_digest(keys: pd.DataFrame) -> str:
    text = keys.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validate_score_alignment(
    *,
    source_scores: pd.DataFrame,
    selected_pairs_csv: Path,
    pair_source_csv: Path,
    dataset: str,
    split: str,
    repo_root: Path,
) -> dict[str, Any]:
    selected = pd.read_csv(selected_pairs_csv)
    selected_keys = _normalized_pair_keys(selected, dataset=dataset, split=split)
    score_subset = source_scores[
        (source_scores["dataset"].astype(str) == dataset)
        & (source_scores["split"].astype(str).str.lower() == split.lower())
    ].copy()
    score_keys = _normalized_pair_keys(score_subset, dataset=dataset, split=split)
    if len(score_keys) != len(selected_keys):
        raise SourceAfisFinalBenchmarkError(
            f"SourceAFIS score row count for {dataset}/{split} does not match selected pairs: "
            f"{len(score_keys)} != {len(selected_keys)}"
        )
    if not score_keys.equals(selected_keys):
        mismatch_index = next(
            (index for index in range(len(selected_keys)) if not score_keys.iloc[index].equals(selected_keys.iloc[index])),
            None,
        )
        raise SourceAfisFinalBenchmarkError(
            f"SourceAFIS score pair order/content does not match selected pairs for {dataset}/{split}"
            + (f" at row {mismatch_index}." if mismatch_index is not None else ".")
        )
    metadata = build_pair_bundle_metadata(
        dataset=dataset,
        split=split,
        pair_source_path=pair_source_csv,
        repo_root=repo_root,
    )
    return {
        "dataset": dataset,
        "split": split,
        "selected_pairs_csv": str(selected_pairs_csv),
        "materialized_pairs_path": str(selected_pairs_csv),
        "materialized_pairs_sha256": sourceafis.file_sha256(selected_pairs_csv),
        "selected_pairs_row_count": int(len(selected_keys)),
        "sourceafis_score_row_count": int(len(score_keys)),
        "selected_pairs_sha256": sourceafis.file_sha256(selected_pairs_csv),
        "pair_key_sha256": _pair_key_digest(selected_keys),
        "status": "match",
        **metadata,
    }


def _numeric_column(scores: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in scores.columns:
        return pd.Series(default, index=scores.index, dtype=float)
    return pd.to_numeric(scores[column], errors="coerce")


def _pair_total_ms(scores: pd.DataFrame) -> pd.Series:
    extraction_a = _numeric_column(scores, "extraction_latency_ms_a").fillna(0.0)
    extraction_b = _numeric_column(scores, "extraction_latency_ms_b").fillna(0.0)
    verification_wall = _numeric_column(scores, "verification_wall_latency_ms", default=float("nan"))
    verification_reported = _numeric_column(scores, "verification_latency_ms", default=float("nan"))
    verification = verification_wall.where(np.isfinite(verification_wall), verification_reported).fillna(0.0)
    return extraction_a + extraction_b + verification


def _finite(values: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _write_method_meta(path: Path, scores: pd.DataFrame, pair_metadata: dict[str, Any] | None = None) -> None:
    values = _finite(scores["pair_total_ms"]) if "pair_total_ms" in scores.columns else np.asarray([], dtype=float)
    cache_hits = 0
    cache_misses = 0
    for column in ("extraction_cache_hit_a", "extraction_cache_hit_b"):
        if column not in scores.columns:
            continue
        hits = scores[column].astype(str).str.lower().isin({"true", "1", "yes"})
        cache_hits += int(hits.sum())
        cache_misses += int((~hits).sum())
    payload = {
        "method": PROVIDER_ID,
        "avg_ms_pair": float(np.mean(values)) if values.size else float("nan"),
        "p50_ms_pair": float(np.median(values)) if values.size else float("nan"),
        "p95_ms_pair": float(np.quantile(values, 0.95)) if values.size else float("nan"),
        "total_ms": float(np.sum(values)) if values.size else float("nan"),
        "template_cache": {
            "hits": int(cache_hits),
            "misses": int(cache_misses),
        },
    }
    if pair_metadata:
        payload.update(pair_metadata)
        payload["pair_bundle_metadata"] = pair_metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_run_meta(
    path: Path,
    *,
    scores: pd.DataFrame,
    method_meta_json: Path,
    pair_metadata: dict[str, Any] | None = None,
) -> None:
    values = _finite(scores["pair_total_ms"]) if "pair_total_ms" in scores.columns else np.asarray([], dtype=float)
    avg = float(np.mean(values)) if values.size else float("nan")
    payload = {
        "method": PROVIDER_ID,
        "created_at": _utc_now(),
        "timing": {
            "avg_ms_pair_reported": avg,
            "avg_ms_pair_wall": avg,
        },
        "method_meta_json": str(method_meta_json),
        "row": {
            "avg_ms_pair_reported": avg,
            "avg_ms_pair_wall": avg,
            "meta_json": str(method_meta_json),
        },
    }
    if pair_metadata:
        payload.update(pair_metadata)
        payload["pair_bundle_metadata"] = pair_metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _write_sourceafis_score_runs(
    *,
    source_scores: pd.DataFrame,
    selected_paths: dict[tuple[str, str], Path],
    pair_source_paths: dict[tuple[str, str], Path],
    outdir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    repo_root: Path,
) -> tuple[list[final.ScoreRun], list[dict[str, Any]]]:
    runs: list[final.ScoreRun] = []
    validations: list[dict[str, Any]] = []
    score_dir = outdir / "scores"
    run_meta_dir = outdir / "run_meta"
    roc_dir = outdir / "roc"
    score_dir.mkdir(parents=True, exist_ok=True)
    run_meta_dir.mkdir(parents=True, exist_ok=True)
    roc_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for split in splits:
            selected_pairs_csv = selected_paths[(dataset, split)]
            validation = _validate_score_alignment(
                source_scores=source_scores,
                selected_pairs_csv=selected_pairs_csv,
                pair_source_csv=pair_source_paths[(dataset, split)],
                dataset=dataset,
                split=split,
                repo_root=repo_root,
            )
            validations.append(validation)
            subset = source_scores[
                (source_scores["dataset"].astype(str) == dataset)
                & (source_scores["split"].astype(str).str.lower() == split.lower())
            ].copy()
            subset.insert(0, "method", PROVIDER_ID)
            subset["score"] = pd.to_numeric(subset["raw_score"], errors="coerce")
            subset["pair_total_ms"] = _pair_total_ms(subset)
            score_csv = score_dir / f"scores_{dataset}_{PROVIDER_ID}_{split}.csv"
            run_meta_json = run_meta_dir / f"run_{dataset}_{PROVIDER_ID}_{split}.meta.json"
            method_meta_json = score_csv.with_suffix(".meta.json")
            columns = [
                "method",
                "dataset",
                "split",
                "pair_id",
                "label",
                "subject_a",
                "subject_b",
                "finger_position",
                "path_a",
                "path_b",
                "score",
                "raw_score",
                "score_semantics",
                "higher_is_more_similar",
                "dpi_a",
                "dpi_b",
                "provider_id",
                "provider_version",
                "template_format",
                "template_version",
                "extraction_cache_hit_a",
                "extraction_cache_hit_b",
                "extraction_latency_ms_a",
                "extraction_latency_ms_b",
                "verification_latency_ms",
                "verification_wall_latency_ms",
                "pair_total_ms",
                "warnings",
                "error",
            ]
            for column in columns:
                if column not in subset.columns:
                    subset[column] = None
            subset[columns].to_csv(score_csv, index=False)
            pair_metadata = dict(validation)
            pair_metadata.update(
                {
                    "method": PROVIDER_ID,
                    "scores_csv": str(score_csv),
                    "score_count": int(len(subset)),
                }
            )
            _write_method_meta(method_meta_json, subset, pair_metadata=pair_metadata)
            _write_run_meta(
                run_meta_json,
                scores=subset,
                method_meta_json=method_meta_json,
                pair_metadata=pair_metadata,
            )
            runs.append(
                final.ScoreRun(
                    method=PROVIDER_ID,
                    dataset=dataset,
                    split=split,
                    selected_pairs_csv=selected_pairs_csv,
                    scores_csv=score_csv,
                    roc_png=roc_dir / f"roc_{dataset}_{PROVIDER_ID}_{split}.png",
                    run_meta_json=run_meta_json,
                    command=[],
                    elapsed_seconds=None,
                    reused_existing_scores=True,
                )
            )
    return runs, validations


def _read_sourceafis_failures(raw_outdir: Path) -> pd.DataFrame:
    path = raw_outdir / "sourceafis_plain_roll_failures.csv"
    if not path.exists():
        return pd.DataFrame(columns=final.FAILURE_COLUMNS)
    raw = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        rows.append(
            {
                "method": PROVIDER_ID,
                "dataset": row.get("dataset", ""),
                "split": row.get("split", ""),
                "status": "failed",
                "error_type": row.get("error_type", row.get("failure_category", "")),
                "error_message": row.get("error_message", ""),
                "returncode": "",
                "command": "",
                "scores_csv": "",
                "run_meta_json": "",
            }
        )
    return pd.DataFrame(rows, columns=final.FAILURE_COLUMNS)


def _load_pair_audit_reports(
    *,
    outdir: Path,
    selected_paths: dict[tuple[str, str], Path],
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    repo_root: Path,
) -> list[final.PairAuditReport]:
    audit_dir = outdir / "pair_audit"
    reports: list[final.PairAuditReport] = []
    missing = False
    for dataset in datasets:
        for split in splits:
            json_path = audit_dir / f"pair_audit_{dataset}_{split}.json"
            markdown_path = audit_dir / f"pair_audit_{dataset}_{split}.md"
            if not json_path.exists():
                missing = True
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            reports.append(
                final.PairAuditReport(
                    dataset=dataset,
                    split=split,
                    selected_pairs_csv=selected_paths[(dataset, split)],
                    json_path=json_path,
                    markdown_path=markdown_path,
                    summary=payload,
                )
            )
    if not missing:
        return reports

    reports = final.write_pair_audits(
        selected_pairs=selected_paths,
        pair_audit_out=audit_dir,
        repo_root=repo_root,
    )
    final.write_pair_audit_summary_markdown(reports, audit_dir / "pair_audit_summary.md")
    return reports


def _write_manifest_with_sourceafis_metadata(
    path: Path,
    *,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    target_fars: tuple[float, ...],
    pair_audit_reports: list[final.PairAuditReport],
    score_runs: list[final.ScoreRun],
    failures: pd.DataFrame,
    output_paths: dict[str, Path],
    total_runtime_s: float,
    repo_root: Path,
    selected_pair_validations: list[dict[str, Any]],
    raw_sourceafis_outdir: Path,
    sourceafis_raw_reused: bool,
) -> None:
    dataset_statuses = []
    for row in selected_pair_validations:
        selected_rows = pd.read_csv(row["selected_pairs_csv"])
        status = dict(row)
        status.update(
            {
                "compatible": True,
                "reason": "SourceAFIS scored exact audited canonical pairs",
                "n_pairs": int(row["selected_pairs_row_count"]),
                "n_positive": int(
                    pd.to_numeric(selected_rows["label"], errors="coerce").fillna(-1).eq(1).sum()
                ),
                "n_negative": int(
                    pd.to_numeric(selected_rows["label"], errors="coerce").fillna(-1).eq(0).sum()
                ),
                "selected_pairs_csv": row["selected_pairs_csv"],
                "selected_pairs_row_count": int(row["selected_pairs_row_count"]),
                "selected_pairs_sha256": row["selected_pairs_sha256"],
            }
        )
        dataset_statuses.append(status)
    final.write_manifest(
        path,
        datasets=datasets,
        methods=(PROVIDER_ID,),
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
        sample_strategy="selected_pairs_exact",
        sample_seed=0,
        select_pairs_only=False,
        strict_pair_audit=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = OUTPUT_SCHEMA_VERSION
    payload["sourceafis_final_bundle"] = {
        "provider_id": PROVIDER_ID,
        "raw_sourceafis_outdir": str(raw_sourceafis_outdir),
        "sourceafis_raw_reused": bool(sourceafis_raw_reused),
        "sourceafis_rerun_required": not bool(sourceafis_raw_reused),
        "selected_pair_validation": selected_pair_validations,
    }
    payload["python"] = sys.version
    payload["platform"] = platform.platform()
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _sourceafis_raw_matches_selected_pairs(
    *,
    raw_outdir: Path,
    selected_pairs_dir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
) -> bool:
    manifest_path = raw_outdir / "sourceafis_plain_roll_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    statuses = payload.get("datasets", [])
    if not isinstance(statuses, list):
        return False
    by_key = {
        (str(item.get("dataset")), str(item.get("split")).lower()): item
        for item in statuses
        if isinstance(item, dict)
    }
    for dataset in datasets:
        for split in splits:
            selected_path = selected_pairs_dir / f"pairs_{dataset}_{split}.csv"
            if not selected_path.exists():
                return False
            expected = sourceafis.file_sha256(selected_path)
            status = by_key.get((dataset, split.lower()))
            if not status or str(status.get("selected_pairs_sha256", "")) != expected:
                return False
    return True


def _ensure_sourceafis_raw_outputs(
    *,
    raw_outdir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    selected_pairs_dir: Path,
    target_fars: tuple[float, ...],
    force_rerun: bool,
    engine: FingerprintEngine | None,
    repo_root: Path,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
) -> bool:
    if not force_rerun and _sourceafis_outputs_complete(raw_outdir, splits) and _sourceafis_raw_matches_selected_pairs(
        raw_outdir=raw_outdir,
        selected_pairs_dir=selected_pairs_dir,
        datasets=datasets,
        splits=splits,
    ):
        return True

    sourceafis.run_benchmark(
        datasets=datasets,
        splits=splits,
        outdir=raw_outdir,
        target_fars=target_fars,
        limit_per_split=0,
        engine=engine,
        require_enabled_env=engine is None,
        repo_root=repo_root,
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        dpi_strategy=dpi_strategy,
        image_dpi=image_dpi,
        selected_pairs_dir=selected_pairs_dir,
        allow_artifact_selected_pairs_dir=True,
    )
    return False


def run_benchmark(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    outdir: Path = DEFAULT_OUTDIR,
    selected_pairs_dir: Path | None = DEFAULT_SELECTED_PAIRS_DIR,
    pair_audit_dir: Path | None = DEFAULT_PAIR_AUDIT_DIR,
    sourceafis_outdir: Path = DEFAULT_RAW_SOURCEAFIS_OUTDIR,
    target_fars: tuple[float, ...] = DEFAULT_TARGET_FARS,
    force_rerun_sourceafis: bool = False,
    engine: FingerprintEngine | None = None,
    repo_root: Path = REPO_ROOT,
    request_timeout_seconds: float | None = None,
    extract_timeout_seconds: float | None = None,
    verify_timeout_seconds: float | None = None,
    max_retries: int = sourceafis.DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = sourceafis.DEFAULT_RETRY_BACKOFF_SECONDS,
    dpi_strategy: str = sourceafis.DEFAULT_DPI_STRATEGY,
    image_dpi: int | None = None,
) -> dict[str, Path]:
    start = time.perf_counter()
    output = sourceafis.parse_file_uri(outdir, repo_root=repo_root)
    selected_dir = sourceafis.parse_file_uri(selected_pairs_dir, repo_root=repo_root) if selected_pairs_dir is not None else None
    raw_outdir = sourceafis.parse_file_uri(sourceafis_outdir, repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=True)

    selected_paths, pair_source_paths = _copy_selected_pairs(
        selected_pairs_dir=selected_dir,
        outdir=output,
        datasets=datasets,
        splits=splits,
        repo_root=repo_root,
    )
    pair_audit_reports = _load_pair_audit_reports(
        outdir=output,
        selected_paths=selected_paths,
        datasets=datasets,
        splits=splits,
        repo_root=repo_root,
    )

    raw_reused = _ensure_sourceafis_raw_outputs(
        raw_outdir=raw_outdir,
        datasets=datasets,
        splits=splits,
        selected_pairs_dir=output / "selected_pairs",
        target_fars=target_fars,
        force_rerun=bool(force_rerun_sourceafis),
        engine=engine,
        repo_root=repo_root,
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        dpi_strategy=dpi_strategy,
        image_dpi=image_dpi,
    )
    source_scores = _load_sourceafis_scores(raw_outdir, splits)
    score_runs, selected_pair_validations = _write_sourceafis_score_runs(
        source_scores=source_scores,
        selected_paths=selected_paths,
        pair_source_paths=pair_source_paths,
        outdir=output,
        datasets=datasets,
        splits=splits,
        repo_root=repo_root,
    )

    latency = final.build_latency_rows(score_runs)
    thresholds = final.build_threshold_table(score_runs, target_fars)
    metrics = final.build_metrics_table(score_runs, thresholds, latency)
    positive_only_metrics = final.build_positive_only_metrics_table(metrics)
    negative_only_metrics = final.build_negative_only_metrics_table(metrics)
    threshold_sweep = final.build_threshold_sweep_table(score_runs, latency)
    tar_far_distribution = final.build_tar_far_distribution_table(threshold_sweep)
    failures = _read_sourceafis_failures(raw_outdir)

    paths: dict[str, Path] = {
        "thresholds": output / "plain_roll_final_thresholds.csv",
        "metrics": output / "plain_roll_final_metrics.csv",
        "positive_only_metrics": output / "plain_roll_final_positive_only_metrics.csv",
        "negative_only_metrics": output / "plain_roll_final_negative_only_metrics.csv",
        "threshold_sweep": output / "plain_roll_final_threshold_sweep.csv",
        "tar_far_distribution": output / "plain_roll_final_tar_far_distribution.csv",
        "latency_summary": output / "plain_roll_final_latency_summary.csv",
        "failures": output / "plain_roll_final_failures.csv",
        "summary": output / "plain_roll_final_summary.md",
        "manifest": output / "plain_roll_final_manifest.json",
        "raw_sourceafis_manifest": raw_outdir / "sourceafis_plain_roll_manifest.json",
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

    markdown_dir = output / "final_markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        md_path = markdown_dir / f"{dataset}_{PROVIDER_ID}_plain_roll_final.md"
        md_path.write_text(
            final.render_method_dataset_markdown(
                method=PROVIDER_ID,
                dataset=dataset,
                metrics=metrics,
                thresholds=thresholds,
                tar_far_distribution=tar_far_distribution,
                latency=latency,
                pair_audit_reports=pair_audit_reports,
            ),
            encoding="utf-8",
        )
        paths[f"markdown_{dataset}_{PROVIDER_ID}"] = md_path

    total_runtime_s = time.perf_counter() - start
    paths["summary"].write_text(
        final.render_combined_markdown(
            metrics=metrics,
            thresholds=thresholds,
            tar_far_distribution=tar_far_distribution,
            latency=latency,
            failures=failures,
            dataset_statuses=[
                {
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "n_pairs": row["selected_pairs_row_count"],
                    "n_positive": "",
                    "n_negative": "",
                    "pairs_csv": row["selected_pairs_csv"],
                    "selected_pairs_csv": row["selected_pairs_csv"],
                }
                for row in selected_pair_validations
            ],
            pair_audit_reports=pair_audit_reports,
            output_paths=paths,
            total_runtime_s=total_runtime_s,
        ),
        encoding="utf-8",
    )
    _write_manifest_with_sourceafis_metadata(
        paths["manifest"],
        datasets=datasets,
        splits=splits,
        target_fars=target_fars,
        pair_audit_reports=pair_audit_reports,
        score_runs=score_runs,
        failures=failures,
        output_paths=paths,
        total_runtime_s=total_runtime_s,
        repo_root=repo_root,
        selected_pair_validations=selected_pair_validations,
        raw_sourceafis_outdir=raw_outdir,
        sourceafis_raw_reused=raw_reused,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build final SourceAFIS plain-vs-roll evidence from audited selected pairs. "
            "Existing SourceAFIS raw outputs are reused when complete unless --force_rerun_sourceafis is set."
        )
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--selected_pairs_dir",
        default="",
        help="Optional non-artifact exact-pairs directory. Default uses canonical data/manifests pairs.",
    )
    parser.add_argument(
        "--pair_audit_dir",
        default="",
        help="Deprecated; pair audits are regenerated from the pair source used by this run.",
    )
    parser.add_argument("--sourceafis_outdir", default=str(DEFAULT_RAW_SOURCEAFIS_OUTDIR))
    parser.add_argument("--target_far", type=float, nargs="*", default=list(DEFAULT_TARGET_FARS))
    parser.add_argument("--force_rerun_sourceafis", action="store_true")
    parser.add_argument("--request_timeout_seconds", type=float, default=None)
    parser.add_argument("--extract_timeout_seconds", type=float, default=None)
    parser.add_argument("--verify_timeout_seconds", type=float, default=None)
    parser.add_argument("--max_retries", type=int, default=sourceafis.DEFAULT_MAX_RETRIES)
    parser.add_argument("--retry_backoff_seconds", type=float, default=sourceafis.DEFAULT_RETRY_BACKOFF_SECONDS)
    parser.add_argument("--dpi_strategy", choices=sourceafis.DPI_STRATEGIES, default=sourceafis.DEFAULT_DPI_STRATEGY)
    parser.add_argument("--image_dpi", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_benchmark(
            datasets=_parse_csv_arg(args.datasets),
            splits=tuple(item.lower() for item in _parse_csv_arg(args.splits)),
            outdir=sourceafis.parse_file_uri(args.outdir),
            selected_pairs_dir=sourceafis.parse_file_uri(args.selected_pairs_dir)
            if str(args.selected_pairs_dir).strip()
            else None,
            pair_audit_dir=sourceafis.parse_file_uri(args.pair_audit_dir)
            if str(args.pair_audit_dir).strip()
            else None,
            sourceafis_outdir=sourceafis.parse_file_uri(args.sourceafis_outdir),
            target_fars=tuple(float(item) for item in args.target_far),
            force_rerun_sourceafis=bool(args.force_rerun_sourceafis),
            request_timeout_seconds=args.request_timeout_seconds,
            extract_timeout_seconds=args.extract_timeout_seconds,
            verify_timeout_seconds=args.verify_timeout_seconds,
            max_retries=int(args.max_retries),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
            dpi_strategy=str(args.dpi_strategy),
            image_dpi=args.image_dpi,
        )
    except (SourceAfisFinalBenchmarkError, sourceafis.SourceAfisBenchmarkError) as exc:
        print(f"SourceAFIS final plain/roll benchmark failed: {exc}", file=sys.stderr)
        return 2

    print("Wrote SourceAFIS final Plain/Roll benchmark artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
