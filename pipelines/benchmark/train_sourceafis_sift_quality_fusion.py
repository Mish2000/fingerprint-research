from __future__ import annotations

import argparse
import json
import os
import platform
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
from scripts.diagnostics import run_sourceafis_plain_roll_benchmark as sourceafis
from src.fpbench.universal import calibration
from src.fpbench.universal.fusion_features import METHOD_NAME, PairScoreSpec, build_feature_tables
from src.fpbench.universal.pair_bundle_metadata import SD300_DATASETS, build_pair_bundle_metadata, file_sha256


DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_TRAIN_SCORE_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_train_scores_v2_anatomical_full_pairs"
    / "scores"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "plain_roll_final_fusion_v1_v2_anatomical_full_pairs"
)
SOURCEAFIS_METHOD = "sourceafis_open"
SIFT_PLAIN_ROLL_METHOD = "sift_plain_roll_v2"
OPTIONAL_SIFT_METHOD = "sift"


class FusionTrainingError(RuntimeError):
    """Raised when fusion training inputs or protocol checks fail."""


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
    raise FusionTrainingError(f"Could not locate pairs_{split}.csv for dataset={dataset!r}.")


def _score_path(score_dir: Path, dataset: str, method: str, split: str) -> Path:
    return score_dir / f"scores_{dataset}_{method}_{split}.csv"


def _sourceafis_train_path(score_dir: Path, dataset: str) -> Path:
    return _score_path(score_dir, dataset, SOURCEAFIS_METHOD, "train")


def _sift_plain_roll_train_path(score_dir: Path, dataset: str) -> Path:
    return _score_path(score_dir, dataset, SIFT_PLAIN_ROLL_METHOD, "train")


def _regular_sift_train_path(score_dir: Path, dataset: str) -> Path:
    return _score_path(score_dir, dataset, OPTIONAL_SIFT_METHOD, "train")


def _run_subprocess(cmd: list[str], *, cwd: Path) -> None:
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(cwd)
    proc = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise FusionTrainingError(
            "Command failed with exit code "
            f"{proc.returncode}: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _attach_pair_traceability(*, dataset: str, pairs_csv: Path, scores_csv: Path) -> None:
    pairs = pd.read_csv(pairs_csv)
    scores = pd.read_csv(scores_csv)
    if len(pairs) != len(scores):
        raise FusionTrainingError(
            f"Score row count does not match pairs for {scores_csv}: scores={len(scores)} pairs={len(pairs)}"
        )
    finger_col = "finger_position" if "finger_position" in pairs.columns else "frgp" if "frgp" in pairs.columns else None
    updated = scores.copy()
    updated["dataset"] = dataset
    for column in ("pair_id", "label", "split", "subject_a", "subject_b", "path_a", "path_b"):
        if column in pairs.columns:
            updated[column] = pairs[column].to_numpy()
    if finger_col is not None:
        updated["finger_position"] = pairs[finger_col].astype(str).to_numpy()
        updated["frgp"] = pairs[finger_col].astype(str).to_numpy()
    updated.to_csv(scores_csv, index=False)


def generate_sift_plain_roll_train_score(
    *,
    dataset: str,
    pairs_csv: Path,
    outdir: Path,
    repo_root: Path,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    run = final.build_evaluate_command(
        method=SIFT_PLAIN_ROLL_METHOD,
        dataset=dataset,
        split="train",
        selected_pairs_csv=pairs_csv,
        outdir=outdir,
        repo_root=repo_root,
    )
    _run_subprocess(run.command, cwd=repo_root)
    _attach_pair_traceability(dataset=dataset, pairs_csv=pairs_csv, scores_csv=run.scores_csv)
    return run.scores_csv


def _sourceafis_pair_total_ms(scores: pd.DataFrame) -> pd.Series:
    extraction_a = pd.to_numeric(scores.get("extraction_latency_ms_a", 0.0), errors="coerce").fillna(0.0)
    extraction_b = pd.to_numeric(scores.get("extraction_latency_ms_b", 0.0), errors="coerce").fillna(0.0)
    verification_wall = pd.to_numeric(scores.get("verification_wall_latency_ms", np.nan), errors="coerce")
    verification_reported = pd.to_numeric(scores.get("verification_latency_ms", np.nan), errors="coerce")
    verification = verification_wall.where(np.isfinite(verification_wall), verification_reported).fillna(0.0)
    return extraction_a + extraction_b + verification


def _write_sourceafis_score_csv(scores: pd.DataFrame, path: Path) -> None:
    out = scores.copy()
    if "method" not in out.columns:
        out.insert(0, "method", SOURCEAFIS_METHOD)
    out["score"] = pd.to_numeric(out["raw_score"], errors="coerce")
    out["pair_total_ms"] = _sourceafis_pair_total_ms(out)
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
        if column not in out.columns:
            out[column] = ""
    path.parent.mkdir(parents=True, exist_ok=True)
    out[columns].to_csv(path, index=False)


def generate_sourceafis_train_scores(
    *,
    datasets: tuple[str, ...],
    outdir: Path,
    repo_root: Path,
    template_cache_dir: Path | None,
    request_timeout_seconds: float | None,
    extract_timeout_seconds: float | None,
    verify_timeout_seconds: float | None,
    max_retries: int,
    retry_backoff_seconds: float,
    dpi_strategy: str,
    image_dpi: int | None,
) -> dict[str, Path]:
    sourceafis.validate_dpi_settings(dpi_strategy=dpi_strategy, image_dpi=image_dpi)
    timeout_overrides = sourceafis._timeout_env_overrides(
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
    )
    previous_env = sourceafis._set_env_overrides(timeout_overrides)
    try:
        engine, provider_metadata = sourceafis.ensure_sourceafis_available(require_enabled_env=True)
        warmup = sourceafis.warmup_sidecar(engine)
        if not bool(warmup.get("ok")):
            raise FusionTrainingError(
                f"SourceAFIS sidecar warmup failed: {warmup.get('error_message') or warmup.get('unavailable_reason')}"
            )
        service_url = str(
            provider_metadata.metadata.get("service_url")
            or os.getenv(sourceafis.SOURCEAFIS_SERVICE_URL_ENV)
            or warmup.get("service_url")
            or ""
        )
        cache = sourceafis.TemplateCache(
            parse_file_uri(template_cache_dir or (outdir / "sourceafis_template_cache"), repo_root=repo_root),
            provider_metadata=provider_metadata,
            service_url=service_url,
            dpi_strategy=dpi_strategy,
            image_dpi=image_dpi,
            repo_root=repo_root,
        )
        retry_config = sourceafis.RetryConfig(
            max_retries=max(int(max_retries), 0),
            retry_backoff_seconds=max(float(retry_backoff_seconds), 0.0),
        )

        generated: dict[str, Path] = {}
        all_failures: list[dict[str, Any]] = []
        for dataset in datasets:
            pairs, status = sourceafis.load_plain_roll_pairs(
                dataset,
                "train",
                repo_root=repo_root,
                limit=0,
                sample_strategy=sourceafis.DEFAULT_SAMPLE_STRATEGY,
                sample_seed=sourceafis.DEFAULT_SAMPLE_SEED,
            )
            if pairs.empty:
                raise FusionTrainingError(f"No SourceAFIS-compatible train pairs for {dataset}: {status}")
            sourceafis.validate_dataset_dpi(
                pairs,
                dataset=dataset,
                dpi_strategy=dpi_strategy,
                image_dpi=image_dpi,
            )
            scored, failures, _events = sourceafis.score_pairs(
                pairs,
                engine=engine,
                cache=cache,
                retry_config=retry_config,
            )
            out_path = _sourceafis_train_path(outdir, dataset)
            _write_sourceafis_score_csv(scored, out_path)
            generated[dataset] = out_path
            all_failures.extend(failures)

        failures_path = outdir / "sourceafis_train_failures.csv"
        pd.DataFrame(all_failures, columns=sourceafis.FAILURE_COLUMNS).to_csv(failures_path, index=False)
        return generated
    finally:
        sourceafis._restore_env(previous_env)


def missing_required_train_scores(*, datasets: tuple[str, ...], score_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for dataset in datasets:
        for path in (_sourceafis_train_path(score_dir, dataset), _sift_plain_roll_train_path(score_dir, dataset)):
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path)
    return missing


def _generation_help(score_dir: Path) -> str:
    return (
        "Missing required train scores. Generate them with:\n"
        "  python pipelines/benchmark/train_sourceafis_sift_quality_fusion.py "
        f"--generate_missing_scores --train_score_dir \"{score_dir}\""
    )


def ensure_train_scores(
    *,
    datasets: tuple[str, ...],
    score_dir: Path,
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
    score_dir.mkdir(parents=True, exist_ok=True)
    missing = missing_required_train_scores(datasets=datasets, score_dir=score_dir)
    if not missing:
        return
    if not generate_missing_scores:
        missing_text = "\n".join(f"  {path}" for path in missing)
        raise FusionTrainingError(f"{_generation_help(score_dir)}\nMissing files:\n{missing_text}")

    if generate_sift_scores:
        for dataset in datasets:
            path = _sift_plain_roll_train_path(score_dir, dataset)
            if path.exists() and path.stat().st_size > 0:
                continue
            generate_sift_plain_roll_train_score(
                dataset=dataset,
                pairs_csv=_pairs_path(dataset, "train", repo_root=repo_root),
                outdir=score_dir,
                repo_root=repo_root,
            )

    if generate_sourceafis_scores:
        sourceafis_missing = [
            dataset
            for dataset in datasets
            if not _sourceafis_train_path(score_dir, dataset).exists()
            or _sourceafis_train_path(score_dir, dataset).stat().st_size == 0
        ]
        if sourceafis_missing:
            generate_sourceafis_train_scores(
                datasets=tuple(sourceafis_missing),
                outdir=score_dir,
                repo_root=repo_root,
                template_cache_dir=sourceafis_template_cache_dir,
                request_timeout_seconds=request_timeout_seconds,
                extract_timeout_seconds=extract_timeout_seconds,
                verify_timeout_seconds=verify_timeout_seconds,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                dpi_strategy=dpi_strategy,
                image_dpi=image_dpi,
            )

    still_missing = missing_required_train_scores(datasets=datasets, score_dir=score_dir)
    if still_missing:
        missing_text = "\n".join(f"  {path}" for path in still_missing)
        raise FusionTrainingError(f"Train score generation finished with missing files:\n{missing_text}")


def build_training_specs(
    *,
    datasets: tuple[str, ...],
    train_score_dir: Path,
    sift_score_dir: Path | None,
    repo_root: Path,
) -> list[PairScoreSpec]:
    specs: list[PairScoreSpec] = []
    for dataset in datasets:
        optional_sift = None
        candidate_dir = sift_score_dir or train_score_dir
        candidate = _regular_sift_train_path(candidate_dir, dataset)
        if candidate.exists() and candidate.stat().st_size > 0:
            optional_sift = candidate
        specs.append(
            PairScoreSpec(
                dataset=dataset,
                split="train",
                pairs_csv=_pairs_path(dataset, "train", repo_root=repo_root),
                sourceafis_scores_csv=_sourceafis_train_path(train_score_dir, dataset),
                sift_plain_roll_scores_csv=_sift_plain_roll_train_path(train_score_dir, dataset),
                sift_scores_csv=optional_sift,
            )
        )
    return specs


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


def train_fusion(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    train_score_dir: Path = DEFAULT_TRAIN_SCORE_DIR,
    outdir: Path = DEFAULT_OUTDIR,
    sift_score_dir: Path | None = None,
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
    save_training_features: bool = False,
    random_state: int = 13,
) -> dict[str, Path]:
    start = time.perf_counter()
    train_scores = parse_file_uri(train_score_dir, repo_root=repo_root)
    output = parse_file_uri(outdir, repo_root=repo_root)
    model_dir = output / "model"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ensure_train_scores(
        datasets=datasets,
        score_dir=train_scores,
        repo_root=repo_root,
        generate_missing_scores=bool(generate_missing_scores),
        generate_sourceafis_scores=bool(generate_sourceafis_scores),
        generate_sift_scores=bool(generate_sift_scores),
        sourceafis_template_cache_dir=sourceafis_template_cache_dir,
        request_timeout_seconds=request_timeout_seconds,
        extract_timeout_seconds=extract_timeout_seconds,
        verify_timeout_seconds=verify_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        dpi_strategy=dpi_strategy,
        image_dpi=image_dpi,
    )

    specs = build_training_specs(
        datasets=datasets,
        train_score_dir=train_scores,
        sift_score_dir=parse_file_uri(sift_score_dir, repo_root=repo_root) if sift_score_dir is not None else None,
        repo_root=repo_root,
    )
    feature_table = build_feature_tables(specs, repo_root=repo_root)
    model, schema = calibration.fit_fusion_model(feature_table, random_state=int(random_state))
    label_counts = feature_table["label"].value_counts().sort_index().to_dict()
    dataset_counts = feature_table.groupby("dataset").size().astype(int).to_dict()

    feature_table_path = model_dir / "training_feature_table.csv"
    if save_training_features:
        feature_table.to_csv(feature_table_path, index=False)

    train_inputs: list[dict[str, Any]] = []
    train_pair_source_sha256: dict[str, str] = {}
    manifest_source_sha256: dict[str, str] = {}
    split_subjects_sha256: dict[str, str] = {}
    source_score_file_sha256s: dict[str, dict[str, str]] = {}
    for spec in specs:
        pair_bundle = build_pair_bundle_metadata(
            dataset=spec.dataset,
            split=spec.split,
            pair_source_path=spec.pairs_csv,
            repo_root=repo_root,
        )
        train_inputs.append(
            {
                "dataset": spec.dataset,
                "pairs_csv": str(spec.pairs_csv),
                "pair_bundle": pair_bundle,
                "sourceafis_scores_csv": str(spec.sourceafis_scores_csv),
                "sift_plain_roll_scores_csv": str(spec.sift_plain_roll_scores_csv),
                "sift_scores_csv": str(spec.sift_scores_csv) if spec.sift_scores_csv is not None else "",
            }
        )
        train_pair_source_sha256[spec.dataset] = str(pair_bundle.get("pair_source_sha256", ""))
        manifest_source_sha256[spec.dataset] = str(pair_bundle.get("manifest_source_sha256", ""))
        split_subjects_sha256[spec.dataset] = str(pair_bundle.get("split_subjects_sha256", ""))
        source_hashes = {
            "sourceafis_train": file_sha256(spec.sourceafis_scores_csv),
            "sift_plain_roll_v2_train": file_sha256(spec.sift_plain_roll_scores_csv),
        }
        if spec.sift_scores_csv is not None:
            source_hashes["sift_train"] = file_sha256(spec.sift_scores_csv)
        source_score_file_sha256s[spec.dataset] = source_hashes

    manifest = {
        "repo_root": str(repo_root),
        "git": _git_info(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "protocol": {
            "fit_splits": ["train"],
            "val_used_for": "model selection and threshold calibration only",
            "test_used_for": "final evaluation only",
            "no_test_leakage": True,
        },
        "run_pair_bundle_version": "sd300_anatomical_full_pairs_v2"
        if set(datasets) & SD300_DATASETS
        else "",
        "sd300_frgp_semantics": "anatomical" if set(datasets) & SD300_DATASETS else "dataset_native",
        "sd300_raw_frgp_available": bool(set(datasets) & SD300_DATASETS),
        "trained_on_splits": ["train"],
        "thresholds_selected_on": "val",
        "test_used_for_training": False,
        "legacy_scores_used": False,
        "artifact_selected_pairs_used_as_input": False,
        "datasets": list(datasets),
        "training_rows": int(len(feature_table)),
        "label_counts": {str(key): int(value) for key, value in label_counts.items()},
        "dataset_counts": {str(key): int(value) for key, value in dataset_counts.items()},
        "train_score_dir": str(train_scores),
        "train_inputs": train_inputs,
        "train_pair_source_sha256": train_pair_source_sha256,
        "manifest_source_sha256": manifest_source_sha256,
        "split_subjects_sha256": split_subjects_sha256,
        "source_score_file_sha256s": source_score_file_sha256s,
        "feature_count": int(len(schema.get("model_features", []))),
        "numeric_feature_count": int(len(schema.get("numeric_features", []))),
        "categorical_features": schema.get("categorical_features", []),
        "model": {
            "type": "sklearn.pipeline.Pipeline",
            "classifier": "LogisticRegression",
            "class_weight": "balanced",
            "scaler": "StandardScaler",
            "random_state": int(random_state),
        },
        "training_feature_table_csv": str(feature_table_path) if save_training_features else "",
        "total_runtime_s": float(time.perf_counter() - start),
        "created_at": _utc_now(),
    }
    paths = calibration.save_model_bundle(model=model, schema=schema, model_dir=model_dir, training_manifest=manifest)
    paths["training_manifest"] = model_dir / "training_manifest.json"
    training_manifest_path = paths["training_manifest"]
    training_payload = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    training_payload["feature_schema_sha256"] = file_sha256(paths["feature_schema"])
    training_payload["model_file_sha256"] = file_sha256(paths["model"])
    training_manifest_path.write_text(json.dumps(training_payload, indent=2, ensure_ascii=True), encoding="utf-8")
    if save_training_features:
        paths["training_feature_table"] = feature_table_path
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train sourceafis_sift_quality_fusion_v1 on full NIST train pairs only."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--train_score_dir", default=str(DEFAULT_TRAIN_SCORE_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--sift_score_dir",
        default="",
        help="Optional directory containing scores_<dataset>_sift_train.csv to include as an extra feature.",
    )
    parser.add_argument(
        "--generate_missing_scores",
        action="store_true",
        help="Generate missing SourceAFIS and SIFT Plain/Roll v2 train score CSVs before fitting.",
    )
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
    parser.add_argument("--save_training_features", action="store_true")
    parser.add_argument("--random_state", type=int, default=13)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = train_fusion(
            datasets=_parse_csv_arg(args.datasets),
            train_score_dir=parse_file_uri(args.train_score_dir),
            outdir=parse_file_uri(args.outdir),
            sift_score_dir=parse_file_uri(args.sift_score_dir) if str(args.sift_score_dir).strip() else None,
            generate_missing_scores=bool(args.generate_missing_scores),
            generate_sourceafis_scores=not bool(args.no_generate_sourceafis),
            generate_sift_scores=not bool(args.no_generate_sift),
            sourceafis_template_cache_dir=parse_file_uri(args.sourceafis_template_cache_dir)
            if str(args.sourceafis_template_cache_dir).strip()
            else None,
            request_timeout_seconds=args.request_timeout_seconds,
            extract_timeout_seconds=args.extract_timeout_seconds,
            verify_timeout_seconds=args.verify_timeout_seconds,
            max_retries=int(args.max_retries),
            retry_backoff_seconds=float(args.retry_backoff_seconds),
            dpi_strategy=str(args.dpi_strategy),
            image_dpi=args.image_dpi,
            save_training_features=bool(args.save_training_features),
            random_state=int(args.random_state),
        )
    except (FusionTrainingError, calibration.FusionCalibrationError, sourceafis.SourceAfisBenchmarkError) as exc:
        print(f"Fusion training failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {METHOD_NAME} model artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
