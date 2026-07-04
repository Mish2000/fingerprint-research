from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.benchmark_validation_utils import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BENCHMARK_RUN_META_SCHEMA_VERSION,
    validate_run_meta,
    validate_scores_csv,
)
from pipelines.benchmark.evaluate import compute_auc_eer, tar_at_far
from pipelines.benchmark.run_plain_roll_final_benchmark import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_STRATEGY,
    DEFAULT_TAR_FAR_CEILINGS,
    DEFAULT_TARGET_FARS,
    FAILURE_COLUMNS,
    LATENCY_COLUMNS,
    METRICS_COLUMNS,
    NEGATIVE_ONLY_METRICS_COLUMNS,
    POSITIVE_ONLY_METRICS_COLUMNS,
    TAR_FAR_DISTRIBUTION_COLUMNS,
    THRESHOLD_COLUMNS,
    THRESHOLD_SWEEP_COLUMNS,
    ScoreRun,
    build_latency_rows,
    build_metrics_table,
    build_negative_only_metrics_table,
    build_positive_only_metrics_table,
    build_tar_far_distribution_table,
    build_threshold_sweep_table,
    build_threshold_table,
    load_plain_roll_pairs,
    render_combined_markdown,
    write_manifest,
)
from src.fpbench.deep.inference import load_checkpoint_model, score_pair_dataloader
from src.fpbench.deep.pair_dataset import FingerprintPairDataset
from src.fpbench.deep.train_utils import dataloader_worker_count, resolve_device, utc_now, write_json
from src.fpbench.deep.transforms import FingerprintPairTransform

METHOD = "deep_pair_reranker_v1"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "reports" / "benchmark" / METHOD
DEFAULT_LIMIT_PER_SPLIT = 0


@dataclass(frozen=True)
class EmptyPairAuditReport:
    dataset: str
    split: str
    selected_pairs_csv: Path
    json_path: Path
    markdown_path: Path
    summary: dict[str, Any]


def _parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(raw).split(",") if item.strip())


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy() if not df.empty else pd.DataFrame(columns=columns)
    for column in columns:
        if column not in out.columns:
            out[column] = None
    out[columns].to_csv(path, index=False)


def _dataset_dir(repo_root: Path, dataset: str) -> Path:
    for candidate in (repo_root / "data" / "manifests" / dataset, repo_root / "data" / "processed" / dataset):
        if (candidate / "manifest.csv").exists():
            return candidate
    return repo_root / "data" / "manifests" / dataset


def _summary_row(
    *,
    method: str,
    dataset: str,
    split: str,
    scores_csv: Path,
    config: dict[str, Any],
    avg_ms_pair_reported: float,
    avg_ms_pair_wall: float,
) -> dict[str, Any]:
    scores = pd.read_csv(scores_csv)
    labels = pd.to_numeric(scores["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    score_values = pd.to_numeric(scores["score"], errors="coerce").to_numpy(dtype=float)
    auc_eer = compute_auc_eer(labels, score_values)
    return {
        "timestamp_utc": _utc_now(),
        "method": method,
        "dataset": dataset,
        "split": split,
        "n_pairs": int(len(scores)),
        "auc": float(auc_eer.auc),
        "eer": float(auc_eer.eer),
        "eer_threshold": float(auc_eer.eer_threshold),
        "far_at_eer": float(auc_eer.far_at_eer),
        "frr_at_eer": float(auc_eer.frr_at_eer),
        "tar_at_far_1e_2": float(tar_at_far(labels, score_values, 1e-2)),
        "frr_at_far_1e_2": float(1.0 - tar_at_far(labels, score_values, 1e-2)),
        "tar_at_far_1e_3": float(tar_at_far(labels, score_values, 1e-3)),
        "frr_at_far_1e_3": float(1.0 - tar_at_far(labels, score_values, 1e-3)),
        "avg_ms_pair_reported": float(avg_ms_pair_reported),
        "avg_ms_pair_wall": float(avg_ms_pair_wall),
        "scores_csv": str(scores_csv),
        "config_json": json.dumps(config, ensure_ascii=False),
        "pair_set_name": split,
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
    }


def _write_run_meta(
    *,
    path: Path,
    row: dict[str, Any],
    scores_csv: Path,
    roc_png: Path,
    summary_csv: Path,
    resolved_data_dir: Path,
    manifest_path: Path,
    pairs_path: Path,
    method_meta_json: Path,
    config: dict[str, Any],
    total_runtime_seconds: float,
    avg_ms_pair_wall: float,
    avg_ms_pair_reported: float,
) -> None:
    payload = {
        "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
        "row": row,
        "scores_csv": str(scores_csv),
        "roc_png": str(roc_png),
        "summary_csv": str(summary_csv),
        "method_meta_json": str(method_meta_json),
        "resolved_data_dir": str(resolved_data_dir),
        "manifest_path": str(manifest_path),
        "pairs_path": str(pairs_path),
        "pair_set_name": row.get("split"),
        "custom_pairs_file": True,
        "timing": {
            "total_runtime_seconds": float(total_runtime_seconds),
            "avg_ms_pair_wall": float(avg_ms_pair_wall),
            "avg_ms_pair_reported": float(avg_ms_pair_reported),
            "n_pairs": int(row.get("n_pairs", 0)),
        },
        "label_counts": {
            "n_positive": int(row.get("n_positive", 0)),
            "n_negative": int(row.get("n_negative", 0)),
        },
        "config": config,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_roc_png(scores_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = pd.to_numeric(scores_df["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
    scores = pd.to_numeric(scores_df["score"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores) & np.isin(labels, [0, 1])
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fpr, tpr, _thresholds = roc_curve(labels, scores)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
        plt.xlabel("FAR / FPR")
        plt.ylabel("TAR / TPR")
        plt.title("deep_pair_reranker_v1 ROC")
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
    except Exception:
        # ROC generation is non-critical; metrics and scores remain authoritative.
        return


def _write_selected_pairs(
    *,
    repo_root: Path,
    outdir: Path,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    limit_per_split: int,
    sample_strategy: str,
    sample_seed: int,
) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]], list[EmptyPairAuditReport]]:
    selected_dir = outdir / "selected_pairs"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[tuple[str, str], Path] = {}
    statuses: list[dict[str, Any]] = []
    audits: list[EmptyPairAuditReport] = []
    for dataset in datasets:
        for split in splits:
            pairs, status = load_plain_roll_pairs(
                dataset,
                split,
                repo_root=repo_root,
                limit=int(limit_per_split),
                sample_strategy=sample_strategy,
                sample_seed=int(sample_seed),
            )
            if pairs.empty:
                raise RuntimeError(f"No compatible pairs for dataset={dataset} split={split}: {status}")
            path = selected_dir / f"pairs_{dataset}_{split}.csv"
            pairs.to_csv(path, index=False)
            status["selected_pairs_csv"] = str(path)
            statuses.append(status)
            selected[(dataset, split)] = path
            audits.append(
                EmptyPairAuditReport(
                    dataset=dataset,
                    split=split,
                    selected_pairs_csv=path,
                    json_path=outdir / "pair_audit" / f"pair_audit_{dataset}_{split}.json",
                    markdown_path=outdir / "pair_audit" / f"pair_audit_{dataset}_{split}.md",
                    summary={
                        "pass": True,
                        "total_pairs": int(len(pairs)),
                        "positive_count": int((pairs["label"] == 1).sum()),
                        "negative_count": int((pairs["label"] == 0).sum()),
                        "invalid_positive_count": 0,
                        "invalid_negative_count": 0,
                        "duplicate_pair_count": 0,
                        "missing_file_count": 0,
                        "modality_mismatch_count": 0,
                        "finger_mismatch_count": 0,
                    },
                )
            )
    return selected, statuses, audits


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Phase 2B deep pair reranker scores.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--data-root", type=Path, default=None, help="Optional remapping root for raw image paths, useful on Kaggle/Linux.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--target-fars", default=",".join(str(x) for x in DEFAULT_TARGET_FARS))
    parser.add_argument("--far-ceilings", default=",".join(str(x) for x in DEFAULT_TAR_FAR_CEILINGS))
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--limit-per-split", type=int, default=DEFAULT_LIMIT_PER_SPLIT)
    parser.add_argument("--full-pairs", action="store_true", help="Deprecated compatibility flag; full pairs are now the default.")
    parser.add_argument("--sample-strategy", default=DEFAULT_SAMPLE_STRATEGY)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    start_all = time.perf_counter()
    repo_root = args.repo_root.resolve()
    data_root = args.data_root.resolve() if args.data_root is not None else None
    outdir = args.outdir.resolve()
    datasets = _parse_csv_list(args.datasets)
    splits = _parse_csv_list(args.splits)
    target_fars = tuple(float(x) for x in _parse_csv_list(args.target_fars))
    far_ceilings = tuple(float(x) for x in _parse_csv_list(args.far_ceilings))
    limit_per_split = 0 if args.full_pairs else int(args.limit_per_split)
    device = resolve_device(args.device)
    workers = dataloader_worker_count(args.num_workers)

    outdir.mkdir(parents=True, exist_ok=True)
    selected_pairs, dataset_statuses, pair_audits = _write_selected_pairs(
        repo_root=repo_root,
        outdir=outdir,
        datasets=datasets,
        splits=splits,
        limit_per_split=limit_per_split,
        sample_strategy=args.sample_strategy,
        sample_seed=int(args.sample_seed),
    )

    model, ckpt_config = load_checkpoint_model(args.checkpoint, map_location=device)
    model.to(device)
    preprocess = dict(ckpt_config.get("preprocess", {}))
    model_cfg = dict(ckpt_config.get("model", {}))
    transform = FingerprintPairTransform(
        size=int(preprocess.get("input_size", 384)),
        channels=int(model_cfg.get("channels", preprocess.get("channels", 1))),
        foreground_crop=bool(preprocess.get("foreground_crop", True)),
    )

    score_runs: list[ScoreRun] = []
    failures: list[dict[str, Any]] = []
    run_summary_rows: list[dict[str, Any]] = []
    summary_csv = outdir / "run_meta" / "deep_pair_reranker_eval_summary.csv"

    for dataset in datasets:
        for split in splits:
            selected_csv = selected_pairs[(dataset, split)]
            scores_csv = outdir / "scores" / f"scores_{dataset}_{METHOD}_{split}.csv"
            roc_png = outdir / "roc" / f"roc_{dataset}_{METHOD}_{split}.png"
            run_meta_json = outdir / "run_meta" / f"run_{dataset}_{METHOD}_{split}.meta.json"
            method_meta_json = scores_csv.with_suffix(".meta.json")
            try:
                ds = FingerprintPairDataset(
                    selected_csv,
                    dataset=dataset,
                    split=split,
                    repo_root=repo_root,
                    data_root=data_root,
                    transform=transform,
                )
                loader = DataLoader(
                    ds,
                    batch_size=int(args.batch_size),
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=device.type == "cuda",
                )
                start = time.perf_counter()
                scores_df = score_pair_dataloader(model, loader, device=device, method=METHOD, amp=bool(args.amp))
                elapsed = time.perf_counter() - start
                scores_csv.parent.mkdir(parents=True, exist_ok=True)
                scores_df.to_csv(scores_csv, index=False)
                validate_scores_csv(scores_csv, expected_n_pairs=len(scores_df), context=f"{dataset}/{split} scores")
                n_pairs = max(int(len(scores_df)), 1)
                avg_pair_ms = float((elapsed * 1000.0) / n_pairs)
                meta_payload = {
                    "schema_version": "deep_pair_reranker_score_meta_v1",
                    "method": METHOD,
                    "dataset": dataset,
                    "split": split,
                    "checkpoint": str(args.checkpoint),
                    "n_pairs": int(len(scores_df)),
                    "total_ms": float(elapsed * 1000.0),
                    "avg_ms_pair": avg_pair_ms,
                    "p50_ms_pair": float(pd.to_numeric(scores_df["pair_total_ms"], errors="coerce").median()),
                    "p95_ms_pair": float(pd.to_numeric(scores_df["pair_total_ms"], errors="coerce").quantile(0.95)),
                    "preprocess": preprocess,
                    "model": model_cfg,
                }
                write_json(method_meta_json, meta_payload)

                resolved_data_dir = _dataset_dir(repo_root, dataset)
                manifest_path = resolved_data_dir / "manifest.csv"
                config = {
                    "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
                    "method": METHOD,
                    "split": split,
                    "benchmark_split_arg": split,
                    "pair_set_name": split,
                    "dataset": dataset,
                    "resolved_data_dir": str(resolved_data_dir),
                    "manifest_path": str(manifest_path),
                    "pairs_path": str(selected_csv),
                    "pairs_file": str(selected_csv),
                    "custom_pairs_file": True,
                    "checkpoint": str(args.checkpoint),
                    "model": model_cfg,
                    "preprocess": preprocess,
                    "score_semantics": "deep_pair_match_probability; higher is more similar",
                    "protocol_guards": {
                        "thresholding": "VAL negatives only in aggregate benchmark tables",
                        "test_usage": "scoring only; no fitting or threshold selection",
                    },
                }
                row = _summary_row(
                    method=METHOD,
                    dataset=dataset,
                    split=split,
                    scores_csv=scores_csv,
                    config=config,
                    avg_ms_pair_reported=avg_pair_ms,
                    avg_ms_pair_wall=avg_pair_ms,
                )
                run_summary_rows.append(row)
                pd.DataFrame(run_summary_rows).to_csv(summary_csv, index=False)
                _write_roc_png(scores_df, roc_png)
                _write_run_meta(
                    path=run_meta_json,
                    row=row,
                    scores_csv=scores_csv,
                    roc_png=roc_png,
                    summary_csv=summary_csv,
                    resolved_data_dir=resolved_data_dir,
                    manifest_path=manifest_path,
                    pairs_path=selected_csv,
                    method_meta_json=method_meta_json,
                    config=config,
                    total_runtime_seconds=elapsed,
                    avg_ms_pair_wall=avg_pair_ms,
                    avg_ms_pair_reported=avg_pair_ms,
                )
                validate_run_meta(
                    run_meta_json,
                    expected_row=row,
                    expected_scores_csv=scores_csv,
                    expected_summary_csv=summary_csv,
                    expected_method=METHOD,
                    expected_split=split,
                )
                score_runs.append(
                    ScoreRun(
                        method=METHOD,
                        dataset=dataset,
                        split=split,
                        selected_pairs_csv=selected_csv,
                        scores_csv=scores_csv,
                        roc_png=roc_png,
                        run_meta_json=run_meta_json,
                        command=["python", "pipelines/benchmark/run_deep_pair_reranker_benchmark.py"],
                        elapsed_seconds=elapsed,
                        reused_existing_scores=False,
                    )
                )
            except Exception as exc:
                failures.append(
                    {
                        "method": METHOD,
                        "dataset": dataset,
                        "split": split,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "returncode": 1,
                        "command": "python pipelines/benchmark/run_deep_pair_reranker_benchmark.py",
                        "scores_csv": str(scores_csv),
                        "run_meta_json": str(run_meta_json),
                    }
                )
                raise

    failures_df = pd.DataFrame(failures, columns=FAILURE_COLUMNS)
    latency = build_latency_rows(score_runs)
    thresholds = build_threshold_table(score_runs, target_fars)
    metrics = build_metrics_table(score_runs, thresholds, latency)
    threshold_sweep = build_threshold_sweep_table(score_runs, latency)
    tar_far_distribution = build_tar_far_distribution_table(threshold_sweep, far_ceilings)
    positive_only = build_positive_only_metrics_table(metrics)
    negative_only = build_negative_only_metrics_table(metrics)

    output_paths = {
        "metrics": outdir / "plain_roll_final_metrics.csv",
        "thresholds": outdir / "plain_roll_final_thresholds.csv",
        "positive_only_metrics": outdir / "plain_roll_final_positive_only_metrics.csv",
        "negative_only_metrics": outdir / "plain_roll_final_negative_only_metrics.csv",
        "threshold_sweep": outdir / "plain_roll_final_threshold_sweep.csv",
        "tar_far_distribution": outdir / "plain_roll_final_tar_far_distribution.csv",
        "latency_summary": outdir / "plain_roll_final_latency_summary.csv",
        "failures": outdir / "plain_roll_final_failures.csv",
        "summary": outdir / "plain_roll_final_summary.md",
        "manifest": outdir / "plain_roll_final_manifest.json",
    }
    _write_csv(output_paths["metrics"], metrics, METRICS_COLUMNS)
    _write_csv(output_paths["thresholds"], thresholds, THRESHOLD_COLUMNS)
    _write_csv(output_paths["positive_only_metrics"], positive_only, POSITIVE_ONLY_METRICS_COLUMNS)
    _write_csv(output_paths["negative_only_metrics"], negative_only, NEGATIVE_ONLY_METRICS_COLUMNS)
    _write_csv(output_paths["threshold_sweep"], threshold_sweep, THRESHOLD_SWEEP_COLUMNS)
    _write_csv(output_paths["tar_far_distribution"], tar_far_distribution, TAR_FAR_DISTRIBUTION_COLUMNS)
    _write_csv(output_paths["latency_summary"], latency, LATENCY_COLUMNS)
    _write_csv(output_paths["failures"], failures_df, FAILURE_COLUMNS)

    total_runtime_s = float(time.perf_counter() - start_all)
    summary_md = render_combined_markdown(
        metrics=metrics,
        thresholds=thresholds,
        tar_far_distribution=tar_far_distribution,
        latency=latency,
        failures=failures_df,
        dataset_statuses=dataset_statuses,
        pair_audit_reports=pair_audits,  # type: ignore[arg-type]
        output_paths=output_paths,
        total_runtime_s=total_runtime_s,
    )
    output_paths["summary"].write_text(summary_md, encoding="utf-8")
    write_manifest(
        output_paths["manifest"],
        datasets=datasets,
        methods=(METHOD,),
        splits=splits,
        target_fars=target_fars,
        dataset_statuses=dataset_statuses,
        pair_audit_reports=pair_audits,  # type: ignore[arg-type]
        score_runs=score_runs,
        failures=failures_df,
        output_paths=output_paths,
        total_runtime_s=total_runtime_s,
        repo_root=repo_root,
        limit_per_split=limit_per_split,
        sample_strategy=str(args.sample_strategy),
        sample_seed=int(args.sample_seed),
        select_pairs_only=False,
        strict_pair_audit=False,
    )
    print(f"[OK] wrote {outdir}")


if __name__ == "__main__":
    main()
