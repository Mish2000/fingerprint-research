from __future__ import annotations

import argparse
import csv
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
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_METHODS = ("sift", "sift_plain_roll_v2")
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)
OUTDIR = "artifacts/reports/benchmark/sift_plain_roll_v2_external_validation"
METHOD_VARIANTS = (
    ("sift", "current_score"),
    ("sift", "inliers"),
    ("sift_plain_roll_v2", "official_score"),
)


@dataclass(frozen=True)
class ScoreRun:
    dataset: str
    dataset_dir: Path
    split: str
    pairs_csv: Path
    method: str
    score_csv: Path
    roc_png: Path
    run_meta_json: Path
    command: list[str]


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _resolve_dataset_dir(dataset: str) -> Path | None:
    for base in ("data/manifests", "data/processed"):
        candidate = REPO_ROOT / base / dataset
        if (candidate / "manifest.csv").exists():
            return candidate
    return None


def _pairs_path(dataset_dir: Path, split: str) -> Path | None:
    candidates = [
        dataset_dir / f"pairs_{split}.csv",
        dataset_dir / "pairs" / f"pairs_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _looks_plain_roll_pair(path_a: str, path_b: str) -> bool:
    left = path_a.lower().replace("\\", "/")
    right = path_b.lower().replace("\\", "/")
    return "plain" in left and ("roll" in right or "rolled" in right)


def _dataset_status(dataset: str, dataset_dir: Path, split: str, pairs_csv: Path) -> dict[str, Any]:
    df = pd.read_csv(pairs_csv)
    required = {"label", "split", "path_a", "path_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        return {
            "dataset": dataset,
            "split": split,
            "dataset_dir": str(dataset_dir),
            "pairs_csv": str(pairs_csv),
            "compatible": False,
            "reason": f"missing required columns: {missing}",
            "n_pairs": int(len(df)),
        }

    labels = pd.to_numeric(df["label"], errors="coerce").fillna(-1).astype(int)
    plain_roll_mask = [
        _looks_plain_roll_pair(str(row.path_a), str(row.path_b))
        for row in df[["path_a", "path_b"]].itertuples(index=False)
    ]
    split_values = sorted(str(x) for x in df["split"].dropna().unique().tolist())
    sample = df.head(min(50, len(df)))
    sampled_paths_exist = True
    first_missing = ""
    for _, row in sample.iterrows():
        for column in ("path_a", "path_b"):
            candidate = parse_file_uri(str(row[column]))
            if not candidate.exists():
                sampled_paths_exist = False
                first_missing = str(candidate)
                break
        if not sampled_paths_exist:
            break

    compatible = bool(len(df)) and all(plain_roll_mask) and sampled_paths_exist
    reason = "plain-to-roll pairs with sampled image paths present" if compatible else "not plain-to-roll compatible"
    if not all(plain_roll_mask):
        reason = "at least one pair path is not plain-to-roll"
    if not sampled_paths_exist:
        reason = f"sampled image path missing: {first_missing}"

    return {
        "dataset": dataset,
        "split": split,
        "dataset_dir": str(dataset_dir),
        "pairs_csv": str(pairs_csv),
        "compatible": compatible,
        "reason": reason,
        "n_pairs": int(len(df)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "split_values": split_values,
        "all_paths_plain_to_roll": bool(all(plain_roll_mask)),
        "sampled_paths_exist": sampled_paths_exist,
    }


def _cmd_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(x) for x in command])


def _build_score_run(
    *,
    dataset: str,
    dataset_dir: Path,
    split: str,
    pairs_csv: Path,
    method: str,
    outdir: Path,
    limit: int,
) -> ScoreRun:
    scores_dir = outdir / "scores" / dataset
    score_csv = scores_dir / f"scores_{dataset}_{method}_{split}.csv"
    roc_png = scores_dir / f"roc_{dataset}_{method}_{split}.png"
    run_meta_json = scores_dir / f"run_{dataset}_{method}_{split}.meta.json"
    summary_csv = scores_dir / "evaluation_runs_summary.csv"
    command = [
        sys.executable,
        str(REPO_ROOT / "pipelines" / "benchmark" / "evaluate.py"),
        "--method",
        method,
        "--split",
        split,
        "--limit",
        str(int(limit)),
        "--dataset",
        dataset,
        "--data_dir",
        str(dataset_dir),
        "--pairs_file",
        str(pairs_csv),
        "--pair_set_name",
        split,
        "--summary_csv",
        str(summary_csv),
        "--out_scores",
        str(score_csv),
        "--out_roc",
        str(roc_png),
        "--out_run_meta",
        str(run_meta_json),
    ]
    return ScoreRun(
        dataset=dataset,
        dataset_dir=dataset_dir,
        split=split,
        pairs_csv=pairs_csv,
        method=method,
        score_csv=score_csv,
        roc_png=roc_png,
        run_meta_json=run_meta_json,
        command=command,
    )


def discover_score_runs(
    *,
    datasets: list[str],
    splits: list[str],
    outdir: Path,
    limit: int,
    allow_non_plain_roll: bool,
) -> tuple[list[ScoreRun], list[dict[str, Any]]]:
    runs: list[ScoreRun] = []
    statuses: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_dir = _resolve_dataset_dir(dataset)
        if dataset_dir is None:
            statuses.append(
                {
                    "dataset": dataset,
                    "split": "",
                    "compatible": False,
                    "reason": "no manifest dataset dir found",
                }
            )
            continue
        for split in splits:
            pairs_csv = _pairs_path(dataset_dir, split)
            if pairs_csv is None:
                statuses.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "dataset_dir": str(dataset_dir),
                        "compatible": False,
                        "reason": "no pairs CSV found",
                    }
                )
                continue
            status = _dataset_status(dataset, dataset_dir, split, pairs_csv)
            statuses.append(status)
            if not allow_non_plain_roll and not bool(status["compatible"]):
                continue
            for method in DEFAULT_METHODS:
                runs.append(
                    _build_score_run(
                        dataset=dataset,
                        dataset_dir=dataset_dir,
                        split=split,
                        pairs_csv=pairs_csv,
                        method=method,
                        outdir=outdir,
                        limit=limit,
                    )
                )
    return runs, statuses


def _run_command(run: ScoreRun, *, log_handle: Any, reuse_existing: bool) -> dict[str, Any]:
    run.score_csv.parent.mkdir(parents=True, exist_ok=True)
    if reuse_existing and run.score_csv.exists() and run.run_meta_json.exists():
        message = f"[REUSE] {run.dataset}/{run.split}/{run.method}: {run.score_csv}"
        print(message)
        log_handle.write(message + "\n\n")
        return {
            "dataset": run.dataset,
            "split": run.split,
            "method": run.method,
            "command": _cmd_text(run.command),
            "exit_code": 0,
            "reused_existing": True,
            "score_csv": str(run.score_csv),
            "run_meta_json": str(run.run_meta_json),
        }

    start = time.perf_counter()
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(REPO_ROOT)
    env.setdefault("PYTHONHASHSEED", "0")
    command_text = _cmd_text(run.command)
    header = f"[RUN] {run.dataset}/{run.split}/{run.method}\n{command_text}"
    print(header)
    log_handle.write(header + "\n")
    proc = subprocess.run(
        run.command,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if proc.stdout:
        log_handle.write("[STDOUT]\n")
        log_handle.write(proc.stdout.rstrip() + "\n")
    if proc.stderr:
        log_handle.write("[STDERR]\n")
        log_handle.write(proc.stderr.rstrip() + "\n")
    log_handle.write(f"[EXIT] {proc.returncode} elapsed_s={elapsed:.3f}\n\n")
    print(f"[DONE] {run.dataset}/{run.split}/{run.method} exit={proc.returncode} elapsed={elapsed:.1f}s")
    if proc.returncode != 0:
        raise RuntimeError(f"Score generation failed for {run.dataset}/{run.split}/{run.method}")
    return {
        "dataset": run.dataset,
        "split": run.split,
        "method": run.method,
        "command": command_text,
        "exit_code": int(proc.returncode),
        "elapsed_s": float(elapsed),
        "reused_existing": False,
        "score_csv": str(run.score_csv),
        "run_meta_json": str(run.run_meta_json),
    }


def _safe_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _variant_scores(method: str, df: pd.DataFrame) -> dict[str, np.ndarray]:
    if method == "sift":
        variants = {"current_score": _safe_numeric(df["score"])}
        if "inliers" in df.columns:
            variants["inliers"] = _safe_numeric(df["inliers"])
        return variants
    if method == "sift_plain_roll_v2":
        return {"official_score": _safe_numeric(df["score"])}
    raise ValueError(f"Unsupported method: {method}")


def _threshold_for_far(negative_scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    scores = np.asarray(negative_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), 0, float("nan")
    n_negative = int(scores.size)
    for threshold in sorted(float(x) for x in np.unique(scores)):
        false_accepts = int(np.sum(scores >= threshold))
        actual_far = false_accepts / n_negative
        if actual_far <= float(target_far):
            return float(threshold), false_accepts, float(actual_far)
    threshold = math.nextafter(float(np.max(scores)), math.inf)
    return float(threshold), 0, 0.0


def _confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = labels == 1
    negatives = labels == 0
    accepted = scores >= float(threshold) if math.isfinite(float(threshold)) else np.zeros_like(scores, dtype=bool)
    ta = int(np.sum(accepted & positives))
    fr = int(np.sum((~accepted) & positives))
    fa = int(np.sum(accepted & negatives))
    tr = int(np.sum((~accepted) & negatives))
    n_positive = int(np.sum(positives))
    n_negative = int(np.sum(negatives))
    tar = float(ta / n_positive) if n_positive else float("nan")
    far = float(fa / n_negative) if n_negative else float("nan")
    return {
        "tar": tar,
        "frr": float(1.0 - tar) if math.isfinite(tar) else float("nan"),
        "far": far,
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _auc_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)
    except ValueError:
        return float("nan"), float("nan"), float("nan")
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr))) if fpr.size else 0
    return float(auc), float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def _median_column(df: pd.DataFrame, mask: np.ndarray, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df.loc[mask, column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _score_frames(runs: list[ScoreRun]) -> dict[tuple[str, str, str], pd.DataFrame]:
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for run in runs:
        if not run.score_csv.exists():
            raise FileNotFoundError(f"Missing score CSV: {run.score_csv}")
        df = pd.read_csv(run.score_csv)
        missing = {"label", "split", "path_a", "path_b", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"{run.score_csv} missing required columns: {sorted(missing)}")
        df = df.copy()
        df["dataset"] = run.dataset
        df["method"] = run.method
        df["source_split"] = run.split
        df["source_scores_csv"] = str(run.score_csv)
        frames[(run.dataset, run.method, run.split)] = df
    return frames


def build_validation_tables(
    runs: list[ScoreRun],
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
    top_n_cases: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frames = _score_frames(runs)
    thresholds: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    false_accepts: list[dict[str, Any]] = []
    false_rejects: list[dict[str, Any]] = []
    datasets = sorted({run.dataset for run in runs})

    for dataset in datasets:
        for method in DEFAULT_METHODS:
            val = frames.get((dataset, method, "val"))
            test = frames.get((dataset, method, "test"))
            if val is None or test is None:
                continue
            val_labels = pd.to_numeric(val["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
            test_labels = pd.to_numeric(test["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
            for variant, val_scores in _variant_scores(method, val).items():
                test_scores = _variant_scores(method, test)[variant]
                val_neg_scores = val_scores[val_labels == 0]
                for target_far in target_fars:
                    threshold, calibration_fa, calibration_far = _threshold_for_far(
                        val_neg_scores,
                        float(target_far),
                    )
                    thresholds.append(
                        {
                            "dataset": dataset,
                            "method": method,
                            "variant": variant,
                            "target_far": float(target_far),
                            "threshold": float(threshold),
                            "calibration_split": "val",
                            "calibration_negative_count": int(np.sum(val_labels == 0)),
                            "calibration_false_accepts": int(calibration_fa),
                            "calibration_far": float(calibration_far),
                            "threshold_source": f"{dataset} val negatives | target FAR <= {float(target_far):.2%}",
                        }
                    )
                    for split, df, labels, scores in (
                        ("val", val, val_labels, val_scores),
                        ("test", test, test_labels, test_scores),
                    ):
                        counts = _confusion(labels, scores, threshold)
                        auc, eer, eer_threshold = _auc_eer(labels, scores)
                        accepted = np.asarray(scores, dtype=float) >= float(threshold) if math.isfinite(threshold) else np.zeros(
                            len(df),
                            dtype=bool,
                        )
                        labels_arr = np.asarray(labels, dtype=int)
                        accepted_pos = accepted & (labels_arr == 1)
                        rejected_pos = (~accepted) & (labels_arr == 1)
                        false_accepted_neg = accepted & (labels_arr == 0)
                        metrics.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "method": method,
                                "variant": variant,
                                "target_far": float(target_far),
                                "threshold": float(threshold),
                                "far": float(counts["far"]),
                                "tar": float(counts["tar"]),
                                "frr": float(counts["frr"]),
                                "ta": int(counts["ta"]),
                                "fr": int(counts["fr"]),
                                "fa": int(counts["fa"]),
                                "tr": int(counts["tr"]),
                                "n_positive": int(counts["n_positive"]),
                                "n_negative": int(counts["n_negative"]),
                                "auc": float(auc),
                                "eer": float(eer),
                                "eer_threshold": float(eer_threshold),
                                "median_score_true_accepts": _median_column(df.assign(_score=scores), accepted_pos, "_score"),
                                "median_score_false_rejects": _median_column(df.assign(_score=scores), rejected_pos, "_score"),
                                "median_score_false_accepts": _median_column(df.assign(_score=scores), false_accepted_neg, "_score"),
                                "median_inliers_true_accepts": _median_column(df, accepted_pos, "inliers"),
                                "median_inliers_false_rejects": _median_column(df, rejected_pos, "inliers"),
                                "median_inliers_false_accepts": _median_column(df, false_accepted_neg, "inliers"),
                                "median_matches_true_accepts": _median_column(df, accepted_pos, "matches"),
                                "median_matches_false_rejects": _median_column(df, rejected_pos, "matches"),
                                "median_matches_false_accepts": _median_column(df, false_accepted_neg, "matches"),
                                "source_scores_csv": str(df["source_scores_csv"].iloc[0]) if len(df) else "",
                            }
                        )

                    test_case_df = test.copy()
                    test_case_df["_score_value"] = np.asarray(test_scores, dtype=float)
                    test_case_df["_accepted"] = (
                        test_case_df["_score_value"].to_numpy(dtype=float) >= float(threshold)
                        if math.isfinite(threshold)
                        else False
                    )
                    fa_subset = test_case_df[(test_case_df["label"].astype(int) == 0) & test_case_df["_accepted"]].copy()
                    fa_subset = fa_subset.sort_values("_score_value", ascending=False).head(int(top_n_cases))
                    fr_subset = test_case_df[(test_case_df["label"].astype(int) == 1) & (~test_case_df["_accepted"])].copy()
                    fr_subset = fr_subset.sort_values("_score_value", ascending=False).head(int(top_n_cases))
                    for rank, (_, row) in enumerate(fa_subset.iterrows(), start=1):
                        false_accepts.append(_case_row(dataset, method, variant, target_far, threshold, rank, row))
                    for rank, (_, row) in enumerate(fr_subset.iterrows(), start=1):
                        false_rejects.append(_case_row(dataset, method, variant, target_far, threshold, rank, row))

    return (
        pd.DataFrame(thresholds),
        pd.DataFrame(metrics),
        pd.DataFrame(false_accepts),
        pd.DataFrame(false_rejects),
    )


def _case_row(
    dataset: str,
    method: str,
    variant: str,
    target_far: float,
    threshold: float,
    rank: int,
    row: pd.Series,
) -> dict[str, Any]:
    score = float(row["_score_value"])
    out = {
        "dataset": dataset,
        "split": "test",
        "method": method,
        "variant": variant,
        "target_far": float(target_far),
        "threshold": float(threshold),
        "rank": int(rank),
        "score": score,
        "score_minus_threshold": float(score - float(threshold)) if math.isfinite(float(threshold)) else float("nan"),
        "label": int(row["label"]),
        "path_a": str(row["path_a"]),
        "path_b": str(row["path_b"]),
        "source_scores_csv": str(row.get("source_scores_csv", "")),
    }
    for column in ("pair_id", "subject_a", "subject_b", "frgp", "inliers", "matches", "k1", "k2"):
        if column in row:
            value = row[column]
            if pd.isna(value):
                value = ""
            out[column] = value
    return out


def _load_professor_continuity() -> tuple[pd.DataFrame, str]:
    try:
        from scripts.diagnostics.official_sift_plain_roll_v2_comparison import (
            build_professor_combined_comparison,
        )

        report = build_professor_combined_comparison()
        report = report.copy()
        report["report_section"] = "professor_combined_val_test_1pct"
        return report, ""
    except Exception as exc:  # pragma: no cover - optional continuity artifact
        return pd.DataFrame(), str(exc)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def _metric_row(metrics: pd.DataFrame, dataset: str, method: str, variant: str, target_far: float) -> pd.Series | None:
    subset = metrics[
        (metrics["dataset"] == dataset)
        & (metrics["split"] == "test")
        & (metrics["method"] == method)
        & (metrics["variant"] == variant)
        & np.isclose(metrics["target_far"].astype(float), float(target_far))
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _answer_rows(metrics: pd.DataFrame) -> list[str]:
    datasets = sorted(metrics["dataset"].dropna().unique().tolist()) if not metrics.empty else []
    lines: list[str] = []
    for dataset in datasets:
        base = _metric_row(metrics, dataset, "sift", "current_score", 0.01)
        v2 = _metric_row(metrics, dataset, "sift_plain_roll_v2", "official_score", 0.01)
        inliers = _metric_row(metrics, dataset, "sift", "inliers", 0.01)
        if base is None or v2 is None:
            continue
        delta = float(v2["tar"]) - float(base["tar"])
        inlier_text = ""
        if inliers is not None:
            inlier_delta = float(v2["tar"]) - float(inliers["tar"])
            inlier_text = f"; vs SIFT inliers-only delta {inlier_delta:+.1%}"
        lines.append(
            f"- {dataset} TEST @1% FAR: v2 TAR {_pct(v2['tar'])} / FAR {_pct(v2['far'])}; "
            f"canonical SIFT TAR {_pct(base['tar'])} / FAR {_pct(base['far'])}; "
            f"delta {delta:+.1%}{inlier_text}."
        )
    return lines


def _generalization_answer(metrics: pd.DataFrame) -> str:
    datasets = sorted(metrics["dataset"].dropna().unique().tolist()) if not metrics.empty else []
    deltas: list[float] = []
    for dataset in datasets:
        base = _metric_row(metrics, dataset, "sift", "current_score", 0.01)
        v2 = _metric_row(metrics, dataset, "sift_plain_roll_v2", "official_score", 0.01)
        if base is not None and v2 is not None:
            deltas.append(float(v2["tar"]) - float(base["tar"]))
    if deltas and all(delta > 0 for delta in deltas):
        return "Yes, with caution: v2 improves TEST TAR over canonical SIFT at the 1% FAR operating point on every compatible full dataset evaluated, but absolute TAR is still far from a solved plain-vs-roll verifier."
    if any(delta > 0 for delta in deltas):
        return "Partially: v2 improves at least one compatible full TEST set, but the gain is not consistent across all external datasets."
    return "No: the full TEST sets did not reproduce a v2 improvement over canonical SIFT at the 1% FAR operating point."


def _dangerous_fa_answer(metrics: pd.DataFrame) -> str:
    one_pct = metrics[
        (metrics["split"] == "test")
        & (metrics["method"] == "sift_plain_roll_v2")
        & (metrics["variant"] == "official_score")
        & np.isclose(metrics["target_far"].astype(float), 0.01)
    ].copy()
    if one_pct.empty:
        return "Cannot assess: no v2 1% FAR TEST rows were produced."
    worst_far = float(one_pct["far"].max())
    worst_fa = int(one_pct["fa"].max())
    worst_n = int(one_pct.loc[one_pct["far"].astype(float).idxmax(), "n_negative"])
    if worst_far <= 0.02:
        return (
            f"No dangerous spike at the calibrated 1% point: worst v2 TEST FAR is {_pct(worst_far)} "
            f"({worst_fa}/{worst_n} negatives). The error cases still need visual audit before any deployment claim."
        )
    return (
        f"Yes, caution: worst v2 TEST FAR at the calibrated 1% point is {_pct(worst_far)} "
        f"({worst_fa}/{worst_n} negatives), indicating validation-set calibration does not fully control TEST false accepts."
    )


def _bottleneck_answer(metrics: pd.DataFrame) -> str:
    ten_pct = metrics[
        (metrics["split"] == "test")
        & (metrics["method"] == "sift_plain_roll_v2")
        & (metrics["variant"] == "official_score")
        & np.isclose(metrics["target_far"].astype(float), 0.10)
    ].copy()
    if ten_pct.empty:
        return "Cannot assess from the generated tables."
    avg_tar_10 = float(ten_pct["tar"].mean())
    median_fr_inliers = float(pd.to_numeric(ten_pct["median_inliers_false_rejects"], errors="coerce").median())
    if avg_tar_10 < 0.75:
        return (
            "The next bottleneck is more likely algorithmic matching plus image quality/crop/overlap than score calibration alone: "
            f"even at the 10% FAR target, average v2 TEST TAR is {_pct(avg_tar_10)}, and false rejects still show limited inlier evidence "
            f"(median false-reject inliers around {_fmt(median_fr_inliers, 1)})."
        )
    return (
        "Score calibration is now a plausible next bottleneck because v2 recovers most positives at relaxed FAR; "
        "the remaining work should still include visual review of crop/overlap failures."
    )


def _build_summary_csv(metrics: pd.DataFrame, professor: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_rows = metrics[metrics["split"] == "test"].copy()
    baseline_lookup = {
        (row["dataset"], row["target_far"]): row
        for _, row in test_rows[
            (test_rows["method"] == "sift") & (test_rows["variant"] == "current_score")
        ].iterrows()
    }
    for _, row in test_rows.iterrows():
        base = baseline_lookup.get((row["dataset"], row["target_far"]))
        rows.append(
            {
                "report_section": "full_dataset_test",
                "dataset": row["dataset"],
                "split": row["split"],
                "target_far": row["target_far"],
                "method": row["method"],
                "variant": row["variant"],
                "threshold": row["threshold"],
                "tar": row["tar"],
                "far": row["far"],
                "ta": row["ta"],
                "fr": row["fr"],
                "fa": row["fa"],
                "tr": row["tr"],
                "n_positive": row["n_positive"],
                "n_negative": row["n_negative"],
                "tar_delta_vs_sift_current": (
                    float(row["tar"]) - float(base["tar"]) if base is not None else float("nan")
                ),
                "far_delta_vs_sift_current": (
                    float(row["far"]) - float(base["far"]) if base is not None else float("nan")
                ),
            }
        )
    if not professor.empty:
        for _, row in professor.iterrows():
            rows.append(
                {
                    "report_section": "professor_combined_val_test_1pct",
                    "dataset": "nist_sd300b_professor_1000",
                    "split": "val+test",
                    "target_far": row.get("target_far", 0.01),
                    "method": row.get("method", ""),
                    "variant": row.get("variant", ""),
                    "threshold": row.get("threshold", float("nan")),
                    "tar": row.get("combined_tar", float("nan")),
                    "far": row.get("combined_far", float("nan")),
                    "ta": row.get("combined_ta", ""),
                    "fr": row.get("combined_fr", ""),
                    "fa": row.get("combined_fa", ""),
                    "tr": row.get("combined_tr", ""),
                    "n_positive": row.get("n_combined_positive", ""),
                    "n_negative": row.get("n_combined_negative", ""),
                    "tar_delta_vs_sift_current": float("nan"),
                    "far_delta_vs_sift_current": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _render_markdown(
    *,
    statuses: list[dict[str, Any]],
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    professor: pd.DataFrame,
    professor_error: str,
    command_log: Path,
    run_manifest: Path,
) -> str:
    lines = [
        "# SIFT Plain/Roll v2 External Validation",
        "",
        "Validation-only run. No parameters, algorithms, UI defaults, or canonical SIFT settings were tuned or promoted.",
        "",
        "## Explicit Answers",
        "",
        f"- Does v2 generalize beyond professor_1000? {_generalization_answer(metrics)}",
    ]
    b_base = _metric_row(metrics, "nist_sd300b", "sift", "current_score", 0.01)
    b_v2 = _metric_row(metrics, "nist_sd300b", "sift_plain_roll_v2", "official_score", 0.01)
    if b_base is not None and b_v2 is not None:
        lines.append(
            f"- Is the improvement consistent on full SD300B TEST? "
            f"{'Yes' if float(b_v2['tar']) > float(b_base['tar']) else 'No'}: "
            f"v2 TAR {_pct(b_v2['tar'])} vs canonical SIFT {_pct(b_base['tar'])} at 1% FAR."
        )
    c_base = _metric_row(metrics, "nist_sd300c", "sift", "current_score", 0.01)
    c_v2 = _metric_row(metrics, "nist_sd300c", "sift_plain_roll_v2", "official_score", 0.01)
    if c_base is not None and c_v2 is not None:
        lines.append(
            f"- Is the improvement consistent on SD300C, if available? "
            f"{'Yes' if float(c_v2['tar']) > float(c_base['tar']) else 'No'}: "
            f"v2 TAR {_pct(c_v2['tar'])} vs canonical SIFT {_pct(c_base['tar'])} at 1% FAR."
        )
    else:
        lines.append("- Is the improvement consistent on SD300C, if available? SD300C was not evaluated because no compatible pair protocol was available.")
    lines.extend(
        [
            f"- Does v2 increase false accepts in a dangerous way? {_dangerous_fa_answer(metrics)}",
            "- Should v2 remain research-only? Yes. This run is external validation evidence, not enough for canonical/default/showcase promotion.",
            f"- Next bottleneck: {_bottleneck_answer(metrics)}",
            "",
            "## Dataset Compatibility",
            "",
            "| dataset | split | compatible | pairs | positives | negatives | reason | pairs CSV |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for status in statuses:
        lines.append(
            f"| {status.get('dataset', '')} | {status.get('split', '')} | {bool(status.get('compatible', False))} | "
            f"{status.get('n_pairs', '')} | {status.get('n_positive', '')} | {status.get('n_negative', '')} | "
            f"{status.get('reason', '')} | `{status.get('pairs_csv', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Main TEST Findings At 1% FAR",
            "",
        ]
    )
    lines.extend(_answer_rows(metrics))
    lines.extend(
        [
            "",
            "## Full TEST Operating Points",
            "",
            "| dataset | method | variant | target FAR | threshold | TEST TAR | TEST FAR | TA | FR | FA | TR |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    test_rows = metrics[metrics["split"] == "test"].copy()
    test_rows = test_rows.sort_values(["dataset", "method", "variant", "target_far"])
    for _, row in test_rows.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['variant']} | {_pct(row['target_far'])} | "
            f"{_fmt(row['threshold'], 6)} | {_pct(row['tar'])} | {_pct(row['far'])} | "
            f"{int(row['ta'])} | {int(row['fr'])} | {int(row['fa'])} | {int(row['tr'])} |"
        )
    lines.extend(
        [
            "",
            "## Threshold Calibration",
            "",
            "Thresholds are calibrated from VAL negatives only, then evaluated on VAL and TEST separately.",
            "",
            "| dataset | method | variant | target FAR | threshold | VAL calibration FAR | VAL calibration FA / negatives |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    ordered_thresholds = thresholds.sort_values(["dataset", "method", "variant", "target_far"])
    for _, row in ordered_thresholds.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['method']} | {row['variant']} | {_pct(row['target_far'])} | "
            f"{_fmt(row['threshold'], 6)} | {_pct(row['calibration_far'])} | "
            f"{int(row['calibration_false_accepts'])}/{int(row['calibration_negative_count'])} |"
        )
    lines.extend(
        [
            "",
            "## Secondary Professor Continuity",
            "",
            "This is not the main research conclusion. It is included only to connect the external validation back to the earlier professor-facing selected 1000 positive / 1000 negative view.",
            "",
        ]
    )
    if professor.empty:
        lines.append(f"Professor continuity table unavailable: {professor_error or 'unknown error'}")
    else:
        lines.extend(
            [
                "| method | variant | target FAR | threshold | combined TAR | combined FAR | TA | FA |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in professor.sort_values(["method", "variant"]).iterrows():
            lines.append(
                f"| {row['method']} | {row['variant']} | {_pct(row['target_far'])} | {_fmt(row['threshold'], 6)} | "
                f"{_pct(row['combined_tar'])} | {_pct(row['combined_far'])} | "
                f"{int(row['combined_ta'])}/{int(row['n_combined_positive'])} | "
                f"{int(row['combined_fa'])}/{int(row['n_combined_negative'])} |"
            )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Command log: `{command_log}`",
            f"- Run manifest: `{run_manifest}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _git_info() -> dict[str, Any]:
    def run_git(args: list[str]) -> tuple[int, str]:
        proc = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True)
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


def write_outputs(
    *,
    outdir: Path,
    statuses: list[dict[str, Any]],
    runs: list[ScoreRun],
    command_results: list[dict[str, Any]],
    target_fars: tuple[float, ...],
    top_n_cases: int,
    execute: bool,
    reuse_existing: bool,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    thresholds, metrics, false_accepts, false_rejects = build_validation_tables(
        runs,
        target_fars=target_fars,
        top_n_cases=top_n_cases,
    )
    professor, professor_error = _load_professor_continuity()
    summary = _build_summary_csv(metrics, professor)

    paths = {
        "markdown": outdir / "external_validation_summary.md",
        "summary_csv": outdir / "external_validation_summary.csv",
        "manifest": outdir / "run_manifest.json",
        "thresholds": outdir / "per_dataset_thresholds.csv",
        "metrics": outdir / "per_dataset_metrics.csv",
        "false_accepts": outdir / "false_accepts_top_cases.csv",
        "false_rejects": outdir / "false_rejects_top_cases.csv",
    }
    thresholds.to_csv(paths["thresholds"], index=False)
    metrics.to_csv(paths["metrics"], index=False)
    false_accepts.to_csv(paths["false_accepts"], index=False)
    false_rejects.to_csv(paths["false_rejects"], index=False)
    summary.to_csv(paths["summary_csv"], index=False)
    command_log = outdir / "command_log.txt"
    paths["markdown"].write_text(
        _render_markdown(
            statuses=statuses,
            metrics=metrics,
            thresholds=thresholds,
            professor=professor,
            professor_error=professor_error,
            command_log=command_log,
            run_manifest=paths["manifest"],
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "sift_plain_roll_v2_external_validation_v1",
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "repo_root": str(REPO_ROOT),
        "git": _git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "target_fars": [float(x) for x in target_fars],
        "methods": list(DEFAULT_METHODS),
        "variants": [{"method": method, "variant": variant} for method, variant in METHOD_VARIANTS],
        "execute": bool(execute),
        "reuse_existing": bool(reuse_existing),
        "validation_scope": "full compatible NIST SD300B/SD300C plain-vs-roll val/test pairs; no tuning",
        "promotion_status": "research_only_not_canonical_not_default_not_showcase",
        "dataset_statuses": statuses,
        "commands": command_results,
        "score_runs": [
            {
                "dataset": run.dataset,
                "split": run.split,
                "method": run.method,
                "pairs_csv": str(run.pairs_csv),
                "score_csv": str(run.score_csv),
                "run_meta_json": str(run.run_meta_json),
                "command": _cmd_text(run.command),
            }
            for run in runs
        ],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "External validation for experimental sift_plain_roll_v2 on compatible "
            "full NIST SD300B/SD300C val/test plain-vs-roll pairs."
        )
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    parser.add_argument("--top_n_cases", type=int, default=25)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run official evaluate.py commands before building the report. Default is command/report dry-run.",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="When --execute is set, skip score generation for score CSVs and run meta files that already exist.",
    )
    parser.add_argument(
        "--allow_non_plain_roll",
        action="store_true",
        help="Do not skip pair files whose paths do not validate as plain-to-roll.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    outdir = parse_file_uri(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    command_log = outdir / "command_log.txt"

    runs, statuses = discover_score_runs(
        datasets=datasets,
        splits=splits,
        outdir=outdir,
        limit=int(args.limit),
        allow_non_plain_roll=bool(args.allow_non_plain_roll),
    )
    if not runs:
        command_log.write_text("No compatible validation score commands were generated.\n", encoding="utf-8")
        print("No compatible validation score commands were generated.")
        return 2

    command_results: list[dict[str, Any]] = []
    with command_log.open("w", encoding="utf-8") as log_handle:
        log_handle.write("SIFT Plain/Roll v2 external validation command log\n")
        log_handle.write(f"Working directory: {REPO_ROOT}\n")
        log_handle.write(f"Environment: FPRJ_ROOT={REPO_ROOT}\n")
        log_handle.write(f"Invocation: {_cmd_text([sys.executable, *sys.argv])}\n\n")
        for run in runs:
            log_handle.write(
                f"[COMMAND-PLAN] {run.dataset}/{run.split}/{run.method}: {_cmd_text(run.command)}\n"
            )
        log_handle.write("\n")
        if args.execute:
            for run in runs:
                result = _run_command(run, log_handle=log_handle, reuse_existing=bool(args.reuse_existing))
                command_results.append(result)
        else:
            for run in runs:
                command_results.append(
                    {
                        "dataset": run.dataset,
                        "split": run.split,
                        "method": run.method,
                        "command": _cmd_text(run.command),
                        "exit_code": None,
                        "dry_run": True,
                        "score_csv": str(run.score_csv),
                        "run_meta_json": str(run.run_meta_json),
                    }
                )
            log_handle.write("Dry run only. Re-run with --execute to generate scores.\n")

    if not args.execute:
        print(f"Wrote command plan: {command_log}")
        return 0

    paths = write_outputs(
        outdir=outdir,
        statuses=statuses,
        runs=runs,
        command_results=command_results,
        target_fars=tuple(float(x) for x in args.target_far),
        top_n_cases=int(args.top_n_cases),
        execute=bool(args.execute),
        reuse_existing=bool(args.reuse_existing),
    )
    print("Wrote external validation artifacts:")
    print(f"  {command_log}")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
