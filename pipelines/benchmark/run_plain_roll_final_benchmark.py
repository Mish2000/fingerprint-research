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
from typing import Any, Iterable

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.benchmark.evaluate import compute_auc_eer


DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_METHODS = ("sift_plain_roll_v2", "sift", "harris", "classic_v2", "minutiae")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_TARGET_FARS = (0.01, 0.005)
DEFAULT_TAR_FAR_CEILINGS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.10)
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_final_v1"
DEFAULT_SAMPLE_STRATEGY = "balanced_spread"
DEFAULT_SAMPLE_SEED = 13
DEFAULT_LIMIT_PER_SPLIT = 1400
OUTPUT_SCHEMA_VERSION = "plain_roll_final_benchmark_v1"
PAIR_AUDIT_SCHEMA_VERSION = "plain_roll_pair_audit_v1"
RUNTIME_ESTIMATE_SCHEMA_VERSION = "plain_roll_runtime_estimate_v1"
DEFAULT_ESTIMATE_SAFETY_FACTOR = 1.25

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

METRICS_COLUMNS = [
    "method",
    "dataset",
    "split",
    "target_far",
    "threshold",
    "threshold_split",
    "threshold_val_far",
    "threshold_val_false_accepts",
    "tar",
    "far",
    "frr",
    "ta",
    "fr",
    "fa",
    "tr",
    "n_positive",
    "n_negative",
    "n_scored",
    "n_unscored",
    "auc",
    "eer",
    "eer_threshold",
    "enough_negatives_for_target",
    "minimum_negatives_for_target",
    "score_count",
    "score_min",
    "score_p05",
    "score_p25",
    "score_median",
    "score_mean",
    "score_p75",
    "score_p95",
    "score_max",
    "positive_score_mean",
    "negative_score_mean",
    "avg_ms_pair_reported",
    "avg_ms_pair_wall",
    "scores_csv",
    "run_meta_json",
    "method_meta_json",
    "selected_pairs_csv",
]

POSITIVE_ONLY_METRICS_COLUMNS = [
    "method",
    "dataset",
    "split",
    "target_far",
    "threshold",
    "n_positive",
    "true_accepts",
    "false_rejects",
    "tar",
    "frr",
    "auc",
    "eer",
    "avg_ms_pair_reported",
    "scores_csv",
]

NEGATIVE_ONLY_METRICS_COLUMNS = [
    "method",
    "dataset",
    "split",
    "target_far",
    "threshold",
    "threshold_val_far",
    "threshold_val_false_accepts",
    "n_negative",
    "false_accepts",
    "true_rejects",
    "far",
    "tnr",
    "auc",
    "eer",
    "avg_ms_pair_reported",
    "scores_csv",
]

THRESHOLD_SWEEP_COLUMNS = [
    "method",
    "dataset",
    "split",
    "threshold",
    "n_positive",
    "n_negative",
    "true_accepts",
    "false_rejects",
    "false_accepts",
    "true_rejects",
    "tar",
    "far",
    "frr",
    "tnr",
    "score_count",
    "avg_ms_pair_reported",
    "scores_csv",
]

TAR_FAR_DISTRIBUTION_COLUMNS = [
    "method",
    "dataset",
    "split",
    "far_ceiling",
    "threshold",
    "actual_far",
    "tar",
    "frr",
    "tnr",
    "ta",
    "fr",
    "fa",
    "tr",
    "n_positive",
    "n_negative",
    "selection_status",
    "selection_rule",
    "scores_csv",
]

LATENCY_COLUMNS = [
    "method",
    "dataset",
    "split",
    "n_pairs",
    "avg_ms_pair_reported",
    "avg_ms_pair_wall",
    "avg_ms_pair_score_csv",
    "p50_ms_pair_score_csv",
    "p95_ms_pair_score_csv",
    "total_ms_score_csv",
    "meta_avg_ms_pair",
    "meta_p50_ms_pair",
    "meta_p95_ms_pair",
    "meta_total_ms",
    "cache_hits",
    "cache_misses",
    "scores_csv",
    "run_meta_json",
    "method_meta_json",
]

FAILURE_COLUMNS = [
    "method",
    "dataset",
    "split",
    "status",
    "error_type",
    "error_message",
    "returncode",
    "command",
    "scores_csv",
    "run_meta_json",
]

SCORE_TRACEABILITY_COLUMNS = [
    "dataset",
    "pair_id",
    "subject_a",
    "subject_b",
    "finger_position",
    "frgp",
]


class PlainRollFinalBenchmarkError(RuntimeError):
    """Raised for final plain-vs-roll benchmark setup or protocol failures."""


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


@dataclass(frozen=True)
class ScoreRun:
    method: str
    dataset: str
    split: str
    selected_pairs_csv: Path
    scores_csv: Path
    roc_png: Path
    run_meta_json: Path
    command: list[str]
    elapsed_seconds: float | None = None
    reused_existing_scores: bool = False


@dataclass(frozen=True)
class PairAuditReport:
    dataset: str
    split: str
    selected_pairs_csv: Path
    json_path: Path
    markdown_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class RuntimeHistorySample:
    method: str
    dataset: str
    split: str
    elapsed_seconds: float
    n_pairs: int | None
    source_manifest: str


@dataclass(frozen=True)
class RuntimeEstimateRow:
    method: str
    dataset: str
    split: str
    n_pairs: int
    status: str
    estimate_seconds: float | None
    estimate_seconds_with_safety: float | None
    estimate_source: str
    confidence: str
    history_samples: int
    reused_existing_scores: bool
    scores_csv: str


def parse_file_uri(raw: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_csv_arg(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    return tuple(item.strip() for item in raw_items if item.strip())


def _safe_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _pairs_path(dataset: str, split: str, *, repo_root: Path = REPO_ROOT) -> Path | None:
    candidates = [
        repo_root / "data" / "manifests" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "manifests" / dataset / "pairs" / f"pairs_{split}.csv",
        repo_root / "data" / "processed" / dataset / f"pairs_{split}.csv",
        repo_root / "data" / "processed" / dataset / "pairs" / f"pairs_{split}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _dataset_dir(dataset: str, *, repo_root: Path = REPO_ROOT) -> Path:
    candidates = [
        repo_root / "data" / "manifests" / dataset,
        repo_root / "data" / "processed" / dataset,
    ]
    for candidate in candidates:
        if (candidate / "manifest.csv").exists():
            return candidate
    checked = [str(path) for path in candidates]
    raise PlainRollFinalBenchmarkError(f"Could not locate manifest.csv for dataset={dataset!r}. Checked: {checked}")


def _one_plain_one_roll(path_a: Any, path_b: Any) -> bool:
    left = str(path_a).lower().replace("\\", "/")
    right = str(path_b).lower().replace("\\", "/")
    left_plain = "plain" in left
    right_plain = "plain" in right
    left_roll = "roll" in left or "rolled" in left
    right_roll = "roll" in right or "rolled" in right
    return (left_plain and right_roll) or (left_roll and right_plain)


def _target_label_counts(df: pd.DataFrame, limit: int) -> tuple[int, int]:
    positives_available = int((df["label"] == 1).sum())
    negatives_available = int((df["label"] == 0).sum())
    positive_target = min(int(math.ceil(limit / 2)), positives_available)
    negative_target = min(int(math.floor(limit / 2)), negatives_available)
    remaining = max(int(limit) - positive_target - negative_target, 0)
    if remaining and positives_available > positive_target:
        add = min(remaining, positives_available - positive_target)
        positive_target += add
        remaining -= add
    if remaining and negatives_available > negative_target:
        add = min(remaining, negatives_available - negative_target)
        negative_target += add
    return positive_target, negative_target


def _spread_sample(group: pd.DataFrame, count: int, rng: np.random.Generator) -> pd.DataFrame:
    if count <= 0:
        return group.head(0)
    if len(group) <= count:
        return group
    positions = np.arange(len(group))
    bins = np.array_split(positions, count)
    chosen_positions = [int(rng.choice(bucket)) for bucket in bins if len(bucket)]
    return group.iloc[chosen_positions]


def _limit_pairs(
    df: pd.DataFrame,
    limit: int,
    *,
    sample_strategy: str,
    sample_seed: int,
) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)

    positive_target, negative_target = _target_label_counts(df, int(limit))
    positives_all = df[df["label"] == 1]
    negatives_all = df[df["label"] == 0]
    if sample_strategy == "first":
        positives = positives_all.head(positive_target)
        negatives = negatives_all.head(negative_target)
    elif sample_strategy == "balanced_spread":
        rng = np.random.default_rng(int(sample_seed))
        positives = _spread_sample(positives_all, positive_target, rng)
        negatives = _spread_sample(negatives_all, negative_target, rng)
    else:
        raise ValueError(f"Unsupported sample_strategy: {sample_strategy!r}")

    limited = pd.concat([positives, negatives], ignore_index=False).sort_values("_source_order")
    return limited.head(limit).reset_index(drop=True)


def load_plain_roll_pairs(
    dataset: str,
    split: str,
    *,
    repo_root: Path = REPO_ROOT,
    limit: int = DEFAULT_LIMIT_PER_SPLIT,
    sample_strategy: str = DEFAULT_SAMPLE_STRATEGY,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pairs_csv = _pairs_path(dataset, split, repo_root=repo_root)
    if pairs_csv is None:
        return pd.DataFrame(), {
            "dataset": dataset,
            "split": split,
            "compatible": False,
            "reason": "no pairs CSV found",
        }

    df = pd.read_csv(pairs_csv)
    required = {"pair_id", "label", "split", "subject_a", "subject_b", "path_a", "path_b"}
    missing = sorted(required - set(df.columns))
    if missing:
        return pd.DataFrame(), {
            "dataset": dataset,
            "split": split,
            "pairs_csv": str(pairs_csv),
            "compatible": False,
            "reason": f"missing required columns: {missing}",
            "n_pairs": int(len(df)),
        }

    finger_col = "frgp" if "frgp" in df.columns else "finger_id" if "finger_id" in df.columns else None
    if finger_col is None:
        return pd.DataFrame(), {
            "dataset": dataset,
            "split": split,
            "pairs_csv": str(pairs_csv),
            "compatible": False,
            "reason": "missing frgp/finger_id column for same-finger protocol",
            "n_pairs": int(len(df)),
        }

    normalized = df.copy()
    normalized["_source_order"] = np.arange(len(normalized))
    normalized["label"] = pd.to_numeric(normalized["label"], errors="coerce").fillna(-1).astype(int)
    normalized["split"] = normalized["split"].astype(str).str.strip().str.lower()
    normalized["subject_a"] = normalized["subject_a"].astype(str)
    normalized["subject_b"] = normalized["subject_b"].astype(str)
    normalized["finger_position"] = normalized[finger_col].astype(str)
    normalized["dataset"] = dataset

    split_mask = normalized["split"] == split.lower()
    label_mask = normalized["label"].isin([0, 1])
    finger_mask = normalized["finger_position"].astype(str).str.strip() != ""
    plain_roll_mask = normalized.apply(lambda row: _one_plain_one_roll(row["path_a"], row["path_b"]), axis=1)
    same_subject = normalized["subject_a"] == normalized["subject_b"]
    protocol_mask = ((normalized["label"] == 1) & same_subject) | ((normalized["label"] == 0) & ~same_subject)
    filtered_before_sampling = normalized[split_mask & label_mask & finger_mask & plain_roll_mask & protocol_mask].copy()
    filtered = _limit_pairs(
        filtered_before_sampling,
        int(limit),
        sample_strategy=sample_strategy,
        sample_seed=int(sample_seed),
    )

    status = {
        "dataset": dataset,
        "split": split,
        "pairs_csv": str(pairs_csv),
        "compatible": bool(len(filtered)),
        "reason": "positive/negative same-finger plain-vs-roll pairs"
        if len(filtered)
        else "no pairs survived plain-vs-roll protocol filtering",
        "n_pairs": int(len(filtered)),
        "n_positive": int((filtered["label"] == 1).sum()) if len(filtered) else 0,
        "n_negative": int((filtered["label"] == 0).sum()) if len(filtered) else 0,
        "source_n_pairs": int(len(df)),
        "protocol_eligible_pairs": int(len(filtered_before_sampling)),
        "filtered_out_pairs": int(len(df) - len(filtered_before_sampling)),
        "sampled_out_pairs": int(len(filtered_before_sampling) - len(filtered)),
        "finger_column": finger_col,
        "sample_strategy": sample_strategy,
        "sample_seed": int(sample_seed),
        "limit_per_split": int(limit),
    }
    columns = [
        "dataset",
        "split",
        "pair_id",
        "label",
        "subject_a",
        "subject_b",
        "finger_position",
        "path_a",
        "path_b",
    ]
    if finger_col not in columns:
        columns.insert(7, finger_col)
    return filtered[columns].reset_index(drop=True), status


def write_selected_pairs(
    *,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    outdir: Path,
    repo_root: Path,
    limit_per_split: int,
    sample_strategy: str,
    sample_seed: int,
) -> tuple[dict[tuple[str, str], Path], list[dict[str, Any]]]:
    selected_dir = outdir / "selected_pairs"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_paths: dict[tuple[str, str], Path] = {}
    statuses: list[dict[str, Any]] = []

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
            statuses.append(status)
            if pairs.empty:
                raise PlainRollFinalBenchmarkError(
                    f"No compatible pairs for dataset={dataset!r} split={split!r}: {status.get('reason')}"
                )
            path = selected_dir / f"pairs_{dataset}_{split}.csv"
            pairs.to_csv(path, index=False)
            selected_paths[(dataset, split)] = path
            status["selected_pairs_csv"] = str(path)
    return selected_paths, statuses


def _path_for_audit(raw: Any, *, repo_root: Path) -> Path:
    return parse_file_uri(str(raw), repo_root=repo_root)


def _path_key(raw: Any, *, repo_root: Path) -> str:
    try:
        return str(_path_for_audit(raw, repo_root=repo_root)).replace("\\", "/").lower()
    except Exception:
        return str(raw).replace("\\", "/").lower()


def _subject_values(row: pd.Series) -> tuple[str, str]:
    for left, right in (
        ("subject_a", "subject_b"),
        ("subject_id_a", "subject_id_b"),
        ("identity_a", "identity_b"),
        ("identity_id_a", "identity_id_b"),
        ("person_a", "person_b"),
        ("person_id_a", "person_id_b"),
    ):
        if left in row.index and right in row.index:
            return str(row.get(left, "")).strip(), str(row.get(right, "")).strip()
    return "", ""


def _finger_values(row: pd.Series) -> tuple[str, str, str]:
    for left, right in (
        ("frgp_a", "frgp_b"),
        ("finger_id_a", "finger_id_b"),
        ("finger_position_a", "finger_position_b"),
        ("finger_a", "finger_b"),
        ("position_a", "position_b"),
    ):
        if left in row.index and right in row.index:
            return str(row.get(left, "")).strip(), str(row.get(right, "")).strip(), f"{left}/{right}"
    for shared in ("frgp", "finger_id", "finger_position", "finger", "position"):
        if shared in row.index:
            value = str(row.get(shared, "")).strip()
            return value, value, shared
    return "", "", ""


def _finger_bucket(row: pd.Series) -> str:
    left, right, _source = _finger_values(row)
    if left and right and left == right:
        return left
    if left or right:
        return f"{left or 'missing'}->{right or 'missing'}"
    return "unknown"


def audit_pair_dataframe(
    pairs: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    selected_pairs_csv: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    duplicate_keys: set[tuple[str, str]] = set()
    seen_keys: set[tuple[str, str]] = set()
    invalid_examples: list[dict[str, Any]] = []

    counters = {
        "invalid_positive_count": 0,
        "invalid_negative_count": 0,
        "invalid_label_count": 0,
        "duplicate_pair_count": 0,
        "missing_file_count": 0,
        "modality_mismatch_count": 0,
        "subject_mismatch_among_positives": 0,
        "same_subject_negatives": 0,
        "finger_mismatch_count": 0,
        "same_path_count": 0,
    }

    for row_index, row in pairs.iterrows():
        violations: list[str] = []
        try:
            label = int(row.get("label"))
        except (TypeError, ValueError):
            label = -1
            violations.append("label_not_0_or_1")
            counters["invalid_label_count"] += 1

        path_a = str(row.get("path_a", "")).strip()
        path_b = str(row.get("path_b", "")).strip()
        path_a_key = _path_key(path_a, repo_root=repo_root)
        path_b_key = _path_key(path_b, repo_root=repo_root)

        if label not in {0, 1}:
            if "label_not_0_or_1" not in violations:
                violations.append("label_not_0_or_1")
                counters["invalid_label_count"] += 1

        if not path_a or not path_b or path_a_key == path_b_key:
            violations.append("same_or_missing_path")
            counters["same_path_count"] += 1

        missing_refs = 0
        for raw_path in (path_a, path_b):
            if not raw_path or not _path_for_audit(raw_path, repo_root=repo_root).exists():
                missing_refs += 1
        if missing_refs:
            violations.append("missing_file")
            counters["missing_file_count"] += int(missing_refs)

        if not _one_plain_one_roll(path_a, path_b):
            violations.append("modality_mismatch")
            counters["modality_mismatch_count"] += 1

        unordered_key = tuple(sorted((path_a_key, path_b_key)))
        if unordered_key in seen_keys:
            duplicate_keys.add(unordered_key)
            violations.append("duplicate_unordered_pair")
            counters["duplicate_pair_count"] += 1
        else:
            seen_keys.add(unordered_key)

        subject_a, subject_b = _subject_values(row)
        same_subject = bool(subject_a and subject_b and subject_a == subject_b)
        finger_a, finger_b, finger_source = _finger_values(row)
        same_finger = bool(finger_a and finger_b and finger_a == finger_b)
        if not same_finger:
            violations.append("finger_mismatch")
            counters["finger_mismatch_count"] += 1

        if label == 1:
            if not same_subject:
                violations.append("positive_subject_mismatch")
                counters["subject_mismatch_among_positives"] += 1
        elif label == 0:
            if same_subject:
                violations.append("negative_same_subject")
                counters["same_subject_negatives"] += 1

        if violations:
            if label == 1:
                counters["invalid_positive_count"] += 1
            elif label == 0:
                counters["invalid_negative_count"] += 1
            if len(invalid_examples) < 100:
                invalid_examples.append(
                    {
                        "row_index": int(row_index),
                        "pair_id": row.get("pair_id", ""),
                        "label": label,
                        "subject_a": subject_a,
                        "subject_b": subject_b,
                        "finger_a": finger_a,
                        "finger_b": finger_b,
                        "finger_source": finger_source,
                        "path_a": path_a,
                        "path_b": path_b,
                        "violations": violations,
                    }
                )

    labels = pd.to_numeric(pairs.get("label", pd.Series(dtype=int)), errors="coerce")
    positive_mask = labels == 1
    negative_mask = labels == 0
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    positive_by_finger = (
        pairs[positive_mask].apply(_finger_bucket, axis=1).value_counts().sort_index().astype(int).to_dict()
        if len(pairs)
        else {}
    )
    negative_by_finger = (
        pairs[negative_mask].apply(_finger_bucket, axis=1).value_counts().sort_index().astype(int).to_dict()
        if len(pairs)
        else {}
    )

    strict_pass = (
        counters["invalid_positive_count"] == 0
        and counters["invalid_negative_count"] == 0
        and counters["invalid_label_count"] == 0
        and counters["duplicate_pair_count"] == 0
        and counters["missing_file_count"] == 0
        and counters["modality_mismatch_count"] == 0
        and counters["subject_mismatch_among_positives"] == 0
        and counters["same_subject_negatives"] == 0
        and counters["finger_mismatch_count"] == 0
        and positive_count > 0
        and negative_count > 0
    )

    return {
        "schema_version": PAIR_AUDIT_SCHEMA_VERSION,
        "dataset": dataset,
        "split": split,
        "selected_pairs_csv": str(selected_pairs_csv),
        "total_pairs": int(len(pairs)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        **counters,
        "positive_count_by_finger_position": {str(k): int(v) for k, v in positive_by_finger.items()},
        "negative_count_by_finger_position": {str(k): int(v) for k, v in negative_by_finger.items()},
        "pass": bool(strict_pass),
        "invalid_pair_examples": invalid_examples,
    }


def audit_pair_csv(
    pairs_csv: Path,
    *,
    dataset: str,
    split: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    return audit_pair_dataframe(
        pd.read_csv(pairs_csv),
        dataset=dataset,
        split=split,
        selected_pairs_csv=pairs_csv,
        repo_root=repo_root,
    )


def render_pair_audit_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Pair Audit: {summary['dataset']} {summary['split']}",
        "",
        f"Selected pairs: `{summary['selected_pairs_csv']}`",
        f"Pass: `{bool(summary['pass'])}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in (
        "total_pairs",
        "positive_count",
        "negative_count",
        "invalid_positive_count",
        "invalid_negative_count",
        "duplicate_pair_count",
        "missing_file_count",
        "modality_mismatch_count",
        "subject_mismatch_among_positives",
        "same_subject_negatives",
        "finger_mismatch_count",
        "same_path_count",
        "invalid_label_count",
    ):
        lines.append(f"| {key} | {summary.get(key, 0)} |")

    lines.extend(["", "## Finger Counts", "", "| finger position | positives | negatives |", "| --- | ---: | ---: |"])
    positives = summary.get("positive_count_by_finger_position", {}) or {}
    negatives = summary.get("negative_count_by_finger_position", {}) or {}
    for finger in sorted(set(positives) | set(negatives)):
        lines.append(f"| {finger} | {int(positives.get(finger, 0))} | {int(negatives.get(finger, 0))} |")

    examples = summary.get("invalid_pair_examples", []) or []
    if examples:
        lines.extend(["", "## Invalid Examples", ""])
        for item in examples[:25]:
            lines.append(
                f"- row {item.get('row_index')} pair `{item.get('pair_id')}`: "
                f"{', '.join(str(v) for v in item.get('violations', []))}"
            )
    return "\n".join(lines) + "\n"


def write_pair_audits(
    *,
    selected_pairs: dict[tuple[str, str], Path],
    pair_audit_out: Path,
    repo_root: Path,
) -> list[PairAuditReport]:
    pair_audit_out.mkdir(parents=True, exist_ok=True)
    reports: list[PairAuditReport] = []
    for (dataset, split), pairs_csv in sorted(selected_pairs.items()):
        summary = audit_pair_csv(pairs_csv, dataset=dataset, split=split, repo_root=repo_root)
        json_path = pair_audit_out / f"pair_audit_{dataset}_{split}.json"
        markdown_path = pair_audit_out / f"pair_audit_{dataset}_{split}.md"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")
        markdown_path.write_text(render_pair_audit_markdown(summary), encoding="utf-8")
        reports.append(
            PairAuditReport(
                dataset=dataset,
                split=split,
                selected_pairs_csv=pairs_csv,
                json_path=json_path,
                markdown_path=markdown_path,
                summary=summary,
            )
        )
    return reports


def write_pair_audit_summary_markdown(reports: list[PairAuditReport], path: Path) -> None:
    lines = [
        "# Pair Audit Summary",
        "",
        "| dataset | split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for report in reports:
        s = report.summary
        lines.append(
            f"| {report.dataset} | {report.split} | {bool(s['pass'])} | {int(s['total_pairs'])} | "
            f"{int(s['positive_count'])} | {int(s['negative_count'])} | "
            f"{int(s['invalid_positive_count'])} | {int(s['invalid_negative_count'])} | "
            f"{int(s['missing_file_count'])} | {int(s['duplicate_pair_count'])} | "
            f"{int(s['modality_mismatch_count'])} | {int(s['finger_mismatch_count'])} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def enforce_strict_pair_audit(reports: list[PairAuditReport]) -> None:
    failed = [report for report in reports if not bool(report.summary.get("pass"))]
    if not failed:
        return
    details = []
    for report in failed:
        s = report.summary
        details.append(
            f"{report.dataset}/{report.split}: invalid_pos={s['invalid_positive_count']} "
            f"invalid_neg={s['invalid_negative_count']} missing_files={s['missing_file_count']} "
            f"duplicates={s['duplicate_pair_count']} modality={s['modality_mismatch_count']} "
            f"finger={s['finger_mismatch_count']} pos={s['positive_count']} neg={s['negative_count']}"
        )
    raise PlainRollFinalBenchmarkError("Strict pair audit failed: " + "; ".join(details))


def _run_subprocess(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FPRJ_ROOT"] = str(cwd)
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, check=False)


def build_evaluate_command(
    *,
    method: str,
    dataset: str,
    split: str,
    selected_pairs_csv: Path,
    outdir: Path,
    repo_root: Path,
) -> ScoreRun:
    dataset_dir = _dataset_dir(dataset, repo_root=repo_root)
    evaluate_py = repo_root / "pipelines" / "benchmark" / "evaluate.py"
    scores_csv = outdir / f"scores_{dataset}_{method}_{split}.csv"
    roc_png = outdir / f"roc_{dataset}_{method}_{split}.png"
    run_meta_json = outdir / f"run_{dataset}_{method}_{split}.meta.json"
    summary_csv = outdir / "evaluate_results_summary.csv"
    cmd = [
        sys.executable,
        str(evaluate_py),
        "--method",
        method,
        "--dataset",
        dataset,
        "--split",
        split,
        "--data_dir",
        str(dataset_dir),
        "--pairs_file",
        str(selected_pairs_csv),
        "--pair_set_name",
        split,
        "--limit",
        "0",
        "--out_scores",
        str(scores_csv),
        "--out_roc",
        str(roc_png),
        "--out_run_meta",
        str(run_meta_json),
        "--summary_csv",
        str(summary_csv),
    ]
    return ScoreRun(
        method=method,
        dataset=dataset,
        split=split,
        selected_pairs_csv=selected_pairs_csv,
        scores_csv=scores_csv,
        roc_png=roc_png,
        run_meta_json=run_meta_json,
        command=cmd,
    )


def ensure_score_csv_traceability(*, selected_pairs_csv: Path, scores_csv: Path) -> None:
    selected = pd.read_csv(selected_pairs_csv)
    scores = pd.read_csv(scores_csv)
    if len(selected) != len(scores):
        raise PlainRollFinalBenchmarkError(
            f"Score CSV row count does not match selected pairs for {scores_csv}: "
            f"scores={len(scores)} selected_pairs={len(selected)}"
        )

    trace_columns = [column for column in SCORE_TRACEABILITY_COLUMNS if column in selected.columns]
    if not trace_columns:
        return

    updated = scores.copy()
    for column in trace_columns:
        updated[column] = selected[column].to_numpy()
    updated.to_csv(scores_csv, index=False)


def _traceability_failure_row(run: ScoreRun, exc: Exception) -> dict[str, Any]:
    return {
        "method": run.method,
        "dataset": run.dataset,
        "split": run.split,
        "status": "traceability_failed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "returncode": 0,
        "command": " ".join(run.command),
        "scores_csv": str(run.scores_csv),
        "run_meta_json": str(run.run_meta_json),
    }


def run_selected_methods(
    *,
    methods: tuple[str, ...],
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    selected_pairs: dict[tuple[str, str], Path],
    outdir: Path,
    repo_root: Path,
    reuse_existing_scores: bool,
    continue_on_method_failure: bool,
) -> tuple[list[ScoreRun], list[dict[str, Any]]]:
    score_runs: list[ScoreRun] = []
    failures: list[dict[str, Any]] = []
    for method in methods:
        for dataset in datasets:
            for split in splits:
                run = build_evaluate_command(
                    method=method,
                    dataset=dataset,
                    split=split,
                    selected_pairs_csv=selected_pairs[(dataset, split)],
                    outdir=outdir,
                    repo_root=repo_root,
                )
                if reuse_existing_scores and run.scores_csv.exists():
                    try:
                        ensure_score_csv_traceability(
                            selected_pairs_csv=run.selected_pairs_csv,
                            scores_csv=run.scores_csv,
                        )
                    except PlainRollFinalBenchmarkError as exc:
                        row = _traceability_failure_row(run, exc)
                        failures.append(row)
                        if not continue_on_method_failure:
                            raise
                        continue
                    score_runs.append(
                        ScoreRun(
                            method=run.method,
                            dataset=run.dataset,
                            split=run.split,
                            selected_pairs_csv=run.selected_pairs_csv,
                            scores_csv=run.scores_csv,
                            roc_png=run.roc_png,
                            run_meta_json=run.run_meta_json,
                            command=run.command,
                            elapsed_seconds=0.0,
                            reused_existing_scores=True,
                        )
                    )
                    continue

                start = time.perf_counter()
                proc = _run_subprocess(run.command, cwd=repo_root)
                elapsed = time.perf_counter() - start
                if proc.returncode != 0:
                    row = {
                        "method": method,
                        "dataset": dataset,
                        "split": split,
                        "status": "failed",
                        "error_type": "CalledProcessError",
                        "error_message": (proc.stderr or proc.stdout or "").strip(),
                        "returncode": int(proc.returncode),
                        "command": " ".join(run.command),
                        "scores_csv": str(run.scores_csv),
                        "run_meta_json": str(run.run_meta_json),
                    }
                    failures.append(row)
                    if not continue_on_method_failure:
                        raise PlainRollFinalBenchmarkError(
                            f"Method failed for method={method!r} dataset={dataset!r} split={split!r}: "
                            f"{row['error_message']}"
                        )
                    continue
                if not run.scores_csv.exists():
                    row = {
                        "method": method,
                        "dataset": dataset,
                        "split": split,
                        "status": "missing_scores",
                        "error_type": "FileNotFoundError",
                        "error_message": f"evaluate.py did not create {run.scores_csv}",
                        "returncode": int(proc.returncode),
                        "command": " ".join(run.command),
                        "scores_csv": str(run.scores_csv),
                        "run_meta_json": str(run.run_meta_json),
                    }
                    failures.append(row)
                    if not continue_on_method_failure:
                        raise PlainRollFinalBenchmarkError(str(row["error_message"]))
                    continue
                try:
                    ensure_score_csv_traceability(
                        selected_pairs_csv=run.selected_pairs_csv,
                        scores_csv=run.scores_csv,
                    )
                except PlainRollFinalBenchmarkError as exc:
                    row = _traceability_failure_row(run, exc)
                    failures.append(row)
                    if not continue_on_method_failure:
                        raise
                    continue
                score_runs.append(
                    ScoreRun(
                        method=run.method,
                        dataset=run.dataset,
                        split=run.split,
                        selected_pairs_csv=run.selected_pairs_csv,
                        scores_csv=run.scores_csv,
                        roc_png=run.roc_png,
                        run_meta_json=run.run_meta_json,
                        command=run.command,
                        elapsed_seconds=elapsed,
                        reused_existing_scores=False,
                    )
                )
    return score_runs, failures


def _finite_labels_scores(labels: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    labels_arr = np.asarray(labels, dtype=int)
    scores_arr = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores_arr) & np.isin(labels_arr, [0, 1])
    return labels_arr[valid], scores_arr[valid]


def _minimum_negatives_for_target(target_far: float) -> int:
    if target_far <= 0:
        return -1
    return int(math.ceil(1.0 / float(target_far)))


def select_threshold_from_val_negatives(labels: Any, scores: Any, target_far: float) -> ThresholdSelection:
    target = float(target_far)
    if target < 0:
        raise ValueError("target_far must be non-negative.")

    labels_arr, scores_arr = _finite_labels_scores(labels, scores)
    negative_scores = scores_arr[labels_arr == 0]
    positive_count = int(np.sum(labels_arr == 1))
    negative_count = int(negative_scores.size)
    minimum_negatives = _minimum_negatives_for_target(target)
    enough_negatives = bool(negative_count >= minimum_negatives) if minimum_negatives > 0 else False

    if negative_count == 0:
        return ThresholdSelection(
            target_far=target,
            threshold=float("nan"),
            false_accepts=0,
            actual_far=float("nan"),
            negative_count=0,
            positive_count=positive_count,
            enough_negatives_for_target=False,
            minimum_negatives_for_target=minimum_negatives,
        )

    for threshold in sorted(float(x) for x in np.unique(negative_scores)):
        false_accepts = int(np.sum(negative_scores >= threshold))
        actual_far = false_accepts / negative_count
        if actual_far <= target + 1e-15:
            return ThresholdSelection(
                target_far=target,
                threshold=float(threshold),
                false_accepts=false_accepts,
                actual_far=float(actual_far),
                negative_count=negative_count,
                positive_count=positive_count,
                enough_negatives_for_target=enough_negatives,
                minimum_negatives_for_target=minimum_negatives,
            )

    threshold = math.nextafter(float(np.max(negative_scores)), math.inf)
    return ThresholdSelection(
        target_far=target,
        threshold=float(threshold),
        false_accepts=0,
        actual_far=0.0,
        negative_count=negative_count,
        positive_count=positive_count,
        enough_negatives_for_target=enough_negatives,
        minimum_negatives_for_target=minimum_negatives,
    )


def compute_confusion(labels: Any, scores: Any, threshold: float) -> dict[str, Any]:
    labels_arr, scores_arr = _finite_labels_scores(labels, scores)
    positives = labels_arr == 1
    negatives = labels_arr == 0
    if math.isfinite(float(threshold)):
        accepted = scores_arr >= float(threshold)
    else:
        accepted = np.zeros_like(scores_arr, dtype=bool)

    ta = int(np.sum(accepted & positives))
    fr = int(np.sum((~accepted) & positives))
    fa = int(np.sum(accepted & negatives))
    tr = int(np.sum((~accepted) & negatives))
    n_positive = int(np.sum(positives))
    n_negative = int(np.sum(negatives))
    tar = float(ta / n_positive) if n_positive else float("nan")
    far = float(fa / n_negative) if n_negative else float("nan")
    tnr = float(tr / n_negative) if n_negative else float("nan")
    return {
        "tar": tar,
        "far": far,
        "frr": float(1.0 - tar) if math.isfinite(tar) else float("nan"),
        "tnr": tnr,
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def score_distribution_summary(scores: Any) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "score_count": 0,
            "score_min": float("nan"),
            "score_p05": float("nan"),
            "score_p25": float("nan"),
            "score_median": float("nan"),
            "score_mean": float("nan"),
            "score_p75": float("nan"),
            "score_p95": float("nan"),
            "score_max": float("nan"),
        }
    return {
        "score_count": int(values.size),
        "score_min": float(np.min(values)),
        "score_p05": float(np.quantile(values, 0.05)),
        "score_p25": float(np.quantile(values, 0.25)),
        "score_median": float(np.median(values)),
        "score_mean": float(np.mean(values)),
        "score_p75": float(np.quantile(values, 0.75)),
        "score_p95": float(np.quantile(values, 0.95)),
        "score_max": float(np.max(values)),
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _method_meta_candidates(scores_csv: Path, method: str) -> list[Path]:
    if method in {"classic_v2", "harris", "sift", "sift_plain_roll_v2"}:
        return [Path(str(scores_csv) + ".meta.json"), scores_csv.with_suffix(".meta.json")]
    return [scores_csv.with_suffix(".meta.json"), Path(str(scores_csv) + ".meta.json")]


def resolve_method_meta(scores_csv: Path, method: str) -> Path | None:
    for candidate in _method_meta_candidates(scores_csv, method):
        if candidate.exists():
            return candidate
    return None


def _first_finite(*values: Any) -> float:
    for value in values:
        number = _safe_float(value)
        if math.isfinite(number):
            return number
    return float("nan")


def _run_meta_timing(run_meta_json: Path) -> dict[str, Any]:
    run_meta = _read_json(run_meta_json)
    timing = run_meta.get("timing") if isinstance(run_meta.get("timing"), dict) else {}
    row = run_meta.get("row") if isinstance(run_meta.get("row"), dict) else {}
    return {
        "avg_ms_pair_reported": _first_finite(timing.get("avg_ms_pair_reported"), row.get("avg_ms_pair_reported")),
        "avg_ms_pair_wall": _first_finite(timing.get("avg_ms_pair_wall"), row.get("avg_ms_pair_wall")),
        "method_meta_json": run_meta.get("method_meta_json") or row.get("meta_json") or "",
    }


def _score_pair_timing(df: pd.DataFrame) -> np.ndarray:
    if "pair_total_ms" in df.columns:
        values = pd.to_numeric(df["pair_total_ms"], errors="coerce").to_numpy(dtype=float)
        return values[np.isfinite(values)]
    required = {"extract_a_ms", "extract_b_ms", "match_ms"}
    if required <= set(df.columns):
        values = (
            pd.to_numeric(df["extract_a_ms"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["extract_b_ms"], errors="coerce").fillna(0.0)
            + pd.to_numeric(df["match_ms"], errors="coerce").fillna(0.0)
        ).to_numpy(dtype=float)
        return values[np.isfinite(values)]
    return np.asarray([], dtype=float)


def _cache_count(meta: dict[str, Any], key: str) -> int | None:
    direct = _safe_int(meta.get(key))
    if direct is not None:
        return direct
    for nested_key in ("template_cache", "feature_cache", "cache"):
        nested = meta.get(nested_key)
        if isinstance(nested, dict):
            nested_value = _safe_int(nested.get(key.replace("cache_", "")))
            if nested_value is not None:
                return nested_value
            nested_value = _safe_int(nested.get(key))
            if nested_value is not None:
                return nested_value
    return None


def build_latency_rows(score_runs: list[ScoreRun]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in score_runs:
        scores = pd.read_csv(run.scores_csv)
        values = _score_pair_timing(scores)
        run_timing = _run_meta_timing(run.run_meta_json)
        method_meta_path = resolve_method_meta(run.scores_csv, run.method)
        method_meta = _read_json(method_meta_path)
        rows.append(
            {
                "method": run.method,
                "dataset": run.dataset,
                "split": run.split,
                "n_pairs": int(len(scores)),
                "avg_ms_pair_reported": run_timing["avg_ms_pair_reported"],
                "avg_ms_pair_wall": run_timing["avg_ms_pair_wall"],
                "avg_ms_pair_score_csv": float(np.mean(values)) if values.size else float("nan"),
                "p50_ms_pair_score_csv": float(np.median(values)) if values.size else float("nan"),
                "p95_ms_pair_score_csv": float(np.quantile(values, 0.95)) if values.size else float("nan"),
                "total_ms_score_csv": float(np.sum(values)) if values.size else float("nan"),
                "meta_avg_ms_pair": _safe_float(method_meta.get("avg_ms_pair")),
                "meta_p50_ms_pair": _safe_float(method_meta.get("p50_ms_pair")),
                "meta_p95_ms_pair": _safe_float(method_meta.get("p95_ms_pair")),
                "meta_total_ms": _safe_float(method_meta.get("total_ms")),
                "cache_hits": _cache_count(method_meta, "cache_hits"),
                "cache_misses": _cache_count(method_meta, "cache_misses"),
                "scores_csv": str(run.scores_csv),
                "run_meta_json": str(run.run_meta_json),
                "method_meta_json": str(method_meta_path) if method_meta_path is not None else str(run_timing["method_meta_json"]),
            }
        )
    return pd.DataFrame(rows, columns=LATENCY_COLUMNS)


def _score_run_lookup(score_runs: list[ScoreRun]) -> dict[tuple[str, str, str], ScoreRun]:
    return {(run.method, run.dataset, run.split): run for run in score_runs}


def build_threshold_table(score_runs: list[ScoreRun], target_fars: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = _score_run_lookup(score_runs)
    keys = sorted({(method, dataset) for method, dataset, _split in lookup})
    for method, dataset in keys:
        val_run = lookup.get((method, dataset, "val"))
        if val_run is None:
            continue
        val_scores = pd.read_csv(val_run.scores_csv)
        labels = pd.to_numeric(val_scores["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        scores = pd.to_numeric(val_scores["score"], errors="coerce").to_numpy(dtype=float)
        for target_far in target_fars:
            selection = select_threshold_from_val_negatives(labels, scores, float(target_far))
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
                    "selection_rule": "most permissive threshold from VAL negative scores with VAL FAR <= target",
                    "higher_is_more_similar": True,
                    "scores_csv": str(val_run.scores_csv),
                }
            )
    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def build_metrics_table(
    score_runs: list[ScoreRun],
    thresholds: pd.DataFrame,
    latency: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = _score_run_lookup(score_runs)
    latency_lookup = {
        (str(row.method), str(row.dataset), str(row.split)): row
        for row in latency.itertuples(index=False)
    }
    for threshold_row in thresholds.itertuples(index=False):
        scored_splits = sorted(
            split
            for method, dataset, split in lookup
            if method == threshold_row.method and dataset == threshold_row.dataset
        )
        for split in scored_splits:
            run = lookup.get((threshold_row.method, threshold_row.dataset, split))
            if run is None:
                continue
            score_df = pd.read_csv(run.scores_csv)
            labels_all = pd.to_numeric(score_df["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
            scores_all = pd.to_numeric(score_df["score"], errors="coerce").to_numpy(dtype=float)
            labels, scores = _finite_labels_scores(labels_all, scores_all)
            counts = compute_confusion(labels, scores, float(threshold_row.threshold))
            auc_eer = compute_auc_eer(labels, scores)
            summary = score_distribution_summary(scores)
            positive_scores = scores[labels == 1]
            negative_scores = scores[labels == 0]
            timing = latency_lookup.get((threshold_row.method, threshold_row.dataset, split))
            rows.append(
                {
                    "method": threshold_row.method,
                    "dataset": threshold_row.dataset,
                    "split": split,
                    "target_far": float(threshold_row.target_far),
                    "threshold": float(threshold_row.threshold),
                    "threshold_split": "val",
                    "threshold_val_far": float(threshold_row.calibration_far),
                    "threshold_val_false_accepts": int(threshold_row.calibration_false_accepts),
                    "tar": float(counts["tar"]),
                    "far": float(counts["far"]),
                    "frr": float(counts["frr"]),
                    "ta": int(counts["ta"]),
                    "fr": int(counts["fr"]),
                    "fa": int(counts["fa"]),
                    "tr": int(counts["tr"]),
                    "n_positive": int(counts["n_positive"]),
                    "n_negative": int(counts["n_negative"]),
                    "n_scored": int(scores.size),
                    "n_unscored": int(len(score_df) - scores.size),
                    "auc": float(auc_eer.auc),
                    "eer": float(auc_eer.eer),
                    "eer_threshold": float(auc_eer.eer_threshold),
                    "enough_negatives_for_target": bool(threshold_row.enough_negatives_for_target),
                    "minimum_negatives_for_target": int(threshold_row.minimum_negatives_for_target),
                    **summary,
                    "positive_score_mean": float(np.mean(positive_scores)) if positive_scores.size else float("nan"),
                    "negative_score_mean": float(np.mean(negative_scores)) if negative_scores.size else float("nan"),
                    "avg_ms_pair_reported": getattr(timing, "avg_ms_pair_reported", float("nan")),
                    "avg_ms_pair_wall": getattr(timing, "avg_ms_pair_wall", float("nan")),
                    "scores_csv": str(run.scores_csv),
                    "run_meta_json": str(run.run_meta_json),
                    "method_meta_json": getattr(timing, "method_meta_json", ""),
                    "selected_pairs_csv": str(run.selected_pairs_csv),
                }
            )
    return pd.DataFrame(rows, columns=METRICS_COLUMNS)


def _candidate_thresholds(scores: Any) -> list[float]:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    unique_scores = sorted((float(value) for value in np.unique(values)), reverse=True)
    no_accept_threshold = math.nextafter(float(np.max(values)), math.inf)
    return [float(no_accept_threshold), *unique_scores]


def build_threshold_sweep_table(score_runs: list[ScoreRun], latency: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    latency_lookup = {
        (str(row.method), str(row.dataset), str(row.split)): row
        for row in latency.itertuples(index=False)
    }
    for run in sorted(score_runs, key=lambda item: (item.method, item.dataset, item.split)):
        score_df = pd.read_csv(run.scores_csv)
        labels_all = pd.to_numeric(score_df["label"], errors="coerce").fillna(-1).to_numpy(dtype=int)
        scores_all = pd.to_numeric(score_df["score"], errors="coerce").to_numpy(dtype=float)
        labels, scores = _finite_labels_scores(labels_all, scores_all)
        score_count = int(scores.size)
        timing = latency_lookup.get((run.method, run.dataset, run.split))
        for threshold in _candidate_thresholds(scores):
            counts = compute_confusion(labels, scores, float(threshold))
            rows.append(
                {
                    "method": run.method,
                    "dataset": run.dataset,
                    "split": run.split,
                    "threshold": float(threshold),
                    "n_positive": int(counts["n_positive"]),
                    "n_negative": int(counts["n_negative"]),
                    "true_accepts": int(counts["ta"]),
                    "false_rejects": int(counts["fr"]),
                    "false_accepts": int(counts["fa"]),
                    "true_rejects": int(counts["tr"]),
                    "tar": float(counts["tar"]),
                    "far": float(counts["far"]),
                    "frr": float(counts["frr"]),
                    "tnr": float(counts["tnr"]),
                    "score_count": score_count,
                    "avg_ms_pair_reported": getattr(timing, "avg_ms_pair_reported", float("nan")),
                    "scores_csv": str(run.scores_csv),
                }
            )
    return pd.DataFrame(rows, columns=THRESHOLD_SWEEP_COLUMNS)


def build_tar_far_distribution_table(
    threshold_sweep: pd.DataFrame,
    far_ceilings: tuple[float, ...] = DEFAULT_TAR_FAR_CEILINGS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selection_rule = "highest TAR with actual FAR <= ceiling; ties choose the highest threshold (more conservative)"
    if threshold_sweep.empty:
        return pd.DataFrame(columns=TAR_FAR_DISTRIBUTION_COLUMNS)

    sort_columns = ["method", "dataset", "split", "threshold"]
    sweep = threshold_sweep.copy().sort_values(sort_columns, ascending=[True, True, True, False])
    for (method, dataset, split), group in sweep.groupby(["method", "dataset", "split"], sort=True):
        group = group.copy()
        group["_far"] = pd.to_numeric(group["far"], errors="coerce")
        group["_tar"] = pd.to_numeric(group["tar"], errors="coerce")
        group["_threshold"] = pd.to_numeric(group["threshold"], errors="coerce")
        first = group.iloc[0]
        for ceiling in far_ceilings:
            ceiling = float(ceiling)
            eligible = group[np.isfinite(group["_far"]) & (group["_far"] <= ceiling + 1e-15)].copy()
            if eligible.empty:
                rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "split": split,
                        "far_ceiling": ceiling,
                        "threshold": float("nan"),
                        "actual_far": float("nan"),
                        "tar": float("nan"),
                        "frr": float("nan"),
                        "tnr": float("nan"),
                        "ta": None,
                        "fr": None,
                        "fa": None,
                        "tr": None,
                        "n_positive": _safe_int(first.get("n_positive")),
                        "n_negative": _safe_int(first.get("n_negative")),
                        "selection_status": "no_threshold",
                        "selection_rule": selection_rule,
                        "scores_csv": str(first.get("scores_csv", "")),
                    }
                )
                continue

            finite_tar = eligible[np.isfinite(eligible["_tar"])].copy()
            if finite_tar.empty:
                chosen = eligible.sort_values(["_threshold", "_far"], ascending=[False, True]).iloc[0]
                status = "selected_no_positive_pairs"
            else:
                best_tar = float(finite_tar["_tar"].max())
                tied = finite_tar[np.isclose(finite_tar["_tar"], best_tar, rtol=0.0, atol=1e-15)].copy()
                chosen = tied.sort_values(["_threshold", "_far"], ascending=[False, True]).iloc[0]
                status = "selected"

            rows.append(
                {
                    "method": method,
                    "dataset": dataset,
                    "split": split,
                    "far_ceiling": ceiling,
                    "threshold": float(chosen["threshold"]),
                    "actual_far": float(chosen["far"]),
                    "tar": float(chosen["tar"]),
                    "frr": float(chosen["frr"]),
                    "tnr": float(chosen["tnr"]),
                    "ta": int(chosen["true_accepts"]),
                    "fr": int(chosen["false_rejects"]),
                    "fa": int(chosen["false_accepts"]),
                    "tr": int(chosen["true_rejects"]),
                    "n_positive": int(chosen["n_positive"]),
                    "n_negative": int(chosen["n_negative"]),
                    "selection_status": status,
                    "selection_rule": selection_rule,
                    "scores_csv": str(chosen["scores_csv"]),
                }
            )
    return pd.DataFrame(rows, columns=TAR_FAR_DISTRIBUTION_COLUMNS)


def _safe_rate(numerator: Any, denominator: Any) -> float:
    denom = _safe_float(denominator)
    if not math.isfinite(denom) or denom == 0:
        return float("nan")
    num = _safe_float(numerator)
    return float(num / denom) if math.isfinite(num) else float("nan")


def build_positive_only_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(columns=POSITIVE_ONLY_METRICS_COLUMNS)
    for row in metrics.itertuples(index=False):
        rows.append(
            {
                "method": row.method,
                "dataset": row.dataset,
                "split": row.split,
                "target_far": float(row.target_far),
                "threshold": float(row.threshold),
                "n_positive": int(row.n_positive),
                "true_accepts": int(row.ta),
                "false_rejects": int(row.fr),
                "tar": float(row.tar),
                "frr": float(row.frr),
                "auc": float(row.auc),
                "eer": float(row.eer),
                "avg_ms_pair_reported": row.avg_ms_pair_reported,
                "scores_csv": row.scores_csv,
            }
        )
    return pd.DataFrame(rows, columns=POSITIVE_ONLY_METRICS_COLUMNS)


def build_negative_only_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if metrics.empty:
        return pd.DataFrame(columns=NEGATIVE_ONLY_METRICS_COLUMNS)
    for row in metrics.itertuples(index=False):
        tnr = _safe_rate(row.tr, row.n_negative)
        rows.append(
            {
                "method": row.method,
                "dataset": row.dataset,
                "split": row.split,
                "target_far": float(row.target_far),
                "threshold": float(row.threshold),
                "threshold_val_far": float(row.threshold_val_far),
                "threshold_val_false_accepts": int(row.threshold_val_false_accepts),
                "n_negative": int(row.n_negative),
                "false_accepts": int(row.fa),
                "true_rejects": int(row.tr),
                "far": float(row.far),
                "tnr": float(tnr),
                "auc": float(row.auc),
                "eer": float(row.eer),
                "avg_ms_pair_reported": row.avg_ms_pair_reported,
                "scores_csv": row.scores_csv,
            }
        )
    return pd.DataFrame(rows, columns=NEGATIVE_ONLY_METRICS_COLUMNS)


def _write_csv(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = None
    out[columns].to_csv(path, index=False)


def _fmt_float(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "NA"
    return f"{100.0 * number:.2f}%"


def _fmt_int(value: Any) -> str:
    number = _safe_int(value)
    return str(number) if number is not None else "NA"


def render_combined_markdown(
    *,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    tar_far_distribution: pd.DataFrame,
    latency: pd.DataFrame,
    failures: pd.DataFrame,
    dataset_statuses: list[dict[str, Any]],
    pair_audit_reports: list[PairAuditReport],
    output_paths: dict[str, Path],
    total_runtime_s: float,
) -> str:
    lines = [
        "# Plain/Roll Final Benchmark",
        "",
        f"Created: `{_utc_now()}`",
        f"Total runtime: `{total_runtime_s:.2f}s`",
        "",
        "## Protocol",
        "",
        "- Datasets: NIST SD300B and NIST SD300C unless overridden.",
        "- Splits: VAL and TEST.",
        "- Pair filter: one plain capture and one rolled capture.",
        "- Labels: positive pairs must share subject, negative pairs must use different subjects.",
        "- Finger protocol: selected pairs preserve `frgp` or `finger_id` as `finger_position`.",
        "- Thresholds: calibrated on VAL negative scores only and applied unchanged to VAL and TEST.",
        "",
        "Although scoring may be executed on one selected-pair CSV for reproducibility, positive and negative outcomes are audited and reported separately. TAR/FRR are computed only from positive pairs, and FAR/TNR are computed only from negative pairs.",
        "",
        "## Expert TAR/FAR Distribution Summary",
        "",
        "- Fixed operating points show selected calibrated thresholds from VAL negatives applied unchanged to VAL and TEST.",
        "- The threshold sweep shows the full behavior across candidate thresholds from each score CSV.",
        "- TAR/FRR are computed only from positive pairs.",
        "- FAR/TNR are computed only from negative pairs.",
        "- FA means negative pairs incorrectly accepted as matches.",
        "- TR means negative pairs correctly rejected.",
        "- TAR/FAR distribution rows maximize TAR within each FAR ceiling; tied TAR rows use the highest threshold as the more conservative operating point.",
        "",
        "| method | dataset | split | FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    test_distribution = (
        tar_far_distribution[tar_far_distribution["split"].astype(str).str.lower() == "test"].copy()
        if not tar_far_distribution.empty
        else tar_far_distribution
    )
    if test_distribution.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    for _, row in test_distribution.sort_values(["method", "dataset", "far_ceiling"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['split']} | {_fmt_pct(row['far_ceiling'])} | "
            f"{_fmt_float(row['threshold'], 6)} | {_fmt_pct(row['actual_far'])} | {_fmt_pct(row['tar'])} | "
            f"{_fmt_int(row['ta'])} | {_fmt_int(row['fr'])} | {_fmt_int(row['fa'])} | {_fmt_int(row['tr'])} |"
        )
    lines.extend(
        [
            "",
            "## TEST Operating Points",
            "",
            "| method | dataset | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER | avg ms/pair |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    test_metrics = metrics[metrics["split"].astype(str).str.lower() == "test"].copy() if not metrics.empty else metrics
    for _, row in test_metrics.sort_values(["method", "dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['dataset']} | {_fmt_pct(row['target_far'])} | "
            f"{_fmt_float(row['threshold'], 6)} | {_fmt_pct(row['tar'])} | {_fmt_pct(row['far'])} | "
            f"{_fmt_pct(row['frr'])} | {int(row['ta'])}/{int(row['fr'])}/{int(row['fa'])}/{int(row['tr'])} | "
            f"{_fmt_float(row['auc'])} | {_fmt_float(row['eer'])} | {_fmt_float(row['avg_ms_pair_reported'], 3)} |"
        )

    lines.extend(
        [
            "",
            "## VAL Calibration",
            "",
            "| method | dataset | target FAR | threshold | VAL FAR | false accepts / negatives | enough negatives |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in thresholds.sort_values(["method", "dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['dataset']} | {_fmt_pct(row['target_far'])} | "
            f"{_fmt_float(row['threshold'], 6)} | {_fmt_pct(row['calibration_far'])} | "
            f"{int(row['calibration_false_accepts'])}/{int(row['calibration_negative_count'])} | "
            f"{bool(row['enough_negatives_for_target'])} |"
        )

    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| method | dataset | split | N | reported avg ms | score CSV p50 ms | score CSV p95 ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in latency.sort_values(["method", "dataset", "split"]).iterrows():
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['split']} | {int(row['n_pairs'])} | "
            f"{_fmt_float(row['avg_ms_pair_reported'], 3)} | {_fmt_float(row['p50_ms_pair_score_csv'], 3)} | "
            f"{_fmt_float(row['p95_ms_pair_score_csv'], 3)} |"
        )

    if not failures.empty:
        lines.extend(["", "## Failures", ""])
        for _, row in failures.iterrows():
            lines.append(
                f"- {row['method']} {row['dataset']} {row['split']}: {row['status']} "
                f"({row['error_type']}) {row['error_message']}"
            )

    lines.extend(
        [
            "",
            "## Pair Audit",
            "",
            "| dataset | split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for report in pair_audit_reports:
        s = report.summary
        lines.append(
            f"| {report.dataset} | {report.split} | {bool(s['pass'])} | {int(s['total_pairs'])} | "
            f"{int(s['positive_count'])} | {int(s['negative_count'])} | "
            f"{int(s['invalid_positive_count'])} | {int(s['invalid_negative_count'])} | "
            f"{int(s['missing_file_count'])} | {int(s['duplicate_pair_count'])} | "
            f"{int(s['modality_mismatch_count'])} | {int(s['finger_mismatch_count'])} |"
        )

    lines.extend(["", "## Selected Pair Sets", ""])
    for status in sorted(dataset_statuses, key=lambda item: (item.get("dataset", ""), item.get("split", ""))):
        lines.append(
            f"- {status.get('dataset')} {status.get('split')}: {status.get('n_pairs')} pairs "
            f"({status.get('n_positive')} positive, {status.get('n_negative')} negative), "
            f"source `{status.get('pairs_csv')}`, selected `{status.get('selected_pairs_csv')}`"
        )

    lines.extend(["", "## Artifacts", ""])
    for key, path in sorted(output_paths.items()):
        lines.append(f"- {key}: `{path}`")
    return "\n".join(lines) + "\n"


def render_method_dataset_markdown(
    *,
    method: str,
    dataset: str,
    metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    tar_far_distribution: pd.DataFrame,
    latency: pd.DataFrame,
    pair_audit_reports: list[PairAuditReport],
) -> str:
    sub_metrics = metrics[(metrics["method"] == method) & (metrics["dataset"] == dataset)].copy()
    sub_thresholds = thresholds[(thresholds["method"] == method) & (thresholds["dataset"] == dataset)].copy()
    sub_distribution = tar_far_distribution[
        (tar_far_distribution["method"] == method) & (tar_far_distribution["dataset"] == dataset)
    ].copy()
    sub_latency = latency[(latency["method"] == method) & (latency["dataset"] == dataset)].copy()
    sub_audits = [report for report in pair_audit_reports if report.dataset == dataset]
    lines = [
        f"# {method} {dataset} Plain/Roll Final",
        "",
        "Final comparable evidence produced by `pipelines/benchmark/run_plain_roll_final_benchmark.py`.",
        "",
        "## Fixed operating points",
        "",
        "| split | target FAR | threshold | TAR | FAR | FRR | TA/FR/FA/TR | AUC | EER |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in sub_metrics.sort_values(["split", "target_far"]).iterrows():
        lines.append(
            f"| {row['split']} | {_fmt_pct(row['target_far'])} | {_fmt_float(row['threshold'], 6)} | "
            f"{_fmt_pct(row['tar'])} | {_fmt_pct(row['far'])} | {_fmt_pct(row['frr'])} | "
            f"{int(row['ta'])}/{int(row['fr'])}/{int(row['fa'])}/{int(row['tr'])} | "
            f"{_fmt_float(row['auc'])} | {_fmt_float(row['eer'])} |"
        )

    lines.extend(
        [
            "",
            "## TAR vs FAR Distribution",
            "",
            "| FAR ceiling | threshold | actual FAR | TAR | TA | FR | FA | TR |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    test_distribution = (
        sub_distribution[sub_distribution["split"].astype(str).str.lower() == "test"].copy()
        if not sub_distribution.empty
        else sub_distribution
    )
    if test_distribution.empty:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA |")
    for _, row in test_distribution.sort_values("far_ceiling").iterrows():
        lines.append(
            f"| {_fmt_pct(row['far_ceiling'])} | {_fmt_float(row['threshold'], 6)} | "
            f"{_fmt_pct(row['actual_far'])} | {_fmt_pct(row['tar'])} | "
            f"{_fmt_int(row['ta'])} | {_fmt_int(row['fr'])} | {_fmt_int(row['fa'])} | {_fmt_int(row['tr'])} |"
        )

    lines.extend(
        [
            "",
            "## Positive-only verification evidence",
            "",
            "| split | target FAR | threshold | positive pairs | true accepts | false rejects | TAR | FRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in sub_metrics.sort_values(["split", "target_far"]).iterrows():
        lines.append(
            f"| {row['split']} | {_fmt_pct(row['target_far'])} | {_fmt_float(row['threshold'], 6)} | "
            f"{int(row['n_positive'])} | {int(row['ta'])} | {int(row['fr'])} | "
            f"{_fmt_pct(row['tar'])} | {_fmt_pct(row['frr'])} |"
        )

    lines.extend(
        [
            "",
            "## Negative-only impostor evidence",
            "",
            "| split | target FAR | threshold | VAL FAR | VAL false accepts | negative pairs | false accepts | true rejects | FAR | TNR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in sub_metrics.sort_values(["split", "target_far"]).iterrows():
        tnr = _safe_rate(row["tr"], row["n_negative"])
        lines.append(
            f"| {row['split']} | {_fmt_pct(row['target_far'])} | {_fmt_float(row['threshold'], 6)} | "
            f"{_fmt_pct(row['threshold_val_far'])} | {int(row['threshold_val_false_accepts'])} | "
            f"{int(row['n_negative'])} | {int(row['fa'])} | {int(row['tr'])} | "
            f"{_fmt_pct(row['far'])} | {_fmt_pct(tnr)} |"
        )
    lines.extend(["", "## Thresholds", ""])
    for _, row in sub_thresholds.sort_values("target_far").iterrows():
        lines.append(
            f"- Target FAR {_fmt_pct(row['target_far'])}: threshold {_fmt_float(row['threshold'], 6)} "
            f"from VAL negatives, VAL FAR {_fmt_pct(row['calibration_far'])}."
        )
    lines.extend(["", "## Latency", ""])
    for _, row in sub_latency.sort_values("split").iterrows():
        lines.append(
            f"- {row['split']}: reported avg {_fmt_float(row['avg_ms_pair_reported'], 3)} ms/pair, "
            f"score CSV p50 {_fmt_float(row['p50_ms_pair_score_csv'], 3)} ms, "
            f"p95 {_fmt_float(row['p95_ms_pair_score_csv'], 3)} ms."
        )
    lines.extend(
        [
            "",
            "## Pair audit summary",
            "",
            "| split | pass | pairs | positives | negatives | invalid positives | invalid negatives | missing files | duplicates | modality mismatches | finger mismatches |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    if not sub_audits:
        lines.append("| NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |")
    for report in sorted(sub_audits, key=lambda item: item.split):
        s = report.summary
        lines.append(
            f"| {report.split} | {bool(s['pass'])} | {int(s['total_pairs'])} | "
            f"{int(s['positive_count'])} | {int(s['negative_count'])} | "
            f"{int(s['invalid_positive_count'])} | {int(s['invalid_negative_count'])} | "
            f"{int(s['missing_file_count'])} | {int(s['duplicate_pair_count'])} | "
            f"{int(s['modality_mismatch_count'])} | {int(s['finger_mismatch_count'])} |"
        )
    return "\n".join(lines) + "\n"


def _duration_seconds_to_text(value: float | None) -> str:
    number = _safe_float(value)
    if not math.isfinite(number):
        return "unknown"
    seconds = max(int(round(number)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _runtime_history_pairs_by_split(payload: dict[str, Any]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in payload.get("selected_pair_sets", []) or []:
        if not isinstance(row, dict):
            continue
        dataset = str(row.get("dataset", "")).strip()
        split = str(row.get("split", "")).strip().lower()
        n_pairs = _safe_int(row.get("n_pairs"))
        if dataset and split and n_pairs is not None and n_pairs > 0:
            out[(dataset, split)] = int(n_pairs)
    return out


def load_runtime_history(manifest_path: Path | None, *, repo_root: Path = REPO_ROOT) -> list[RuntimeHistorySample]:
    """Load measured score-run runtimes from a previous final benchmark manifest.

    Only positive, finite elapsed_seconds values are used. Existing-score reuse rows
    are intentionally skipped because they describe cache hits, not scoring cost.
    """

    if manifest_path is None:
        return []
    path = parse_file_uri(manifest_path, repo_root=repo_root)
    if not path.exists():
        raise PlainRollFinalBenchmarkError(f"Runtime estimate manifest does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlainRollFinalBenchmarkError(f"Runtime estimate manifest is not valid JSON: {path}") from exc

    n_pairs_by_split = _runtime_history_pairs_by_split(payload)
    samples: list[RuntimeHistorySample] = []
    for row in payload.get("score_runs", []) or []:
        if not isinstance(row, dict):
            continue
        if bool(row.get("reused_existing_scores")):
            continue
        elapsed = _safe_float(row.get("elapsed_seconds"))
        if not math.isfinite(elapsed) or elapsed <= 0:
            continue
        method = str(row.get("method", "")).strip()
        dataset = str(row.get("dataset", "")).strip()
        split = str(row.get("split", "")).strip().lower()
        if not method or not dataset or not split:
            continue
        n_pairs = _safe_int(row.get("n_pairs"))
        if n_pairs is None:
            n_pairs = n_pairs_by_split.get((dataset, split))
        samples.append(
            RuntimeHistorySample(
                method=method,
                dataset=dataset,
                split=split,
                elapsed_seconds=float(elapsed),
                n_pairs=int(n_pairs) if n_pairs is not None and n_pairs > 0 else None,
                source_manifest=str(path),
            )
        )
    return samples


def _scaled_history_seconds(sample: RuntimeHistorySample, current_n_pairs: int) -> float:
    if sample.n_pairs is None or sample.n_pairs <= 0 or current_n_pairs <= 0:
        return float(sample.elapsed_seconds)
    return float(sample.elapsed_seconds) * (float(current_n_pairs) / float(sample.n_pairs))


def _mean_runtime_from_samples(samples: list[RuntimeHistorySample], current_n_pairs: int) -> float | None:
    values = [_scaled_history_seconds(sample, current_n_pairs) for sample in samples]
    values = [value for value in values if math.isfinite(value) and value > 0]
    if not values:
        return None
    return float(sum(values) / len(values))


def estimate_seconds_for_score_run(
    *,
    method: str,
    dataset: str,
    split: str,
    n_pairs: int,
    history: list[RuntimeHistorySample],
) -> tuple[float | None, str, str, int]:
    """Estimate one score run with progressively weaker fallbacks.

    Returns: (seconds, source_label, confidence, sample_count).
    """

    candidates: tuple[tuple[str, str, list[RuntimeHistorySample]], ...] = (
        (
            "exact_method_dataset_split",
            "high",
            [s for s in history if s.method == method and s.dataset == dataset and s.split == split],
        ),
        (
            "method_dataset_average",
            "medium",
            [s for s in history if s.method == method and s.dataset == dataset],
        ),
        (
            "method_average",
            "medium",
            [s for s in history if s.method == method],
        ),
        (
            "dataset_average_weak_fallback",
            "low",
            [s for s in history if s.dataset == dataset],
        ),
        (
            "global_average_weak_fallback",
            "low",
            list(history),
        ),
    )
    for source, confidence, samples in candidates:
        estimated = _mean_runtime_from_samples(samples, n_pairs)
        if estimated is not None:
            return estimated, source, confidence, len(samples)
    return None, "no_history", "unknown", 0


def _dataset_status_n_pairs(dataset_statuses: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for status in dataset_statuses:
        dataset = str(status.get("dataset", "")).strip()
        split = str(status.get("split", "")).strip().lower()
        n_pairs = _safe_int(status.get("n_pairs"))
        if dataset and split and n_pairs is not None:
            out[(dataset, split)] = int(n_pairs)
    return out


def build_score_run_plan(
    *,
    methods: tuple[str, ...],
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    selected_pairs: dict[tuple[str, str], Path],
    outdir: Path,
    repo_root: Path,
) -> list[ScoreRun]:
    runs: list[ScoreRun] = []
    for method in methods:
        for dataset in datasets:
            for split in splits:
                runs.append(
                    build_evaluate_command(
                        method=method,
                        dataset=dataset,
                        split=split,
                        selected_pairs_csv=selected_pairs[(dataset, split)],
                        outdir=outdir,
                        repo_root=repo_root,
                    )
                )
    return runs


def build_runtime_estimate(
    *,
    methods: tuple[str, ...],
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    selected_pairs: dict[tuple[str, str], Path],
    dataset_statuses: list[dict[str, Any]],
    outdir: Path,
    repo_root: Path,
    reuse_existing_scores: bool,
    estimate_from_manifest: Path | None,
    estimate_safety_factor: float,
) -> dict[str, Any]:
    if estimate_safety_factor <= 0:
        raise PlainRollFinalBenchmarkError("estimate_safety_factor must be positive.")

    history = load_runtime_history(estimate_from_manifest, repo_root=repo_root)
    n_pairs_lookup = _dataset_status_n_pairs(dataset_statuses)
    rows: list[RuntimeEstimateRow] = []
    for run in build_score_run_plan(
        methods=methods,
        datasets=datasets,
        splits=splits,
        selected_pairs=selected_pairs,
        outdir=outdir,
        repo_root=repo_root,
    ):
        n_pairs = int(n_pairs_lookup.get((run.dataset, run.split), 0))
        if reuse_existing_scores and run.scores_csv.exists():
            rows.append(
                RuntimeEstimateRow(
                    method=run.method,
                    dataset=run.dataset,
                    split=run.split,
                    n_pairs=n_pairs,
                    status="reuse_existing_scores",
                    estimate_seconds=0.0,
                    estimate_seconds_with_safety=0.0,
                    estimate_source="existing_scores_csv",
                    confidence="high",
                    history_samples=0,
                    reused_existing_scores=True,
                    scores_csv=str(run.scores_csv),
                )
            )
            continue

        estimated, source, confidence, sample_count = estimate_seconds_for_score_run(
            method=run.method,
            dataset=run.dataset,
            split=run.split,
            n_pairs=n_pairs,
            history=history,
        )
        if estimated is None:
            status = "unknown"
            safe_estimated = None
        else:
            status = "estimated" if confidence != "low" else "estimated_low_confidence"
            safe_estimated = float(estimated) * float(estimate_safety_factor)
        rows.append(
            RuntimeEstimateRow(
                method=run.method,
                dataset=run.dataset,
                split=run.split,
                n_pairs=n_pairs,
                status=status,
                estimate_seconds=float(estimated) if estimated is not None else None,
                estimate_seconds_with_safety=float(safe_estimated) if safe_estimated is not None else None,
                estimate_source=source,
                confidence=confidence,
                history_samples=sample_count,
                reused_existing_scores=False,
                scores_csv=str(run.scores_csv),
            )
        )

    known_base = sum(row.estimate_seconds or 0.0 for row in rows)
    known_safe = sum(row.estimate_seconds_with_safety or 0.0 for row in rows)
    summary = {
        "schema_version": RUNTIME_ESTIMATE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "estimate_from_manifest": str(parse_file_uri(estimate_from_manifest, repo_root=repo_root))
        if estimate_from_manifest is not None
        else "",
        "estimate_safety_factor": float(estimate_safety_factor),
        "history_sample_count": int(len(history)),
        "planned_score_runs": int(len(rows)),
        "reused_existing_score_runs": int(sum(1 for row in rows if row.reused_existing_scores)),
        "estimated_score_runs": int(sum(1 for row in rows if row.estimate_seconds is not None and not row.reused_existing_scores)),
        "low_confidence_score_runs": int(sum(1 for row in rows if row.confidence == "low")),
        "unknown_score_runs": int(sum(1 for row in rows if row.estimate_seconds is None)),
        "estimated_seconds_base_known": float(known_base),
        "estimated_seconds_with_safety_known": float(known_safe),
        "estimated_duration_base_known": _duration_seconds_to_text(known_base),
        "estimated_duration_with_safety_known": _duration_seconds_to_text(known_safe),
    }
    return {
        "summary": summary,
        "rows": [row.__dict__ for row in rows],
    }


def render_runtime_estimate_markdown(estimate: dict[str, Any]) -> str:
    summary = estimate.get("summary", {})
    rows = estimate.get("rows", []) or []
    lines = [
        "# Plain/Roll Final Benchmark Runtime Estimate",
        "",
        f"Created: `{summary.get('created_at', '')}`",
        f"Runtime history manifest: `{summary.get('estimate_from_manifest', '') or 'none'}`",
        f"History samples used: `{int(summary.get('history_sample_count', 0))}`",
        f"Safety factor: `{float(summary.get('estimate_safety_factor', 1.0)):.2f}`",
        "",
        "## Summary",
        "",
        f"- Planned score runs: `{int(summary.get('planned_score_runs', 0))}`",
        f"- Reused existing score runs: `{int(summary.get('reused_existing_score_runs', 0))}`",
        f"- Estimated new score runs: `{int(summary.get('estimated_score_runs', 0))}`",
        f"- Low-confidence estimates: `{int(summary.get('low_confidence_score_runs', 0))}`",
        f"- Unknown runs: `{int(summary.get('unknown_score_runs', 0))}`",
        f"- Base known ETA: `{summary.get('estimated_duration_base_known', 'unknown')}`",
        f"- Safety-factor ETA: `{summary.get('estimated_duration_with_safety_known', 'unknown')}`",
        "",
    ]
    if int(summary.get("unknown_score_runs", 0)):
        lines.extend(
            [
                "> Some runs have no runtime history. Run a small pilot such as `--limit_per_split 100` first if you need a tighter ETA.",
                "",
            ]
        )
    if int(summary.get("low_confidence_score_runs", 0)):
        lines.extend(
            [
                "> Low-confidence rows use dataset/global fallback history from other methods, so treat them as rough upper/lower planning hints only.",
                "",
            ]
        )

    lines.extend(
        [
            "## Planned Runs",
            "",
            "| method | dataset | split | pairs | status | confidence | source | samples | base ETA | safety ETA |",
            "| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        base = _duration_seconds_to_text(row.get("estimate_seconds"))
        safe = _duration_seconds_to_text(row.get("estimate_seconds_with_safety"))
        lines.append(
            f"| {row.get('method')} | {row.get('dataset')} | {row.get('split')} | "
            f"{int(row.get('n_pairs') or 0)} | {row.get('status')} | {row.get('confidence')} | "
            f"{row.get('estimate_source')} | {int(row.get('history_samples') or 0)} | {base} | {safe} |"
        )
    return "\n".join(lines) + "\n"


def write_runtime_estimate_artifacts(
    *,
    estimate: dict[str, Any],
    outdir: Path,
) -> dict[str, Path]:
    json_path = outdir / "plain_roll_runtime_estimate.json"
    markdown_path = outdir / "plain_roll_runtime_estimate.md"
    json_path.write_text(json.dumps(estimate, indent=2, ensure_ascii=True), encoding="utf-8")
    markdown_path.write_text(render_runtime_estimate_markdown(estimate), encoding="utf-8")
    return {
        "runtime_estimate_json": json_path,
        "runtime_estimate_markdown": markdown_path,
    }


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


def output_schema() -> dict[str, list[str]]:
    return {
        "plain_roll_final_thresholds.csv": THRESHOLD_COLUMNS,
        "plain_roll_final_metrics.csv": METRICS_COLUMNS,
        "plain_roll_final_positive_only_metrics.csv": POSITIVE_ONLY_METRICS_COLUMNS,
        "plain_roll_final_negative_only_metrics.csv": NEGATIVE_ONLY_METRICS_COLUMNS,
        "plain_roll_final_threshold_sweep.csv": THRESHOLD_SWEEP_COLUMNS,
        "plain_roll_final_tar_far_distribution.csv": TAR_FAR_DISTRIBUTION_COLUMNS,
        "plain_roll_final_latency_summary.csv": LATENCY_COLUMNS,
        "plain_roll_final_failures.csv": FAILURE_COLUMNS,
    }


def write_manifest(
    path: Path,
    *,
    datasets: tuple[str, ...],
    methods: tuple[str, ...],
    splits: tuple[str, ...],
    target_fars: tuple[float, ...],
    dataset_statuses: list[dict[str, Any]],
    pair_audit_reports: list[PairAuditReport],
    score_runs: list[ScoreRun],
    failures: pd.DataFrame,
    output_paths: dict[str, Path],
    total_runtime_s: float,
    repo_root: Path,
    limit_per_split: int,
    sample_strategy: str,
    sample_seed: int,
    select_pairs_only: bool,
    strict_pair_audit: bool,
    estimate_only: bool = False,
    estimate_from_manifest: Path | None = None,
    estimate_safety_factor: float = DEFAULT_ESTIMATE_SAFETY_FACTOR,
    runtime_estimate: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "git": _git_info(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "datasets": list(datasets),
        "methods": list(methods),
        "splits": list(splits),
        "target_fars": [float(x) for x in target_fars],
        "limit_per_split": int(limit_per_split),
        "sample_strategy": sample_strategy,
        "sample_seed": int(sample_seed),
        "select_pairs_only": bool(select_pairs_only),
        "estimate_only": bool(estimate_only),
        "estimate_from_manifest": str(estimate_from_manifest) if estimate_from_manifest is not None else "",
        "estimate_safety_factor": float(estimate_safety_factor),
        "strict_pair_audit": bool(strict_pair_audit),
        "pair_protocol": {
            "plain_roll_filter": "exactly one path contains plain and the other contains roll/rolled",
            "labels": "labels must be 0 or 1",
            "positive_rule": "label 1 requires subject_a == subject_b",
            "negative_rule": "label 0 requires subject_a != subject_b",
            "finger_position": "frgp or finger_id is preserved as finger_position",
        },
        "pair_audit_schema_version": PAIR_AUDIT_SCHEMA_VERSION,
        "pair_audits": [
            {
                "dataset": report.dataset,
                "split": report.split,
                "selected_pairs_csv": str(report.selected_pairs_csv),
                "json": str(report.json_path),
                "markdown": str(report.markdown_path),
                "pass": bool(report.summary.get("pass")),
                "summary": {
                    key: report.summary.get(key)
                    for key in (
                        "total_pairs",
                        "positive_count",
                        "negative_count",
                        "invalid_positive_count",
                        "invalid_negative_count",
                        "duplicate_pair_count",
                        "missing_file_count",
                        "modality_mismatch_count",
                        "subject_mismatch_among_positives",
                        "same_subject_negatives",
                        "finger_mismatch_count",
                    )
                },
            }
            for report in pair_audit_reports
        ],
        "score_semantics": {
            "higher_score_more_similar": True,
            "threshold_calibration": "VAL negative scores only",
            "threshold_application": "unchanged threshold applied to VAL and TEST",
            "threshold_sweep": "candidate thresholds are the finite score values plus a no-accept threshold above the maximum finite score",
            "tar_far_distribution": "highest TAR with actual FAR <= ceiling; tied TAR rows choose the highest threshold as the more conservative operating point",
        },
        "selected_pair_sets": dataset_statuses,
        "score_runs": [
            {
                "method": run.method,
                "dataset": run.dataset,
                "split": run.split,
                "selected_pairs_csv": str(run.selected_pairs_csv),
                "scores_csv": str(run.scores_csv),
                "roc_png": str(run.roc_png),
                "run_meta_json": str(run.run_meta_json),
                "command": run.command,
                "elapsed_seconds": run.elapsed_seconds,
                "reused_existing_scores": bool(run.reused_existing_scores),
            }
            for run in score_runs
        ],
        "failure_count": int(len(failures)),
        "outputs": {key: str(value) for key, value in output_paths.items()},
        "output_schema": output_schema(),
        "total_runtime_s": float(total_runtime_s),
    }
    if runtime_estimate is not None:
        payload["runtime_estimate"] = runtime_estimate.get("summary", {})
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_benchmark(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    outdir: Path = DEFAULT_OUTDIR,
    target_fars: tuple[float, ...] = DEFAULT_TARGET_FARS,
    limit_per_split: int = DEFAULT_LIMIT_PER_SPLIT,
    sample_strategy: str = DEFAULT_SAMPLE_STRATEGY,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    reuse_existing_scores: bool = False,
    continue_on_method_failure: bool = False,
    select_pairs_only: bool = False,
    estimate_only: bool = False,
    estimate_from_manifest: Path | None = None,
    estimate_safety_factor: float = DEFAULT_ESTIMATE_SAFETY_FACTOR,
    strict_pair_audit: bool = False,
    pair_audit_out: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    if "sourceafis_open" in set(methods) or "sourceafis" in set(methods):
        raise PlainRollFinalBenchmarkError("SourceAFIS is intentionally excluded from this generic runner.")
    if sample_strategy not in {"first", "balanced_spread"}:
        raise PlainRollFinalBenchmarkError(f"Unsupported sample_strategy: {sample_strategy!r}")
    if "val" not in splits:
        raise PlainRollFinalBenchmarkError("The final benchmark requires the val split for threshold calibration.")

    start = time.perf_counter()
    output = parse_file_uri(outdir, repo_root=repo_root)
    output.mkdir(parents=True, exist_ok=True)

    selected_pairs, dataset_statuses = write_selected_pairs(
        datasets=datasets,
        splits=splits,
        outdir=output,
        repo_root=repo_root,
        limit_per_split=int(limit_per_split),
        sample_strategy=sample_strategy,
        sample_seed=int(sample_seed),
    )
    audit_dir = parse_file_uri(pair_audit_out, repo_root=repo_root) if pair_audit_out is not None else (output / "pair_audit")
    pair_audit_reports = write_pair_audits(
        selected_pairs=selected_pairs,
        pair_audit_out=audit_dir,
        repo_root=repo_root,
    )
    pair_audit_summary = audit_dir / "pair_audit_summary.md"
    write_pair_audit_summary_markdown(pair_audit_reports, pair_audit_summary)
    if strict_pair_audit:
        enforce_strict_pair_audit(pair_audit_reports)

    paths: dict[str, Path] = {
        "pair_audit_summary": pair_audit_summary,
        "manifest": output / "plain_roll_final_manifest.json",
    }
    for (dataset, split), selected_path in selected_pairs.items():
        paths[f"selected_pairs_{dataset}_{split}"] = selected_path
    for report in pair_audit_reports:
        paths[f"pair_audit_json_{report.dataset}_{report.split}"] = report.json_path
        paths[f"pair_audit_markdown_{report.dataset}_{report.split}"] = report.markdown_path

    if estimate_only:
        runtime_estimate = build_runtime_estimate(
            methods=methods,
            datasets=datasets,
            splits=splits,
            selected_pairs=selected_pairs,
            dataset_statuses=dataset_statuses,
            outdir=output,
            repo_root=repo_root,
            reuse_existing_scores=bool(reuse_existing_scores),
            estimate_from_manifest=estimate_from_manifest,
            estimate_safety_factor=float(estimate_safety_factor),
        )
        estimate_paths = write_runtime_estimate_artifacts(
            estimate=runtime_estimate,
            outdir=output,
        )
        paths.update(estimate_paths)
        total_runtime_s = time.perf_counter() - start
        empty_failures = pd.DataFrame(columns=FAILURE_COLUMNS)
        write_manifest(
            paths["manifest"],
            datasets=datasets,
            methods=methods,
            splits=splits,
            target_fars=tuple(float(x) for x in target_fars),
            dataset_statuses=dataset_statuses,
            pair_audit_reports=pair_audit_reports,
            score_runs=[],
            failures=empty_failures,
            output_paths=paths,
            total_runtime_s=total_runtime_s,
            repo_root=repo_root,
            limit_per_split=int(limit_per_split),
            sample_strategy=sample_strategy,
            sample_seed=int(sample_seed),
            select_pairs_only=False,
            strict_pair_audit=bool(strict_pair_audit),
            estimate_only=True,
            estimate_from_manifest=parse_file_uri(estimate_from_manifest, repo_root=repo_root)
            if estimate_from_manifest is not None
            else None,
            estimate_safety_factor=float(estimate_safety_factor),
            runtime_estimate=runtime_estimate,
        )
        print(render_runtime_estimate_markdown(runtime_estimate))
        return paths

    if select_pairs_only:
        total_runtime_s = time.perf_counter() - start
        empty_failures = pd.DataFrame(columns=FAILURE_COLUMNS)
        write_manifest(
            paths["manifest"],
            datasets=datasets,
            methods=methods,
            splits=splits,
            target_fars=tuple(float(x) for x in target_fars),
            dataset_statuses=dataset_statuses,
            pair_audit_reports=pair_audit_reports,
            score_runs=[],
            failures=empty_failures,
            output_paths=paths,
            total_runtime_s=total_runtime_s,
            repo_root=repo_root,
            limit_per_split=int(limit_per_split),
            sample_strategy=sample_strategy,
            sample_seed=int(sample_seed),
            select_pairs_only=True,
            strict_pair_audit=bool(strict_pair_audit),
        )
        return paths

    score_runs, failure_rows = run_selected_methods(
        methods=methods,
        datasets=datasets,
        splits=splits,
        selected_pairs=selected_pairs,
        outdir=output,
        repo_root=repo_root,
        reuse_existing_scores=bool(reuse_existing_scores),
        continue_on_method_failure=bool(continue_on_method_failure),
    )
    if not score_runs:
        raise PlainRollFinalBenchmarkError("No method score CSVs were produced.")

    latency = build_latency_rows(score_runs)
    thresholds = build_threshold_table(score_runs, tuple(float(x) for x in target_fars))
    metrics = build_metrics_table(score_runs, thresholds, latency)
    positive_only_metrics = build_positive_only_metrics_table(metrics)
    negative_only_metrics = build_negative_only_metrics_table(metrics)
    threshold_sweep = build_threshold_sweep_table(score_runs, latency)
    tar_far_distribution = build_tar_far_distribution_table(threshold_sweep)
    failures = pd.DataFrame(failure_rows, columns=FAILURE_COLUMNS)

    paths.update(
        {
            "thresholds": output / "plain_roll_final_thresholds.csv",
            "metrics": output / "plain_roll_final_metrics.csv",
            "positive_only_metrics": output / "plain_roll_final_positive_only_metrics.csv",
            "negative_only_metrics": output / "plain_roll_final_negative_only_metrics.csv",
            "threshold_sweep": output / "plain_roll_final_threshold_sweep.csv",
            "tar_far_distribution": output / "plain_roll_final_tar_far_distribution.csv",
            "latency_summary": output / "plain_roll_final_latency_summary.csv",
            "failures": output / "plain_roll_final_failures.csv",
            "summary": output / "plain_roll_final_summary.md",
        }
    )

    _write_csv(paths["thresholds"], thresholds, THRESHOLD_COLUMNS)
    _write_csv(paths["metrics"], metrics, METRICS_COLUMNS)
    _write_csv(paths["positive_only_metrics"], positive_only_metrics, POSITIVE_ONLY_METRICS_COLUMNS)
    _write_csv(paths["negative_only_metrics"], negative_only_metrics, NEGATIVE_ONLY_METRICS_COLUMNS)
    _write_csv(paths["threshold_sweep"], threshold_sweep, THRESHOLD_SWEEP_COLUMNS)
    _write_csv(paths["tar_far_distribution"], tar_far_distribution, TAR_FAR_DISTRIBUTION_COLUMNS)
    _write_csv(paths["latency_summary"], latency, LATENCY_COLUMNS)
    _write_csv(paths["failures"], failures, FAILURE_COLUMNS)

    markdown_dir = output / "final_markdown"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    for method in sorted({run.method for run in score_runs}):
        for dataset in sorted({run.dataset for run in score_runs if run.method == method}):
            md_path = markdown_dir / f"{dataset}_{method}_plain_roll_final.md"
            md_path.write_text(
                render_method_dataset_markdown(
                    method=method,
                    dataset=dataset,
                    metrics=metrics,
                    thresholds=thresholds,
                    tar_far_distribution=tar_far_distribution,
                    latency=latency,
                    pair_audit_reports=pair_audit_reports,
                ),
                encoding="utf-8",
            )
            paths[f"markdown_{dataset}_{method}"] = md_path

    total_runtime_s = time.perf_counter() - start
    paths["summary"].write_text(
        render_combined_markdown(
            metrics=metrics,
            thresholds=thresholds,
            tar_far_distribution=tar_far_distribution,
            latency=latency,
            failures=failures,
            dataset_statuses=dataset_statuses,
            pair_audit_reports=pair_audit_reports,
            output_paths=paths,
            total_runtime_s=total_runtime_s,
        ),
        encoding="utf-8",
    )
    write_manifest(
        paths["manifest"],
        datasets=datasets,
        methods=methods,
        splits=splits,
        target_fars=tuple(float(x) for x in target_fars),
        dataset_statuses=dataset_statuses,
        pair_audit_reports=pair_audit_reports,
        score_runs=score_runs,
        failures=failures,
        output_paths=paths,
        total_runtime_s=total_runtime_s,
        repo_root=repo_root,
        limit_per_split=int(limit_per_split),
        sample_strategy=sample_strategy,
        sample_seed=int(sample_seed),
        select_pairs_only=False,
        strict_pair_audit=bool(strict_pair_audit),
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the official generic NIST plain-vs-roll final benchmark for non-SourceAFIS methods. "
            "Thresholds are calibrated on VAL negative scores only and applied unchanged to TEST."
        )
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--target_far", type=float, nargs="*", default=list(DEFAULT_TARGET_FARS))
    parser.add_argument("--limit_per_split", type=int, default=DEFAULT_LIMIT_PER_SPLIT)
    parser.add_argument(
        "--sample_strategy",
        choices=("first", "balanced_spread"),
        default=DEFAULT_SAMPLE_STRATEGY,
    )
    parser.add_argument("--sample_seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--reuse_existing_scores", action="store_true")
    parser.add_argument("--estimate_only", action="store_true", help="Write pair audits and a runtime ETA, then exit before scoring.")
    parser.add_argument(
        "--estimate_from_manifest",
        default="",
        help="Previous plain_roll_final_manifest.json to use as runtime history for --estimate_only.",
    )
    parser.add_argument(
        "--estimate_safety_factor",
        type=float,
        default=DEFAULT_ESTIMATE_SAFETY_FACTOR,
        help="Multiplier applied to known runtime estimates in --estimate_only output.",
    )
    parser.add_argument("--continue_on_method_failure", action="store_true")
    parser.add_argument(
        "--select_pairs_only",
        action="store_true",
        help="Write selected VAL/TEST pair CSVs and pair audits, then exit before method scoring.",
    )
    parser.add_argument(
        "--strict_pair_audit",
        action="store_true",
        help="Fail before scoring unless every selected pair audit passes strict positive/negative checks.",
    )
    parser.add_argument(
        "--pair_audit_out",
        default="",
        help="Optional pair-audit output directory. Defaults to <outdir>/pair_audit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_benchmark(
            datasets=_parse_csv_arg(args.datasets),
            methods=_parse_csv_arg(args.methods),
            splits=tuple(item.lower() for item in _parse_csv_arg(args.splits)),
            outdir=parse_file_uri(args.outdir),
            target_fars=tuple(float(x) for x in args.target_far),
            limit_per_split=int(args.limit_per_split),
            sample_strategy=str(args.sample_strategy),
            sample_seed=int(args.sample_seed),
            reuse_existing_scores=bool(args.reuse_existing_scores),
            continue_on_method_failure=bool(args.continue_on_method_failure),
            select_pairs_only=bool(args.select_pairs_only),
            estimate_only=bool(args.estimate_only),
            estimate_from_manifest=parse_file_uri(args.estimate_from_manifest)
            if str(args.estimate_from_manifest).strip()
            else None,
            estimate_safety_factor=float(args.estimate_safety_factor),
            strict_pair_audit=bool(args.strict_pair_audit),
            pair_audit_out=parse_file_uri(args.pair_audit_out) if str(args.pair_audit_out).strip() else None,
            repo_root=REPO_ROOT,
        )
    except PlainRollFinalBenchmarkError as exc:
        print(f"Plain/Roll final benchmark failed: {exc}", file=sys.stderr)
        return 2

    print("Wrote Plain/Roll final benchmark artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
