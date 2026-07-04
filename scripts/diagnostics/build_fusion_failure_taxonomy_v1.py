from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
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

from src.fpbench.universal.calibration import select_threshold_from_negative_scores  # noqa: E402
from src.fpbench.universal.fusion_features import METHOD_NAME, add_quality_features  # noqa: E402


SCHEMA_VERSION = "fusion_failure_taxonomy_v1"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
DEFAULT_TARGET_FAR = 0.01
DEFAULT_FUSION_DIR = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_final_fusion_v1_full_pairs"
)
DEFAULT_SIFT_SCORE_DIR = REPO_ROOT / "artifacts" / "reports" / "benchmark" / "plain_roll_full_scores_v1" / "sift"
DEFAULT_OUTDIR = REPO_ROOT / "artifacts" / "reports" / "diagnostics" / "fusion_failure_taxonomy_v1"
DEFAULT_COMPARISON_CSV = DEFAULT_FUSION_DIR / "plain_roll_final_statistical_comparison.csv"

TARGET_LABEL = "far_1pct"
CATEGORY_COLUMNS = [
    "both_correct",
    "both_wrong",
    "rescued_positive",
    "lost_positive",
    "fixed_false_accept",
    "new_false_accept",
]
CORRECTNESS_COLUMNS = [
    "both_correct",
    "sourceafis_only_correct",
    "fusion_only_correct",
    "both_wrong",
]
SCORE_METHODS = [
    ("sourceafis", "sourceafis_score", "sourceafis_threshold_at_far_1pct"),
    ("sift_plain_roll_v2", "sift_plain_roll_v2_score", "sift_plain_roll_v2_threshold_at_far_1pct"),
    ("fusion", "fusion_score", "fusion_threshold_at_far_1pct"),
]
QUALITY_FEATURE_BASES = [
    "image_read_ok",
    "width",
    "height",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "contrast_proxy",
    "foreground_ratio",
    "sharpness_laplacian_var",
    "edge_density",
]
QUALITY_BAND_METRICS = {
    "contrast": "pair_min_contrast_proxy",
    "sharpness": "pair_min_sharpness_laplacian_var",
    "foreground_ratio": "pair_min_foreground_ratio",
    "contrast_delta": "pair_contrast_proxy_abs_delta",
    "sharpness_delta": "pair_sharpness_laplacian_var_abs_delta",
    "overall_quality": "pair_quality_index",
}
REQUIRED_OUTPUT_NAMES = [
    "failure_taxonomy_pairs",
    "failure_taxonomy_summary",
    "failure_taxonomy_by_dataset",
    "failure_taxonomy_by_finger",
    "failure_taxonomy_by_score_band",
    "failure_taxonomy_by_quality_band",
    "rescued_positive_examples",
    "lost_positive_examples",
    "fixed_false_accept_examples",
    "new_false_accept_examples",
]


class FusionFailureTaxonomyError(RuntimeError):
    """Raised when the fusion failure taxonomy cannot be built."""


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


def _parse_csv_arg(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    return tuple(item.strip() for item in raw_items if item.strip())


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{100.0 * number:.2f}%"


def _fmt_float(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number:.{digits}g}"


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


def _fusion_score_path(fusion_dir: Path, dataset: str, split: str) -> Path:
    return fusion_dir / "scores" / f"scores_{dataset}_{METHOD_NAME}_{split}.csv"


def _sift_score_path(sift_score_dir: Path, dataset: str, split: str) -> Path:
    return sift_score_dir / f"scores_{dataset}_sift_plain_roll_v2_{split}.csv"


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], *, table_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise FusionFailureTaxonomyError(f"{table_name} is missing required columns: {missing}")


def _normalise_pair_keys(df: pd.DataFrame, *, dataset: str, split: str) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = dataset if "dataset" not in out.columns else out["dataset"].fillna(dataset)
    out["split"] = split if "split" not in out.columns else out["split"].fillna(split)
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(-1).astype(int)
    return out[(out["dataset"] == dataset) & (out["split"] == split.lower())].copy()


def _read_fusion_scores(path: Path, *, dataset: str, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FusionFailureTaxonomyError(f"Missing fusion score CSV: {path}")
    df = pd.read_csv(path)
    required = [
        "dataset",
        "split",
        "pair_id",
        "label",
        "subject_a",
        "subject_b",
        "path_a",
        "path_b",
        "score",
        "sourceafis_score",
        "sift_plain_roll_v2_score",
    ]
    _ensure_columns(df, required, table_name=str(path))
    out = _normalise_pair_keys(df, dataset=dataset, split=split)
    if "finger_position" not in out.columns and "frgp" in out.columns:
        out["finger_position"] = out["frgp"]
    if "frgp" not in out.columns and "finger_position" in out.columns:
        out["frgp"] = out["finger_position"]
    _ensure_columns(out, ["finger_position", "frgp"], table_name=str(path))
    out = out.rename(columns={"score": "fusion_score"})
    out["sourceafis_score"] = pd.to_numeric(out["sourceafis_score"], errors="coerce")
    out["sift_plain_roll_v2_score"] = pd.to_numeric(out["sift_plain_roll_v2_score"], errors="coerce")
    out["fusion_score"] = pd.to_numeric(out["fusion_score"], errors="coerce")
    return out.reset_index(drop=True)


def _read_sift_geometry(path: Path, *, dataset: str, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FusionFailureTaxonomyError(f"Missing SIFT Plain/Roll v2 score CSV: {path}")
    df = pd.read_csv(path)
    _ensure_columns(df, ["dataset", "split", "pair_id", "label", "score"], table_name=str(path))
    out = _normalise_pair_keys(df, dataset=dataset, split=split)
    keep = ["dataset", "split", "pair_id", "label", "score"]
    for column in ("inliers", "matches", "k1", "k2", "pair_total_ms", "extract_a_ms", "extract_b_ms", "match_ms"):
        if column in out.columns:
            keep.append(column)
    renamed = out[keep].rename(
        columns={
            "label": "sift_plain_roll_v2_label",
            "score": "sift_plain_roll_v2_score_from_sift_csv",
            "inliers": "sift_plain_roll_v2_inliers",
            "matches": "sift_plain_roll_v2_matches",
            "k1": "sift_plain_roll_v2_k1",
            "k2": "sift_plain_roll_v2_k2",
            "pair_total_ms": "sift_plain_roll_v2_pair_total_ms",
            "extract_a_ms": "sift_plain_roll_v2_extract_a_ms",
            "extract_b_ms": "sift_plain_roll_v2_extract_b_ms",
            "match_ms": "sift_plain_roll_v2_match_ms",
        }
    )
    duplicates = renamed.duplicated(["dataset", "split", "pair_id"], keep=False)
    if bool(duplicates.any()):
        examples = renamed.loc[duplicates, ["dataset", "split", "pair_id"]].head(5).to_dict("records")
        raise FusionFailureTaxonomyError(f"SIFT score table has duplicate pair keys: {examples}")
    return renamed.reset_index(drop=True)


def load_pair_scores(
    *,
    datasets: tuple[str, ...],
    splits: tuple[str, ...],
    fusion_dir: Path,
    sift_score_dir: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for split in splits:
            fusion = _read_fusion_scores(_fusion_score_path(fusion_dir, dataset, split), dataset=dataset, split=split)
            sift = _read_sift_geometry(_sift_score_path(sift_score_dir, dataset, split), dataset=dataset, split=split)
            merged = fusion.merge(
                sift,
                on=["dataset", "split", "pair_id"],
                how="left",
                validate="one_to_one",
            )
            if merged["sift_plain_roll_v2_score_from_sift_csv"].isna().any():
                missing = merged.loc[
                    merged["sift_plain_roll_v2_score_from_sift_csv"].isna(),
                    ["dataset", "split", "pair_id"],
                ].head(5)
                raise FusionFailureTaxonomyError(
                    f"Missing SIFT Plain/Roll v2 geometry rows for {dataset}/{split}: "
                    f"{missing.to_dict('records')}"
                )
            labels_match = merged["sift_plain_roll_v2_label"].astype(int) == merged["label"].astype(int)
            if not bool(labels_match.all()):
                examples = merged.loc[
                    ~labels_match,
                    ["dataset", "split", "pair_id", "label", "sift_plain_roll_v2_label"],
                ].head(5)
                raise FusionFailureTaxonomyError(f"SIFT labels do not match fusion labels: {examples.to_dict('records')}")
            score_delta = (
                pd.to_numeric(merged["sift_plain_roll_v2_score"], errors="coerce")
                - pd.to_numeric(merged["sift_plain_roll_v2_score_from_sift_csv"], errors="coerce")
            ).abs()
            mismatched = np.isfinite(score_delta) & (score_delta > 1e-6)
            if bool(mismatched.any()):
                examples = merged.loc[
                    mismatched,
                    ["dataset", "split", "pair_id", "sift_plain_roll_v2_score", "sift_plain_roll_v2_score_from_sift_csv"],
                ].head(5)
                raise FusionFailureTaxonomyError(
                    "Embedded fusion SIFT scores do not match SIFT CSV scores: "
                    f"{examples.to_dict('records')}"
                )
            frames.append(merged.drop(columns=["sift_plain_roll_v2_label"]).reset_index(drop=True))
    if not frames:
        raise FusionFailureTaxonomyError("No pair scores were loaded.")
    pairs = pd.concat(frames, ignore_index=True, sort=False)
    duplicates = pairs.duplicated(["dataset", "split", "pair_id"], keep=False)
    if bool(duplicates.any()):
        examples = pairs.loc[duplicates, ["dataset", "split", "pair_id"]].head(5).to_dict("records")
        raise FusionFailureTaxonomyError(f"Pair taxonomy input has duplicate keys: {examples}")
    return pairs


def _threshold_rows(table: pd.DataFrame, *, target_far: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in table.groupby("dataset", sort=True):
        val = group[group["split"].astype(str).str.lower() == "val"].copy()
        if val.empty:
            raise FusionFailureTaxonomyError(f"Missing VAL rows for threshold calibration: dataset={dataset}")
        labels = pd.to_numeric(val["label"], errors="coerce").fillna(-1).astype(int)
        positive_count = int((labels == 1).sum())
        for method, score_column, _threshold_column in SCORE_METHODS:
            scores = pd.to_numeric(val[score_column], errors="coerce")
            selection = select_threshold_from_negative_scores(
                scores[labels == 0],
                target_far=float(target_far),
                positive_count=positive_count,
            )
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "score_column": score_column,
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
                }
            )
    return pd.DataFrame(rows)


def _score_percentiles(values: pd.Series, reference: pd.Series) -> pd.Series:
    ref = pd.to_numeric(reference, errors="coerce").to_numpy(dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    if ref.size == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    finite = np.isfinite(vals)
    out[finite] = np.searchsorted(ref, vals[finite], side="right") / float(ref.size)
    return pd.Series(out, index=values.index, dtype=float)


def _percentile_band(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(number):
        return "missing"
    if number < 0.25:
        return "p00_p25"
    if number < 0.50:
        return "p25_p50"
    if number < 0.75:
        return "p50_p75"
    if number < 0.90:
        return "p75_p90"
    if number < 0.99:
        return "p90_p99"
    return "p99_p100"


def _quantile_band(value: Any, low: float, high: float, *, high_is_low_quality: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(number):
        return "missing"
    if high_is_low_quality:
        if number <= low:
            return "low_delta"
        if number <= high:
            return "mid_delta"
        return "high_delta"
    if number <= low:
        return "low"
    if number <= high:
        return "mid"
    return "high"


def _truth_decision(accepted: bool) -> str:
    return "accept" if bool(accepted) else "reject"


def _is_correct(label: int, accepted: bool) -> bool:
    return bool(accepted) if int(label) == 1 else not bool(accepted)


def _category(row: pd.Series) -> str:
    label = int(row["label"])
    source_accept = bool(row["sourceafis_accept_at_far_1pct"])
    fusion_accept = bool(row["fusion_accept_at_far_1pct"])
    source_correct = _is_correct(label, source_accept)
    fusion_correct = _is_correct(label, fusion_accept)
    if source_correct and fusion_correct:
        return "both_correct"
    if (not source_correct) and (not fusion_correct):
        return "both_wrong"
    if label == 1 and (not source_accept) and fusion_accept:
        return "rescued_positive"
    if label == 1 and source_accept and (not fusion_accept):
        return "lost_positive"
    if label == 0 and source_accept and (not fusion_accept):
        return "fixed_false_accept"
    if label == 0 and (not source_accept) and fusion_accept:
        return "new_false_accept"
    return "fusion_only_correct" if fusion_correct else "sourceafis_only_correct"


def _correctness_category(row: pd.Series) -> str:
    label = int(row["label"])
    source_correct = _is_correct(label, bool(row["sourceafis_accept_at_far_1pct"]))
    fusion_correct = _is_correct(label, bool(row["fusion_accept_at_far_1pct"]))
    if source_correct and fusion_correct:
        return "both_correct"
    if source_correct and not fusion_correct:
        return "sourceafis_only_correct"
    if fusion_correct and not source_correct:
        return "fusion_only_correct"
    return "both_wrong"


def _add_quality_pair_metrics(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for base in QUALITY_FEATURE_BASES:
        left = pd.to_numeric(out.get(f"a_{base}"), errors="coerce")
        right = pd.to_numeric(out.get(f"b_{base}"), errors="coerce")
        if f"pair_{base}_abs_delta" not in out.columns:
            out[f"pair_{base}_abs_delta"] = (left - right).abs()
    out["pair_min_contrast_proxy"] = pd.concat(
        [
            pd.to_numeric(out.get("a_contrast_proxy"), errors="coerce"),
            pd.to_numeric(out.get("b_contrast_proxy"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    out["pair_min_sharpness_laplacian_var"] = pd.concat(
        [
            pd.to_numeric(out.get("a_sharpness_laplacian_var"), errors="coerce"),
            pd.to_numeric(out.get("b_sharpness_laplacian_var"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    out["pair_min_foreground_ratio"] = pd.concat(
        [
            pd.to_numeric(out.get("a_foreground_ratio"), errors="coerce"),
            pd.to_numeric(out.get("b_foreground_ratio"), errors="coerce"),
        ],
        axis=1,
    ).min(axis=1, skipna=True)
    contrast = pd.to_numeric(out["pair_min_contrast_proxy"], errors="coerce")
    sharpness = np.log1p(pd.to_numeric(out["pair_min_sharpness_laplacian_var"], errors="coerce"))
    foreground = pd.to_numeric(out["pair_min_foreground_ratio"], errors="coerce")
    out["pair_quality_index"] = pd.concat(
        [
            contrast.rank(pct=True),
            pd.Series(sharpness, index=out.index).rank(pct=True),
            foreground.rank(pct=True),
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    return out


def add_thresholds_decisions_and_bands(
    table: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    target_far: float,
) -> pd.DataFrame:
    out = table.copy()
    lookup = {
        (str(row.dataset), str(row.method)): float(row.threshold)
        for row in thresholds.itertuples(index=False)
        if math.isclose(float(row.target_far), float(target_far), rel_tol=0.0, abs_tol=1e-12)
    }
    for method, score_column, threshold_column in SCORE_METHODS:
        out[threshold_column] = np.nan
        for dataset in sorted(out["dataset"].astype(str).unique()):
            key = (dataset, method)
            if key not in lookup:
                raise FusionFailureTaxonomyError(f"Missing threshold for dataset={dataset} method={method}")
            out.loc[out["dataset"].astype(str) == dataset, threshold_column] = float(lookup[key])
        accept_column = f"{method}_accept_at_far_1pct"
        decision_column = f"{method}_decision_at_far_1pct"
        margin_column = f"{method}_score_minus_threshold"
        out[accept_column] = pd.to_numeric(out[score_column], errors="coerce") >= pd.to_numeric(
            out[threshold_column],
            errors="coerce",
        )
        out[decision_column] = out[accept_column].map(_truth_decision)
        out[margin_column] = pd.to_numeric(out[score_column], errors="coerce") - pd.to_numeric(
            out[threshold_column],
            errors="coerce",
        )

    out["label_name"] = np.where(out["label"].astype(int) == 1, "positive", "negative")
    out["sourceafis_correct"] = [
        _is_correct(int(label), bool(accepted))
        for label, accepted in zip(out["label"], out["sourceafis_accept_at_far_1pct"])
    ]
    out["sift_plain_roll_v2_correct"] = [
        _is_correct(int(label), bool(accepted))
        for label, accepted in zip(out["label"], out["sift_plain_roll_v2_accept_at_far_1pct"])
    ]
    out["fusion_correct"] = [
        _is_correct(int(label), bool(accepted))
        for label, accepted in zip(out["label"], out["fusion_accept_at_far_1pct"])
    ]
    out["category"] = out.apply(_category, axis=1)
    out["correctness_category"] = out.apply(_correctness_category, axis=1)

    for dataset, group in out.groupby("dataset", sort=True):
        val = group[group["split"].astype(str).str.lower() == "val"].copy()
        val_negative = val[val["label"].astype(int) == 0].copy()
        for method, score_column, _threshold_column in SCORE_METHODS:
            percentile_column = f"{method}_score_percentile_val"
            negative_percentile_column = f"{method}_negative_score_percentile_val"
            out.loc[group.index, percentile_column] = _score_percentiles(group[score_column], val[score_column]).to_numpy()
            out.loc[group.index, negative_percentile_column] = _score_percentiles(
                group[score_column],
                val_negative[score_column],
            ).to_numpy()
            out.loc[group.index, f"{method}_score_band"] = out.loc[group.index, percentile_column].map(_percentile_band)
    out["sift_score_percentile"] = out["sift_plain_roll_v2_score_percentile_val"]
    out["sift_score_band"] = out["sift_plain_roll_v2_score_band"]

    for dataset, group in out.groupby("dataset", sort=True):
        val = group[group["split"].astype(str).str.lower() == "val"].copy()
        for _label, metric in QUALITY_BAND_METRICS.items():
            values = pd.to_numeric(val[metric], errors="coerce").dropna()
            if values.empty:
                low, high = float("nan"), float("nan")
            else:
                low, high = float(values.quantile(0.25)), float(values.quantile(0.75))
            high_is_low_quality = metric.endswith("_abs_delta")
            out.loc[group.index, f"{metric}_band"] = [
                _quantile_band(value, low, high, high_is_low_quality=high_is_low_quality)
                for value in out.loc[group.index, metric]
            ]
    low_quality_flags = []
    for row in out.itertuples(index=False):
        contrast = getattr(row, "pair_min_contrast_proxy_band", "missing")
        sharpness = getattr(row, "pair_min_sharpness_laplacian_var_band", "missing")
        foreground = getattr(row, "pair_min_foreground_ratio_band", "missing")
        if "missing" in {contrast, sharpness, foreground}:
            low_quality_flags.append("missing")
        elif "low" in {contrast, sharpness, foreground}:
            low_quality_flags.append("low")
        elif {contrast, sharpness, foreground} == {"high"}:
            low_quality_flags.append("high")
        else:
            low_quality_flags.append("mid")
    out["quality_band"] = low_quality_flags

    out["sift_high_sourceafis_low"] = (
        (pd.to_numeric(out["sift_plain_roll_v2_score_percentile_val"], errors="coerce") >= 0.90)
        & (~out["sourceafis_accept_at_far_1pct"].astype(bool))
    )
    out["sourceafis_high_fusion_suppressed_false_accept"] = (
        (out["label"].astype(int) == 0)
        & out["sourceafis_accept_at_far_1pct"].astype(bool)
        & (~out["fusion_accept_at_far_1pct"].astype(bool))
    )
    out["sourceafis_high_fusion_suppressed_false_accept_p90"] = (
        out["sourceafis_high_fusion_suppressed_false_accept"].astype(bool)
        & (pd.to_numeric(out["sourceafis_score_percentile_val"], errors="coerce") >= 0.90)
    )
    return out


def build_failure_taxonomy(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    fusion_dir: Path = DEFAULT_FUSION_DIR,
    sift_score_dir: Path = DEFAULT_SIFT_SCORE_DIR,
    target_far: float = DEFAULT_TARGET_FAR,
    include_quality: bool = True,
    repo_root: Path = REPO_ROOT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_pair_scores(
        datasets=datasets,
        splits=splits,
        fusion_dir=parse_file_uri(fusion_dir, repo_root=repo_root),
        sift_score_dir=parse_file_uri(sift_score_dir, repo_root=repo_root),
    )
    if include_quality:
        raw = add_quality_features(raw, repo_root=repo_root)
    else:
        for prefix in ("a", "b"):
            for feature in QUALITY_FEATURE_BASES:
                raw[f"{prefix}_{feature}"] = np.nan
    raw = _add_quality_pair_metrics(raw)
    thresholds = _threshold_rows(raw, target_far=float(target_far))
    taxonomy = add_thresholds_decisions_and_bands(raw, thresholds, target_far=float(target_far))
    taxonomy["target_far"] = float(target_far)
    taxonomy["target_far_label"] = "1%" if math.isclose(float(target_far), 0.01) else _fmt_pct(target_far)
    taxonomy["taxonomy_schema_version"] = SCHEMA_VERSION
    return taxonomy, thresholds


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame(columns=group_cols)
    for key, group in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        labels = group["label"].astype(int)
        source_accept = group["sourceafis_accept_at_far_1pct"].astype(bool)
        fusion_accept = group["fusion_accept_at_far_1pct"].astype(bool)
        row = {column: value for column, value in zip(group_cols, key)}
        row.update(
            {
                "n_pairs": int(len(group)),
                "n_positive": int((labels == 1).sum()),
                "n_negative": int((labels == 0).sum()),
                "sourceafis_accepts": int(source_accept.sum()),
                "fusion_accepts": int(fusion_accept.sum()),
                "sift_plain_roll_v2_accepts": int(group["sift_plain_roll_v2_accept_at_far_1pct"].astype(bool).sum()),
                "sourceafis_correct": int(group["sourceafis_correct"].astype(bool).sum()),
                "fusion_correct": int(group["fusion_correct"].astype(bool).sum()),
                "sift_plain_roll_v2_correct": int(group["sift_plain_roll_v2_correct"].astype(bool).sum()),
                "sourceafis_false_rejects": int(((labels == 1) & (~source_accept)).sum()),
                "fusion_false_rejects": int(((labels == 1) & (~fusion_accept)).sum()),
                "sourceafis_false_accepts": int(((labels == 0) & source_accept).sum()),
                "fusion_false_accepts": int(((labels == 0) & fusion_accept).sum()),
                "sift_high_sourceafis_low": int(group["sift_high_sourceafis_low"].astype(bool).sum()),
                "sourceafis_high_fusion_suppressed_false_accept": int(
                    group["sourceafis_high_fusion_suppressed_false_accept"].astype(bool).sum()
                ),
            }
        )
        category_counts = group["category"].value_counts()
        for category in CATEGORY_COLUMNS:
            row[f"category_{category}"] = int(category_counts.get(category, 0))
        correctness_counts = group["correctness_category"].value_counts()
        for category in CORRECTNESS_COLUMNS:
            row[f"correctness_{category}"] = int(correctness_counts.get(category, 0))
        row["fusion_accuracy"] = float(row["fusion_correct"] / row["n_pairs"]) if row["n_pairs"] else float("nan")
        row["sourceafis_accuracy"] = (
            float(row["sourceafis_correct"] / row["n_pairs"]) if row["n_pairs"] else float("nan")
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_dataset(taxonomy: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(taxonomy, ["split", "dataset", "label_name"])


def summarize_by_finger(taxonomy: pd.DataFrame) -> pd.DataFrame:
    return _aggregate(taxonomy, ["split", "dataset", "finger_position", "label_name"])


def summarize_by_score_band(taxonomy: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, _score_column, _threshold_column in SCORE_METHODS:
        band_column = f"{method}_score_band"
        frame = taxonomy.copy()
        frame["score_family"] = method
        frame["score_band"] = frame[band_column]
        frames.append(_aggregate(frame, ["split", "dataset", "score_family", "score_band", "label_name"]))

    for flag in ("sift_high_sourceafis_low", "sourceafis_high_fusion_suppressed_false_accept"):
        subset = taxonomy[taxonomy[flag].astype(bool)].copy()
        if subset.empty:
            continue
        subset["score_family"] = "diagnostic_slice"
        subset["score_band"] = flag
        frames.append(_aggregate(subset, ["split", "dataset", "score_family", "score_band", "label_name"]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def summarize_by_quality_band(taxonomy: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metric_label, metric in QUALITY_BAND_METRICS.items():
        frame = taxonomy.copy()
        frame["quality_metric"] = metric_label
        frame["quality_band"] = frame[f"{metric}_band"]
        frames.append(_aggregate(frame, ["split", "dataset", "quality_metric", "quality_band", "label_name"]))
    frame = taxonomy.copy()
    frame["quality_metric"] = "composite"
    frames.append(_aggregate(frame, ["split", "dataset", "quality_metric", "quality_band", "label_name"]))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _output_column_order(df: pd.DataFrame) -> list[str]:
    front = [
        "taxonomy_schema_version",
        "dataset",
        "split",
        "pair_id",
        "label",
        "label_name",
        "subject_a",
        "subject_b",
        "frgp",
        "finger_position",
        "path_a",
        "path_b",
        "sourceafis_score",
        "sift_plain_roll_v2_score",
        "fusion_score",
        "sourceafis_threshold_at_far_1pct",
        "sift_plain_roll_v2_threshold_at_far_1pct",
        "fusion_threshold_at_far_1pct",
        "sourceafis_decision_at_far_1pct",
        "sift_plain_roll_v2_decision_at_far_1pct",
        "fusion_decision_at_far_1pct",
        "sourceafis_score_minus_threshold",
        "sift_plain_roll_v2_score_minus_threshold",
        "fusion_score_minus_threshold",
        "sift_score_percentile",
        "sourceafis_score_percentile_val",
        "fusion_score_percentile_val",
        "sourceafis_score_band",
        "sift_score_band",
        "fusion_score_band",
        "category",
        "correctness_category",
        "sourceafis_correct",
        "sift_plain_roll_v2_correct",
        "fusion_correct",
        "sift_high_sourceafis_low",
        "sourceafis_high_fusion_suppressed_false_accept",
        "sourceafis_high_fusion_suppressed_false_accept_p90",
        "quality_band",
    ]
    quality = []
    for prefix in ("a", "b"):
        quality.extend(f"{prefix}_{name}" for name in QUALITY_FEATURE_BASES)
    quality.extend(
        [
            "pair_min_contrast_proxy",
            "pair_min_sharpness_laplacian_var",
            "pair_min_foreground_ratio",
            "pair_quality_index",
            "pair_contrast_proxy_abs_delta",
            "pair_sharpness_laplacian_var_abs_delta",
            "pair_min_contrast_proxy_band",
            "pair_min_sharpness_laplacian_var_band",
            "pair_min_foreground_ratio_band",
            "pair_quality_index_band",
        ]
    )
    ordered = [column for column in [*front, *quality] if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)
    return ordered


def _example_rows(taxonomy: pd.DataFrame, category: str) -> pd.DataFrame:
    rows = taxonomy[(taxonomy["split"].astype(str).str.lower() == "test") & (taxonomy["category"] == category)].copy()
    if rows.empty:
        return rows
    if category == "rescued_positive":
        rows = rows.sort_values(
            ["fusion_score_minus_threshold", "sourceafis_score_minus_threshold"],
            ascending=[False, True],
        )
    elif category == "lost_positive":
        rows = rows.sort_values(
            ["sourceafis_score_minus_threshold", "fusion_score_minus_threshold"],
            ascending=[False, True],
        )
    elif category == "fixed_false_accept":
        rows = rows.sort_values(
            ["sourceafis_score_minus_threshold", "fusion_score_minus_threshold"],
            ascending=[False, True],
        )
    elif category == "new_false_accept":
        rows = rows.sort_values(
            ["fusion_score_minus_threshold", "sourceafis_score_minus_threshold"],
            ascending=[False, True],
        )
    return rows


def _category_table_lines(summary: pd.DataFrame, *, split: str) -> list[str]:
    rows = summary[summary["split"].astype(str).str.lower() == split].copy()
    if rows.empty:
        return ["| none |  | 0 | 0 | 0 | 0 | 0 | 0 | 0 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    lines = [
        "| dataset | label | pairs | rescued | lost | fixed FA | new FA | both correct | both wrong |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in rows.sort_values(["dataset", "label_name"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['label_name']} | {int(row['n_pairs'])} | "
            f"{int(row['category_rescued_positive'])} | {int(row['category_lost_positive'])} | "
            f"{int(row['category_fixed_false_accept'])} | {int(row['category_new_false_accept'])} | "
            f"{int(row['category_both_correct'])} | {int(row['category_both_wrong'])} |"
        )
    return lines


def _test_dataset_headlines(taxonomy: pd.DataFrame) -> pd.DataFrame:
    test = taxonomy[taxonomy["split"].astype(str).str.lower() == "test"].copy()
    return _aggregate(test, ["dataset"])


def _top_group_counts(df: pd.DataFrame, count_column: str, *, limit: int = 4) -> str:
    if df.empty or count_column not in df.columns:
        return "none"
    top = df.sort_values(count_column, ascending=False).head(limit)
    parts = [
        f"{row.dataset}/finger {row.finger_position}: {int(getattr(row, count_column))}"
        for row in top.itertuples(index=False)
        if int(getattr(row, count_column)) > 0
    ]
    return ", ".join(parts) if parts else "none"


def render_summary(
    *,
    taxonomy: pd.DataFrame,
    thresholds: pd.DataFrame,
    by_dataset: pd.DataFrame,
    by_finger: pd.DataFrame,
    by_quality_band: pd.DataFrame,
    comparison_csv: Path | None,
    target_far: float,
) -> str:
    test_headline = _test_dataset_headlines(taxonomy)
    test = taxonomy[taxonomy["split"].astype(str).str.lower() == "test"].copy()
    val = taxonomy[taxonomy["split"].astype(str).str.lower() == "val"].copy()

    total_rescued = int((test["category"] == "rescued_positive").sum())
    total_lost = int((test["category"] == "lost_positive").sum())
    total_fixed = int((test["category"] == "fixed_false_accept").sum())
    total_new_fa = int((test["category"] == "new_false_accept").sum())
    fusion_fr = int(((test["label"].astype(int) == 1) & (~test["fusion_accept_at_far_1pct"].astype(bool))).sum())
    fusion_fa = int(((test["label"].astype(int) == 0) & test["fusion_accept_at_far_1pct"].astype(bool)).sum())
    both_wrong_positive = int(((test["label"].astype(int) == 1) & (test["category"] == "both_wrong")).sum())
    both_wrong_negative = int(((test["label"].astype(int) == 0) & (test["category"] == "both_wrong")).sum())
    sift_high_source_low = int(test["sift_high_sourceafis_low"].astype(bool).sum())
    source_high_suppressed = int(test["sourceafis_high_fusion_suppressed_false_accept"].astype(bool).sum())

    finger_positive = by_finger[
        (by_finger["split"].astype(str).str.lower() == "test") & (by_finger["label_name"] == "positive")
    ].copy()
    finger_negative = by_finger[
        (by_finger["split"].astype(str).str.lower() == "test") & (by_finger["label_name"] == "negative")
    ].copy()

    low_quality_errors = by_quality_band[
        (by_quality_band["split"].astype(str).str.lower() == "test")
        & (by_quality_band["quality_metric"] == "composite")
        & (by_quality_band["quality_band"] == "low")
    ].copy()
    low_quality_fusion_errors = int(
        low_quality_errors.get("fusion_false_rejects", pd.Series(dtype=int)).sum()
        + low_quality_errors.get("fusion_false_accepts", pd.Series(dtype=int)).sum()
    )

    lines = [
        "# Fusion Failure Taxonomy v1",
        "",
        f"Target operating point: `{_fmt_pct(target_far)}` FAR. Thresholds for SourceAFIS, SIFT Plain/Roll v2, and fusion are recalibrated inside this diagnostic from VAL negatives only, then applied unchanged to VAL and TEST rows.",
        "",
        f"Input comparison report: `{comparison_csv}`." if comparison_csv else "Input comparison report: not provided.",
        "",
        "## TEST Headline Counts",
        "",
        "| dataset | pairs | positives | negatives | SourceAFIS TA/FA | fusion TA/FA | rescued | lost | fixed FA | new FA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in test_headline.sort_values("dataset").iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['n_pairs'])} | {int(row['n_positive'])} | {int(row['n_negative'])} | "
            f"{int(row['sourceafis_accepts'] - row['sourceafis_false_accepts'])}/{int(row['sourceafis_false_accepts'])} | "
            f"{int(row['fusion_accepts'] - row['fusion_false_accepts'])}/{int(row['fusion_false_accepts'])} | "
            f"{int(row['category_rescued_positive'])} | {int(row['category_lost_positive'])} | "
            f"{int(row['category_fixed_false_accept'])} | {int(row['category_new_false_accept'])} |"
        )
    lines.extend(
        [
            "",
            "## TEST Category Detail",
            "",
            *_category_table_lines(by_dataset, split="test"),
            "",
            "## VAL Category Detail",
            "",
            *_category_table_lines(by_dataset, split="val"),
            "",
            "## Thresholds",
            "",
            "| dataset | method | threshold | VAL false accepts | VAL FAR |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in thresholds.sort_values(["dataset", "method"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {row['method']} | {_fmt_float(row['threshold'], 6)} | "
            f"{int(row['calibration_false_accepts'])} | {_fmt_pct(row['calibration_far'])} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Slices",
            "",
            f"- TEST SIFT-high / SourceAFIS-low pairs: {sift_high_source_low}.",
            f"- TEST SourceAFIS accepts but fusion suppresses negative pairs: {source_high_suppressed}.",
            f"- Worst TEST positive false-reject fingers after fusion: {_top_group_counts(finger_positive, 'fusion_false_rejects')}.",
            f"- Worst TEST negative false-accept fingers after fusion: {_top_group_counts(finger_negative, 'fusion_false_accepts')}.",
            f"- TEST fusion errors in composite low-quality bands: {low_quality_fusion_errors}.",
            "",
            "## Recommendations",
            "",
            f"- Remaining fusion failures are dominated by {fusion_fr} TEST false rejects and {fusion_fa} TEST false accepts. The hard core is {both_wrong_positive} positive pairs that both SourceAFIS and fusion reject, plus {both_wrong_negative} negative pairs that both accept.",
            f"- Good deep pairwise reranker candidates are the discordant and near-threshold cases: {total_rescued + total_lost} positive disagreements, {total_fixed + total_new_fa} negative disagreements, and especially SIFT-high / SourceAFIS-low positives where a learned pair model could use local ridge evidence instead of only score-level fusion.",
            "- Good calibration/reranking candidates are SourceAFIS-high false accepts that fusion already suppresses; those cases show the quality/SIFT features carry useful counter-evidence and could be made more reliable with a pairwise model.",
            f"- Likely data or quality limitations are the low-quality composite-band errors and the both-wrong residuals, where all available classical signals agree with the wrong decision. Those should be inspected visually before treating them as learnable model failures.",
        ]
    )
    if not val.empty:
        lines.append(
            f"- VAL remains separate in the CSVs and summary tables: {len(val)} VAL rows are reported separately from {len(test)} TEST rows."
        )
    return "\n".join(lines) + "\n"


def assert_counts_match_statistical_comparison(
    taxonomy: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    target_far: float = DEFAULT_TARGET_FAR,
) -> None:
    required = {
        "dataset",
        "split",
        "target_far",
        "rescued_positives",
        "lost_positives",
        "fixed_false_accepts",
        "new_false_accepts",
        "fusion_ta",
        "fusion_fa",
        "sourceafis_ta",
        "sourceafis_fa",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise AssertionError(f"Statistical comparison CSV missing columns: {sorted(missing)}")
    rows = comparison[
        (comparison["split"].astype(str).str.lower() == "test")
        & np.isclose(pd.to_numeric(comparison["target_far"], errors="coerce"), float(target_far))
    ].copy()
    if rows.empty:
        raise AssertionError(f"No TEST statistical comparison rows for target_far={target_far}")
    test = taxonomy[taxonomy["split"].astype(str).str.lower() == "test"].copy()
    failures: list[str] = []
    for row in rows.itertuples(index=False):
        dataset = str(row.dataset)
        group = test[test["dataset"].astype(str) == dataset].copy()
        labels = group["label"].astype(int)
        source_accept = group["sourceafis_accept_at_far_1pct"].astype(bool)
        fusion_accept = group["fusion_accept_at_far_1pct"].astype(bool)
        observed = {
            "rescued_positives": int((group["category"] == "rescued_positive").sum()),
            "lost_positives": int((group["category"] == "lost_positive").sum()),
            "fixed_false_accepts": int((group["category"] == "fixed_false_accept").sum()),
            "new_false_accepts": int((group["category"] == "new_false_accept").sum()),
            "fusion_ta": int(((labels == 1) & fusion_accept).sum()),
            "fusion_fa": int(((labels == 0) & fusion_accept).sum()),
            "sourceafis_ta": int(((labels == 1) & source_accept).sum()),
            "sourceafis_fa": int(((labels == 0) & source_accept).sum()),
        }
        for column, value in observed.items():
            expected = int(getattr(row, column))
            if int(value) != expected:
                failures.append(f"{dataset} {column}: observed={value} expected={expected}")
    if failures:
        raise AssertionError("Failure taxonomy counts do not match statistical comparison: " + "; ".join(failures))


def write_outputs(
    *,
    taxonomy: pd.DataFrame,
    thresholds: pd.DataFrame,
    outdir: Path,
    comparison_csv: Path | None,
    target_far: float,
    repo_root: Path,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    by_dataset = summarize_by_dataset(taxonomy)
    by_finger = summarize_by_finger(taxonomy)
    by_score_band = summarize_by_score_band(taxonomy)
    by_quality_band = summarize_by_quality_band(taxonomy)

    ordered_columns = _output_column_order(taxonomy)
    paths = {
        "failure_taxonomy_pairs": outdir / "failure_taxonomy_pairs.csv",
        "failure_taxonomy_summary": outdir / "failure_taxonomy_summary.md",
        "failure_taxonomy_by_dataset": outdir / "failure_taxonomy_by_dataset.csv",
        "failure_taxonomy_by_finger": outdir / "failure_taxonomy_by_finger.csv",
        "failure_taxonomy_by_score_band": outdir / "failure_taxonomy_by_score_band.csv",
        "failure_taxonomy_by_quality_band": outdir / "failure_taxonomy_by_quality_band.csv",
        "rescued_positive_examples": outdir / "rescued_positive_examples.csv",
        "lost_positive_examples": outdir / "lost_positive_examples.csv",
        "fixed_false_accept_examples": outdir / "fixed_false_accept_examples.csv",
        "new_false_accept_examples": outdir / "new_false_accept_examples.csv",
        "thresholds": outdir / "failure_taxonomy_thresholds.csv",
        "manifest": outdir / "run_manifest.json",
    }
    taxonomy[ordered_columns].to_csv(paths["failure_taxonomy_pairs"], index=False)
    by_dataset.to_csv(paths["failure_taxonomy_by_dataset"], index=False)
    by_finger.to_csv(paths["failure_taxonomy_by_finger"], index=False)
    by_score_band.to_csv(paths["failure_taxonomy_by_score_band"], index=False)
    by_quality_band.to_csv(paths["failure_taxonomy_by_quality_band"], index=False)
    thresholds.to_csv(paths["thresholds"], index=False)

    for category in ("rescued_positive", "lost_positive", "fixed_false_accept", "new_false_accept"):
        _example_rows(taxonomy, category)[ordered_columns].to_csv(paths[f"{category}_examples"], index=False)

    paths["failure_taxonomy_summary"].write_text(
        render_summary(
            taxonomy=taxonomy,
            thresholds=thresholds,
            by_dataset=by_dataset,
            by_finger=by_finger,
            by_quality_band=by_quality_band,
            comparison_csv=comparison_csv,
            target_far=float(target_far),
        ),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "outdir": str(outdir),
        "target_far": float(target_far),
        "threshold_protocol": "thresholds selected from VAL negatives only",
        "test_protocol": "TEST rows are reported as final taxonomy counts; VAL rows remain separate",
        "row_counts": {
            "all_pairs": int(len(taxonomy)),
            "val_pairs": int((taxonomy["split"].astype(str).str.lower() == "val").sum()),
            "test_pairs": int((taxonomy["split"].astype(str).str.lower() == "test").sum()),
            "rescued_positive_test": int(
                ((taxonomy["split"].astype(str).str.lower() == "test") & (taxonomy["category"] == "rescued_positive")).sum()
            ),
            "lost_positive_test": int(
                ((taxonomy["split"].astype(str).str.lower() == "test") & (taxonomy["category"] == "lost_positive")).sum()
            ),
            "fixed_false_accept_test": int(
                (
                    (taxonomy["split"].astype(str).str.lower() == "test")
                    & (taxonomy["category"] == "fixed_false_accept")
                ).sum()
            ),
            "new_false_accept_test": int(
                ((taxonomy["split"].astype(str).str.lower() == "test") & (taxonomy["category"] == "new_false_accept")).sum()
            ),
        },
        "git": _git_info(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return paths


def run_taxonomy(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    fusion_dir: Path = DEFAULT_FUSION_DIR,
    sift_score_dir: Path = DEFAULT_SIFT_SCORE_DIR,
    outdir: Path = DEFAULT_OUTDIR,
    comparison_csv: Path | None = DEFAULT_COMPARISON_CSV,
    target_far: float = DEFAULT_TARGET_FAR,
    include_quality: bool = True,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    taxonomy, thresholds = build_failure_taxonomy(
        datasets=datasets,
        splits=splits,
        fusion_dir=parse_file_uri(fusion_dir, repo_root=repo_root),
        sift_score_dir=parse_file_uri(sift_score_dir, repo_root=repo_root),
        target_far=float(target_far),
        include_quality=bool(include_quality),
        repo_root=repo_root,
    )
    comparison_path = parse_file_uri(comparison_csv, repo_root=repo_root) if comparison_csv is not None else None
    if comparison_path is not None and comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        assert_counts_match_statistical_comparison(taxonomy, comparison, target_far=float(target_far))
    paths = write_outputs(
        taxonomy=taxonomy,
        thresholds=thresholds,
        outdir=parse_file_uri(outdir, repo_root=repo_root),
        comparison_csv=comparison_path if comparison_path is not None and comparison_path.exists() else None,
        target_far=float(target_far),
        repo_root=repo_root,
    )
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Phase 2A failure taxonomy for sourceafis_sift_quality_fusion_v1."
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--fusion_dir", default=str(DEFAULT_FUSION_DIR))
    parser.add_argument("--sift_score_dir", default=str(DEFAULT_SIFT_SCORE_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--comparison_csv", default=str(DEFAULT_COMPARISON_CSV))
    parser.add_argument("--target_far", type=float, default=DEFAULT_TARGET_FAR)
    parser.add_argument("--skip_quality", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_taxonomy(
            datasets=_parse_csv_arg(args.datasets),
            splits=tuple(split.lower() for split in _parse_csv_arg(args.splits)),
            fusion_dir=parse_file_uri(args.fusion_dir),
            sift_score_dir=parse_file_uri(args.sift_score_dir),
            outdir=parse_file_uri(args.outdir),
            comparison_csv=parse_file_uri(args.comparison_csv) if str(args.comparison_csv).strip() else None,
            target_far=float(args.target_far),
            include_quality=not bool(args.skip_quality),
            repo_root=REPO_ROOT,
        )
    except (FusionFailureTaxonomyError, AssertionError) as exc:
        print(f"Fusion failure taxonomy failed: {exc}", file=sys.stderr)
        return 2

    print("Wrote fusion failure taxonomy artifacts:")
    for name in REQUIRED_OUTPUT_NAMES:
        print(f"  {paths[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
