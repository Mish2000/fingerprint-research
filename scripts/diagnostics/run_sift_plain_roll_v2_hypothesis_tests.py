from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.matchers.matching_baseline import (  # noqa: E402
    SIFTConfig,
    match_sift,
    ransac_inliers_for_model,
    score_sift_plain_roll_v2_counts,
    sift_extract,
)
from src.fpbench.preprocess.preprocess import (  # noqa: E402
    PreprocessConfig,
    extract_fingerprint_roi,
    preprocess_image,
    resize_pad_to_square,
)


DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "sift_plain_roll_v2_external_validation"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "sift_plain_roll_v2_hypothesis_tests"
)
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)
PAIR_KEYS = ("dataset", "split", "label", "path_a", "path_b")
TARGET_SIZE = 768
NFEATURES = 3000
BLUR_KSIZE = 0
LOWE_RATIO = 0.75
RANSAC_THRESH = 3.0
V2_GEOMETRY_MODEL = "affine_full_2d"
RESEARCH_PREFIX = "research_only::"
FRGP_FOCUS = (5, 10)
SOURCE_SCORE_COLUMNS = ("canonical_current_score", "sift_inliers_score", "v2_official_score")
CROP_PADDING_FRACTIONS = (0.15,)
ROLL_CROP_GRID = (
    ("left_60", 0.00, 0.60),
    ("center_60", 0.20, 0.80),
    ("right_60", 0.40, 1.00),
)


@dataclass(frozen=True)
class CandidateSpec:
    candidate_name: str
    candidate_family: str
    probe_kind: str
    parameters: dict[str, Any]
    research_only: bool = True


@dataclass(frozen=True)
class FingerprintName:
    subject: str
    capture: str
    ppi: int | None
    frgp: int | None


@dataclass(frozen=True)
class Snapshot:
    keypoints: tuple[Any, ...]
    descriptors: np.ndarray | None
    meta: dict[str, Any]


def _candidate_name(family: str, variant: str) -> str:
    return f"{RESEARCH_PREFIX}{family}:{variant}"


CROP_CANDIDATES = tuple(
    CandidateSpec(
        candidate_name=_candidate_name("crop_overlap_probe_v1", f"pad{int(round(pad * 100)):02d}"),
        candidate_family="crop_overlap_probe_v1",
        probe_kind="image_scalar",
        parameters={
            "padding_fraction": float(pad),
            "padding_policy": "round(padding_fraction * max(foreground_bbox_width, foreground_bbox_height)); clipped",
            "preprocess": "v2 preprocess first, ROI bbox crop, resize_pad_to_square back to target_size",
            "geometry_model": V2_GEOMETRY_MODEL,
        },
    )
    for pad in CROP_PADDING_FRACTIONS
)
ROLL_MULTICROP_CANDIDATE = CandidateSpec(
    candidate_name=_candidate_name("roll_multicrop_overlap_probe_v1", "grid3_max"),
    candidate_family="roll_multicrop_overlap_probe_v1",
    probe_kind="image_scalar",
    parameters={
        "roll_crop_grid": ROLL_CROP_GRID,
        "plain_query": "standard v2 preprocessed plain image",
        "decision_rule": "max score over fixed roll ROI crop grid",
        "geometry_model": V2_GEOMETRY_MODEL,
    },
)
GEOMETRY_CANDIDATES = (
    CandidateSpec(
        candidate_name=_candidate_name("geometry_probe_v1", "affine_full_2d"),
        candidate_family="geometry_probe_v1",
        probe_kind="image_scalar",
        parameters={"geometry_model": "affine_full_2d", "preprocess": "standard v2"},
    ),
    CandidateSpec(
        candidate_name=_candidate_name("geometry_probe_v1", "affine_partial_2d"),
        candidate_family="geometry_probe_v1",
        probe_kind="image_scalar",
        parameters={"geometry_model": "affine_partial_2d", "preprocess": "standard v2"},
    ),
    CandidateSpec(
        candidate_name=_candidate_name("geometry_probe_v1", "homography"),
        candidate_family="geometry_probe_v1",
        probe_kind="image_scalar",
        parameters={"geometry_model": "homography", "preprocess": "standard v2"},
    ),
    CandidateSpec(
        candidate_name=_candidate_name("geometry_probe_v1", "best_affine_partial_or_full"),
        candidate_family="geometry_probe_v1",
        probe_kind="image_scalar",
        parameters={
            "geometry_model": "best_score_among_affine_partial_2d_and_affine_full_2d",
            "preprocess": "standard v2",
        },
    ),
)
FUSION_CANDIDATES = (
    CandidateSpec(
        candidate_name=_candidate_name("fusion_probe_v1", "max_norm_mean"),
        candidate_family="fusion_probe_v1",
        probe_kind="fusion_scalar",
        parameters={
            "source_columns": SOURCE_SCORE_COLUMNS,
            "normalization": "divide each source score by its dataset VAL max",
            "fusion": "mean of normalized source scores",
        },
    ),
    CandidateSpec(
        candidate_name=_candidate_name("fusion_probe_v1", "max_norm_max"),
        candidate_family="fusion_probe_v1",
        probe_kind="fusion_scalar",
        parameters={
            "source_columns": SOURCE_SCORE_COLUMNS,
            "normalization": "divide each source score by its dataset VAL max",
            "fusion": "max of normalized source scores",
        },
    ),
    CandidateSpec(
        candidate_name=_candidate_name("fusion_probe_v1", "conservative_or"),
        candidate_family="fusion_probe_v1",
        probe_kind="fusion_target_rule",
        parameters={
            "source_columns": SOURCE_SCORE_COLUMNS,
            "rule": "OR of source-specific VAL thresholds calibrated at target_far / number_of_sources",
            "score": "max(source_score / source_threshold)",
            "threshold": 1.0,
        },
    ),
)


def candidate_registry() -> tuple[CandidateSpec, ...]:
    return (*CROP_CANDIDATES, ROLL_MULTICROP_CANDIDATE, *GEOMETRY_CANDIDATES, *FUSION_CANDIDATES)


def assert_research_only_candidate_names(specs: tuple[CandidateSpec, ...] | list[CandidateSpec]) -> None:
    unsafe = [
        spec.candidate_name
        for spec in specs
        if not spec.research_only
        or not str(spec.candidate_name).startswith(RESEARCH_PREFIX)
        or spec.candidate_name in {"sift", "sift_plain_roll_v2"}
    ]
    if unsafe:
        raise AssertionError(f"All hypothesis-test candidates must be research-only/script-local: {unsafe}")


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def parse_fingerprint_filename(raw_path: str | Path) -> FingerprintName:
    name = re.split(r"[\\/]", str(raw_path).strip())[-1]
    stem = Path(name).stem
    match = re.match(
        r"^(?P<subject>\d+)_(?P<capture>plain|roll|rolled)_(?P<ppi>\d+)_(?P<frgp>\d+)$",
        stem,
        flags=re.IGNORECASE,
    )
    if not match:
        return FingerprintName(subject="", capture="", ppi=None, frgp=None)
    capture = match.group("capture").lower()
    if capture == "rolled":
        capture = "roll"
    return FingerprintName(
        subject=match.group("subject"),
        capture=capture,
        ppi=int(match.group("ppi")),
        frgp=int(match.group("frgp")),
    )


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _score_csv(input_dir: Path, dataset: str, method: str, split: str) -> Path:
    path = input_dir / "scores" / dataset / f"scores_{dataset}_{method}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing source score CSV: {path}")
    return path


def _load_method_scores(input_dir: Path, dataset: str, method: str, split: str) -> pd.DataFrame:
    path = _score_csv(input_dir, dataset, method, split)
    df = pd.read_csv(path)
    missing = {"label", "split", "path_a", "path_b", "score", "inliers", "matches", "k1", "k2"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["dataset"] = dataset
    df["split"] = df["split"].astype(str).str.lower()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["source_scores_csv"] = str(path)
    return df


def _assert_unique_pairs(df: pd.DataFrame, label: str) -> None:
    duplicates = df.duplicated(list(PAIR_KEYS), keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, list(PAIR_KEYS)].head(5).to_dict(orient="records")
        raise ValueError(f"{label} has duplicate pair keys, sample={sample}")


def load_aligned_source_scores(
    input_dir: str | Path,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> pd.DataFrame:
    input_path = parse_file_uri(input_dir)
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for split in splits:
            sift = _load_method_scores(input_path, dataset, "sift", split)
            v2 = _load_method_scores(input_path, dataset, "sift_plain_roll_v2", split)
            _assert_unique_pairs(sift, f"{dataset}/{split} canonical SIFT")
            _assert_unique_pairs(v2, f"{dataset}/{split} SIFT Plain/Roll v2")
            canonical = sift.rename(
                columns={
                    "score": "canonical_current_score",
                    "inliers": "canonical_inliers",
                    "matches": "canonical_matches",
                    "k1": "canonical_k1",
                    "k2": "canonical_k2",
                    "source_scores_csv": "canonical_source_scores_csv",
                }
            )
            canonical["sift_inliers_score"] = canonical["canonical_inliers"]
            canonical = canonical[
                [
                    *PAIR_KEYS,
                    "canonical_current_score",
                    "sift_inliers_score",
                    "canonical_inliers",
                    "canonical_matches",
                    "canonical_k1",
                    "canonical_k2",
                    "canonical_source_scores_csv",
                ]
            ].copy()
            v2_scores = v2.rename(
                columns={
                    "score": "v2_official_score",
                    "inliers": "v2_inliers",
                    "matches": "v2_matches",
                    "k1": "v2_k1",
                    "k2": "v2_k2",
                    "source_scores_csv": "v2_source_scores_csv",
                }
            )
            v2_scores = v2_scores[
                [
                    *PAIR_KEYS,
                    "v2_official_score",
                    "v2_inliers",
                    "v2_matches",
                    "v2_k1",
                    "v2_k2",
                    "v2_source_scores_csv",
                ]
            ].copy()
            aligned = canonical.merge(v2_scores, on=list(PAIR_KEYS), how="inner", validate="one_to_one")
            if len(aligned) != len(canonical) or len(aligned) != len(v2_scores):
                raise ValueError(
                    f"{dataset}/{split} source score alignment mismatch: "
                    f"canonical={len(canonical)} v2={len(v2_scores)} aligned={len(aligned)}"
                )
            frames.append(aligned)
    out = pd.concat(frames, ignore_index=True, sort=False)
    meta_a = out["path_a"].map(parse_fingerprint_filename)
    meta_b = out["path_b"].map(parse_fingerprint_filename)
    out["subject_a"] = [item.subject for item in meta_a]
    out["subject_b"] = [item.subject for item in meta_b]
    out["capture_a"] = [item.capture for item in meta_a]
    out["capture_b"] = [item.capture for item in meta_b]
    out["ppi_a"] = [item.ppi for item in meta_a]
    out["ppi_b"] = [item.ppi for item in meta_b]
    out["frgp_a"] = [item.frgp for item in meta_a]
    out["frgp_b"] = [item.frgp for item in meta_b]
    out["frgp"] = out["frgp_a"].where(out["frgp_a"].notna(), out["frgp_b"])
    for column in (
        "canonical_current_score",
        "sift_inliers_score",
        "v2_official_score",
        "v2_inliers",
        "v2_matches",
    ):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    return out.reset_index(drop=True)


def threshold_for_far(negative_scores: np.ndarray | pd.Series | list[float], target_far: float) -> tuple[float, int, float]:
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
    return auc, float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def calibrate_reference_v2_thresholds(
    source_scores: pd.DataFrame,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset, group in source_scores.groupby("dataset", sort=True):
        val = group[group["split"].astype(str).str.lower() == "val"]
        val_labels = pd.to_numeric(val["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        neg_scores = _safe_numeric(val["v2_official_score"])[val_labels == 0]
        for target_far in target_fars:
            threshold, calibration_fa, calibration_far = threshold_for_far(neg_scores, float(target_far))
            rows.append(
                {
                    "dataset": dataset,
                    "target_far": float(target_far),
                    "v2_threshold": float(threshold),
                    "calibration_split": "val",
                    "calibration_false_accepts": int(calibration_fa),
                    "calibration_far": float(calibration_far),
                }
            )
    return pd.DataFrame(rows)


def calibrate_scalar_candidate_thresholds(
    candidate_scores: pd.DataFrame,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    if candidate_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (dataset, candidate_name), group in candidate_scores.groupby(["dataset", "candidate_name"], sort=True):
        spec_family = str(group["candidate_family"].iloc[0])
        probe_kind = str(group.get("probe_kind", pd.Series([""])).iloc[0])
        val = group[group["split"].astype(str).str.lower() == "val"]
        val_labels = pd.to_numeric(val["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        val_neg_scores = _safe_numeric(val["score"])[val_labels == 0]
        for target_far in target_fars:
            threshold, calibration_fa, calibration_far = threshold_for_far(val_neg_scores, float(target_far))
            selected = bool(math.isfinite(calibration_far) and calibration_far <= float(target_far) + 1e-12)
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_family": spec_family,
                    "candidate_name": candidate_name,
                    "probe_kind": probe_kind,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "threshold_kind": "scalar_score",
                    "calibration_split": "val",
                    "calibration_negative_count": int(np.sum(val_labels == 0)),
                    "calibration_false_accepts": int(calibration_fa),
                    "calibration_far": float(calibration_far),
                    "source_thresholds_json": "",
                    "fit_params_json": "",
                    "selected_by_val": selected,
                    "val_safety_status": "selected_val_far_within_target" if selected else "rejected_missing_or_unsafe_val_far",
                    "research_only": True,
                }
            )
    return pd.DataFrame(rows)


def _fit_fusion_normalizers(source_scores: pd.DataFrame) -> dict[str, dict[str, float]]:
    normalizers: dict[str, dict[str, float]] = {}
    for dataset, group in source_scores.groupby("dataset", sort=True):
        val = group[group["split"].astype(str).str.lower() == "val"]
        dataset_norms: dict[str, float] = {}
        for column in SOURCE_SCORE_COLUMNS:
            values = pd.to_numeric(val[column], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            max_value = float(np.max(values)) if values.size else 0.0
            dataset_norms[column] = max_value if max_value > 0.0 else 1.0
        normalizers[str(dataset)] = dataset_norms
    return normalizers


def build_fusion_probe_scores(source_scores: pd.DataFrame) -> pd.DataFrame:
    normalizers = _fit_fusion_normalizers(source_scores)
    rows: list[dict[str, Any]] = []
    spec_by_name = {spec.candidate_name: spec for spec in FUSION_CANDIDATES}
    scalar_names = [
        _candidate_name("fusion_probe_v1", "max_norm_mean"),
        _candidate_name("fusion_probe_v1", "max_norm_max"),
    ]
    for _, row in source_scores.iterrows():
        dataset = str(row["dataset"])
        norms = normalizers[dataset]
        values = np.asarray([_to_float(row[column], 0.0) / norms[column] for column in SOURCE_SCORE_COLUMNS], dtype=float)
        fused = {
            scalar_names[0]: float(np.mean(values)),
            scalar_names[1]: float(np.max(values)),
        }
        for candidate_name, score in fused.items():
            spec = spec_by_name[candidate_name]
            rows.append(
                {
                    "dataset": dataset,
                    "split": str(row["split"]).lower(),
                    "label": int(row["label"]),
                    "path_a": str(row["path_a"]),
                    "path_b": str(row["path_b"]),
                    "frgp": "" if pd.isna(row.get("frgp")) else int(row["frgp"]),
                    "candidate_name": candidate_name,
                    "candidate_family": spec.candidate_family,
                    "probe_kind": spec.probe_kind,
                    "score": score,
                    "matches": "",
                    "inliers": "",
                    "k1": "",
                    "k2": "",
                    "diagnostic_json": "",
                    "research_only": True,
                }
            )
    return pd.DataFrame(rows)


def _source_thresholds_for_or(val_negatives: pd.DataFrame, source_budget_far: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for column in SOURCE_SCORE_COLUMNS:
        threshold, _, _ = threshold_for_far(_safe_numeric(val_negatives[column]), source_budget_far)
        thresholds[column] = float(threshold)
    return thresholds


def _or_rule_scores(frame: pd.DataFrame, source_thresholds: dict[str, float]) -> np.ndarray:
    pieces: list[np.ndarray] = []
    for column in SOURCE_SCORE_COLUMNS:
        threshold = float(source_thresholds.get(column, float("nan")))
        values = _safe_numeric(frame[column])
        if math.isfinite(threshold) and threshold > 0.0:
            pieces.append(values / threshold)
        else:
            pieces.append(np.zeros_like(values, dtype=float))
    return np.max(np.vstack(pieces), axis=0) if pieces else np.zeros(len(frame), dtype=float)


def calibrate_conservative_or_thresholds(
    source_scores: pd.DataFrame,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    candidate_name = _candidate_name("fusion_probe_v1", "conservative_or")
    rows: list[dict[str, Any]] = []
    for dataset, group in source_scores.groupby("dataset", sort=True):
        val = group[group["split"].astype(str).str.lower() == "val"].copy()
        val_labels = pd.to_numeric(val["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        val_negatives = val[val_labels == 0].copy()
        for target_far in target_fars:
            source_budget = float(target_far) / float(len(SOURCE_SCORE_COLUMNS))
            source_thresholds = _source_thresholds_for_or(val_negatives, source_budget)
            val_scores = _or_rule_scores(val, source_thresholds)
            counts = _confusion(val_labels, val_scores, 1.0)
            selected = bool(math.isfinite(counts["far"]) and counts["far"] <= float(target_far) + 1e-12)
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_family": "fusion_probe_v1",
                    "candidate_name": candidate_name,
                    "probe_kind": "fusion_target_rule",
                    "target_far": float(target_far),
                    "threshold": 1.0,
                    "threshold_kind": "conservative_or_target_rule",
                    "calibration_split": "val",
                    "calibration_negative_count": int(np.sum(val_labels == 0)),
                    "calibration_false_accepts": int(counts["fa"]),
                    "calibration_far": float(counts["far"]),
                    "source_thresholds_json": json.dumps(source_thresholds, sort_keys=True),
                    "fit_params_json": json.dumps({"source_budget_far": source_budget}, sort_keys=True),
                    "selected_by_val": selected,
                    "val_safety_status": (
                        "selected_val_far_within_target"
                        if selected
                        else "rejected_val_far_over_target_after_or_union"
                    ),
                    "research_only": True,
                }
            )
    return pd.DataFrame(rows)


def select_candidates_from_val(candidate_metrics_val: pd.DataFrame, *, target_far: float = 0.01) -> pd.DataFrame:
    """Return one VAL-selected candidate per family without consulting TEST rows."""
    if candidate_metrics_val.empty:
        return pd.DataFrame()
    val = candidate_metrics_val[
        (candidate_metrics_val["split"].astype(str).str.lower() == "val")
        & np.isclose(pd.to_numeric(candidate_metrics_val["target_far"], errors="coerce"), float(target_far))
        & (candidate_metrics_val["selected_by_val"].astype(bool))
    ].copy()
    if val.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for (_dataset, family), group in val.groupby(["dataset", "candidate_family"], sort=True):
        ranked = group.sort_values(["tar", "far", "candidate_name"], ascending=[False, True, True])
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


@lru_cache(maxsize=8192)
def _load_gray(path_str: str) -> np.ndarray:
    path = parse_file_uri(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _sift_snapshot_from_processed(processed: np.ndarray, meta: dict[str, Any]) -> Snapshot:
    keypoints, descriptors = sift_extract(processed, None, SIFTConfig(nfeatures=NFEATURES))
    return Snapshot(tuple(keypoints or []), descriptors, dict(meta))


@lru_cache(maxsize=8192)
def _standard_snapshot(path_str: str) -> Snapshot:
    gray = _load_gray(path_str)
    processed = preprocess_image(gray, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))
    return _sift_snapshot_from_processed(
        processed,
        {
            "preprocess": "standard_v2",
            "image_width": int(processed.shape[1]),
            "image_height": int(processed.shape[0]),
        },
    )


def _foreground_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    foreground = mask > 0
    if not bool(np.any(foreground)):
        return None
    ys, xs = np.where(foreground)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _clip_bbox(x0: int, y0: int, x1: int, y1: int, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(int(x0), int(width) - 1))
    y0 = max(0, min(int(y0), int(height) - 1))
    x1 = max(x0 + 1, min(int(x1), int(width)))
    y1 = max(y0 + 1, min(int(y1), int(height)))
    return x0, y0, x1, y1


def _resize_crop_for_sift(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
    resized = resize_pad_to_square(crop, TARGET_SIZE)
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return resized.astype(np.uint8)


@lru_cache(maxsize=8192)
def _crop_overlap_snapshot(path_str: str, padding_fraction: float) -> Snapshot:
    gray = _load_gray(path_str)
    processed = preprocess_image(gray, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))
    roi = extract_fingerprint_roi(processed)
    bbox = _foreground_bbox(roi.mask) if roi.is_valid else None
    h, w = processed.shape[:2]
    fallback = bbox is None
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = bbox
    bw = x1 - x0
    bh = y1 - y0
    pad_px = int(round(float(padding_fraction) * float(max(bw, bh))))
    x0p, y0p, x1p, y1p = _clip_bbox(x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px, w, h)
    crop = processed[y0p:y1p, x0p:x1p]
    resized = _resize_crop_for_sift(crop)
    return _sift_snapshot_from_processed(
        resized,
        {
            "preprocess": "crop_overlap_probe_v1",
            "padding_fraction": float(padding_fraction),
            "padding_px": int(pad_px),
            "roi_valid": bool(roi.is_valid),
            "roi_failure_reason": roi.failure_reason or "",
            "fallback_full_image": bool(fallback),
            "bbox": [int(x0), int(y0), int(x1), int(y1)],
            "padded_bbox": [int(x0p), int(y0p), int(x1p), int(y1p)],
            "foreground_fraction": float(roi.foreground_fraction),
        },
    )


@lru_cache(maxsize=8192)
def _roll_crop_snapshot(path_str: str, crop_index: int) -> Snapshot:
    label, x_frac0, x_frac1 = ROLL_CROP_GRID[int(crop_index)]
    gray = _load_gray(path_str)
    processed = preprocess_image(gray, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))
    roi = extract_fingerprint_roi(processed)
    bbox = _foreground_bbox(roi.mask) if roi.is_valid else None
    h, w = processed.shape[:2]
    fallback = bbox is None
    if bbox is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        x0, y0, x1, y1 = bbox
    bw = max(x1 - x0, 1)
    win_x0 = x0 + int(round(float(x_frac0) * bw))
    win_x1 = x0 + int(round(float(x_frac1) * bw))
    win_x0, y0c, win_x1, y1c = _clip_bbox(win_x0, y0, win_x1, y1, w, h)
    crop = processed[y0c:y1c, win_x0:win_x1]
    resized = _resize_crop_for_sift(crop)
    return _sift_snapshot_from_processed(
        resized,
        {
            "preprocess": "roll_multicrop_overlap_probe_v1",
            "crop_index": int(crop_index),
            "crop_label": label,
            "roi_valid": bool(roi.is_valid),
            "roi_failure_reason": roi.failure_reason or "",
            "fallback_full_image": bool(fallback),
            "roi_bbox": [int(x0), int(y0), int(x1), int(y1)],
            "crop_bbox": [int(win_x0), int(y0c), int(win_x1), int(y1c)],
            "crop_x_fraction": [float(x_frac0), float(x_frac1)],
            "foreground_fraction": float(roi.foreground_fraction),
        },
    )


def _score_snapshots(snapshot_a: Snapshot, snapshot_b: Snapshot, *, ransac_model: str) -> dict[str, Any]:
    kps_a = list(snapshot_a.keypoints)
    kps_b = list(snapshot_b.keypoints)
    desc_a = snapshot_a.descriptors
    desc_b = snapshot_b.descriptors
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return {
            "score": 0.0,
            "matches": 0,
            "inliers": 0,
            "k1": int(len(kps_a)),
            "k2": int(len(kps_b)),
            "diagnostic": {"reason": "missing_descriptors", "model": ransac_model},
        }
    good = match_sift(desc_a, desc_b, ratio=LOWE_RATIO)
    inliers, _mask = ransac_inliers_for_model(
        kps_a,
        kps_b,
        good,
        ransac_model=ransac_model,
        ransac_thresh=RANSAC_THRESH,
    )
    return {
        "score": score_sift_plain_roll_v2_counts(matches=len(good), inliers=inliers),
        "matches": int(len(good)),
        "inliers": int(inliers),
        "k1": int(len(kps_a)),
        "k2": int(len(kps_b)),
        "diagnostic": {"model": ransac_model},
    }


def _base_score_row(
    row: pd.Series,
    spec: CandidateSpec,
    result: dict[str, Any],
    *,
    diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "dataset": str(row["dataset"]),
        "split": str(row["split"]).lower(),
        "label": int(row["label"]),
        "path_a": str(row["path_a"]),
        "path_b": str(row["path_b"]),
        "frgp": "" if pd.isna(row.get("frgp")) else int(row["frgp"]),
        "candidate_name": spec.candidate_name,
        "candidate_family": spec.candidate_family,
        "probe_kind": spec.probe_kind,
        "score": float(result["score"]),
        "matches": int(result["matches"]),
        "inliers": int(result["inliers"]),
        "k1": int(result["k1"]),
        "k2": int(result["k2"]),
        "diagnostic_json": json.dumps(diagnostic or result.get("diagnostic", {}), sort_keys=True),
        "research_only": True,
    }
    return payload


def _compute_geometry_scores(group: pd.DataFrame) -> pd.DataFrame:
    spec_by_name = {spec.candidate_name: spec for spec in GEOMETRY_CANDIDATES}
    rows: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        snap_a = _standard_snapshot(str(row["path_a"]))
        snap_b = _standard_snapshot(str(row["path_b"]))
        model_results: dict[str, dict[str, Any]] = {}
        for model in ("affine_full_2d", "affine_partial_2d", "homography"):
            model_results[model] = _score_snapshots(snap_a, snap_b, ransac_model=model)
            spec = spec_by_name[_candidate_name("geometry_probe_v1", model)]
            rows.append(_base_score_row(row, spec, model_results[model], diagnostic={"geometry_model": model}))
        full = model_results["affine_full_2d"]
        partial = model_results["affine_partial_2d"]
        winner_model = "affine_partial_2d" if float(partial["score"]) > float(full["score"]) else "affine_full_2d"
        winner = partial if winner_model == "affine_partial_2d" else full
        spec = spec_by_name[_candidate_name("geometry_probe_v1", "best_affine_partial_or_full")]
        rows.append(
            _base_score_row(
                row,
                spec,
                winner,
                diagnostic={
                    "geometry_model": "best_affine_partial_or_full",
                    "winning_geometry_model": winner_model,
                    "affine_full_2d_score": float(full["score"]),
                    "affine_partial_2d_score": float(partial["score"]),
                },
            )
        )
    return pd.DataFrame(rows)


def _compute_crop_scores(group: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        for spec in CROP_CANDIDATES:
            pad = float(spec.parameters["padding_fraction"])
            snap_a = _crop_overlap_snapshot(str(row["path_a"]), pad)
            snap_b = _crop_overlap_snapshot(str(row["path_b"]), pad)
            result = _score_snapshots(snap_a, snap_b, ransac_model=V2_GEOMETRY_MODEL)
            diagnostic = {
                "geometry_model": V2_GEOMETRY_MODEL,
                "plain_crop": snap_a.meta,
                "roll_crop": snap_b.meta,
            }
            rows.append(_base_score_row(row, spec, result, diagnostic=diagnostic))
    return pd.DataFrame(rows)


def _compute_roll_multicrop_scores(group: pd.DataFrame) -> pd.DataFrame:
    spec = ROLL_MULTICROP_CANDIDATE
    rows: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        snap_a = _standard_snapshot(str(row["path_a"]))
        best: dict[str, Any] | None = None
        best_crop_meta: dict[str, Any] = {}
        per_crop: list[dict[str, Any]] = []
        for crop_index in range(len(ROLL_CROP_GRID)):
            snap_b = _roll_crop_snapshot(str(row["path_b"]), crop_index)
            result = _score_snapshots(snap_a, snap_b, ransac_model=V2_GEOMETRY_MODEL)
            crop_record = {
                "crop_index": int(crop_index),
                "crop_label": str(snap_b.meta.get("crop_label", "")),
                "score": float(result["score"]),
                "matches": int(result["matches"]),
                "inliers": int(result["inliers"]),
                "crop_bbox": snap_b.meta.get("crop_bbox", []),
            }
            per_crop.append(crop_record)
            if best is None or float(result["score"]) > float(best["score"]):
                best = result
                best_crop_meta = dict(snap_b.meta)
        if best is None:
            best = {"score": 0.0, "matches": 0, "inliers": 0, "k1": int(len(snap_a.keypoints)), "k2": 0}
        diagnostic = {
            "geometry_model": V2_GEOMETRY_MODEL,
            "winning_crop_index": int(best_crop_meta.get("crop_index", -1)),
            "winning_crop_label": str(best_crop_meta.get("crop_label", "")),
            "winning_crop_bbox": best_crop_meta.get("crop_bbox", []),
            "roll_roi_bbox": best_crop_meta.get("roi_bbox", []),
            "per_crop_scores": per_crop,
        }
        rows.append(_base_score_row(row, spec, best, diagnostic=diagnostic))
    return pd.DataFrame(rows)


def _cache_path(outdir: Path, family: str, dataset: str, split: str) -> Path:
    return outdir / "candidate_score_cache" / f"{family}_{dataset}_{split}.csv"


def _compute_family_scores(
    source_scores: pd.DataFrame,
    *,
    family: str,
    outdir: Path,
    reuse_existing: bool,
    limit_per_split: int,
    log_handle: Any,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    family_fn = {
        "geometry_probe_v1": _compute_geometry_scores,
        "crop_overlap_probe_v1": _compute_crop_scores,
        "roll_multicrop_overlap_probe_v1": _compute_roll_multicrop_scores,
    }[family]
    for (dataset, split), group in source_scores.groupby(["dataset", "split"], sort=True):
        cache = _cache_path(outdir, family, str(dataset), str(split))
        cache.parent.mkdir(parents=True, exist_ok=True)
        if reuse_existing and cache.exists():
            message = f"[REUSE] {family} {dataset}/{split}: {cache}"
            print(message)
            log_handle.write(message + "\n")
            frames.append(pd.read_csv(cache))
            continue
        work = group.copy()
        if int(limit_per_split) > 0:
            work = work.head(int(limit_per_split)).copy()
        message = f"[RUN] {family} {dataset}/{split} pairs={len(work)}"
        print(message)
        log_handle.write(message + "\n")
        start = time.perf_counter()
        scored = family_fn(work)
        elapsed = time.perf_counter() - start
        scored["runtime_family"] = family
        scored["runtime_dataset"] = str(dataset)
        scored["runtime_split"] = str(split)
        scored["runtime_elapsed_s_for_file"] = float(elapsed)
        scored.to_csv(cache, index=False)
        log_handle.write(f"[DONE] {family} {dataset}/{split} elapsed_s={elapsed:.3f} cache={cache}\n")
        frames.append(scored)
        _clear_image_caches()
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _clear_image_caches() -> None:
    _standard_snapshot.cache_clear()
    _crop_overlap_snapshot.cache_clear()
    _roll_crop_snapshot.cache_clear()
    _load_gray.cache_clear()


def build_image_probe_scores(
    source_scores: pd.DataFrame,
    *,
    outdir: Path,
    families: tuple[str, ...],
    reuse_existing: bool,
    limit_per_split: int,
    log_handle: Any,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for family in families:
        if family not in {"geometry_probe_v1", "crop_overlap_probe_v1", "roll_multicrop_overlap_probe_v1"}:
            continue
        frames.append(
            _compute_family_scores(
                source_scores,
                family=family,
                outdir=outdir,
                reuse_existing=reuse_existing,
                limit_per_split=limit_per_split,
                log_handle=log_handle,
            )
        )
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _candidate_scores_for_threshold(
    source_scores: pd.DataFrame,
    scalar_scores: pd.DataFrame,
    threshold_row: pd.Series,
) -> pd.DataFrame:
    candidate_name = str(threshold_row["candidate_name"])
    dataset = str(threshold_row["dataset"])
    if str(threshold_row["threshold_kind"]) == "conservative_or_target_rule":
        source_thresholds = json.loads(str(threshold_row["source_thresholds_json"]))
        frame = source_scores[source_scores["dataset"] == dataset].copy()
        frame["candidate_name"] = candidate_name
        frame["candidate_family"] = str(threshold_row["candidate_family"])
        frame["probe_kind"] = str(threshold_row["probe_kind"])
        frame["score"] = _or_rule_scores(frame, source_thresholds)
        frame["matches"] = ""
        frame["inliers"] = ""
        frame["k1"] = ""
        frame["k2"] = ""
        frame["diagnostic_json"] = str(threshold_row["source_thresholds_json"])
        frame["research_only"] = True
        return frame[
            [
                "dataset",
                "split",
                "label",
                "path_a",
                "path_b",
                "frgp",
                "candidate_name",
                "candidate_family",
                "probe_kind",
                "score",
                "matches",
                "inliers",
                "k1",
                "k2",
                "diagnostic_json",
                "research_only",
            ]
        ].copy()
    return scalar_scores[
        (scalar_scores["dataset"] == dataset) & (scalar_scores["candidate_name"] == candidate_name)
    ].copy()


def _severity_from_v2(row: pd.Series) -> str:
    if int(row["label"]) != 1 or bool(row["v2_accepted"]):
        return ""
    ratio = _to_float(row.get("v2_score_margin_ratio"))
    if ratio >= -0.10:
        return "near_miss"
    if ratio >= -0.50:
        return "moderate_margin_failure"
    return "hard_score_failure"


def build_candidate_decisions(
    source_scores: pd.DataFrame,
    scalar_candidate_scores: pd.DataFrame,
    candidate_thresholds: pd.DataFrame,
    v2_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    threshold_lookup = {
        (str(row["dataset"]), round(float(row["target_far"]), 6)): float(row["v2_threshold"])
        for _, row in v2_thresholds.iterrows()
    }
    frames: list[pd.DataFrame] = []
    for _, threshold_row in candidate_thresholds.iterrows():
        scored = _candidate_scores_for_threshold(source_scores, scalar_candidate_scores, threshold_row)
        if scored.empty:
            continue
        dataset = str(threshold_row["dataset"])
        target_far = float(threshold_row["target_far"])
        threshold = float(threshold_row["threshold"])
        v2_threshold = threshold_lookup[(dataset, round(target_far, 6))]
        keep_source = [
            *PAIR_KEYS,
            "subject_a",
            "subject_b",
            "frgp_a",
            "frgp_b",
            "canonical_current_score",
            "sift_inliers_score",
            "v2_official_score",
            "v2_inliers",
            "v2_matches",
            "v2_k1",
            "v2_k2",
        ]
        merged = scored.merge(
            source_scores[keep_source],
            on=list(PAIR_KEYS),
            how="left",
            validate="many_to_one",
            suffixes=("", "_source"),
        )
        if "frgp_source" in merged.columns:
            merged["frgp"] = merged["frgp"].where(merged["frgp"].astype(str).str.len() > 0, merged["frgp_source"])
            merged = merged.drop(columns=["frgp_source"])
        merged["target_far"] = target_far
        merged["threshold"] = threshold
        merged["selected_by_val"] = bool(threshold_row["selected_by_val"])
        merged["val_safety_status"] = str(threshold_row["val_safety_status"])
        merged["candidate_accepted"] = pd.to_numeric(merged["score"], errors="coerce").fillna(-math.inf) >= threshold
        merged["candidate_score_margin"] = pd.to_numeric(merged["score"], errors="coerce") - threshold
        merged["candidate_score_margin_ratio"] = merged["candidate_score_margin"] / threshold if threshold else float("nan")
        merged["v2_threshold"] = v2_threshold
        merged["v2_accepted"] = pd.to_numeric(merged["v2_official_score"], errors="coerce").fillna(-math.inf) >= v2_threshold
        merged["v2_score_margin"] = pd.to_numeric(merged["v2_official_score"], errors="coerce") - v2_threshold
        merged["v2_score_margin_ratio"] = merged["v2_score_margin"] / v2_threshold if v2_threshold else float("nan")
        merged["v2_failure_severity"] = merged.apply(_severity_from_v2, axis=1)
        merged["research_only"] = True
        frames.append(merged)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_candidate_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "split", "candidate_family", "candidate_name", "target_far"]
    for keys, group in decisions.groupby(group_cols, sort=True):
        dataset, split, family, candidate_name, target_far = keys
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        scores = pd.to_numeric(group["score"], errors="coerce").fillna(float("nan")).to_numpy(dtype=float)
        threshold = float(group["threshold"].iloc[0])
        counts = _confusion(labels, scores, threshold)
        auc, eer, eer_threshold = _auc_eer(labels, scores)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "candidate_family": family,
                "candidate_name": candidate_name,
                "target_far": float(target_far),
                "threshold": threshold,
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
                "selected_by_val": bool(group["selected_by_val"].iloc[0]),
                "val_safety_status": str(group["val_safety_status"].iloc[0]),
                "research_only": True,
            }
        )
    return pd.DataFrame(rows)


def build_v2_reference_metrics(
    source_scores: pd.DataFrame,
    v2_thresholds: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = {
        (str(row["dataset"]), round(float(row["target_far"]), 6)): float(row["v2_threshold"])
        for _, row in v2_thresholds.iterrows()
    }
    for (dataset, split), group in source_scores.groupby(["dataset", "split"], sort=True):
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        scores = _safe_numeric(group["v2_official_score"])
        for target_far in sorted({float(x[1]) for x in lookup if x[0] == str(dataset)}):
            threshold = lookup[(str(dataset), round(float(target_far), 6))]
            counts = _confusion(labels, scores, threshold)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "candidate_family": "reference_v2",
                    "candidate_name": "sift_plain_roll_v2_official_reference",
                    "target_far": float(target_far),
                    "threshold": threshold,
                    "far": float(counts["far"]),
                    "tar": float(counts["tar"]),
                    "frr": float(counts["frr"]),
                    "ta": int(counts["ta"]),
                    "fr": int(counts["fr"]),
                    "fa": int(counts["fa"]),
                    "tr": int(counts["tr"]),
                    "n_positive": int(counts["n_positive"]),
                    "n_negative": int(counts["n_negative"]),
                    "research_only": False,
                }
            )
    return pd.DataFrame(rows)


def build_per_frgp_candidate_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "split", "candidate_family", "candidate_name", "target_far", "frgp"]
    for keys, group in decisions.groupby(group_cols, dropna=False, sort=True):
        dataset, split, family, candidate_name, target_far, frgp = keys
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        scores = pd.to_numeric(group["score"], errors="coerce").fillna(float("nan")).to_numpy(dtype=float)
        threshold = float(group["threshold"].iloc[0])
        counts = _confusion(labels, scores, threshold)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "candidate_family": family,
                "candidate_name": candidate_name,
                "target_far": float(target_far),
                "frgp": "" if pd.isna(frgp) or str(frgp) == "" else int(float(frgp)),
                "far": float(counts["far"]),
                "tar": float(counts["tar"]),
                "ta": int(counts["ta"]),
                "fr": int(counts["fr"]),
                "fa": int(counts["fa"]),
                "tr": int(counts["tr"]),
                "n_positive": int(counts["n_positive"]),
                "n_negative": int(counts["n_negative"]),
                "selected_by_val": bool(group["selected_by_val"].iloc[0]),
                "research_only": True,
            }
        )
    return pd.DataFrame(rows)


def build_candidate_decision_overlap(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if decisions.empty:
        return pd.DataFrame()
    group_cols = ["dataset", "split", "candidate_family", "candidate_name", "target_far"]
    for keys, group in decisions.groupby(group_cols, sort=True):
        dataset, split, family, candidate_name, target_far = keys
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int)
        cand = group["candidate_accepted"].astype(bool)
        v2 = group["v2_accepted"].astype(bool)
        positives = labels == 1
        negatives = labels == 0
        hard_v2_fr = positives & (~v2) & (group["v2_failure_severity"] == "hard_score_failure")
        row = {
            "dataset": dataset,
            "split": split,
            "candidate_family": family,
            "candidate_name": candidate_name,
            "target_far": float(target_far),
            "selected_by_val": bool(group["selected_by_val"].iloc[0]),
            "positive_both_accept": int(np.sum(positives & v2 & cand)),
            "positive_candidate_rescue_vs_v2": int(np.sum(positives & (~v2) & cand)),
            "positive_candidate_lost_vs_v2": int(np.sum(positives & v2 & (~cand))),
            "positive_both_reject": int(np.sum(positives & (~v2) & (~cand))),
            "negative_both_reject": int(np.sum(negatives & (~v2) & (~cand))),
            "negative_candidate_new_false_accept_vs_v2": int(np.sum(negatives & (~v2) & cand)),
            "negative_candidate_fixed_false_accept_vs_v2": int(np.sum(negatives & v2 & (~cand))),
            "negative_both_false_accept": int(np.sum(negatives & v2 & cand)),
            "hard_v2_false_rejects": int(np.sum(hard_v2_fr)),
            "hard_v2_false_rejects_rescued": int(np.sum(hard_v2_fr & cand)),
            "n_positive": int(np.sum(positives)),
            "n_negative": int(np.sum(negatives)),
            "research_only": True,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_candidate_case_tables(decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if decisions.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_cols = [
        "dataset",
        "split",
        "candidate_family",
        "candidate_name",
        "target_far",
        "selected_by_val",
        "label",
        "frgp",
        "path_a",
        "path_b",
        "score",
        "threshold",
        "candidate_score_margin",
        "candidate_score_margin_ratio",
        "v2_official_score",
        "v2_threshold",
        "v2_accepted",
        "v2_failure_severity",
        "matches",
        "inliers",
        "k1",
        "k2",
        "diagnostic_json",
        "research_only",
    ]
    false_accepts = decisions[(decisions["label"].astype(int) == 0) & (decisions["candidate_accepted"].astype(bool))].copy()
    false_accepts["high_confidence_false_accept"] = (
        pd.to_numeric(false_accepts["candidate_score_margin_ratio"], errors="coerce").fillna(0.0) >= 0.50
    )
    false_accepts = false_accepts.sort_values(
        ["dataset", "split", "target_far", "candidate_name", "score"],
        ascending=[True, True, True, True, False],
    )
    false_rejects = decisions[(decisions["label"].astype(int) == 1) & (~decisions["candidate_accepted"].astype(bool))].copy()
    false_rejects["was_hard_v2_false_reject"] = false_rejects["v2_failure_severity"] == "hard_score_failure"
    false_rejects = false_rejects.sort_values(
        ["dataset", "split", "target_far", "candidate_name", "score"],
        ascending=[True, True, True, True, False],
    )
    fa_cols = [*base_cols, "high_confidence_false_accept"]
    fr_cols = [*base_cols, "was_hard_v2_false_reject"]
    return false_accepts[fa_cols].reset_index(drop=True), false_rejects[fr_cols].reset_index(drop=True)


def build_runtime_summary(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    if candidate_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    if "runtime_elapsed_s_for_file" in candidate_scores.columns:
        for (family, dataset, split), group in candidate_scores.groupby(
            ["runtime_family", "runtime_dataset", "runtime_split"],
            dropna=True,
            sort=True,
        ):
            elapsed = _to_float(group["runtime_elapsed_s_for_file"].iloc[0], 0.0)
            pairs = len(group[list(PAIR_KEYS)].drop_duplicates())
            rows.append(
                {
                    "candidate_family": family,
                    "dataset": dataset,
                    "split": split,
                    "n_pairs": int(pairs),
                    "elapsed_s": float(elapsed),
                    "avg_ms_pair": float(1000.0 * elapsed / max(pairs, 1)),
                    "source": "image_probe_cache_file",
                }
            )
    fusion = candidate_scores[candidate_scores["candidate_family"] == "fusion_probe_v1"] if "candidate_family" in candidate_scores else pd.DataFrame()
    if not fusion.empty:
        for (dataset, split), group in fusion.groupby(["dataset", "split"], sort=True):
            pairs = len(group[list(PAIR_KEYS)].drop_duplicates())
            rows.append(
                {
                    "candidate_family": "fusion_probe_v1",
                    "dataset": dataset,
                    "split": split,
                    "n_pairs": int(pairs),
                    "elapsed_s": 0.0,
                    "avg_ms_pair": 0.0,
                    "source": "existing_score_columns_only",
                }
            )
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)


def assert_required_output_coverage(
    candidate_metrics: pd.DataFrame,
    per_frgp_candidate_metrics: pd.DataFrame,
    *,
    required_datasets: tuple[str, ...] = DEFAULT_DATASETS,
    required_frgps: tuple[int, ...] = FRGP_FOCUS,
) -> None:
    datasets = set(str(x) for x in candidate_metrics.get("dataset", pd.Series(dtype=str)).dropna().unique())
    missing_datasets = sorted(set(required_datasets) - datasets)
    if missing_datasets:
        raise AssertionError(f"candidate metrics missing required datasets: {missing_datasets}")
    frgps = set(
        int(float(x))
        for x in per_frgp_candidate_metrics.get("frgp", pd.Series(dtype=object)).dropna().tolist()
        if str(x) != ""
    )
    missing_frgps = sorted(set(required_frgps) - frgps)
    if missing_frgps:
        raise AssertionError(f"per-FRGP candidate metrics missing required FRGP groups: {missing_frgps}")


def assert_v2_scores_unchanged(before: pd.DataFrame, after_source_scores: pd.DataFrame) -> None:
    before_cols = before[list(PAIR_KEYS) + ["v2_official_score"]].copy().sort_values(list(PAIR_KEYS)).reset_index(drop=True)
    after_cols = (
        after_source_scores[list(PAIR_KEYS) + ["v2_official_score"]]
        .copy()
        .sort_values(list(PAIR_KEYS))
        .reset_index(drop=True)
    )
    try:
        pd.testing.assert_frame_equal(before_cols, after_cols, check_dtype=False, check_exact=False, rtol=0, atol=0)
    except AssertionError as exc:
        raise AssertionError("research probes must not mutate existing v2 official scores") from exc


def _metric_lookup(metrics: pd.DataFrame, *, split: str, target_far: float) -> pd.DataFrame:
    return metrics[
        (metrics["split"].astype(str).str.lower() == split)
        & np.isclose(pd.to_numeric(metrics["target_far"], errors="coerce"), float(target_far))
    ].copy()


def _reference_metric(v2_metrics: pd.DataFrame, dataset: str, split: str, target_far: float) -> pd.Series | None:
    subset = v2_metrics[
        (v2_metrics["dataset"] == dataset)
        & (v2_metrics["split"].astype(str).str.lower() == split)
        & np.isclose(pd.to_numeric(v2_metrics["target_far"], errors="coerce"), float(target_far))
    ]
    return None if subset.empty else subset.iloc[0]


def _best_frgp_family(
    per_frgp: pd.DataFrame,
    v2_per_frgp: pd.DataFrame,
    *,
    split: str,
    target_far: float = 0.01,
) -> tuple[pd.Series | None, float]:
    rows: list[dict[str, Any]] = []
    candidate = per_frgp[
        (per_frgp["split"].astype(str).str.lower() == split)
        & np.isclose(pd.to_numeric(per_frgp["target_far"], errors="coerce"), float(target_far))
        & (per_frgp["selected_by_val"].astype(bool))
        & (per_frgp["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
    ].copy()
    if candidate.empty:
        return None, float("nan")
    ref = v2_per_frgp[
        (v2_per_frgp["split"].astype(str).str.lower() == split)
        & np.isclose(pd.to_numeric(v2_per_frgp["target_far"], errors="coerce"), float(target_far))
        & (v2_per_frgp["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
    ].copy()
    for (dataset, family, name), group in candidate.groupby(["dataset", "candidate_family", "candidate_name"], sort=True):
        deltas: list[float] = []
        for _, row in group.iterrows():
            r = ref[(ref["dataset"] == dataset) & (ref["frgp"].astype(str) == str(row["frgp"]))]
            if not r.empty:
                deltas.append(float(row["tar"]) - float(r.iloc[0]["tar"]))
        if deltas:
            rows.append(
                {
                    "dataset": dataset,
                    "candidate_family": family,
                    "candidate_name": name,
                    "mean_frgp_5_10_tar_delta": float(np.mean(deltas)),
                }
            )
    if not rows:
        return None, float("nan")
    ranking = pd.DataFrame(rows).groupby(["candidate_family", "candidate_name"], sort=True)[
        "mean_frgp_5_10_tar_delta"
    ].mean()
    best_key = ranking.sort_values(ascending=False).index[0]
    best_delta = float(ranking.loc[best_key])
    row = pd.Series({"candidate_family": best_key[0], "candidate_name": best_key[1]})
    return row, best_delta


def build_v2_per_frgp_metrics(source_scores: pd.DataFrame, v2_thresholds: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (str(row["dataset"]), round(float(row["target_far"]), 6)): float(row["v2_threshold"])
        for _, row in v2_thresholds.iterrows()
    }
    rows: list[dict[str, Any]] = []
    for (dataset, split, frgp), group in source_scores.groupby(["dataset", "split", "frgp"], dropna=False, sort=True):
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        scores = _safe_numeric(group["v2_official_score"])
        for target_far in sorted({float(x[1]) for x in lookup if x[0] == str(dataset)}):
            threshold = lookup[(str(dataset), round(float(target_far), 6))]
            counts = _confusion(labels, scores, threshold)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "target_far": float(target_far),
                    "frgp": "" if pd.isna(frgp) or str(frgp) == "" else int(float(frgp)),
                    "far": float(counts["far"]),
                    "tar": float(counts["tar"]),
                    "ta": int(counts["ta"]),
                    "fr": int(counts["fr"]),
                    "fa": int(counts["fa"]),
                    "tr": int(counts["tr"]),
                    "n_positive": int(counts["n_positive"]),
                    "n_negative": int(counts["n_negative"]),
                }
            )
    return pd.DataFrame(rows)


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


def render_summary(
    *,
    outdir: Path,
    candidate_metrics: pd.DataFrame,
    v2_metrics: pd.DataFrame,
    per_frgp: pd.DataFrame,
    v2_per_frgp: pd.DataFrame,
    overlap: pd.DataFrame,
    false_accepts: pd.DataFrame,
    selected_val: pd.DataFrame,
    command_text: str,
) -> str:
    best_val_row, best_val_delta = _best_frgp_family(per_frgp, v2_per_frgp, split="val", target_far=0.01)
    best_test_delta = float("nan")
    if best_val_row is not None:
        best_name = str(best_val_row["candidate_name"])
        test_candidate = per_frgp[
            (per_frgp["split"].astype(str).str.lower() == "test")
            & (per_frgp["candidate_name"] == best_name)
            & np.isclose(pd.to_numeric(per_frgp["target_far"], errors="coerce"), 0.01)
            & (per_frgp["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
        ].copy()
        test_ref = v2_per_frgp[
            (v2_per_frgp["split"].astype(str).str.lower() == "test")
            & np.isclose(pd.to_numeric(v2_per_frgp["target_far"], errors="coerce"), 0.01)
            & (v2_per_frgp["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
        ].copy()
        deltas = []
        for _, row in test_candidate.iterrows():
            ref = test_ref[(test_ref["dataset"] == row["dataset"]) & (test_ref["frgp"].astype(str) == str(row["frgp"]))]
            if not ref.empty:
                deltas.append(float(row["tar"]) - float(ref.iloc[0]["tar"]))
        if deltas:
            best_test_delta = float(np.mean(deltas))

    one_pct_test = _metric_lookup(candidate_metrics, split="test", target_far=0.01)
    safe_improvements: list[str] = []
    for _, row in one_pct_test.iterrows():
        ref = _reference_metric(v2_metrics, str(row["dataset"]), "test", 0.01)
        if ref is None:
            continue
        if bool(row["selected_by_val"]) and float(row["tar"]) > float(ref["tar"]) and float(row["far"]) <= 0.01 + 1e-12:
            safe_improvements.append(
                f"{row['dataset']} `{row['candidate_name']}` TAR {_pct(row['tar'])} vs v2 {_pct(ref['tar'])}, FAR {_pct(row['far'])}"
            )

    test_overlap_1pct = overlap[
        (overlap["split"].astype(str).str.lower() == "test")
        & np.isclose(pd.to_numeric(overlap["target_far"], errors="coerce"), 0.01)
        & (overlap["selected_by_val"].astype(bool))
    ].copy()
    hard_rescue = pd.DataFrame()
    if not test_overlap_1pct.empty:
        hard_rescue = (
            test_overlap_1pct.groupby(["candidate_family", "candidate_name"], sort=True)[
                "hard_v2_false_rejects_rescued"
            ]
            .sum()
            .reset_index()
            .sort_values("hard_v2_false_rejects_rescued", ascending=False)
        )
    high_conf = false_accepts[
        (false_accepts["split"].astype(str).str.lower() == "test")
        & np.isclose(pd.to_numeric(false_accepts["target_far"], errors="coerce"), 0.01)
        & (false_accepts["high_confidence_false_accept"].astype(bool))
    ].copy()
    high_conf_summary = (
        high_conf.groupby(["candidate_family", "candidate_name"], sort=True).size().reset_index(name="count")
        if not high_conf.empty
        else pd.DataFrame(columns=["candidate_family", "candidate_name", "count"])
    )
    family_best = one_pct_test[one_pct_test["selected_by_val"].astype(bool)].copy()
    family_notes: dict[str, str] = {}
    for family, group in family_best.groupby("candidate_family", sort=True):
        rows = []
        for _, row in group.iterrows():
            ref = _reference_metric(v2_metrics, str(row["dataset"]), "test", 0.01)
            if ref is not None:
                rows.append(
                    {
                        "tar_delta": float(row["tar"]) - float(ref["tar"]),
                        "far": float(row["far"]),
                        "candidate_name": str(row["candidate_name"]),
                    }
                )
        if rows:
            best = sorted(rows, key=lambda x: (x["tar_delta"], -x["far"]), reverse=True)[0]
            family_notes[family] = (
                f"best TEST 1% row `{best['candidate_name']}` delta {_fmt(best['tar_delta'], 4)} "
                f"at FAR {_pct(best['far'])}"
            )

    val_answer = (
        "No VAL-selected FRGP 5/10 candidate was available."
        if best_val_row is None
        else (
            f"`{best_val_row['candidate_family']}` via `{best_val_row['candidate_name']}` "
            f"with mean FRGP 5/10 TAR delta {_fmt(best_val_delta, 4)} on VAL."
        )
    )
    test_answer = (
        "No, the VAL-selected FRGP 5/10 winner did not show a measurable TEST delta."
        if not math.isfinite(best_test_delta)
        else f"TEST mean FRGP 5/10 TAR delta for that same candidate is {_fmt(best_test_delta, 4)}."
    )
    safe_answer = "No candidate met the strict TEST TAR-improvement and FAR<=1% condition."
    if safe_improvements:
        safe_answer = "Yes: " + " ; ".join(safe_improvements[:8])
    hard_answer = "No selected candidate rescued hard v2 false rejects at TEST 1%."
    if not hard_rescue.empty:
        top = hard_rescue.iloc[0]
        hard_answer = (
            f"`{top['candidate_name']}` rescued {int(top['hard_v2_false_rejects_rescued'])} "
            "hard v2 false rejects at TEST 1% among VAL-selected candidates."
        )
    high_answer = "No high-confidence TEST 1% false accepts were introduced by the reported candidates."
    if not high_conf_summary.empty:
        high_answer = "; ".join(
            f"`{row['candidate_name']}`: {int(row['count'])}"
            for _, row in high_conf_summary.sort_values("count", ascending=False).head(10).iterrows()
        )

    crop_note = family_notes.get("crop_overlap_probe_v1", "no selected crop candidate row at TEST 1%")
    geometry_note = family_notes.get("geometry_probe_v1", "no selected geometry candidate row at TEST 1%")
    fusion_note = family_notes.get("fusion_probe_v1", "no selected fusion candidate row at TEST 1%")
    next_direction = "Treat this run as hypothesis evidence, not implementation approval."
    if safe_improvements:
        if hard_rescue.empty:
            next_direction = "Prioritize the safest VAL-selected low-FAR candidate family for a real implementation follow-up, then revalidate under the same locked protocol."
        else:
            top_family = str(hard_rescue.iloc[0]["candidate_family"])
            next_direction = (
                f"Use `{top_family}` as the next implementation direction, with FAR controls carried forward "
                "and fusion considered only as a guarded decision layer if it stays safe."
            )

    selected_lines = ["| dataset | family | selected candidate | VAL TAR | VAL FAR |", "| --- | --- | --- | ---: | ---: |"]
    if selected_val.empty:
        selected_lines.append("|  |  | none |  |  |")
    else:
        for _, row in selected_val.sort_values(["dataset", "candidate_family"]).iterrows():
            selected_lines.append(
                f"| {row['dataset']} | {row['candidate_family']} | `{row['candidate_name']}` | "
                f"{_pct(row['tar'])} | {_pct(row['far'])} |"
            )

    lines = [
        "# SIFT Plain/Roll v2 Hypothesis Tests",
        "",
        "Scope: controlled research-only probes for the remaining SIFT Plain/Roll v2 failures. No matcher behavior, canonical SIFT behavior, existing `sift_plain_roll_v2` behavior, production/API thresholds, UI/default/showcase behavior, or canonical method registry entries are changed.",
        "",
        "Protocol: candidate thresholds and candidate selection use VAL only. TEST is reported once for the fixed candidate families and is not used to choose parameters, weights, padding, crop grids, geometry models, or fusion rules.",
        "",
        f"Output folder: `{outdir}`",
        f"Command: `{command_text}`",
        "",
        "## Required Answers",
        "",
        f"1. Which family improves FRGP 5/10 the most on VAL? {val_answer}",
        f"2. Does that improvement survive on TEST? {test_answer}",
        f"3. Does any candidate improve v2 TEST TAR at 1% FAR without increasing FAR dangerously? {safe_answer}",
        f"4. Which candidate rescues the most hard false rejects? {hard_answer}",
        f"5. Which candidate introduces high-confidence false accepts? {high_answer}",
        f"6. Is crop/overlap normalization enough by itself? {crop_note}. Treat as enough only if it improves TEST TAR at 1% with FAR<=1% and rescues hard failures.",
        f"7. Is geometry enough by itself? {geometry_note}. Treat as enough only if the gain survives TEST safely.",
        f"8. Does fusion provide low-FAR benefit without solving hard failures? {fusion_note}. Compare this to the hard-FR rescue table before promoting any fusion idea.",
        f"9. What should be the next implementation direction? {next_direction}",
        "",
        "## VAL-Selected Candidates At 1% FAR",
        "",
        *selected_lines,
        "",
        "## Guardrails",
        "",
        "- All probe candidate names are script-local and prefixed with `research_only::`.",
        "- `configs/thresholds.yaml`, `configs/methods.yaml`, `apps/api`, `apps/ui`, and benchmark defaults are not edited by this script.",
        "- Crop paddings, roll crop grid, geometry models, and fusion rules are fixed before TEST reporting.",
        "- Conservative OR fusion rules are marked rejected when the VAL union FAR exceeds the target.",
    ]
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _families_from_args(raw: str) -> tuple[str, ...]:
    if not raw or raw.strip().lower() == "all":
        return (
            "crop_overlap_probe_v1",
            "roll_multicrop_overlap_probe_v1",
            "geometry_probe_v1",
            "fusion_probe_v1",
        )
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    allowed = {"crop_overlap_probe_v1", "roll_multicrop_overlap_probe_v1", "geometry_probe_v1", "fusion_probe_v1"}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown family values: {unknown}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run research-only SIFT Plain/Roll v2 hypothesis probes.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="External validation score artifact folder.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="New hypothesis-test output folder.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated splits, normally val,test.")
    parser.add_argument("--families", default="all", help="Comma-separated candidate families or all.")
    parser.add_argument("--limit-per-split", type=int, default=0, help="Debug cap per dataset/split; 0 means full.")
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse cached candidate score CSVs when present.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite required summary/report CSVs in outdir.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    specs = candidate_registry()
    assert_research_only_candidate_names(specs)
    outdir = parse_file_uri(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    required_outputs = {
        "summary": outdir / "hypothesis_test_summary.md",
        "metrics_val": outdir / "candidate_metrics_val.csv",
        "metrics_test": outdir / "candidate_metrics_test.csv",
        "thresholds": outdir / "candidate_thresholds.csv",
        "per_frgp": outdir / "per_frgp_candidate_metrics.csv",
        "overlap": outdir / "candidate_decision_overlap.csv",
        "false_accepts": outdir / "candidate_false_accepts.csv",
        "false_rejects": outdir / "candidate_false_rejects.csv",
        "runtime": outdir / "candidate_runtime_summary.csv",
        "command_log": outdir / "command_log.txt",
        "manifest": outdir / "run_manifest.json",
    }
    if not bool(args.overwrite):
        existing = [path for key, path in required_outputs.items() if key != "command_log" and path.exists()]
        if existing:
            raise FileExistsError(
                "Required output files already exist. Pass --overwrite to replace files in the hypothesis-test folder: "
                + ", ".join(str(path) for path in existing[:5])
            )

    command_text = " ".join([Path(sys.executable).name, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]])
    with required_outputs["command_log"].open("w", encoding="utf-8", newline="\n") as log_handle:
        log_handle.write("SIFT Plain/Roll v2 hypothesis-test command log\n")
        log_handle.write(f"started_at_utc={datetime.now(timezone.utc).isoformat()}\n")
        log_handle.write(f"command={command_text}\n")
        log_handle.write(f"cwd={REPO_ROOT}\n\n")
        datasets = tuple(item.strip() for item in str(args.datasets).split(",") if item.strip())
        splits = tuple(item.strip().lower() for item in str(args.splits).split(",") if item.strip())
        families = _families_from_args(str(args.families))
        source_scores = load_aligned_source_scores(args.input_dir, datasets=datasets, splits=splits)
        source_before = source_scores.copy(deep=True)
        log_handle.write(f"loaded_source_scores rows={len(source_scores)} datasets={datasets} splits={splits}\n")

        scalar_frames: list[pd.DataFrame] = []
        image_families = tuple(f for f in families if f != "fusion_probe_v1")
        if image_families:
            scalar_frames.append(
                build_image_probe_scores(
                    source_scores,
                    outdir=outdir,
                    families=image_families,
                    reuse_existing=bool(args.reuse_existing),
                    limit_per_split=int(args.limit_per_split),
                    log_handle=log_handle,
                )
            )
        if "fusion_probe_v1" in families:
            start = time.perf_counter()
            fusion_scores = build_fusion_probe_scores(source_scores)
            elapsed = time.perf_counter() - start
            fusion_scores["runtime_elapsed_s_for_file"] = elapsed
            scalar_frames.append(fusion_scores)
            fusion_cache = outdir / "candidate_score_cache" / "fusion_probe_v1_scores.csv"
            fusion_cache.parent.mkdir(parents=True, exist_ok=True)
            fusion_scores.to_csv(fusion_cache, index=False)
            log_handle.write(f"[DONE] fusion_probe_v1 scalar scores elapsed_s={elapsed:.3f} cache={fusion_cache}\n")

        scalar_candidate_scores = (
            pd.concat([frame for frame in scalar_frames if frame is not None and not frame.empty], ignore_index=True, sort=False)
            if scalar_frames
            else pd.DataFrame()
        )
        scalar_thresholds = calibrate_scalar_candidate_thresholds(scalar_candidate_scores)
        threshold_frames = [scalar_thresholds]
        if "fusion_probe_v1" in families:
            threshold_frames.append(calibrate_conservative_or_thresholds(source_scores))
        candidate_thresholds = pd.concat(
            [frame for frame in threshold_frames if frame is not None and not frame.empty],
            ignore_index=True,
            sort=False,
        )
        v2_thresholds = calibrate_reference_v2_thresholds(source_scores)
        decisions = build_candidate_decisions(source_scores, scalar_candidate_scores, candidate_thresholds, v2_thresholds)
        assert_v2_scores_unchanged(source_before, source_scores)
        metrics = build_candidate_metrics(decisions)
        metrics_val = metrics[metrics["split"].astype(str).str.lower() == "val"].copy()
        metrics_test = metrics[metrics["split"].astype(str).str.lower() == "test"].copy()
        per_frgp = build_per_frgp_candidate_metrics(decisions)
        overlap = build_candidate_decision_overlap(decisions)
        false_accepts, false_rejects = build_candidate_case_tables(decisions)
        runtime = build_runtime_summary(scalar_candidate_scores)
        v2_metrics = build_v2_reference_metrics(source_scores, v2_thresholds)
        v2_per_frgp = build_v2_per_frgp_metrics(source_scores, v2_thresholds)
        selected_val = select_candidates_from_val(metrics_val, target_far=0.01)

        assert_required_output_coverage(pd.concat([metrics_val, metrics_test], ignore_index=True), per_frgp, required_datasets=datasets)

        _write_csv(required_outputs["thresholds"], candidate_thresholds)
        _write_csv(required_outputs["metrics_val"], metrics_val)
        _write_csv(required_outputs["metrics_test"], metrics_test)
        _write_csv(required_outputs["per_frgp"], per_frgp)
        _write_csv(required_outputs["overlap"], overlap)
        _write_csv(required_outputs["false_accepts"], false_accepts)
        _write_csv(required_outputs["false_rejects"], false_rejects)
        _write_csv(required_outputs["runtime"], runtime)
        summary = render_summary(
            outdir=outdir,
            candidate_metrics=metrics,
            v2_metrics=v2_metrics,
            per_frgp=per_frgp,
            v2_per_frgp=v2_per_frgp,
            overlap=overlap,
            false_accepts=false_accepts,
            selected_val=selected_val,
            command_text=command_text,
        )
        required_outputs["summary"].write_text(summary, encoding="utf-8", newline="\n")

        manifest = {
            "schema_version": "sift_plain_roll_v2_hypothesis_tests_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(REPO_ROOT),
            "input_dir": str(parse_file_uri(args.input_dir)),
            "outdir": str(outdir),
            "datasets": list(datasets),
            "splits": list(splits),
            "target_fars": list(TARGET_FARS),
            "protocol": {
                "candidate_selection_split": "val",
                "threshold_calibration_split": "val",
                "test_usage": "locked final reporting only; no TEST parameter tuning",
                "research_only": True,
                "production_thresholds_yaml_changed": False,
                "canonical_sift_changed": False,
                "sift_plain_roll_v2_changed": False,
            },
            "v2_parameters_reused": {
                "target_size": TARGET_SIZE,
                "nfeatures": NFEATURES,
                "blur_ksize": BLUR_KSIZE,
                "lowe_ratio": LOWE_RATIO,
                "ransac_thresh": RANSAC_THRESH,
                "score": "inliers_times_inlier_ratio_times_log1p_matches",
                "geometry_model": V2_GEOMETRY_MODEL,
            },
            "candidate_registry": [
                {
                    "candidate_name": spec.candidate_name,
                    "candidate_family": spec.candidate_family,
                    "probe_kind": spec.probe_kind,
                    "parameters": spec.parameters,
                    "research_only": spec.research_only,
                }
                for spec in specs
            ],
            "limits": {"limit_per_split": int(args.limit_per_split)},
            "outputs": {key: str(path) for key, path in required_outputs.items()},
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        }
        required_outputs["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        log_handle.write("\n[OUTPUTS]\n")
        for key, path in required_outputs.items():
            log_handle.write(f"{key}={path}\n")
    return required_outputs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    outputs = run(args)
    print("Wrote SIFT Plain/Roll v2 hypothesis-test artifacts:")
    for path in outputs.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
