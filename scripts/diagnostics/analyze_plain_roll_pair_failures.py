from __future__ import annotations

import argparse
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.preprocess.preprocess import (  # noqa: E402
    PreprocessConfig,
    extract_fingerprint_roi,
    preprocess_image,
)


DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_professor_1000_pos_neg"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_plain_roll_diagnostics"
)
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)
PAIR_SETS = ("positive_1000", "negative_1000")


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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


def _split_flags(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    for sep in (";", "|", ","):
        if sep in text:
            return {part.strip() for part in text.split(sep) if part.strip()}
    return {text}


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


@lru_cache(maxsize=8192)
def _load_gray(path_str: str) -> np.ndarray:
    path = parse_file_uri(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


@lru_cache(maxsize=8192)
def image_quality_metrics(path_str: str, target_size: int) -> dict[str, Any]:
    original = _load_gray(path_str)
    processed = preprocess_image(original, PreprocessConfig(target_size=int(target_size)))
    roi_result = extract_fingerprint_roi(processed)
    mask = roi_result.mask if roi_result.is_valid else np.zeros_like(processed, dtype=np.uint8)
    foreground = mask > 0
    values = processed[foreground] if np.any(foreground) else processed.reshape(-1)
    stats = _safe_stats(values)
    lap = cv2.Laplacian(processed, cv2.CV_64F)
    lap_values = lap[foreground] if np.any(foreground) else lap.reshape(-1)
    foreground_ratio = float(np.mean(foreground)) if mask.size else 0.0
    contrast_p95_p05 = float(stats["p95"] - stats["p05"]) if math.isfinite(stats["p95"]) else float("nan")
    return {
        "height": int(original.shape[0]),
        "width": int(original.shape[1]),
        "processed_height": int(processed.shape[0]),
        "processed_width": int(processed.shape[1]),
        "foreground_ratio": foreground_ratio,
        "roi_valid": bool(roi_result.is_valid),
        "roi_failure_reason": roi_result.failure_reason or "",
        "contrast_std": float(stats["std"]),
        "contrast_p95_p05": contrast_p95_p05,
        "mean_intensity": float(stats["mean"]),
        "sharpness_lapvar": float(np.var(lap_values)) if lap_values.size else float("nan"),
        "black_pixel_ratio": float(np.mean(processed <= 5)),
        "white_pixel_ratio": float(np.mean(processed >= 250)),
    }


@lru_cache(maxsize=8192)
def sift_descriptor_snapshot(
    path_str: str,
    target_size: int,
    nfeatures: int,
    blur_ksize: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    original = _load_gray(path_str)
    processed = preprocess_image(
        original,
        PreprocessConfig(target_size=int(target_size), blur_ksize=int(blur_ksize)),
    )
    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    keypoints, descriptors = sift.detectAndCompute(processed, None)
    return keypoints or [], descriptors


def _estimate_sift_inliers(
    kps_a: list[cv2.KeyPoint],
    kps_b: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    *,
    model: str,
    ransac_thresh: float,
) -> tuple[int, np.ndarray | None]:
    if len(matches) < 3:
        return 0, None
    pts_a = np.float32([kps_a[m.queryIdx].pt for m in matches])
    pts_b = np.float32([kps_b[m.trainIdx].pt for m in matches])
    model_key = str(model).strip().lower()
    mask = None
    if model_key == "homography":
        if len(matches) < 8:
            return 0, None
        _, mask = cv2.findHomography(
            pts_a,
            pts_b,
            cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
        )
    elif model_key == "affine_partial_2d":
        _, mask = cv2.estimateAffinePartial2D(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99,
        )
    elif model_key == "affine_full_2d":
        _, mask = cv2.estimateAffine2D(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99,
        )
    else:
        raise ValueError(f"Unsupported SIFT recompute ransac model: {model}")
    if mask is None:
        return 0, None
    flat = mask.ravel().astype(bool)
    return int(np.sum(flat)), flat


def recompute_sift_match_diagnostics(
    path_a: str,
    path_b: str,
    *,
    target_size: int,
    nfeatures: int,
    ratio: float,
    ransac_thresh: float,
    ransac_model: str,
    blur_ksize: int,
) -> dict[str, Any]:
    kps_a, desc_a = sift_descriptor_snapshot(path_a, int(target_size), int(nfeatures), int(blur_ksize))
    kps_b, desc_b = sift_descriptor_snapshot(path_b, int(target_size), int(nfeatures), int(blur_ksize))
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return {
            "sift_knn_query_descriptors": int(0 if desc_a is None else len(desc_a)),
            "sift_knn_pairs_with_2_neighbors": 0,
            "sift_lowe_good_matches": 0,
            "sift_inliers_recomputed": 0,
            "sift_inlier_ratio_recomputed": 0.0,
            "sift_median_good_match_distance": float("nan"),
            "sift_median_inlier_distance": float("nan"),
            "sift_recompute_model": str(ransac_model),
            "sift_recompute_ratio": float(ratio),
            "sift_recompute_ransac_thresh": float(ransac_thresh),
            "sift_recompute_nfeatures": int(nfeatures),
            "sift_recompute_blur_ksize": int(blur_ksize),
        }

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for item in knn:
        if len(item) == 2:
            first, second = item
            if first.distance < float(ratio) * second.distance:
                good.append(first)
    inliers, inlier_mask = _estimate_sift_inliers(
        kps_a,
        kps_b,
        good,
        model=str(ransac_model),
        ransac_thresh=float(ransac_thresh),
    )
    good_distances = np.asarray([match.distance for match in good], dtype=float)
    if inlier_mask is not None and len(inlier_mask) == len(good):
        inlier_distances = good_distances[inlier_mask]
    else:
        inlier_distances = np.asarray([], dtype=float)
    return {
        "sift_knn_query_descriptors": int(len(desc_a)),
        "sift_knn_pairs_with_2_neighbors": int(sum(1 for item in knn if len(item) == 2)),
        "sift_lowe_good_matches": int(len(good)),
        "sift_inliers_recomputed": int(inliers),
        "sift_inlier_ratio_recomputed": float(inliers / len(good)) if good else 0.0,
        "sift_median_good_match_distance": float(np.median(good_distances)) if good_distances.size else float("nan"),
        "sift_median_inlier_distance": float(np.median(inlier_distances)) if inlier_distances.size else float("nan"),
        "sift_recompute_model": str(ransac_model),
        "sift_recompute_ratio": float(ratio),
        "sift_recompute_ransac_thresh": float(ransac_thresh),
        "sift_recompute_nfeatures": int(nfeatures),
        "sift_recompute_blur_ksize": int(blur_ksize),
    }


def _quality_suspect(metrics_a: dict[str, Any], metrics_b: dict[str, Any]) -> bool:
    for metrics in (metrics_a, metrics_b):
        foreground_ratio = _to_float(metrics.get("foreground_ratio"), 0.0)
        contrast = _to_float(metrics.get("contrast_std"), 0.0)
        sharpness = _to_float(metrics.get("sharpness_lapvar"), 0.0)
        if not bool(metrics.get("roi_valid")):
            return True
        if foreground_ratio < 0.02 or foreground_ratio > 0.90:
            return True
        if contrast < 12.0 or sharpness < 20.0:
            return True
    return False


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _load_thresholds(benchmark_dir: Path) -> dict[str, float]:
    path = benchmark_dir / "calibration" / "thresholds_far_1pct_from_val.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration threshold CSV: {path}")
    df = pd.read_csv(path)
    missing = {"method", "threshold"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return {
        str(row["method"]): float(row["threshold"])
        for _, row in df.iterrows()
        if pd.notna(row["threshold"])
    }


def _score_path(benchmark_dir: Path, method: str, pair_set: str) -> Path:
    path = benchmark_dir / f"scores_{method}_{pair_set}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing score CSV: {path}")
    return path


def _load_method_scores(
    benchmark_dir: Path,
    method: str,
    *,
    limit_per_pair_set: int = 0,
) -> pd.DataFrame:
    frames = []
    for pair_set in PAIR_SETS:
        path = _score_path(benchmark_dir, method, pair_set)
        df = pd.read_csv(path)
        df = df.copy()
        df["pair_set"] = pair_set
        df["method"] = method
        df["source_scores_csv"] = str(path)
        if limit_per_pair_set > 0:
            df = df.head(int(limit_per_pair_set)).copy()
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _common_pair_fields(row: pd.Series, threshold: float, target_size: int) -> dict[str, Any]:
    score = _to_float(row.get("score"), 0.0)
    path_a = str(row["path_a"])
    path_b = str(row["path_b"])
    metrics_a = image_quality_metrics(path_a, int(target_size))
    metrics_b = image_quality_metrics(path_b, int(target_size))
    return {
        "method": str(row.get("method", "")),
        "pair_set": str(row.get("pair_set", "")),
        "label": _to_int(row.get("label")),
        "split": str(row.get("split", "")),
        "path_a": path_a,
        "path_b": path_b,
        "score": float(score),
        "threshold_far_1pct": float(threshold),
        "accepted": bool(score >= float(threshold)),
        **_prefix_metrics("a", metrics_a),
        **_prefix_metrics("b", metrics_b),
    }


def _sift_failure_reason(row: dict[str, Any]) -> str:
    if bool(row["accepted"]):
        return "accepted"
    k1 = _to_int(row.get("keypoints_a"))
    k2 = _to_int(row.get("keypoints_b"))
    raw_matches = _to_int(row.get("raw_matches"))
    good_matches = _to_int(row.get("good_matches"))
    inlier_ratio = _to_float(row.get("inlier_ratio"), 0.0)
    score = _to_float(row.get("score"), 0.0)
    preprocessing_suspect = bool(row.get("preprocessing_or_mask_suspect"))
    if k1 == 0 or k2 == 0:
        return "no_keypoints"
    if min(k1, k2) < 32:
        return "too_few_keypoints"
    if raw_matches == 0 or good_matches == 0:
        return "no_matches"
    if good_matches < 8:
        return "too_few_good_matches"
    if inlier_ratio < 0.25:
        return "low_inlier_ratio"
    if score > 0.0 and good_matches >= 8:
        return "low_score_despite_matches"
    if preprocessing_suspect:
        return "preprocessing_or_mask_suspect"
    return "unknown"


def build_sift_diagnostics(
    benchmark_dir: Path,
    thresholds: dict[str, float],
    target_size: int,
    limit: int,
    *,
    recompute_sift_diagnostics: bool = False,
    sift_nfeatures: int = 1500,
    sift_ratio: float = 0.75,
    sift_ransac_thresh: float = 3.0,
    sift_ransac_model: str = "homography",
    sift_blur_ksize: int = 3,
) -> pd.DataFrame:
    threshold = thresholds.get("sift")
    if threshold is None:
        raise ValueError("No calibrated threshold found for method=sift")
    df = _load_method_scores(benchmark_dir, "sift", limit_per_pair_set=limit)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out = _common_pair_fields(row, threshold, target_size)
        k1 = _to_int(row.get("k1"))
        k2 = _to_int(row.get("k2"))
        good_matches = _to_int(row.get("matches"))
        inliers = _to_int(row.get("inliers"))
        raw_matches = k1 if k1 > 0 and k2 > 0 else 0
        inlier_ratio = float(inliers / good_matches) if good_matches > 0 else 0.0
        preprocessing_suspect = _quality_suspect(
            {key[2:]: value for key, value in out.items() if key.startswith("a_")},
            {key[2:]: value for key, value in out.items() if key.startswith("b_")},
        )
        out.update(
            {
                "keypoints_a": k1,
                "keypoints_b": k2,
                "raw_matches": int(raw_matches),
                "raw_matches_source": "derived_from_query_descriptors",
                "good_matches": int(good_matches),
                "inliers": int(inliers),
                "inlier_ratio": float(inlier_ratio),
                "score_zero": bool(_to_float(row.get("score"), 0.0) == 0.0),
                "preprocessing_or_mask_suspect": bool(preprocessing_suspect),
            }
        )
        if recompute_sift_diagnostics:
            out.update(
                recompute_sift_match_diagnostics(
                    str(out["path_a"]),
                    str(out["path_b"]),
                    target_size=int(target_size),
                    nfeatures=int(sift_nfeatures),
                    ratio=float(sift_ratio),
                    ransac_thresh=float(sift_ransac_thresh),
                    ransac_model=str(sift_ransac_model),
                    blur_ksize=int(sift_blur_ksize),
                )
            )
        out["failure_reason"] = _sift_failure_reason(out)
        rows.append(out)
    return pd.DataFrame(rows)


def _minutiae_failure_reason(row: dict[str, Any]) -> str:
    if bool(row["accepted"]):
        return "accepted"
    min_count = min(_to_int(row.get("minutiae_count_a")), _to_int(row.get("minutiae_count_b")))
    matched = _to_int(row.get("matched_minutiae_count"))
    dense = bool(row.get("dense_skeleton"))
    sparse = bool(row.get("sparse_minutiae"))
    score = _to_float(row.get("score"), 0.0)
    if sparse:
        return "sparse_minutiae"
    if dense:
        return "dense_skeleton"
    if matched < 8:
        return "too_few_matches"
    if score < _to_float(row.get("threshold_far_1pct"), 0.0):
        return "low_score"
    if min_count <= 0:
        return "sparse_minutiae"
    return "unknown"


def build_minutiae_diagnostics(
    benchmark_dir: Path,
    thresholds: dict[str, float],
    target_size: int,
    limit: int,
) -> pd.DataFrame:
    threshold = thresholds.get("minutiae")
    if threshold is None:
        raise ValueError("No calibrated threshold found for method=minutiae")
    df = _load_method_scores(benchmark_dir, "minutiae", limit_per_pair_set=limit)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out = _common_pair_fields(row, threshold, target_size)
        flags_a = _split_flags(row.get("extraction_quality_flags_a"))
        flags_b = _split_flags(row.get("extraction_quality_flags_b"))
        all_flags = flags_a | flags_b
        minutiae_a = _to_int(row.get("minutiae_count_a", row.get("minutiae_a")))
        minutiae_b = _to_int(row.get("minutiae_count_b", row.get("minutiae_b")))
        dense = (
            "dense_skeleton" in all_flags
            or _to_float(row.get("skeleton_density_a"), 0.0) > 0.090
            or _to_float(row.get("skeleton_density_b"), 0.0) > 0.090
        )
        sparse = (
            min(minutiae_a, minutiae_b) < 12
            or "no_minutiae" in all_flags
            or "sparse_skeleton" in all_flags
        )
        out.update(
            {
                "minutiae_count_a": minutiae_a,
                "minutiae_count_b": minutiae_b,
                "matched_minutiae_count": _to_int(row.get("matched_minutiae")),
                "tentative_minutiae_count": _to_int(row.get("tentative_minutiae")),
                "dense_skeleton": bool(dense),
                "sparse_minutiae": bool(sparse),
                "skeleton_density_a": _to_float(row.get("skeleton_density_a")),
                "skeleton_density_b": _to_float(row.get("skeleton_density_b")),
                "skeleton_foreground_pixels_a": _to_int(row.get("skeleton_foreground_pixels_a")),
                "skeleton_foreground_pixels_b": _to_int(row.get("skeleton_foreground_pixels_b")),
                "raw_candidate_endings_a": _to_int(row.get("raw_candidate_endings_a")),
                "raw_candidate_endings_b": _to_int(row.get("raw_candidate_endings_b")),
                "raw_candidate_bifurcations_a": _to_int(row.get("raw_candidate_bifurcations_a")),
                "raw_candidate_bifurcations_b": _to_int(row.get("raw_candidate_bifurcations_b")),
                "saturated_by_max_minutiae_a": _bool_value(row.get("saturated_by_max_minutiae_a")),
                "saturated_by_max_minutiae_b": _bool_value(row.get("saturated_by_max_minutiae_b")),
                "extraction_quality_flags_a": ";".join(sorted(flags_a)),
                "extraction_quality_flags_b": ";".join(sorted(flags_b)),
                "raw_alignment_score": _to_float(row.get("raw_alignment_score")),
                "score_multiplier": _to_float(row.get("score_multiplier")),
                "score_component_template_quality": _to_float(row.get("score_component_template_quality")),
                "score_component_ambiguity": _to_float(row.get("score_component_ambiguity")),
                "score_component_transform_plausibility": _to_float(
                    row.get("score_component_transform_plausibility")
                ),
                "transform_angle_deg": _to_float(row.get("transform_angle_deg")),
                "transform_dx": _to_float(row.get("transform_dx")),
                "transform_dy": _to_float(row.get("transform_dy")),
            }
        )
        out["failure_reason"] = _minutiae_failure_reason(out)
        rows.append(out)
    return pd.DataFrame(rows)


def _threshold_for_far(negative_scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    scores = np.asarray(negative_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), 0, float("nan")
    n_negative = int(scores.size)
    for threshold in sorted(float(x) for x in np.unique(scores)):
        false_accepts = int(np.sum(scores >= threshold))
        far = false_accepts / n_negative
        if far <= float(target_far):
            return float(threshold), false_accepts, float(far)
    threshold = math.nextafter(float(np.max(scores)), math.inf)
    return float(threshold), 0, 0.0


def _score_methods_in_dir(benchmark_dir: Path) -> list[str]:
    prefix = "scores_"
    suffix = "_positive_1000.csv"
    methods: list[str] = []
    for path in sorted(benchmark_dir.glob(f"{prefix}*{suffix}")):
        name = path.name
        method = name[len(prefix) : -len(suffix)]
        if (benchmark_dir / f"scores_{method}_negative_1000.csv").exists():
            methods.append(method)
    return methods


def build_threshold_sweep(benchmark_dir: Path, target_fars: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method in _score_methods_in_dir(benchmark_dir):
        positive = pd.read_csv(_score_path(benchmark_dir, method, "positive_1000"))
        negative = pd.read_csv(_score_path(benchmark_dir, method, "negative_1000"))
        pos_scores = pd.to_numeric(positive["score"], errors="coerce").to_numpy(dtype=float)
        neg_scores = pd.to_numeric(negative["score"], errors="coerce").to_numpy(dtype=float)
        pos_scores = pos_scores[np.isfinite(pos_scores)]
        neg_scores = neg_scores[np.isfinite(neg_scores)]
        n_pos = int(pos_scores.size)
        n_neg = int(neg_scores.size)
        for target_far in target_fars:
            threshold, false_accepts, actual_far = _threshold_for_far(neg_scores, float(target_far))
            true_accepts = int(np.sum(pos_scores >= threshold)) if math.isfinite(threshold) else 0
            false_rejects = int(n_pos - true_accepts)
            true_rejects = int(n_neg - false_accepts)
            tar = float(true_accepts / n_pos) if n_pos else float("nan")
            far = float(false_accepts / n_neg) if n_neg else float("nan")
            rows.append(
                {
                    "method": method,
                    "threshold": float(threshold),
                    "target_far": float(target_far),
                    "actual_far": float(actual_far if math.isfinite(actual_far) else far),
                    "tar": tar,
                    "frr": float(1.0 - tar) if math.isfinite(tar) else float("nan"),
                    "true_accepts": int(true_accepts),
                    "false_rejects": int(false_rejects),
                    "false_accepts": int(false_accepts),
                    "true_rejects": int(true_rejects),
                }
            )
    return pd.DataFrame(rows)


def _top_rows(df: pd.DataFrame, *, label: int, accepted: bool, ascending: bool, top_n: int) -> pd.DataFrame:
    subset = df[(df["label"].astype(int) == int(label)) & (df["accepted"].astype(bool) == bool(accepted))].copy()
    if subset.empty:
        return subset
    subset["score"] = pd.to_numeric(subset["score"], errors="coerce")
    subset = subset.sort_values(["method", "score"], ascending=[True, ascending])
    return subset.groupby("method", group_keys=False).head(int(top_n)).reset_index(drop=True)


def _summary_metric(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in df.columns:
        return {"n": 0, "median": None, "mean": None}
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"n": 0, "median": None, "mean": None}
    return {"n": int(values.size), "median": float(np.median(values)), "mean": float(np.mean(values))}


def _reason_counts_markdown(title: str, df: pd.DataFrame) -> list[str]:
    rows = [f"## {title}", "", "| method | failure_reason | count | rate |", "| --- | --- | ---: | ---: |"]
    for method, method_df in df.groupby("method"):
        rejected_positive = method_df[
            (method_df["label"].astype(int) == 1) & (~method_df["accepted"].astype(bool))
        ]
        denom = max(int(len(rejected_positive)), 1)
        counts = rejected_positive["failure_reason"].value_counts()
        if counts.empty:
            rows.append(f"| {method} | none | 0 | 0 |")
        for reason, count in counts.items():
            rows.append(f"| {method} | {reason} | {int(count)} | {count / denom:.3f} |")
    return rows


def _render_sweep_table(sweep: pd.DataFrame) -> list[str]:
    rows = [
        "## Threshold Sweep",
        "",
        "| method | target FAR | threshold | actual FAR | TAR | FRR | TA | FR | FA | TR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in sweep.sort_values(["method", "target_far"]).iterrows():
        rows.append(
            "| {method} | {target_far:.3f} | {threshold:.6g} | {actual_far:.3f} | "
            "{tar:.3f} | {frr:.3f} | {true_accepts} | {false_rejects} | "
            "{false_accepts} | {true_rejects} |".format(**row.to_dict())
        )
    return rows


def _fmt_metric(stats: dict[str, Any], key: str = "median") -> str:
    value = stats.get(key)
    return "n/a" if value is None else f"{float(value):.3f}"


def render_summary(
    *,
    benchmark_dir: Path,
    thresholds: dict[str, float],
    sift: pd.DataFrame,
    minutiae: pd.DataFrame,
    sweep: pd.DataFrame,
) -> str:
    combined = pd.concat([sift, minutiae], ignore_index=True, sort=False)
    lines = [
        "# NIST SD300B Plain-vs-Roll Pair Diagnostics",
        "",
        f"Source benchmark folder: `{benchmark_dir}`",
        "",
        "## Calibrated FAR <= 1% Operating Point",
        "",
        "| method | threshold | positive accepts | positive rejects | negative false accepts | negative true rejects | TAR | FAR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, df in (("sift", sift), ("minutiae", minutiae)):
        positives = df[df["label"].astype(int) == 1]
        negatives = df[df["label"].astype(int) == 0]
        ta = int(positives["accepted"].sum())
        fr = int(len(positives) - ta)
        fa = int(negatives["accepted"].sum())
        tr = int(len(negatives) - fa)
        tar = ta / max(len(positives), 1)
        far = fa / max(len(negatives), 1)
        lines.append(
            f"| {method} | {thresholds[method]:.6g} | {ta} | {fr} | {fa} | {tr} | {tar:.3f} | {far:.3f} |"
        )

    lines.extend([""])
    lines.extend(_reason_counts_markdown("Positive Reject Failure Reasons", combined))

    sift_pos_rejects = sift[(sift["label"].astype(int) == 1) & (~sift["accepted"].astype(bool))]
    sift_pos_accepts = sift[(sift["label"].astype(int) == 1) & (sift["accepted"].astype(bool))]
    min_pos_rejects = minutiae[(minutiae["label"].astype(int) == 1) & (~minutiae["accepted"].astype(bool))]
    min_pos_accepts = minutiae[(minutiae["label"].astype(int) == 1) & (minutiae["accepted"].astype(bool))]

    lines.extend(
        [
            "",
            "## SIFT Matching Signals",
            "",
            "| group | median good matches | median inliers | median inlier ratio | median score |",
            "| --- | ---: | ---: | ---: | ---: |",
            (
                f"| accepted positives | {_fmt_metric(_summary_metric(sift_pos_accepts, 'good_matches'))} | "
                f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'inliers'))} | "
                f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'inlier_ratio'))} | "
                f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'score'))} |"
            ),
            (
                f"| rejected positives | {_fmt_metric(_summary_metric(sift_pos_rejects, 'good_matches'))} | "
                f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'inliers'))} | "
                f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'inlier_ratio'))} | "
                f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'score'))} |"
            ),
        ]
    )
    if "sift_knn_pairs_with_2_neighbors" in sift.columns:
        lines.extend(
            [
                "",
                "## Recomputed SIFT KNN Diagnostics",
                "",
                "| group | median KNN query descriptors | median KNN pairs with 2 neighbors | median Lowe-ratio matches | median inliers | median inlier ratio | median good-match distance | median inlier distance |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| accepted positives | {_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_knn_query_descriptors'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_knn_pairs_with_2_neighbors'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_lowe_good_matches'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_inliers_recomputed'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_inlier_ratio_recomputed'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_median_good_match_distance'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_accepts, 'sift_median_inlier_distance'))} |"
                ),
                (
                    f"| rejected positives | {_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_knn_query_descriptors'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_knn_pairs_with_2_neighbors'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_lowe_good_matches'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_inliers_recomputed'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_inlier_ratio_recomputed'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_median_good_match_distance'))} | "
                    f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'sift_median_inlier_distance'))} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Note: the legacy `raw_matches` column in `pair_diagnostics_sift.csv` is derived from query keypoints/descriptors, not from raw BFMatcher KNN output. Use the `sift_knn_*` columns when `--recompute_sift_diagnostics` is enabled.",
            "",
            "## Minutiae Matching Signals",
            "",
            "| group | median matched minutiae | median minutiae A | median minutiae B | dense skeleton rate | median score |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, df in (("accepted positives", min_pos_accepts), ("rejected positives", min_pos_rejects)):
        dense_rate = float(df["dense_skeleton"].mean()) if len(df) else float("nan")
        lines.append(
            f"| {name} | {_fmt_metric(_summary_metric(df, 'matched_minutiae_count'))} | "
            f"{_fmt_metric(_summary_metric(df, 'minutiae_count_a'))} | "
            f"{_fmt_metric(_summary_metric(df, 'minutiae_count_b'))} | "
            f"{dense_rate:.3f} | {_fmt_metric(_summary_metric(df, 'score'))} |"
        )

    low_score_match_count = int((sift_pos_rejects["failure_reason"] == "low_score_despite_matches").sum())
    few_good_match_count = int((sift_pos_rejects["failure_reason"] == "too_few_good_matches").sum())
    low_inlier_count = int((sift_pos_rejects["failure_reason"] == "low_inlier_ratio").sum())
    dense_count = int((min_pos_rejects["failure_reason"] == "dense_skeleton").sum())
    sparse_count = int((min_pos_rejects["failure_reason"] == "sparse_minutiae").sum())
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"- SIFT rejects are mostly not keypoint-starved: rejected positives still have a median of "
                f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'keypoints_a'))}/"
                f"{_fmt_metric(_summary_metric(sift_pos_rejects, 'keypoints_b'))} keypoints. "
                f"The main SIFT reject buckets are {low_score_match_count} `low_score_despite_matches`, "
                f"{few_good_match_count} `too_few_good_matches`, and {low_inlier_count} `low_inlier_ratio`."
            ),
            (
                "- That points toward plain-vs-roll overlap, distortion, or descriptor/geometry instability "
                "rather than label inversion or empty feature extraction."
            ),
            (
                f"- Minutiae failures are dominated by extraction quality and matching sparsity signals: "
                f"{dense_count} rejected positives are `dense_skeleton`, and {sparse_count} are `sparse_minutiae`."
            ),
            "- Relaxing thresholds improves TAR, but the sweep shows the tradeoff directly against false accepts.",
            "",
        ]
    )
    lines.extend(_render_sweep_table(sweep))
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    outdir: Path,
    benchmark_dir: Path,
    thresholds: dict[str, float],
    sift: pd.DataFrame,
    minutiae: pd.DataFrame,
    sweep: pd.DataFrame,
    top_n: int,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat([sift, minutiae], ignore_index=True, sort=False)
    positive_failures = _top_rows(combined, label=1, accepted=False, ascending=True, top_n=top_n)
    positive_successes = _top_rows(combined, label=1, accepted=True, ascending=False, top_n=top_n)
    negative_false_accepts = _top_rows(combined, label=0, accepted=True, ascending=False, top_n=top_n)

    paths = {
        "pair_diagnostics_sift": outdir / "pair_diagnostics_sift.csv",
        "pair_diagnostics_minutiae": outdir / "pair_diagnostics_minutiae.csv",
        "diagnostics_summary": outdir / "diagnostics_summary.md",
        "threshold_sweep_summary": outdir / "threshold_sweep_summary.csv",
        "top_positive_failures": outdir / "top_positive_failures.csv",
        "top_positive_successes": outdir / "top_positive_successes.csv",
        "top_negative_false_accepts": outdir / "top_negative_false_accepts.csv",
    }
    sift.to_csv(paths["pair_diagnostics_sift"], index=False)
    minutiae.to_csv(paths["pair_diagnostics_minutiae"], index=False)
    sweep.to_csv(paths["threshold_sweep_summary"], index=False)
    positive_failures.to_csv(paths["top_positive_failures"], index=False)
    positive_successes.to_csv(paths["top_positive_successes"], index=False)
    negative_false_accepts.to_csv(paths["top_negative_false_accepts"], index=False)
    paths["diagnostics_summary"].write_text(
        render_summary(
            benchmark_dir=benchmark_dir,
            thresholds=thresholds,
            sift=sift,
            minutiae=minutiae,
            sweep=sweep,
        ),
        encoding="utf-8",
    )
    return paths


def run_analysis(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    target_size: int = 512,
    target_fars: tuple[float, ...] = TARGET_FARS,
    top_n: int = 25,
    limit_per_pair_set: int = 0,
    recompute_sift_diagnostics: bool = False,
    sift_nfeatures: int = 1500,
    sift_ratio: float = 0.75,
    sift_ransac_thresh: float = 3.0,
    sift_ransac_model: str = "homography",
    sift_blur_ksize: int = 3,
) -> dict[str, Path]:
    benchmark = parse_file_uri(benchmark_dir)
    output = parse_file_uri(outdir)
    thresholds = _load_thresholds(benchmark)
    sift = build_sift_diagnostics(
        benchmark,
        thresholds,
        int(target_size),
        int(limit_per_pair_set),
        recompute_sift_diagnostics=bool(recompute_sift_diagnostics),
        sift_nfeatures=int(sift_nfeatures),
        sift_ratio=float(sift_ratio),
        sift_ransac_thresh=float(sift_ransac_thresh),
        sift_ransac_model=str(sift_ransac_model),
        sift_blur_ksize=int(sift_blur_ksize),
    )
    minutiae = build_minutiae_diagnostics(benchmark, thresholds, int(target_size), int(limit_per_pair_set))
    sweep = build_threshold_sweep(benchmark, tuple(float(x) for x in target_fars))
    return write_outputs(
        outdir=output,
        benchmark_dir=benchmark,
        thresholds=thresholds,
        sift=sift,
        minutiae=minutiae,
        sweep=sweep,
        top_n=int(top_n),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose NIST SD300B plain-vs-roll pair failures.")
    parser.add_argument("--benchmark_dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--target_size", type=int, default=512)
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument(
        "--limit_per_pair_set",
        type=int,
        default=0,
        help="Optional smoke-test limit applied separately to positive_1000 and negative_1000.",
    )
    parser.add_argument("--recompute_sift_diagnostics", action="store_true")
    parser.add_argument("--sift_nfeatures", type=int, default=1500)
    parser.add_argument("--sift_ratio", type=float, default=0.75)
    parser.add_argument("--sift_ransac_thresh", type=float, default=3.0)
    parser.add_argument(
        "--sift_ransac_model",
        type=str,
        default="homography",
        choices=["homography", "affine_partial_2d", "affine_full_2d"],
    )
    parser.add_argument("--sift_blur_ksize", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = run_analysis(
        args.benchmark_dir,
        args.outdir,
        target_size=int(args.target_size),
        target_fars=tuple(float(x) for x in args.target_far),
        top_n=int(args.top_n),
        limit_per_pair_set=int(args.limit_per_pair_set),
        recompute_sift_diagnostics=bool(args.recompute_sift_diagnostics),
        sift_nfeatures=int(args.sift_nfeatures),
        sift_ratio=float(args.sift_ratio),
        sift_ransac_thresh=float(args.sift_ransac_thresh),
        sift_ransac_model=str(args.sift_ransac_model),
        sift_blur_ksize=int(args.sift_blur_ksize),
    )
    print("Wrote diagnostics:")
    for path in paths.values():
        print(f"  {path}")
    print("Summary:")
    print(f"  {paths['diagnostics_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
