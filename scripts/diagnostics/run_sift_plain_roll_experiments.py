from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.preprocess.preprocess import PreprocessConfig, preprocess_image  # noqa: E402


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
TARGET_SIZES = (512, 768, 1024)
NFEATURES = (1500, 3000, 5000)
RATIOS = (0.70, 0.75, 0.80, 0.85)
RANSAC_MODELS = ("homography", "affine_partial_2d", "affine_full_2d")
RANSAC_THRESHOLDS = (3.0, 4.0, 6.0)
BLUR_KSIZES = (0, 3)
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)
SCORE_VARIANTS = (
    "current_score",
    "inliers",
    "inliers_over_matches",
    "inliers_times_inlier_ratio",
    "inlier_ratio_times_log1p_matches",
    "inliers_times_inlier_ratio_times_log1p_matches",
    "inliers_over_sqrt_matches",
)
FOCUSED_CONFIGS = (
    {
        "config_name": "affine768_nf3000_ratio075_ransac3_blur0",
        "target_size": 768,
        "nfeatures": 3000,
        "blur_ksize": 0,
        "ratio": 0.75,
        "ransac_model": "affine_full_2d",
        "ransac_thresh": 3.0,
    },
    {
        "config_name": "affine768_nf3000_ratio075_ransac6_blur0",
        "target_size": 768,
        "nfeatures": 3000,
        "blur_ksize": 0,
        "ratio": 0.75,
        "ransac_model": "affine_full_2d",
        "ransac_thresh": 6.0,
    },
    {
        "config_name": "affine768_nf5000_ratio075_ransac6_blur0",
        "target_size": 768,
        "nfeatures": 5000,
        "blur_ksize": 0,
        "ratio": 0.75,
        "ransac_model": "affine_full_2d",
        "ransac_thresh": 6.0,
    },
    {
        "config_name": "affine768_nf1500_ratio085_ransac6_blur3",
        "target_size": 768,
        "nfeatures": 1500,
        "blur_ksize": 3,
        "ratio": 0.85,
        "ransac_model": "affine_full_2d",
        "ransac_thresh": 6.0,
    },
    {
        "config_name": "baseline_current_approx",
        "target_size": 512,
        "nfeatures": 1500,
        "blur_ksize": 3,
        "ratio": 0.75,
        "ransac_model": "homography",
        "ransac_thresh": 4.0,
    },
)


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load_gray(path_str: str) -> np.ndarray:
    img = cv2.imread(str(parse_file_uri(path_str)), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path_str}")
    return img


@lru_cache(maxsize=32768)
def _snapshot(path_str: str, target_size: int, nfeatures: int, blur_ksize: int) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    gray = _load_gray(path_str)
    processed = preprocess_image(
        gray,
        PreprocessConfig(target_size=int(target_size), blur_ksize=int(blur_ksize)),
    )
    sift = cv2.SIFT_create(nfeatures=int(nfeatures))
    keypoints, descriptors = sift.detectAndCompute(processed, None)
    return keypoints or [], descriptors


@lru_cache(maxsize=32768)
def _knn_records(
    path_a: str,
    path_b: str,
    target_size: int,
    nfeatures: int,
    blur_ksize: int,
) -> tuple[int, int, int, int, tuple[tuple[int, int, float, float], ...]]:
    kps_a, desc_a = _snapshot(path_a, int(target_size), int(nfeatures), int(blur_ksize))
    kps_b, desc_b = _snapshot(path_b, int(target_size), int(nfeatures), int(blur_ksize))
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return len(kps_a), len(kps_b), 0 if desc_a is None else len(desc_a), 0, ()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    records: list[tuple[int, int, float, float]] = []
    for item in knn:
        if len(item) == 2:
            first, second = item
            records.append((int(first.queryIdx), int(first.trainIdx), float(first.distance), float(second.distance)))
    return len(kps_a), len(kps_b), len(desc_a), int(len(records)), tuple(records)


def _estimate_inliers(
    path_a: str,
    path_b: str,
    records: list[tuple[int, int, float, float]],
    *,
    target_size: int,
    nfeatures: int,
    blur_ksize: int,
    ransac_model: str,
    ransac_thresh: float,
) -> int:
    min_matches = 8 if ransac_model == "homography" else 3
    if len(records) < min_matches:
        return 0
    kps_a, _ = _snapshot(path_a, int(target_size), int(nfeatures), int(blur_ksize))
    kps_b, _ = _snapshot(path_b, int(target_size), int(nfeatures), int(blur_ksize))
    pts_a = np.float32([kps_a[q].pt for q, _t, _d1, _d2 in records])
    pts_b = np.float32([kps_b[t].pt for _q, t, _d1, _d2 in records])
    mask = None
    if ransac_model == "homography":
        _, mask = cv2.findHomography(
            pts_a,
            pts_b,
            cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
        )
    elif ransac_model == "affine_partial_2d":
        _, mask = cv2.estimateAffinePartial2D(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99,
        )
    elif ransac_model == "affine_full_2d":
        _, mask = cv2.estimateAffine2D(
            pts_a,
            pts_b,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99,
        )
    else:
        raise ValueError(f"Unsupported ransac_model={ransac_model!r}")
    return int(mask.ravel().sum()) if mask is not None else 0


def _score_values(pair_df: pd.DataFrame, variant: str) -> np.ndarray:
    inliers = pd.to_numeric(pair_df["inliers"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    matches = pd.to_numeric(pair_df["matches"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    k1 = pd.to_numeric(pair_df["k1"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    k2 = pd.to_numeric(pair_df["k2"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    denom_matches = np.maximum(matches, 1.0)
    ratio = np.divide(inliers, denom_matches, out=np.zeros_like(inliers), where=denom_matches > 0)
    if variant == "current_score":
        denom = np.maximum(np.minimum(k1, k2), 1.0)
        return np.divide(inliers, denom, out=np.zeros_like(inliers), where=denom > 0)
    if variant == "inliers":
        return inliers
    if variant == "inliers_over_matches":
        return ratio
    if variant == "inliers_times_inlier_ratio":
        return inliers * ratio
    if variant == "inlier_ratio_times_log1p_matches":
        return ratio * np.log1p(np.maximum(matches, 0.0))
    if variant == "inliers_times_inlier_ratio_times_log1p_matches":
        return inliers * ratio * np.log1p(np.maximum(matches, 0.0))
    if variant == "inliers_over_sqrt_matches":
        return np.divide(inliers, np.sqrt(denom_matches), out=np.zeros_like(inliers), where=denom_matches > 0)
    raise ValueError(f"Unknown score variant: {variant}")


def _auc_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
    except ValueError:
        return float("nan"), float("nan")
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr))) if fpr.size else 0
    return auc, float((fpr[idx] + fnr[idx]) / 2.0)


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


def _load_selected_pairs(benchmark_dir: Path, limit_per_pair_set: int) -> pd.DataFrame:
    frames = []
    for pair_set in ("positive_1000", "negative_1000"):
        path = benchmark_dir / "selected_pairs" / f"{pair_set}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing selected pairs: {path}")
        df = pd.read_csv(path)
        if int(limit_per_pair_set) > 0:
            df = df.head(int(limit_per_pair_set)).copy()
        df["pair_set"] = pair_set
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _config_id(config: dict[str, Any]) -> str:
    payload = "|".join(f"{key}={config[key]}" for key in sorted(config))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _iter_configs(
    *,
    target_sizes: tuple[int, ...],
    nfeatures_values: tuple[int, ...],
    ratios: tuple[float, ...],
    ransac_models: tuple[str, ...],
    ransac_thresholds: tuple[float, ...],
    blur_ksizes: tuple[int, ...],
    max_configs: int = 0,
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for target_size in target_sizes:
        for nfeatures in nfeatures_values:
            for blur_ksize in blur_ksizes:
                for ratio in ratios:
                    for ransac_model in ransac_models:
                        for ransac_thresh in ransac_thresholds:
                            config = {
                                "target_size": int(target_size),
                                "nfeatures": int(nfeatures),
                                "blur_ksize": int(blur_ksize),
                                "ratio": float(ratio),
                                "ransac_model": str(ransac_model),
                                "ransac_thresh": float(ransac_thresh),
                            }
                            config["config_id"] = _config_id(config)
                            configs.append(config)
                            if int(max_configs) > 0 and len(configs) >= int(max_configs):
                                return configs
    return configs


def _evaluate_config(pairs: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair_index, row in pairs.reset_index(drop=True).iterrows():
        path_a = str(row["path_a"])
        path_b = str(row["path_b"])
        k1, k2, knn_query, knn_pairs_2, records = _knn_records(
            path_a,
            path_b,
            int(config["target_size"]),
            int(config["nfeatures"]),
            int(config["blur_ksize"]),
        )
        good = [
            item
            for item in records
            if float(item[2]) < float(config["ratio"]) * float(item[3])
        ]
        inliers = _estimate_inliers(
            path_a,
            path_b,
            list(good),
            target_size=int(config["target_size"]),
            nfeatures=int(config["nfeatures"]),
            blur_ksize=int(config["blur_ksize"]),
            ransac_model=str(config["ransac_model"]),
            ransac_thresh=float(config["ransac_thresh"]),
        )
        rows.append(
            {
                "pair_index": int(pair_index),
                "pair_set": str(row.get("pair_set", "")),
                "label": int(row["label"]),
                "split": str(row.get("split", "")),
                "path_a": path_a,
                "path_b": path_b,
                "k1": int(k1),
                "k2": int(k2),
                "knn_query_descriptors": int(knn_query),
                "knn_pairs_with_2_neighbors": int(knn_pairs_2),
                "matches": int(len(good)),
                "inliers": int(inliers),
            }
        )
    return pd.DataFrame(rows)


def _summarize_config(
    pair_scores: pd.DataFrame,
    config: dict[str, Any],
    *,
    score_variants: tuple[str, ...],
    target_fars: tuple[float, ...],
    limit_per_pair_set: int,
    elapsed_s: float,
) -> list[dict[str, Any]]:
    labels = pd.to_numeric(pair_scores["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    rows: list[dict[str, Any]] = []
    for variant in score_variants:
        values = _score_values(pair_scores, variant)
        auc, eer = _auc_eer(labels, values)
        pos_scores = values[labels == 1]
        neg_scores = values[labels == 0]
        n_pos = int(np.isfinite(pos_scores).sum())
        n_neg = int(np.isfinite(neg_scores).sum())
        for target_far in target_fars:
            threshold, false_accepts, actual_far = _threshold_for_far(neg_scores, float(target_far))
            true_accepts = int(np.sum(pos_scores[np.isfinite(pos_scores)] >= threshold)) if math.isfinite(threshold) else 0
            tar = float(true_accepts / n_pos) if n_pos else float("nan")
            rows.append(
                {
                    "config_id": config["config_id"],
                    "target_size": int(config["target_size"]),
                    "nfeatures": int(config["nfeatures"]),
                    "blur_ksize": int(config["blur_ksize"]),
                    "ratio": float(config["ratio"]),
                    "ransac_model": str(config["ransac_model"]),
                    "ransac_thresh": float(config["ransac_thresh"]),
                    "score_variant": variant,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "actual_far": float(actual_far),
                    "tar": float(tar),
                    "true_accepts": int(true_accepts),
                    "false_accepts": int(false_accepts),
                    "n_positive": int(n_pos),
                    "n_negative": int(n_neg),
                    "auc": float(auc),
                    "eer": float(eer),
                    "median_matches": float(np.median(pair_scores["matches"])) if len(pair_scores) else float("nan"),
                    "median_inliers": float(np.median(pair_scores["inliers"])) if len(pair_scores) else float("nan"),
                    "mean_matches": float(np.mean(pair_scores["matches"])) if len(pair_scores) else float("nan"),
                    "mean_inliers": float(np.mean(pair_scores["inliers"])) if len(pair_scores) else float("nan"),
                    "limit_per_pair_set": int(limit_per_pair_set),
                    "elapsed_s_for_config": float(elapsed_s),
                    "exploratory_same_selected_set": True,
                }
            )
    return rows


def _operating_counts(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
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
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    return {
        "tar": float(ta / n_pos) if n_pos else float("nan"),
        "far": float(fa / n_neg) if n_neg else float("nan"),
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_pos,
        "n_negative": n_neg,
    }


def _median_signal(df: pd.DataFrame, mask: np.ndarray, column: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df.loc[mask, column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _false_accept_examples(
    pair_scores: pd.DataFrame,
    values: np.ndarray,
    mask: np.ndarray,
    *,
    top_n: int = 5,
) -> str:
    subset = pair_scores.loc[mask, ["path_a", "path_b"]].copy()
    subset["score_value"] = np.asarray(values, dtype=float)[mask]
    subset = subset.sort_values("score_value", ascending=False).head(int(top_n))
    examples = []
    for _, row in subset.iterrows():
        examples.append(f"{row['score_value']:.6g}: {row['path_a']} -> {row['path_b']}")
    return " ; ".join(examples)


def _summarize_official_config(
    pair_scores: pd.DataFrame,
    config: dict[str, Any],
    *,
    score_variants: tuple[str, ...],
    target_fars: tuple[float, ...],
    elapsed_s: float,
) -> list[dict[str, Any]]:
    labels = pd.to_numeric(pair_scores["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    splits = pair_scores["split"].astype(str).str.strip().str.lower().to_numpy()
    val_mask = splits == "val"
    test_mask = splits == "test"
    rows: list[dict[str, Any]] = []
    for variant in score_variants:
        values = _score_values(pair_scores, variant)
        val_labels = labels[val_mask]
        val_scores = values[val_mask]
        test_labels = labels[test_mask]
        test_scores = values[test_mask]
        val_negative_scores = val_scores[val_labels == 0]
        for target_far in target_fars:
            threshold, calibration_false_accepts, calibration_far = _threshold_for_far(
                val_negative_scores,
                float(target_far),
            )
            val = _operating_counts(val_labels, val_scores, threshold)
            test = _operating_counts(test_labels, test_scores, threshold)
            accepted = np.asarray(values, dtype=float) >= float(threshold) if math.isfinite(threshold) else np.zeros(
                len(pair_scores),
                dtype=bool,
            )
            test_positive = test_mask & (labels == 1)
            test_negative = test_mask & (labels == 0)
            accepted_test_positive = test_positive & accepted
            rejected_test_positive = test_positive & (~accepted)
            false_accepted_test_negative = test_negative & accepted
            rows.append(
                {
                    "config_id": config["config_id"],
                    "config_name": config["config_name"],
                    "target_size": int(config["target_size"]),
                    "nfeatures": int(config["nfeatures"]),
                    "blur_ksize": int(config["blur_ksize"]),
                    "ratio": float(config["ratio"]),
                    "ransac_model": str(config["ransac_model"]),
                    "ransac_thresh": float(config["ransac_thresh"]),
                    "score_variant": variant,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "calibration_false_accepts": int(calibration_false_accepts),
                    "calibration_far": float(calibration_far),
                    "val_far": float(val["far"]),
                    "val_tar": float(val["tar"]),
                    "test_far": float(test["far"]),
                    "test_tar": float(test["tar"]),
                    "test_ta": int(test["ta"]),
                    "test_fr": int(test["fr"]),
                    "test_fa": int(test["fa"]),
                    "test_tr": int(test["tr"]),
                    "n_val_positive": int(val["n_positive"]),
                    "n_val_negative": int(val["n_negative"]),
                    "n_test_positive": int(test["n_positive"]),
                    "n_test_negative": int(test["n_negative"]),
                    "median_good_matches_accepted_test_positive": _median_signal(
                        pair_scores,
                        accepted_test_positive,
                        "matches",
                    ),
                    "median_inliers_accepted_test_positive": _median_signal(
                        pair_scores,
                        accepted_test_positive,
                        "inliers",
                    ),
                    "median_good_matches_rejected_test_positive": _median_signal(
                        pair_scores,
                        rejected_test_positive,
                        "matches",
                    ),
                    "median_inliers_rejected_test_positive": _median_signal(
                        pair_scores,
                        rejected_test_positive,
                        "inliers",
                    ),
                    "median_good_matches_false_accepted_test_negative": _median_signal(
                        pair_scores,
                        false_accepted_test_negative,
                        "matches",
                    ),
                    "median_inliers_false_accepted_test_negative": _median_signal(
                        pair_scores,
                        false_accepted_test_negative,
                        "inliers",
                    ),
                    "false_accept_examples": _false_accept_examples(
                        pair_scores,
                        values,
                        false_accepted_test_negative,
                    ),
                    "elapsed_s_for_config": float(elapsed_s),
                    "official_val_test": True,
                }
            )
    return rows


def _safe_output_paths(outdir: Path, stem: str, *, overwrite: bool) -> tuple[Path, Path]:
    csv_path = outdir / f"{stem}.csv"
    md_path = outdir / f"{stem}.md"
    if overwrite or (not csv_path.exists() and not md_path.exists()):
        return csv_path, md_path
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return outdir / f"{stem}_{stamp}.csv", outdir / f"{stem}_{stamp}.md"


def _render_markdown(matrix: pd.DataFrame, *, benchmark_dir: Path, n_configs: int, n_pairs: int) -> str:
    mode = "smoke" if int(matrix["limit_per_pair_set"].max()) > 0 else "full"
    lines = [
        "# SIFT Plain-vs-Roll Experiment Matrix",
        "",
        f"Source benchmark folder: `{benchmark_dir}`",
        f"Run mode: `{mode}`",
        f"Evaluated configs: {int(n_configs)}",
        f"Pairs per config: {int(n_pairs)}",
        "",
        "This is diagnostics/research only. It does not change official benchmark defaults or score semantics.",
        "Thresholds and TAR/FAR tradeoffs are exploratory because they use the selected pair set directly.",
        "",
        "## Best At FAR Targets",
        "",
        "| target FAR | rank | config | score variant | TAR | actual FAR | AUC | threshold | target size | nfeatures | ratio | model | ransac | blur | TA | FA |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for target_far in (0.01, 0.05):
        sub = matrix[np.isclose(matrix["target_far"].astype(float), target_far)].copy()
        sub = sub.sort_values(["tar", "actual_far", "auc", "median_inliers"], ascending=[False, True, False, False]).head(15)
        for rank, (_, row) in enumerate(sub.iterrows(), start=1):
            lines.append(
                f"| {target_far:.3f} | {rank} | {row['config_id']} | {row['score_variant']} | {row['tar']:.3f} | "
                f"{row['actual_far']:.3f} | {row['auc']:.4f} | {row['threshold']:.6g} | {int(row['target_size'])} | "
                f"{int(row['nfeatures'])} | {row['ratio']:.2f} | {row['ransac_model']} | {row['ransac_thresh']:.1f} | "
                f"{int(row['blur_ksize'])} | {int(row['true_accepts'])} | {int(row['false_accepts'])} |"
            )
    return "\n".join(lines) + "\n"


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _recommendation_text(matrix: pd.DataFrame) -> list[str]:
    one_pct = matrix[np.isclose(matrix["target_far"].astype(float), 0.01)].copy()
    if one_pct.empty:
        return ["Recommendation: no `sift_plain_roll_v2` decision can be made because no 1% FAR rows were produced."]
    baseline_rows = one_pct[
        (one_pct["config_name"] == "baseline_current_approx") & (one_pct["score_variant"] == "current_score")
    ]
    if baseline_rows.empty:
        return ["Recommendation: no `sift_plain_roll_v2` decision can be made because the baseline row is missing."]
    baseline = baseline_rows.iloc[0]
    baseline_tar = float(baseline["test_tar"])
    eligible = one_pct[
        (one_pct["config_name"] != "baseline_current_approx")
        & (one_pct["test_far"].astype(float) <= 0.01 + 1e-12)
        & (one_pct["test_tar"].astype(float) >= baseline_tar + 0.05 - 1e-12)
    ].copy()
    if eligible.empty:
        best = one_pct[one_pct["test_far"].astype(float) <= 0.01 + 1e-12].copy()
        best = best.sort_values(["test_tar", "test_far"], ascending=[False, True]).head(1)
        if best.empty:
            return [
                (
                    "Recommendation: do not promote a `sift_plain_roll_v2` experimental candidate. "
                    "No focused SIFT row stayed at test FAR <= 1%."
                )
            ]
        row = best.iloc[0]
        delta = float(row["test_tar"]) - baseline_tar
        return [
            (
                "Recommendation: do not promote a `sift_plain_roll_v2` experimental candidate yet. "
                f"The best FAR<=1% focused row is `{row['config_name']}` / `{row['score_variant']}` "
                f"with test TAR {_fmt(row['test_tar'])} versus baseline {_fmt(baseline_tar)} "
                f"(delta {delta:+.3f}), below the +0.050 criterion."
            )
        ]
    eligible = eligible.sort_values(["test_tar", "test_far", "val_tar"], ascending=[False, True, False])
    row = eligible.iloc[0]
    delta = float(row["test_tar"]) - baseline_tar
    return [
        (
            "Recommendation: promote a `sift_plain_roll_v2` experimental candidate. "
            f"`{row['config_name']}` / `{row['score_variant']}` reaches test TAR {_fmt(row['test_tar'])} "
            f"at test FAR {_fmt(row['test_far'])}, improving over the baseline current-score TAR "
            f"{_fmt(baseline_tar)} by {delta:+.3f}."
        )
    ]


def _render_focused_markdown(matrix: pd.DataFrame, *, benchmark_dir: Path, n_configs: int, n_pairs: int) -> str:
    lines = [
        "# Focused SIFT Val/Test Experiments",
        "",
        f"Source benchmark folder: `{benchmark_dir}`",
        f"Evaluated configs: {int(n_configs)}",
        f"Pairs per config: {int(n_pairs)}",
        "",
        "Thresholds are calibrated only on original `val` negatives. Test metrics use only original `test` rows.",
        "",
        "## 1% FAR Ranking",
        "",
        "| rank | config | variant | threshold | val FAR | val TAR | test FAR | test TAR | TA | FR | FA | TR | runtime s | med accepted pos matches/inliers | med rejected pos matches/inliers | med false-accepted neg matches/inliers |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    one_pct = matrix[np.isclose(matrix["target_far"].astype(float), 0.01)].copy()
    one_pct = one_pct.sort_values(["test_tar", "test_far", "val_tar"], ascending=[False, True, False])
    for rank, (_, row) in enumerate(one_pct.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['config_name']} | {row['score_variant']} | {_fmt(row['threshold'], 6)} | "
            f"{_fmt(row['val_far'])} | {_fmt(row['val_tar'])} | {_fmt(row['test_far'])} | "
            f"{_fmt(row['test_tar'])} | {int(row['test_ta'])} | {int(row['test_fr'])} | "
            f"{int(row['test_fa'])} | {int(row['test_tr'])} | {_fmt(row['elapsed_s_for_config'], 1)} | "
            f"{_fmt(row['median_good_matches_accepted_test_positive'])}/{_fmt(row['median_inliers_accepted_test_positive'])} | "
            f"{_fmt(row['median_good_matches_rejected_test_positive'])}/{_fmt(row['median_inliers_rejected_test_positive'])} | "
            f"{_fmt(row['median_good_matches_false_accepted_test_negative'])}/{_fmt(row['median_inliers_false_accepted_test_negative'])} |"
        )
    lines.extend(
        [
            "",
            "## False Accept Examples At 1% FAR",
            "",
            "| config | variant | examples |",
            "| --- | --- | --- |",
        ]
    )
    examples = one_pct[one_pct["test_fa"].astype(int) > 0].copy()
    if examples.empty:
        lines.append("| all | all | none at the calibrated 1% FAR operating point |")
    else:
        for _, row in examples.head(25).iterrows():
            text = str(row["false_accept_examples"]).replace("|", "/")
            lines.append(f"| {row['config_name']} | {row['score_variant']} | {text} |")
    lines.extend(["", "## Recommendation", ""])
    lines.extend(f"- {line}" for line in _recommendation_text(matrix))
    return "\n".join(lines) + "\n"


def run_experiments(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    limit_per_pair_set: int = 0,
    max_configs: int = 0,
    overwrite: bool = False,
    target_sizes: tuple[int, ...] = TARGET_SIZES,
    nfeatures_values: tuple[int, ...] = NFEATURES,
    ratios: tuple[float, ...] = RATIOS,
    ransac_models: tuple[str, ...] = RANSAC_MODELS,
    ransac_thresholds: tuple[float, ...] = RANSAC_THRESHOLDS,
    blur_ksizes: tuple[int, ...] = BLUR_KSIZES,
    score_variants: tuple[str, ...] = SCORE_VARIANTS,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> dict[str, Path]:
    benchmark = parse_file_uri(benchmark_dir)
    output = parse_file_uri(outdir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = _load_selected_pairs(benchmark, int(limit_per_pair_set))
    configs = _iter_configs(
        target_sizes=tuple(int(x) for x in target_sizes),
        nfeatures_values=tuple(int(x) for x in nfeatures_values),
        ratios=tuple(float(x) for x in ratios),
        ransac_models=tuple(str(x) for x in ransac_models),
        ransac_thresholds=tuple(float(x) for x in ransac_thresholds),
        blur_ksizes=tuple(int(x) for x in blur_ksizes),
        max_configs=int(max_configs),
    )
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for idx, config in enumerate(configs, start=1):
        t0 = time.perf_counter()
        pair_scores = _evaluate_config(pairs, config)
        elapsed = time.perf_counter() - t0
        rows.extend(
            _summarize_config(
                pair_scores,
                config,
                score_variants=tuple(score_variants),
                target_fars=tuple(float(x) for x in target_fars),
                limit_per_pair_set=int(limit_per_pair_set),
                elapsed_s=elapsed,
            )
        )
        print(
            f"[sift-experiment] {idx}/{len(configs)} config={config['config_id']} "
            f"target={config['target_size']} nf={config['nfeatures']} ratio={config['ratio']} "
            f"model={config['ransac_model']} thresh={config['ransac_thresh']} blur={config['blur_ksize']} "
            f"elapsed={elapsed:.1f}s"
        )
    matrix = pd.DataFrame(rows)
    matrix["total_elapsed_s"] = float(time.perf_counter() - t_start)
    csv_path, md_path = _safe_output_paths(output, "sift_experiment_matrix", overwrite=bool(overwrite))
    matrix.to_csv(csv_path, index=False)
    md_path.write_text(
        _render_markdown(matrix, benchmark_dir=benchmark, n_configs=len(configs), n_pairs=len(pairs)),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path}


def run_focused_official_val_test(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    score_variants: tuple[str, ...] = SCORE_VARIANTS,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> dict[str, Path]:
    benchmark = parse_file_uri(benchmark_dir)
    output = parse_file_uri(outdir)
    output.mkdir(parents=True, exist_ok=True)
    pairs = _load_selected_pairs(benchmark, 0)
    configs: list[dict[str, Any]] = []
    for raw_config in FOCUSED_CONFIGS:
        config = dict(raw_config)
        config["config_id"] = _config_id(config)
        configs.append(config)
    rows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    for idx, config in enumerate(configs, start=1):
        t0 = time.perf_counter()
        pair_scores = _evaluate_config(pairs, config)
        elapsed = time.perf_counter() - t0
        rows.extend(
            _summarize_official_config(
                pair_scores,
                config,
                score_variants=tuple(score_variants),
                target_fars=tuple(float(x) for x in target_fars),
                elapsed_s=elapsed,
            )
        )
        print(
            f"[focused-sift-val-test] {idx}/{len(configs)} config={config['config_name']} "
            f"target={config['target_size']} nf={config['nfeatures']} ratio={config['ratio']} "
            f"model={config['ransac_model']} thresh={config['ransac_thresh']} blur={config['blur_ksize']} "
            f"elapsed={elapsed:.1f}s"
        )
    matrix = pd.DataFrame(rows)
    matrix["total_elapsed_s"] = float(time.perf_counter() - t_start)
    csv_path = output / "focused_sift_val_test_experiments.csv"
    md_path = output / "focused_sift_val_test_experiments.md"
    matrix.to_csv(csv_path, index=False)
    md_path.write_text(
        _render_focused_markdown(matrix, benchmark_dir=benchmark, n_configs=len(configs), n_pairs=len(pairs)),
        encoding="utf-8",
    )
    return {"csv": csv_path, "markdown": md_path}


def _parse_int_list(values: list[str] | None, default: tuple[int, ...]) -> tuple[int, ...]:
    return default if not values else tuple(int(x) for x in values)


def _parse_float_list(values: list[str] | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return default if not values else tuple(float(x) for x in values)


def _parse_str_list(values: list[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    return default if not values else tuple(str(x) for x in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled SIFT plain-vs-roll diagnostics experiments.")
    parser.add_argument("--benchmark_dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument(
        "--focused_official_val_test",
        action="store_true",
        help="Run the five focused full selected-pair configs with val-calibrated/test-evaluated metrics.",
    )
    parser.add_argument("--limit_per_pair_set", type=int, default=0)
    parser.add_argument("--max_configs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target_size", nargs="*", default=None)
    parser.add_argument("--nfeatures", nargs="*", default=None)
    parser.add_argument("--ratio", nargs="*", default=None)
    parser.add_argument("--ransac_model", nargs="*", default=None)
    parser.add_argument("--ransac_thresh", nargs="*", default=None)
    parser.add_argument("--blur_ksize", nargs="*", default=None)
    parser.add_argument("--score_variant", nargs="*", default=None)
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.focused_official_val_test):
        paths = run_focused_official_val_test(
            args.benchmark_dir,
            args.outdir,
            score_variants=_parse_str_list(args.score_variant, SCORE_VARIANTS),
            target_fars=tuple(float(x) for x in args.target_far),
        )
        print("Wrote focused official SIFT val/test experiments:")
        for path in paths.values():
            print(f"  {path}")
        return 0
    paths = run_experiments(
        args.benchmark_dir,
        args.outdir,
        limit_per_pair_set=int(args.limit_per_pair_set),
        max_configs=int(args.max_configs),
        overwrite=bool(args.overwrite),
        target_sizes=_parse_int_list(args.target_size, TARGET_SIZES),
        nfeatures_values=_parse_int_list(args.nfeatures, NFEATURES),
        ratios=_parse_float_list(args.ratio, RATIOS),
        ransac_models=_parse_str_list(args.ransac_model, RANSAC_MODELS),
        ransac_thresholds=_parse_float_list(args.ransac_thresh, RANSAC_THRESHOLDS),
        blur_ksizes=_parse_int_list(args.blur_ksize, BLUR_KSIZES),
        score_variants=_parse_str_list(args.score_variant, SCORE_VARIANTS),
        target_fars=tuple(float(x) for x in args.target_far),
    )
    print("Wrote SIFT experiment matrix:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
