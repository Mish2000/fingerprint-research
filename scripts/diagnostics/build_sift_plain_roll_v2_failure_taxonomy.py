from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
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


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2])).resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.generate_sift_plain_roll_v2_visual_audit import (  # noqa: E402
    _fit_width,
    _label,
    _pad_to_width,
    _side_by_side,
    _sift_v2_views,
    parse_file_uri,
)
from src.fpbench.preprocess.preprocess import PreprocessConfig, extract_fingerprint_roi, preprocess_image  # noqa: E402


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
    / "sift_plain_roll_v2_failure_taxonomy"
)
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)
PAIR_KEYS = ("dataset", "split", "label", "path_a", "path_b")
METHOD_VARIANTS = (
    ("sift", "current_score", "canonical"),
    ("sift", "inliers", "sift_inliers"),
    ("sift_plain_roll_v2", "official_score", "v2"),
)
FRGP_FOCUS_GROUPS = (10, 5, 3, 7)
REQUIRED_FRGP_FOCUS_GROUPS = (5, 10)
OVERLAP_GEOMETRY_DIAGNOSTIC_COLUMNS = (
    "plain_foreground_area_px",
    "roll_foreground_area_px",
    "plain_foreground_bbox_area_px",
    "roll_foreground_bbox_area_px",
    "plain_foreground_bbox_coverage",
    "roll_foreground_bbox_coverage",
    "plain_foreground_mask_coverage",
    "roll_foreground_mask_coverage",
    "foreground_area_ratio_plain_over_roll",
    "foreground_area_ratio_abs_log",
    "crop_coverage_proxy",
    "crop_coverage_imbalance_abs",
    "plain_keypoints_per_1000_bbox_px",
    "roll_keypoints_per_1000_bbox_px",
    "matches_per_1000_min_keypoints",
    "inliers_per_1000_min_keypoints",
    "inlier_ratio",
    "affine_inlier_spatial_spread_plain",
    "affine_inlier_spatial_spread_roll",
    "affine_inlier_spatial_spread_pair",
    "affine_inlier_clustered_tiny_region",
    "affine_inlier_spread_computed",
    "score_severity",
    "taxonomy",
)
VISUAL_RECOMPUTED_COLUMNS = ("matches_recomputed", "inliers_recomputed", "score_recomputed")
VISUAL_SOURCE_COMPARISON_COLUMNS = ("v2_matches", "v2_inliers", "v2_score")
VISUAL_RECOMPUTED_COMPARISON_COLUMNS = (
    *VISUAL_SOURCE_COMPARISON_COLUMNS,
    *VISUAL_RECOMPUTED_COLUMNS,
)


@dataclass(frozen=True)
class FingerprintName:
    subject: str
    capture: str
    ppi: int | None
    frgp: int | None


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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _fmt(value: Any, digits: int = 4) -> str:
    num = _to_float(value)
    if not math.isfinite(num):
        return ""
    return f"{num:.{digits}g}"


def _pct(value: Any) -> str:
    num = _to_float(value)
    if not math.isfinite(num):
        return ""
    return f"{100.0 * num:.2f}%"


def _cmd_text(command: list[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def parse_fingerprint_filename(raw_path: str | Path) -> FingerprintName:
    """Parse NIST SD300B/C names like SUBJECT_plain_1000_03.png or SUBJECT_roll_2000_10.png."""
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


def _score_csv(input_dir: Path, dataset: str, method: str, split: str) -> Path:
    path = input_dir / "scores" / dataset / f"scores_{dataset}_{method}_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing score CSV: {path}")
    return path


def _load_score_csv(input_dir: Path, dataset: str, method: str, split: str) -> pd.DataFrame:
    path = _score_csv(input_dir, dataset, method, split)
    df = pd.read_csv(path)
    missing = {"label", "split", "path_a", "path_b", "score", "inliers", "matches", "k1", "k2"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["dataset"] = dataset
    df["source_scores_csv"] = str(path)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    df["split"] = df["split"].astype(str).str.lower()
    return df


def _assert_unique_pairs(df: pd.DataFrame, label: str) -> None:
    duplicates = df.duplicated(list(PAIR_KEYS), keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, list(PAIR_KEYS)].head(5).to_dict(orient="records")
        raise ValueError(f"{label} has duplicate aligned pair keys, sample={sample}")


def _rename_score_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "score": f"{prefix}_score",
            "inliers": f"{prefix}_inliers",
            "matches": f"{prefix}_matches",
            "k1": f"{prefix}_k1",
            "k2": f"{prefix}_k2",
            "source_scores_csv": f"{prefix}_source_scores_csv",
        }
    )
    keep = [
        *PAIR_KEYS,
        f"{prefix}_score",
        f"{prefix}_inliers",
        f"{prefix}_matches",
        f"{prefix}_k1",
        f"{prefix}_k2",
        f"{prefix}_source_scores_csv",
    ]
    return renamed[keep].copy()


def build_aligned_test_pairs(input_dir: str | Path, datasets: tuple[str, ...] = DEFAULT_DATASETS) -> pd.DataFrame:
    input_path = parse_file_uri(input_dir)
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        sift = _load_score_csv(input_path, dataset, "sift", "test")
        v2 = _load_score_csv(input_path, dataset, "sift_plain_roll_v2", "test")
        sift = sift.loc[sift["split"] == "test"].copy()
        v2 = v2.loc[v2["split"] == "test"].copy()
        _assert_unique_pairs(sift, f"{dataset} canonical SIFT")
        _assert_unique_pairs(v2, f"{dataset} SIFT Plain/Roll v2")

        canonical = _rename_score_columns(sift, "canonical")
        v2_scores = _rename_score_columns(v2, "v2")
        aligned = canonical.merge(v2_scores, on=list(PAIR_KEYS), how="inner", validate="one_to_one")
        if len(aligned) != len(canonical) or len(aligned) != len(v2_scores):
            raise ValueError(
                f"{dataset} alignment mismatch: canonical={len(canonical)} v2={len(v2_scores)} aligned={len(aligned)}"
            )
        frames.append(aligned)

    out = pd.concat(frames, ignore_index=True, sort=False)
    out["sift_inliers_score"] = pd.to_numeric(out["canonical_inliers"], errors="coerce").fillna(0).astype(int)
    out["canonical_inlier_ratio"] = out["canonical_inliers"] / out["canonical_matches"].clip(lower=1)
    out["v2_inlier_ratio"] = out["v2_inliers"] / out["v2_matches"].clip(lower=1)

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

    front = [
        "dataset",
        "split",
        "label",
        "subject_a",
        "subject_b",
        "frgp",
        "frgp_a",
        "frgp_b",
        "ppi_a",
        "ppi_b",
        "capture_a",
        "capture_b",
        "path_a",
        "path_b",
    ]
    rest = [col for col in out.columns if col not in front]
    return out[front + rest].copy()


def load_thresholds(input_dir: str | Path) -> pd.DataFrame:
    path = parse_file_uri(input_dir) / "per_dataset_thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing threshold CSV: {path}")
    df = pd.read_csv(path)
    missing = {"dataset", "method", "variant", "target_far", "threshold"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    df = df.copy()
    df["target_far"] = pd.to_numeric(df["target_far"], errors="raise")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    return df


def _threshold_lookup(thresholds: pd.DataFrame) -> dict[tuple[str, str, str, float], float]:
    out: dict[tuple[str, str, str, float], float] = {}
    for _, row in thresholds.iterrows():
        out[
            (
                str(row["dataset"]),
                str(row["method"]),
                str(row["variant"]),
                round(float(row["target_far"]), 6),
            )
        ] = float(row["threshold"])
    return out


def build_pair_decisions(
    aligned: pd.DataFrame,
    thresholds: pd.DataFrame,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    lookup = _threshold_lookup(thresholds)
    frames: list[pd.DataFrame] = []
    for target_far in target_fars:
        target_key = round(float(target_far), 6)
        frame = aligned.copy()
        frame["target_far"] = float(target_far)
        for dataset in sorted(frame["dataset"].unique()):
            mask = frame["dataset"] == dataset
            try:
                canonical_threshold = lookup[(dataset, "sift", "current_score", target_key)]
                inliers_threshold = lookup[(dataset, "sift", "inliers", target_key)]
                v2_threshold = lookup[(dataset, "sift_plain_roll_v2", "official_score", target_key)]
            except KeyError as exc:
                raise ValueError(f"Missing threshold for {dataset} target_far={target_far}") from exc
            frame.loc[mask, "canonical_threshold"] = canonical_threshold
            frame.loc[mask, "sift_inliers_threshold"] = inliers_threshold
            frame.loc[mask, "v2_threshold"] = v2_threshold

        frame["canonical_accepted"] = frame["canonical_score"] >= frame["canonical_threshold"]
        frame["sift_inliers_accepted"] = frame["sift_inliers_score"] >= frame["sift_inliers_threshold"]
        frame["v2_accepted"] = frame["v2_score"] >= frame["v2_threshold"]
        frame["canonical_score_margin"] = frame["canonical_score"] - frame["canonical_threshold"]
        frame["v2_score_margin"] = frame["v2_score"] - frame["v2_threshold"]
        frame["canonical_score_margin_ratio"] = frame["canonical_score_margin"] / frame["canonical_threshold"]
        frame["v2_score_margin_ratio"] = frame["v2_score_margin"] / frame["v2_threshold"]
        frame["decision_overlap"] = frame.apply(_decision_overlap_category, axis=1)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["target_far_label"] = out["target_far"].map(lambda x: f"{100.0 * float(x):.1f}%")
    return out


def _decision_overlap_category(row: pd.Series) -> str:
    canonical = bool(row["canonical_accepted"])
    v2 = bool(row["v2_accepted"])
    label = _to_int(row["label"])
    if label == 1:
        if canonical and v2:
            return "both_accept"
        if (not canonical) and v2:
            return "v2_rescue"
        if canonical and (not v2):
            return "v2_lost"
        return "both_reject"
    if (not canonical) and (not v2):
        return "both_reject"
    if (not canonical) and v2:
        return "v2_new_false_accept"
    if canonical and (not v2):
        return "v2_fixed_false_accept"
    return "both_false_accept"


def summarize_decision_overlap(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    positive_categories = ("both_accept", "v2_rescue", "v2_lost", "both_reject")
    negative_categories = ("both_reject", "v2_new_false_accept", "v2_fixed_false_accept", "both_false_accept")
    for (dataset, target_far), group in decisions.groupby(["dataset", "target_far"], sort=True):
        positives = group[group["label"].astype(int) == 1]
        negatives = group[group["label"].astype(int) == 0]
        row: dict[str, Any] = {
            "dataset": dataset,
            "target_far": float(target_far),
            "n_positive": int(len(positives)),
            "n_negative": int(len(negatives)),
        }
        pos_counts = positives["decision_overlap"].value_counts()
        neg_counts = negatives["decision_overlap"].value_counts()
        for category in positive_categories:
            count = int(pos_counts.get(category, 0))
            row[f"positive_{category}"] = count
            row[f"positive_{category}_rate"] = count / max(int(len(positives)), 1)
        for category in negative_categories:
            count = int(neg_counts.get(category, 0))
            row[f"negative_{category}"] = count
            row[f"negative_{category}_rate"] = count / max(int(len(negatives)), 1)
        row["canonical_ta"] = int((positives["canonical_accepted"]).sum())
        row["v2_ta"] = int((positives["v2_accepted"]).sum())
        row["canonical_fa"] = int((negatives["canonical_accepted"]).sum())
        row["v2_fa"] = int((negatives["v2_accepted"]).sum())
        row["v2_net_ta_vs_canonical"] = int(row["v2_ta"] - row["canonical_ta"])
        row["v2_net_fa_vs_canonical"] = int(row["v2_fa"] - row["canonical_fa"])
        rows.append(row)
    return pd.DataFrame(rows)


def _score_severity(row: pd.Series) -> str:
    ratio = _to_float(row.get("v2_score_margin_ratio"))
    if ratio >= -0.10:
        return "near_miss"
    if ratio >= -0.50:
        return "moderate_margin_failure"
    return "hard_score_failure"


def _positive_failure_category(row: pd.Series) -> str:
    margin_ratio = _to_float(row.get("v2_score_margin_ratio"))
    matches = _to_int(row.get("v2_matches"))
    inliers = _to_int(row.get("v2_inliers"))
    k1 = _to_int(row.get("v2_k1"))
    k2 = _to_int(row.get("v2_k2"))
    min_keypoints = min(k1, k2)

    if margin_ratio >= -0.10:
        return "near_miss"
    if matches < 8:
        return "low_match_failure"
    if matches >= 24 and inliers < 8:
        return "possible_geometry_failure"
    if min_keypoints >= 1000 and matches < 16:
        return "possible_overlap_or_crop_failure"
    if inliers < 6:
        return "low_inlier_failure"
    if margin_ratio >= -0.50:
        return "moderate_margin_failure"
    if margin_ratio < -0.50:
        return "hard_score_failure"
    return "ambiguous"


def build_positive_failure_taxonomy(decisions: pd.DataFrame, target_far: float = 0.01) -> pd.DataFrame:
    target = decisions[np.isclose(decisions["target_far"], float(target_far))].copy()
    failures = target[(target["label"].astype(int) == 1) & (~target["v2_accepted"].astype(bool))].copy()
    failures["score_severity"] = failures.apply(_score_severity, axis=1)
    failures["taxonomy"] = failures.apply(_positive_failure_category, axis=1)
    failures["taxonomy_flags"] = failures.apply(_positive_failure_flags, axis=1)
    sort_cols = ["dataset", "v2_score_margin_ratio", "v2_score", "v2_matches"]
    return failures.sort_values(sort_cols, ascending=[True, False, False, False]).reset_index(drop=True)


def _positive_failure_flags(row: pd.Series) -> str:
    flags: list[str] = []
    margin_ratio = _to_float(row.get("v2_score_margin_ratio"))
    matches = _to_int(row.get("v2_matches"))
    inliers = _to_int(row.get("v2_inliers"))
    min_keypoints = min(_to_int(row.get("v2_k1")), _to_int(row.get("v2_k2")))
    if margin_ratio >= -0.10:
        flags.append("near_miss")
    if -0.50 <= margin_ratio < -0.10:
        flags.append("moderate_margin")
    if margin_ratio < -0.50:
        flags.append("hard_margin")
    if matches < 8:
        flags.append("low_matches")
    if inliers < 6:
        flags.append("low_inliers")
    if min_keypoints >= 1000 and matches < 16:
        flags.append("keypoints_present_low_matches")
    if matches >= 24 and inliers < 8:
        flags.append("many_matches_low_inliers")
    return ";".join(flags) if flags else "ambiguous"


def _negative_false_accept_category(row: pd.Series) -> str:
    margin_ratio = _to_float(row.get("v2_score_margin_ratio"))
    inliers = _to_int(row.get("v2_inliers"))
    matches = _to_int(row.get("v2_matches"))
    inlier_threshold = _to_float(row.get("sift_inliers_threshold"), default=12.0)
    if margin_ratio <= 0.10:
        return "near_threshold_false_accept"
    if margin_ratio >= 0.50:
        return "high_confidence_false_accept"
    if inliers >= int(math.ceil(inlier_threshold)):
        return "high_inlier_false_accept"
    if matches >= 40:
        return "high_match_false_accept"
    if matches >= 24 and inliers >= 8:
        return "possible_ridge_texture_collision"
    return "needs_visual_review"


def _negative_false_accept_flags(row: pd.Series) -> str:
    flags: list[str] = []
    margin_ratio = _to_float(row.get("v2_score_margin_ratio"))
    inliers = _to_int(row.get("v2_inliers"))
    matches = _to_int(row.get("v2_matches"))
    inlier_threshold = _to_float(row.get("sift_inliers_threshold"), default=12.0)
    if margin_ratio <= 0.10:
        flags.append("near_threshold")
    if margin_ratio >= 0.50:
        flags.append("high_confidence")
    if inliers >= int(math.ceil(inlier_threshold)):
        flags.append("high_inlier")
    if matches >= 40:
        flags.append("high_match")
    if matches >= 24 and inliers >= 8:
        flags.append("ridge_texture_collision_candidate")
    return ";".join(flags) if flags else "needs_visual_review"


def build_negative_false_accept_taxonomy(decisions: pd.DataFrame, target_far: float = 0.01) -> pd.DataFrame:
    target = decisions[np.isclose(decisions["target_far"], float(target_far))].copy()
    false_accepts = target[(target["label"].astype(int) == 0) & (target["v2_accepted"].astype(bool))].copy()
    false_accepts["taxonomy"] = false_accepts.apply(_negative_false_accept_category, axis=1)
    false_accepts["taxonomy_flags"] = false_accepts.apply(_negative_false_accept_flags, axis=1)
    return false_accepts.sort_values(["dataset", "v2_score", "v2_score_margin_ratio"], ascending=[True, False, False]).reset_index(drop=True)


def build_per_frgp_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, target_far, frgp), group in decisions.groupby(["dataset", "target_far", "frgp"], dropna=False, sort=True):
        for method, variant, prefix in METHOD_VARIANTS:
            accepted_col = {
                "canonical": "canonical_accepted",
                "sift_inliers": "sift_inliers_accepted",
                "v2": "v2_accepted",
            }[prefix]
            positives = group[group["label"].astype(int) == 1]
            negatives = group[group["label"].astype(int) == 0]
            pos_accept = positives[accepted_col].astype(bool) if not positives.empty else pd.Series(dtype=bool)
            neg_accept = negatives[accepted_col].astype(bool) if not negatives.empty else pd.Series(dtype=bool)
            ta = int(pos_accept.sum())
            fr = int((~pos_accept).sum())
            fa = int(neg_accept.sum())
            tr = int((~neg_accept).sum())
            n_positive = int(len(positives))
            n_negative = int(len(negatives))
            rows.append(
                {
                    "dataset": dataset,
                    "split": "test",
                    "method": method,
                    "variant": variant,
                    "target_far": float(target_far),
                    "frgp": "" if pd.isna(frgp) else int(frgp),
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "tar": ta / n_positive if n_positive else float("nan"),
                    "far": fa / n_negative if n_negative else float("nan"),
                    "ta": ta,
                    "fr": fr,
                    "fa": fa,
                    "tr": tr,
                }
            )
    return pd.DataFrame(rows).sort_values(["dataset", "method", "variant", "target_far", "frgp"]).reset_index(drop=True)


def _load_gray(path_str: str) -> np.ndarray:
    path = parse_file_uri(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _case_slug(row: pd.Series, group_name: str, rank: int) -> str:
    dataset = re.sub(r"[^A-Za-z0-9]+", "_", str(row["dataset"])).strip("_")
    subject_a = re.sub(r"[^A-Za-z0-9]+", "_", str(row.get("subject_a", ""))).strip("_") or "a"
    subject_b = re.sub(r"[^A-Za-z0-9]+", "_", str(row.get("subject_b", ""))).strip("_") or "b"
    frgp = re.sub(r"[^A-Za-z0-9]+", "_", str(row.get("frgp", ""))).strip("_") or "frgp"
    return f"{dataset}_{group_name}_{rank:02d}_{subject_a}_{subject_b}_f{frgp}"


def _bool_text(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _put_text_lines(canvas: np.ndarray, lines: list[str], *, x: int = 12, y: int = 28, dy: int = 27) -> None:
    for idx, line in enumerate(lines):
        cv2.putText(
            canvas,
            line[:180],
            (x, y + idx * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _make_decision_case_sheet(
    row: pd.Series,
    group_name: str,
    rank: int,
    sheet_dir: Path,
    args: argparse.Namespace,
    filename_stem: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    img_a = _load_gray(str(row["path_a"]))
    img_b = _load_gray(str(row["path_b"]))
    proc_a = preprocess_image(img_a, PreprocessConfig(target_size=int(args.target_size), blur_ksize=int(args.blur_ksize)))
    proc_b = preprocess_image(img_b, PreprocessConfig(target_size=int(args.target_size), blur_ksize=int(args.blur_ksize)))
    header = np.full((116, 1500, 3), 32, dtype=np.uint8)
    margin = _to_float(row.get("v2_score_margin"))
    margin_ratio = _to_float(row.get("v2_score_margin_ratio"))
    lines = [
        (
            f"{group_name} #{rank} {row['dataset']} label={int(row['label'])} "
            f"subject={row.get('subject_a', '')}->{row.get('subject_b', '')} frgp={row.get('frgp', '')}"
        ),
        (
            f"canonical score={_fmt(row.get('canonical_score'), 6)} threshold={_fmt(row.get('canonical_threshold'), 6)} "
            f"accepted={_bool_text(row.get('canonical_accepted'))} | "
            f"SIFT inliers score={_fmt(row.get('sift_inliers_score'), 6)} threshold={_fmt(row.get('sift_inliers_threshold'), 6)} "
            f"accepted={_bool_text(row.get('sift_inliers_accepted'))}"
        ),
        (
            f"v2 score={_fmt(row.get('v2_score'), 6)} threshold={_fmt(row.get('v2_threshold'), 6)} "
            f"accepted={_bool_text(row.get('v2_accepted'))} matches={_to_int(row.get('v2_matches'))} "
            f"inliers={_to_int(row.get('v2_inliers'))} margin={_fmt(margin, 6)} ratio={_pct(margin_ratio)}"
        ),
        f"path_a={Path(str(row['path_a'])).name} | path_b={Path(str(row['path_b'])).name}",
    ]
    _put_text_lines(header, lines)
    keypoints, good_matches, inliers, diagnostics = _sift_v2_views(
        img_a,
        img_b,
        target_size=int(args.target_size),
        nfeatures=int(args.nfeatures),
        blur_ksize=int(args.blur_ksize),
        ratio=float(args.ratio),
        ransac_thresh=float(args.ransac_thresh),
    )
    rows = [
        header,
        _side_by_side(img_a, img_b, "raw plain", "raw roll", height=300),
        _side_by_side(proc_a, proc_b, "preprocessed plain", "preprocessed roll", height=300),
        keypoints,
        good_matches,
        inliers,
    ]
    width = max(item.shape[1] for item in rows)
    padded = [_pad_to_width(item, width) for item in rows]
    gutter = np.full((12, width, 3), 245, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for item in padded:
        if stacked:
            stacked.append(gutter)
        stacked.append(item)
    sheet = np.vstack(stacked)
    sheet_path = sheet_dir / f"{filename_stem or _case_slug(row, group_name, rank)}.png"
    cv2.imwrite(str(sheet_path), sheet)
    return sheet_path, {
        "group": group_name,
        "rank": int(rank),
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "label": int(row["label"]),
        "subject_a": str(row.get("subject_a", "")),
        "subject_b": str(row.get("subject_b", "")),
        "frgp": "" if pd.isna(row.get("frgp")) else int(row.get("frgp")),
        "path_a": str(row["path_a"]),
        "path_b": str(row["path_b"]),
        "canonical_score": _to_float(row.get("canonical_score")),
        "canonical_threshold": _to_float(row.get("canonical_threshold")),
        "canonical_accepted": bool(row.get("canonical_accepted")),
        "sift_inliers_score": _to_float(row.get("sift_inliers_score")),
        "sift_inliers_threshold": _to_float(row.get("sift_inliers_threshold")),
        "sift_inliers_accepted": bool(row.get("sift_inliers_accepted")),
        "v2_score": _to_float(row.get("v2_score")),
        "v2_threshold": _to_float(row.get("v2_threshold")),
        "v2_accepted": bool(row.get("v2_accepted")),
        "v2_matches": _to_int(row.get("v2_matches")),
        "v2_inliers": _to_int(row.get("v2_inliers")),
        "v2_score_margin": _to_float(row.get("v2_score_margin")),
        "v2_score_margin_ratio": _to_float(row.get("v2_score_margin_ratio")),
        "sheet": str(sheet_path),
        **diagnostics,
    }


def _select_visual_groups(decisions_1pct: pd.DataFrame, top_n: int) -> dict[str, pd.DataFrame]:
    groups: dict[str, list[pd.DataFrame]] = {
        "v2_rescued_high_confidence": [],
        "v2_rescued_near_threshold": [],
        "v2_lost": [],
        "both_rejected_near_miss": [],
        "both_rejected_hard_failure": [],
        "v2_false_accept_top_score": [],
        "v2_false_accept_near_threshold": [],
        "canonical_false_accept_fixed_by_v2": [],
    }
    for _, dataset_group in decisions_1pct.groupby("dataset", sort=True):
        positives = dataset_group[dataset_group["label"].astype(int) == 1]
        negatives = dataset_group[dataset_group["label"].astype(int) == 0]
        rescued = positives[positives["decision_overlap"] == "v2_rescue"]
        both_rejected = positives[positives["decision_overlap"] == "both_reject"]
        false_accepts = negatives[negatives["v2_accepted"].astype(bool)]
        fixed = negatives[negatives["decision_overlap"] == "v2_fixed_false_accept"]

        groups["v2_rescued_high_confidence"].append(
            rescued.sort_values(["v2_score_margin_ratio", "v2_score"], ascending=False).head(int(top_n))
        )
        groups["v2_rescued_near_threshold"].append(
            rescued[rescued["v2_score_margin_ratio"] <= 0.10]
            .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[True, True])
            .head(int(top_n))
        )
        groups["v2_lost"].append(
            positives[positives["decision_overlap"] == "v2_lost"]
            .sort_values(["v2_score_margin_ratio", "canonical_score_margin_ratio"], ascending=[True, False])
            .head(int(top_n))
        )
        groups["both_rejected_near_miss"].append(
            both_rejected[both_rejected["v2_score_margin_ratio"] >= -0.10]
            .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[False, False])
            .head(int(top_n))
        )
        groups["both_rejected_hard_failure"].append(
            both_rejected[both_rejected["v2_score_margin_ratio"] < -0.50]
            .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[True, True])
            .head(int(top_n))
        )
        groups["v2_false_accept_top_score"].append(
            false_accepts.sort_values(["v2_score", "v2_score_margin_ratio"], ascending=False).head(int(top_n))
        )
        groups["v2_false_accept_near_threshold"].append(
            false_accepts[false_accepts["v2_score_margin_ratio"] <= 0.10]
            .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[True, True])
            .head(int(top_n))
        )
        groups["canonical_false_accept_fixed_by_v2"].append(
            fixed.sort_values(["canonical_score", "canonical_score_margin_ratio"], ascending=False).head(int(top_n))
        )

    return {
        name: pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
        for name, frames in groups.items()
    }


def generate_visual_audit(decisions: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> pd.DataFrame:
    sheet_dir = outdir / "visual_audit_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    decisions_1pct = decisions[np.isclose(decisions["target_far"], 0.01)].copy()
    groups = _select_visual_groups(decisions_1pct, int(args.visual_top_n))
    rows: list[dict[str, Any]] = []
    for group_name, group_df in groups.items():
        if group_df.empty:
            rows.append({"group": group_name, "rank": 0, "sheet": "", "note": "no matching cases"})
            continue
        for rank, (_, row) in enumerate(group_df.iterrows(), start=1):
            sheet_path, payload = _make_decision_case_sheet(row, group_name, rank, sheet_dir, args)
            payload["sheet_relative"] = str(sheet_path.relative_to(outdir)).replace("\\", "/")
            rows.append(payload)
    return pd.DataFrame(rows)


def _target_far_rows(decisions: pd.DataFrame, target_far: float = 0.01) -> pd.DataFrame:
    return decisions[np.isclose(pd.to_numeric(decisions["target_far"], errors="coerce"), float(target_far))].copy()


def _row_score_severity(row: pd.Series) -> str:
    label = _to_int(row.get("label"))
    v2_accepted = _to_bool(row.get("v2_accepted"))
    if label == 1 and not v2_accepted:
        return _score_severity(row)
    if label == 0 and v2_accepted:
        taxonomy = _negative_false_accept_category(row)
        if taxonomy == "near_threshold_false_accept":
            return "near_threshold_false_accept"
        if taxonomy == "high_confidence_false_accept":
            return "high_confidence_false_accept"
        return "other_false_accept"
    if label == 1:
        return "v2_true_accept"
    return "v2_true_reject"


def _row_taxonomy(row: pd.Series) -> str:
    label = _to_int(row.get("label"))
    v2_accepted = _to_bool(row.get("v2_accepted"))
    if label == 1 and not v2_accepted:
        return _positive_failure_category(row)
    if label == 0 and v2_accepted:
        return _negative_false_accept_category(row)
    if label == 1:
        return "v2_true_accept"
    return "v2_true_reject"


def _safe_div(numerator: Any, denominator: Any) -> float:
    denom = _to_float(denominator)
    if not math.isfinite(denom) or denom == 0.0:
        return float("nan")
    return _to_float(numerator) / denom


@lru_cache(maxsize=8192)
def _foreground_bbox_metrics(path_str: str, target_size: int, blur_ksize: int) -> dict[str, Any]:
    img = _load_gray(path_str)
    processed = preprocess_image(
        img,
        PreprocessConfig(target_size=int(target_size), blur_ksize=int(blur_ksize)),
    )
    roi = extract_fingerprint_roi(processed)
    mask = roi.mask if roi.is_valid else np.zeros_like(processed, dtype=np.uint8)
    foreground = mask > 0
    image_area = int(processed.shape[0] * processed.shape[1])
    foreground_area = int(np.sum(foreground))
    bbox_area = 0
    bbox_width = 0
    bbox_height = 0
    if foreground_area > 0:
        ys, xs = np.where(foreground)
        bbox_width = int(xs.max() - xs.min() + 1)
        bbox_height = int(ys.max() - ys.min() + 1)
        bbox_area = int(bbox_width * bbox_height)
    return {
        "image_width": int(processed.shape[1]),
        "image_height": int(processed.shape[0]),
        "roi_valid": bool(roi.is_valid),
        "roi_failure_reason": roi.failure_reason or "",
        "foreground_area_px": foreground_area,
        "foreground_bbox_width": bbox_width,
        "foreground_bbox_height": bbox_height,
        "foreground_bbox_area_px": bbox_area,
        "foreground_mask_coverage": float(foreground_area / image_area) if image_area else float("nan"),
        "foreground_bbox_coverage": float(bbox_area / image_area) if image_area else float("nan"),
        "foreground_fill_ratio": float(foreground_area / max(bbox_area, 1)),
    }


def _with_row_taxonomy(decisions: pd.DataFrame) -> pd.DataFrame:
    out = decisions.copy()
    out["score_severity"] = out.apply(_row_score_severity, axis=1)
    out["taxonomy"] = out.apply(_row_taxonomy, axis=1)
    return out


def build_overlap_geometry_diagnostics(
    decisions: pd.DataFrame,
    *,
    target_far: float = 0.01,
    target_size: int = 768,
    blur_ksize: int = 0,
) -> pd.DataFrame:
    target = _with_row_taxonomy(_target_far_rows(decisions, target_far=target_far))
    rows: list[dict[str, Any]] = []
    for _, row in target.iterrows():
        plain = _foreground_bbox_metrics(str(row["path_a"]), int(target_size), int(blur_ksize))
        roll = _foreground_bbox_metrics(str(row["path_b"]), int(target_size), int(blur_ksize))
        k1 = _to_int(row.get("v2_k1"))
        k2 = _to_int(row.get("v2_k2"))
        matches = _to_int(row.get("v2_matches"))
        inliers = _to_int(row.get("v2_inliers"))
        min_keypoints = max(min(k1, k2), 1)
        foreground_area_ratio = _safe_div(plain["foreground_area_px"], roll["foreground_area_px"])
        foreground_area_abs_log = (
            abs(math.log(foreground_area_ratio))
            if math.isfinite(foreground_area_ratio) and foreground_area_ratio > 0.0
            else float("nan")
        )
        plain_bbox_coverage = _to_float(plain["foreground_bbox_coverage"])
        roll_bbox_coverage = _to_float(roll["foreground_bbox_coverage"])
        crop_coverage_proxy = min(plain_bbox_coverage, roll_bbox_coverage)
        crop_coverage_imbalance = abs(plain_bbox_coverage - roll_bbox_coverage)
        rows.append(
            {
                "dataset": str(row["dataset"]),
                "split": str(row["split"]),
                "label": _to_int(row.get("label")),
                "subject_a": str(row.get("subject_a", "")),
                "subject_b": str(row.get("subject_b", "")),
                "frgp": "" if pd.isna(row.get("frgp")) else int(row.get("frgp")),
                "path_a": str(row["path_a"]),
                "path_b": str(row["path_b"]),
                "target_far": float(row["target_far"]),
                "decision_overlap": str(row.get("decision_overlap", "")),
                "canonical_score": _to_float(row.get("canonical_score")),
                "canonical_threshold": _to_float(row.get("canonical_threshold")),
                "canonical_accepted": _to_bool(row.get("canonical_accepted")),
                "v2_score": _to_float(row.get("v2_score")),
                "v2_threshold": _to_float(row.get("v2_threshold")),
                "v2_accepted": _to_bool(row.get("v2_accepted")),
                "v2_score_margin": _to_float(row.get("v2_score_margin")),
                "v2_score_margin_ratio": _to_float(row.get("v2_score_margin_ratio")),
                "v2_k1": k1,
                "v2_k2": k2,
                "v2_matches": matches,
                "v2_inliers": inliers,
                "plain_roi_valid": bool(plain["roi_valid"]),
                "roll_roi_valid": bool(roll["roi_valid"]),
                "plain_roi_failure_reason": str(plain["roi_failure_reason"]),
                "roll_roi_failure_reason": str(roll["roi_failure_reason"]),
                "plain_foreground_area_px": int(plain["foreground_area_px"]),
                "roll_foreground_area_px": int(roll["foreground_area_px"]),
                "plain_foreground_bbox_area_px": int(plain["foreground_bbox_area_px"]),
                "roll_foreground_bbox_area_px": int(roll["foreground_bbox_area_px"]),
                "plain_foreground_bbox_coverage": plain_bbox_coverage,
                "roll_foreground_bbox_coverage": roll_bbox_coverage,
                "plain_foreground_mask_coverage": _to_float(plain["foreground_mask_coverage"]),
                "roll_foreground_mask_coverage": _to_float(roll["foreground_mask_coverage"]),
                "plain_foreground_fill_ratio": _to_float(plain["foreground_fill_ratio"]),
                "roll_foreground_fill_ratio": _to_float(roll["foreground_fill_ratio"]),
                "foreground_area_ratio_plain_over_roll": foreground_area_ratio,
                "foreground_area_ratio_abs_log": foreground_area_abs_log,
                "crop_coverage_proxy": crop_coverage_proxy,
                "crop_coverage_imbalance_abs": crop_coverage_imbalance,
                "plain_keypoints_per_1000_bbox_px": 1000.0 * _safe_div(k1, plain["foreground_bbox_area_px"]),
                "roll_keypoints_per_1000_bbox_px": 1000.0 * _safe_div(k2, roll["foreground_bbox_area_px"]),
                "matches_per_1000_min_keypoints": 1000.0 * matches / float(min_keypoints),
                "inliers_per_1000_min_keypoints": 1000.0 * inliers / float(min_keypoints),
                "inlier_ratio": float(inliers / max(matches, 1)),
                "affine_inlier_spatial_spread_plain": float("nan"),
                "affine_inlier_spatial_spread_roll": float("nan"),
                "affine_inlier_spatial_spread_pair": float("nan"),
                "affine_inlier_clustered_tiny_region": "",
                "affine_inlier_spread_computed": False,
                "affine_inlier_spread_note": "not_computed_source_score_csv_has_counts_not_inlier_coordinates",
                "score_severity": str(row["score_severity"]),
                "taxonomy": str(row["taxonomy"]),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def assert_no_diagnostic_decision_inputs(decisions: pd.DataFrame) -> None:
    present = sorted(set(OVERLAP_GEOMETRY_DIAGNOSTIC_COLUMNS).intersection(decisions.columns))
    if present:
        raise AssertionError(f"Decision rows already contain overlap/geometry diagnostic columns: {present}")


def _frgp_focus_slug(row: pd.Series, category: str, rank: int) -> str:
    dataset = re.sub(r"[^A-Za-z0-9]+", "_", str(row["dataset"])).strip("_")
    subject_a = re.sub(r"[^A-Za-z0-9]+", "_", str(row.get("subject_a", ""))).strip("_") or "a"
    subject_b = re.sub(r"[^A-Za-z0-9]+", "_", str(row.get("subject_b", ""))).strip("_") or "b"
    frgp = _to_int(row.get("frgp"))
    category_clean = re.sub(r"[^A-Za-z0-9]+", "_", str(category)).strip("_")
    return f"frgp_focus_{dataset}_f{frgp}_{category_clean}_{rank:02d}_{subject_a}_{subject_b}"


def _focus_category_candidates(group: pd.DataFrame) -> dict[str, pd.DataFrame]:
    positives = group[group["label"].astype(int) == 1].copy()
    negatives = group[group["label"].astype(int) == 0].copy()
    false_rejects = positives[~positives["v2_accepted"].map(_to_bool)].copy()
    return {
        "v2_rescued_positives": positives[positives["decision_overlap"] == "v2_rescue"]
        .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[False, False]),
        "v2_hard_false_rejects": false_rejects[false_rejects["score_severity"] == "hard_score_failure"]
        .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[True, True]),
        "v2_near_miss_false_rejects": false_rejects[false_rejects["score_severity"] == "near_miss"]
        .sort_values(["v2_score_margin_ratio", "v2_score"], ascending=[False, False]),
        "v2_false_accepts": negatives[negatives["v2_accepted"].map(_to_bool)]
        .sort_values(["v2_score", "v2_score_margin_ratio"], ascending=[False, False]),
        "canonical_false_accepts_fixed_by_v2": negatives[negatives["decision_overlap"] == "v2_fixed_false_accept"]
        .sort_values(["canonical_score", "canonical_score_margin_ratio"], ascending=[False, False]),
    }


def generate_frgp_focus_visual_audit(
    decisions: pd.DataFrame,
    outdir: Path,
    args: argparse.Namespace,
    *,
    focus_frgps: tuple[int, ...] = FRGP_FOCUS_GROUPS,
    target_far: float = 0.01,
) -> pd.DataFrame:
    sheet_dir = outdir / "visual_audit_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    top_n = int(getattr(args, "frgp_focus_top_n", 2))
    target = _with_row_taxonomy(_target_far_rows(decisions, target_far=target_far))
    rows: list[dict[str, Any]] = []
    for dataset in sorted(target["dataset"].astype(str).unique()):
        dataset_group = target[target["dataset"].astype(str) == dataset]
        for frgp in focus_frgps:
            group = dataset_group[dataset_group["frgp"].astype("Int64") == int(frgp)].copy()
            candidates_by_category = _focus_category_candidates(group) if not group.empty else {}
            for category in (
                "v2_rescued_positives",
                "v2_hard_false_rejects",
                "v2_near_miss_false_rejects",
                "v2_false_accepts",
                "canonical_false_accepts_fixed_by_v2",
            ):
                candidates = candidates_by_category.get(category, pd.DataFrame())
                if candidates.empty:
                    rows.append(
                        {
                            "group": f"frgp_focus_f{frgp}_{category}",
                            "rank": 0,
                            "dataset": dataset,
                            "focus_frgp": int(frgp),
                            "focus_category": category,
                            "focus_total_cases": 0,
                            "sheet": "",
                            "sheet_relative": "",
                            "note": "no matching cases",
                        }
                    )
                    continue
                total = int(len(candidates))
                for rank, (_, row) in enumerate(candidates.head(top_n).iterrows(), start=1):
                    group_name = f"frgp_focus_f{frgp}_{category}"
                    sheet_path, payload = _make_decision_case_sheet(
                        row,
                        group_name,
                        rank,
                        sheet_dir,
                        args,
                        filename_stem=_frgp_focus_slug(row, category, rank),
                    )
                    payload.update(
                        {
                            "focus_frgp": int(frgp),
                            "focus_category": category,
                            "focus_total_cases": total,
                            "score_severity": str(row.get("score_severity", "")),
                            "taxonomy": str(row.get("taxonomy", "")),
                            "sheet_relative": str(sheet_path.relative_to(outdir)).replace("\\", "/"),
                        }
                    )
                    rows.append(payload)
    return pd.DataFrame(rows)


def _resolve_case_sheet_path(row: pd.Series, outdir: Path) -> Path | None:
    raw_sheet = row.get("sheet", "")
    sheet = "" if pd.isna(raw_sheet) else str(raw_sheet).strip()
    if not sheet:
        raw_rel = row.get("sheet_relative", "")
        rel = "" if pd.isna(raw_rel) else str(raw_rel).strip()
        if not rel:
            return None
        sheet = rel
    path = Path(sheet)
    if not path.is_absolute():
        path = outdir / path
    return path


def assert_visual_case_sheets_and_recomputed_values(cases: pd.DataFrame | Path, outdir: Path) -> None:
    df = pd.read_csv(cases) if isinstance(cases, Path) else cases.copy()
    if df.empty:
        raise AssertionError("No visual audit case rows were available.")
    missing_sheets: list[str] = []
    missing_recomputed_columns: list[str] = []
    missing_source_columns: list[str] = []
    missing_values: list[str] = []
    mismatches: list[str] = []
    for _, row in df.iterrows():
        sheet_path = _resolve_case_sheet_path(row, outdir)
        if sheet_path is None:
            continue
        missing_recomputed = [col for col in VISUAL_RECOMPUTED_COLUMNS if col not in row.index]
        missing_source = [col for col in VISUAL_SOURCE_COMPARISON_COLUMNS if col not in row.index]
        if missing_recomputed:
            missing_recomputed_columns.append(f"{sheet_path.name}: {', '.join(missing_recomputed)}")
        if missing_source:
            missing_source_columns.append(f"{sheet_path.name}: {', '.join(missing_source)}")
        if missing_recomputed or missing_source:
            continue
        blank = [
            col
            for col in VISUAL_RECOMPUTED_COMPARISON_COLUMNS
            if pd.isna(row[col]) or str(row[col]).strip() == ""
        ]
        if blank:
            missing_values.append(f"{sheet_path.name}: {', '.join(blank)}")
            continue
        if not sheet_path.exists():
            missing_sheets.append(str(sheet_path))
            continue
        if _to_int(row["matches_recomputed"]) != _to_int(row["v2_matches"]):
            mismatches.append(f"{sheet_path.name}: matches")
        if _to_int(row["inliers_recomputed"]) != _to_int(row["v2_inliers"]):
            mismatches.append(f"{sheet_path.name}: inliers")
        if abs(_to_float(row["score_recomputed"]) - _to_float(row["v2_score"])) > 1e-9:
            mismatches.append(f"{sheet_path.name}: score")
    if missing_recomputed_columns:
        sample = "; ".join(missing_recomputed_columns[:5])
        raise AssertionError(
            f"Visual audit rows with sheets are missing recomputation columns ({len(missing_recomputed_columns)}): {sample}"
        )
    if missing_source_columns:
        sample = "; ".join(missing_source_columns[:5])
        raise AssertionError(f"Visual audit rows with sheets are missing source comparison columns ({len(missing_source_columns)}): {sample}")
    if missing_values:
        sample = "; ".join(missing_values[:5])
        raise AssertionError(f"Visual audit rows with sheets have blank recomputation values ({len(missing_values)}): {sample}")
    if missing_sheets:
        sample = ", ".join(missing_sheets[:5])
        raise AssertionError(f"Missing visual audit sheets ({len(missing_sheets)}): {sample}")
    if mismatches:
        sample = ", ".join(mismatches[:5])
        raise AssertionError(f"Recomputed visual case values differ from source CSV ({len(mismatches)}): {sample}")


def _decision_key_set(df: pd.DataFrame) -> set[tuple[str, str, int, str, str, float]]:
    keys: set[tuple[str, str, int, str, str, float]] = set()
    for _, row in df.iterrows():
        keys.add(
            (
                str(row["dataset"]),
                str(row["split"]),
                _to_int(row["label"]),
                str(row["path_a"]),
                str(row["path_b"]),
                round(_to_float(row["target_far"]), 6),
            )
        )
    return keys


def assert_taxonomy_outputs_complete(outdir: Path, decisions: pd.DataFrame, target_far: float = 0.01) -> tuple[int, int]:
    positive_path = outdir / "v2_positive_failure_taxonomy.csv"
    negative_path = outdir / "v2_negative_false_accept_taxonomy.csv"
    if not positive_path.exists() or not negative_path.exists():
        raise AssertionError("Taxonomy CSV outputs are missing.")
    positive = pd.read_csv(positive_path)
    negative = pd.read_csv(negative_path)
    target = _target_far_rows(decisions, target_far=target_far)
    expected_positive = target[(target["label"].astype(int) == 1) & (~target["v2_accepted"].map(_to_bool))].copy()
    expected_negative = target[(target["label"].astype(int) == 0) & (target["v2_accepted"].map(_to_bool))].copy()
    if len(positive) != len(expected_positive):
        raise AssertionError(f"Positive taxonomy count mismatch: {len(positive)} != {len(expected_positive)}")
    if len(negative) != len(expected_negative):
        raise AssertionError(f"Negative false-accept taxonomy count mismatch: {len(negative)} != {len(expected_negative)}")
    if _decision_key_set(positive) != _decision_key_set(expected_positive):
        raise AssertionError("Positive taxonomy keys do not match all v2 positive false rejects.")
    if _decision_key_set(negative) != _decision_key_set(expected_negative):
        raise AssertionError("Negative taxonomy keys do not match all v2 false accepts.")
    return int(len(positive)), int(len(negative))


def assert_frgp_focus_coverage(
    cases: pd.DataFrame,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    required_frgps: tuple[int, ...] = REQUIRED_FRGP_FOCUS_GROUPS,
) -> None:
    with_sheets = cases[cases.get("sheet", pd.Series(dtype=str)).astype(str).str.strip() != ""].copy()
    missing: list[str] = []
    for dataset in datasets:
        for frgp in required_frgps:
            mask = (with_sheets["dataset"].astype(str) == dataset) & (
                pd.to_numeric(with_sheets["focus_frgp"], errors="coerce").astype("Int64") == int(frgp)
            )
            if not bool(mask.any()):
                missing.append(f"{dataset} FRGP {frgp}")
    if missing:
        raise AssertionError(f"Missing FRGP-focused generated sheets for: {', '.join(missing)}")


def _median_text(df: pd.DataFrame, column: str, digits: int = 4) -> str:
    if df.empty or column not in df:
        return ""
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return ""
    return _fmt(float(values.median()), digits)


def _count_mask(df: pd.DataFrame, mask: pd.Series) -> int:
    return int(mask.fillna(False).sum()) if len(mask) else 0


def _focus_count_rows(decisions: pd.DataFrame, focus_frgps: tuple[int, ...]) -> list[dict[str, Any]]:
    target = _with_row_taxonomy(_target_far_rows(decisions, target_far=0.01))
    rows: list[dict[str, Any]] = []
    for (dataset, frgp), group in target[target["frgp"].isin(focus_frgps)].groupby(["dataset", "frgp"], sort=True):
        positives = group[group["label"].astype(int) == 1]
        negatives = group[group["label"].astype(int) == 0]
        false_rejects = positives[~positives["v2_accepted"].map(_to_bool)]
        rows.append(
            {
                "dataset": dataset,
                "frgp": int(frgp),
                "v2_rescued_positives": int((positives["decision_overlap"] == "v2_rescue").sum()),
                "v2_hard_false_rejects": int((false_rejects["score_severity"] == "hard_score_failure").sum()),
                "v2_near_miss_false_rejects": int((false_rejects["score_severity"] == "near_miss").sum()),
                "v2_false_accepts": int(negatives["v2_accepted"].map(_to_bool).sum()),
                "canonical_false_accepts_fixed_by_v2": int((negatives["decision_overlap"] == "v2_fixed_false_accept").sum()),
                "v2_tar": float(positives["v2_accepted"].map(_to_bool).sum() / max(len(positives), 1)),
            }
        )
    return rows


def render_frgp_focus_summary(
    decisions: pd.DataFrame,
    frgp_cases: pd.DataFrame,
    diagnostics: pd.DataFrame,
    *,
    focus_frgps: tuple[int, ...] = FRGP_FOCUS_GROUPS,
) -> str:
    target = _with_row_taxonomy(_target_far_rows(decisions, target_far=0.01))
    focus = target[target["frgp"].isin(focus_frgps)].copy()
    problem = focus[focus["frgp"].isin([5, 10])]
    comparison = focus[focus["frgp"].isin([3, 7])]
    problem_fr = problem[(problem["label"].astype(int) == 1) & (~problem["v2_accepted"].map(_to_bool))]
    comparison_fr = comparison[(comparison["label"].astype(int) == 1) & (~comparison["v2_accepted"].map(_to_bool))]
    problem_hard = problem_fr[problem_fr["score_severity"] == "hard_score_failure"]
    comparison_hard = comparison_fr[comparison_fr["score_severity"] == "hard_score_failure"]
    low_match = int((problem_hard["v2_matches"] < 8).sum())
    many_matches_low_inliers = int(((problem_hard["v2_matches"] >= 24) & (problem_hard["v2_inliers"] < 8)).sum())
    focus_fa = problem[(problem["label"].astype(int) == 0) & (problem["v2_accepted"].map(_to_bool))]
    fa_counts = focus_fa["taxonomy"].value_counts()
    problem_diag = diagnostics[diagnostics["frgp"].isin([5, 10])] if not diagnostics.empty else pd.DataFrame()
    comparison_diag = diagnostics[diagnostics["frgp"].isin([3, 7])] if not diagnostics.empty else pd.DataFrame()
    problem_diag_hard = problem_diag[
        (problem_diag["label"].astype(int) == 1) & (problem_diag["score_severity"] == "hard_score_failure")
    ] if not problem_diag.empty else pd.DataFrame()
    comparison_diag_hard = comparison_diag[
        (comparison_diag["label"].astype(int) == 1) & (comparison_diag["score_severity"] == "hard_score_failure")
    ] if not comparison_diag.empty else pd.DataFrame()

    lines = [
        "# FRGP-Focused SIFT Plain/Roll v2 Visual Audit",
        "",
        "Scope: diagnostic-only review of existing 1% FAR TEST decisions. No matcher behavior, thresholds, canonical SIFT behavior, or UI/default/showcase behavior is changed.",
        "",
        "## Direct Answers",
        "",
        (
            f"- Are FRGP 5/10 failures visually different from FRGP 3/7? Yes in concentration and severity: "
            f"FRGP 5/10 contain {len(problem_hard)} hard v2 false rejects in the focused groups versus "
            f"{len(comparison_hard)} in FRGP 3/7. The sheets indicate a stronger version of the same "
            "plain/roll overlap and geometry stress, not a calibration-only edge case."
        ),
        (
            f"- Is the issue mainly missing overlap/crop? Crop/overlap is implicated, especially where the "
            f"median crop coverage proxy for hard FRGP 5/10 rejects is {_median_text(problem_diag_hard, 'crop_coverage_proxy')} "
            f"versus {_median_text(comparison_diag_hard, 'crop_coverage_proxy')} for FRGP 3/7, but it is not the only factor."
        ),
        (
            f"- Is the issue mainly too few matches? Not mainly: only {low_match}/{len(problem_hard)} hard FRGP 5/10 "
            "false rejects have fewer than 8 tentative Lowe-ratio matches."
        ),
        (
            f"- Is the issue many tentative matches but too few affine inliers? Often yes: {many_matches_low_inliers}/{len(problem_hard)} "
            "hard FRGP 5/10 false rejects have at least 24 tentative matches but fewer than 8 affine inliers."
        ),
        (
            f"- Are false accepts in FRGP 5/10 high-confidence or near-threshold? "
            f"{int(fa_counts.get('high_confidence_false_accept', 0))}/{len(focus_fa)} are high-confidence and "
            f"{int(fa_counts.get('near_threshold_false_accept', 0))}/{len(focus_fa)} are near-threshold by the existing score-margin taxonomy."
        ),
        "",
        "## Focus Counts",
        "",
        "| dataset | FRGP | rescued positives | hard FR | near-miss FR | v2 FA | fixed canonical FA | v2 TAR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in _focus_count_rows(decisions, focus_frgps):
        lines.append(
            f"| {row['dataset']} | {row['frgp']} | {row['v2_rescued_positives']} | "
            f"{row['v2_hard_false_rejects']} | {row['v2_near_miss_false_rejects']} | "
            f"{row['v2_false_accepts']} | {row['canonical_false_accepts_fixed_by_v2']} | {_pct(row['v2_tar'])} |"
        )
    generated = frgp_cases[frgp_cases.get("sheet", pd.Series(dtype=str)).astype(str).str.strip() != ""]
    lines.extend(
        [
            "",
            "## Generated Sheets",
            "",
            f"Generated {len(generated)} focused sheets. See `frgp_focus_cases.csv`; sheet filenames use `visual_audit_sheets/frgp_focus_*.png`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _profile_row(label: str, group: pd.DataFrame) -> dict[str, str]:
    return {
        "label": label,
        "n": str(int(len(group))),
        "median_crop": _median_text(group, "crop_coverage_proxy"),
        "median_area_abs_log": _median_text(group, "foreground_area_ratio_abs_log"),
        "median_matches": _median_text(group, "v2_matches"),
        "median_inliers": _median_text(group, "v2_inliers"),
        "median_inlier_ratio": _median_text(group, "inlier_ratio"),
        "median_matches_per_1000_kp": _median_text(group, "matches_per_1000_min_keypoints"),
    }


def render_overlap_geometry_summary(diagnostics: pd.DataFrame) -> str:
    problem = diagnostics[diagnostics["frgp"].isin([5, 10])].copy()
    comparison = diagnostics[diagnostics["frgp"].isin([3, 7])].copy()
    rescued = diagnostics[(diagnostics["label"].astype(int) == 1) & (diagnostics["decision_overlap"] == "v2_rescue")]
    hard_fr = diagnostics[(diagnostics["label"].astype(int) == 1) & (diagnostics["score_severity"] == "hard_score_failure")]
    near_fr = diagnostics[(diagnostics["label"].astype(int) == 1) & (diagnostics["score_severity"] == "near_miss")]
    high_conf_fa = diagnostics[
        (diagnostics["label"].astype(int) == 0) & (diagnostics["taxonomy"] == "high_confidence_false_accept")
    ]
    true_accepts = diagnostics[(diagnostics["label"].astype(int) == 1) & (diagnostics["v2_accepted"].map(_to_bool))]
    rows = [
        _profile_row("FRGP 5/10", problem),
        _profile_row("FRGP 3/7", comparison),
        _profile_row("v2 rescued positives", rescued),
        _profile_row("hard false rejects", hard_fr),
        _profile_row("near-miss false rejects", near_fr),
        _profile_row("high-confidence false accepts", high_conf_fa),
        _profile_row("true accepts", true_accepts),
    ]
    lines = [
        "# SIFT Plain/Roll v2 Overlap/Crop/Geometry Diagnostics",
        "",
        "Scope: diagnostic-only features computed from existing 1% FAR TEST decisions. These columns are not used for decisions, thresholding, or matcher behavior.",
        "",
        "## Direct Comparisons",
        "",
        (
            f"- FRGP 5/10 vs FRGP 3/7: FRGP 5/10 have lower TAR and more hard false rejects; "
            f"median matches are {_median_text(problem, 'v2_matches')} versus {_median_text(comparison, 'v2_matches')}, "
            f"with median inlier ratios {_median_text(problem, 'inlier_ratio')} versus {_median_text(comparison, 'inlier_ratio')}."
        ),
        (
            f"- v2 rescued positives vs hard false rejects: rescued positives have median "
            f"{_median_text(rescued, 'v2_inliers')} inliers and {_median_text(rescued, 'inlier_ratio')} inlier ratio, "
            f"while hard false rejects have median {_median_text(hard_fr, 'v2_inliers')} inliers and "
            f"{_median_text(hard_fr, 'inlier_ratio')} inlier ratio."
        ),
        (
            f"- High-confidence false accepts vs true accepts: high-confidence FAs have median "
            f"{_median_text(high_conf_fa, 'v2_matches')} matches and {_median_text(high_conf_fa, 'v2_inliers')} inliers, "
            f"versus {_median_text(true_accepts, 'v2_matches')} matches and {_median_text(true_accepts, 'v2_inliers')} inliers for true accepts."
        ),
        (
            f"- Near-miss false rejects vs hard false rejects: near misses have median score margin ratio "
            f"{_median_text(near_fr, 'v2_score_margin_ratio')} and hard rejects have "
            f"{_median_text(hard_fr, 'v2_score_margin_ratio')}; this supports the current finding that calibration alone is unlikely to clear most remaining rejects."
        ),
        "- Affine inlier spatial-spread columns are present but intentionally blank in the full table because the source score CSVs retain counts, not inlier coordinates.",
        "",
        "## Median Diagnostic Profile",
        "",
        "| group | n | crop proxy | abs log area ratio | matches | inliers | inlier ratio | matches/1000 min kp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['n']} | {row['median_crop']} | {row['median_area_abs_log']} | "
            f"{row['median_matches']} | {row['median_inliers']} | {row['median_inlier_ratio']} | "
            f"{row['median_matches_per_1000_kp']} |"
        )
    return "\n".join(lines) + "\n"


def render_decision_overlap_markdown(summary: pd.DataFrame) -> str:
    lines = [
        "# SIFT Plain/Roll v2 Decision Overlap Summary",
        "",
        "All rows are TEST pairs aligned by dataset, split, label, path_a, and path_b. Canonical means SIFT current_score.",
        "",
        "| dataset | target FAR | positive both accept | v2 rescue | v2 lost | positive both reject | negative both reject | new FA | fixed FA | both FA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.sort_values(["dataset", "target_far"]).iterrows():
        lines.append(
            f"| {row['dataset']} | {_pct(row['target_far'])} | "
            f"{int(row['positive_both_accept'])} | {int(row['positive_v2_rescue'])} | "
            f"{int(row['positive_v2_lost'])} | {int(row['positive_both_reject'])} | "
            f"{int(row['negative_both_reject'])} | {int(row['negative_v2_new_false_accept'])} | "
            f"{int(row['negative_v2_fixed_false_accept'])} | {int(row['negative_both_false_accept'])} |"
        )
    return "\n".join(lines) + "\n"


def _counts_table(df: pd.DataFrame, column: str) -> list[str]:
    lines = ["| dataset | category | count |", "| --- | --- | ---: |"]
    if df.empty:
        lines.append("|  | none | 0 |")
        return lines
    counts = df.groupby(["dataset", column]).size().reset_index(name="count").sort_values(["dataset", "count"], ascending=[True, False])
    for _, row in counts.iterrows():
        lines.append(f"| {row['dataset']} | {row[column]} | {int(row['count'])} |")
    return lines


def _frgp_answer(per_frgp: pd.DataFrame) -> str:
    v2 = per_frgp[
        (per_frgp["method"] == "sift_plain_roll_v2")
        & (per_frgp["variant"] == "official_score")
        & (np.isclose(per_frgp["target_far"], 0.01))
    ].copy()
    if v2.empty:
        return "Per-FRGP concentration could not be assessed."
    pieces: list[str] = []
    for dataset, group in v2.groupby("dataset", sort=True):
        group = group.sort_values(["fr", "tar"], ascending=[False, True])
        top = group.head(3)
        top_text = ", ".join(
            f"FRGP {int(row.frgp)}: {int(row.fr)}/{int(row.n_positive)} FR (TAR {_pct(row.tar)})"
            for row in top.itertuples(index=False)
        )
        tar_min = float(group["tar"].min())
        tar_max = float(group["tar"].max())
        pieces.append(f"{dataset}: worst remaining misses are {top_text}; per-finger TAR spans {_pct(tar_min)} to {_pct(tar_max)}.")
    return " ".join(pieces)


def _one_pct_overlap_rows(overlap: pd.DataFrame) -> pd.DataFrame:
    return overlap[np.isclose(overlap["target_far"], 0.01)].copy().sort_values("dataset")


def _dominant_false_accept_answer(false_accepts: pd.DataFrame) -> str:
    if false_accepts.empty:
        return "No v2 false accepts at 1% FAR."
    counts = false_accepts["taxonomy"].value_counts()
    near = int(counts.get("near_threshold_false_accept", 0))
    high = int(counts.get("high_confidence_false_accept", 0))
    return f"At 1% FAR, {near}/{len(false_accepts)} v2 false accepts are near-threshold and {high}/{len(false_accepts)} are high-confidence by the transparent score-margin rules."


def _dominant_false_reject_answer(failures: pd.DataFrame) -> str:
    if failures.empty:
        return "No v2 false rejects at 1% FAR."
    severity = failures["score_severity"].value_counts()
    near = int(severity.get("near_miss", 0))
    hard = int(severity.get("hard_score_failure", 0))
    moderate = int(severity.get("moderate_margin_failure", 0))
    return f"At 1% FAR, remaining v2 false rejects are {near}/{len(failures)} near misses, {moderate}/{len(failures)} moderate-margin failures, and {hard}/{len(failures)} hard score failures."


def render_failure_taxonomy_summary(
    *,
    overlap: pd.DataFrame,
    positive_failures: pd.DataFrame,
    negative_false_accepts: pd.DataFrame,
    per_frgp: pd.DataFrame,
    visual_cases: pd.DataFrame,
    visual_review_notes: str,
    next_direction: str,
) -> str:
    one_pct = _one_pct_overlap_rows(overlap)
    lines = [
        "# SIFT Plain/Roll v2 Pair-Level Failure Taxonomy",
        "",
        "Scope: diagnostic analysis only. This report reuses existing external validation score CSVs and thresholds; it does not tune parameters, change algorithms, change canonical SIFT, or alter UI/default/showcase behavior.",
        "",
        "## Direct Answers",
        "",
    ]
    if one_pct.empty:
        lines.append("- 1% FAR overlap rows were not available.")
    else:
        total_rescue = int(one_pct["positive_v2_rescue"].sum())
        total_lost = int(one_pct["positive_v2_lost"].sum())
        lines.append(
            f"- v2 is mostly adding true positives at 1% FAR rather than trading them away: {total_rescue} rescued positives versus {total_lost} lost positives across SD300B/C."
        )
        for _, row in one_pct.iterrows():
            lines.append(
                f"- {row['dataset']} at 1% FAR: v2 rescues {int(row['positive_v2_rescue'])} positives, loses {int(row['positive_v2_lost'])}, and introduces {int(row['negative_v2_new_false_accept'])} new false accepts; it also fixes {int(row['negative_v2_fixed_false_accept'])} canonical false accepts."
            )
    lines.append(f"- {_dominant_false_accept_answer(negative_false_accepts)}")
    lines.append(f"- {_dominant_false_reject_answer(positive_failures)}")
    lines.append(f"- {_frgp_answer(per_frgp)}")
    if visual_review_notes.strip():
        lines.append(f"- Visual audit read: {visual_review_notes.strip()}")
    else:
        lines.append("- Visual audit read: image sheets were generated for manual inspection; no manual visual note was supplied to this run.")
    if next_direction.strip():
        lines.append(f"- Most likely next research direction: {next_direction.strip()}")
    else:
        lines.append(
            "- Most likely next research direction: D. hybrid decision/fusion, with B/C as follow-up diagnostics; v2 adds many positives at strict FAR but leaves many hard plain-vs-roll failures."
        )
    lines.extend(
        [
            "",
            "## 1% FAR Decision Overlap",
            "",
            "| dataset | canonical TA | v2 TA | v2 rescue | v2 lost | canonical FA | v2 FA | new FA | fixed FA |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in one_pct.iterrows():
        lines.append(
            f"| {row['dataset']} | {int(row['canonical_ta'])} | {int(row['v2_ta'])} | "
            f"{int(row['positive_v2_rescue'])} | {int(row['positive_v2_lost'])} | "
            f"{int(row['canonical_fa'])} | {int(row['v2_fa'])} | "
            f"{int(row['negative_v2_new_false_accept'])} | {int(row['negative_v2_fixed_false_accept'])} |"
        )
    lines.extend(["", "## v2 Positive Failure Taxonomy at 1% FAR", ""])
    lines.extend(_counts_table(positive_failures, "taxonomy"))
    lines.extend(["", "## v2 Positive Score Severity at 1% FAR", ""])
    lines.extend(_counts_table(positive_failures, "score_severity"))
    lines.extend(["", "## v2 False Accept Taxonomy at 1% FAR", ""])
    lines.extend(_counts_table(negative_false_accepts, "taxonomy"))
    lines.extend(
        [
            "",
            "## Visual Audit Bundle",
            "",
            f"Generated case rows: {int(len(visual_cases))}. See `visual_audit_index.md` and `visual_audit_cases.csv`.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_visual_audit_index(visual_cases: pd.DataFrame, outdir: Path) -> str:
    lines = [
        "# SIFT Plain/Roll v2 Failure Taxonomy Visual Audit",
        "",
        "Each sheet shows raw and preprocessed plain/roll images plus SIFT v2 keypoints, Lowe-ratio matches, affine inliers, and text overlays with canonical, SIFT-inliers, and v2 decisions at 1% FAR.",
        "",
        "| group | rank | dataset | label | subject/frgp | canonical | SIFT inliers | v2 | margin | sheet |",
        "| --- | ---: | --- | ---: | --- | --- | --- | --- | ---: | --- |",
    ]
    if visual_cases.empty:
        lines.append("| none | 0 |  |  |  |  |  |  |  |  |")
        return "\n".join(lines) + "\n"
    for _, row in visual_cases.iterrows():
        if not str(row.get("sheet", "")).strip():
            lines.append(f"| {row.get('group', '')} | {row.get('rank', 0)} |  |  | no matching cases |  |  |  |  |  |")
            continue
        rel = str(row.get("sheet_relative") or Path(str(row["sheet"])).relative_to(outdir)).replace("\\", "/")
        subject = f"{row.get('subject_a', '')}->{row.get('subject_b', '')} / {row.get('frgp', '')}"
        canonical = f"{_fmt(row.get('canonical_score'), 5)} >= {_fmt(row.get('canonical_threshold'), 5)}: {_bool_text(row.get('canonical_accepted'))}"
        inliers = f"{_fmt(row.get('sift_inliers_score'), 5)} >= {_fmt(row.get('sift_inliers_threshold'), 5)}: {_bool_text(row.get('sift_inliers_accepted'))}"
        v2 = f"{_fmt(row.get('v2_score'), 5)} >= {_fmt(row.get('v2_threshold'), 5)}: {_bool_text(row.get('v2_accepted'))}"
        lines.append(
            f"| {row.get('group', '')} | {int(row.get('rank', 0))} | {row.get('dataset', '')} | "
            f"{int(row.get('label', 0))} | {subject} | {canonical} | {inliers} | {v2} | "
            f"{_fmt(row.get('v2_score_margin'), 5)} | [{Path(rel).name}]({rel}) |"
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


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": "sift_plain_roll_v2_failure_taxonomy_v1",
            "timestamp_utc": _timestamp_utc(),
            "repo_root": str(REPO_ROOT),
            "outdir": str(path.parent),
        }
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Existing run manifest is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Existing run manifest must contain a JSON object: {path}")
    return loaded


def write_focus_diagnostics_from_existing(
    *,
    outdir: Path,
    datasets: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    decisions_path = outdir / "aligned_test_pair_decisions.csv"
    visual_cases_path = outdir / "visual_audit_cases.csv"
    if not decisions_path.exists():
        raise FileNotFoundError(f"Missing existing aligned decisions CSV: {decisions_path}")
    if not visual_cases_path.exists():
        raise FileNotFoundError(f"Missing existing visual audit cases CSV: {visual_cases_path}")

    decisions = pd.read_csv(decisions_path)
    assert_no_diagnostic_decision_inputs(decisions)
    positive_count, negative_count = assert_taxonomy_outputs_complete(outdir, decisions, target_far=0.01)
    assert_visual_case_sheets_and_recomputed_values(visual_cases_path, outdir)

    focus_frgps = tuple(
        int(item.strip())
        for item in str(getattr(args, "frgp_focus_groups", ",".join(str(x) for x in FRGP_FOCUS_GROUPS))).split(",")
        if item.strip()
    )
    diagnostics = build_overlap_geometry_diagnostics(
        decisions,
        target_far=0.01,
        target_size=int(args.target_size),
        blur_ksize=int(args.blur_ksize),
    )
    frgp_cases = generate_frgp_focus_visual_audit(
        decisions,
        outdir,
        args,
        focus_frgps=focus_frgps,
        target_far=0.01,
    )
    assert_frgp_focus_coverage(frgp_cases, datasets=datasets, required_frgps=REQUIRED_FRGP_FOCUS_GROUPS)
    assert_visual_case_sheets_and_recomputed_values(frgp_cases, outdir)

    paths = {
        "frgp_focus_summary": outdir / "frgp_focus_summary.md",
        "frgp_focus_cases": outdir / "frgp_focus_cases.csv",
        "overlap_geometry_diagnostics": outdir / "overlap_geometry_diagnostics.csv",
        "overlap_geometry_summary": outdir / "overlap_geometry_summary.md",
        "manifest": outdir / "run_manifest.json",
    }
    diagnostics.to_csv(paths["overlap_geometry_diagnostics"], index=False)
    frgp_cases.to_csv(paths["frgp_focus_cases"], index=False)
    paths["frgp_focus_summary"].write_text(
        render_frgp_focus_summary(decisions, frgp_cases, diagnostics, focus_frgps=focus_frgps),
        encoding="utf-8",
    )
    paths["overlap_geometry_summary"].write_text(
        render_overlap_geometry_summary(diagnostics),
        encoding="utf-8",
    )

    command_log = outdir / "command_log.txt"
    with command_log.open("a", encoding="utf-8") as handle:
        handle.write("\nFocused FRGP/overlap diagnostics appended\n")
        handle.write(f"Invocation: {_cmd_text([sys.executable, *sys.argv])}\n")
        handle.write(f"Asserted positive false rejects at 1% FAR: {positive_count}\n")
        handle.write(f"Asserted v2 false accepts at 1% FAR: {negative_count}\n")
        for key, path in paths.items():
            handle.write(f"- {key}: {path}\n")

    manifest = _load_manifest(paths["manifest"])
    manifest["focus_diagnostics"] = {
        "timestamp_utc": _timestamp_utc(),
        "invocation": _cmd_text([sys.executable, *sys.argv]),
        "parameters": {
            "datasets": list(datasets),
            "target_far": 0.01,
            "frgp_focus_groups": [int(x) for x in focus_frgps],
            "required_frgp_focus_groups": [int(x) for x in REQUIRED_FRGP_FOCUS_GROUPS],
            "frgp_focus_top_n": int(args.frgp_focus_top_n),
            "target_size": int(args.target_size),
            "nfeatures": int(args.nfeatures),
            "blur_ksize": int(args.blur_ksize),
            "ratio": float(args.ratio),
            "ransac_thresh": float(args.ransac_thresh),
        },
        "row_counts": {
            "aligned_decision_rows": int(len(decisions)),
            "visual_cases_existing": int(len(pd.read_csv(visual_cases_path))),
            "positive_false_rejects_1pct": int(positive_count),
            "negative_false_accepts_1pct": int(negative_count),
            "frgp_focus_cases": int(len(frgp_cases)),
            "overlap_geometry_diagnostics": int(len(diagnostics)),
        },
        "input_sources": {
            "aligned_decisions": str(decisions_path),
            "visual_cases": str(visual_cases_path),
            "positive_taxonomy": str(outdir / "v2_positive_failure_taxonomy.csv"),
            "negative_taxonomy": str(outdir / "v2_negative_false_accept_taxonomy.csv"),
        },
        "outputs": {key: str(path) for key, path in paths.items() if key != "manifest"},
        "git": _git_info(),
    }
    manifest.setdefault("outputs", {})
    if isinstance(manifest["outputs"], dict):
        manifest["outputs"].update({key: str(path) for key, path in paths.items()})
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def write_report(
    *,
    input_dir: Path,
    outdir: Path,
    datasets: tuple[str, ...],
    target_fars: tuple[float, ...],
    args: argparse.Namespace,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    aligned = build_aligned_test_pairs(input_dir, datasets=datasets)
    thresholds = load_thresholds(input_dir)
    decisions = build_pair_decisions(aligned, thresholds, target_fars=target_fars)
    overlap = summarize_decision_overlap(decisions)
    positive_failures = build_positive_failure_taxonomy(decisions, target_far=0.01)
    negative_false_accepts = build_negative_false_accept_taxonomy(decisions, target_far=0.01)
    per_frgp = build_per_frgp_metrics(decisions)
    visual_cases = pd.DataFrame()
    if not bool(args.skip_visual_audit):
        visual_cases = generate_visual_audit(decisions, outdir, args)

    paths = {
        "failure_summary": outdir / "failure_taxonomy_summary.md",
        "decision_overlap_csv": outdir / "decision_overlap_summary.csv",
        "decision_overlap_md": outdir / "decision_overlap_summary.md",
        "aligned_decisions": outdir / "aligned_test_pair_decisions.csv",
        "positive_taxonomy": outdir / "v2_positive_failure_taxonomy.csv",
        "negative_taxonomy": outdir / "v2_negative_false_accept_taxonomy.csv",
        "per_frgp": outdir / "per_frgp_metrics.csv",
        "visual_cases": outdir / "visual_audit_cases.csv",
        "visual_index": outdir / "visual_audit_index.md",
        "command_log": outdir / "command_log.txt",
        "manifest": outdir / "run_manifest.json",
    }

    decisions.to_csv(paths["aligned_decisions"], index=False)
    overlap.to_csv(paths["decision_overlap_csv"], index=False)
    positive_failures.to_csv(paths["positive_taxonomy"], index=False)
    negative_false_accepts.to_csv(paths["negative_taxonomy"], index=False)
    per_frgp.to_csv(paths["per_frgp"], index=False)
    visual_cases.to_csv(paths["visual_cases"], index=False)
    paths["decision_overlap_md"].write_text(render_decision_overlap_markdown(overlap), encoding="utf-8")
    paths["visual_index"].write_text(render_visual_audit_index(visual_cases, outdir), encoding="utf-8")
    paths["failure_summary"].write_text(
        render_failure_taxonomy_summary(
            overlap=overlap,
            positive_failures=positive_failures,
            negative_false_accepts=negative_false_accepts,
            per_frgp=per_frgp,
            visual_cases=visual_cases,
            visual_review_notes=str(args.visual_review_notes),
            next_direction=str(args.next_direction),
        ),
        encoding="utf-8",
    )

    invocation = _cmd_text([sys.executable, *sys.argv])
    command_log = [
        "SIFT Plain/Roll v2 failure taxonomy command log",
        f"Working directory: {REPO_ROOT}",
        f"Invocation: {invocation}",
        f"Input artifact dir: {input_dir}",
        f"Output dir: {outdir}",
        f"Datasets: {', '.join(datasets)}",
        f"Target FARs: {', '.join(str(x) for x in target_fars)}",
        "Mode: existing score CSVs only; no score generation, parameter tuning, algorithm changes, or UI/default/showcase changes.",
        "",
        "Outputs:",
    ]
    command_log.extend(f"- {key}: {path}" for key, path in paths.items())
    paths["command_log"].write_text("\n".join(command_log) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "sift_plain_roll_v2_failure_taxonomy_v1",
        "timestamp_utc": _timestamp_utc(),
        "repo_root": str(REPO_ROOT),
        "input_dir": str(input_dir),
        "outdir": str(outdir),
        "datasets": list(datasets),
        "target_fars": [float(x) for x in target_fars],
        "alignment_keys": list(PAIR_KEYS),
        "method_variants": [
            {"method": method, "variant": variant, "prefix": prefix}
            for method, variant, prefix in METHOD_VARIANTS
        ],
        "taxonomy_rules": {
            "positive_false_rejects": [
                "near_miss: v2 score margin ratio >= -10%",
                "low_match_failure: matches < 8",
                "possible_geometry_failure: matches >= 24 and inliers < 8",
                "possible_overlap_or_crop_failure: min(k1,k2) >= 1000 and matches < 16",
                "low_inlier_failure: inliers < 6",
                "moderate_margin_failure: -50% <= v2 score margin ratio < -10%",
                "hard_score_failure: v2 score margin ratio < -50%",
            ],
            "negative_false_accepts": [
                "near_threshold_false_accept: v2 score margin ratio <= 10%",
                "high_confidence_false_accept: v2 score margin ratio >= 50%",
                "high_inlier_false_accept: v2 inliers >= SIFT-inliers 1% threshold",
                "high_match_false_accept: matches >= 40",
                "possible_ridge_texture_collision: matches >= 24 and inliers >= 8",
            ],
        },
        "visual_audit": {
            "generated": not bool(args.skip_visual_audit),
            "top_n_per_dataset_group": int(args.visual_top_n),
            "target_size": int(args.target_size),
            "nfeatures": int(args.nfeatures),
            "blur_ksize": int(args.blur_ksize),
            "ratio": float(args.ratio),
            "ransac_thresh": float(args.ransac_thresh),
            "visual_review_notes": str(args.visual_review_notes),
            "next_direction": str(args.next_direction),
        },
        "row_counts": {
            "aligned_base_pairs": int(len(aligned)),
            "aligned_decision_rows": int(len(decisions)),
            "positive_failures_1pct": int(len(positive_failures)),
            "negative_false_accepts_1pct": int(len(negative_false_accepts)),
            "visual_cases": int(len(visual_cases)),
        },
        "git": _git_info(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "outputs": {key: str(path) for key, path in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build pair-level decision overlap and failure taxonomy report for SIFT Plain/Roll v2."
    )
    parser.add_argument("--input_dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    parser.add_argument("--visual_top_n", type=int, default=4)
    parser.add_argument("--skip_visual_audit", action="store_true")
    parser.add_argument(
        "--focus_diagnostics_only",
        action="store_true",
        help="Read existing taxonomy artifacts and write only FRGP focus plus overlap/crop/geometry diagnostics.",
    )
    parser.add_argument("--frgp_focus_top_n", type=int, default=2)
    parser.add_argument("--frgp_focus_groups", default=",".join(str(x) for x in FRGP_FOCUS_GROUPS))
    parser.add_argument("--visual_review_notes", default="")
    parser.add_argument("--next_direction", default="")
    parser.add_argument("--target_size", type=int, default=768)
    parser.add_argument("--nfeatures", type=int, default=3000)
    parser.add_argument("--blur_ksize", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--ransac_thresh", type=float, default=3.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = parse_file_uri(args.input_dir)
    outdir = parse_file_uri(args.outdir)
    datasets = tuple(item.strip() for item in str(args.datasets).split(",") if item.strip())
    if not datasets:
        raise ValueError("No datasets requested.")
    target_fars = tuple(float(item) for item in args.target_far)
    if bool(args.focus_diagnostics_only):
        paths = write_focus_diagnostics_from_existing(
            outdir=outdir,
            datasets=datasets,
            args=args,
        )
    else:
        paths = write_report(
            input_dir=input_dir,
            outdir=outdir,
            datasets=datasets,
            target_fars=target_fars,
            args=args,
        )
    print("Wrote SIFT Plain/Roll v2 failure taxonomy artifacts:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
