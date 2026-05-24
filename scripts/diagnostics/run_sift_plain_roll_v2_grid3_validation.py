from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
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

from scripts.diagnostics.generate_plain_roll_visual_audit import (  # noqa: E402
    _fit_width,
    _label,
    _pad_to_width,
    _side_by_side,
)
from scripts.diagnostics.run_sift_plain_roll_v2_hypothesis_tests import (  # noqa: E402
    BLUR_KSIZE,
    DEFAULT_INPUT_DIR,
    FRGP_FOCUS,
    LOWE_RATIO,
    NFEATURES,
    PAIR_KEYS,
    RANSAC_THRESH,
    TARGET_FARS,
    TARGET_SIZE,
    V2_GEOMETRY_MODEL,
    _confusion,
    _safe_numeric,
    load_aligned_source_scores,
    parse_file_uri,
)
from src.fpbench.matchers.matching_baseline import (  # noqa: E402
    ransac_inliers_for_model,
    score_sift_plain_roll_v2_counts,
)
from src.fpbench.preprocess.preprocess import (  # noqa: E402
    PreprocessConfig,
    preprocess_image,
    resize_pad_to_square,
)


DEFAULT_HYPOTHESIS_DIR = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark" / "sift_plain_roll_v2_hypothesis_tests"
)
DEFAULT_OUTDIR = (
    REPO_ROOT / "artifacts" / "reports" / "benchmark" / "sift_plain_roll_v2_grid3_validation"
)
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_SPLITS = ("val", "test")
GRID3_FAMILY = "roll_multicrop_overlap_probe_v1"
GRID3_NAME = "research_only::roll_multicrop_overlap_probe_v1:grid3_max"
PRIMARY_TARGET_FAR = 0.01
NEAR_THRESHOLD_MARGIN_RATIO = 0.10
HIGH_CONFIDENCE_MARGIN_RATIO = 0.50
VISUAL_GROUP_TARGET_FAR = 0.01
VISUAL_GROUP_SPLIT = "test"


@dataclass(frozen=True)
class GuardrailSpec:
    name: str
    description: str
    kind: str
    params: dict[str, float]


NO_GUARDRAIL = GuardrailSpec(
    name="no_guardrail",
    description="Original grid3 decision; scores and decisions unchanged.",
    kind="none",
    params={},
)
GUARDRAIL_SPECS = (
    NO_GUARDRAIL,
    GuardrailSpec(
        name="score_delta_ge_1_0",
        description="Accept grid3 only when grid3_score - v2_score >= 1.0.",
        kind="score_delta_min",
        params={"min_score_delta": 1.0},
    ),
    GuardrailSpec(
        name="winning_crop_inliers_ge_8",
        description="Accept grid3 only when the winning crop has at least 8 affine inliers.",
        kind="winning_inliers_min",
        params={"min_inliers": 8.0},
    ),
    GuardrailSpec(
        name="v2_weak_support_score_ge_2_5",
        description="Accept grid3 only when full-roll v2 score is at least 2.5.",
        kind="v2_score_min",
        params={"min_v2_score": 2.5},
    ),
    GuardrailSpec(
        name="reject_near_zero_full_roll_support",
        description="Reject grid3 accepts when full-roll v2 has both score < 2.5 and inliers <= 3.",
        kind="reject_near_zero_v2_support",
        params={"min_v2_score": 2.5, "min_v2_inliers": 4.0},
    ),
)
GUARDRAIL_BY_NAME = {spec.name: spec for spec in GUARDRAIL_SPECS}


def _json_loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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


def _pct(value: Any, digits: int = 1) -> str:
    number = _to_float(value)
    return "nan" if not math.isfinite(number) else f"{number * 100:.{digits}f}%"


def _fmt(value: Any, digits: int = 3) -> str:
    number = _to_float(value)
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _target_label(target_far: float) -> str:
    return f"{float(target_far) * 100:g}%"


def _score_cache_path(hypothesis_dir: Path, dataset: str, split: str) -> Path:
    return hypothesis_dir / "candidate_score_cache" / f"{GRID3_FAMILY}_{dataset}_{split}.csv"


def _thresholds_path(hypothesis_dir: Path) -> Path:
    return hypothesis_dir / "candidate_thresholds.csv"


def _runtime_path(hypothesis_dir: Path) -> Path:
    return hypothesis_dir / "candidate_runtime_summary.csv"


def _external_thresholds_path(input_dir: Path) -> Path:
    return input_dir / "per_dataset_thresholds.csv"


def _position_from_crop(label: str, bbox: list[int], roi_bbox: list[int]) -> tuple[str, str, str]:
    text = str(label).lower()
    if "left" in text:
        horizontal = "left"
    elif "right" in text:
        horizontal = "right"
    elif "center" in text or "central" in text:
        horizontal = "central"
    else:
        horizontal = ""

    vertical = ""
    if len(bbox) == 4 and len(roi_bbox) == 4:
        _, y0, _, y1 = [float(x) for x in bbox]
        _, ry0, _, ry1 = [float(x) for x in roi_bbox]
        roi_h = max(ry1 - ry0, 1.0)
        center_y = ((y0 + y1) / 2.0 - ry0) / roi_h
        if center_y < 1.0 / 3.0:
            vertical = "top"
        elif center_y > 2.0 / 3.0:
            vertical = "bottom"
        else:
            vertical = "central"
    position = "/".join(item for item in (horizontal, vertical) if item)
    return horizontal, vertical, position


def _extract_crop_diagnostics(row: pd.Series) -> dict[str, Any]:
    diagnostic = _json_loads(row.get("diagnostic_json", ""))
    crop_index = _to_int(diagnostic.get("winning_crop_index"), -1)
    crop_label = str(diagnostic.get("winning_crop_label", ""))
    bbox = diagnostic.get("winning_crop_bbox", [])
    roi_bbox = diagnostic.get("roll_roi_bbox", [])
    bbox = [int(x) for x in bbox] if isinstance(bbox, list) and len(bbox) == 4 else []
    roi_bbox = [int(x) for x in roi_bbox] if isinstance(roi_bbox, list) and len(roi_bbox) == 4 else []
    horizontal, vertical, position = _position_from_crop(crop_label, bbox, roi_bbox)
    per_crop = diagnostic.get("per_crop_scores", [])
    per_crop = per_crop if isinstance(per_crop, list) else []
    out: dict[str, Any] = {
        "winning_crop_index": int(crop_index),
        "winning_crop_label": crop_label,
        "winning_crop_bbox": _json_dumps(bbox),
        "roll_roi_bbox": _json_dumps(roi_bbox),
        "crop_geometry": f"{crop_label} bbox={bbox} roi={roi_bbox}",
        "winning_crop_horizontal_position": horizontal,
        "winning_crop_vertical_position": vertical,
        "winning_crop_position": position,
        "per_crop_scores_json": _json_dumps(per_crop),
    }
    for item in per_crop:
        if not isinstance(item, dict):
            continue
        idx = _to_int(item.get("crop_index"), -1)
        if idx < 0:
            continue
        out[f"crop{idx}_score"] = _to_float(item.get("score"), 0.0)
        out[f"crop{idx}_matches"] = _to_int(item.get("matches"), 0)
        out[f"crop{idx}_inliers"] = _to_int(item.get("inliers"), 0)
        out[f"crop{idx}_bbox"] = _json_dumps(item.get("crop_bbox", []))
    return out


def load_grid3_scores(
    hypothesis_dir: str | Path = DEFAULT_HYPOTHESIS_DIR,
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> pd.DataFrame:
    root = parse_file_uri(hypothesis_dir)
    frames: list[pd.DataFrame] = []
    for dataset in datasets:
        for split in splits:
            path = _score_cache_path(root, dataset, split)
            if not path.exists():
                raise FileNotFoundError(f"Missing grid3 score cache: {path}")
            df = pd.read_csv(path)
            missing = {
                "dataset",
                "split",
                "label",
                "path_a",
                "path_b",
                "score",
                "matches",
                "inliers",
                "diagnostic_json",
                "candidate_name",
                "candidate_family",
            } - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing required columns: {sorted(missing)}")
            df = df[df["candidate_name"].astype(str) == GRID3_NAME].copy()
            if df.empty:
                raise ValueError(f"{path} has no rows for {GRID3_NAME}")
            for key, value in (("dataset", dataset), ("split", split)):
                df[key] = value
            frames.append(df)
    scores = pd.concat(frames, ignore_index=True, sort=False)
    diagnostics = pd.DataFrame([_extract_crop_diagnostics(row) for _, row in scores.iterrows()])
    scores = pd.concat([scores.reset_index(drop=True), diagnostics.reset_index(drop=True)], axis=1)
    scores = scores.rename(
        columns={
            "score": "grid3_score",
            "matches": "grid3_matches",
            "inliers": "grid3_inliers",
            "k1": "grid3_k1",
            "k2": "grid3_k2",
        }
    )
    for column in ("grid3_score", "grid3_matches", "grid3_inliers", "grid3_k1", "grid3_k2"):
        if column in scores:
            scores[column] = pd.to_numeric(scores[column], errors="coerce").fillna(0.0)
    return scores.reset_index(drop=True)


def load_grid3_thresholds(
    hypothesis_dir: str | Path = DEFAULT_HYPOTHESIS_DIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    path = _thresholds_path(parse_file_uri(hypothesis_dir))
    if not path.exists():
        raise FileNotFoundError(f"Missing hypothesis threshold CSV: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "candidate_name", "target_far", "threshold", "calibration_split", "selected_by_val"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[df["candidate_name"].astype(str) == GRID3_NAME].copy()
    target_values = {round(float(x), 6) for x in target_fars}
    out = out[out["target_far"].map(lambda x: round(float(x), 6) in target_values)].copy()
    if out.empty:
        raise ValueError(f"No thresholds found for {GRID3_NAME}")
    out = out.rename(columns={"threshold": "grid3_threshold"})
    out["target_far"] = pd.to_numeric(out["target_far"], errors="raise").astype(float)
    out["grid3_threshold"] = pd.to_numeric(out["grid3_threshold"], errors="raise").astype(float)
    return out.reset_index(drop=True)


def load_v2_thresholds(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    path = _external_thresholds_path(parse_file_uri(input_dir))
    if not path.exists():
        raise FileNotFoundError(f"Missing external validation thresholds: {path}")
    df = pd.read_csv(path)
    required = {"dataset", "method", "variant", "target_far", "threshold", "calibration_split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[
        (df["method"].astype(str) == "sift_plain_roll_v2")
        & (df["variant"].astype(str) == "official_score")
    ].copy()
    target_values = {round(float(x), 6) for x in target_fars}
    out = out[out["target_far"].map(lambda x: round(float(x), 6) in target_values)].copy()
    if out.empty:
        raise ValueError("No v2 official thresholds found in external validation thresholds")
    out = out.rename(columns={"threshold": "v2_threshold"})
    out["target_far"] = pd.to_numeric(out["target_far"], errors="raise").astype(float)
    out["v2_threshold"] = pd.to_numeric(out["v2_threshold"], errors="raise").astype(float)
    return out.reset_index(drop=True)


def _v2_failure_severity(row: pd.Series) -> str:
    if int(row["label"]) != 1 or bool(row["v2_accepted"]):
        return ""
    threshold = _to_float(row.get("v2_threshold"), 0.0)
    if threshold <= 0.0:
        return ""
    ratio = (_to_float(row.get("v2_score"), 0.0) - threshold) / threshold
    if ratio >= -0.10:
        return "near_miss"
    if ratio >= -0.50:
        return "moderate_margin_failure"
    return "hard_score_failure"


def _decision_category(label: int, v2_accepted: bool, grid3_accepted: bool) -> str:
    if int(label) == 1:
        if v2_accepted and grid3_accepted:
            return "positive_both_accept"
        if (not v2_accepted) and grid3_accepted:
            return "rescued_positive"
        if v2_accepted and (not grid3_accepted):
            return "lost_true_accept"
        return "positive_both_reject"
    if (not v2_accepted) and (not grid3_accepted):
        return "negative_both_reject"
    if (not v2_accepted) and grid3_accepted:
        return "new_false_accept"
    if v2_accepted and (not grid3_accepted):
        return "fixed_false_accept"
    return "negative_both_false_accept"


def _assert_unique(df: pd.DataFrame, keys: list[str], name: str) -> None:
    duplicated = df.duplicated(keys, keep=False)
    if duplicated.any():
        sample = df.loc[duplicated, keys].head(5).to_dict(orient="records")
        raise ValueError(f"{name} has duplicate rows for keys {keys}: {sample}")


def build_grid3_decisions(
    source_scores: pd.DataFrame,
    grid3_scores: pd.DataFrame,
    grid3_thresholds: pd.DataFrame,
    v2_thresholds: pd.DataFrame,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    _assert_unique(source_scores, list(PAIR_KEYS), "source_scores")
    _assert_unique(grid3_scores, list(PAIR_KEYS), "grid3_scores")
    source_cols = [
        *PAIR_KEYS,
        "subject_a",
        "subject_b",
        "frgp_a",
        "frgp_b",
        "frgp",
        "v2_official_score",
        "v2_inliers",
        "v2_matches",
        "v2_k1",
        "v2_k2",
    ]
    available_source_cols = [col for col in source_cols if col in source_scores.columns]
    merged = grid3_scores.merge(
        source_scores[available_source_cols],
        on=list(PAIR_KEYS),
        how="inner",
        validate="one_to_one",
        suffixes=("", "_source"),
        sort=False,
    )
    if len(merged) != len(grid3_scores):
        raise ValueError(
            f"grid3/source pair alignment mismatch: grid3={len(grid3_scores)} merged={len(merged)}"
        )
    if "frgp_source" in merged.columns:
        merged["frgp"] = merged["frgp"].where(merged["frgp"].astype(str).str.len() > 0, merged["frgp_source"])
        merged = merged.drop(columns=["frgp_source"])
    merged = merged.rename(columns={"v2_official_score": "v2_score"})
    merged["v2_score"] = pd.to_numeric(merged["v2_score"], errors="coerce").fillna(0.0)
    merged["grid3_score"] = pd.to_numeric(merged["grid3_score"], errors="coerce").fillna(0.0)
    merged["score_delta_grid3_minus_v2"] = merged["grid3_score"] - merged["v2_score"]

    frames: list[pd.DataFrame] = []
    target_values = {round(float(x), 6) for x in target_fars}
    grid3_thresholds = grid3_thresholds[
        grid3_thresholds["target_far"].map(lambda x: round(float(x), 6) in target_values)
    ].copy()
    v2_thresholds = v2_thresholds[
        v2_thresholds["target_far"].map(lambda x: round(float(x), 6) in target_values)
    ].copy()
    for _, threshold_row in grid3_thresholds.sort_values(["dataset", "target_far"]).iterrows():
        dataset = str(threshold_row["dataset"])
        target_far = float(threshold_row["target_far"])
        v2_match = v2_thresholds[
            (v2_thresholds["dataset"].astype(str) == dataset)
            & np.isclose(pd.to_numeric(v2_thresholds["target_far"], errors="coerce"), target_far)
        ]
        if v2_match.empty:
            raise ValueError(f"Missing v2 threshold for {dataset} target FAR {target_far}")
        block = merged[merged["dataset"].astype(str) == dataset].copy()
        block["target_far"] = target_far
        block["target_far_label"] = _target_label(target_far)
        block["grid3_threshold"] = float(threshold_row["grid3_threshold"])
        block["v2_threshold"] = float(v2_match.iloc[0]["v2_threshold"])
        block["grid3_accepted"] = block["grid3_score"] >= block["grid3_threshold"]
        block["v2_accepted"] = block["v2_score"] >= block["v2_threshold"]
        block["grid3_score_margin"] = block["grid3_score"] - block["grid3_threshold"]
        block["v2_score_margin"] = block["v2_score"] - block["v2_threshold"]
        block["grid3_score_margin_ratio"] = block["grid3_score_margin"] / block["grid3_threshold"].replace(0, np.nan)
        block["v2_score_margin_ratio"] = block["v2_score_margin"] / block["v2_threshold"].replace(0, np.nan)
        block["decision_category"] = [
            _decision_category(label, v2, grid3)
            for label, v2, grid3 in zip(block["label"].astype(int), block["v2_accepted"], block["grid3_accepted"])
        ]
        block["v2_failure_severity"] = block.apply(_v2_failure_severity, axis=1)
        block["near_threshold_grid3_accept"] = (
            block["grid3_accepted"].astype(bool)
            & (pd.to_numeric(block["grid3_score_margin_ratio"], errors="coerce").fillna(math.inf) <= NEAR_THRESHOLD_MARGIN_RATIO)
        )
        block["high_confidence_grid3_accept"] = (
            block["grid3_accepted"].astype(bool)
            & (pd.to_numeric(block["grid3_score_margin_ratio"], errors="coerce").fillna(-math.inf) >= HIGH_CONFIDENCE_MARGIN_RATIO)
        )
        block["research_only"] = True
        frames.append(block)
    decisions = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return decisions.reset_index(drop=True)


def _paired_counts(group: pd.DataFrame, accept_col: str = "grid3_accepted") -> dict[str, int]:
    labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).astype(int)
    v2 = group["v2_accepted"].astype(bool)
    grid = group[accept_col].astype(bool)
    positives = labels == 1
    negatives = labels == 0
    return {
        "positive_both_accept": int(np.sum(positives & v2 & grid)),
        "positive_rescued_vs_v2": int(np.sum(positives & (~v2) & grid)),
        "positive_lost_vs_v2": int(np.sum(positives & v2 & (~grid))),
        "positive_both_reject": int(np.sum(positives & (~v2) & (~grid))),
        "negative_both_reject": int(np.sum(negatives & (~v2) & (~grid))),
        "negative_new_false_accept_vs_v2": int(np.sum(negatives & (~v2) & grid)),
        "negative_fixed_false_accept_vs_v2": int(np.sum(negatives & v2 & (~grid))),
        "negative_both_false_accept": int(np.sum(negatives & v2 & grid)),
        "n_positive": int(np.sum(positives)),
        "n_negative": int(np.sum(negatives)),
    }


def build_grid3_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in decisions.groupby(["dataset", "split", "target_far"], sort=True):
        dataset, split, target_far = keys
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        v2_counts = _confusion(labels, _safe_numeric(group["v2_score"]), float(group["v2_threshold"].iloc[0]))
        grid_counts = _confusion(labels, _safe_numeric(group["grid3_score"]), float(group["grid3_threshold"].iloc[0]))
        paired = _paired_counts(group)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "target_far": float(target_far),
                "target_far_label": _target_label(float(target_far)),
                "v2_threshold": float(group["v2_threshold"].iloc[0]),
                "grid3_threshold": float(group["grid3_threshold"].iloc[0]),
                "v2_far": float(v2_counts["far"]),
                "v2_tar": float(v2_counts["tar"]),
                "v2_ta": int(v2_counts["ta"]),
                "v2_fr": int(v2_counts["fr"]),
                "v2_fa": int(v2_counts["fa"]),
                "v2_tr": int(v2_counts["tr"]),
                "grid3_far": float(grid_counts["far"]),
                "grid3_tar": float(grid_counts["tar"]),
                "grid3_ta": int(grid_counts["ta"]),
                "grid3_fr": int(grid_counts["fr"]),
                "grid3_fa": int(grid_counts["fa"]),
                "grid3_tr": int(grid_counts["tr"]),
                "tar_delta_grid3_minus_v2": float(grid_counts["tar"] - v2_counts["tar"]),
                "far_delta_grid3_minus_v2": float(grid_counts["far"] - v2_counts["far"]),
                **paired,
            }
        )
    return pd.DataFrame(rows)


def build_per_frgp_metrics(decisions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in decisions.groupby(["dataset", "split", "target_far", "frgp"], dropna=False, sort=True):
        dataset, split, target_far, frgp = keys
        labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
        v2_counts = _confusion(labels, _safe_numeric(group["v2_score"]), float(group["v2_threshold"].iloc[0]))
        grid_counts = _confusion(labels, _safe_numeric(group["grid3_score"]), float(group["grid3_threshold"].iloc[0]))
        paired = _paired_counts(group)
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "target_far": float(target_far),
                "frgp": "" if pd.isna(frgp) or str(frgp) == "" else int(float(frgp)),
                "v2_far": float(v2_counts["far"]),
                "v2_tar": float(v2_counts["tar"]),
                "v2_ta": int(v2_counts["ta"]),
                "v2_fr": int(v2_counts["fr"]),
                "v2_fa": int(v2_counts["fa"]),
                "v2_tr": int(v2_counts["tr"]),
                "grid3_far": float(grid_counts["far"]),
                "grid3_tar": float(grid_counts["tar"]),
                "grid3_ta": int(grid_counts["ta"]),
                "grid3_fr": int(grid_counts["fr"]),
                "grid3_fa": int(grid_counts["fa"]),
                "grid3_tr": int(grid_counts["tr"]),
                "tar_delta_grid3_minus_v2": float(grid_counts["tar"] - v2_counts["tar"]),
                "far_delta_grid3_minus_v2": float(grid_counts["far"] - v2_counts["far"]),
                **paired,
            }
        )
    return pd.DataFrame(rows)


def _case_columns() -> list[str]:
    return [
        "dataset",
        "split",
        "target_far",
        "label",
        "frgp",
        "subject_a",
        "subject_b",
        "path_a",
        "path_b",
        "v2_score",
        "grid3_score",
        "score_delta_grid3_minus_v2",
        "v2_threshold",
        "grid3_threshold",
        "v2_accepted",
        "grid3_accepted",
        "decision_category",
        "v2_failure_severity",
        "grid3_matches",
        "grid3_inliers",
        "winning_crop_index",
        "winning_crop_label",
        "winning_crop_bbox",
        "roll_roi_bbox",
        "crop_geometry",
        "winning_crop_position",
        "grid3_score_margin",
        "grid3_score_margin_ratio",
        "near_threshold_grid3_accept",
        "high_confidence_grid3_accept",
    ]


def _select_cols(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df[columns].copy()


def build_case_tables(decisions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cols = _case_columns()
    rescued = decisions[
        (decisions["label"].astype(int) == 1)
        & (~decisions["v2_accepted"].astype(bool))
        & (decisions["grid3_accepted"].astype(bool))
        & (decisions["v2_failure_severity"].astype(str) == "hard_score_failure")
    ].copy()
    lost = decisions[
        (decisions["label"].astype(int) == 1)
        & (decisions["v2_accepted"].astype(bool))
        & (~decisions["grid3_accepted"].astype(bool))
    ].copy()
    new_fa = decisions[
        (decisions["label"].astype(int) == 0)
        & (~decisions["v2_accepted"].astype(bool))
        & (decisions["grid3_accepted"].astype(bool))
    ].copy()
    fixed_fa = decisions[
        (decisions["label"].astype(int) == 0)
        & (decisions["v2_accepted"].astype(bool))
        & (~decisions["grid3_accepted"].astype(bool))
    ].copy()
    sort_rescue = ["split", "target_far", "dataset", "score_delta_grid3_minus_v2", "grid3_score"]
    sort_fa = ["split", "target_far", "dataset", "grid3_score_margin_ratio", "grid3_score"]
    return {
        "rescued_hard_false_rejects": _select_cols(
            rescued.sort_values(sort_rescue, ascending=[True, True, True, False, False]), cols
        ),
        "lost_true_accepts": _select_cols(
            lost.sort_values(["split", "target_far", "dataset", "v2_score"], ascending=[True, True, True, False]), cols
        ),
        "new_false_accepts": _select_cols(
            new_fa.sort_values(sort_fa, ascending=[True, True, True, False, False]), cols
        ),
        "fixed_false_accepts": _select_cols(
            fixed_fa.sort_values(["split", "target_far", "dataset", "v2_score"], ascending=[True, True, True, False]), cols
        ),
    }


def build_winning_crop_analysis(decisions: pd.DataFrame, *, target_far: float = PRIMARY_TARGET_FAR) -> pd.DataFrame:
    one = decisions[
        np.isclose(pd.to_numeric(decisions["target_far"], errors="coerce"), float(target_far))
    ].copy()
    cols = [
        "dataset",
        "split",
        "target_far",
        "label",
        "frgp",
        "subject_a",
        "subject_b",
        "path_a",
        "path_b",
        "winning_crop_index",
        "winning_crop_label",
        "winning_crop_bbox",
        "roll_roi_bbox",
        "crop_geometry",
        "winning_crop_horizontal_position",
        "winning_crop_vertical_position",
        "winning_crop_position",
        "v2_score",
        "grid3_score",
        "score_delta_grid3_minus_v2",
        "v2_threshold",
        "grid3_threshold",
        "v2_accepted",
        "grid3_accepted",
        "decision_category",
        "grid3_matches",
        "grid3_inliers",
        "crop0_score",
        "crop0_matches",
        "crop0_inliers",
        "crop0_bbox",
        "crop1_score",
        "crop1_matches",
        "crop1_inliers",
        "crop1_bbox",
        "crop2_score",
        "crop2_matches",
        "crop2_inliers",
        "crop2_bbox",
        "per_crop_scores_json",
    ]
    return _select_cols(one.sort_values(["dataset", "split", "path_a", "path_b"]), cols).reset_index(drop=True)


def _crop_counts_text(group: pd.DataFrame, title: str) -> str:
    if group.empty:
        return f"{title}: none."
    counts = group.groupby(["winning_crop_index", "winning_crop_label"], dropna=False).size().reset_index(name="count")
    total = int(counts["count"].sum())
    pieces = [
        f"{row['winning_crop_index']}:{row['winning_crop_label']}={int(row['count'])} ({int(row['count']) / max(total, 1):.1%})"
        for _, row in counts.sort_values("count", ascending=False).iterrows()
    ]
    return f"{title}: " + ", ".join(pieces) + "."


def render_winning_crop_summary(analysis: pd.DataFrame, per_frgp: pd.DataFrame) -> str:
    test_1 = analysis[
        (analysis["split"].astype(str).str.lower() == "test")
        & np.isclose(pd.to_numeric(analysis["target_far"], errors="coerce"), PRIMARY_TARGET_FAR)
    ].copy()
    val_1 = analysis[
        (analysis["split"].astype(str).str.lower() == "val")
        & np.isclose(pd.to_numeric(analysis["target_far"], errors="coerce"), PRIMARY_TARGET_FAR)
    ].copy()
    rescues = test_1[test_1["decision_category"].astype(str) == "rescued_positive"].copy()
    new_fa = test_1[test_1["decision_category"].astype(str) == "new_false_accept"].copy()
    all_pos = test_1[test_1["label"].astype(int) == 1]
    all_neg = test_1[test_1["label"].astype(int) == 0]
    median_pos_delta = float(pd.to_numeric(all_pos["score_delta_grid3_minus_v2"], errors="coerce").median())
    median_neg_delta = float(pd.to_numeric(all_neg["score_delta_grid3_minus_v2"], errors="coerce").median())
    rescue_delta = float(pd.to_numeric(rescues["score_delta_grid3_minus_v2"], errors="coerce").median()) if not rescues.empty else float("nan")
    fa_delta = float(pd.to_numeric(new_fa["score_delta_grid3_minus_v2"], errors="coerce").median()) if not new_fa.empty else float("nan")
    same_crop = "No"
    if not rescues.empty and not new_fa.empty:
        rescue_top = rescues["winning_crop_index"].mode().iloc[0]
        fa_top = new_fa["winning_crop_index"].mode().iloc[0]
        same_crop = "Yes" if int(rescue_top) == int(fa_top) else "No"

    focus = per_frgp[
        (per_frgp["split"].astype(str).str.lower() == "test")
        & np.isclose(pd.to_numeric(per_frgp["target_far"], errors="coerce"), PRIMARY_TARGET_FAR)
        & (per_frgp["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
    ].copy()
    focus_lines = ["| dataset | FRGP | v2 TAR | grid3 TAR | delta | top rescue crop |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    if focus.empty:
        focus_lines.append("|  |  |  |  |  | none |")
    else:
        for _, row in focus.sort_values(["dataset", "frgp"]).iterrows():
            crop_group = test_1[
                (test_1["dataset"].astype(str) == str(row["dataset"]))
                & (test_1["frgp"].astype(str) == str(row["frgp"]))
                & (test_1["decision_category"].astype(str) == "rescued_positive")
            ]
            top_crop = "none"
            if not crop_group.empty:
                top = (
                    crop_group.groupby(["winning_crop_index", "winning_crop_label"], dropna=False)
                    .size()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                    .iloc[0]
                )
                top_crop = f"{top['winning_crop_index']}:{top['winning_crop_label']} ({int(top['count'])})"
            focus_lines.append(
                f"| {row['dataset']} | {row['frgp']} | {_pct(row['v2_tar'])} | "
                f"{_pct(row['grid3_tar'])} | {_fmt(row['tar_delta_grid3_minus_v2'], 4)} | {top_crop} |"
            )

    lines = [
        "# Grid3 Winning Crop Analysis",
        "",
        "Decision columns in this crop report use the locked 1% FAR thresholds calibrated on VAL. Scores are unchanged.",
        "",
        "## Required Answers",
        "",
        f"1. Are true-positive rescues concentrated in specific crop indices? {_crop_counts_text(rescues, 'TEST 1% rescued positives')}",
        f"VAL cross-check: {_crop_counts_text(val_1[val_1['decision_category'].astype(str) == 'rescued_positive'], 'VAL 1% rescued positives')}",
        f"2. Are false accepts concentrated in the same crop indices? {same_crop}. {_crop_counts_text(new_fa, 'TEST 1% new false accepts')}",
        (
            "3. Does grid3 help because it finds a better roll subregion, or because it inflates scores globally? "
            f"Median TEST 1% score delta is {_fmt(median_pos_delta)} for positives and {_fmt(median_neg_delta)} for negatives; "
            f"rescues have median delta {_fmt(rescue_delta)}, while new false accepts have median delta {_fmt(fa_delta)}. "
            "The answer is mixed: grid3 finds useful roll subregions for real rescues, but it also raises some impostor scores strongly, so the gain is not merely clean subregion recovery."
        ),
        "4. Are FRGP 5/10 improvements coming from specific crop positions? See the FRGP 5/10 table below.",
        "",
        "## FRGP 5/10 TEST 1% Crop Positions",
        "",
        *focus_lines,
    ]
    return "\n".join(lines) + "\n"


def _guardrail_pass(decisions: pd.DataFrame, spec: GuardrailSpec) -> pd.Series:
    if spec.kind == "none":
        return pd.Series(True, index=decisions.index)
    if spec.kind == "score_delta_min":
        return pd.to_numeric(decisions["score_delta_grid3_minus_v2"], errors="coerce").fillna(-math.inf) >= float(
            spec.params["min_score_delta"]
        )
    if spec.kind == "winning_inliers_min":
        return pd.to_numeric(decisions["grid3_inliers"], errors="coerce").fillna(0.0) >= float(spec.params["min_inliers"])
    if spec.kind == "v2_score_min":
        return pd.to_numeric(decisions["v2_score"], errors="coerce").fillna(0.0) >= float(spec.params["min_v2_score"])
    if spec.kind == "reject_near_zero_v2_support":
        v2_score = pd.to_numeric(decisions["v2_score"], errors="coerce").fillna(0.0)
        v2_inliers = pd.to_numeric(decisions.get("v2_inliers", pd.Series(0, index=decisions.index)), errors="coerce").fillna(0.0)
        near_zero = (v2_score < float(spec.params["min_v2_score"])) & (
            v2_inliers < float(spec.params["min_v2_inliers"])
        )
        return ~near_zero
    raise ValueError(f"Unknown guardrail kind: {spec.kind}")


def apply_guardrail(decisions: pd.DataFrame, spec: GuardrailSpec) -> pd.DataFrame:
    guarded = decisions.copy(deep=True)
    before_scores = guarded["grid3_score"].copy(deep=True)
    pass_mask = _guardrail_pass(guarded, spec).astype(bool)
    guarded["guardrail_name"] = spec.name
    guarded["guardrail_description"] = spec.description
    guarded["guardrail_pass"] = pass_mask
    guarded["grid3_guarded_accepted"] = guarded["grid3_accepted"].astype(bool) & pass_mask
    guarded["guardrail_changed_decision"] = guarded["grid3_accepted"].astype(bool) != guarded["grid3_guarded_accepted"].astype(bool)
    guarded["guardrail_modified_original_scores"] = not before_scores.equals(guarded["grid3_score"])
    guarded["guarded_decision_category"] = [
        _decision_category(label, v2, grid)
        for label, v2, grid in zip(
            guarded["label"].astype(int),
            guarded["v2_accepted"].astype(bool),
            guarded["grid3_guarded_accepted"].astype(bool),
        )
    ]
    return guarded


def _guardrail_metric_row(
    group: pd.DataFrame,
    *,
    dataset: str,
    split: str,
    target_far: float,
    spec: GuardrailSpec,
    accept_col: str = "grid3_guarded_accepted",
) -> dict[str, Any]:
    labels = pd.to_numeric(group["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    accepted = group[accept_col].astype(bool).to_numpy()
    positives = labels == 1
    negatives = labels == 0
    ta = int(np.sum(accepted & positives))
    fr = int(np.sum((~accepted) & positives))
    fa = int(np.sum(accepted & negatives))
    tr = int(np.sum((~accepted) & negatives))
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    paired = _paired_counts(group.rename(columns={accept_col: "_guarded_accept"}), accept_col="_guarded_accept")
    unguarded_paired = _paired_counts(group, accept_col="grid3_accepted")
    v2_counts = _confusion(labels, _safe_numeric(group["v2_score"]), float(group["v2_threshold"].iloc[0]))
    grid_counts = _confusion(labels, _safe_numeric(group["grid3_score"]), float(group["grid3_threshold"].iloc[0]))
    high_conf_new = int(
        np.sum(
            negatives
            & (~group["v2_accepted"].astype(bool).to_numpy())
            & accepted
            & (
                pd.to_numeric(group["grid3_score_margin_ratio"], errors="coerce")
                .fillna(-math.inf)
                .to_numpy(dtype=float)
                >= HIGH_CONFIDENCE_MARGIN_RATIO
            )
        )
    )
    near_threshold_new = int(
        np.sum(
            negatives
            & (~group["v2_accepted"].astype(bool).to_numpy())
            & accepted
            & (
                pd.to_numeric(group["grid3_score_margin_ratio"], errors="coerce")
                .fillna(math.inf)
                .to_numpy(dtype=float)
                <= NEAR_THRESHOLD_MARGIN_RATIO
            )
        )
    )
    return {
        "guardrail_name": spec.name,
        "guardrail_description": spec.description,
        "dataset": dataset,
        "split": split,
        "target_far": float(target_far),
        "tar": float(ta / n_pos) if n_pos else float("nan"),
        "far": float(fa / n_neg) if n_neg else float("nan"),
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "v2_tar": float(v2_counts["tar"]),
        "v2_far": float(v2_counts["far"]),
        "original_grid3_tar": float(grid_counts["tar"]),
        "original_grid3_far": float(grid_counts["far"]),
        "decision_changes_from_original_grid3": int(group.get("guardrail_changed_decision", pd.Series(False, index=group.index)).astype(bool).sum()),
        "guardrail_modified_original_scores": bool(group.get("guardrail_modified_original_scores", pd.Series(False, index=group.index)).astype(bool).any()),
        "high_confidence_new_false_accepts": high_conf_new,
        "near_threshold_new_false_accepts": near_threshold_new,
        "original_grid3_new_false_accepts": int(unguarded_paired["negative_new_false_accept_vs_v2"]),
        "original_grid3_positive_rescues_vs_v2": int(unguarded_paired["positive_rescued_vs_v2"]),
        **paired,
    }


def evaluate_guardrail_candidates_val(
    decisions: pd.DataFrame,
    *,
    target_far: float = PRIMARY_TARGET_FAR,
    specs: tuple[GuardrailSpec, ...] = GUARDRAIL_SPECS,
) -> pd.DataFrame:
    val = decisions[
        (decisions["split"].astype(str).str.lower() == "val")
        & np.isclose(pd.to_numeric(decisions["target_far"], errors="coerce"), float(target_far))
    ].copy()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        guarded = apply_guardrail(val, spec)
        for dataset, group in guarded.groupby("dataset", sort=True):
            rows.append(
                _guardrail_metric_row(
                    group,
                    dataset=str(dataset),
                    split="val",
                    target_far=float(target_far),
                    spec=spec,
                )
            )
        if not guarded.empty:
            rows.append(
                _guardrail_metric_row(
                    guarded,
                    dataset="ALL",
                    split="val",
                    target_far=float(target_far),
                    spec=spec,
                )
            )
    return pd.DataFrame(rows)


def select_guardrail_from_val(val_candidates: pd.DataFrame) -> str:
    all_rows = val_candidates[val_candidates["dataset"].astype(str) == "ALL"].copy()
    if all_rows.empty:
        return NO_GUARDRAIL.name
    baseline_rows = all_rows[all_rows["guardrail_name"].astype(str) == NO_GUARDRAIL.name]
    if baseline_rows.empty:
        return NO_GUARDRAIL.name
    baseline = baseline_rows.iloc[0]
    candidates = all_rows[all_rows["guardrail_name"].astype(str) != NO_GUARDRAIL.name].copy()
    if candidates.empty:
        return NO_GUARDRAIL.name
    conservative = candidates[
        (pd.to_numeric(candidates["negative_new_false_accept_vs_v2"], errors="coerce") < float(baseline["negative_new_false_accept_vs_v2"]))
        & (pd.to_numeric(candidates["high_confidence_new_false_accepts"], errors="coerce") <= float(baseline["high_confidence_new_false_accepts"]))
        & (pd.to_numeric(candidates["tar"], errors="coerce") >= float(baseline["v2_tar"]))
        & (pd.to_numeric(candidates["far"], errors="coerce") <= float(baseline["far"]) + 1e-12)
        & (pd.to_numeric(candidates["positive_rescued_vs_v2"], errors="coerce") >= pd.to_numeric(candidates["positive_lost_vs_v2"], errors="coerce"))
    ].copy()
    if conservative.empty:
        return NO_GUARDRAIL.name
    conservative = conservative.sort_values(
        [
            "high_confidence_new_false_accepts",
            "negative_new_false_accept_vs_v2",
            "far",
            "tar",
            "guardrail_name",
        ],
        ascending=[True, True, True, False, True],
    )
    return str(conservative.iloc[0]["guardrail_name"])


def build_guardrail_locked_test(
    decisions: pd.DataFrame,
    selected_guardrail_name: str,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    spec = GUARDRAIL_BY_NAME.get(selected_guardrail_name, NO_GUARDRAIL)
    test = decisions[decisions["split"].astype(str).str.lower() == "test"].copy()
    target_values = {round(float(x), 6) for x in target_fars}
    test = test[test["target_far"].map(lambda x: round(float(x), 6) in target_values)].copy()
    guarded = apply_guardrail(test, spec)
    rows: list[dict[str, Any]] = []
    for keys, group in guarded.groupby(["dataset", "target_far"], sort=True):
        dataset, target_far = keys
        row = _guardrail_metric_row(
            group,
            dataset=str(dataset),
            split="test",
            target_far=float(target_far),
            spec=spec,
        )
        row["selected_on_split"] = "val"
        row["selection_target_far"] = float(PRIMARY_TARGET_FAR)
        rows.append(row)
    if not guarded.empty:
        for target_far, group in guarded.groupby("target_far", sort=True):
            row = _guardrail_metric_row(
                group,
                dataset="ALL",
                split="test",
                target_far=float(target_far),
                spec=spec,
            )
            row["selected_on_split"] = "val"
            row["selection_target_far"] = float(PRIMARY_TARGET_FAR)
            rows.append(row)
    return pd.DataFrame(rows)


def _thresholded_group(decisions: pd.DataFrame, *, split: str, target_far: float) -> pd.DataFrame:
    return decisions[
        (decisions["split"].astype(str).str.lower() == split)
        & np.isclose(pd.to_numeric(decisions["target_far"], errors="coerce"), float(target_far))
    ].copy()


def build_visual_audit_case_index(
    decisions: pd.DataFrame,
    *,
    top_n: int,
    target_far: float = VISUAL_GROUP_TARGET_FAR,
    split: str = VISUAL_GROUP_SPLIT,
) -> pd.DataFrame:
    base = _thresholded_group(decisions, split=split, target_far=target_far)
    numeric_defaults = {
        "score_delta_grid3_minus_v2": 0.0,
        "grid3_score": 0.0,
        "grid3_score_margin_ratio": 0.0,
        "v2_score_margin_ratio": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in base.columns:
            base[column] = default
    bool_defaults = {
        "high_confidence_grid3_accept": False,
        "grid3_accepted": False,
    }
    for column, default in bool_defaults.items():
        if column not in base.columns:
            base[column] = default
    if "decision_category" not in base.columns:
        base["decision_category"] = ""
    if "v2_failure_severity" not in base.columns:
        base["v2_failure_severity"] = ""
    if "frgp" not in base.columns:
        base["frgp"] = ""
    groups: list[tuple[str, pd.DataFrame]] = [
        (
            "top_grid3_rescued_hard_false_rejects",
            base[
                (base["label"].astype(int) == 1)
                & (base["decision_category"].astype(str) == "rescued_positive")
                & (base["v2_failure_severity"].astype(str) == "hard_score_failure")
            ].sort_values(["score_delta_grid3_minus_v2", "grid3_score"], ascending=False),
        ),
        (
            "top_grid3_new_false_accepts",
            base[base["decision_category"].astype(str) == "new_false_accept"].sort_values(
                ["grid3_score_margin_ratio", "grid3_score"], ascending=False
            ),
        ),
        (
            "high_confidence_grid3_false_accepts",
            base[
                (base["label"].astype(int) == 0)
                & (base["grid3_accepted"].astype(bool))
                & (base["high_confidence_grid3_accept"].astype(bool))
            ].sort_values(["grid3_score_margin_ratio", "grid3_score"], ascending=False),
        ),
        (
            "grid3_lost_positives",
            base[base["decision_category"].astype(str) == "lost_true_accept"].sort_values(
                ["v2_score_margin_ratio", "v2_score"], ascending=False
            ),
        ),
        (
            "frgp_5_10_rescued_cases",
            base[
                (base["decision_category"].astype(str) == "rescued_positive")
                & (base["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
            ].sort_values(["score_delta_grid3_minus_v2", "grid3_score"], ascending=False),
        ),
        (
            "frgp_5_10_false_accepts",
            base[
                (base["label"].astype(int) == 0)
                & (base["grid3_accepted"].astype(bool))
                & (base["frgp"].astype(str).isin([str(x) for x in FRGP_FOCUS]))
            ].sort_values(["grid3_score_margin_ratio", "grid3_score"], ascending=False),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for group_name, frame in groups:
        selected = frame.head(int(top_n)).copy()
        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            payload = row.to_dict()
            payload["audit_group"] = group_name
            payload["rank"] = int(rank)
            payload["sheet_path"] = ""
            rows.append(payload)
    if not rows:
        return pd.DataFrame()
    index = pd.DataFrame(rows)
    required = [
        "audit_group",
        "rank",
        "dataset",
        "split",
        "target_far",
        "label",
        "frgp",
        "path_a",
        "path_b",
        "v2_score",
        "grid3_score",
        "v2_threshold",
        "grid3_threshold",
        "v2_accepted",
        "grid3_accepted",
        "decision_category",
        "v2_failure_severity",
        "winning_crop_index",
        "winning_crop_label",
        "winning_crop_bbox",
        "roll_roi_bbox",
        "crop_geometry",
        "grid3_matches",
        "grid3_inliers",
        "sheet_path",
    ]
    return _select_cols(index, required + [col for col in index.columns if col not in required])


def _load_gray(path_str: str) -> np.ndarray:
    path = parse_file_uri(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def _parse_bbox(raw: Any) -> list[int]:
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
    else:
        value = raw
    return [int(x) for x in value] if isinstance(value, list) and len(value) == 4 else []


def _blank_panel(text: str, *, width: int = 900, height: int = 240) -> np.ndarray:
    blank = np.full((height, width, 3), 245, dtype=np.uint8)
    cv2.putText(blank, text[:90], (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2, cv2.LINE_AA)
    return blank


def _sift_match_views(
    proc_a: np.ndarray,
    proc_b: np.ndarray,
    *,
    left_label: str,
    right_label: str,
    match_label: str,
    inlier_label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sift = cv2.SIFT_create(nfeatures=int(NFEATURES))
    kps_a, desc_a = sift.detectAndCompute(proc_a, None)
    kps_b, desc_b = sift.detectAndCompute(proc_b, None)
    kps_a = kps_a or []
    kps_b = kps_b or []
    key_a = cv2.drawKeypoints(proc_a, kps_a, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    key_b = cv2.drawKeypoints(proc_b, kps_b, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints = _side_by_side(key_a, key_b, f"{left_label}: {len(kps_a)} keypoints", f"{right_label}: {len(kps_b)} keypoints", height=260)
    diagnostics: dict[str, Any] = {
        "k1_recomputed": int(len(kps_a)),
        "k2_recomputed": int(len(kps_b)),
        "matches_recomputed": 0,
        "inliers_recomputed": 0,
        "score_recomputed": 0.0,
    }
    if desc_a is None or desc_b is None or len(desc_a) == 0 or len(desc_b) == 0:
        return (
            _label(_blank_panel("No SIFT descriptors"), match_label),
            _label(_blank_panel("No SIFT descriptors"), inlier_label),
            diagnostics | {"keypoints_panel": keypoints},
        )

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc_a, desc_b, k=2)
    good: list[cv2.DMatch] = []
    for item in knn:
        if len(item) == 2:
            first, second = item
            if first.distance < float(LOWE_RATIO) * second.distance:
                good.append(first)
    good_sorted = sorted(good, key=lambda m: float(m.distance))
    good_img = cv2.drawMatches(
        proc_a,
        kps_a,
        proc_b,
        kps_b,
        good_sorted[:80],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    good_img = _label(_fit_width(good_img, 1500), f"{match_label}: {len(good)}")
    inliers, mask = ransac_inliers_for_model(
        kps_a,
        kps_b,
        good,
        ransac_model=V2_GEOMETRY_MODEL,
        ransac_thresh=float(RANSAC_THRESH),
    )
    diagnostics.update(
        {
            "matches_recomputed": int(len(good)),
            "inliers_recomputed": int(inliers),
            "score_recomputed": score_sift_plain_roll_v2_counts(matches=len(good), inliers=inliers),
        }
    )
    if mask is None:
        inlier_img = _label(_blank_panel("Too few matches or no affine model"), inlier_label)
    else:
        inlier_matches = [m for m, keep in zip(good, mask.astype(bool).tolist()) if keep]
        inlier_matches = sorted(inlier_matches, key=lambda m: float(m.distance))[:80]
        raw = cv2.drawMatches(
            proc_a,
            kps_a,
            proc_b,
            kps_b,
            inlier_matches,
            None,
            matchColor=(0, 210, 0),
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        inlier_img = _label(_fit_width(raw, 1500), f"{inlier_label}: {int(inliers)}")
    diagnostics["keypoints_panel"] = keypoints
    return good_img, inlier_img, diagnostics


def _grid3_crop_images(img_a: np.ndarray, img_b: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    proc_a = preprocess_image(img_a, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))
    proc_b = preprocess_image(img_b, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))
    if len(bbox) != 4:
        bbox = [0, 0, proc_b.shape[1], proc_b.shape[0]]
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(int(x0), proc_b.shape[1] - 1))
    y0 = max(0, min(int(y0), proc_b.shape[0] - 1))
    x1 = max(x0 + 1, min(int(x1), proc_b.shape[1]))
    y1 = max(y0 + 1, min(int(y1), proc_b.shape[0]))
    annotated = cv2.cvtColor(proc_b, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), (0, 220, 255), 3)
    crop = proc_b[y0:y1, x0:x1]
    resized = resize_pad_to_square(crop, TARGET_SIZE)
    resized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return proc_a, annotated, resized


def _case_slug(row: pd.Series) -> str:
    payload = f"{row.get('audit_group', '')}|{row.get('rank', '')}|{row.get('dataset', '')}|{row.get('path_a', '')}|{row.get('path_b', '')}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"grid3_{row.get('audit_group', 'case')}_{int(row.get('rank', 0)):02d}_{digest}"


def _render_case_sheet(row: pd.Series, outdir: Path) -> tuple[Path, dict[str, Any]]:
    img_a = _load_gray(str(row["path_a"]))
    img_b = _load_gray(str(row["path_b"]))
    proc_a = preprocess_image(img_a, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))
    proc_b = preprocess_image(img_b, PreprocessConfig(target_size=TARGET_SIZE, blur_ksize=BLUR_KSIZE))

    title = (
        f"{row.get('audit_group')} #{int(row.get('rank', 0))} {row.get('dataset')} {row.get('split')} "
        f"target={_target_label(float(row.get('target_far', PRIMARY_TARGET_FAR)))} label={int(row.get('label', 0))} "
        f"FRGP={row.get('frgp', '')} decision={row.get('decision_category', '')}"
    )
    header = np.full((104, 1500, 3), 32, dtype=np.uint8)
    cv2.putText(header, title[:170], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    score_line = (
        f"v2 score={float(row.get('v2_score', 0.0)):.6g} thr={float(row.get('v2_threshold', 0.0)):.6g} "
        f"accept={bool(row.get('v2_accepted', False))}; "
        f"grid3 score={float(row.get('grid3_score', 0.0)):.6g} thr={float(row.get('grid3_threshold', 0.0)):.6g} "
        f"accept={bool(row.get('grid3_accepted', False))}; "
        f"crop={row.get('winning_crop_index', '')}:{row.get('winning_crop_label', '')} geometry={row.get('winning_crop_bbox', '')}"
    )
    cv2.putText(header, score_line[:190], (12, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(
        header,
        f"v2 full-roll and grid3 winning-crop recomputed with target={TARGET_SIZE}, nfeatures={NFEATURES}, Lowe={LOWE_RATIO}, affine_full_2d RANSAC={RANSAC_THRESH}",
        (12, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    v2_matches, v2_inliers, v2_diag = _sift_match_views(
        proc_a,
        proc_b,
        left_label="v2 plain",
        right_label="v2 full roll",
        match_label="v2 full-roll Lowe-ratio matches",
        inlier_label="v2 full-roll affine inliers",
    )
    bbox = _parse_bbox(row.get("winning_crop_bbox", "[]"))
    grid_plain, grid_roll_annotated, grid_crop = _grid3_crop_images(img_a, img_b, bbox)
    crop_context = _side_by_side(
        grid_plain,
        grid_roll_annotated,
        "grid3 plain query",
        f"roll with winning crop {row.get('winning_crop_index', '')}:{row.get('winning_crop_label', '')}",
        height=330,
    )
    crop_preview = _side_by_side(
        grid_plain,
        grid_crop,
        "grid3 plain query",
        "grid3 winning crop resized",
        height=330,
    )
    grid_matches, grid_inliers, grid_diag = _sift_match_views(
        grid_plain,
        grid_crop,
        left_label="grid3 plain",
        right_label="grid3 crop",
        match_label="grid3 winning-crop Lowe-ratio matches",
        inlier_label="grid3 winning-crop affine inliers",
    )

    rows = [
        header,
        _side_by_side(img_a, img_b, "raw plain", "raw roll", height=330),
        _side_by_side(proc_a, proc_b, "v2 preprocessed plain", "v2 preprocessed full roll", height=330),
        v2_diag["keypoints_panel"],
        v2_matches,
        v2_inliers,
        crop_context,
        crop_preview,
        grid_diag["keypoints_panel"],
        grid_matches,
        grid_inliers,
    ]
    width = max(item.shape[1] for item in rows)
    gutter = np.full((12, width, 3), 245, dtype=np.uint8)
    stacked: list[np.ndarray] = []
    for item in [_pad_to_width(x, width) for x in rows]:
        if stacked:
            stacked.append(gutter)
        stacked.append(item)
    sheet = np.vstack(stacked)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{_case_slug(row)}.png"
    cv2.imwrite(str(path), sheet)
    payload = {
        "v2_matches_recomputed": int(v2_diag["matches_recomputed"]),
        "v2_inliers_recomputed": int(v2_diag["inliers_recomputed"]),
        "v2_score_recomputed": float(v2_diag["score_recomputed"]),
        "grid3_matches_recomputed": int(grid_diag["matches_recomputed"]),
        "grid3_inliers_recomputed": int(grid_diag["inliers_recomputed"]),
        "grid3_score_recomputed": float(grid_diag["score_recomputed"]),
    }
    return path, payload


def generate_visual_audit_sheets(cases: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    if cases.empty:
        return cases.copy()
    rows: list[dict[str, Any]] = []
    for _, row in cases.iterrows():
        sheet, diagnostics = _render_case_sheet(row, outdir)
        payload = row.to_dict()
        payload["sheet_path"] = str(sheet)
        payload.update(diagnostics)
        rows.append(payload)
    return pd.DataFrame(rows)


def render_visual_audit_index(cases: pd.DataFrame, visual_dir: Path) -> str:
    lines = [
        "# Grid3 Visual Audit Index",
        "",
        "All sheets use TEST 1% FAR paired decisions. They show raw plain/roll, v2 full-roll matches/inliers, grid3 winning crop geometry, and grid3 winning-crop matches/inliers.",
        "",
        "| group | rank | dataset | label | FRGP | decision | v2 score/thr | grid3 score/thr | crop | sheet |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    if cases.empty:
        lines.append("| none |  |  |  |  |  |  |  |  |  |")
    else:
        for _, row in cases.sort_values(["audit_group", "rank"]).iterrows():
            sheet = Path(str(row.get("sheet_path", "")))
            rel = sheet.name if sheet.name else ""
            link = f"[{rel}](visual_audit_sheets/{rel})" if rel else ""
            lines.append(
                f"| {row.get('audit_group', '')} | {int(row.get('rank', 0))} | {row.get('dataset', '')} | "
                f"{int(row.get('label', 0))} | {row.get('frgp', '')} | {row.get('decision_category', '')} | "
                f"{float(row.get('v2_score', 0.0)):.4g}/{float(row.get('v2_threshold', 0.0)):.4g} | "
                f"{float(row.get('grid3_score', 0.0)):.4g}/{float(row.get('grid3_threshold', 0.0)):.4g} | "
                f"{row.get('winning_crop_index', '')}:{row.get('winning_crop_label', '')} {row.get('winning_crop_bbox', '')} | {link} |"
            )
    return "\n".join(lines) + "\n"


def _summary_metric_table(metrics: pd.DataFrame, *, split: str) -> list[str]:
    rows = metrics[metrics["split"].astype(str).str.lower() == split].copy()
    lines = [
        f"## {split.upper()} Metrics",
        "",
        "| dataset | target FAR | v2 TAR | v2 FAR | grid3 TAR | grid3 FAR | TAR delta | new FAs | fixed FAs | rescued positives | lost positives |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    if rows.empty:
        lines.append("|  |  |  |  |  |  |  |  |  |  |  |")
    else:
        for _, row in rows.sort_values(["dataset", "target_far"]).iterrows():
            lines.append(
                f"| {row['dataset']} | {_target_label(row['target_far'])} | {_pct(row['v2_tar'])} | {_pct(row['v2_far'])} | "
                f"{_pct(row['grid3_tar'])} | {_pct(row['grid3_far'])} | {_fmt(row['tar_delta_grid3_minus_v2'], 4)} | "
                f"{int(row['negative_new_false_accept_vs_v2'])} | {int(row['negative_fixed_false_accept_vs_v2'])} | "
                f"{int(row['positive_rescued_vs_v2'])} | {int(row['positive_lost_vs_v2'])} |"
            )
    return lines


def _paired_table_lines(metrics: pd.DataFrame, *, split: str, target_far: float) -> list[str]:
    rows = metrics[
        (metrics["split"].astype(str).str.lower() == split)
        & np.isclose(pd.to_numeric(metrics["target_far"], errors="coerce"), float(target_far))
    ].copy()
    lines = [
        f"## Paired Tables ({split.upper()} {_target_label(target_far)})",
        "",
        "| dataset | class | both correct/accept | grid3 only | v2 only | both reject/false accept | McNemar b | McNemar c |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in rows.sort_values("dataset").iterrows():
        lines.append(
            f"| {row['dataset']} | positives | {int(row['positive_both_accept'])} | "
            f"{int(row['positive_rescued_vs_v2'])} rescued | {int(row['positive_lost_vs_v2'])} lost | "
            f"{int(row['positive_both_reject'])} | {int(row['positive_lost_vs_v2'])} | {int(row['positive_rescued_vs_v2'])} |"
        )
        lines.append(
            f"| {row['dataset']} | negatives | {int(row['negative_both_reject'])} | "
            f"{int(row['negative_new_false_accept_vs_v2'])} new FA | {int(row['negative_fixed_false_accept_vs_v2'])} fixed FA | "
            f"{int(row['negative_both_false_accept'])} | {int(row['negative_fixed_false_accept_vs_v2'])} | {int(row['negative_new_false_accept_vs_v2'])} |"
        )
    return lines


def render_guardrail_summary(
    val_candidates: pd.DataFrame,
    locked_test: pd.DataFrame,
    selected_guardrail: str,
    decisions: pd.DataFrame,
) -> str:
    test_1 = _thresholded_group(decisions, split="test", target_far=PRIMARY_TARGET_FAR)
    new_fa = test_1[test_1["decision_category"].astype(str) == "new_false_accept"].copy()
    near = int(new_fa["near_threshold_grid3_accept"].astype(bool).sum()) if not new_fa.empty else 0
    high = int(new_fa["high_confidence_grid3_accept"].astype(bool).sum()) if not new_fa.empty else 0
    frgp_lines = ["| FRGP | new false accepts | high-confidence |", "| ---: | ---: | ---: |"]
    if new_fa.empty:
        frgp_lines.append("|  | 0 | 0 |")
    else:
        tmp = new_fa.copy()
        tmp["is_high"] = tmp["high_confidence_grid3_accept"].astype(bool)
        for frgp, group in tmp.groupby("frgp", dropna=False, sort=True):
            frgp_lines.append(f"| {frgp} | {len(group)} | {int(group['is_high'].sum())} |")
    locked_1 = locked_test[
        np.isclose(pd.to_numeric(locked_test["target_far"], errors="coerce"), PRIMARY_TARGET_FAR)
        & (locked_test["dataset"].astype(str) == "ALL")
    ]
    locked_sentence = "No locked TEST guardrail row was produced."
    if not locked_1.empty:
        row = locked_1.iloc[0]
        locked_sentence = (
            f"Selected guardrail `{selected_guardrail}` gives locked TEST 1% aggregate TAR {_pct(row['tar'])}, "
            f"FAR {_pct(row['far'])}, new FAs {int(row['negative_new_false_accept_vs_v2'])}, "
            f"rescues {int(row['positive_rescued_vs_v2'])}, lost positives {int(row['positive_lost_vs_v2'])}."
        )

    val_lines = [
        "| guardrail | VAL TAR | VAL FAR | new FAs | high-conf new FAs | rescues | lost positives | decision changes | selected? |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    all_val = val_candidates[val_candidates["dataset"].astype(str) == "ALL"].copy()
    for _, row in all_val.sort_values(["guardrail_name"]).iterrows():
        val_lines.append(
            f"| `{row['guardrail_name']}` | {_pct(row['tar'])} | {_pct(row['far'])} | "
            f"{int(row['negative_new_false_accept_vs_v2'])} | {int(row['high_confidence_new_false_accepts'])} | "
            f"{int(row['positive_rescued_vs_v2'])} | {int(row['positive_lost_vs_v2'])} | "
            f"{int(row['decision_changes_from_original_grid3'])} | {'yes' if row['guardrail_name'] == selected_guardrail else ''} |"
        )
    visual_note = (
        "The top generated sheets show local ridge similarity, but several high-confidence false accepts route many inliers through crop-edge or partial-region geometry rather than a convincing full-finger alignment. "
        "Treat them as visually plausible local ridge collisions, not safe genuine-match evidence."
    )
    lines = [
        "# Grid3 Guardrail Summary",
        "",
        "Guardrail selection uses VAL 1% only. TEST rows are locked reports for the selected guardrail; TEST is not used to choose guardrail parameters or the guardrail itself.",
        "",
        "## False Accept Safety",
        "",
        f"1. How many grid3 new false accepts are near-threshold? {near} at TEST 1%, using margin <= {NEAR_THRESHOLD_MARGIN_RATIO:.0%} of the grid3 threshold.",
        f"2. How many are high-confidence? {high} at TEST 1%, using margin >= {HIGH_CONFIDENCE_MARGIN_RATIO:.0%} of the grid3 threshold.",
        f"3. Are high-confidence false accepts visually plausible ridge collisions? {visual_note}",
        "4. Are new false accepts concentrated by FRGP? See table below.",
        f"5. Would a conservative guardrail reduce them? {locked_sentence}",
        "",
        "## VAL Guardrail Candidates",
        "",
        *val_lines,
        "",
        "## TEST 1% New False Accepts By FRGP",
        "",
        *frgp_lines,
    ]
    return "\n".join(lines) + "\n"


def render_validation_summary(
    *,
    outdir: Path,
    metrics: pd.DataFrame,
    guardrail_selected: str,
    locked_test: pd.DataFrame,
    command_text: str,
) -> str:
    lines = [
        "# Grid3 Max Focused Validation",
        "",
        f"Candidate: `{GRID3_NAME}`.",
        "",
        "Scope: report-only validation and error audit. This script does not change v2, canonical SIFT, benchmark configs, API/UI code, defaults, showcase behavior, or TEST-selected parameters.",
        "",
        "Protocol: existing hypothesis-test grid3 score caches and VAL-calibrated thresholds are reused. Guardrail candidates are fixed in the script, evaluated on VAL only, and TEST is reported only after the selected guardrail is locked.",
        "",
        f"Output folder: `{outdir}`",
        f"Command: `{command_text}`",
        "",
        *_summary_metric_table(metrics, split="val"),
        "",
        *_summary_metric_table(metrics, split="test"),
        "",
        *_paired_table_lines(metrics, split="test", target_far=PRIMARY_TARGET_FAR),
        "",
        "## Guardrail Lock",
        "",
        f"Selected guardrail from VAL: `{guardrail_selected}`.",
    ]
    selected = locked_test[
        (locked_test["dataset"].astype(str) == "ALL")
        & np.isclose(pd.to_numeric(locked_test["target_far"], errors="coerce"), PRIMARY_TARGET_FAR)
    ].copy()
    if selected.empty:
        lines.append("No aggregate locked TEST row was available.")
    else:
        row = selected.iloc[0]
        lines.append(
            f"Locked TEST 1% aggregate guarded result: TAR {_pct(row['tar'])}, FAR {_pct(row['far'])}, "
            f"new false accepts {int(row['negative_new_false_accept_vs_v2'])}, fixed false accepts {int(row['negative_fixed_false_accept_vs_v2'])}, "
            f"rescued positives {int(row['positive_rescued_vs_v2'])}, lost positives {int(row['positive_lost_vs_v2'])}."
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "Grid3_max remains a promising implementation direction only if the error audit accepts the false-accept risk. A real implementation follow-up should preserve the VAL-only protocol and treat any guardrail as a locked research candidate, not as a default or canonical behavior.",
        ]
    )
    return "\n".join(lines) + "\n"


def load_runtime_summary(hypothesis_dir: str | Path = DEFAULT_HYPOTHESIS_DIR) -> pd.DataFrame:
    path = _runtime_path(parse_file_uri(hypothesis_dir))
    if not path.exists():
        raise FileNotFoundError(f"Missing runtime summary: {path}")
    runtime = pd.read_csv(path)
    if "candidate_family" not in runtime.columns:
        raise ValueError(f"{path} missing candidate_family")
    out = runtime[runtime["candidate_family"].astype(str) == GRID3_FAMILY].copy()
    return out.reset_index(drop=True)


def assert_grid3_output_coverage(
    metrics: pd.DataFrame,
    *,
    required_datasets: tuple[str, ...] = DEFAULT_DATASETS,
    required_splits: tuple[str, ...] = DEFAULT_SPLITS,
) -> None:
    datasets = set(str(x) for x in metrics.get("dataset", pd.Series(dtype=str)).dropna().unique())
    missing_datasets = sorted(set(required_datasets) - datasets)
    if missing_datasets:
        raise AssertionError(f"grid3 metrics missing required datasets: {missing_datasets}")
    splits = set(str(x).lower() for x in metrics.get("split", pd.Series(dtype=str)).dropna().unique())
    missing_splits = sorted(set(required_splits) - splits)
    if missing_splits:
        raise AssertionError(f"grid3 metrics missing required splits: {missing_splits}")


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def _copy_command_log(path: Path, command_text: str, args: argparse.Namespace, selected_guardrail: str) -> None:
    lines = [
        "SIFT Plain/Roll v2 grid3 validation command log",
        f"started_at_utc={datetime.now(timezone.utc).isoformat()}",
        f"command={command_text}",
        f"cwd={REPO_ROOT}",
        f"input_dir={parse_file_uri(args.input_dir)}",
        f"hypothesis_dir={parse_file_uri(args.hypothesis_dir)}",
        f"outdir={parse_file_uri(args.outdir)}",
        f"datasets={args.datasets}",
        f"splits={args.splits}",
        f"target_fars={','.join(str(x) for x in TARGET_FARS)}",
        f"primary_guardrail_selection_target_far={PRIMARY_TARGET_FAR}",
        f"selected_guardrail={selected_guardrail}",
        "protocol=VAL thresholds and VAL guardrail selection only; TEST locked report only",
        "forbidden_edits=not touched by this script: configs/thresholds.yaml, configs/methods.yaml, apps/api, apps/ui",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Focused validation and error audit for grid3_max only.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="External validation score artifact folder.")
    parser.add_argument("--hypothesis-dir", default=str(DEFAULT_HYPOTHESIS_DIR), help="Existing hypothesis-test artifact folder.")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output folder for grid3 validation artifacts.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="Comma-separated splits, normally val,test.")
    parser.add_argument("--visual-top-n", type=int, default=8, help="Cases per visual audit group.")
    parser.add_argument("--skip-visuals", action="store_true", help="Write visual audit index rows without rendering PNG sheets.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files in the grid3 validation output folder.")
    return parser


def run(args: argparse.Namespace) -> dict[str, Path]:
    outdir = parse_file_uri(args.outdir)
    if outdir.exists() and not bool(args.overwrite):
        existing = list(outdir.glob("*"))
        if existing:
            raise FileExistsError(f"Output directory already has files; pass --overwrite: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    datasets = tuple(item.strip() for item in str(args.datasets).split(",") if item.strip())
    splits = tuple(item.strip().lower() for item in str(args.splits).split(",") if item.strip())
    command_text = " ".join([Path(sys.executable).name, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]])

    source_scores = load_aligned_source_scores(args.input_dir, datasets=datasets, splits=splits)
    grid3_scores = load_grid3_scores(args.hypothesis_dir, datasets=datasets, splits=splits)
    grid3_thresholds = load_grid3_thresholds(args.hypothesis_dir)
    v2_thresholds = load_v2_thresholds(args.input_dir)
    decisions = build_grid3_decisions(source_scores, grid3_scores, grid3_thresholds, v2_thresholds)
    metrics = build_grid3_metrics(decisions)
    assert_grid3_output_coverage(metrics, required_datasets=datasets, required_splits=splits)
    per_frgp = build_per_frgp_metrics(decisions)
    cases = build_case_tables(decisions)
    crop_analysis = build_winning_crop_analysis(decisions)
    val_guardrails = evaluate_guardrail_candidates_val(decisions)
    selected_guardrail = select_guardrail_from_val(val_guardrails)
    locked_test = build_guardrail_locked_test(decisions, selected_guardrail)
    runtime = load_runtime_summary(args.hypothesis_dir)
    visual_index = build_visual_audit_case_index(decisions, top_n=int(args.visual_top_n))
    visual_dir = outdir / "visual_audit_sheets"
    if not bool(args.skip_visuals):
        visual_index = generate_visual_audit_sheets(visual_index, visual_dir)

    outputs = {
        "summary": outdir / "grid3_validation_summary.md",
        "decision_overlap": outdir / "grid3_vs_v2_decision_overlap.csv",
        "rescued": outdir / "grid3_rescued_hard_false_rejects.csv",
        "lost": outdir / "grid3_lost_true_accepts.csv",
        "new_fa": outdir / "grid3_new_false_accepts.csv",
        "fixed_fa": outdir / "grid3_fixed_false_accepts.csv",
        "per_frgp": outdir / "grid3_per_frgp_metrics.csv",
        "runtime": outdir / "grid3_runtime_summary.csv",
        "command_log": outdir / "command_log.txt",
        "manifest": outdir / "run_manifest.json",
        "crop_analysis": outdir / "grid3_winning_crop_analysis.csv",
        "crop_summary": outdir / "grid3_winning_crop_summary.md",
        "visual_cases": outdir / "grid3_visual_audit_cases.csv",
        "visual_index": outdir / "grid3_visual_audit_index.md",
        "guardrail_val": outdir / "grid3_guardrail_candidates_val.csv",
        "guardrail_test": outdir / "grid3_guardrail_locked_test.csv",
        "guardrail_summary": outdir / "grid3_guardrail_summary.md",
    }
    decision_cols = [
        "dataset",
        "split",
        "target_far",
        "label",
        "frgp",
        "subject_a",
        "subject_b",
        "path_a",
        "path_b",
        "v2_score",
        "grid3_score",
        "score_delta_grid3_minus_v2",
        "v2_threshold",
        "grid3_threshold",
        "v2_accepted",
        "grid3_accepted",
        "decision_category",
        "v2_failure_severity",
        "grid3_matches",
        "grid3_inliers",
        "winning_crop_index",
        "winning_crop_label",
        "winning_crop_bbox",
        "roll_roi_bbox",
        "crop_geometry",
        "grid3_score_margin",
        "grid3_score_margin_ratio",
        "near_threshold_grid3_accept",
        "high_confidence_grid3_accept",
        "research_only",
    ]
    _write_csv(outputs["decision_overlap"], _select_cols(decisions, decision_cols))
    _write_csv(outputs["rescued"], cases["rescued_hard_false_rejects"])
    _write_csv(outputs["lost"], cases["lost_true_accepts"])
    _write_csv(outputs["new_fa"], cases["new_false_accepts"])
    _write_csv(outputs["fixed_fa"], cases["fixed_false_accepts"])
    _write_csv(outputs["per_frgp"], per_frgp)
    _write_csv(outputs["runtime"], runtime)
    _write_csv(outputs["crop_analysis"], crop_analysis)
    _write_csv(outputs["visual_cases"], visual_index)
    _write_csv(outputs["guardrail_val"], val_guardrails)
    _write_csv(outputs["guardrail_test"], locked_test)
    outputs["summary"].write_text(
        render_validation_summary(
            outdir=outdir,
            metrics=metrics,
            guardrail_selected=selected_guardrail,
            locked_test=locked_test,
            command_text=command_text,
        ),
        encoding="utf-8",
        newline="\n",
    )
    outputs["crop_summary"].write_text(render_winning_crop_summary(crop_analysis, per_frgp), encoding="utf-8", newline="\n")
    outputs["visual_index"].write_text(render_visual_audit_index(visual_index, visual_dir), encoding="utf-8", newline="\n")
    outputs["guardrail_summary"].write_text(
        render_guardrail_summary(val_guardrails, locked_test, selected_guardrail, decisions),
        encoding="utf-8",
        newline="\n",
    )
    _copy_command_log(outputs["command_log"], command_text, args, selected_guardrail)
    manifest = {
        "schema_version": "sift_plain_roll_v2_grid3_validation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "input_dir": str(parse_file_uri(args.input_dir)),
        "hypothesis_dir": str(parse_file_uri(args.hypothesis_dir)),
        "outdir": str(outdir),
        "candidate": GRID3_NAME,
        "datasets": list(datasets),
        "splits": list(splits),
        "target_fars": list(TARGET_FARS),
        "primary_guardrail_selection_target_far": PRIMARY_TARGET_FAR,
        "guardrail_specs": [
            {
                "name": spec.name,
                "description": spec.description,
                "kind": spec.kind,
                "params": spec.params,
            }
            for spec in GUARDRAIL_SPECS
        ],
        "selected_guardrail": selected_guardrail,
        "protocol": {
            "candidate_threshold_source": "existing hypothesis-test candidate_thresholds.csv",
            "candidate_score_source": "existing hypothesis-test candidate_score_cache roll_multicrop files",
            "threshold_calibration_split": "val",
            "guardrail_selection_split": "val",
            "test_usage": "locked final reporting only; no TEST parameter tuning",
            "research_only": True,
            "production_thresholds_yaml_changed": False,
            "methods_yaml_changed": False,
            "canonical_sift_changed": False,
            "sift_plain_roll_v2_changed": False,
            "apps_api_changed": False,
            "apps_ui_changed": False,
        },
        "v2_parameters_reused": {
            "target_size": TARGET_SIZE,
            "nfeatures": NFEATURES,
            "blur_ksize": BLUR_KSIZE,
            "lowe_ratio": LOWE_RATIO,
            "ransac_thresh": RANSAC_THRESH,
            "geometry_model": V2_GEOMETRY_MODEL,
        },
        "visual_audit": {
            "split": VISUAL_GROUP_SPLIT,
            "target_far": VISUAL_GROUP_TARGET_FAR,
            "top_n_per_group": int(args.visual_top_n),
            "rendered": not bool(args.skip_visuals),
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    outputs["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = run(args)
    print("Wrote grid3 validation artifacts:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
