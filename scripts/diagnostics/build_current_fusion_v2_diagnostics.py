from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

for _optional_pandas_module in ("numexpr", "bottleneck"):
    sys.modules.setdefault(_optional_pandas_module, None)

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_METHOD = "sourceafis_sift_quality_deep_fusion_v2"
CANONICAL_BENCHMARK_REL = Path(
    "artifacts/reports/benchmark/sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep"
)
CANONICAL_SCORES_REL = CANONICAL_BENCHMARK_REL / "scores"
DEFAULT_TAXONOMY_OUTDIR_REL = Path(
    "artifacts/reports/diagnostics/sourceafis_sift_quality_deep_fusion_v2_current_failure_taxonomy"
)
DEFAULT_OUTCOMES_OUTDIR_REL = Path(
    "artifacts/reports/diagnostics/true_accept_failures_across_methods_current"
)
DATASETS = ("nist_sd300b", "nist_sd300c")
SPLITS = ("val", "test")
TARGET_FAR = 0.01
EXPECTED_TEST_COUNTS = {
    "nist_sd300b": {"pairs": 3556, "positives": 889, "negatives": 2667},
    "nist_sd300c": {"pairs": 3556, "positives": 889, "negatives": 2667},
}
EXPECTED_FUSION_TEST_METRICS = {
    "nist_sd300b": {"TAR": 753 / 889, "FAR": 23 / 2667},
    "nist_sd300c": {"TAR": 757 / 889, "FAR": 24 / 2667},
}
FUSION_METRIC_TOLERANCE = 0.001
FUSION_ALIAS = "fusion_v2"
PRIMARY_BASELINE_ALIAS = "sourceafis"


@dataclass(frozen=True)
class MethodSpec:
    alias: str
    score_column: str
    display_name: str


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    false_accepts: int
    actual_far: float
    negative_count: int
    positive_count: int
    minimum_negatives_for_target: int
    enough_negatives_for_target: bool


METHOD_SPECS = (
    MethodSpec(FUSION_ALIAS, "score", "Fusion v2 final score"),
    MethodSpec(PRIMARY_BASELINE_ALIAS, "sourceafis_score", "SourceAFIS score"),
    MethodSpec("sift_score", "sift_score", "SIFT score"),
    MethodSpec("sift_inliers", "sift_inliers", "SIFT inliers"),
    MethodSpec("sift_matches", "sift_matches", "SIFT matches"),
    MethodSpec("deep_score", "deep_score", "Deep score"),
    MethodSpec("deep_logit", "deep_logit", "Deep logit"),
)
METHOD_BY_ALIAS = {spec.alias: spec for spec in METHOD_SPECS}
CONTEXT_COLUMNS = [
    "dataset",
    "split",
    "pair_id",
    "label",
    "subject_a",
    "subject_b",
    "finger_position",
    "frgp",
    "path_a",
    "path_b",
]
TAXONOMY_FLAGS = [
    "fusion_rescued_positive_from_sourceafis",
    "fusion_lost_positive_vs_sourceafis",
    "fusion_fixed_sourceafis_false_accept",
    "fusion_new_false_accept_vs_sourceafis",
    "common_false_reject_all_methods",
    "fusion_false_reject_rescued_by_component",
]


class CurrentDiagnosticsError(ValueError):
    """Raised when current diagnostics cannot be produced safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _score_path(repo_root: Path, dataset: str, split: str) -> Path:
    return (
        repo_root
        / CANONICAL_SCORES_REL
        / f"scores_{dataset}_{CANONICAL_METHOD}_{split}.csv"
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _minimum_negatives_for_target(target_far: float) -> int:
    if float(target_far) <= 0:
        return -1
    return int(math.ceil(1.0 / float(target_far)))


def select_threshold_from_val_negatives(
    val_labels: pd.Series,
    val_scores: pd.Series,
    *,
    target_far: float,
) -> ThresholdSelection:
    labels = pd.to_numeric(val_labels, errors="coerce").fillna(-1).astype(int)
    scores = pd.to_numeric(val_scores, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(scores) & np.isin(labels.to_numpy(dtype=int), [0, 1])
    labels_arr = labels.to_numpy(dtype=int)[valid]
    scores_arr = scores[valid]
    negatives = scores_arr[labels_arr == 0]
    positives = scores_arr[labels_arr == 1]
    negative_count = int(negatives.size)
    positive_count = int(positives.size)
    minimum_negatives = _minimum_negatives_for_target(float(target_far))
    enough_negatives = bool(negative_count >= minimum_negatives) if minimum_negatives > 0 else False
    if negative_count == 0:
        raise CurrentDiagnosticsError("Cannot calibrate threshold without VAL negatives.")

    for threshold in sorted(float(value) for value in np.unique(negatives)):
        false_accepts = int(np.sum(negatives >= threshold))
        actual_far = float(false_accepts / negative_count)
        if actual_far <= float(target_far) + 1e-15:
            return ThresholdSelection(
                threshold=float(threshold),
                false_accepts=false_accepts,
                actual_far=actual_far,
                negative_count=negative_count,
                positive_count=positive_count,
                minimum_negatives_for_target=minimum_negatives,
                enough_negatives_for_target=enough_negatives,
            )

    threshold = math.nextafter(float(np.max(negatives)), math.inf)
    return ThresholdSelection(
        threshold=float(threshold),
        false_accepts=0,
        actual_far=0.0,
        negative_count=negative_count,
        positive_count=positive_count,
        minimum_negatives_for_target=minimum_negatives,
        enough_negatives_for_target=enough_negatives,
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def confusion_at_threshold(labels: pd.Series, scores: pd.Series, threshold: float) -> dict[str, Any]:
    label_arr = pd.to_numeric(labels, errors="coerce").fillna(-1).astype(int).to_numpy(dtype=int)
    score_arr = pd.to_numeric(scores, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(score_arr) & np.isin(label_arr, [0, 1])
    label_arr = label_arr[valid]
    score_arr = score_arr[valid]
    positive = label_arr == 1
    negative = label_arr == 0
    accepted = score_arr >= float(threshold)
    ta = int(np.sum(accepted & positive))
    fr = int(np.sum(~accepted & positive))
    fa = int(np.sum(accepted & negative))
    tr = int(np.sum(~accepted & negative))
    positives = int(np.sum(positive))
    negatives = int(np.sum(negative))
    return {
        "n_pairs": int(label_arr.size),
        "positives": positives,
        "negatives": negatives,
        "TA": ta,
        "FR": fr,
        "FA": fa,
        "TR": tr,
        "TAR": _safe_rate(ta, positives),
        "FAR": _safe_rate(fa, negatives),
        "FRR": _safe_rate(fr, positives),
        "TNR": _safe_rate(tr, negatives),
    }


def _outcome_series(labels: pd.Series, accepted: pd.Series) -> pd.Series:
    labels_int = pd.to_numeric(labels, errors="coerce").fillna(-1).astype(int)
    accept_bool = accepted.astype(bool)
    return pd.Series(
        np.select(
            [
                (labels_int == 1) & accept_bool,
                (labels_int == 1) & ~accept_bool,
                (labels_int == 0) & accept_bool,
                (labels_int == 0) & ~accept_bool,
            ],
            ["TA", "FR", "FA", "TR"],
            default="INVALID",
        ),
        index=labels.index,
    )


def read_score_file(path: Path, *, dataset: str, split: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical score file: {path}")
    df = pd.read_csv(path)
    required = set(CONTEXT_COLUMNS) | {spec.score_column for spec in METHOD_SPECS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise CurrentDiagnosticsError(f"{path} is missing required column(s): {missing}")

    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out = out[(out["dataset"] == dataset) & (out["split"] == split)].copy()
    if out.empty:
        raise CurrentDiagnosticsError(f"{path} has no rows for dataset={dataset}, split={split}.")
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    for column in ("finger_position", "frgp", "subject_a", "subject_b"):
        out[column] = out[column].astype(str).str.strip()
    for spec in METHOD_SPECS:
        out[spec.score_column] = pd.to_numeric(out[spec.score_column], errors="coerce")
        if out[spec.score_column].isna().any():
            examples = out.loc[out[spec.score_column].isna(), ["dataset", "split", "pair_id"]].head(5)
            raise CurrentDiagnosticsError(
                f"{path} has non-numeric values in {spec.score_column}: {examples.to_dict('records')}"
            )
    dup = out.duplicated(["dataset", "split", "pair_id"], keep=False)
    if bool(dup.any()):
        examples = out.loc[dup, ["dataset", "split", "pair_id"]].head(5).to_dict("records")
        raise CurrentDiagnosticsError(f"{path} has duplicate pair keys: {examples}")
    return out.reset_index(drop=True)


def load_score_frames(repo_root: Path, datasets: Iterable[str]) -> tuple[dict[tuple[str, str], pd.DataFrame], dict[str, Any]]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    manifest_files: dict[str, Any] = {}
    for dataset in datasets:
        for split in SPLITS:
            path = _score_path(repo_root, dataset, split)
            frame = read_score_file(path, dataset=str(dataset), split=split)
            frames[(str(dataset), split)] = frame
            labels = frame["label"].astype(int)
            manifest_files[f"{dataset}_{split}"] = {
                "path": str(path),
                "sha256": _sha256(path),
                "rows": int(len(frame)),
                "positives": int(labels.eq(1).sum()),
                "negatives": int(labels.eq(0).sum()),
            }
    return frames, manifest_files


def validate_test_counts(frames: dict[tuple[str, str], pd.DataFrame], datasets: Iterable[str]) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for dataset in datasets:
        frame = frames[(str(dataset), "test")]
        labels = frame["label"].astype(int)
        counts = {
            "pairs": int(len(frame)),
            "positives": int(labels.eq(1).sum()),
            "negatives": int(labels.eq(0).sum()),
        }
        expected = EXPECTED_TEST_COUNTS[str(dataset)]
        if counts != expected:
            raise CurrentDiagnosticsError(
                f"{dataset} TEST protocol count mismatch. Expected {expected}, got {counts}."
            )
        rows[str(dataset)] = counts
    return rows


def build_thresholds_and_metrics(
    frames: dict[tuple[str, str], pd.DataFrame],
    *,
    repo_root: Path,
    datasets: Iterable[str],
    target_far: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    threshold_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for dataset in datasets:
        val = frames[(str(dataset), "val")]
        for spec in METHOD_SPECS:
            selection = select_threshold_from_val_negatives(
                val["label"],
                val[spec.score_column],
                target_far=float(target_far),
            )
            threshold_rows.append(
                {
                    "method_alias": spec.alias,
                    "method_display_name": spec.display_name,
                    "score_column": spec.score_column,
                    "dataset": str(dataset),
                    "target_far": float(target_far),
                    "threshold": float(selection.threshold),
                    "calibration_split": "val",
                    "calibration_negative_count": int(selection.negative_count),
                    "calibration_positive_count": int(selection.positive_count),
                    "calibration_false_accepts": int(selection.false_accepts),
                    "calibration_far": float(selection.actual_far),
                    "minimum_negatives_for_target": int(selection.minimum_negatives_for_target),
                    "enough_negatives_for_target": bool(selection.enough_negatives_for_target),
                    "selection_rule": "lowest VAL negative-score threshold with VAL FAR <= target",
                    "higher_is_more_similar": True,
                    "scores_csv": str(_score_path(repo_root, str(dataset), "val")),
                }
            )
            for split in SPLITS:
                split_frame = frames[(str(dataset), split)]
                counts = confusion_at_threshold(
                    split_frame["label"],
                    split_frame[spec.score_column],
                    selection.threshold,
                )
                metric_rows.append(
                    {
                        "method_alias": spec.alias,
                        "method_display_name": spec.display_name,
                        "score_column": spec.score_column,
                        "dataset": str(dataset),
                        "split": split,
                        "target_far": float(target_far),
                        "threshold": float(selection.threshold),
                        "threshold_source_split": "val",
                        **counts,
                    }
                )
    return pd.DataFrame(threshold_rows), pd.DataFrame(metric_rows)


def validate_fusion_metrics(metrics: pd.DataFrame, *, target_far: float) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    for dataset, expected in EXPECTED_FUSION_TEST_METRICS.items():
        row = metrics[
            (metrics["method_alias"] == FUSION_ALIAS)
            & (metrics["dataset"] == dataset)
            & (metrics["split"] == "test")
            & (metrics["target_far"].astype(float).round(12) == round(float(target_far), 12))
        ]
        if row.empty:
            raise CurrentDiagnosticsError(f"Missing Fusion v2 TEST metrics for {dataset}.")
        record = row.iloc[0].to_dict()
        actual_tar = float(record["TAR"])
        actual_far = float(record["FAR"])
        if abs(actual_tar - expected["TAR"]) > FUSION_METRIC_TOLERANCE:
            raise CurrentDiagnosticsError(
                f"{dataset} Fusion v2 TEST TAR mismatch. Expected about {expected['TAR']:.6f}, got {actual_tar:.6f}."
            )
        if abs(actual_far - expected["FAR"]) > FUSION_METRIC_TOLERANCE:
            raise CurrentDiagnosticsError(
                f"{dataset} Fusion v2 TEST FAR mismatch. Expected about {expected['FAR']:.6f}, got {actual_far:.6f}."
            )
        validation[dataset] = {
            "target_far": float(target_far),
            "threshold": float(record["threshold"]),
            "TA": int(record["TA"]),
            "FR": int(record["FR"]),
            "FA": int(record["FA"]),
            "TR": int(record["TR"]),
            "TAR": actual_tar,
            "FAR": actual_far,
            "expected_TAR": float(expected["TAR"]),
            "expected_FAR": float(expected["FAR"]),
        }
    return validation


def build_all_method_outcomes(
    frames: dict[tuple[str, str], pd.DataFrame],
    thresholds: pd.DataFrame,
    *,
    datasets: Iterable[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for dataset in datasets:
        test = frames[(str(dataset), "test")]
        for spec in METHOD_SPECS:
            threshold_row = thresholds[
                (thresholds["dataset"] == str(dataset)) & (thresholds["method_alias"] == spec.alias)
            ].iloc[0]
            threshold = float(threshold_row["threshold"])
            accepted = test[spec.score_column].astype(float) >= threshold
            out = test[CONTEXT_COLUMNS].copy()
            out["target_far"] = float(threshold_row["target_far"])
            out["method_alias"] = spec.alias
            out["method_display_name"] = spec.display_name
            out["score_column"] = spec.score_column
            out["score"] = test[spec.score_column].astype(float)
            out["threshold"] = threshold
            out["threshold_source_split"] = "val"
            out["score_margin"] = out["score"] - threshold
            out["accepted"] = accepted.astype(bool)
            out["outcome"] = _outcome_series(out["label"], out["accepted"])
            rows.append(out)
    return pd.concat(rows, ignore_index=True, sort=False)


def summarize_method_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["method_alias", "method_display_name", "score_column", "dataset", "target_far", "threshold"]
    for keys, sub in outcomes.groupby(group_cols, dropna=False, sort=True):
        record = dict(zip(group_cols, keys))
        labels = sub["label"].astype(int)
        positives = int(labels.eq(1).sum())
        negatives = int(labels.eq(0).sum())
        ta = int(sub["outcome"].eq("TA").sum())
        fr = int(sub["outcome"].eq("FR").sum())
        fa = int(sub["outcome"].eq("FA").sum())
        tr = int(sub["outcome"].eq("TR").sum())
        record.update(
            {
                "split": "test",
                "pairs": int(len(sub)),
                "positives": positives,
                "negatives": negatives,
                "TA": ta,
                "FR": fr,
                "FA": fa,
                "TR": tr,
                "TAR": _safe_rate(ta, positives),
                "FAR": _safe_rate(fa, negatives),
                "FRR": _safe_rate(fr, positives),
                "TNR": _safe_rate(tr, negatives),
                "threshold_source_split": "val",
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _method_lists_from_matrix(row: pd.Series, aliases: Iterable[str], *, accepted: bool) -> str:
    wanted: list[str] = []
    for alias in aliases:
        if bool(row[f"{alias}_accepted"]) is bool(accepted):
            wanted.append(alias)
    return ",".join(wanted)


def build_positive_pair_matrix(outcomes: pd.DataFrame) -> pd.DataFrame:
    aliases = [spec.alias for spec in METHOD_SPECS]
    positives = outcomes[outcomes["label"].astype(int) == 1].copy()
    context = positives[CONTEXT_COLUMNS + ["target_far"]].drop_duplicates(["dataset", "pair_id", "target_far"])
    matrix = context.reset_index(drop=True)
    for alias in aliases:
        sub = positives[positives["method_alias"] == alias][
            ["dataset", "pair_id", "target_far", "score", "threshold", "score_margin", "accepted", "outcome"]
        ].rename(
            columns={
                "score": f"{alias}_score",
                "threshold": f"{alias}_threshold",
                "score_margin": f"{alias}_score_margin",
                "accepted": f"{alias}_accepted",
                "outcome": f"{alias}_outcome",
            }
        )
        matrix = matrix.merge(sub, on=["dataset", "pair_id", "target_far"], how="left", validate="one_to_one")

    accept_cols = [f"{alias}_accepted" for alias in aliases]
    outcome_cols = [f"{alias}_outcome" for alias in aliases]
    matrix["true_accept_method_count"] = matrix[outcome_cols].eq("TA").sum(axis=1).astype(int)
    matrix["false_reject_method_count"] = matrix[outcome_cols].eq("FR").sum(axis=1).astype(int)
    matrix["all_methods_false_reject"] = matrix["false_reject_method_count"].eq(len(aliases))
    matrix["accepted_by_methods"] = matrix.apply(
        lambda row: _method_lists_from_matrix(row, aliases, accepted=True),
        axis=1,
    )
    matrix["rejected_by_methods"] = matrix.apply(
        lambda row: _method_lists_from_matrix(row, aliases, accepted=False),
        axis=1,
    )
    return matrix


def build_common_false_rejects(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix[matrix["all_methods_false_reject"].astype(bool)].copy()


def build_method_specific_false_rejects(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aliases = [spec.alias for spec in METHOD_SPECS]
    for record in matrix.to_dict("records"):
        if int(record["true_accept_method_count"]) <= 0:
            continue
        for alias in aliases:
            if record.get(f"{alias}_outcome") != "FR":
                continue
            rows.append(
                {
                    "dataset": record["dataset"],
                    "split": record["split"],
                    "pair_id": record["pair_id"],
                    "target_far": record["target_far"],
                    "method_alias": alias,
                    "score": record.get(f"{alias}_score"),
                    "threshold": record.get(f"{alias}_threshold"),
                    "score_margin": record.get(f"{alias}_score_margin"),
                    "accepted_by_methods": record["accepted_by_methods"],
                    "rejected_by_methods": record["rejected_by_methods"],
                    "true_accept_method_count": record["true_accept_method_count"],
                    "false_reject_method_count": record["false_reject_method_count"],
                    "subject_a": record["subject_a"],
                    "subject_b": record["subject_b"],
                    "finger_position": record["finger_position"],
                    "frgp": record["frgp"],
                    "path_a": record["path_a"],
                    "path_b": record["path_b"],
                }
            )
    return pd.DataFrame(rows)


def build_pairwise_complementarity(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    aliases = [spec.alias for spec in METHOD_SPECS]
    for dataset, sub in matrix.groupby("dataset", sort=True):
        positives = int(len(sub))
        for base in aliases:
            for other in aliases:
                if base == other:
                    continue
                base_outcome = sub[f"{base}_outcome"]
                other_outcome = sub[f"{other}_outcome"]
                both_ta = int((base_outcome.eq("TA") & other_outcome.eq("TA")).sum())
                both_fr = int((base_outcome.eq("FR") & other_outcome.eq("FR")).sum())
                base_fr_other_ta = int((base_outcome.eq("FR") & other_outcome.eq("TA")).sum())
                base_ta_other_fr = int((base_outcome.eq("TA") & other_outcome.eq("FR")).sum())
                union_ta = int((base_outcome.eq("TA") | other_outcome.eq("TA")).sum())
                rows.append(
                    {
                        "dataset": dataset,
                        "target_far": float(sub["target_far"].iloc[0]),
                        "base_method": base,
                        "other_method": other,
                        "positives": positives,
                        "base_TA": int(base_outcome.eq("TA").sum()),
                        "base_FR": int(base_outcome.eq("FR").sum()),
                        "other_TA": int(other_outcome.eq("TA").sum()),
                        "other_FR": int(other_outcome.eq("FR").sum()),
                        "both_TA": both_ta,
                        "both_FR": both_fr,
                        "base_FR_other_TA_rescued_by_other": base_fr_other_ta,
                        "base_TA_other_FR_lost_to_other": base_ta_other_fr,
                        "union_TA": union_ta,
                        "union_TAR": _safe_rate(union_ta, positives),
                    }
                )
    return pd.DataFrame(rows)


def _score_band(values: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=[-np.inf, 0.25, 0.50, 0.75, 0.90, 0.97, np.inf],
        labels=["<=0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-0.97", ">0.97"],
    ).astype(str)


def _disagreement_type(row: pd.Series) -> str:
    method_count = len(METHOD_SPECS)
    accepted_count = int(row["accepted_method_count"])
    fusion_accept = bool(row[f"{FUSION_ALIAS}_accepted"])
    source_accept = bool(row[f"{PRIMARY_BASELINE_ALIAS}_accepted"])
    if accepted_count == method_count:
        return "all_methods_accept"
    if accepted_count == 0:
        return "all_methods_reject"
    if fusion_accept and not source_accept:
        return "fusion_accepts_sourceafis_rejects"
    if source_accept and not fusion_accept:
        return "sourceafis_accepts_fusion_rejects"
    if fusion_accept:
        return "fusion_accepts_partial_component_disagreement"
    return "fusion_rejects_partial_component_disagreement"


def _taxonomy_label(row: pd.Series) -> str:
    label = int(row["label"])
    fusion_accept = bool(row[f"{FUSION_ALIAS}_accepted"])
    source_accept = bool(row[f"{PRIMARY_BASELINE_ALIAS}_accepted"])
    accepted_count = int(row["accepted_method_count"])
    if label == 1 and fusion_accept:
        if not source_accept:
            return "fusion_rescued_positive_from_sourceafis"
        return "fusion_true_accept"
    if label == 1 and not fusion_accept:
        if accepted_count > 0:
            return "fusion_false_reject_rescued_by_component"
        return "common_false_reject_all_methods"
    if label == 0 and fusion_accept:
        if not source_accept:
            return "fusion_new_false_accept_vs_sourceafis"
        return "fusion_and_sourceafis_false_accept"
    if label == 0 and source_accept:
        return "fusion_fixed_sourceafis_false_accept"
    return "fusion_true_reject"


def build_failure_taxonomy_pairs(
    frames: dict[tuple[str, str], pd.DataFrame],
    thresholds: pd.DataFrame,
    *,
    datasets: Iterable[str],
    target_far: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    aliases = [spec.alias for spec in METHOD_SPECS]
    for dataset in datasets:
        test = frames[(str(dataset), "test")]
        out = test[CONTEXT_COLUMNS].copy()
        out["target_far"] = float(target_far)
        for spec in METHOD_SPECS:
            threshold = float(
                thresholds[(thresholds["dataset"] == str(dataset)) & (thresholds["method_alias"] == spec.alias)].iloc[0][
                    "threshold"
                ]
            )
            score = test[spec.score_column].astype(float)
            accepted = score >= threshold
            out[f"{spec.alias}_score"] = score
            out[f"{spec.alias}_threshold"] = threshold
            out[f"{spec.alias}_score_margin"] = score - threshold
            out[f"{spec.alias}_accepted"] = accepted.astype(bool)
            out[f"{spec.alias}_outcome"] = _outcome_series(out["label"], out[f"{spec.alias}_accepted"])
        accept_cols = [f"{alias}_accepted" for alias in aliases]
        outcome_cols = [f"{alias}_outcome" for alias in aliases]
        out["accepted_method_count"] = out[accept_cols].sum(axis=1).astype(int)
        out["rejected_method_count"] = len(aliases) - out["accepted_method_count"]
        out["true_accept_method_count"] = out[outcome_cols].eq("TA").sum(axis=1).astype(int)
        out["false_reject_method_count"] = out[outcome_cols].eq("FR").sum(axis=1).astype(int)
        out["false_accept_method_count"] = out[outcome_cols].eq("FA").sum(axis=1).astype(int)
        out["true_reject_method_count"] = out[outcome_cols].eq("TR").sum(axis=1).astype(int)
        out["accepted_by_methods"] = out.apply(lambda row: _method_lists_from_matrix(row, aliases, accepted=True), axis=1)
        out["rejected_by_methods"] = out.apply(lambda row: _method_lists_from_matrix(row, aliases, accepted=False), axis=1)
        out["method_acceptance_pattern"] = out[accept_cols].apply(
            lambda row: "|".join(alias for alias in aliases if bool(row[f"{alias}_accepted"])),
            axis=1,
        )
        out["method_acceptance_pattern"] = out["method_acceptance_pattern"].mask(
            out["method_acceptance_pattern"].eq(""),
            "__none__",
        )
        out["method_disagreement_type"] = out.apply(_disagreement_type, axis=1)
        out["fusion_score_band"] = _score_band(out[f"{FUSION_ALIAS}_score"])
        out["taxonomy"] = out.apply(_taxonomy_label, axis=1)
        labels = out["label"].astype(int)
        out["fusion_rescued_positive_from_sourceafis"] = (
            labels.eq(1) & ~out[f"{PRIMARY_BASELINE_ALIAS}_accepted"].astype(bool) & out[f"{FUSION_ALIAS}_accepted"].astype(bool)
        )
        out["fusion_lost_positive_vs_sourceafis"] = (
            labels.eq(1) & out[f"{PRIMARY_BASELINE_ALIAS}_accepted"].astype(bool) & ~out[f"{FUSION_ALIAS}_accepted"].astype(bool)
        )
        out["fusion_fixed_sourceafis_false_accept"] = (
            labels.eq(0) & out[f"{PRIMARY_BASELINE_ALIAS}_accepted"].astype(bool) & ~out[f"{FUSION_ALIAS}_accepted"].astype(bool)
        )
        out["fusion_new_false_accept_vs_sourceafis"] = (
            labels.eq(0) & ~out[f"{PRIMARY_BASELINE_ALIAS}_accepted"].astype(bool) & out[f"{FUSION_ALIAS}_accepted"].astype(bool)
        )
        out["common_false_reject_all_methods"] = labels.eq(1) & out["false_reject_method_count"].eq(len(aliases))
        out["fusion_false_reject_rescued_by_component"] = (
            labels.eq(1) & ~out[f"{FUSION_ALIAS}_accepted"].astype(bool) & out["true_accept_method_count"].gt(0)
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True, sort=False)


def summarize_taxonomy(pairs: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in pairs.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        labels = sub["label"].astype(int)
        positives = int(labels.eq(1).sum())
        negatives = int(labels.eq(0).sum())
        row.update(
            {
                "pairs": int(len(sub)),
                "positives": positives,
                "negatives": negatives,
                "fusion_TA": int(sub[f"{FUSION_ALIAS}_outcome"].eq("TA").sum()),
                "fusion_FR": int(sub[f"{FUSION_ALIAS}_outcome"].eq("FR").sum()),
                "fusion_FA": int(sub[f"{FUSION_ALIAS}_outcome"].eq("FA").sum()),
                "fusion_TR": int(sub[f"{FUSION_ALIAS}_outcome"].eq("TR").sum()),
                "fusion_TAR": _safe_rate(int(sub[f"{FUSION_ALIAS}_outcome"].eq("TA").sum()), positives),
                "fusion_FAR": _safe_rate(int(sub[f"{FUSION_ALIAS}_outcome"].eq("FA").sum()), negatives),
            }
        )
        for flag in TAXONOMY_FLAGS:
            row[flag] = int(sub[flag].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _format_value(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if math.isnan(value):
            return "nan"
        if abs(value) <= 1 and value != 0:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value).replace("|", "\\|")


def _markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    display = df.head(max_rows).copy() if max_rows else df.copy()
    cols = list(display.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def write_failure_taxonomy_summary(
    path: Path,
    *,
    by_dataset: pd.DataFrame,
    by_finger: pd.DataFrame,
    by_disagreement: pd.DataFrame,
    thresholds: pd.DataFrame,
    fusion_validation: dict[str, Any],
    target_far: float,
) -> None:
    fusion_thresholds = thresholds[thresholds["method_alias"] == FUSION_ALIAS][
        ["dataset", "target_far", "threshold", "calibration_negative_count", "calibration_false_accepts", "calibration_far"]
    ]
    validation_df = pd.DataFrame.from_dict(fusion_validation, orient="index").reset_index(names="dataset")
    lines = [
        "# Current Fusion v2 failure taxonomy",
        "",
        "These diagnostics are rebuilt from the canonical Fusion v2 statistical score files only.",
        "Thresholds are selected from VAL negatives and then frozen for TEST.",
        "",
        f"Primary operating point: target FAR `{target_far}`.",
        "",
        "## Fusion v2 validation",
        "",
        _markdown_table(validation_df[["dataset", "threshold", "TA", "FR", "FA", "TR", "TAR", "FAR"]]),
        "",
        "## VAL threshold calibration",
        "",
        _markdown_table(fusion_thresholds),
        "",
        "## Taxonomy by dataset",
        "",
        _markdown_table(by_dataset),
        "",
        "## Worst fingers by Fusion v2 false rejects",
        "",
        _markdown_table(by_finger.sort_values(["fusion_FR", "dataset"], ascending=[False, True]).head(20)),
        "",
        "## Method disagreement",
        "",
        _markdown_table(by_disagreement.sort_values(["pairs", "dataset"], ascending=[False, True]).head(30)),
        "",
        "## Baseline comparison",
        "",
        "`rescued`, `lost`, `fixed false accept`, and `new false accept` are defined relative to SourceAFIS-only at its own VAL-calibrated threshold.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_true_accept_failure_summary(
    path: Path,
    *,
    method_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    common_fr: pd.DataFrame,
    method_specific_fr: pd.DataFrame,
    target_far: float,
) -> None:
    display = method_summary[
        ["method_alias", "dataset", "threshold", "positives", "TA", "FR", "TAR", "negatives", "FA", "TR", "FAR"]
    ].copy()
    preferred = pairwise[
        (pairwise["base_method"].isin([PRIMARY_BASELINE_ALIAS, FUSION_ALIAS]))
        & (pairwise["other_method"].isin([PRIMARY_BASELINE_ALIAS, FUSION_ALIAS, "deep_score", "deep_logit", "sift_score"]))
    ].copy()
    lines = [
        "# Current true-accept failures across methods",
        "",
        "All method thresholds are calibrated from each method's own VAL negatives and applied frozen to TEST.",
        f"Primary operating point: target FAR `{target_far}`.",
        "",
        "## Method outcome summary",
        "",
        _markdown_table(display),
        "",
        "## Pairwise complementarity",
        "",
        _markdown_table(preferred.head(40)),
        "",
        "## Compact counts",
        "",
        f"Common false-reject positive pairs across all methods: `{len(common_fr)}`.",
        f"Method-specific false-reject rows: `{len(method_specific_fr)}`.",
        "",
        "Filtered/removal-style recomputations are intentionally not reported here as benchmark results.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _prepare_clean_output_dir(path: Path, repo_root: Path) -> None:
    resolved = path.resolve()
    if not _is_relative_to(resolved, repo_root):
        raise CurrentDiagnosticsError(f"Refusing to write outside repo root: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def write_outputs(
    *,
    repo_root: Path,
    taxonomy_outdir: Path,
    outcomes_outdir: Path,
    pairs: pd.DataFrame,
    all_method_outcomes: pd.DataFrame,
    method_summary: pd.DataFrame,
    positive_matrix: pd.DataFrame,
    pairwise: pd.DataFrame,
    common_fr: pd.DataFrame,
    method_specific_fr: pd.DataFrame,
    thresholds: pd.DataFrame,
    metrics: pd.DataFrame,
    input_files: dict[str, Any],
    test_counts: dict[str, Any],
    fusion_validation: dict[str, Any],
    target_far: float,
) -> dict[str, Path]:
    _prepare_clean_output_dir(taxonomy_outdir, repo_root)
    _prepare_clean_output_dir(outcomes_outdir, repo_root)

    by_dataset = summarize_taxonomy(pairs, ["dataset"])
    by_finger = summarize_taxonomy(pairs, ["dataset", "finger_position", "frgp"])
    by_score_band = summarize_taxonomy(pairs, ["dataset", "fusion_score_band"])
    by_disagreement = summarize_taxonomy(pairs, ["dataset", "method_disagreement_type", "method_acceptance_pattern"])

    taxonomy_outputs: dict[str, Path] = {
        "failure_taxonomy_pairs.csv": taxonomy_outdir / "failure_taxonomy_pairs.csv",
        "failure_taxonomy_by_dataset.csv": taxonomy_outdir / "failure_taxonomy_by_dataset.csv",
        "failure_taxonomy_by_finger.csv": taxonomy_outdir / "failure_taxonomy_by_finger.csv",
        "failure_taxonomy_by_score_band.csv": taxonomy_outdir / "failure_taxonomy_by_score_band.csv",
        "failure_taxonomy_by_method_disagreement.csv": taxonomy_outdir / "failure_taxonomy_by_method_disagreement.csv",
        "rescued_positive_examples.csv": taxonomy_outdir / "rescued_positive_examples.csv",
        "lost_positive_examples.csv": taxonomy_outdir / "lost_positive_examples.csv",
        "fixed_false_accept_examples.csv": taxonomy_outdir / "fixed_false_accept_examples.csv",
        "new_false_accept_examples.csv": taxonomy_outdir / "new_false_accept_examples.csv",
        "failure_taxonomy_summary.md": taxonomy_outdir / "failure_taxonomy_summary.md",
        "current_diagnostics_manifest.json": taxonomy_outdir / "current_diagnostics_manifest.json",
    }

    pairs.to_csv(taxonomy_outputs["failure_taxonomy_pairs.csv"], index=False)
    by_dataset.to_csv(taxonomy_outputs["failure_taxonomy_by_dataset.csv"], index=False)
    by_finger.to_csv(taxonomy_outputs["failure_taxonomy_by_finger.csv"], index=False)
    by_score_band.to_csv(taxonomy_outputs["failure_taxonomy_by_score_band.csv"], index=False)
    by_disagreement.to_csv(taxonomy_outputs["failure_taxonomy_by_method_disagreement.csv"], index=False)
    pairs[pairs["fusion_rescued_positive_from_sourceafis"]].sort_values(
        [f"{FUSION_ALIAS}_score_margin", f"{PRIMARY_BASELINE_ALIAS}_score_margin"],
        ascending=[False, True],
    ).head(100).to_csv(taxonomy_outputs["rescued_positive_examples.csv"], index=False)
    pairs[pairs["fusion_lost_positive_vs_sourceafis"]].sort_values(
        [f"{PRIMARY_BASELINE_ALIAS}_score_margin", f"{FUSION_ALIAS}_score_margin"],
        ascending=[False, True],
    ).head(100).to_csv(taxonomy_outputs["lost_positive_examples.csv"], index=False)
    pairs[pairs["fusion_fixed_sourceafis_false_accept"]].sort_values(
        [f"{PRIMARY_BASELINE_ALIAS}_score_margin", f"{FUSION_ALIAS}_score_margin"],
        ascending=[False, True],
    ).head(100).to_csv(taxonomy_outputs["fixed_false_accept_examples.csv"], index=False)
    pairs[pairs["fusion_new_false_accept_vs_sourceafis"]].sort_values(
        [f"{FUSION_ALIAS}_score_margin", f"{PRIMARY_BASELINE_ALIAS}_score_margin"],
        ascending=[False, True],
    ).head(100).to_csv(taxonomy_outputs["new_false_accept_examples.csv"], index=False)
    write_failure_taxonomy_summary(
        taxonomy_outputs["failure_taxonomy_summary.md"],
        by_dataset=by_dataset,
        by_finger=by_finger,
        by_disagreement=by_disagreement,
        thresholds=thresholds,
        fusion_validation=fusion_validation,
        target_far=float(target_far),
    )

    common_manifest = {
        "created_at": _utc_now(),
        "canonical_benchmark_artifact": str(repo_root / CANONICAL_BENCHMARK_REL),
        "canonical_method": CANONICAL_METHOD,
        "target_far": float(target_far),
        "threshold_protocol": "Each method threshold is computed from its own VAL negatives only.",
        "test_threshold_application": "TEST thresholds are frozen from VAL.",
        "input_score_files": input_files,
        "methods": [spec.__dict__ for spec in METHOD_SPECS],
        "test_count_validation": test_counts,
        "fusion_test_metric_validation": fusion_validation,
        "metric_tolerance": FUSION_METRIC_TOLERANCE,
        "thresholds": thresholds.to_dict("records"),
        "metrics": metrics.to_dict("records"),
        "stale_legacy_quarantine": str(repo_root / "artifacts/reports/diagnostics/legacy_stale_20260629"),
    }
    taxonomy_manifest = {
        "schema_version": "current_fusion_v2_failure_taxonomy_v1",
        "output_dir": str(taxonomy_outdir),
        "baseline_for_rescued_lost_fixed_new": PRIMARY_BASELINE_ALIAS,
        "generated_files": sorted(taxonomy_outputs),
        **common_manifest,
    }
    taxonomy_outputs["current_diagnostics_manifest.json"].write_text(
        json.dumps(taxonomy_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    outcome_outputs: dict[str, Path] = {
        "all_method_outcomes.csv": outcomes_outdir / "all_method_outcomes.csv",
        "positive_pair_outcome_matrix.csv": outcomes_outdir / "positive_pair_outcome_matrix.csv",
        "method_outcome_summary.csv": outcomes_outdir / "method_outcome_summary.csv",
        "pairwise_complementarity_summary.csv": outcomes_outdir / "pairwise_complementarity_summary.csv",
        "common_false_rejects_all_methods.csv": outcomes_outdir / "common_false_rejects_all_methods.csv",
        "method_specific_false_rejects.csv": outcomes_outdir / "method_specific_false_rejects.csv",
        "true_accept_failure_summary.md": outcomes_outdir / "true_accept_failure_summary.md",
        "current_diagnostics_manifest.json": outcomes_outdir / "current_diagnostics_manifest.json",
    }
    all_method_outcomes.to_csv(outcome_outputs["all_method_outcomes.csv"], index=False)
    positive_matrix.to_csv(outcome_outputs["positive_pair_outcome_matrix.csv"], index=False)
    method_summary.to_csv(outcome_outputs["method_outcome_summary.csv"], index=False)
    pairwise.to_csv(outcome_outputs["pairwise_complementarity_summary.csv"], index=False)
    common_fr.to_csv(outcome_outputs["common_false_rejects_all_methods.csv"], index=False)
    method_specific_fr.to_csv(outcome_outputs["method_specific_false_rejects.csv"], index=False)
    write_true_accept_failure_summary(
        outcome_outputs["true_accept_failure_summary.md"],
        method_summary=method_summary,
        pairwise=pairwise,
        common_fr=common_fr,
        method_specific_fr=method_specific_fr,
        target_far=float(target_far),
    )
    outcomes_manifest = {
        "schema_version": "current_true_accept_failures_across_methods_v1",
        "output_dir": str(outcomes_outdir),
        "generated_files": sorted(outcome_outputs),
        **common_manifest,
    }
    outcome_outputs["current_diagnostics_manifest.json"].write_text(
        json.dumps(outcomes_manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return {**taxonomy_outputs, **outcome_outputs}


def run_current_diagnostics(
    *,
    repo_root: Path,
    taxonomy_outdir: Path,
    outcomes_outdir: Path,
    datasets: Iterable[str] = DATASETS,
    target_far: float = TARGET_FAR,
) -> dict[str, Path]:
    repo_root = repo_root.resolve()
    datasets = tuple(str(dataset) for dataset in datasets)
    frames, input_files = load_score_frames(repo_root, datasets)
    test_counts = validate_test_counts(frames, datasets)
    thresholds, metrics = build_thresholds_and_metrics(
        frames,
        repo_root=repo_root,
        datasets=datasets,
        target_far=float(target_far),
    )
    fusion_validation = validate_fusion_metrics(metrics, target_far=float(target_far))
    all_method_outcomes = build_all_method_outcomes(frames, thresholds, datasets=datasets)
    method_summary = summarize_method_outcomes(all_method_outcomes)
    positive_matrix = build_positive_pair_matrix(all_method_outcomes)
    common_fr = build_common_false_rejects(positive_matrix)
    method_specific_fr = build_method_specific_false_rejects(positive_matrix)
    pairwise = build_pairwise_complementarity(positive_matrix)
    pairs = build_failure_taxonomy_pairs(frames, thresholds, datasets=datasets, target_far=float(target_far))

    return write_outputs(
        repo_root=repo_root,
        taxonomy_outdir=taxonomy_outdir,
        outcomes_outdir=outcomes_outdir,
        pairs=pairs,
        all_method_outcomes=all_method_outcomes,
        method_summary=method_summary,
        positive_matrix=positive_matrix,
        pairwise=pairwise,
        common_fr=common_fr,
        method_specific_fr=method_specific_fr,
        thresholds=thresholds,
        metrics=metrics,
        input_files=input_files,
        test_counts=test_counts,
        fusion_validation=fusion_validation,
        target_far=float(target_far),
    )


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild current Fusion v2 diagnostics from canonical statistical score files."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--taxonomy-outdir", default=str(DEFAULT_TAXONOMY_OUTDIR_REL))
    parser.add_argument("--outcomes-outdir", default=str(DEFAULT_OUTCOMES_OUTDIR_REL))
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--target-far", type=float, default=TARGET_FAR)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    taxonomy_outdir = _repo_path(repo_root, args.taxonomy_outdir)
    outcomes_outdir = _repo_path(repo_root, args.outcomes_outdir)
    outputs = run_current_diagnostics(
        repo_root=repo_root,
        taxonomy_outdir=taxonomy_outdir,
        outcomes_outdir=outcomes_outdir,
        datasets=parse_csv_list(args.datasets),
        target_far=float(args.target_far),
    )
    print("[done] current diagnostics rebuilt")
    print("taxonomy:", taxonomy_outdir)
    print("all-method outcomes:", outcomes_outdir)
    for name, path in sorted(outputs.items()):
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
