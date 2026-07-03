from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# The local environment has numexpr/bottleneck wheels built against NumPy 1.x.
# Pandas treats them as optional, so block them before import to avoid noisy
# extension tracebacks while keeping the analysis pure-Python/NumPy/Pandas.
sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)

import numpy as np
import pandas as pd


DATASETS = ("nist_sd300b", "nist_sd300c")
SPLITS = ("val", "test")
TARGET_TARS = (0.90, 0.95, 0.99, 0.995, 1.00)
PRIMARY_HIGH_RECALL_TARGET_TARS = (0.99, 0.995, 1.00)
PRIMARY_HIGH_RECALL_TARGET_TAR_KEYS = {round(target, 12) for target in PRIMARY_HIGH_RECALL_TARGET_TARS}
TARGET_FARS = (0.005, 0.01)
OUTPUT_DIR = Path("artifacts/reports/benchmark/pre_final_menachem_experiments_sd300_anatomical_v2")
SPLIT_ORACLE_MODE = "split_oracle_descriptive"
VAL_CALIBRATED_MODE = "val_calibrated_apply_to_test"
DESCRIPTIVE_ORACLE_ROLE = "descriptive_oracle_only"
PUBLISHABLE_PROTOCOL_ROLE = "publishable_protocol"

AUDIT_ALLOWLIST = Path("artifacts/reports/benchmark/final_sd300_anatomical_v2_artifact_audit/final_artifact_allowlist.csv")
AUDIT_LEGACY_CANDIDATES = Path(
    "artifacts/reports/benchmark/final_sd300_anatomical_v2_artifact_audit/legacy_artifact_candidates.csv"
)
COMPAT_DDP_DIR = Path("artifacts/reports/benchmark/deep_pair_reranker_fast_ddp_full_pairs")
FORBIDDEN_COMPAT_TRAIN_SCORES = Path("artifacts/reports/benchmark/deep_pair_reranker_fast_ddp_train_scores")


@dataclass(frozen=True)
class MethodSpec:
    display_name: str
    method_key: str
    method_id: str
    root: Path
    score_template: str
    threshold_csv: Path
    is_deep_fusion: bool = False

    def score_path(self, dataset: str, split: str) -> Path:
        return self.root / self.score_template.format(dataset=dataset, method=self.method_id, split=split)


METHODS = (
    MethodSpec(
        display_name="SourceAFIS v2",
        method_key="sourceafis_v2",
        method_id="sourceafis_open",
        root=Path("artifacts/reports/benchmark/plain_roll_final_sourceafis_v2_anatomical_full_pairs"),
        score_template="scores/scores_{dataset}_{method}_{split}.csv",
        threshold_csv=Path(
            "artifacts/reports/benchmark/plain_roll_final_sourceafis_v2_anatomical_full_pairs/plain_roll_final_thresholds.csv"
        ),
    ),
    MethodSpec(
        display_name="Fusion v1 v2",
        method_key="fusion_v1_v2",
        method_id="sourceafis_sift_quality_fusion_v1",
        root=Path("artifacts/reports/benchmark/plain_roll_final_fusion_v1_v2_anatomical_full_pairs"),
        score_template="scores/scores_{dataset}_{method}_{split}.csv",
        threshold_csv=Path(
            "artifacts/reports/benchmark/plain_roll_final_fusion_v1_v2_anatomical_full_pairs/plain_roll_final_thresholds.csv"
        ),
    ),
    MethodSpec(
        display_name="Deep Fusion v2 Statistical",
        method_key="deep_fusion_v2_statistical",
        method_id="sourceafis_sift_quality_deep_fusion_v2",
        root=Path("artifacts/reports/benchmark/sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep"),
        score_template="scores/scores_{dataset}_{method}_{split}.csv",
        threshold_csv=Path(
            "artifacts/reports/benchmark/sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep/plain_roll_final_thresholds.csv"
        ),
        is_deep_fusion=True,
    ),
    MethodSpec(
        display_name="Deep Fusion v2 Manual Group Weighted 45/15/30/10",
        method_key="deep_fusion_v2_manual_45_15_30_10",
        method_id="sourceafis_sift_quality_deep_group_weighted_fusion_v2",
        root=Path(
            "artifacts/reports/benchmark/sourceafis_sift_quality_deep_group_weighted_fusion_v2_manual_45_15_30_10_anatomical_v2_ddpdeep"
        ),
        score_template="scores/ablation/scores_{dataset}_{method}_{split}.csv",
        threshold_csv=Path(
            "artifacts/reports/benchmark/sourceafis_sift_quality_deep_group_weighted_fusion_v2_manual_45_15_30_10_anatomical_v2_ddpdeep/plain_roll_final_thresholds.csv"
        ),
        is_deep_fusion=True,
    ),
    MethodSpec(
        display_name="Deep Fusion v2 Auto Group Weighted",
        method_key="deep_fusion_v2_auto_val_tar_at_far_001",
        method_id="sourceafis_sift_quality_deep_group_weighted_fusion_v2",
        root=Path(
            "artifacts/reports/benchmark/sourceafis_sift_quality_deep_group_weighted_fusion_v2_auto_val_tar_at_far_001_anatomical_v2_ddpdeep"
        ),
        score_template="scores/ablation/scores_{dataset}_{method}_{split}.csv",
        threshold_csv=Path(
            "artifacts/reports/benchmark/sourceafis_sift_quality_deep_group_weighted_fusion_v2_auto_val_tar_at_far_001_anatomical_v2_ddpdeep/plain_roll_final_thresholds.csv"
        ),
        is_deep_fusion=True,
    ),
)

DEEP_FUSION_FAILURE_KEYS = {
    "Deep Fusion v2 Statistical": "statistical",
    "Deep Fusion v2 Manual Group Weighted 45/15/30/10": "manual",
    "Deep Fusion v2 Auto Group Weighted": "auto",
}

FRGP_NAMES = {
    1: "right_thumb",
    2: "right_index",
    3: "right_middle",
    4: "right_ring",
    5: "right_little",
    6: "left_thumb",
    7: "left_index",
    8: "left_middle",
    9: "left_ring",
    10: "left_little",
}
FINGER_TYPES = {
    1: "thumb",
    2: "index",
    3: "middle",
    4: "ring",
    5: "little",
    6: "thumb",
    7: "index",
    8: "middle",
    9: "ring",
    10: "little",
}
SIDES = {
    1: "right",
    2: "right",
    3: "right",
    4: "right",
    5: "right",
    6: "left",
    7: "left",
    8: "left",
    9: "left",
    10: "left",
}


class InputTracker:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self._paths: set[Path] = set()

    def add(self, path: Path) -> Path:
        resolved = path if path.is_absolute() else self.repo_root / path
        resolved = resolved.resolve()
        self._paths.add(resolved)
        return resolved

    def read_csv(self, path: Path, label: str) -> pd.DataFrame:
        resolved = self.add(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing {label}: {resolved}")
        frame = pd.read_csv(resolved)
        if frame.empty:
            raise ValueError(f"{label} is empty: {resolved}")
        return frame

    def files(self) -> list[Path]:
        return sorted(self._paths, key=lambda item: relpath(item, self.repo_root))

    def sha256(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in self.files():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result[relpath(path, self.repo_root)] = digest
        return result


def relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--outdir", default=str(OUTPUT_DIR))
    return parser.parse_args(argv)


def _require_not_forbidden(path: Path) -> None:
    normalized = path.as_posix().replace("\\", "/")
    forbidden = FORBIDDEN_COMPAT_TRAIN_SCORES.as_posix()
    if normalized.startswith(forbidden):
        raise ValueError(f"Forbidden legacy/known-do-not-use input selected: {path}")


def load_score(repo_root: Path, tracker: InputTracker, spec: MethodSpec, dataset: str, split: str) -> pd.DataFrame:
    rel = spec.score_path(dataset, split)
    _require_not_forbidden(rel)
    df = tracker.read_csv(rel, f"{spec.display_name} {dataset} {split} scores").copy()
    required = {"dataset", "split", "pair_id", "label", "subject_a", "subject_b", "path_a", "path_b", "score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{rel} missing required columns: {missing}")
    if "frgp" not in df.columns:
        if "finger_position" not in df.columns:
            raise ValueError(f"{rel} missing frgp/finger_position column")
        df["frgp"] = df["finger_position"]
    if "finger_position" not in df.columns:
        df["finger_position"] = df["frgp"]
    if "higher_is_more_similar" in df.columns:
        higher = df["higher_is_more_similar"].astype(str).str.lower().str.strip()
        if not higher.isin({"true", "1", "yes"}).all():
            raise ValueError(f"{rel} contains scores that are not marked higher-is-more-similar")
    out = df.copy()
    out["method"] = spec.display_name
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    out["score"] = pd.to_numeric(out["score"], errors="raise")
    out["frgp"] = pd.to_numeric(out["frgp"], errors="raise").astype(int)
    if not set(out["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{rel} contains non-binary labels")
    if out["score"].isna().any():
        raise ValueError(f"{rel} contains NaN scores")
    return out


def load_thresholds(tracker: InputTracker, spec: MethodSpec) -> pd.DataFrame:
    _require_not_forbidden(spec.threshold_csv)
    df = tracker.read_csv(spec.threshold_csv, f"{spec.display_name} thresholds").copy()
    required = {"dataset", "target_far", "threshold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{spec.threshold_csv} missing required columns: {missing}")
    df["dataset"] = df["dataset"].astype(str).str.strip()
    df["target_far"] = pd.to_numeric(df["target_far"], errors="raise")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    return df


def threshold_for_far(thresholds: pd.DataFrame, dataset: str, target_far: float) -> float:
    sub = thresholds[
        (thresholds["dataset"].astype(str) == dataset)
        & np.isclose(pd.to_numeric(thresholds["target_far"], errors="coerce"), float(target_far), rtol=0.0, atol=1e-12)
    ]
    if sub.empty:
        raise ValueError(f"No threshold for dataset={dataset}, target_far={target_far}")
    return float(sub.iloc[0]["threshold"])


def positive_threshold_for_tar(scores: pd.DataFrame, target_tar: float) -> tuple[float, int]:
    positives = scores.loc[scores["label"] == 1, "score"].sort_values(ascending=False).to_numpy()
    if len(positives) == 0:
        raise ValueError("Cannot select TAR threshold without positive pairs")
    required_accepts = int(math.ceil(float(target_tar) * len(positives) - 1e-12))
    required_accepts = min(max(required_accepts, 1), len(positives))
    return float(positives[required_accepts - 1]), required_accepts


def metrics_at_threshold(scores: pd.DataFrame, threshold: float) -> dict[str, Any]:
    labels = scores["label"].astype(int)
    accepted = pd.to_numeric(scores["score"], errors="raise") >= float(threshold)
    positive = labels == 1
    negative = labels == 0
    ta = int((accepted & positive).sum())
    fr = int((~accepted & positive).sum())
    fa = int((accepted & negative).sum())
    tr = int((~accepted & negative).sum())
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    tar = ta / n_positive if n_positive else float("nan")
    frr = fr / n_positive if n_positive else float("nan")
    far = fa / n_negative if n_negative else float("nan")
    return {
        "achieved_TAR": tar,
        "achieved_FAR": far,
        "FRR": frr,
        "TA": ta,
        "FR": fr,
        "FA": fa,
        "TR": tr,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def build_threshold_to_target_tar(score_cache: dict[tuple[str, str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in METHODS:
        for dataset in DATASETS:
            for split in SPLITS:
                split_scores = score_cache[(spec.display_name, dataset, split)]
                for target_tar in TARGET_TARS:
                    threshold, required_accepts = positive_threshold_for_tar(split_scores, target_tar)
                    metrics = metrics_at_threshold(split_scores, threshold)
                    rows.append(
                        {
                            "method": spec.display_name,
                            "method_key": spec.method_key,
                            "dataset": dataset,
                            "split": split,
                            "target_tar": float(target_tar),
                            "threshold": threshold,
                            **metrics,
                            "analysis_mode": SPLIT_ORACLE_MODE,
                            "protocol_role": DESCRIPTIVE_ORACLE_ROLE,
                            "is_oracle": True,
                            "val_threshold": "",
                            "val_TAR": "",
                            "val_FAR": "",
                            "test_TAR_at_val_threshold": "",
                            "test_FAR_at_val_threshold": "",
                            "notes": (
                                "Descriptive split-local threshold; uses the same split to select the maximum threshold. "
                                f"Required positive accepts={required_accepts}."
                            ),
                        }
                    )

            for target_tar in TARGET_TARS:
                val_scores = score_cache[(spec.display_name, dataset, "val")]
                test_scores = score_cache[(spec.display_name, dataset, "test")]
                val_threshold, required_accepts = positive_threshold_for_tar(val_scores, target_tar)
                val_metrics = metrics_at_threshold(val_scores, val_threshold)
                test_metrics = metrics_at_threshold(test_scores, val_threshold)
                shared = {
                    "method": spec.display_name,
                    "method_key": spec.method_key,
                    "dataset": dataset,
                    "target_tar": float(target_tar),
                    "threshold": val_threshold,
                    "analysis_mode": VAL_CALIBRATED_MODE,
                    "protocol_role": PUBLISHABLE_PROTOCOL_ROLE,
                    "is_oracle": False,
                    "val_threshold": val_threshold,
                    "val_TAR": val_metrics["achieved_TAR"],
                    "val_FAR": val_metrics["achieved_FAR"],
                    "test_TAR_at_val_threshold": test_metrics["achieved_TAR"],
                    "test_FAR_at_val_threshold": test_metrics["achieved_FAR"],
                }
                rows.append(
                    {
                        **shared,
                        "split": "val",
                        **val_metrics,
                        "notes": (
                            "VAL calibration row; threshold chosen on VAL only as maximum threshold satisfying target TAR. "
                            f"Required positive accepts={required_accepts}."
                        ),
                    }
                )
                rows.append(
                    {
                        **shared,
                        "split": "test",
                        **test_metrics,
                        "notes": "TEST application row; threshold was selected on VAL only.",
                    }
                )
    columns = [
        "method",
        "method_key",
        "dataset",
        "split",
        "target_tar",
        "threshold",
        "achieved_TAR",
        "achieved_FAR",
        "FRR",
        "TA",
        "FR",
        "FA",
        "TR",
        "n_positive",
        "n_negative",
        "analysis_mode",
        "protocol_role",
        "is_oracle",
        "val_threshold",
        "val_TAR",
        "val_FAR",
        "test_TAR_at_val_threshold",
        "test_FAR_at_val_threshold",
        "notes",
    ]
    return pd.DataFrame(rows, columns=columns)


def finger_meta(frgp: int) -> dict[str, str]:
    return {
        "finger_name": FRGP_NAMES.get(int(frgp), f"unknown_{frgp}"),
        "finger_type": FINGER_TYPES.get(int(frgp), "unknown"),
        "side": SIDES.get(int(frgp), "unknown"),
    }


def percentile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(q)) if len(values) else float("nan")


def positive_group_summary(
    *,
    scores: pd.DataFrame,
    method: str,
    dataset: str,
    split: str,
    target_far: float,
    threshold: float,
    aggregation_level: str,
    group_cols: list[str],
) -> list[dict[str, Any]]:
    positives = scores[scores["label"] == 1].copy()
    positives["accepted"] = pd.to_numeric(positives["score"], errors="raise") >= float(threshold)
    rows: list[dict[str, Any]] = []
    for keys, group in positives.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = {col: value for col, value in zip(group_cols, keys)}
        n_positive = int(len(group))
        ta = int(group["accepted"].sum())
        fr = n_positive - ta
        row = {
            "aggregation_level": aggregation_level,
            "method": method,
            "dataset": dataset,
            "split": split,
            "target_far": float(target_far),
            "threshold": float(threshold),
            "frgp": "",
            "finger_name": "",
            "finger_type": "",
            "side": "",
            "n_positive_pairs": n_positive,
            "TA": ta,
            "FR": fr,
            "TAR": ta / n_positive if n_positive else float("nan"),
            "FRR": fr / n_positive if n_positive else float("nan"),
            "mean_positive_score": float(pd.to_numeric(group["score"], errors="coerce").mean()),
            "median_positive_score": float(pd.to_numeric(group["score"], errors="coerce").median()),
            "std_positive_score": float(pd.to_numeric(group["score"], errors="coerce").std()),
            "p10_positive_score": percentile(group["score"], 0.10),
            "p25_positive_score": percentile(group["score"], 0.25),
            "p75_positive_score": percentile(group["score"], 0.75),
            "p90_positive_score": percentile(group["score"], 0.90),
        }
        if aggregation_level == "frgp":
            frgp = int(values["frgp"])
            row["frgp"] = frgp
            row.update(finger_meta(frgp))
        elif aggregation_level == "finger_type":
            row["finger_type"] = str(values["finger_type"])
            row["finger_name"] = "all"
            row["side"] = "both"
        rows.append(row)
    return rows


def build_finger_type_quality_summary(
    score_cache: dict[tuple[str, str, str], pd.DataFrame],
    threshold_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in METHODS:
        thresholds = threshold_cache[spec.display_name]
        for dataset in DATASETS:
            for split in SPLITS:
                scores = score_cache[(spec.display_name, dataset, split)].copy()
                scores["finger_type"] = scores["frgp"].map(lambda value: FINGER_TYPES.get(int(value), "unknown"))
                for target_far in TARGET_FARS:
                    threshold = threshold_for_far(thresholds, dataset, target_far)
                    rows.extend(
                        positive_group_summary(
                            scores=scores,
                            method=spec.display_name,
                            dataset=dataset,
                            split=split,
                            target_far=target_far,
                            threshold=threshold,
                            aggregation_level="frgp",
                            group_cols=["frgp"],
                        )
                    )
                    rows.extend(
                        positive_group_summary(
                            scores=scores,
                            method=spec.display_name,
                            dataset=dataset,
                            split=split,
                            target_far=target_far,
                            threshold=threshold,
                            aggregation_level="finger_type",
                            group_cols=["finger_type"],
                        )
                    )
    columns = [
        "aggregation_level",
        "method",
        "dataset",
        "split",
        "target_far",
        "threshold",
        "frgp",
        "finger_name",
        "finger_type",
        "side",
        "n_positive_pairs",
        "TA",
        "FR",
        "TAR",
        "FRR",
        "mean_positive_score",
        "median_positive_score",
        "std_positive_score",
        "p10_positive_score",
        "p25_positive_score",
        "p75_positive_score",
        "p90_positive_score",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_finger_type_failure_details(
    score_cache: dict[tuple[str, str, str], pd.DataFrame],
    threshold_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    deep_specs = [spec for spec in METHODS if spec.is_deep_fusion]
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for split in SPLITS:
            base: pd.DataFrame | None = None
            for spec in deep_specs:
                key = DEEP_FUSION_FAILURE_KEYS[spec.display_name]
                scores = score_cache[(spec.display_name, dataset, split)].copy()
                columns = ["dataset", "split", "pair_id", "label", "subject_a", "subject_b", "frgp", "path_a", "path_b", "score"]
                sub = scores[columns].rename(columns={"score": f"score_{key}"})
                merge_key = ["dataset", "split", "pair_id"]
                if base is None:
                    base = sub
                else:
                    base = base.merge(
                        sub[merge_key + [f"score_{key}"]],
                        on=merge_key,
                        how="inner",
                        validate="one_to_one",
                    )
            if base is None:
                continue
            positives = base[base["label"].astype(int) == 1].copy()
            for target_far in TARGET_FARS:
                thresholds = {
                    DEEP_FUSION_FAILURE_KEYS[spec.display_name]: threshold_for_far(
                        threshold_cache[spec.display_name], dataset, target_far
                    )
                    for spec in deep_specs
                }
                working = positives.copy()
                for key, threshold in thresholds.items():
                    working[f"threshold_{key}"] = float(threshold)
                    working[f"failed_{key}"] = pd.to_numeric(working[f"score_{key}"], errors="raise") < float(threshold)
                failed = working[
                    working[[f"failed_{key}" for key in ("statistical", "manual", "auto")]].any(axis=1)
                ].copy()
                for _, row in failed.sort_values(["pair_id"]).iterrows():
                    frgp = int(row["frgp"])
                    meta = finger_meta(frgp)
                    rows.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "target_far": float(target_far),
                            "pair_id": row["pair_id"],
                            "subject_id": row["subject_a"],
                            "frgp": frgp,
                            "finger_name": meta["finger_name"],
                            "finger_type": meta["finger_type"],
                            "side": meta["side"],
                            "path_plain": row["path_a"],
                            "path_roll": row["path_b"],
                            "failed_statistical": bool(row["failed_statistical"]),
                            "failed_manual": bool(row["failed_manual"]),
                            "failed_auto": bool(row["failed_auto"]),
                            "score_statistical": float(row["score_statistical"]),
                            "score_manual": float(row["score_manual"]),
                            "score_auto": float(row["score_auto"]),
                            "threshold_statistical": float(row["threshold_statistical"]),
                            "threshold_manual": float(row["threshold_manual"]),
                            "threshold_auto": float(row["threshold_auto"]),
                        }
                    )
    columns = [
        "dataset",
        "split",
        "target_far",
        "pair_id",
        "subject_id",
        "frgp",
        "finger_name",
        "finger_type",
        "side",
        "path_plain",
        "path_roll",
        "failed_statistical",
        "failed_manual",
        "failed_auto",
        "score_statistical",
        "score_manual",
        "score_auto",
        "threshold_statistical",
        "threshold_manual",
        "threshold_auto",
    ]
    return pd.DataFrame(rows, columns=columns)


def _tar_far_group_metrics(
    *,
    scores: pd.DataFrame,
    threshold: float,
    group_values: dict[str, Any],
    aggregation_level: str,
) -> dict[str, Any]:
    labels = scores["label"].astype(int)
    accepted = pd.to_numeric(scores["score"], errors="raise") >= float(threshold)
    positive = labels == 1
    negative = labels == 0
    ta = int((accepted & positive).sum())
    fr = int((~accepted & positive).sum())
    fa = int((accepted & negative).sum())
    tr = int((~accepted & negative).sum())
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    row: dict[str, Any] = {
        "aggregation_level": aggregation_level,
        "frgp": "",
        "finger_name": "",
        "finger_type": "",
        "side": "",
        "n_negative_pairs": n_negative,
        "FA": fa,
        "TR": tr,
        "FAR": fa / n_negative if n_negative else float("nan"),
        "n_positive_pairs": n_positive,
        "TA": ta,
        "FR": fr,
        "TAR": ta / n_positive if n_positive else float("nan"),
        "FRR": fr / n_positive if n_positive else float("nan"),
    }
    if aggregation_level == "frgp":
        frgp = int(group_values["frgp"])
        row["frgp"] = frgp
        row.update(finger_meta(frgp))
    elif aggregation_level == "finger_type":
        row["finger_type"] = str(group_values["finger_type"])
        row["finger_name"] = "all"
        row["side"] = "both"
    return row


def _validate_finger_type_far_reconstructs_overall(
    *,
    threshold_row: pd.Series,
    finger_type_rows: list[dict[str, Any]],
) -> None:
    total_negative = int(sum(int(row["n_negative_pairs"]) for row in finger_type_rows))
    total_fa = int(sum(int(row["FA"]) for row in finger_type_rows))
    if total_negative != int(threshold_row["n_negative"]):
        raise ValueError(
            "Finger-type negative count mismatch for "
            f"{threshold_row['method']} {threshold_row['dataset']} {threshold_row['split']} "
            f"{threshold_row['analysis_mode']} target_tar={threshold_row['target_tar']}: "
            f"grouped={total_negative} overall={int(threshold_row['n_negative'])}"
        )
    reconstructed_far = total_fa / total_negative if total_negative else float("nan")
    expected_far = float(threshold_row["achieved_FAR"])
    if not math.isclose(reconstructed_far, expected_far, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Finger-type FAR reconstruction mismatch for "
            f"{threshold_row['method']} {threshold_row['dataset']} {threshold_row['split']} "
            f"{threshold_row['analysis_mode']} target_tar={threshold_row['target_tar']}: "
            f"grouped_fa={total_fa}, grouped_negatives={total_negative}, "
            f"reconstructed_far={reconstructed_far}, threshold_table_far={expected_far}"
        )


def build_finger_type_far_at_target_tar(
    threshold_to_target: pd.DataFrame,
    score_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    required = {
        "method",
        "method_key",
        "dataset",
        "split",
        "target_tar",
        "threshold",
        "achieved_FAR",
        "n_negative",
        "analysis_mode",
    }
    missing = sorted(required - set(threshold_to_target.columns))
    if missing:
        raise ValueError(f"threshold_to_target_tar.csv missing required columns: {missing}")

    source = threshold_to_target[
        threshold_to_target["target_tar"].map(lambda value: round(float(value), 12) in PRIMARY_HIGH_RECALL_TARGET_TAR_KEYS)
    ].copy()
    source["target_tar"] = pd.to_numeric(source["target_tar"], errors="raise")
    source["threshold"] = pd.to_numeric(source["threshold"], errors="raise")
    source["achieved_FAR"] = pd.to_numeric(source["achieved_FAR"], errors="raise")
    source["n_negative"] = pd.to_numeric(source["n_negative"], errors="raise").astype(int)

    rows: list[dict[str, Any]] = []
    for _, threshold_row in source.sort_values(
        ["method", "dataset", "analysis_mode", "target_tar", "split"], kind="stable"
    ).iterrows():
        method = str(threshold_row["method"])
        method_key = str(threshold_row["method_key"])
        dataset = str(threshold_row["dataset"])
        split = str(threshold_row["split"])
        analysis_mode = str(threshold_row["analysis_mode"])
        target_tar = float(threshold_row["target_tar"])
        threshold = float(threshold_row["threshold"])
        scores = score_cache[(method, dataset, split)].copy()
        scores["finger_type"] = scores["frgp"].map(lambda value: FINGER_TYPES.get(int(value), "unknown"))
        common = {
            "method": method,
            "method_key": method_key,
            "dataset": dataset,
            "split": split,
            "analysis_mode": analysis_mode,
            "target_tar": target_tar,
            "threshold": threshold,
        }
        for frgp, group in scores.groupby("frgp", sort=True, dropna=False):
            rows.append(
                {
                    **common,
                    **_tar_far_group_metrics(
                        scores=group,
                        threshold=threshold,
                        group_values={"frgp": int(frgp)},
                        aggregation_level="frgp",
                    ),
                }
            )

        current_finger_rows: list[dict[str, Any]] = []
        for finger_type, group in scores.groupby("finger_type", sort=True, dropna=False):
            finger_row = {
                **common,
                **_tar_far_group_metrics(
                    scores=group,
                    threshold=threshold,
                    group_values={"finger_type": str(finger_type)},
                    aggregation_level="finger_type",
                ),
            }
            current_finger_rows.append(finger_row)
            rows.append(finger_row)
        _validate_finger_type_far_reconstructs_overall(
            threshold_row=threshold_row,
            finger_type_rows=current_finger_rows,
        )

    columns = [
        "aggregation_level",
        "method",
        "method_key",
        "dataset",
        "split",
        "analysis_mode",
        "target_tar",
        "threshold",
        "frgp",
        "finger_name",
        "finger_type",
        "side",
        "n_negative_pairs",
        "FA",
        "TR",
        "FAR",
        "n_positive_pairs",
        "TA",
        "FR",
        "TAR",
        "FRR",
    ]
    return pd.DataFrame(rows, columns=columns)


def _outcomes_for_scores(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = scores.copy()
    out["accepted"] = pd.to_numeric(out["score"], errors="raise") >= float(threshold)
    positive = out["label"].astype(int) == 1
    out["outcome"] = np.select(
        [
            positive & out["accepted"],
            positive & ~out["accepted"],
            ~positive & out["accepted"],
            ~positive & ~out["accepted"],
        ],
        ["TA", "FR", "FA", "TR"],
        default="",
    )
    return out


def _threshold_lookup(threshold_to_target: pd.DataFrame) -> dict[tuple[str, str, float], float]:
    required = {"method_key", "dataset", "split", "analysis_mode", "target_tar", "threshold"}
    missing = sorted(required - set(threshold_to_target.columns))
    if missing:
        raise ValueError(f"threshold_to_target_tar.csv missing required columns: {missing}")

    source = threshold_to_target[
        (threshold_to_target["analysis_mode"] == VAL_CALIBRATED_MODE)
        & (threshold_to_target["split"] == "test")
        & threshold_to_target["target_tar"].map(lambda value: round(float(value), 12) in PRIMARY_HIGH_RECALL_TARGET_TAR_KEYS)
    ].copy()
    source["target_tar"] = pd.to_numeric(source["target_tar"], errors="raise")
    source["threshold"] = pd.to_numeric(source["threshold"], errors="raise")

    lookup: dict[tuple[str, str, float], float] = {}
    for _, row in source.iterrows():
        key = (str(row["method_key"]), str(row["dataset"]), round(float(row["target_tar"]), 12))
        if key in lookup:
            raise ValueError(f"Duplicate VAL-calibrated TEST threshold row: {key}")
        lookup[key] = float(row["threshold"])
    return lookup


def build_high_recall_pair_outcomes(
    threshold_to_target: pd.DataFrame,
    score_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> pd.DataFrame:
    thresholds = _threshold_lookup(threshold_to_target)
    rows: list[pd.DataFrame] = []
    for spec in METHODS:
        for dataset in DATASETS:
            scores = score_cache[(spec.display_name, dataset, "test")].copy()
            scores["finger_type"] = scores["frgp"].map(lambda value: FINGER_TYPES.get(int(value), "unknown"))
            scores["side"] = scores["frgp"].map(lambda value: SIDES.get(int(value), "unknown"))
            scores["finger_name"] = scores["frgp"].map(lambda value: FRGP_NAMES.get(int(value), f"unknown_{value}"))
            for target_tar in PRIMARY_HIGH_RECALL_TARGET_TARS:
                threshold = thresholds[(spec.method_key, dataset, round(float(target_tar), 12))]
                decisions = _outcomes_for_scores(scores, threshold)
                decisions["method_id"] = spec.method_id
                decisions["method_key"] = spec.method_key
                decisions["method_label"] = spec.display_name
                decisions["analysis_mode"] = VAL_CALIBRATED_MODE
                decisions["protocol_role"] = PUBLISHABLE_PROTOCOL_ROLE
                decisions["target_tar"] = float(target_tar)
                decisions["threshold"] = threshold
                rows.append(decisions)

    if rows:
        result = pd.concat(rows, ignore_index=True)
    else:
        result = pd.DataFrame()
    columns = [
        "method_id",
        "method_key",
        "method_label",
        "dataset",
        "split",
        "analysis_mode",
        "protocol_role",
        "target_tar",
        "threshold",
        "pair_id",
        "label",
        "score",
        "accepted",
        "outcome",
        "subject_a",
        "subject_b",
        "finger_position",
        "frgp",
        "finger_type",
        "side",
        "finger_name",
        "path_a",
        "path_b",
    ]
    return result.loc[:, columns]


def build_high_recall_false_reject_details(pair_outcomes: pd.DataFrame) -> pd.DataFrame:
    details = pair_outcomes[
        (pair_outcomes["label"].astype(int) == 1) & (~pair_outcomes["accepted"].astype(bool))
    ].copy()
    return details.sort_values(
        ["method_label", "dataset", "target_tar", "finger_type", "frgp", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def build_high_recall_false_accept_details(pair_outcomes: pd.DataFrame) -> pd.DataFrame:
    details = pair_outcomes[
        (pair_outcomes["label"].astype(int) == 0) & (pair_outcomes["accepted"].astype(bool))
    ].copy()
    return details.sort_values(
        ["method_label", "dataset", "target_tar", "finger_type", "frgp", "pair_id"], kind="stable"
    ).reset_index(drop=True)


def _method_summary_row(
    *,
    group: pd.DataFrame,
    dataset: str,
    threshold: Any,
    threshold_scope: str,
) -> dict[str, Any]:
    outcomes = group["outcome"].value_counts()
    ta = int(outcomes.get("TA", 0))
    fr = int(outcomes.get("FR", 0))
    fa = int(outcomes.get("FA", 0))
    tr = int(outcomes.get("TR", 0))
    n_positive = ta + fr
    n_negative = fa + tr
    return {
        "method_id": str(group.iloc[0]["method_id"]),
        "method_key": str(group.iloc[0]["method_key"]),
        "method_label": str(group.iloc[0]["method_label"]),
        "dataset": dataset,
        "split": "test",
        "analysis_mode": VAL_CALIBRATED_MODE,
        "protocol_role": PUBLISHABLE_PROTOCOL_ROLE,
        "target_tar": float(group.iloc[0]["target_tar"]),
        "threshold": threshold,
        "threshold_scope": threshold_scope,
        "TA": ta,
        "FR": fr,
        "FA": fa,
        "TR": tr,
        "TAR": ta / n_positive if n_positive else float("nan"),
        "FAR": fa / n_negative if n_negative else float("nan"),
        "FRR": fr / n_positive if n_positive else float("nan"),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def build_high_recall_method_comparison(pair_outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, group in pair_outcomes.groupby(
        ["method_key", "method_id", "method_label", "dataset", "target_tar"], sort=True, dropna=False
    ):
        rows.append(
            _method_summary_row(
                group=group,
                dataset=str(group.iloc[0]["dataset"]),
                threshold=float(group.iloc[0]["threshold"]),
                threshold_scope="dataset_val_threshold",
            )
        )

    for _, group in pair_outcomes.groupby(
        ["method_key", "method_id", "method_label", "target_tar"], sort=True, dropna=False
    ):
        rows.append(
            _method_summary_row(
                group=group,
                dataset="combined_sd300b_sd300c",
                threshold="per_dataset",
                threshold_scope="per_dataset_val_thresholds",
            )
        )

    columns = [
        "method_id",
        "method_key",
        "method_label",
        "dataset",
        "split",
        "analysis_mode",
        "protocol_role",
        "target_tar",
        "threshold",
        "threshold_scope",
        "TA",
        "FR",
        "FA",
        "TR",
        "TAR",
        "FAR",
        "FRR",
        "n_positive",
        "n_negative",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dataset", "target_tar", "method_label"], kind="stable"
    ).reset_index(drop=True)


def build_low_far_verification_summary(
    score_cache: dict[tuple[str, str, str], pd.DataFrame],
    threshold_cache: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    method = "Deep Fusion v2 Statistical"
    rows: list[dict[str, Any]] = []
    thresholds = threshold_cache[method]
    for dataset in DATASETS:
        threshold = threshold_for_far(thresholds, dataset, 0.01)
        scores = score_cache[(method, dataset, "test")]
        metrics = metrics_at_threshold(scores, threshold)
        rows.append(
            {
                "method_label": method,
                "dataset": dataset,
                "split": "test",
                "analysis_mode": "low_far_verification",
                "target_far": 0.01,
                "threshold": threshold,
                "TA": metrics["TA"],
                "FR": metrics["FR"],
                "FA": metrics["FA"],
                "TR": metrics["TR"],
                "TAR": metrics["achieved_TAR"],
                "FAR": metrics["achieved_FAR"],
                "FRR": metrics["FRR"],
                "n_positive": metrics["n_positive"],
                "n_negative": metrics["n_negative"],
            }
        )

    combined = pd.DataFrame(rows)
    ta = int(combined["TA"].sum())
    fr = int(combined["FR"].sum())
    fa = int(combined["FA"].sum())
    tr = int(combined["TR"].sum())
    n_positive = ta + fr
    n_negative = fa + tr
    rows.append(
        {
            "method_label": method,
            "dataset": "combined_sd300b_sd300c",
            "split": "test",
            "analysis_mode": "low_far_verification",
            "target_far": 0.01,
            "threshold": "per_dataset",
            "TA": ta,
            "FR": fr,
            "FA": fa,
            "TR": tr,
            "TAR": ta / n_positive if n_positive else float("nan"),
            "FAR": fa / n_negative if n_negative else float("nan"),
            "FRR": fr / n_positive if n_positive else float("nan"),
            "n_positive": n_positive,
            "n_negative": n_negative,
        }
    )
    return pd.DataFrame(rows)


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.6g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    return str(value).replace("|", "\\|")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    data = frame.loc[:, columns].copy()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in data.iterrows():
        body.append("| " + " | ".join(_format_cell(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *body])


def render_threshold_md(thresholds: pd.DataFrame) -> str:
    oracle = thresholds[
        (thresholds["analysis_mode"] == SPLIT_ORACLE_MODE)
        & (thresholds["split"] == "test")
        & thresholds["target_tar"].isin(PRIMARY_HIGH_RECALL_TARGET_TARS)
    ].sort_values(["method", "dataset", "target_tar"])
    calibrated = thresholds[
        (thresholds["analysis_mode"] == VAL_CALIBRATED_MODE)
        & (thresholds["split"] == "test")
        & thresholds["target_tar"].isin(PRIMARY_HIGH_RECALL_TARGET_TARS)
    ].sort_values(["method", "dataset", "target_tar"])
    lines = [
        "# Threshold to target TAR",
        "",
        "Acceptance rule: `score >= threshold`. Scores are higher-is-better.",
        "The split-oracle mode is descriptive only; TEST rows in that mode use TEST to choose a threshold.",
        "The VAL-calibrated mode chooses the threshold on VAL and applies it unchanged to TEST.",
        "",
        "## TEST split-oracle descriptive at 99%, 99.5%, and 100% TAR",
        "",
        markdown_table(
            oracle,
            ["method", "dataset", "target_tar", "threshold", "achieved_TAR", "achieved_FAR", "TA", "FR", "FA", "TR"],
        ),
        "",
        "## TEST at VAL-calibrated thresholds for 99%, 99.5%, and 100% VAL TAR",
        "",
        markdown_table(
            calibrated,
            [
                "method",
                "dataset",
                "target_tar",
                "threshold",
                "achieved_TAR",
                "achieved_FAR",
                "val_TAR",
                "val_FAR",
                "TA",
                "FR",
                "FA",
                "TR",
            ],
        ),
        "",
        f"Total CSV rows: {len(thresholds)}.",
    ]
    return "\n".join(lines) + "\n"


def hardest_finger_types(summary: pd.DataFrame) -> pd.DataFrame:
    sub = summary[
        (summary["method"] == "Deep Fusion v2 Statistical")
        & (summary["split"] == "test")
        & np.isclose(pd.to_numeric(summary["target_far"], errors="coerce"), 0.01, rtol=0.0, atol=1e-12)
        & (summary["aggregation_level"] == "finger_type")
    ].copy()
    rows: list[pd.Series] = []
    for _, group in sub.groupby("dataset", sort=True):
        ordered = group.sort_values(["FRR", "mean_positive_score", "finger_type"], ascending=[False, True, True])
        rows.append(ordered.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame()


def render_finger_md(summary: pd.DataFrame, failures: pd.DataFrame) -> str:
    hardest = hardest_finger_types(summary)
    stat_test = summary[
        (summary["method"] == "Deep Fusion v2 Statistical")
        & (summary["split"] == "test")
        & np.isclose(pd.to_numeric(summary["target_far"], errors="coerce"), 0.01, rtol=0.0, atol=1e-12)
        & (summary["aggregation_level"] == "finger_type")
    ].sort_values(["dataset", "FRR", "finger_type"], ascending=[True, False, True])
    thumb_notes: list[str] = []
    for dataset, group in stat_test.groupby("dataset", sort=True):
        thumbs = group[group["finger_type"] == "thumb"]
        thumb_frr = float(thumbs.iloc[0]["FRR"]) if not thumbs.empty else float("nan")
        max_frr = float(pd.to_numeric(group["FRR"], errors="coerce").max())
        if math.isclose(thumb_frr, max_frr, rel_tol=0.0, abs_tol=1e-12):
            thumb_notes.append(f"- {dataset}: thumbs are tied for hardest at FRR={thumb_frr:.6g}.")
        else:
            thumb_notes.append(f"- {dataset}: thumbs are not hardest; thumb FRR={thumb_frr:.6g}, max FRR={max_frr:.6g}.")
    failure_counts = (
        failures.groupby(["dataset", "split", "target_far"], sort=True)
        .size()
        .reset_index(name="positive_pairs_failed_at_least_one_deep_method")
    )
    lines = [
        "# Finger type quality summary",
        "",
        "Thresholds are the method-local `plain_roll_final_thresholds.csv` values calibrated on VAL target FAR.",
        "Rows aggregate positive plain-roll pairs only; failures are positive pairs rejected at the threshold.",
        "",
        "## Hardest finger type: Deep Fusion v2 Statistical @ 1% FAR on TEST",
        "",
        markdown_table(
            hardest,
            ["dataset", "finger_type", "n_positive_pairs", "TA", "FR", "TAR", "FRR", "mean_positive_score"],
        ),
        "",
        "## Thumb check",
        "",
        "\n".join(thumb_notes) if thumb_notes else "_No thumb rows._",
        "",
        "## Deep Fusion v2 Statistical finger-type rows @ 1% FAR on TEST",
        "",
        markdown_table(
            stat_test,
            ["dataset", "finger_type", "n_positive_pairs", "TA", "FR", "TAR", "FRR", "mean_positive_score"],
        ),
        "",
        "## Failure-detail row counts",
        "",
        markdown_table(
            failure_counts,
            ["dataset", "split", "target_far", "positive_pairs_failed_at_least_one_deep_method"],
        ),
        "",
        f"Summary CSV rows: {len(summary)}. Failure detail CSV rows: {len(failures)}.",
    ]
    return "\n".join(lines) + "\n"


def render_finger_type_far_at_target_tar_md(far_by_finger: pd.DataFrame) -> str:
    stat_test = far_by_finger[
        (far_by_finger["method"] == "Deep Fusion v2 Statistical")
        & (far_by_finger["split"] == "test")
        & (far_by_finger["aggregation_level"] == "finger_type")
    ].copy()
    stat_test = stat_test[
        stat_test["target_tar"].map(lambda value: round(float(value), 12) in PRIMARY_HIGH_RECALL_TARGET_TAR_KEYS)
    ]
    stat_test = stat_test.sort_values(["analysis_mode", "target_tar", "dataset", "finger_type"], kind="stable")

    all_finger = far_by_finger[far_by_finger["aggregation_level"] == "finger_type"].copy()
    reconstruction = (
        all_finger.groupby(["method", "dataset", "split", "analysis_mode", "target_tar"], sort=True)
        .agg(n_negative_pairs=("n_negative_pairs", "sum"), FA=("FA", "sum"))
        .reset_index()
    )
    reconstruction["overall_FAR_from_finger_types"] = reconstruction["FA"] / reconstruction["n_negative_pairs"]

    lines = [
        "# Finger type FAR at target TAR thresholds",
        "",
        "Input threshold table: `threshold_to_target_tar.csv`.",
        "Acceptance rule: `score >= threshold`.",
        "For `split_oracle_descriptive`, each split uses its own threshold row.",
        "For `val_calibrated_apply_to_test`, VAL and TEST rows use the VAL-selected threshold recorded in the table.",
        "",
        "## Deep Fusion v2 Statistical TEST finger-type FAR",
        "",
        markdown_table(
            stat_test,
            [
                "dataset",
                "analysis_mode",
                "target_tar",
                "finger_type",
                "threshold",
                "n_negative_pairs",
                "FA",
                "TR",
                "FAR",
                "n_positive_pairs",
                "TA",
                "FR",
                "TAR",
                "FRR",
            ],
        ),
        "",
        "## Reconstructed overall FAR from finger-type rows",
        "",
        markdown_table(
            reconstruction,
            [
                "method",
                "dataset",
                "split",
                "analysis_mode",
                "target_tar",
                "n_negative_pairs",
                "FA",
                "overall_FAR_from_finger_types",
            ],
        ),
        "",
        f"CSV rows: {len(far_by_finger)}.",
    ]
    return "\n".join(lines) + "\n"


def _format_percent(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(number):
        return ""
    return f"{number * 100:.{digits}f}%"


def _statistical_combined_rows(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = comparison[
        (comparison["method_label"] == "Deep Fusion v2 Statistical")
        & (comparison["dataset"] == "combined_sd300b_sd300c")
        & (comparison["analysis_mode"] == VAL_CALIBRATED_MODE)
    ].copy()
    return rows.sort_values("target_tar", kind="stable")


def _combined_finger_type_problem_rows(far_by_finger: pd.DataFrame, target_tar: float) -> pd.DataFrame:
    source = far_by_finger[
        (far_by_finger["method"] == "Deep Fusion v2 Statistical")
        & (far_by_finger["dataset"].isin(DATASETS))
        & (far_by_finger["split"] == "test")
        & (far_by_finger["analysis_mode"] == VAL_CALIBRATED_MODE)
        & (far_by_finger["aggregation_level"] == "finger_type")
        & np.isclose(pd.to_numeric(far_by_finger["target_tar"], errors="coerce"), target_tar, rtol=0.0, atol=1e-12)
    ].copy()
    if source.empty:
        return pd.DataFrame()
    grouped = (
        source.groupby("finger_type", sort=True)
        .agg(
            n_positive_pairs=("n_positive_pairs", "sum"),
            TA=("TA", "sum"),
            FR=("FR", "sum"),
            n_negative_pairs=("n_negative_pairs", "sum"),
            FA=("FA", "sum"),
            TR=("TR", "sum"),
        )
        .reset_index()
    )
    grouped["TAR"] = grouped["TA"] / grouped["n_positive_pairs"]
    grouped["FRR"] = grouped["FR"] / grouped["n_positive_pairs"]
    grouped["FAR"] = grouped["FA"] / grouped["n_negative_pairs"]
    return grouped.sort_values(["FRR", "FAR", "finger_type"], ascending=[False, False, True], kind="stable")


def render_high_recall_threshold_summary(comparison: pd.DataFrame) -> str:
    stat = _statistical_combined_rows(comparison)
    stat_display = stat[["target_tar", "TAR", "FAR", "FR", "FA", "n_positive", "n_negative"]].copy()
    stat_display["TAR"] = stat_display["TAR"].map(_format_percent)
    stat_display["FAR"] = stat_display["FAR"].map(_format_percent)

    combined_995 = comparison[
        (comparison["dataset"] == "combined_sd300b_sd300c")
        & np.isclose(pd.to_numeric(comparison["target_tar"], errors="coerce"), 0.995, rtol=0.0, atol=1e-12)
    ].copy()
    combined_995 = combined_995.sort_values(["FAR", "method_label"], ascending=[True, True], kind="stable")
    combined_995_display = combined_995[["method_label", "TAR", "FAR", "FR", "FA"]].copy()
    combined_995_display["TAR"] = combined_995_display["TAR"].map(_format_percent)
    combined_995_display["FAR"] = combined_995_display["FAR"].map(_format_percent)

    lines = [
        "# High-recall threshold summary",
        "",
        "Primary protocol: `val_calibrated_apply_to_test`. The threshold is selected on VAL positives, then applied unchanged to TEST.",
        f"Primary target TARs: {', '.join(str(target) for target in PRIMARY_HIGH_RECALL_TARGET_TARS)}.",
        "",
        "## Operating points",
        "",
        "- 99% TAR is the least expensive high-recall screen here; it leaves a small false-reject tail while keeping FAR lower than the stricter targets.",
        "- 99.5% TAR is the most interesting practical operating point because it halves the remaining false rejects versus 99% for Deep Fusion v2 Statistical, with a moderate additional FAR cost.",
        "- 100% TAR is an upper-bound screening mode: the threshold must be low enough to accept every VAL positive, so it can drive very high TEST FAR.",
        "",
        "## Deep Fusion v2 Statistical, combined TEST",
        "",
        markdown_table(stat_display, ["target_tar", "TAR", "FAR", "FR", "FA", "n_positive", "n_negative"]),
        "",
        "## Combined TEST method comparison at 99.5% target TAR",
        "",
        markdown_table(combined_995_display, ["method_label", "TAR", "FAR", "FR", "FA"]),
        "",
        "## Protocol note",
        "",
        "`split_oracle_descriptive` rows are descriptive only because they choose thresholds from the same split being evaluated. "
        "`val_calibrated_apply_to_test` is the publishable protocol because TEST is only used after the VAL threshold has been frozen.",
    ]
    return "\n".join(lines) + "\n"


def render_pre_final_menachem_experiments_summary(
    comparison: pd.DataFrame,
    low_far_summary: pd.DataFrame,
    far_by_finger: pd.DataFrame,
) -> str:
    low_far = low_far_summary[low_far_summary["dataset"] == "combined_sd300b_sd300c"].iloc[0]
    stat = _statistical_combined_rows(comparison)
    stat_display = stat[["target_tar", "TAR", "FAR", "FR", "FA", "n_positive", "n_negative"]].copy()
    stat_display["TAR"] = stat_display["TAR"].map(_format_percent)
    stat_display["FAR"] = stat_display["FAR"].map(_format_percent)

    finger_995 = _combined_finger_type_problem_rows(far_by_finger, 0.995)
    finger_columns = ["finger_type", "TAR", "FAR", "FR", "FA", "n_positive_pairs", "n_negative_pairs"]
    if finger_995.empty:
        finger_display = pd.DataFrame(columns=finger_columns)
    else:
        finger_display = finger_995[finger_columns].copy()
    if not finger_display.empty:
        finger_display["TAR"] = finger_display["TAR"].map(_format_percent)
        finger_display["FAR"] = finger_display["FAR"].map(_format_percent)

    low_far_sentence = (
        "Previous canonical Fusion v2 verification mode: "
        f"TAR={_format_percent(low_far['TAR'])} at FAR={_format_percent(low_far['FAR'])} "
        f"(target FAR={float(low_far['target_far']):.3g}, combined SD300B+SD300C TEST)."
    )

    lines = [
        "# Pre-final Menachem experiment summary",
        "",
        low_far_sentence,
        "",
        "The Menachem experiment asks a different question: instead of maximizing TAR at a fixed low FAR, it asks what FAR cost is required to reach near-99%, 99.5%, and 100% TAR in a screening setting.",
        "",
        "## Deep Fusion v2 Statistical, high-recall screening",
        "",
        "VAL-calibrated threshold applied to TEST, combined SD300B+SD300C:",
        "",
        markdown_table(stat_display, ["target_tar", "TAR", "FAR", "FR", "FA", "n_positive", "n_negative"]),
        "",
        "100% TEST TAR is possible here, but it is expensive: the combined TEST FAR rises sharply because the VAL-calibrated threshold has to accept every VAL positive.",
        "The 99.5% operating point is the strongest practical/research candidate: it cuts the combined false rejects from 20 to 10 versus 99%, while FAR rises from the mid-20% range to the low-30% range.",
        "",
        "## Finger-type signal at 99.5%",
        "",
        "Little fingers should stay highlighted in follow-up analysis; at 99.5% they carry the largest finger-type FAR burden, while the small remaining false-reject tail is spread across several finger types.",
        "",
        markdown_table(
            finger_display,
            ["finger_type", "TAR", "FAR", "FR", "FA", "n_positive_pairs", "n_negative_pairs"],
        ),
        "",
        "## Publishable protocol",
        "",
        "`val_calibrated_apply_to_test` is the publishable protocol. `split_oracle_descriptive` is useful for understanding split-local upper bounds, but it is not a publishable TEST protocol because TEST participates in threshold selection.",
    ]
    return "\n".join(lines) + "\n"


def validate_compatibility_ddp_scores(
    repo_root: Path,
    tracker: InputTracker,
    score_cache: dict[tuple[str, str, str], pd.DataFrame],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    deep_specs = [spec for spec in METHODS if spec.is_deep_fusion]
    for dataset in DATASETS:
        for split in SPLITS:
            ddp_rel = COMPAT_DDP_DIR / "scores" / f"scores_{dataset}_deep_pair_reranker_fast_ddp_{split}.csv"
            _require_not_forbidden(ddp_rel)
            ddp = tracker.read_csv(ddp_rel, f"compatibility DDP {dataset} {split} scores")
            ddp = ddp[["dataset", "split", "pair_id", "label", "score"]].copy()
            ddp["pair_id"] = ddp["pair_id"].astype(str).str.strip()
            ddp["label"] = pd.to_numeric(ddp["label"], errors="raise").astype(int)
            ddp["score"] = pd.to_numeric(ddp["score"], errors="raise")
            for spec in deep_specs:
                raw_path = spec.score_path(dataset, split)
                raw = tracker.read_csv(raw_path, f"{spec.display_name} deep-score check {dataset} {split}")
                if "deep_score" not in raw.columns:
                    raise ValueError(f"{raw_path} missing deep_score for compatibility check")
                raw = raw[["dataset", "split", "pair_id", "label", "deep_score"]].copy()
                raw["pair_id"] = raw["pair_id"].astype(str).str.strip()
                raw["label"] = pd.to_numeric(raw["label"], errors="raise").astype(int)
                raw["deep_score"] = pd.to_numeric(raw["deep_score"], errors="raise")
                merged = raw.merge(ddp, on=["dataset", "split", "pair_id", "label"], validate="one_to_one")
                if len(merged) != len(raw):
                    raise ValueError(
                        f"Compatibility DDP score check dropped rows for {spec.display_name} {dataset} {split}"
                    )
                diff = (merged["deep_score"] - merged["score"]).abs()
                mismatches = int((diff > 1e-6).sum())
                if mismatches:
                    raise ValueError(
                        f"Compatibility DDP score mismatch for {spec.display_name} {dataset} {split}: {mismatches}"
                    )
                results.append(
                    {
                        "method": spec.display_name,
                        "dataset": dataset,
                        "split": split,
                        "rows_checked": int(len(merged)),
                        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
                    }
                )
    return results


def existing_created_at(manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    value = data.get("created_at")
    return str(value) if value else None


def write_manifest(
    *,
    repo_root: Path,
    outdir: Path,
    tracker: InputTracker,
    output_files: list[Path],
    compatibility_checks: list[dict[str, Any]],
) -> None:
    manifest_path = outdir / "pre_final_experiments_manifest.json"
    created_at = existing_created_at(manifest_path) or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    outputs = [relpath(path, repo_root) for path in output_files]
    tracked_inputs = [relpath(path, repo_root) for path in tracker.files()]
    manifest = {
        "created_at": created_at,
        "generated_at": generated_at,
        "repo_root": str(repo_root),
        "script_path": relpath(Path(__file__), repo_root),
        "input_files": tracked_inputs,
        "input_score_files": [path for path in tracked_inputs if "/scores/" in path],
        "input_threshold_files": [path for path in tracked_inputs if path.endswith("plain_roll_final_thresholds.csv")],
        "input_file_sha256": tracker.sha256(),
        "output_files": outputs,
        "methods": [spec.display_name for spec in METHODS],
        "datasets": list(DATASETS),
        "splits": list(SPLITS),
        "target_fars": list(TARGET_FARS),
        "target_tars": list(TARGET_TARS),
        "primary_high_recall_target_tars": list(PRIMARY_HIGH_RECALL_TARGET_TARS),
        "score_higher_is_better": True,
        "acceptance_rule": "score >= threshold",
        "threshold_selection_rule": "maximum threshold satisfying target TAR",
        "sd300_frgp_semantics": "anatomical",
        "legacy_8finger_results_used": False,
        "test_used_for_training": False,
        "analysis_modes": [SPLIT_ORACLE_MODE, VAL_CALIBRATED_MODE],
        "publishable_protocol": VAL_CALIBRATED_MODE,
        "descriptive_only_protocol": SPLIT_ORACLE_MODE,
        "threshold_target_tar_rule_detail": (
            "For a target TAR, positive scores are sorted descending and the threshold is the "
            "ceil(target_tar * n_positive)-th positive score; ties may make achieved TAR higher."
        ),
        "target_far_threshold_source": "method-local plain_roll_final_thresholds.csv, calibrated on VAL",
        "legacy_low_far_outputs": ["finger_type_quality_summary.csv", "finger_type_failure_details.csv"],
        "row_level_high_recall_diagnostics": [
            "high_recall_pair_outcomes.csv",
            "high_recall_false_reject_details.csv",
            "high_recall_false_accept_details.csv",
        ],
        "compatibility_current_v2_score_check": compatibility_checks,
        "forbidden_inputs_not_used": [FORBIDDEN_COMPAT_TRAIN_SCORES.as_posix()],
        "notes": [
            "No training was run.",
            "Kaggle was not run.",
            "Deep Fusion was not rerun; only existing final score CSVs were analyzed.",
            "Manifests and pair bundles were not modified.",
            "Legacy candidate directories were not used as final-result inputs.",
            "The explicitly forbidden deep_pair_reranker_fast_ddp_train_scores path was not read.",
            "finger_type_far_at_target_tar validates that finger-type FA and negative counts reconstruct overall FAR.",
            "val_calibrated_apply_to_test is the publishable protocol: thresholds are selected on VAL and frozen before TEST.",
            "split_oracle_descriptive is descriptive only and must not be presented as the publishable TEST protocol.",
            "high_recall_* outputs use target_tar operating points, not target_far operating points.",
            "finger_type_failure_details.csv is a legacy low-FAR diagnostic and is not the row-level high-recall diagnostic.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    tracker = InputTracker(repo_root)
    tracker.read_csv(AUDIT_ALLOWLIST, "final artifact allowlist")
    tracker.read_csv(AUDIT_LEGACY_CANDIDATES, "legacy artifact candidates")

    score_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    threshold_cache: dict[str, pd.DataFrame] = {}
    for spec in METHODS:
        threshold_cache[spec.display_name] = load_thresholds(tracker, spec)
        for dataset in DATASETS:
            for split in SPLITS:
                score_cache[(spec.display_name, dataset, split)] = load_score(repo_root, tracker, spec, dataset, split)

    compatibility_checks = validate_compatibility_ddp_scores(repo_root, tracker, score_cache)

    threshold_to_target = build_threshold_to_target_tar(score_cache)
    finger_summary = build_finger_type_quality_summary(score_cache, threshold_cache)
    failure_details = build_finger_type_failure_details(score_cache, threshold_cache)
    far_at_tar = build_finger_type_far_at_target_tar(threshold_to_target, score_cache)
    high_recall_pair_outcomes = build_high_recall_pair_outcomes(threshold_to_target, score_cache)
    high_recall_false_rejects = build_high_recall_false_reject_details(high_recall_pair_outcomes)
    high_recall_false_accepts = build_high_recall_false_accept_details(high_recall_pair_outcomes)
    high_recall_method_comparison = build_high_recall_method_comparison(high_recall_pair_outcomes)
    low_far_summary = build_low_far_verification_summary(score_cache, threshold_cache)

    threshold_csv = outdir / "threshold_to_target_tar.csv"
    threshold_md = outdir / "threshold_to_target_tar.md"
    finger_csv = outdir / "finger_type_quality_summary.csv"
    finger_md = outdir / "finger_type_quality_summary.md"
    failure_csv = outdir / "finger_type_failure_details.csv"
    far_at_tar_csv = outdir / "finger_type_far_at_target_tar.csv"
    far_at_tar_md = outdir / "finger_type_far_at_target_tar.md"
    high_recall_pair_outcomes_csv = outdir / "high_recall_pair_outcomes.csv"
    high_recall_false_rejects_csv = outdir / "high_recall_false_reject_details.csv"
    high_recall_false_accepts_csv = outdir / "high_recall_false_accept_details.csv"
    high_recall_method_comparison_csv = outdir / "high_recall_method_comparison.csv"
    high_recall_threshold_summary_md = outdir / "high_recall_threshold_summary.md"
    pre_final_summary_md = outdir / "pre_final_menachem_experiments_summary.md"
    manifest_json = outdir / "pre_final_experiments_manifest.json"

    threshold_to_target.to_csv(threshold_csv, index=False)
    threshold_md.write_text(render_threshold_md(threshold_to_target), encoding="utf-8")
    finger_summary.to_csv(finger_csv, index=False)
    finger_md.write_text(render_finger_md(finger_summary, failure_details), encoding="utf-8")
    failure_details.to_csv(failure_csv, index=False)
    far_at_tar.to_csv(far_at_tar_csv, index=False)
    far_at_tar_md.write_text(render_finger_type_far_at_target_tar_md(far_at_tar), encoding="utf-8")
    high_recall_pair_outcomes.to_csv(high_recall_pair_outcomes_csv, index=False)
    high_recall_false_rejects.to_csv(high_recall_false_rejects_csv, index=False)
    high_recall_false_accepts.to_csv(high_recall_false_accepts_csv, index=False)
    high_recall_method_comparison.to_csv(high_recall_method_comparison_csv, index=False)
    high_recall_threshold_summary_md.write_text(
        render_high_recall_threshold_summary(high_recall_method_comparison), encoding="utf-8"
    )
    pre_final_summary_md.write_text(
        render_pre_final_menachem_experiments_summary(
            high_recall_method_comparison,
            low_far_summary,
            far_at_tar,
        ),
        encoding="utf-8",
    )
    write_manifest(
        repo_root=repo_root,
        outdir=outdir,
        tracker=tracker,
        output_files=[
            threshold_csv,
            threshold_md,
            finger_csv,
            finger_md,
            failure_csv,
            far_at_tar_csv,
            far_at_tar_md,
            high_recall_pair_outcomes_csv,
            high_recall_false_rejects_csv,
            high_recall_false_accepts_csv,
            high_recall_method_comparison_csv,
            high_recall_threshold_summary_md,
            pre_final_summary_md,
            manifest_json,
        ],
        compatibility_checks=compatibility_checks,
    )

    print(f"[done] wrote {relpath(outdir, repo_root)}")
    print(f"[rows] threshold_to_target_tar={len(threshold_to_target)}")
    print(f"[rows] finger_type_quality_summary={len(finger_summary)}")
    print(f"[rows] finger_type_failure_details={len(failure_details)}")
    print(f"[rows] finger_type_far_at_target_tar={len(far_at_tar)}")
    print(f"[rows] high_recall_pair_outcomes={len(high_recall_pair_outcomes)}")
    print(f"[rows] high_recall_false_reject_details={len(high_recall_false_rejects)}")
    print(f"[rows] high_recall_false_accept_details={len(high_recall_false_accepts)}")
    print(f"[rows] high_recall_method_comparison={len(high_recall_method_comparison)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
