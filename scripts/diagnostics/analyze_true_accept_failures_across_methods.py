from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_TARGET_FARS = (0.005, 0.01)
DEFAULT_SPLIT = "test"

DEFAULT_METHOD_SPECS: dict[str, tuple[str, str]] = {
    "sourceafis": (
        "plain_roll_final_sourceafis_v2_anatomical_full_pairs",
        "sourceafis_open",
    ),
    "fusion_v1": (
        "plain_roll_final_fusion_v1_v2_anatomical_full_pairs",
        "sourceafis_sift_quality_fusion_v1",
    ),
    "fusion_v2": (
        "sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep",
        "sourceafis_sift_quality_deep_fusion_v2",
    ),
    "group_manual_45_15_30_10": (
        "sourceafis_sift_quality_deep_group_weighted_fusion_v2_manual_45_15_30_10_anatomical_v2_ddpdeep",
        "sourceafis_sift_quality_deep_group_weighted_fusion_v2",
    ),
    "group_auto_tar_far_001": (
        "sourceafis_sift_quality_deep_group_weighted_fusion_v2_auto_val_tar_at_far_001_anatomical_v2_ddpdeep",
        "sourceafis_sift_quality_deep_group_weighted_fusion_v2",
    ),
}

KEY_COLUMNS = ["dataset", "split", "pair_id"]
DEFAULT_CONTEXT_COLUMNS = [
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


class TrueAcceptFailureAnalysisError(ValueError):
    """Raised when scores/thresholds cannot be loaded or compared safely."""


@dataclass(frozen=True)
class MethodSpec:
    alias: str
    benchmark_dir: Path
    method_id: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_csv_list(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return tuple(str(item).strip() for item in value if str(item).strip())


def parse_float_list(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        raw = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raw = list(value)
    return tuple(float(item) for item in raw)


def _sanitize_alias(value: str) -> str:
    alias = str(value).strip()
    if not alias:
        raise TrueAcceptFailureAnalysisError("Method alias cannot be empty.")
    bad = [char for char in alias if not (char.isalnum() or char in "_-.")]
    if bad:
        raise TrueAcceptFailureAnalysisError(
            f"Method alias {alias!r} contains unsupported character(s): {sorted(set(bad))}. "
            "Use only letters, digits, underscore, hyphen or dot."
        )
    return alias


def parse_method_specs(
    methods: str,
    *,
    repo_root: Path,
    custom_specs: Iterable[str] = (),
) -> list[MethodSpec]:
    """Parse requested methods.

    Built-in aliases are listed in DEFAULT_METHOD_SPECS. Custom specs can be provided as:
        alias=relative_or_absolute_benchmark_dir:method_id
    """

    specs: dict[str, tuple[str, str]] = dict(DEFAULT_METHOD_SPECS)
    for raw in custom_specs:
        text = str(raw).strip()
        if not text:
            continue
        if "=" not in text or ":" not in text:
            raise TrueAcceptFailureAnalysisError(
                "Custom method specs must use alias=benchmark_dir:method_id, got: " + text
            )
        alias, remainder = text.split("=", 1)
        benchmark_dir, method_id = remainder.rsplit(":", 1)
        specs[_sanitize_alias(alias)] = (benchmark_dir.strip(), method_id.strip())

    selected = parse_csv_list(methods)
    if not selected:
        raise TrueAcceptFailureAnalysisError("At least one method must be selected.")

    out: list[MethodSpec] = []
    benchmark_root = repo_root / "artifacts" / "reports" / "benchmark"
    for alias in selected:
        alias = _sanitize_alias(alias)
        if alias not in specs:
            known = ", ".join(sorted(specs))
            raise TrueAcceptFailureAnalysisError(f"Unknown method alias {alias!r}. Known aliases: {known}")
        benchmark_dir_text, method_id = specs[alias]
        benchmark_dir = Path(benchmark_dir_text)
        if not benchmark_dir.is_absolute():
            benchmark_dir = benchmark_root / benchmark_dir
        out.append(MethodSpec(alias=alias, benchmark_dir=benchmark_dir.resolve(), method_id=method_id))
    return out


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise TrueAcceptFailureAnalysisError(f"{label} is empty: {path}")
    return df


def _find_score_path(spec: MethodSpec, dataset: str, split: str) -> Path:
    candidates = [
        spec.benchmark_dir / "scores" / f"scores_{dataset}_{spec.method_id}_{split}.csv",
        spec.benchmark_dir / "scores" / "ablation" / f"scores_{dataset}_{spec.method_id}_{split}.csv",
        spec.benchmark_dir / f"scores_{dataset}_{spec.method_id}_{split}.csv",
        spec.benchmark_dir / f"{spec.alias}_plain_roll_scores_{split}.csv",
        spec.benchmark_dir / f"{spec.method_id}_plain_roll_scores_{split}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find score CSV for method={spec.alias}, dataset={dataset}, split={split}. Searched:\n  {searched}"
    )


def _threshold_path(spec: MethodSpec) -> Path:
    candidates = [
        spec.benchmark_dir / "plain_roll_final_thresholds.csv",
        spec.benchmark_dir / f"{spec.alias}_plain_roll_thresholds.csv",
        spec.benchmark_dir / f"{spec.method_id}_plain_roll_thresholds.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _score_column(df: pd.DataFrame) -> str:
    for column in ("score", "raw_score", "similarity", "match_score"):
        if column in df.columns:
            return column
    raise TrueAcceptFailureAnalysisError("Score CSV is missing score/raw_score/similarity/match_score column.")


def _normalize_scores(df: pd.DataFrame, *, dataset: str, split: str, score_path: Path) -> pd.DataFrame:
    required = {"pair_id", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise TrueAcceptFailureAnalysisError(f"Score CSV is missing required columns {missing}: {score_path}")

    out = df.copy()
    out["dataset"] = dataset if "dataset" not in out.columns else out["dataset"].fillna(dataset)
    out["split"] = split if "split" not in out.columns else out["split"].fillna(split)
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)

    score_col = _score_column(out)
    out["score"] = pd.to_numeric(out[score_col], errors="coerce")
    if out["score"].isna().any():
        examples = out.loc[out["score"].isna(), KEY_COLUMNS].head(5).to_dict("records")
        raise TrueAcceptFailureAnalysisError(f"Non-numeric scores in {score_path}: {examples}")

    if "finger_position" not in out.columns:
        out["finger_position"] = out["frgp"] if "frgp" in out.columns else "__missing__"
    if "frgp" not in out.columns:
        out["frgp"] = out["finger_position"]

    for column in DEFAULT_CONTEXT_COLUMNS:
        if column not in out.columns:
            out[column] = np.nan

    out = out[(out["dataset"] == dataset) & (out["split"] == split.lower())].copy()
    if out.empty:
        raise TrueAcceptFailureAnalysisError(f"No rows for dataset={dataset}, split={split} in {score_path}")
    dup = out.duplicated(KEY_COLUMNS, keep=False)
    if bool(dup.any()):
        examples = out.loc[dup, KEY_COLUMNS].head(5).to_dict("records")
        raise TrueAcceptFailureAnalysisError(f"Duplicate pair keys in {score_path}: {examples}")
    return out[DEFAULT_CONTEXT_COLUMNS + ["score"]].reset_index(drop=True)


def _threshold_for(thresholds: pd.DataFrame, *, dataset: str, method_id: str, target_far: float) -> float:
    df = thresholds.copy()
    if "dataset" not in df.columns or "target_far" not in df.columns or "threshold" not in df.columns:
        raise TrueAcceptFailureAnalysisError("Threshold CSV must contain dataset, target_far and threshold columns.")
    mask = (df["dataset"].astype(str).str.strip() == dataset) & (
        pd.to_numeric(df["target_far"], errors="coerce").round(12) == round(float(target_far), 12)
    )
    if "method" in df.columns:
        method_mask = df["method"].astype(str).str.strip() == method_id
        if bool((mask & method_mask).any()):
            mask = mask & method_mask
    sub = df[mask]
    if sub.empty:
        raise TrueAcceptFailureAnalysisError(
            f"No threshold for dataset={dataset}, method={method_id}, target_far={target_far}."
        )
    return float(pd.to_numeric(sub.iloc[0]["threshold"], errors="raise"))


def load_method_outcomes(
    spec: MethodSpec,
    *,
    dataset: str,
    split: str,
    target_far: float,
) -> pd.DataFrame:
    score_path = _find_score_path(spec, dataset=dataset, split=split)
    threshold_csv = _threshold_path(spec)
    scores = _normalize_scores(_read_csv(score_path, f"{spec.alias} scores"), dataset=dataset, split=split, score_path=score_path)
    thresholds = _read_csv(threshold_csv, f"{spec.alias} thresholds")
    threshold = _threshold_for(thresholds, dataset=dataset, method_id=spec.method_id, target_far=float(target_far))

    out = scores.copy()
    out["method_alias"] = spec.alias
    out["method_id"] = spec.method_id
    out["target_far"] = float(target_far)
    out["threshold"] = float(threshold)
    out["accepted"] = pd.to_numeric(out["score"], errors="coerce") >= threshold
    out["positive"] = out["label"].astype(int) == 1
    out["negative"] = out["label"].astype(int) == 0
    out["outcome"] = np.select(
        [out["positive"] & out["accepted"], out["positive"] & ~out["accepted"], out["negative"] & out["accepted"], out["negative"] & ~out["accepted"]],
        ["TA", "FR", "FA", "TR"],
        default="UNKNOWN",
    )
    out["score_csv"] = str(score_path)
    out["thresholds_csv"] = str(threshold_csv)
    return out


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_method_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["method_alias", "method_id", "dataset", "split", "target_far"]
    for keys, sub in outcomes.groupby(group_cols, sort=True, dropna=False):
        method_alias, method_id, dataset, split, target_far = keys
        labels = sub["label"].astype(int)
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        ta = int((sub["outcome"] == "TA").sum())
        fr = int((sub["outcome"] == "FR").sum())
        fa = int((sub["outcome"] == "FA").sum())
        tr = int((sub["outcome"] == "TR").sum())
        rows.append(
            {
                "method_alias": method_alias,
                "method_id": method_id,
                "dataset": dataset,
                "split": split,
                "target_far": float(target_far),
                "pairs": int(len(sub)),
                "positives": positives,
                "negatives": negatives,
                "TA": ta,
                "FR": fr,
                "FA": fa,
                "TR": tr,
                "TAR": _safe_rate(ta, positives),
                "FRR": _safe_rate(fr, positives),
                "FAR": _safe_rate(fa, negatives),
                "threshold": float(sub["threshold"].iloc[0]),
                "score_csv": str(sub["score_csv"].iloc[0]),
                "thresholds_csv": str(sub["thresholds_csv"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def build_positive_outcome_matrix(outcomes: pd.DataFrame, method_aliases: list[str]) -> pd.DataFrame:
    positives = outcomes[outcomes["label"].astype(int) == 1].copy()
    if positives.empty:
        return pd.DataFrame()

    context_cols = [
        "dataset",
        "split",
        "target_far",
        "pair_id",
        "label",
        "subject_a",
        "subject_b",
        "finger_position",
        "frgp",
        "path_a",
        "path_b",
    ]
    base = positives[context_cols].drop_duplicates(["dataset", "split", "target_far", "pair_id"])
    dup = base.duplicated(["dataset", "split", "target_far", "pair_id"], keep=False)
    if bool(dup.any()):
        examples = base.loc[dup, ["dataset", "split", "target_far", "pair_id"]].head(5).to_dict("records")
        raise TrueAcceptFailureAnalysisError(f"Positive context contains duplicate pair keys: {examples}")

    matrix = base.copy()
    join_key = ["dataset", "split", "target_far", "pair_id"]
    for alias in method_aliases:
        part = positives[positives["method_alias"] == alias][join_key + ["score", "threshold", "accepted", "outcome"]].copy()
        part = part.rename(
            columns={
                "score": f"{alias}_score",
                "threshold": f"{alias}_threshold",
                "accepted": f"{alias}_accepted",
                "outcome": f"{alias}_outcome",
            }
        )
        matrix = matrix.merge(part, on=join_key, how="left", validate="one_to_one")
        if matrix[f"{alias}_outcome"].isna().any():
            examples = matrix.loc[matrix[f"{alias}_outcome"].isna(), join_key].head(5).to_dict("records")
            raise TrueAcceptFailureAnalysisError(f"Missing positive outcomes for method={alias}: {examples}")
        matrix[f"{alias}_is_fr"] = matrix[f"{alias}_outcome"].astype(str) == "FR"

    fr_cols = [f"{alias}_is_fr" for alias in method_aliases]
    matrix["false_reject_method_count"] = matrix[fr_cols].sum(axis=1).astype(int)
    matrix["true_accept_method_count"] = int(len(method_aliases)) - matrix["false_reject_method_count"]
    matrix["false_reject_methods"] = matrix.apply(
        lambda row: ",".join(alias for alias in method_aliases if bool(row[f"{alias}_is_fr"])), axis=1
    )
    matrix["true_accept_methods"] = matrix.apply(
        lambda row: ",".join(alias for alias in method_aliases if not bool(row[f"{alias}_is_fr"])), axis=1
    )
    matrix["all_methods_false_reject"] = matrix["false_reject_method_count"] == len(method_aliases)
    matrix["all_methods_true_accept"] = matrix["false_reject_method_count"] == 0
    matrix["mixed_outcome_across_methods"] = (~matrix["all_methods_false_reject"]) & (~matrix["all_methods_true_accept"])
    return matrix.sort_values(["dataset", "target_far", "pair_id"]).reset_index(drop=True)


def build_false_reject_sets(matrix: pd.DataFrame, method_aliases: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in matrix.iterrows():
        for alias in method_aliases:
            if bool(row[f"{alias}_is_fr"]):
                rescued_by = [other for other in method_aliases if other != alias and not bool(row[f"{other}_is_fr"])]
                also_rejected_by = [other for other in method_aliases if other != alias and bool(row[f"{other}_is_fr"])]
                rows.append(
                    {
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "target_far": float(row["target_far"]),
                        "pair_id": row["pair_id"],
                        "method_alias": alias,
                        "score": float(row[f"{alias}_score"]),
                        "threshold": float(row[f"{alias}_threshold"]),
                        "subject_a": row.get("subject_a", np.nan),
                        "subject_b": row.get("subject_b", np.nan),
                        "finger_position": row.get("finger_position", np.nan),
                        "frgp": row.get("frgp", np.nan),
                        "accepted_by_other_method_count": len(rescued_by),
                        "accepted_by_methods": ",".join(rescued_by),
                        "also_false_rejected_by_methods": ",".join(also_rejected_by),
                        "all_methods_false_reject": bool(row["all_methods_false_reject"]),
                    }
                )
    return pd.DataFrame(rows)


def build_method_specific_false_rejects(false_rejects: pd.DataFrame) -> pd.DataFrame:
    if false_rejects.empty:
        return false_rejects.copy()
    return false_rejects[false_rejects["accepted_by_other_method_count"] > 0].copy().reset_index(drop=True)


def build_common_false_rejects(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return matrix.copy()
    return matrix[matrix["all_methods_false_reject"]].copy().reset_index(drop=True)


def build_pattern_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, sub in matrix.groupby(["dataset", "split", "target_far"], sort=True, dropna=False):
        dataset, split, target_far = keys
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "target_far": float(target_far),
                "positive_pairs": int(len(sub)),
                "all_methods_true_accept": int(sub["all_methods_true_accept"].sum()),
                "all_methods_false_reject": int(sub["all_methods_false_reject"].sum()),
                "mixed_outcome_across_methods": int(sub["mixed_outcome_across_methods"].sum()),
                "any_method_false_reject": int((sub["false_reject_method_count"] > 0).sum()),
                "any_method_true_accept": int((sub["true_accept_method_count"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_pairwise_complementarity(matrix: pd.DataFrame, method_aliases: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if matrix.empty:
        return pd.DataFrame()
    for keys, sub in matrix.groupby(["dataset", "split", "target_far"], sort=True, dropna=False):
        dataset, split, target_far = keys
        for base in method_aliases:
            base_fr = sub[f"{base}_is_fr"].astype(bool)
            for other in method_aliases:
                if other == base:
                    continue
                other_fr = sub[f"{other}_is_fr"].astype(bool)
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split,
                        "target_far": float(target_far),
                        "base_method": base,
                        "other_method": other,
                        "positive_pairs": int(len(sub)),
                        "both_TA": int((~base_fr & ~other_fr).sum()),
                        "both_FR": int((base_fr & other_fr).sum()),
                        "base_FR_other_TA_rescued_by_other": int((base_fr & ~other_fr).sum()),
                        "base_TA_other_FR_lost_by_other": int((~base_fr & other_fr).sum()),
                    }
                )
    return pd.DataFrame(rows)


def build_sanity_rerun_without_own_fr(method_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in method_summary.iterrows():
        positives = int(row["positives"])
        original_ta = int(row["TA"])
        original_fr = int(row["FR"])
        negatives = int(row["negatives"])
        fa = int(row["FA"])
        tr = int(row["TR"])
        remaining_positives = positives - original_fr
        rows.append(
            {
                "method_alias": row["method_alias"],
                "method_id": row["method_id"],
                "dataset": row["dataset"],
                "split": row["split"],
                "target_far": float(row["target_far"]),
                "diagnostic_filter": "remove this method's own positive false rejects only",
                "original_positives": positives,
                "removed_positive_false_rejects": original_fr,
                "remaining_positives": remaining_positives,
                "TA_after_filter": original_ta,
                "FR_after_filter": 0,
                "TAR_after_filter": _safe_rate(original_ta, remaining_positives),
                "negatives_unchanged": negatives,
                "FA_unchanged": fa,
                "TR_unchanged": tr,
                "FAR_unchanged": _safe_rate(fa, negatives),
                "expected_TAR_100_percent": bool(remaining_positives == original_ta and original_fr >= 0),
                "benchmark_validity_note": "diagnostic only; this removes TEST positives selected after observing method outcomes",
            }
        )
    return pd.DataFrame(rows)


def build_global_removed_sets_metrics(outcomes: pd.DataFrame, matrix: pd.DataFrame, method_aliases: list[str]) -> pd.DataFrame:
    """Optional diagnostic: remove union/intersection positive FR sets and recompute all methods.

    This answers a subtly different question than removing each method's own FRs. Removing the union of
    all FRs makes all remaining positives accepted by all methods. Removing the intersection only removes
    positives missed by every method, and often leaves method-specific FRs behind.
    """

    if matrix.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, pos_sub in matrix.groupby(["dataset", "split", "target_far"], sort=True, dropna=False):
        dataset, split, target_far = keys
        union_ids = set(pos_sub.loc[pos_sub["false_reject_method_count"] > 0, "pair_id"].astype(str))
        common_ids = set(pos_sub.loc[pos_sub["all_methods_false_reject"], "pair_id"].astype(str))
        sets = {
            "remove_union_of_positive_FRs_from_any_method": union_ids,
            "remove_intersection_of_positive_FRs_from_all_methods": common_ids,
        }
        for filter_name, removed_ids in sets.items():
            for alias in method_aliases:
                sub = outcomes[
                    (outcomes["method_alias"] == alias)
                    & (outcomes["dataset"] == dataset)
                    & (outcomes["split"] == split)
                    & (outcomes["target_far"].astype(float).round(12) == round(float(target_far), 12))
                ].copy()
                keep = ~(sub["label"].astype(int).eq(1) & sub["pair_id"].astype(str).isin(removed_ids))
                kept = sub[keep]
                labels = kept["label"].astype(int)
                positives = int((labels == 1).sum())
                negatives = int((labels == 0).sum())
                ta = int((kept["outcome"] == "TA").sum())
                fr = int((kept["outcome"] == "FR").sum())
                fa = int((kept["outcome"] == "FA").sum())
                tr = int((kept["outcome"] == "TR").sum())
                rows.append(
                    {
                        "method_alias": alias,
                        "dataset": dataset,
                        "split": split,
                        "target_far": float(target_far),
                        "diagnostic_filter": filter_name,
                        "removed_positive_pairs": int(len(removed_ids)),
                        "remaining_positives": positives,
                        "TA": ta,
                        "FR": fr,
                        "FA": fa,
                        "TR": tr,
                        "TAR": _safe_rate(ta, positives),
                        "FAR": _safe_rate(fa, negatives),
                        "benchmark_validity_note": "diagnostic only; positives are removed after observing TEST outcomes",
                    }
                )
    return pd.DataFrame(rows)


def _format_pct(value: float) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "nan"
    return f"{100.0 * float(value):.2f}%"


def write_markdown_summary(
    path: Path,
    *,
    method_summary: pd.DataFrame,
    sanity: pd.DataFrame,
    pattern_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    false_rejects: pd.DataFrame,
    common_fr: pd.DataFrame,
    method_specific_fr: pd.DataFrame,
    method_aliases: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# True-accept failure analysis across methods")
    lines.append("")
    lines.append(f"Created at: `{_utc_now()}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("For an individual positive pair, the outcome is either `TA` (true accept) or `FR` (false reject). `TAR` is the aggregate rate `TA / positives`.")
    lines.append("This report treats cases with no true accept as positive pairs that became `FR` at the already selected threshold.")
    lines.append("")
    lines.append("All filtering below is diagnostic only. It must not be reported as a valid benchmark because the removed positives are selected after observing test outcomes.")
    lines.append("")
    lines.append("## Methods")
    lines.append("")
    for alias in method_aliases:
        lines.append(f"- `{alias}`")
    lines.append("")

    lines.append("## Original positive outcomes by method")
    lines.append("")
    display_cols = ["method_alias", "dataset", "split", "target_far", "positives", "TA", "FR", "TAR", "FA", "TR", "FAR"]
    display = method_summary[display_cols].copy()
    display["TAR"] = display["TAR"].map(_format_pct)
    display["FAR"] = display["FAR"].map(_format_pct)
    lines.append(display.to_markdown(index=False))
    lines.append("")

    lines.append("## Diagnostic rerun after removing each method's own false rejects")
    lines.append("")
    diag_cols = ["method_alias", "dataset", "target_far", "removed_positive_false_rejects", "remaining_positives", "TAR_after_filter", "expected_TAR_100_percent"]
    diag = sanity[diag_cols].copy()
    diag["TAR_after_filter"] = diag["TAR_after_filter"].map(_format_pct)
    lines.append(diag.to_markdown(index=False))
    lines.append("")
    lines.append("As expected, when a method's own positive false rejects are removed, its remaining positive set has `TAR=100%`. This is a sanity check, not an improved benchmark.")
    lines.append("")

    lines.append("## Are the false rejects identical across methods?")
    lines.append("")
    if pattern_summary.empty:
        lines.append("No positive outcome matrix was available.")
    else:
        lines.append(pattern_summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Pairwise complementarity")
    lines.append("")
    lines.append("`base_FR_other_TA_rescued_by_other` counts positives that the base method missed but the other method accepted.")
    lines.append("")
    if pairwise.empty:
        lines.append("No pairwise complementarity table was available.")
    else:
        preferred_pairs = []
        for base, other in [("sourceafis", "fusion_v2"), ("fusion_v1", "fusion_v2"), ("fusion_v2", "group_manual_45_15_30_10"), ("fusion_v2", "group_auto_tar_far_001")]:
            if base in method_aliases and other in method_aliases:
                preferred_pairs.append((base, other))
        subset = pairwise.copy()
        if preferred_pairs:
            mask = False
            for base, other in preferred_pairs:
                mask = mask | ((subset["base_method"] == base) & (subset["other_method"] == other))
            subset = subset[mask]
        lines.append(subset.to_markdown(index=False))
    lines.append("")

    lines.append("## Key output files")
    lines.append("")
    lines.append("- `positive_pair_outcome_matrix.csv`: one row per positive pair, with TA/FR outcome for each method.")
    lines.append("- `false_reject_sets_by_method.csv`: all positive false rejects by method.")
    lines.append("- `common_false_rejects_all_methods.csv`: positives missed by every selected method.")
    lines.append("- `method_specific_false_rejects.csv`: positives missed by one method but accepted by at least one other method.")
    lines.append("- `pairwise_complementarity_summary.csv`: rescue/loss counts for every method pair.")
    lines.append("- `rerun_without_own_false_rejects_metrics.csv`: sanity rerun proving TAR=100% after removing each method's own FRs.")
    lines.append("")

    lines.append("## Compact counts")
    lines.append("")
    lines.append(f"Total false-reject rows by method: `{len(false_rejects)}`")
    lines.append(f"Common false-reject pair rows across all selected methods: `{len(common_fr)}`")
    lines.append(f"Method-specific false-reject rows: `{len(method_specific_fr)}`")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    *,
    repo_root: Path,
    outdir: Path,
    methods: list[MethodSpec],
    datasets: Iterable[str],
    split: str,
    target_fars: Iterable[float],
) -> dict[str, pd.DataFrame]:
    split = str(split).lower().strip()
    outdir.mkdir(parents=True, exist_ok=True)
    method_aliases = [spec.alias for spec in methods]

    frames: list[pd.DataFrame] = []
    for spec in methods:
        for dataset in datasets:
            for target_far in target_fars:
                print(f"[load] {spec.alias} {dataset}/{split} target_far={target_far}")
                frames.append(load_method_outcomes(spec, dataset=str(dataset), split=split, target_far=float(target_far)))
    outcomes = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    method_summary = summarize_method_outcomes(outcomes)
    matrix = build_positive_outcome_matrix(outcomes, method_aliases)
    false_rejects = build_false_reject_sets(matrix, method_aliases)
    common_fr = build_common_false_rejects(matrix)
    method_specific_fr = build_method_specific_false_rejects(false_rejects)
    pattern_summary = build_pattern_summary(matrix)
    pairwise = build_pairwise_complementarity(matrix, method_aliases)
    sanity = build_sanity_rerun_without_own_fr(method_summary)
    global_filtered = build_global_removed_sets_metrics(outcomes, matrix, method_aliases)

    outputs = {
        "all_method_outcomes.csv": outcomes,
        "method_outcome_summary.csv": method_summary,
        "positive_pair_outcome_matrix.csv": matrix,
        "false_reject_sets_by_method.csv": false_rejects,
        "common_false_rejects_all_methods.csv": common_fr,
        "method_specific_false_rejects.csv": method_specific_fr,
        "false_reject_pattern_summary.csv": pattern_summary,
        "pairwise_complementarity_summary.csv": pairwise,
        "rerun_without_own_false_rejects_metrics.csv": sanity,
        "global_removed_sets_diagnostic_metrics.csv": global_filtered,
    }
    for name, df in outputs.items():
        df.to_csv(outdir / name, index=False)

    manifest = {
        "schema_version": "true_accept_failure_analysis_v1",
        "created_at": _utc_now(),
        "repo_root": str(repo_root),
        "outdir": str(outdir),
        "datasets": list(datasets),
        "split": split,
        "target_fars": [float(x) for x in target_fars],
        "methods": [
            {"alias": spec.alias, "benchmark_dir": str(spec.benchmark_dir), "method_id": spec.method_id}
            for spec in methods
        ],
        "validity_note": "Diagnostic only; filtered metrics remove positives after observing method outcomes and are not valid benchmark claims.",
    }
    (outdir / "true_accept_failure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown_summary(
        outdir / "true_accept_failure_summary.md",
        method_summary=method_summary,
        sanity=sanity,
        pattern_summary=pattern_summary,
        pairwise=pairwise,
        false_rejects=false_rejects,
        common_fr=common_fr,
        method_specific_fr=method_specific_fr,
        method_aliases=method_aliases,
    )
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze positive-pair true-accept failures across benchmark methods.")
    parser.add_argument("--repo-root", required=True, help="Repository root, e.g. C:\\fingerprint-research")
    parser.add_argument(
        "--outdir",
        default="artifacts/reports/diagnostics/legacy_stale_20260629/true_accept_failures_across_methods",
        help=(
            "Legacy output directory. Relative paths are resolved under repo root. "
            "Use build_current_fusion_v2_diagnostics.py for current canonical diagnostics."
        ),
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--target-fars", default=",".join(str(x) for x in DEFAULT_TARGET_FARS))
    parser.add_argument(
        "--methods",
        default="sourceafis,fusion_v1,fusion_v2,group_manual_45_15_30_10,group_auto_tar_far_001",
        help="Comma-separated aliases. Built-ins include: " + ",".join(sorted(DEFAULT_METHOD_SPECS)),
    )
    parser.add_argument(
        "--method-spec",
        action="append",
        default=[],
        help="Optional custom method spec: alias=benchmark_dir:method_id. Can be repeated.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    methods = parse_method_specs(args.methods, repo_root=repo_root, custom_specs=args.method_spec)
    datasets = parse_csv_list(args.datasets)
    target_fars = parse_float_list(args.target_fars)

    run_analysis(
        repo_root=repo_root,
        outdir=outdir.resolve(),
        methods=methods,
        datasets=datasets,
        split=str(args.split).lower().strip(),
        target_fars=target_fars,
    )
    print("[done]", outdir.resolve())
    print("summary:", (outdir / "true_accept_failure_summary.md").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
