from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

METHOD_V1 = "sourceafis_sift_quality_fusion_v1"
METHOD_V2 = "sourceafis_sift_quality_deep_fusion_v2"
DEFAULT_DATASETS = ("nist_sd300b", "nist_sd300c")
DEFAULT_TARGET_FAR = 0.01


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    return df


def score_path(repo_root: Path, method_dir: str, method: str, dataset: str, split: str) -> Path:
    return repo_root / "artifacts/reports/benchmark" / method_dir / "scores" / f"scores_{dataset}_{method}_{split}.csv"


def threshold_for(thresholds: pd.DataFrame, dataset: str, target_far: float) -> float:
    sub = thresholds[
        (thresholds["dataset"].astype(str) == dataset)
        & (pd.to_numeric(thresholds["target_far"], errors="coerce").round(12) == round(float(target_far), 12))
    ]
    if sub.empty:
        raise ValueError(f"No threshold for dataset={dataset}, target_far={target_far}")
    return float(sub.iloc[0]["threshold"])


def normalize_scores(df: pd.DataFrame, *, method_prefix: str) -> pd.DataFrame:
    out = df.copy()
    out["dataset"] = out["dataset"].astype(str).str.strip()
    out["split"] = out["split"].astype(str).str.strip().str.lower()
    out["pair_id"] = out["pair_id"].astype(str).str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)
    if "finger_position" not in out.columns:
        if "frgp" in out.columns:
            out["finger_position"] = out["frgp"]
        else:
            out["finger_position"] = "__missing__"
    keep = [
        "dataset", "split", "pair_id", "label", "subject_a", "subject_b", "finger_position", "frgp", "path_a", "path_b", "score"
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep].rename(columns={"score": f"{method_prefix}_score"})
    return out


def build_for_dataset(repo_root: Path, outdir: Path, dataset: str, target_far: float) -> pd.DataFrame:
    v1_dir = "plain_roll_final_fusion_v1_v2_anatomical_full_pairs"
    v2_dir = "sourceafis_sift_quality_deep_fusion_v2_statistical_anatomical_v2_ddpdeep"
    v1_thresholds = read_csv(repo_root / "artifacts/reports/benchmark" / v1_dir / "plain_roll_final_thresholds.csv", "Fusion v1 thresholds")
    v2_thresholds = read_csv(repo_root / "artifacts/reports/benchmark" / v2_dir / "plain_roll_final_thresholds.csv", "Fusion v2 thresholds")
    v1_thr = threshold_for(v1_thresholds, dataset, target_far)
    v2_thr = threshold_for(v2_thresholds, dataset, target_far)

    v1_scores = normalize_scores(read_csv(score_path(repo_root, v1_dir, METHOD_V1, dataset, "test"), f"{dataset} v1 test scores"), method_prefix="v1")
    v2_scores = normalize_scores(read_csv(score_path(repo_root, v2_dir, METHOD_V2, dataset, "test"), f"{dataset} v2 test scores"), method_prefix="v2")
    key = ["dataset", "split", "pair_id"]
    merged = v1_scores.merge(v2_scores[key + ["label", "v2_score"]], on=key, validate="one_to_one", suffixes=("", "_v2_label"))
    if (pd.to_numeric(merged["label_v2_label"], errors="raise").astype(int) != merged["label"].astype(int)).any():
        raise ValueError(f"Label mismatch between v1/v2 scores for {dataset}")
    merged = merged.drop(columns=["label_v2_label"])
    merged["target_far"] = float(target_far)
    merged["v1_threshold"] = float(v1_thr)
    merged["v2_threshold"] = float(v2_thr)
    merged["v1_accept"] = pd.to_numeric(merged["v1_score"], errors="coerce") >= v1_thr
    merged["v2_accept"] = pd.to_numeric(merged["v2_score"], errors="coerce") >= v2_thr
    merged["positive"] = merged["label"].astype(int) == 1
    merged["negative"] = merged["label"].astype(int) == 0
    merged["v2_rescued_positive"] = merged["positive"] & (~merged["v1_accept"]) & merged["v2_accept"]
    merged["v2_lost_positive"] = merged["positive"] & merged["v1_accept"] & (~merged["v2_accept"])
    merged["v2_fixed_false_accept"] = merged["negative"] & merged["v1_accept"] & (~merged["v2_accept"])
    merged["v2_new_false_accept"] = merged["negative"] & (~merged["v1_accept"]) & merged["v2_accept"]
    merged["both_reject_positive"] = merged["positive"] & (~merged["v1_accept"]) & (~merged["v2_accept"])
    merged["both_accept_negative"] = merged["negative"] & merged["v1_accept"] & merged["v2_accept"]
    return merged


def summarize_flags(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    flags = [
        "v2_rescued_positive",
        "v2_lost_positive",
        "v2_fixed_false_accept",
        "v2_new_false_accept",
        "both_reject_positive",
        "both_accept_negative",
    ]
    rows = []
    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update({"pairs": int(len(sub)), "positives": int((sub["label"] == 1).sum()), "negatives": int((sub["label"] == 0).sum())})
        for flag in flags:
            row[flag] = int(sub[flag].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def score_band(values: pd.Series) -> pd.Series:
    return pd.cut(pd.to_numeric(values, errors="coerce"), bins=[-np.inf, 0.25, 0.5, 0.75, 0.9, 0.97, np.inf], labels=["<=0.25", "0.25-0.50", "0.50-0.75", "0.75-0.90", "0.90-0.97", ">0.97"])


def write_summary(path: Path, pairs: pd.DataFrame, by_dataset: pd.DataFrame, by_finger: pd.DataFrame, target_far: float) -> None:
    lines = [f"# Fusion v2 failure taxonomy @ target FAR {target_far}", ""]
    lines.append("Compares `sourceafis_sift_quality_fusion_v1` against `sourceafis_sift_quality_deep_fusion_v2` on TEST only.")
    lines.append("")
    lines.append("## Overall counts")
    lines.append("")
    flags = ["v2_rescued_positive", "v2_lost_positive", "v2_fixed_false_accept", "v2_new_false_accept", "both_reject_positive", "both_accept_negative"]
    overall = {flag: int(pairs[flag].sum()) for flag in flags}
    overall.update({"pairs": int(len(pairs)), "positives": int((pairs["label"] == 1).sum()), "negatives": int((pairs["label"] == 0).sum())})
    lines.append(pd.DataFrame([overall]).to_markdown(index=False))
    lines.append("\n## By dataset\n")
    lines.append(by_dataset.to_markdown(index=False))
    lines.append("\n## Worst fingers by remaining both-rejected positives\n")
    lines.append(by_finger.sort_values("both_reject_positive", ascending=False).head(20).to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--outdir",
        default=(
            "artifacts/reports/diagnostics/legacy_stale_20260629/"
            "sourceafis_sift_quality_deep_fusion_v2_failure_taxonomy"
        ),
        help="Legacy v1-vs-v2 taxonomy output directory. Use build_current_fusion_v2_diagnostics.py for current canonical diagnostics.",
    )
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--target-far", type=float, default=DEFAULT_TARGET_FAR)
    parser.add_argument("--examples-per-type", type=int, default=50)
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    frames = [build_for_dataset(repo_root, outdir, dataset, float(args.target_far)) for dataset in parse_csv_list(args.datasets)]
    pairs = pd.concat(frames, ignore_index=True, sort=False)
    pairs["v2_score_band"] = score_band(pairs["v2_score"])
    pairs["v1_score_band"] = score_band(pairs["v1_score"])

    by_dataset = summarize_flags(pairs, ["dataset"])
    by_finger = summarize_flags(pairs, ["dataset", "finger_position"])
    by_v2_score_band = summarize_flags(pairs, ["dataset", "v2_score_band"])

    pairs.to_csv(outdir / "failure_taxonomy_pairs.csv", index=False)
    by_dataset.to_csv(outdir / "failure_taxonomy_by_dataset.csv", index=False)
    by_finger.to_csv(outdir / "failure_taxonomy_by_finger.csv", index=False)
    by_v2_score_band.to_csv(outdir / "failure_taxonomy_by_v2_score_band.csv", index=False)

    examples = {
        "rescued_positive_examples.csv": pairs[pairs["v2_rescued_positive"]].sort_values("v2_score", ascending=False),
        "lost_positive_examples.csv": pairs[pairs["v2_lost_positive"]].sort_values("v1_score", ascending=False),
        "fixed_false_accept_examples.csv": pairs[pairs["v2_fixed_false_accept"]].sort_values("v1_score", ascending=False),
        "new_false_accept_examples.csv": pairs[pairs["v2_new_false_accept"]].sort_values("v2_score", ascending=False),
    }
    for name, df in examples.items():
        df.head(int(args.examples_per_type)).to_csv(outdir / name, index=False)

    manifest = {
        "schema_version": "sourceafis_sift_quality_deep_fusion_v2_failure_taxonomy_v1",
        "datasets": parse_csv_list(args.datasets),
        "target_far": float(args.target_far),
        "split": "test",
        "method_v1": METHOD_V1,
        "method_v2": METHOD_V2,
    }
    (outdir / "failure_taxonomy_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary(outdir / "failure_taxonomy_summary.md", pairs, by_dataset, by_finger, float(args.target_far))
    print("[done]", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
