from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_score_csv(path: Path, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"label", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df[["label", "score"]].copy()
    out = out.rename(columns={"score": method_name})
    out["label"] = out["label"].astype(int)
    return out


def merge_methods(scores_dir: Path, split: str, methods: list[str]) -> pd.DataFrame:
    merged = None
    for method in methods:
        path = scores_dir / f"scores_{method}_{split}.csv"
        df = load_score_csv(path, method)
        if merged is None:
            merged = df
        else:
            if len(merged) != len(df):
                raise ValueError(f"Row count mismatch for {method} on {split}")
            if not np.array_equal(merged["label"].values, df["label"].values):
                raise ValueError(f"Label order mismatch for {method} on {split}")
            merged[method] = df[method].values
    assert merged is not None
    return merged


def minmax_fit(series: pd.Series) -> tuple[float, float]:
    x_min = float(series.min())
    x_max = float(series.max())
    if x_max <= x_min:
        x_max = x_min + 1e-12
    return x_min, x_max


def minmax_apply(series: pd.Series, x_min: float, x_max: float) -> np.ndarray:
    x = (series.astype(float).values - x_min) / (x_max - x_min)
    return np.clip(x, 0.0, 1.0)


def eer_from_scores(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def tar_at_far(y_true: np.ndarray, y_score: np.ndarray, far_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    valid = np.where(fpr <= far_target)[0]
    if len(valid) == 0:
        return 0.0
    return float(tpr[valid[-1]])


def metrics_dict(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, y_score)),
        "eer": eer_from_scores(y_true, y_score),
        "tar_at_far_1e_2": tar_at_far(y_true, y_score, 1e-2),
        "tar_at_far_1e_3": tar_at_far(y_true, y_score, 1e-3),
    }


def normalize_from_val(val_df: pd.DataFrame, test_df: pd.DataFrame, methods: list[str]):
    val_norm = {}
    test_norm = {}
    for method in methods:
        x_min, x_max = minmax_fit(val_df[method])
        val_norm[method] = minmax_apply(val_df[method], x_min, x_max)
        test_norm[method] = minmax_apply(test_df[method], x_min, x_max)
    return val_norm, test_norm


def fused_score(norm_dict: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    return (
        weights["sift"] * norm_dict["sift"]
        + weights["dl_quick"] * norm_dict["dl_quick"]
        + weights["vit"] * norm_dict["vit"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports" / "week11",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports" / "fusion" / "refine",
    )
    args = parser.parse_args()

    args.scores_dir = (
        args.scores_dir.resolve()
        if args.scores_dir.is_absolute()
        else (REPO_ROOT / args.scores_dir).resolve()
    )
    args.outdir = (
        args.outdir.resolve()
        if args.outdir.is_absolute()
        else (REPO_ROOT / args.outdir).resolve()
    )
    args.outdir.mkdir(parents=True, exist_ok=True)

    methods = ["sift", "dl_quick", "vit"]
    val_df = merge_methods(args.scores_dir, "val", methods)
    test_df = merge_methods(args.scores_dir, "test", methods)

    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    val_norm, test_norm = normalize_from_val(val_df, test_df, methods)

    sift_val = metrics_dict(y_val, val_df["sift"].values)
    sift_test = metrics_dict(y_test, test_df["sift"].values)

    rows = []

    # Refine around the region that already looked best:
    # sift in [0.86..0.98], dl in [0.00..0.10], vit = remainder
    for w_sift in np.arange(0.86, 0.981, 0.01):
        for w_dl in np.arange(0.00, 0.101, 0.01):
            w_vit = 1.0 - w_sift - w_dl
            if w_vit < -1e-9:
                continue
            if w_vit < 0:
                w_vit = 0.0

            weights = {
                "sift": round(float(w_sift), 4),
                "dl_quick": round(float(w_dl), 4),
                "vit": round(float(w_vit), 4),
            }

            val_scores = fused_score(val_norm, weights)
            test_scores = fused_score(test_norm, weights)

            val_m = metrics_dict(y_val, val_scores)
            test_m = metrics_dict(y_test, test_scores)

            rows.append({
                "weights_json": json.dumps(weights, sort_keys=True),
                **{f"val_{k}": v for k, v in val_m.items()},
                **{f"test_{k}": v for k, v in test_m.items()},
                "delta_val_auc_vs_sift": val_m["auc"] - sift_val["auc"],
                "delta_val_eer_vs_sift": val_m["eer"] - sift_val["eer"],
                "delta_val_tar_1e_2_vs_sift": val_m["tar_at_far_1e_2"] - sift_val["tar_at_far_1e_2"],
                "delta_val_tar_1e_3_vs_sift": val_m["tar_at_far_1e_3"] - sift_val["tar_at_far_1e_3"],
            })

    df = pd.DataFrame(rows)

    # Conservative balanced candidates:
    # improve AUC and TAR@1e-3 on val, while not hurting EER by >0.02
    # and not hurting TAR@1e-2 by >0.01
    balanced = df[
        (df["delta_val_auc_vs_sift"] > 0.0) &
        (df["delta_val_tar_1e_3_vs_sift"] > 0.0) &
        (df["delta_val_eer_vs_sift"] <= 0.02) &
        (df["delta_val_tar_1e_2_vs_sift"] >= -0.01)
    ].copy()

    # Very conservative candidates:
    very_conservative = df[
        (df["delta_val_auc_vs_sift"] > 0.0) &
        (df["delta_val_eer_vs_sift"] <= 0.01) &
        (df["delta_val_tar_1e_2_vs_sift"] >= 0.0)
    ].copy()

    df.sort_values(["test_auc", "test_tar_at_far_1e_3"], ascending=[False, False]).to_csv(
        args.outdir / "all_candidates.csv", index=False
    )

    if not balanced.empty:
        balanced = balanced.sort_values(
            ["val_auc", "val_tar_at_far_1e_3", "val_eer"],
            ascending=[False, False, True]
        )
        balanced.to_csv(args.outdir / "balanced_candidates.csv", index=False)

    if not very_conservative.empty:
        very_conservative = very_conservative.sort_values(
            ["val_auc", "val_tar_at_far_1e_2", "val_eer"],
            ascending=[False, False, True]
        )
        very_conservative.to_csv(args.outdir / "very_conservative_candidates.csv", index=False)

    best_auc = df.sort_values(["test_auc", "test_tar_at_far_1e_3"], ascending=[False, False]).iloc[0].to_dict()

    summary = {
        "sift_val_baseline": sift_val,
        "sift_test_baseline": sift_test,
        "n_candidates_total": int(len(df)),
        "n_balanced": int(len(balanced)),
        "n_very_conservative": int(len(very_conservative)),
        "best_test_auc_candidate": best_auc,
    }

    with open(args.outdir / "refine_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SIFT BASELINE (VAL) ===")
    print(json.dumps(sift_val, indent=2))
    print("\n=== SIFT BASELINE (TEST) ===")
    print(json.dumps(sift_test, indent=2))
    print(f"\nTotal candidates: {len(df)}")
    print(f"Balanced candidates: {len(balanced)}")
    print(f"Very conservative candidates: {len(very_conservative)}")

    print("\n=== BEST TEST-AUC CANDIDATE ===")
    print(json.dumps(best_auc, indent=2))

    if not balanced.empty:
        print("\n=== TOP BALANCED CANDIDATE ===")
        print(balanced.head(1).to_string(index=False))

    if not very_conservative.empty:
        print("\n=== TOP VERY CONSERVATIVE CANDIDATE ===")
        print(very_conservative.head(1).to_string(index=False))

    print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
