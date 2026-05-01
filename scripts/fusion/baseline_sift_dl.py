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
    x = np.clip(x, 0.0, 1.0)
    return x


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


def rank_average(series_list: list[pd.Series]) -> np.ndarray:
    ranks = []
    n = len(series_list[0])
    for s in series_list:
        ranks.append(s.rank(method="average", pct=True).values)
    return np.mean(np.stack(ranks, axis=1), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports" / "benchmark" / "current",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports" / "fusion" / "baseline_sift_dl",
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

    methods = ["sift", "dl_quick"]

    val_df = merge_methods(args.scores_dir, "val", methods)
    test_df = merge_methods(args.scores_dir, "test", methods)

    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    # Fit normalization only on VAL
    sift_min, sift_max = minmax_fit(val_df["sift"])
    dl_min, dl_max = minmax_fit(val_df["dl_quick"])

    val_sift = minmax_apply(val_df["sift"], sift_min, sift_max)
    val_dl = minmax_apply(val_df["dl_quick"], dl_min, dl_max)

    test_sift = minmax_apply(test_df["sift"], sift_min, sift_max)
    test_dl = minmax_apply(test_df["dl_quick"], dl_min, dl_max)

    baseline_val = {
        "sift": metrics_dict(y_val, val_df["sift"].values),
        "dl_quick": metrics_dict(y_val, val_df["dl_quick"].values),
    }
    baseline_test = {
        "sift": metrics_dict(y_test, test_df["sift"].values),
        "dl_quick": metrics_dict(y_test, test_df["dl_quick"].values),
    }

    best = None
    best_w = None
    best_val_score = None
    for w in np.linspace(0.0, 1.0, 101):
        fused_val = w * val_sift + (1.0 - w) * val_dl
        auc = roc_auc_score(y_val, fused_val)
        if best is None or auc > best:
            best = float(auc)
            best_w = float(w)
            best_val_score = fused_val

    fused_test = best_w * test_sift + (1.0 - best_w) * test_dl

    weighted_val_metrics = metrics_dict(y_val, best_val_score)
    weighted_test_metrics = metrics_dict(y_test, fused_test)

    rankavg_val = rank_average([val_df["sift"], val_df["dl_quick"]])
    rankavg_test = rank_average([test_df["sift"], test_df["dl_quick"]])

    rankavg_val_metrics = metrics_dict(y_val, rankavg_val)
    rankavg_test_metrics = metrics_dict(y_test, rankavg_test)

    pd.DataFrame({"label": y_val, "score": best_val_score}).to_csv(
        args.outdir / "scores_fusion_sift_dl_val.csv", index=False
    )
    pd.DataFrame({"label": y_test, "score": fused_test}).to_csv(
        args.outdir / "scores_fusion_sift_dl_test.csv", index=False
    )
    pd.DataFrame({"label": y_val, "score": rankavg_val}).to_csv(
        args.outdir / "scores_fusion_rankavg_sift_dl_val.csv", index=False
    )
    pd.DataFrame({"label": y_test, "score": rankavg_test}).to_csv(
        args.outdir / "scores_fusion_rankavg_sift_dl_test.csv", index=False
    )

    report = {
        "scores_dir": str(args.scores_dir),
        "methods": methods,
        "weighted_mean": {
            "best_weight_for_sift": best_w,
            "best_weight_for_dl_quick": 1.0 - best_w,
            "val": weighted_val_metrics,
            "test": weighted_test_metrics,
        },
        "rank_average": {
            "val": rankavg_val_metrics,
            "test": rankavg_test_metrics,
        },
        "single_method_baselines": {
            "val": baseline_val,
            "test": baseline_test,
        },
    }

    with open(args.outdir / "fusion_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== SINGLE METHOD BASELINES (VAL) ===")
    print(json.dumps(baseline_val, indent=2))
    print("\n=== SINGLE METHOD BASELINES (TEST) ===")
    print(json.dumps(baseline_test, indent=2))
    print("\n=== BEST WEIGHTED MEAN (SIFT + DL_QUICK) ===")
    print(json.dumps(report["weighted_mean"], indent=2))
    print("\n=== RANK AVERAGE (SIFT + DL_QUICK) ===")
    print(json.dumps(report["rank_average"], indent=2))
    print(f"\nSaved outputs to: {args.outdir}")


if __name__ == "__main__":
    main()