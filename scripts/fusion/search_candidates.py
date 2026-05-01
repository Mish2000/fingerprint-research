from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


def weighted_score(norm_dict: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = None
    for method, w in weights.items():
        x = w * norm_dict[method]
        out = x if out is None else (out + x)
    assert out is not None
    return out


def generate_weight_grid(methods: list[str], step: float):
    units = int(round(1.0 / step))
    if len(methods) == 2:
        for a in range(units + 1):
            b = units - a
            yield {
                methods[0]: a / units,
                methods[1]: b / units,
            }
    elif len(methods) == 3:
        for a in range(units + 1):
            for b in range(units + 1 - a):
                c = units - a - b
                yield {
                    methods[0]: a / units,
                    methods[1]: b / units,
                    methods[2]: c / units,
                }
    else:
        raise ValueError("Only 2 or 3 methods supported in grid search")


def fit_logreg(val_norm: dict[str, np.ndarray], test_norm: dict[str, np.ndarray], methods: list[str]):
    X_val = np.column_stack([val_norm[m] for m in methods])
    X_test = np.column_stack([test_norm[m] for m in methods])
    return X_val, X_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scores_dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports" / "benchmark" / "current"
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "reports" / "fusion" / "search",
    )
    parser.add_argument("--step", type=float, default=0.05)
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

    methods_all = ["sift", "dl_quick", "vit"]
    val_df = merge_methods(args.scores_dir, "val", methods_all)
    test_df = merge_methods(args.scores_dir, "test", methods_all)

    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    val_norm, test_norm = normalize_from_val(val_df, test_df, methods_all)

    rows = []

    # singles
    for method in methods_all:
        rows.append({
            "candidate": method,
            "type": "single",
            "weights_json": json.dumps({method: 1.0}),
            **{f"val_{k}": v for k, v in metrics_dict(y_val, val_df[method].values).items()},
            **{f"test_{k}": v for k, v in metrics_dict(y_test, test_df[method].values).items()},
        })

    # weighted grids
    candidates = [
        ["sift", "dl_quick"],
        ["sift", "vit"],
        ["dl_quick", "vit"],
        ["sift", "dl_quick", "vit"],
    ]

    for methods in candidates:
        best_auc = None
        best_auc_weights = None
        best_auc_val_scores = None

        best_eer = None
        best_eer_weights = None
        best_eer_val_scores = None

        for weights in generate_weight_grid(methods, args.step):
            fused_val = weighted_score(val_norm, weights)
            m = metrics_dict(y_val, fused_val)

            if best_auc is None or m["auc"] > best_auc:
                best_auc = m["auc"]
                best_auc_weights = weights
                best_auc_val_scores = fused_val

            if best_eer is None or m["eer"] < best_eer:
                best_eer = m["eer"]
                best_eer_weights = weights
                best_eer_val_scores = fused_val

        for objective, weights, val_scores in [
            ("opt_auc", best_auc_weights, best_auc_val_scores),
            ("opt_eer", best_eer_weights, best_eer_val_scores),
        ]:
            fused_test = weighted_score(test_norm, weights)
            rows.append({
                "candidate": "+".join(methods),
                "type": objective,
                "weights_json": json.dumps(weights, sort_keys=True),
                **{f"val_{k}": v for k, v in metrics_dict(y_val, val_scores).items()},
                **{f"test_{k}": v for k, v in metrics_dict(y_test, fused_test).items()},
            })

    # logistic regression fusion
    for methods in [["sift", "dl_quick"], ["sift", "dl_quick", "vit"]]:
        X_val, X_test = fit_logreg(val_norm, test_norm, methods)
        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X_val, y_val)

        val_scores = clf.predict_proba(X_val)[:, 1]
        test_scores = clf.predict_proba(X_test)[:, 1]

        rows.append({
            "candidate": "+".join(methods),
            "type": "logreg",
            "weights_json": json.dumps(
                {
                    "intercept": float(clf.intercept_[0]),
                    **{m: float(w) for m, w in zip(methods, clf.coef_[0])},
                },
                sort_keys=True,
            ),
            **{f"val_{k}": v for k, v in metrics_dict(y_val, val_scores).items()},
            **{f"test_{k}": v for k, v in metrics_dict(y_test, test_scores).items()},
        })

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(["test_auc", "test_tar_at_far_1e_3"], ascending=[False, False]).reset_index(drop=True)

    out_csv = args.outdir / "fusion_search_summary.csv"
    out_df.to_csv(out_csv, index=False)

    best_test_auc = out_df.iloc[0].to_dict()

    with open(args.outdir / "best_by_test_auc.json", "w", encoding="utf-8") as f:
        json.dump(best_test_auc, f, indent=2)

    print("\n=== TOP CANDIDATES BY TEST AUC ===")
    print(out_df[[
        "candidate",
        "type",
        "weights_json",
        "test_auc",
        "test_eer",
        "test_tar_at_far_1e_2",
        "test_tar_at_far_1e_3",
    ]].head(12).to_string(index=False))

    print(f"\nSaved summary to: {out_csv}")


if __name__ == "__main__":
    main()
