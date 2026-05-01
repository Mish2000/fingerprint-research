from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from pipelines.benchmark.benchmark_validation_utils import (
    BENCHMARK_CONFIG_SCHEMA_VERSION,
    BENCHMARK_RUN_META_SCHEMA_VERSION,
    validate_run_meta,
    validate_scores_csv,
    validate_summary_columns,
)


METHOD_NAME = "fusion_balanced_v1"
WEIGHTS = {
    "sift": 0.91,
    "dl_quick": 0.05,
    "vit": 0.04,
}


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


def make_roc_png(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.xscale("log")
    plt.xlim(1e-4, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel("FAR / FPR")
    plt.ylabel("TAR / TPR")
    plt.title(title)
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def resolve_pairs_path(data_dir: Path, split: str) -> Path:
    candidates = [
        data_dir / f"pairs_{split}.csv",
        data_dir / "pairs" / f"pairs_{split}.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def build_config(
    *,
    split: str,
    resolved_data_dir: str,
    dataset_name: str,
    scores_dir: Path,
) -> dict:
    data_dir = Path(resolved_data_dir)
    pairs_path = resolve_pairs_path(data_dir, split)

    return {
        "schema_version": BENCHMARK_CONFIG_SCHEMA_VERSION,
        "method": METHOD_NAME,
        "split": split,
        "dataset": dataset_name,
        "resolved_data_dir": str(data_dir),
        "manifest_path": str((data_dir / "manifest.csv").resolve()),
        "pairs_path": str(pairs_path.resolve()),
        "fusion": {
            "weights": WEIGHTS,
            "normalization": "minmax_fit_on_val_only",
            "source_methods": ["sift", "dl_quick", "vit"],
            "source_scores_dir": str(scores_dir.resolve()),
        },
    }


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
        default=REPO_ROOT / "artifacts" / "reports" / "fusion" / "final_balanced_v1",
    )
    args = parser.parse_args()

    scores_dir = args.scores_dir
    if not scores_dir.is_absolute():
        scores_dir = (REPO_ROOT / scores_dir).resolve()
    else:
        scores_dir = scores_dir.resolve()

    outdir = args.outdir
    if not outdir.is_absolute():
        outdir = (REPO_ROOT / outdir).resolve()
    else:
        outdir = outdir.resolve()

    outdir.mkdir(parents=True, exist_ok=True)

    methods = ["sift", "dl_quick", "vit"]

    val_df = merge_methods(scores_dir, "val", methods)
    test_df = merge_methods(scores_dir, "test", methods)

    y_val = val_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    val_norm, test_norm = normalize_from_val(val_df, test_df, methods)
    fused_val = fused_score(val_norm, WEIGHTS)
    fused_test = fused_score(test_norm, WEIGHTS)

    val_metrics = metrics_dict(y_val, fused_val)
    test_metrics = metrics_dict(y_test, fused_test)

    # Save scores
    scores_val_path = outdir / f"scores_{METHOD_NAME}_val.csv"
    scores_test_path = outdir / f"scores_{METHOD_NAME}_test.csv"

    pd.DataFrame({"label": y_val, "score": fused_val}).to_csv(scores_val_path, index=False)
    pd.DataFrame({"label": y_test, "score": fused_test}).to_csv(scores_test_path, index=False)

    # ROC
    roc_val_path = outdir / f"roc_{METHOD_NAME}_val.png"
    roc_test_path = outdir / f"roc_{METHOD_NAME}_test.png"
    make_roc_png(y_val, fused_val, roc_val_path, f"{METHOD_NAME} ROC (val)")
    make_roc_png(y_test, fused_test, roc_test_path, f"{METHOD_NAME} ROC (test)")

    # Load reference manifest for resolved_data_dir
    run_manifest_path = scores_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    resolved_data_dir = run_manifest["dataset"]["resolved_data_dir"]
    dataset_name = run_manifest["dataset"].get("name", "unknown")

    # Optional latency story: serial execution of the three source methods, taken from week11 results_summary.csv
    reference_summary = pd.read_csv(scores_dir / "results_summary.csv")

    def wall_latency(split: str) -> float:
        sub = reference_summary[
            (reference_summary["split"] == split)
            & (reference_summary["method"].isin(["sift", "dl_quick", "vit"]))
        ].copy()
        return float(sub["avg_ms_pair_wall"].fillna(0.0).sum())

    rows = []
    for split, metrics, scores_path, roc_path, y_true in [
        ("val", val_metrics, scores_val_path, roc_val_path, y_val),
        ("test", test_metrics, scores_test_path, roc_test_path, y_test),
    ]:
        config = build_config(
            split=split,
            resolved_data_dir=resolved_data_dir,
            dataset_name=dataset_name,
            scores_dir=scores_dir,
        )
        config_json = json.dumps(config, sort_keys=True)

        row = {
            "method": METHOD_NAME,
            "split": split,
            "n_pairs": int(len(y_true)),
            "auc": metrics["auc"],
            "eer": metrics["eer"],
            "tar_at_far_1e_2": metrics["tar_at_far_1e_2"],
            "tar_at_far_1e_3": metrics["tar_at_far_1e_3"],
            "scores_csv": str(scores_path),
            "config_json": config_json,
            "avg_ms_pair_reported": "",
            "avg_ms_pair_wall": wall_latency(split),
        }
        rows.append((row, config, roc_path))

    summary_df = pd.DataFrame([r[0] for r in rows])
    summary_csv_path = outdir / "results_summary.csv"
    summary_md_path = outdir / "results_summary.md"
    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write(f"# {METHOD_NAME} Results Summary\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n")

    # Per-split run meta
    for row, config, roc_path in rows:
        meta = {
            "schema_version": BENCHMARK_RUN_META_SCHEMA_VERSION,
            "row": row,
            "scores_csv": row["scores_csv"],
            "roc_png": str(roc_path),
            "summary_csv": str(summary_csv_path),
            "resolved_data_dir": config["resolved_data_dir"],
            "manifest_path": config["manifest_path"],
            "pairs_path": config["pairs_path"],
            "config": config,
        }
        meta_path = outdir / f"run_{METHOD_NAME}_{row['split']}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    # Self-validation
    loaded_summary = pd.read_csv(summary_csv_path)
    validate_summary_columns(loaded_summary, context=str(summary_csv_path))

    for _, row in loaded_summary.iterrows():
        validate_scores_csv(
            row["scores_csv"],
            expected_n_pairs=int(row["n_pairs"]),
            context=f"{row['method']} {row['split']} scores",
        )

        run_meta_path = outdir / f"run_{row['method']}_{row['split']}.meta.json"
        expected_row = {
            "method": row["method"],
            "split": row["split"],
            "n_pairs": row["n_pairs"],
            "auc": row["auc"],
            "eer": row["eer"],
            "tar_at_far_1e_2": row["tar_at_far_1e_2"],
            "tar_at_far_1e_3": row["tar_at_far_1e_3"],
            "scores_csv": row["scores_csv"],
            "config_json": row["config_json"],
        }
        validate_run_meta(
            run_meta_path,
            expected_row=expected_row,
            expected_scores_csv=row["scores_csv"],
            expected_summary_csv=str(summary_csv_path),
            expected_method=row["method"],
            expected_split=row["split"],
        )

    with open(outdir / "package_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": METHOD_NAME,
                "weights": WEIGHTS,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "summary_csv": str(summary_csv_path),
                "summary_md": str(summary_md_path),
                "scores_val_csv": str(scores_val_path),
                "scores_test_csv": str(scores_test_path),
                "roc_val_png": str(roc_val_path),
                "roc_test_png": str(roc_test_path),
                "status": "validated",
            },
            f,
            indent=2,
        )

    print("\n=== PACKAGED FUSION CANDIDATE ===")
    print(json.dumps(
        {
            "method": METHOD_NAME,
            "weights": WEIGHTS,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "summary_csv": str(summary_csv_path),
            "status": "validated",
        },
        indent=2
    ))
    print(f"\nArtifacts written to: {outdir}")


if __name__ == "__main__":
    main()
