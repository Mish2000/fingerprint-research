from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.modules.setdefault("numexpr", None)
sys.modules.setdefault("bottleneck", None)
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_BENCHMARK_DIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_professor_1000_pos_neg"
)
DEFAULT_OUTDIR = (
    REPO_ROOT
    / "artifacts"
    / "reports"
    / "benchmark"
    / "nist_sd300b_plain_roll_diagnostics"
)
TARGET_FARS = (0.005, 0.01, 0.02, 0.05, 0.10)


def parse_file_uri(raw: str | Path) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _safe_numeric(series: pd.Series, default: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(default).to_numpy(dtype=float)


def _variant_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    score = _safe_numeric(df["score"])
    inliers = _safe_numeric(df["inliers"])
    matches = _safe_numeric(df["matches"])
    denom_matches = np.maximum(matches, 1.0)
    inlier_ratio = np.divide(inliers, denom_matches, out=np.zeros_like(inliers), where=denom_matches > 0)
    log_matches = np.log1p(np.maximum(matches, 0.0))
    return {
        "current_score": score,
        "inliers": inliers,
        "inliers_over_matches": inlier_ratio,
        "inliers_times_inlier_ratio": inliers * inlier_ratio,
        "inlier_ratio_times_log1p_matches": inlier_ratio * log_matches,
        "inliers_times_inlier_ratio_times_log1p_matches": inliers * inlier_ratio * log_matches,
        "inliers_over_sqrt_matches": np.divide(
            inliers,
            np.sqrt(denom_matches),
            out=np.zeros_like(inliers),
            where=denom_matches > 0,
        ),
    }


def _auc_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan"), float("nan")
    try:
        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores)
    except ValueError:
        return float("nan"), float("nan")
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fpr - fnr))) if fpr.size else 0
    return auc, float((fpr[idx] + fnr[idx]) / 2.0)


def _threshold_for_far(negative_scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    scores = np.asarray(negative_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), 0, float("nan")
    n_negative = int(scores.size)
    for threshold in sorted(float(x) for x in np.unique(scores)):
        false_accepts = int(np.sum(scores >= threshold))
        far = false_accepts / n_negative
        if far <= float(target_far):
            return float(threshold), false_accepts, float(far)
    threshold = math.nextafter(float(np.max(scores)), math.inf)
    return float(threshold), 0, 0.0


def _load_sift_scores(benchmark_dir: Path) -> pd.DataFrame:
    frames = []
    for pair_set in ("positive_1000", "negative_1000"):
        path = benchmark_dir / f"scores_sift_{pair_set}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing SIFT scores: {path}")
        df = pd.read_csv(path)
        df["pair_set"] = pair_set
        df["source_scores_csv"] = str(path)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    missing = {"label", "score", "inliers", "matches", "k1", "k2"} - set(out.columns)
    if missing:
        raise ValueError(f"SIFT score CSVs missing required columns: {sorted(missing)}")
    return out


def build_sweep(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    benchmark = parse_file_uri(benchmark_dir)
    scores_df = _load_sift_scores(benchmark)
    labels = _safe_numeric(scores_df["label"]).astype(int)
    rows: list[dict[str, Any]] = []
    for variant_name, values in _variant_scores(scores_df).items():
        values = np.asarray(values, dtype=float)
        auc, eer = _auc_eer(labels, values)
        pos_scores = values[labels == 1]
        neg_scores = values[labels == 0]
        n_pos = int(np.isfinite(pos_scores).sum())
        n_neg = int(np.isfinite(neg_scores).sum())
        for target_far in target_fars:
            threshold, false_accepts, actual_far = _threshold_for_far(neg_scores, float(target_far))
            true_accepts = int(np.sum(pos_scores[np.isfinite(pos_scores)] >= threshold)) if math.isfinite(threshold) else 0
            tar = float(true_accepts / n_pos) if n_pos else float("nan")
            rows.append(
                {
                    "variant": variant_name,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "actual_far": float(actual_far),
                    "tar": float(tar),
                    "true_accepts": int(true_accepts),
                    "false_accepts": int(false_accepts),
                    "n_positive": int(n_pos),
                    "n_negative": int(n_neg),
                    "auc": float(auc),
                    "eer": float(eer),
                    "exploratory_same_selected_set": True,
                }
            )
    return pd.DataFrame(rows)


def _render_markdown(sweep: pd.DataFrame, benchmark_dir: Path) -> str:
    lines = [
        "# Exploratory SIFT Score Variant Sweep",
        "",
        f"Source benchmark folder: `{benchmark_dir}`",
        "",
        "This is exploratory: thresholds and tradeoff metrics are estimated on the same selected "
        "`positive_1000` + `negative_1000` set, not calibrated on validation and evaluated separately.",
        "",
        "## Ranking At FAR Targets",
        "",
        "| target FAR | rank | variant | TAR | actual FAR | threshold | TA | FA | AUC | EER |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for target_far, group in sweep.groupby("target_far", sort=True):
        ranked = group.sort_values(["tar", "actual_far", "auc"], ascending=[False, True, False]).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            lines.append(
                f"| {float(target_far):.3f} | {rank} | {row['variant']} | {row['tar']:.3f} | "
                f"{row['actual_far']:.3f} | {row['threshold']:.6g} | {int(row['true_accepts'])} | "
                f"{int(row['false_accepts'])} | {row['auc']:.4f} | {row['eer']:.4f} |"
            )
    lines.extend(["", "## Best By AUC", "", "| rank | variant | AUC | EER |", "| ---: | --- | ---: | ---: |"])
    auc_rows = (
        sweep[["variant", "auc", "eer"]]
        .drop_duplicates()
        .sort_values(["auc", "eer"], ascending=[False, True])
        .reset_index(drop=True)
    )
    for rank, (_, row) in enumerate(auc_rows.iterrows(), start=1):
        lines.append(f"| {rank} | {row['variant']} | {row['auc']:.4f} | {row['eer']:.4f} |")
    return "\n".join(lines) + "\n"


def write_outputs(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    outdir: str | Path = DEFAULT_OUTDIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> dict[str, Path]:
    benchmark = parse_file_uri(benchmark_dir)
    output = parse_file_uri(outdir)
    output.mkdir(parents=True, exist_ok=True)
    sweep = build_sweep(benchmark, target_fars=tuple(float(x) for x in target_fars))
    csv_path = output / "sift_score_variant_sweep.csv"
    md_path = output / "sift_score_variant_sweep.md"
    sweep.to_csv(csv_path, index=False)
    md_path.write_text(_render_markdown(sweep, benchmark), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Exploratory score-variant sweep for existing SIFT score CSVs.")
    parser.add_argument("--benchmark_dir", default=str(DEFAULT_BENCHMARK_DIR))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--target_far", type=float, nargs="*", default=list(TARGET_FARS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = write_outputs(
        args.benchmark_dir,
        args.outdir,
        target_fars=tuple(float(x) for x in args.target_far),
    )
    print("Wrote exploratory SIFT variant sweep:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
