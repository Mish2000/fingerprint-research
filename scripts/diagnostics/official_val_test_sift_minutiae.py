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
PAIR_SETS = ("positive_1000", "negative_1000")


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


def _load_method_scores(benchmark_dir: Path, method: str) -> pd.DataFrame:
    frames = []
    for pair_set in PAIR_SETS:
        path = benchmark_dir / f"scores_{method}_{pair_set}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing score CSV: {path}")
        df = pd.read_csv(path)
        missing = {"label", "split", "path_a", "path_b", "score"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")
        df = df.copy()
        df["pair_set"] = pair_set
        df["source_scores_csv"] = str(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _sift_variant_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    score = _safe_numeric(df["score"])
    inliers = _safe_numeric(df["inliers"])
    matches = _safe_numeric(df["matches"])
    denom_matches = np.maximum(matches, 1.0)
    inlier_ratio = np.divide(inliers, denom_matches, out=np.zeros_like(inliers), where=denom_matches > 0)
    return {
        "current_score": score,
        "inliers": inliers,
        "inliers_times_inlier_ratio": inliers * inlier_ratio,
        "inliers_times_inlier_ratio_times_log1p_matches": (
            inliers * inlier_ratio * np.log1p(np.maximum(matches, 0.0))
        ),
        "inliers_over_sqrt_matches": np.divide(
            inliers,
            np.sqrt(denom_matches),
            out=np.zeros_like(inliers),
            where=denom_matches > 0,
        ),
    }


def _threshold_for_far(negative_scores: np.ndarray, target_far: float) -> tuple[float, int, float]:
    scores = np.asarray(negative_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return float("nan"), 0, float("nan")
    n_negative = int(scores.size)
    for threshold in sorted(float(x) for x in np.unique(scores)):
        false_accepts = int(np.sum(scores >= threshold))
        actual_far = false_accepts / n_negative
        if actual_far <= float(target_far):
            return float(threshold), false_accepts, float(actual_far)
    threshold = math.nextafter(float(np.max(scores)), math.inf)
    return float(threshold), 0, 0.0


def _confusion(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels = labels[valid]
    scores = scores[valid]
    positives = labels == 1
    negatives = labels == 0
    accepted = scores >= float(threshold) if math.isfinite(float(threshold)) else np.zeros_like(scores, dtype=bool)
    ta = int(np.sum(accepted & positives))
    fr = int(np.sum((~accepted) & positives))
    fa = int(np.sum(accepted & negatives))
    tr = int(np.sum((~accepted) & negatives))
    n_positive = int(np.sum(positives))
    n_negative = int(np.sum(negatives))
    return {
        "tar": float(ta / n_positive) if n_positive else float("nan"),
        "far": float(fa / n_negative) if n_negative else float("nan"),
        "ta": ta,
        "fr": fr,
        "fa": fa,
        "tr": tr,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _build_method_rows(
    method: str,
    scores_df: pd.DataFrame,
    variant_scores: dict[str, np.ndarray],
    *,
    target_fars: tuple[float, ...],
) -> list[dict[str, Any]]:
    labels = pd.to_numeric(scores_df["label"], errors="coerce").fillna(0).to_numpy(dtype=int)
    splits = scores_df["split"].astype(str).str.strip().str.lower().to_numpy()
    val_mask = splits == "val"
    test_mask = splits == "test"
    rows: list[dict[str, Any]] = []
    for variant, values in variant_scores.items():
        values = np.asarray(values, dtype=float)
        val_labels = labels[val_mask]
        val_scores = values[val_mask]
        test_labels = labels[test_mask]
        test_scores = values[test_mask]
        val_negative_scores = val_scores[val_labels == 0]
        for target_far in target_fars:
            threshold, calibration_fa, calibration_far = _threshold_for_far(val_negative_scores, float(target_far))
            val = _confusion(val_labels, val_scores, threshold)
            test = _confusion(test_labels, test_scores, threshold)
            rows.append(
                {
                    "method": method,
                    "variant": variant,
                    "target_far": float(target_far),
                    "threshold": float(threshold),
                    "calibration_false_accepts": int(calibration_fa),
                    "calibration_far": float(calibration_far),
                    "val_far": float(val["far"]),
                    "val_tar": float(val["tar"]),
                    "val_ta": int(val["ta"]),
                    "val_fr": int(val["fr"]),
                    "val_fa": int(val["fa"]),
                    "val_tr": int(val["tr"]),
                    "test_far": float(test["far"]),
                    "test_tar": float(test["tar"]),
                    "test_ta": int(test["ta"]),
                    "test_fr": int(test["fr"]),
                    "test_fa": int(test["fa"]),
                    "test_tr": int(test["tr"]),
                    "n_val_positive": int(val["n_positive"]),
                    "n_val_negative": int(val["n_negative"]),
                    "n_test_positive": int(test["n_positive"]),
                    "n_test_negative": int(test["n_negative"]),
                }
            )
    return rows


def build_official_report(
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    *,
    target_fars: tuple[float, ...] = TARGET_FARS,
) -> pd.DataFrame:
    benchmark = parse_file_uri(benchmark_dir)
    sift = _load_method_scores(benchmark, "sift")
    minutiae = _load_method_scores(benchmark, "minutiae")
    rows: list[dict[str, Any]] = []
    rows.extend(_build_method_rows("sift", sift, _sift_variant_scores(sift), target_fars=target_fars))
    rows.extend(
        _build_method_rows(
            "minutiae",
            minutiae,
            {"current_score": _safe_numeric(minutiae["score"])},
            target_fars=target_fars,
        )
    )
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return "nan" if not math.isfinite(number) else f"{number:.{digits}f}"


def _render_markdown(report: pd.DataFrame, *, benchmark_dir: Path) -> str:
    lines = [
        "# Official Val/Test SIFT + Minutiae Evaluation",
        "",
        f"Source benchmark folder: `{benchmark_dir}`",
        "",
        "Thresholds are calibrated only on original `val` rows. Test metrics are evaluated only on original `test` rows.",
        "This report is the official val/test view for the selected professor 1000 positive/negative plain-vs-roll pairs.",
        "",
        "## Operating Points",
        "",
        "| method | variant | target FAR | threshold | val FAR | val TAR | test FAR | test TAR | test TA | test FR | test FA | test TR | n val pos | n val neg | n test pos | n test neg |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    ordered = report.sort_values(["method", "variant", "target_far"]).reset_index(drop=True)
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['method']} | {row['variant']} | {_fmt(row['target_far'])} | {_fmt(row['threshold'], 6)} | "
            f"{_fmt(row['val_far'])} | {_fmt(row['val_tar'])} | {_fmt(row['test_far'])} | "
            f"{_fmt(row['test_tar'])} | {int(row['test_ta'])} | {int(row['test_fr'])} | "
            f"{int(row['test_fa'])} | {int(row['test_tr'])} | {int(row['n_val_positive'])} | "
            f"{int(row['n_val_negative'])} | {int(row['n_test_positive'])} | {int(row['n_test_negative'])} |"
        )
    lines.extend(
        [
            "",
            "## Best Test TAR At FAR <= 1%",
            "",
            "| rank | method | variant | target FAR | threshold | val FAR | test FAR | test TAR | test TA | test FA |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    one_pct = report[np.isclose(report["target_far"].astype(float), 0.01)].copy()
    one_pct = one_pct[one_pct["test_far"].astype(float) <= 0.01 + 1e-12]
    one_pct = one_pct.sort_values(["test_tar", "test_far", "val_tar"], ascending=[False, True, False])
    for rank, (_, row) in enumerate(one_pct.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['method']} | {row['variant']} | {_fmt(row['target_far'])} | "
            f"{_fmt(row['threshold'], 6)} | {_fmt(row['val_far'])} | {_fmt(row['test_far'])} | "
            f"{_fmt(row['test_tar'])} | {int(row['test_ta'])} | {int(row['test_fa'])} |"
        )
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
    report = build_official_report(benchmark, target_fars=tuple(float(x) for x in target_fars))
    csv_path = output / "official_val_test_sift_minutiae.csv"
    md_path = output / "official_val_test_sift_minutiae.md"
    report.to_csv(csv_path, index=False)
    md_path.write_text(_render_markdown(report, benchmark_dir=benchmark), encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Official val-calibrated/test-evaluated SIFT and minutiae report.")
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
    print("Wrote official val/test SIFT + minutiae report:")
    for path in paths.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
