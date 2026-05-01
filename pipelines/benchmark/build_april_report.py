from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = ROOT / "artifacts" / "reports" / "benchmark"
FALLBACK_SUFFIXES = ("_h6", "_harris")

DATASET_ORDER = {
    "nist_sd300b": 0,
    "nist_sd300c": 1,
    "polyu_cross": 2,
    "polyu_3d": 3,
    "unsw_2d3d": 4,
    "l3_sf_v2": 5,
}

METHOD_ORDER = {
    "classic_v2": 0,
    "harris": 1,
    "sift": 2,
    "dl_quick": 3,
    "dedicated": 4,
    "vit": 5,
}

SPLIT_ORDER = {"val": 0, "test": 1, "train": 2}


def infer_dataset(run_dir: Path) -> str:
    manifest = run_dir / "run_manifest.json"
    if manifest.exists():
        try:
            return str(json.loads(manifest.read_text(encoding="utf-8")).get("dataset", run_dir.name))
        except Exception:
            pass

    name = run_dir.name.lower()
    for ds in DATASET_ORDER:
        if ds in name:
            return ds
    return run_dir.name


def read_summary(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    if df.empty:
        raise RuntimeError(f"Empty summary CSV: {summary_csv}")
    return df


def merge_missing_rows(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    seen = set(zip(base["method"].astype(str).str.lower(), base["split"].astype(str).str.lower()))
    mask = ~extra.apply(
        lambda r: (str(r["method"]).lower(), str(r["split"]).lower()) in seen,
        axis=1,
    )
    extra_only = extra[mask].copy()
    if extra_only.empty:
        return base
    return pd.concat([base, extra_only], ignore_index=True)


def load_run_summary(run: str, strict: bool = False) -> Optional[pd.DataFrame]:
    run_dir = BENCH_ROOT / run
    summary_csv = run_dir / "results_summary.csv"

    if not summary_csv.exists():
        if strict:
            raise FileNotFoundError(f"Missing run summary: {summary_csv}")
        print(f"[WARN] Missing run summary: {summary_csv}")
        return None

    df = read_summary(summary_csv)

    for suffix in FALLBACK_SUFFIXES:
        alt_csv = run_dir.parent / f"{run_dir.name}{suffix}" / "results_summary.csv"
        if alt_csv.exists():
            df = merge_missing_rows(df, read_summary(alt_csv))

    df = df.copy()
    df["run"] = run
    df["dataset"] = infer_dataset(run_dir)

    if "avg_ms_pair_reported" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["avg_ms_pair_reported"], errors="coerce")
    else:
        df["latency_ms"] = pd.Series([float("nan")] * len(df))

    if "avg_ms_pair_wall" in df.columns:
        df["latency_ms"] = df["latency_ms"].fillna(
            pd.to_numeric(df["avg_ms_pair_wall"], errors="coerce")
        )

    for col in ("auc", "eer", "tar_at_far_1e_2", "tar_at_far_1e_3", "latency_ms"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["auc_rank"] = (
        out.groupby(["dataset", "split"])["auc"]
        .rank(ascending=False, method="dense")
        .astype("Int64")
    )
    out["eer_rank"] = (
        out.groupby(["dataset", "split"])["eer"]
        .rank(ascending=True, method="dense")
        .astype("Int64")
    )
    out["latency_rank"] = (
        out.groupby(["dataset", "split"])["latency_ms"]
        .rank(ascending=True, method="dense", na_option="bottom")
        .astype("Int64")
    )

    out["_dataset_order"] = out["dataset"].map(DATASET_ORDER).fillna(99).astype(int)
    out["_split_order"] = out["split"].map(SPLIT_ORDER).fillna(99).astype(int)
    out["_method_order"] = out["method"].map(METHOD_ORDER).fillna(99).astype(int)

    out = out.sort_values(
        ["_dataset_order", "_split_order", "_method_order", "run"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)

    return out.drop(columns=["_dataset_order", "_split_order", "_method_order"])


def build_best_methods(df: pd.DataFrame) -> Dict[str, dict]:
    out: Dict[str, dict] = {}

    for (dataset, split), g in df.groupby(["dataset", "split"], sort=False):
        item: Dict[str, dict] = {}

        g_auc = g.dropna(subset=["auc"])
        if not g_auc.empty:
            r = g_auc.sort_values(["auc", "method"], ascending=[False, True]).iloc[0]
            item["best_auc"] = {
                "method": str(r["method"]),
                "auc": float(r["auc"]),
                "run": str(r["run"]),
            }

        g_eer = g.dropna(subset=["eer"])
        if not g_eer.empty:
            r = g_eer.sort_values(["eer", "method"], ascending=[True, True]).iloc[0]
            item["best_eer"] = {
                "method": str(r["method"]),
                "eer": float(r["eer"]),
                "run": str(r["run"]),
            }

        g_lat = g.dropna(subset=["latency_ms"])
        if not g_lat.empty:
            r = g_lat.sort_values(["latency_ms", "method"], ascending=[True, True]).iloc[0]
            item["best_latency"] = {
                "method": str(r["method"]),
                "latency_ms": float(r["latency_ms"]),
                "run": str(r["run"]),
            }

        out[f"{dataset}:{split}"] = item

    return out


def fmt(v: object, digits: int = 4) -> str:
    if pd.isna(v):
        return "-"
    return f"{float(v):.{digits}f}"


def render_markdown(df: pd.DataFrame) -> str:
    lines: List[str] = ["# April Benchmark Comparison", ""]

    for (dataset, split), g in df.groupby(["dataset", "split"], sort=False):
        lines.append(f"## {dataset} | {split}")
        lines.append("")
        lines.append(
            "| method | auc | eer | tar@1e-2 | tar@1e-3 | latency_ms | auc_rank | eer_rank | latency_rank | run |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
        )

        for _, r in g.iterrows():
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(r["method"]),
                        fmt(r.get("auc")),
                        fmt(r.get("eer")),
                        fmt(r.get("tar_at_far_1e_2")),
                        fmt(r.get("tar_at_far_1e_3")),
                        fmt(r.get("latency_ms"), 2),
                        str(r.get("auc_rank", "-")),
                        str(r.get("eer_rank", "-")),
                        str(r.get("latency_rank", "-")),
                        str(r.get("run", "")),
                    ]
                )
                + " |"
            )

        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the April cross-run benchmark comparison bundle.")
    ap.add_argument(
        "--runs",
        type=str,
        default="full_nist_sd300b,full_nist_sd300c_h6,full_polyu_cross_h5",
        help="Comma-separated benchmark run names under artifacts/reports/benchmark/",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/reports/benchmark/april_comparison",
    )
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    runs = [x.strip() for x in args.runs.split(",") if x.strip()]
    if not runs:
        raise ValueError("No runs provided.")

    frames: List[pd.DataFrame] = []
    for run in runs:
        df = load_run_summary(run, strict=args.strict)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError("No benchmark runs were loaded.")

    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = add_ranks(df)

    csv_path = out_dir / "benchmark_comparison.csv"
    md_path = out_dir / "benchmark_comparison.md"
    best_json = out_dir / "best_methods.json"

    df.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(df), encoding="utf-8")
    best_json.write_text(
        json.dumps(build_best_methods(df), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Wrote:", csv_path)
    print("Wrote:", md_path)
    print("Wrote:", best_json)


if __name__ == "__main__":
    main()