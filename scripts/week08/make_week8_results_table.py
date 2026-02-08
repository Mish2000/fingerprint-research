from __future__ import annotations

from pathlib import Path
import pandas as pd


METHODS = ["classic_v2", "dl_quick", "dedicated"]
SPLITS = ["val", "test"]

OUT_DIR = Path("reports/week08")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = Path("reports/week05/results_summary.csv")
OUT_MD = OUT_DIR / "week8_results_v0.md"
OUT_CSV = OUT_DIR / "week8_results_v0.csv"


def _latest_rows(df: pd.DataFrame) -> pd.DataFrame:
    # Make sure timestamps sort correctly
    if "timestamp_utc" in df.columns:
        df = df.copy()
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce")
        df = df.sort_values("timestamp_utc")
    else:
        # fallback: keep file order
        df = df.reset_index().sort_values("index")

    sub = df[df["method"].isin(METHODS) & df["split"].isin(SPLITS)].copy()
    latest = sub.groupby(["method", "split"], as_index=False).tail(1)
    return latest


def _format_table(latest: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "method",
        "split",
        "n_pairs",
        "auc",
        "eer",
        "tar_at_far_1e_2",
        "tar_at_far_1e_3",
        "avg_ms_pair_reported",
        "avg_ms_pair_wall",
    ]
    for c in cols:
        if c not in latest.columns:
            latest[c] = None

    out = latest[cols].copy()

    # Nice ordering
    out["method"] = pd.Categorical(out["method"], METHODS, ordered=True)
    out["split"] = pd.Categorical(out["split"], SPLITS, ordered=True)
    out = out.sort_values(["split", "method"]).reset_index(drop=True)

    # Round numeric columns for readability
    for c in ["auc", "eer", "tar_at_far_1e_2", "tar_at_far_1e_3"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(4)

    for c in ["avg_ms_pair_reported", "avg_ms_pair_wall"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    return out


def _to_markdown_sections(tbl: pd.DataFrame) -> str:
    lines = []
    lines.append("# Week 8 — Results Table v0\n")
    lines.append("Auto-generated from `reports/week05/results_summary.csv`.\n")

    for split in SPLITS:
        t = tbl[tbl["split"] == split].copy()
        if t.empty:
            continue
        lines.append(f"## {split.upper()} (full split)\n")
        lines.append(t.to_markdown(index=False))
        lines.append("\n")

    return "\n".join(lines)


def main():
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing: {RESULTS_CSV}")

    df = pd.read_csv(RESULTS_CSV)
    required = {"method", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"results_summary.csv missing required columns: {missing}")

    latest = _latest_rows(df)
    tbl = _format_table(latest)

    # Save artifacts
    tbl.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(_to_markdown_sections(tbl), encoding="utf-8")

    print("Saved:")
    print(f"  CSV: {OUT_CSV.resolve()}")
    print(f"  MD : {OUT_MD.resolve()}")
    print("\nPreview:\n")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
