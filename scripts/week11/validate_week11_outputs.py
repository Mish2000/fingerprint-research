"""
Week 11 (Step 11): Validate that reports/week11 outputs are complete and sane.

Checks:
- results_summary.csv exists and has expected rows
- metrics are in valid ranges
- per-row artifact paths exist (scores_csv, meta_json when present)
- config_json is valid JSON and matches method/split
- expected ROC files exist in reports/week11/
Writes: reports/week11/validation.ok (and prints a short summary).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTDIR = REPO_ROOT / "reports" / "week11"
SUMMARY_CSV = OUTDIR / "results_summary.csv"
OK_FILE = OUTDIR / "validation.ok"


EXPECTED_METHODS = ["classic_v2", "dl_quick", "dedicated"]
EXPECTED_SPLITS = ["val", "test"]


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def exists_path_maybe(p: str) -> bool:
    # pandas may give NaN as float; we convert to string earlier
    if not p or p.lower() in ("nan", "none"):
        return False
    try:
        return Path(p).exists()
    except Exception:
        return False


def in_01(x: float) -> bool:
    return 0.0 <= float(x) <= 1.0


def main() -> int:
    if not SUMMARY_CSV.exists():
        fail(f"Missing {SUMMARY_CSV.relative_to(REPO_ROOT).as_posix()}")

    df = pd.read_csv(SUMMARY_CSV)

    if df.empty:
        fail("results_summary.csv exists but is empty")

    required_cols = ["method", "split", "n_pairs", "auc", "eer", "tar_at_far_1e_2", "tar_at_far_1e_3"]
    for c in required_cols:
        if c not in df.columns:
            fail(f"Missing required column '{c}' in results_summary.csv")

    # Expect exactly methods x splits rows (unless you intentionally add train later)
    expected_rows = len(EXPECTED_METHODS) * len(EXPECTED_SPLITS)
    if len(df) != expected_rows:
        fail(f"Expected {expected_rows} rows (methods×splits) but found {len(df)} rows")

    # Check method/split coverage
    got_methods = sorted(set(df["method"].astype(str).tolist()))
    got_splits = sorted(set(df["split"].astype(str).tolist()))

    for m in EXPECTED_METHODS:
        if m not in got_methods:
            fail(f"Missing method row(s) for: {m}")
    for s in EXPECTED_SPLITS:
        if s not in got_splits:
            fail(f"Missing split row(s) for: {s}")

    # Per-row checks
    problems: List[str] = []
    for i, r in df.iterrows():
        method = str(r["method"])
        split = str(r["split"])

        n_pairs = int(r["n_pairs"])
        if n_pairs <= 0:
            problems.append(f"row {i}: n_pairs <= 0")

        for k in ["auc", "eer", "tar_at_far_1e_2", "tar_at_far_1e_3"]:
            v = float(r[k])
            if not in_01(v):
                problems.append(f"row {i} ({method}/{split}): {k} not in [0,1] -> {v}")

        # Artifact paths (if present in CSV)
        if "scores_csv" in df.columns:
            scores_csv = str(r.get("scores_csv", ""))
            if scores_csv and scores_csv.lower() not in ("nan", "none") and not exists_path_maybe(scores_csv):
                problems.append(f"row {i} ({method}/{split}): scores_csv not found -> {scores_csv}")

        if "meta_json" in df.columns:
            meta_json = str(r.get("meta_json", ""))
            # meta_json may be empty for classic; only require if present
            if meta_json and meta_json.lower() not in ("nan", "none") and not exists_path_maybe(meta_json):
                problems.append(f"row {i} ({method}/{split}): meta_json not found -> {meta_json}")

        # Expected ROC file by naming convention from run_all
        roc = OUTDIR / f"roc_{method}_{split}.png"
        if not roc.exists():
            problems.append(f"row {i} ({method}/{split}): missing ROC -> {roc.relative_to(REPO_ROOT).as_posix()}")

        # config_json must parse, and match method/split
        if "config_json" in df.columns:
            cfg_raw = r.get("config_json", "")
            try:
                cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else dict(cfg_raw)
                if str(cfg.get("method")) != method:
                    problems.append(f"row {i} ({method}/{split}): config_json.method mismatch -> {cfg.get('method')}")
                if str(cfg.get("split")) != split:
                    problems.append(f"row {i} ({method}/{split}): config_json.split mismatch -> {cfg.get('split')}")
            except Exception as e:
                problems.append(f"row {i} ({method}/{split}): config_json invalid JSON ({e})")

    if problems:
        fail("Validation errors:\n- " + "\n- ".join(problems))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    OK_FILE.write_text(
        "OK\n"
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"summary_csv  : {SUMMARY_CSV}\n"
        f"rows         : {len(df)}\n",
        encoding="utf-8",
    )

    print("[OK] Week 11 outputs validated.")
    print(f"[OK] Wrote {OK_FILE.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())