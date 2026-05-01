from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set, Tuple

import pandas as pd

from pipelines.benchmark.benchmark_validation_utils import (
    validate_run_meta,
    validate_scores_csv,
    validate_summary_columns,
)


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


ROOT = project_root()


def resolve_path(s: str) -> Path:
    if s.startswith("file:"):
        s = s[len("file:"):]
        if len(s) >= 3 and s[0] == "/" and s[2] == ":":
            s = s[1:]
    p = Path(s).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def exists_path_maybe(value: str) -> bool:
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none"}:
        return False
    try:
        return resolve_path(s).exists()
    except Exception:
        return False


def in_01(x) -> bool:
    try:
        v = float(x)
    except Exception:
        return False
    return 0.0 <= v <= 1.0


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a benchmark artifact bundle.")
    ap.add_argument("--outdir", required=True, type=str)
    ap.add_argument("--expected_methods", default="classic_v2,harris,sift,dl_quick,dedicated,vit", type=str)
    ap.add_argument("--expected_splits", default="val,test", type=str)
    args = ap.parse_args()

    outdir = resolve_path(args.outdir)
    summary_csv = outdir / "results_summary.csv"
    ok_file = outdir / "validation.ok"

    expected_methods = [m.strip() for m in args.expected_methods.split(",") if m.strip()]
    expected_splits = [s.strip() for s in args.expected_splits.split(",") if s.strip()]
    expected_combos: Set[Tuple[str, str]] = {(m, s) for m in expected_methods for s in expected_splits}

    if not summary_csv.exists():
        fail(f"Missing summary CSV: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        fail("results_summary.csv exists but is empty")

    try:
        validate_summary_columns(df)
    except Exception as exc:
        fail(str(exc))

    problems: List[str] = []

    actual_combos: Set[Tuple[str, str]] = set(zip(df["method"].astype(str), df["split"].astype(str)))
    missing_combos = sorted(expected_combos - actual_combos)
    extra_combos = sorted(actual_combos - expected_combos)

    if missing_combos:
        problems.append(f"missing method/split combos: {missing_combos}")
    if extra_combos:
        problems.append(f"unexpected method/split combos: {extra_combos}")

    dups = df.duplicated(subset=["method", "split"], keep=False)
    if bool(dups.any()):
        dup_rows = df.loc[dups, ["method", "split"]].astype(str).values.tolist()
        problems.append(f"duplicate rows for method/split: {dup_rows}")

    for i, row in df.iterrows():
        method = str(row["method"])
        split = str(row["split"])

        if method not in expected_methods:
            problems.append(f"row {i}: unexpected method -> {method}")
        if split not in expected_splits:
            problems.append(f"row {i}: unexpected split -> {split}")

        try:
            n_pairs = int(row["n_pairs"])
            if n_pairs <= 0:
                problems.append(f"row {i} ({method}/{split}): n_pairs must be > 0")
        except Exception:
            problems.append(f"row {i} ({method}/{split}): invalid n_pairs -> {row['n_pairs']}")

        for col in ["auc", "eer", "tar_at_far_1e_2", "tar_at_far_1e_3"]:
            if not in_01(row[col]):
                problems.append(f"row {i} ({method}/{split}): {col} out of range -> {row[col]}")

        for col in ["avg_ms_pair_reported", "avg_ms_pair_wall"]:
            if col in df.columns and str(row.get(col, "")).lower() not in {"", "nan", "none"}:
                try:
                    if float(row[col]) < 0:
                        problems.append(f"row {i} ({method}/{split}): {col} must be >= 0")
                except Exception:
                    problems.append(f"row {i} ({method}/{split}): invalid {col} -> {row[col]}")

        scores_csv = str(row.get("scores_csv", ""))
        if not exists_path_maybe(scores_csv):
            problems.append(f"row {i} ({method}/{split}): scores_csv not found -> {scores_csv}")
        else:
            try:
                validate_scores_csv(scores_csv, expected_n_pairs=n_pairs, context=f"row {i} ({method}/{split}) scores_csv")
            except Exception as exc:
                problems.append(str(exc))

        if "meta_json" in df.columns:
            meta_json = str(row.get("meta_json", ""))
            if meta_json.lower() not in {"", "nan", "none"} and not exists_path_maybe(meta_json):
                problems.append(f"row {i} ({method}/{split}): meta_json not found -> {meta_json}")

        roc_path = outdir / f"roc_{method}_{split}.png"
        if not roc_path.exists():
            problems.append(f"row {i} ({method}/{split}): missing ROC -> {roc_path}")

        run_meta = outdir / f"run_{method}_{split}.meta.json"
        if not run_meta.exists():
            problems.append(f"row {i} ({method}/{split}): missing run meta -> {run_meta}")
        else:
            try:
                validate_run_meta(
                    run_meta,
                    expected_row=row.to_dict(),
                    expected_scores_csv=scores_csv,
                    expected_summary_csv=summary_csv,
                    expected_method=method,
                    expected_split=split,
                )
            except Exception as exc:
                problems.append(f"row {i} ({method}/{split}): {exc}")

        if "config_json" in df.columns:
            cfg_raw = row.get("config_json", "")
            if str(cfg_raw).lower() not in {"", "nan", "none"}:
                try:
                    cfg = json.loads(str(cfg_raw))
                    if str(cfg.get("method", "")) != method:
                        problems.append(
                            f"row {i} ({method}/{split}): config_json.method mismatch -> {cfg.get('method')}"
                        )
                    if str(cfg.get("split", "")) != split:
                        problems.append(
                            f"row {i} ({method}/{split}): config_json.split mismatch -> {cfg.get('split')}"
                        )
                except Exception as e:
                    problems.append(f"row {i} ({method}/{split}): invalid config_json ({e})")

    run_manifest = outdir / "run_manifest.json"
    if not run_manifest.exists():
        problems.append(f"missing run_manifest.json -> {run_manifest}")

    if problems:
        fail("Validation errors:\n- " + "\n- ".join(problems))

    ok_file.write_text(
        "OK\n"
        f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n"
        f"summary_csv  : {summary_csv}\n"
        f"rows         : {len(df)}\n",
        encoding="utf-8",
    )

    print("[OK] Benchmark bundle validated.")
    print(f"[OK] Wrote {ok_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())