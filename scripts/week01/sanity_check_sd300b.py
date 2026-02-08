from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd


def _pick_file(processed_dir: Path, keywords: list[str]) -> Path:
    files = list(processed_dir.glob("*.csv")) + list(processed_dir.glob("*.json"))
    # prefer exact keyword hits
    scored = []
    for f in files:
        name = f.name.lower()
        score = sum(1 for k in keywords if k in name)
        if score > 0:
            scored.append((score, f))
    if not scored:
        raise FileNotFoundError(
            f"Could not find file in {processed_dir} with keywords={keywords}. "
            f"Found: {[x.name for x in files]}"
        )
    scored.sort(key=lambda x: (-x[0], x[1].name))
    return scored[0][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed_dir",
        type=str,
        default=r"C:\fingerprint-research\data\processed\nist_sd300b",
        help="Folder containing manifest/pairs produced by prepare_data_sd300b.py",
    )
    args = ap.parse_args()

    processed_dir = Path(args.processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"processed_dir does not exist: {processed_dir}")

    manifest_path = _pick_file(processed_dir, ["manifest"])
    pos_path = _pick_file(processed_dir, ["pos", "pair"])
    neg_path = _pick_file(processed_dir, ["neg", "pair"])
    stats_path = _pick_file(processed_dir, ["stats"])

    df = pd.read_csv(manifest_path)
    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)

    report = {
        "paths": {
            "manifest": str(manifest_path),
            "pairs_pos": str(pos_path),
            "pairs_neg": str(neg_path),
            "stats": str(stats_path),
        },
        "manifest": {},
        "pairs": {},
        "checks": {},
        "problems": [],
    }

    # --- Manifest checks ---
    required_cols = {"dataset", "capture", "subject_id", "frgp", "ppi", "path"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        report["problems"].append(f"manifest missing columns: {missing}")

    report["manifest"]["rows"] = int(len(df))
    report["manifest"]["unique_subjects"] = int(df["subject_id"].nunique()) if "subject_id" in df.columns else None
    report["manifest"]["frgp_values"] = sorted(df["frgp"].unique().tolist()) if "frgp" in df.columns else None
    report["manifest"]["ppi_values"] = sorted(df["ppi"].unique().tolist()) if "ppi" in df.columns else None
    report["manifest"]["capture_counts"] = df["capture"].value_counts().to_dict() if "capture" in df.columns else None

    # Split / leakage sanity (if split exists)
    if "split" in df.columns:
        split_vals = sorted(df["split"].dropna().unique().tolist())
        report["manifest"]["split_values"] = split_vals

        # Identity leakage: a subject appearing in >1 split
        subj_splits = df.groupby("subject_id")["split"].nunique()
        leaked = subj_splits[subj_splits > 1]
        report["checks"]["identity_leakage_subjects"] = int(len(leaked))
        if len(leaked) > 0:
            report["problems"].append(f"IDENTITY LEAKAGE: {len(leaked)} subjects appear in multiple splits.")
    else:
        report["manifest"]["split_values"] = None

    # File existence (fast but still meaningful)
    # Check first 200 paths (you can increase later)
    sample_paths = df["path"].head(200).tolist() if "path" in df.columns else []
    missing_files = [p for p in sample_paths if not Path(p).exists()]
    report["checks"]["missing_files_in_first_200"] = int(len(missing_files))
    if missing_files:
        report["problems"].append(f"Some files do not exist (first 200 sample). Example: {missing_files[:3]}")

    # --- Pair checks (robust to column naming) ---
    def pair_basic_stats(pairs_df: pd.DataFrame, name: str):
        rep = {"rows": int(len(pairs_df)), "columns": list(pairs_df.columns)}
        if "split" in pairs_df.columns:
            rep["by_split"] = pairs_df["split"].value_counts().to_dict()
        if "label" in pairs_df.columns:
            rep["label_values"] = sorted(pairs_df["label"].unique().tolist())
        report["pairs"][name] = rep

    pair_basic_stats(pos, "pos")
    pair_basic_stats(neg, "neg")

    # If columns exist, validate label + subject logic
    def validate_pairs(pairs_df: pd.DataFrame, expected_label: int, kind: str):
        if "label" in pairs_df.columns:
            bad_label = pairs_df[pairs_df["label"] != expected_label]
            if len(bad_label) > 0:
                report["problems"].append(f"{kind}: {len(bad_label)} rows have wrong label (expected {expected_label}).")

        # Try subject_id_a/subject_id_b if present
        if {"subject_id_a", "subject_id_b"}.issubset(pairs_df.columns):
            if kind == "pos":
                bad = pairs_df[pairs_df["subject_id_a"] != pairs_df["subject_id_b"]]
                if len(bad) > 0:
                    report["problems"].append(f"{kind}: {len(bad)} pairs are not same-subject.")
            else:
                bad = pairs_df[pairs_df["subject_id_a"] == pairs_df["subject_id_b"]]
                if len(bad) > 0:
                    report["problems"].append(f"{kind}: {len(bad)} pairs are same-subject but should be negative.")

        # Try file existence if path_a/path_b exist
        if {"path_a", "path_b"}.issubset(pairs_df.columns):
            p_missing = []
            for a, b in pairs_df[["path_a", "path_b"]].head(200).itertuples(index=False):
                if not Path(a).exists() or not Path(b).exists():
                    p_missing.append((a, b))
            if p_missing:
                report["problems"].append(f"{kind}: missing files in first 200 pairs. Example: {p_missing[:1]}")

    validate_pairs(pos, expected_label=1, kind="pos")
    validate_pairs(neg, expected_label=0, kind="neg")

    out_path = processed_dir / "sanity_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SANITY CHECK DONE")
    print(f"Wrote: {out_path}")
    if report["problems"]:
        print("\nProblems found:")
        for p in report["problems"]:
            print(" -", p)
    else:
        print("No problems detected in basic checks.")


if __name__ == "__main__":
    main()
