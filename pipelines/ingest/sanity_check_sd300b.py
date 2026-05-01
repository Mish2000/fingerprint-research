from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from pipelines.ingest.pair_bundle_utils import (
    CANONICAL_PAIR_COLUMNS,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _pick_file(base_dir: Path, keywords: list[str]) -> Path:
    files = list(base_dir.glob("*.csv")) + list(base_dir.glob("*.json"))
    scored = []
    for f in files:
        name = f.name.lower()
        score = sum(1 for k in keywords if k in name)
        if score > 0:
            scored.append((score, f))

    if not scored:
        raise FileNotFoundError(
            f"Could not find file in {base_dir} with keywords={keywords}. "
            f"Found: {[x.name for x in files]}"
        )

    scored.sort(key=lambda x: (-x[0], x[1].name))
    return scored[0][1]


def _resolve_data_dir(input_dir: Path) -> Path:
    """
    Supports both layouts:
    1) data/processed/<dataset>
    2) data/manifests/<dataset>
    """
    candidates: list[Path] = [input_dir]

    try:
        if input_dir.parent.name.lower() == "processed":
            candidates.append(input_dir.parent.parent / "manifests" / input_dir.name)
    except Exception:
        pass

    try:
        if input_dir.parent.name.lower() == "manifests":
            candidates.append(input_dir.parent.parent / "processed" / input_dir.name)
    except Exception:
        pass

    for c in candidates:
        if not c.exists():
            continue

        files = list(c.glob("*.csv")) + list(c.glob("*.json"))
        names = {f.name.lower() for f in files}

        has_manifest = "manifest.csv" in names or any("manifest" in n for n in names)
        has_pos = "pairs_pos.csv" in names or any(("pairs" in n and "pos" in n) for n in names)
        has_neg = "pairs_neg.csv" in names or any(("pairs" in n and "neg" in n) for n in names)

        if has_manifest and has_pos and has_neg:
            return c

    checked = [str(c) for c in candidates]
    raise FileNotFoundError(f"Could not locate SD300B data dir. Checked: {checked}")


def _candidate_remaps(path_str: str, raw_root: Path) -> list[Path]:
    s = str(path_str).strip()
    if not s or s.lower() == "nan":
        return []

    p = Path(s)
    candidates: list[Path] = []

    if p.exists():
        candidates.append(p)

    parts = list(p.parts)
    lower_parts = [str(x).lower() for x in parts]

    # Rebuild from ".../images/1000/png/plain/file.png" tail
    if "images" in lower_parts:
        idx = lower_parts.index("images")
        tail = parts[idx:]  # images/1000/png/plain/filename.png
        candidates.append(raw_root / Path(*tail))

    # Fallback: search by filename under raw_root/images/*/png/{plain|roll}/
    name = p.name
    if name:
        lname = name.lower()
        capture = None
        if "roll" in lname:
            capture = "roll"
        elif "plain" in lname:
            capture = "plain"

        images_root = raw_root / "images"
        if images_root.exists():
            if capture is not None:
                candidates.extend(images_root.glob(f"*/png/{capture}/{name}"))
            candidates.extend(images_root.glob(f"*/png/*/{name}"))

    # de-dup preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        key = str(c)
        if key not in seen:
            seen.add(key)
            out.append(c)

    return out


def _remap_path_str(path_str: str, raw_root: Path) -> str:
    for cand in _candidate_remaps(path_str, raw_root):
        if cand.exists():
            return str(cand.resolve())
    return str(path_str)


def _remap_columns_in_df(df: pd.DataFrame, columns: list[str], raw_root: Path) -> dict[str, int]:
    changes: dict[str, int] = {}

    for col in columns:
        if col not in df.columns:
            continue

        changed = 0
        new_vals = []

        for v in df[col].tolist():
            if pd.isna(v):
                new_vals.append(v)
                continue

            old = str(v)
            new = _remap_path_str(old, raw_root)
            if new != old:
                changed += 1
            new_vals.append(new)

        df[col] = new_vals
        changes[col] = changed

    return changes


def _csvs_to_repair(data_dir: Path) -> list[Path]:
    csvs = list(data_dir.glob("*.csv"))
    pairs_dir = data_dir / "pairs"
    if pairs_dir.exists():
        csvs.extend(list(pairs_dir.glob("*.csv")))

    # de-dup
    uniq = []
    seen = set()
    for p in csvs:
        s = str(p.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    return sorted(uniq)


def _repair_csvs_inplace(data_dir: Path, raw_root: Path) -> dict[str, dict[str, int]]:
    repaired: dict[str, dict[str, int]] = {}

    for csv_path in _csvs_to_repair(data_dir):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        changes = _remap_columns_in_df(df, ["path", "path_a", "path_b"], raw_root)
        total_changed = sum(changes.values())

        if total_changed > 0:
            df.to_csv(csv_path, index=False)

        repaired[str(csv_path.relative_to(data_dir))] = changes

    return repaired


def main():
    rr = repo_root_from_here()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed_dir",
        type=str,
        default=str(rr / "data" / "processed" / "nist_sd300b"),
        help="Dataset folder. Supports either data/processed/<dataset> or data/manifests/<dataset>.",
    )
    ap.add_argument(
        "--raw_root",
        type=str,
        default=str(rr / "data" / "raw" / "sd300b"),
        help="Current SD300B raw root, e.g. <repo>/data/raw/sd300b",
    )
    ap.add_argument(
        "--repair_paths_inplace",
        action="store_true",
        help="Rewrite manifest/pairs CSV files in-place with remapped local paths.",
    )
    args = ap.parse_args()

    input_dir = Path(args.processed_dir).expanduser().resolve()
    data_dir = _resolve_data_dir(input_dir)

    raw_root = Path(args.raw_root).expanduser().resolve()
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    repaired = {}
    if args.repair_paths_inplace:
        repaired = _repair_csvs_inplace(data_dir, raw_root)

    manifest_path = _pick_file(data_dir, ["manifest"])
    pos_path = _pick_file(data_dir, ["pos", "pair"])
    neg_path = _pick_file(data_dir, ["neg", "pair"])
    stats_path = _pick_file(data_dir, ["stats"])

    df = pd.read_csv(manifest_path)
    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)

    manifest_remap = _remap_columns_in_df(df, ["path"], raw_root)
    pos_remap = _remap_columns_in_df(pos, ["path_a", "path_b"], raw_root)
    neg_remap = _remap_columns_in_df(neg, ["path_a", "path_b"], raw_root)

    report = {
        "paths": {
            "data_dir": str(data_dir),
            "raw_root": str(raw_root),
            "manifest": str(manifest_path),
            "pairs_pos": str(pos_path),
            "pairs_neg": str(neg_path),
            "stats": str(stats_path),
        },
        "manifest": {},
        "pairs": {},
        "checks": {
            "remap_counts": {
                "manifest": manifest_remap,
                "pos": pos_remap,
                "neg": neg_remap,
            },
            "repaired_files_inplace": repaired,
        },
        "problems": [],
    }

    required_cols = {"dataset", "capture", "subject_id", "frgp", "ppi", "path"}
    missing = sorted(list(required_cols - set(df.columns)))
    if missing:
        report["problems"].append(f"manifest missing columns: {missing}")

    report["manifest"]["rows"] = int(len(df))
    report["manifest"]["unique_subjects"] = int(df["subject_id"].nunique()) if "subject_id" in df.columns else None
    report["manifest"]["frgp_values"] = sorted(df["frgp"].unique().tolist()) if "frgp" in df.columns else None
    report["manifest"]["ppi_values"] = sorted(df["ppi"].unique().tolist()) if "ppi" in df.columns else None
    report["manifest"]["capture_counts"] = df["capture"].value_counts().to_dict() if "capture" in df.columns else None

    if "split" in df.columns:
        split_vals = sorted(df["split"].dropna().unique().tolist())
        report["manifest"]["split_values"] = split_vals

        subj_splits = df.groupby("subject_id")["split"].nunique()
        leaked = subj_splits[subj_splits > 1]
        report["checks"]["identity_leakage_subjects"] = int(len(leaked))
        if len(leaked) > 0:
            report["problems"].append(f"IDENTITY LEAKAGE: {len(leaked)} subjects appear in multiple splits.")
    else:
        report["manifest"]["split_values"] = None

    sample_paths = df["path"].head(200).tolist() if "path" in df.columns else []
    missing_files = [p for p in sample_paths if not Path(p).exists()]
    report["checks"]["missing_files_in_first_200"] = int(len(missing_files))
    if missing_files:
        report["problems"].append(f"Some files do not exist (first 200 sample). Example: {missing_files[:3]}")

    def pair_basic_stats(pairs_df: pd.DataFrame, name: str):
        rep = {"rows": int(len(pairs_df)), "columns": list(pairs_df.columns)}
        rep["canonical_columns"] = bool(list(pairs_df.columns) == CANONICAL_PAIR_COLUMNS)
        if "split" in pairs_df.columns:
            rep["by_split"] = pairs_df["split"].value_counts().to_dict()
        if "label" in pairs_df.columns:
            rep["label_values"] = sorted(pairs_df["label"].unique().tolist())
        report["pairs"][name] = rep

    pair_basic_stats(pos, "pos")
    pair_basic_stats(neg, "neg")

    def validate_pairs(pairs_df: pd.DataFrame, expected_label: int, kind: str):
        if "label" in pairs_df.columns:
            bad_label = pairs_df[pairs_df["label"] != expected_label]
            if len(bad_label) > 0:
                report["problems"].append(
                    f"{kind}: {len(bad_label)} rows have wrong label (expected {expected_label})."
                )

        subject_a_col = "subject_id_a" if "subject_id_a" in pairs_df.columns else "subject_a" if "subject_a" in pairs_df.columns else None
        subject_b_col = "subject_id_b" if "subject_id_b" in pairs_df.columns else "subject_b" if "subject_b" in pairs_df.columns else None

        if subject_a_col and subject_b_col:
            if kind == "pos":
                bad = pairs_df[pairs_df[subject_a_col] != pairs_df[subject_b_col]]
                if len(bad) > 0:
                    report["problems"].append(f"{kind}: {len(bad)} pairs are not same-subject.")
            else:
                bad = pairs_df[pairs_df[subject_a_col] == pairs_df[subject_b_col]]
                if len(bad) > 0:
                    report["problems"].append(f"{kind}: {len(bad)} pairs are same-subject but should be negative.")

        if {"path_a", "path_b"}.issubset(pairs_df.columns):
            p_missing = []
            for a, b in pairs_df[["path_a", "path_b"]].head(200).itertuples(index=False):
                if not Path(a).exists() or not Path(b).exists():
                    p_missing.append((a, b))
            if p_missing:
                report["problems"].append(
                    f"{kind}: missing files in first 200 pairs. Example: {p_missing[:1]}"
                )

    validate_pairs(pos, expected_label=1, kind="pos")
    validate_pairs(neg, expected_label=0, kind="neg")

    split_subjects_path = data_dir / "pairs" / "split_subjects.json"
    if split_subjects_path.exists():
        try:
            validate_split_subjects_metadata(json.loads(split_subjects_path.read_text(encoding="utf-8")), context=str(split_subjects_path))
        except Exception as exc:
            report["problems"].append(f"split_subjects metadata invalid: {exc}")

    pair_build_meta_path = data_dir / "pairs_split_build.meta.json"
    if pair_build_meta_path.exists():
        try:
            validate_pairs_split_build_meta(json.loads(pair_build_meta_path.read_text(encoding="utf-8")), context=str(pair_build_meta_path))
        except Exception as exc:
            report["problems"].append(f"pairs_split_build metadata invalid: {exc}")

    out_path = data_dir / "sanity_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("SANITY CHECK DONE")
    print(f"Resolved data dir: {data_dir}")
    print(f"Raw root        : {raw_root}")
    if args.repair_paths_inplace:
        print("Path repair     : applied in-place to CSV files")
    else:
        print("Path repair     : in-memory only (CSV files were not modified)")
    print(f"Wrote: {out_path}")

    if report["problems"]:
        print("\nProblems found:")
        for p in report["problems"]:
            print(" -", p)
    else:
        print("No problems detected in basic checks.")


if __name__ == "__main__":
    main()