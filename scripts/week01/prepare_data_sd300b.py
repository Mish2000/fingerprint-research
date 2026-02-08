from __future__ import annotations

"""
Prepare SD300B (1000ppi) processed manifests + split + pairs for Unit 1.

Inputs (required):
  --plain_dir PATH   Directory containing SD300B plain PNGs
  --roll_dir  PATH   Directory containing SD300B roll  PNGs

Outputs (default under <repo>/data/processed/nist_sd300b/):
  - manifest.csv
  - split.json
  - stats.json
  - pairs_pos.csv
  - pairs_neg.csv
  - pairs_train.csv / pairs_val.csv / pairs_test.csv
  - pairs_split_build.meta.json
  - sanity_report.json

Notes:
- Paths written into CSVs are absolute paths resolved on your machine.
- These outputs are meant to be GENERATED locally and should be git-ignored.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# SUBJECT_IMPRESSION_PPI_FRGP.EXT
FNAME_RE = re.compile(
    r"^(?P<subject>\d+)_(?P<impression>.+)_(?P<ppi>\d+)_(?P<frgp>\d+)\.(?P<ext>\w+)$"
)


@dataclass(frozen=True)
class Row:
    dataset: str
    capture: str          # "plain" or "roll" (from folder)
    subject_id: int
    impression: str       # "plain" / "roll" (from filename)
    ppi: int
    frgp: int
    path: str             # absolute path
    split: Optional[str] = None


def repo_root_from_here() -> Path:
    # scripts/week01/prepare_data_sd300b.py -> <repo>
    return Path(__file__).resolve().parents[2]


def parse_file(p: Path, capture: str, dataset: str, *, target_ppi: int, exts: Tuple[str, ...]) -> Optional[Row]:
    if p.suffix.lower().lstrip(".") not in exts:
        return None

    m = FNAME_RE.match(p.name)
    if not m:
        return None

    subject_id = int(m.group("subject"))
    impression = m.group("impression")
    ppi = int(m.group("ppi"))
    frgp = int(m.group("frgp"))

    # Safety: ensure filename impression matches the directory we're scanning
    if impression != capture:
        return None

    # Keep only target ppi (Week-1: 1000)
    if ppi != target_ppi:
        return None

    # Week-1 clean fingerprint matching: keep only finger positions 1..10
    if not (1 <= frgp <= 10):
        return None

    return Row(
        dataset=dataset,
        capture=capture,
        subject_id=subject_id,
        impression=impression,
        ppi=ppi,
        frgp=frgp,
        path=str(p.resolve()),
    )


def build_manifest(plain_dir: Path, roll_dir: Path, dataset: str, *, target_ppi: int, exts: Tuple[str, ...]) -> pd.DataFrame:
    rows: List[Row] = []

    for capture, d in [("plain", plain_dir), ("roll", roll_dir)]:
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")

        files: List[Path] = []
        for ext in exts:
            files.extend(sorted(d.rglob(f"*.{ext}")))

        for p in tqdm(files, desc=f"Scanning {capture}", unit="file"):
            r = parse_file(p, capture=capture, dataset=dataset, target_ppi=target_ppi, exts=exts)
            if r is not None:
                rows.append(r)

    if not rows:
        raise RuntimeError(
            "No valid files were parsed. Check that:\n"
            " - directories are correct\n"
            " - filenames match SUBJECT_IMPRESSION_PPI_FRGP.png\n"
            " - impression in filename matches folder (plain/roll)\n"
            " - PPI matches the requested --ppi (default 1000)\n"
        )

    return pd.DataFrame([r.__dict__ for r in rows])


def split_by_subject(
    df: pd.DataFrame,
    *,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[int]]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1). Got: {train_ratio}")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1). Got: {val_ratio}")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio must be < 1. Got: {train_ratio + val_ratio}")

    subjects = sorted(df["subject_id"].unique().tolist())
    if len(subjects) < 3:
        raise RuntimeError(f"Not enough unique subjects to split safely. Found {len(subjects)}.")

    train_subj, tmp_subj = train_test_split(
        subjects, test_size=(1.0 - train_ratio), random_state=seed, shuffle=True
    )
    # proportion of tmp that should become val
    val_size = val_ratio / (1.0 - train_ratio)
    val_subj, test_subj = train_test_split(
        tmp_subj, test_size=(1.0 - val_size), random_state=seed, shuffle=True
    )

    return {
        "train": sorted(train_subj),
        "val": sorted(val_subj),
        "test": sorted(test_subj),
    }


def assign_split(df: pd.DataFrame, split: Dict[str, List[int]]) -> pd.DataFrame:
    split_map: Dict[int, str] = {}
    for k, ids in split.items():
        for sid in ids:
            split_map[int(sid)] = k

    out = df.copy()
    out["split"] = out["subject_id"].map(split_map)
    return out


def choose_one(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep 1 sample per (subject_id, frgp, capture, split).
    Deterministic: pick lexicographically smallest path.
    """
    df = df.sort_values(["subject_id", "frgp", "capture", "path"])
    return df.groupby(["subject_id", "frgp", "capture", "split"], as_index=False).first()


def make_positive_pairs(df_one: pd.DataFrame) -> pd.DataFrame:
    plain = df_one[df_one["capture"] == "plain"][["subject_id", "frgp", "split", "path"]].rename(columns={"path": "path_a"})
    roll = df_one[df_one["capture"] == "roll"][["subject_id", "frgp", "split", "path"]].rename(columns={"path": "path_b"})

    pos = plain.merge(roll, on=["subject_id", "frgp", "split"], how="inner")
    pos = pos.sort_values(["split", "subject_id", "frgp"]).reset_index(drop=True)

    # For consistency with negatives:
    pos = pos.rename(columns={"subject_id": "subject_a"})
    pos["subject_b"] = pos["subject_a"]

    pos.insert(0, "pair_id", range(len(pos)))
    pos["label"] = 1
    return pos[["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"]]


def make_negative_pairs(df_one: pd.DataFrame, pos: pd.DataFrame, *, seed: int, neg_per_pos: int) -> pd.DataFrame:
    if neg_per_pos < 1:
        raise ValueError(f"neg_per_pos must be >= 1. Got: {neg_per_pos}")

    roll_pool = df_one[df_one["capture"] == "roll"][["split", "frgp", "subject_id", "path"]].copy()
    roll_pool = roll_pool.rename(columns={"path": "path_b", "subject_id": "subject_b"})

    neg_rows = []
    neg_id = 0

    for _, row in pos.iterrows():
        split = str(row["split"])
        frgp = int(row["frgp"])
        subject_a = int(row["subject_a"])
        path_a = str(row["path_a"])

        candidates = roll_pool[
            (roll_pool["split"] == split)
            & (roll_pool["frgp"] == frgp)
            & (roll_pool["subject_b"] != subject_a)
        ]
        if len(candidates) == 0:
            continue

        sampled = candidates.sample(
            n=min(neg_per_pos, len(candidates)),
            replace=False,
            random_state=seed + subject_a * 131 + frgp * 17,
        )

        for _, s in sampled.iterrows():
            neg_rows.append(
                {
                    "pair_id": neg_id,
                    "label": 0,
                    "split": split,
                    "subject_a": subject_a,
                    "subject_b": int(s["subject_b"]),
                    "frgp": frgp,
                    "path_a": path_a,
                    "path_b": str(s["path_b"]),
                }
            )
            neg_id += 1

    neg = pd.DataFrame(neg_rows)
    if len(neg) == 0:
        raise RuntimeError("No negative pairs were generated (unexpected).")
    return neg[["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"]]


def build_split_pairs(pos: pd.DataFrame, neg: pd.DataFrame, split: str) -> pd.DataFrame:
    df = pd.concat([pos[pos["split"] == split], neg[neg["split"] == split]], axis=0, ignore_index=True)
    # Reassign unique pair_id within each split file (simplifies downstream assumptions)
    df = df.reset_index(drop=True)
    df["pair_id"] = range(len(df))
    return df[["pair_id", "label", "split", "subject_a", "subject_b", "frgp", "path_a", "path_b"]]


def sanity_checks(df: pd.DataFrame, split: Dict[str, List[int]], pos: pd.DataFrame, neg: pd.DataFrame) -> Dict[str, object]:
    # Disjoint split subjects
    s_train, s_val, s_test = map(set, (split["train"], split["val"], split["test"]))
    disjoint_ok = (len(s_train & s_val) == 0) and (len(s_train & s_test) == 0) and (len(s_val & s_test) == 0)

    # Subject leakage check for manifest
    leak_rows = 0
    for sp in ["train", "val", "test"]:
        ids = set(split[sp])
        leak_rows += int(((df["split"] == sp) & (~df["subject_id"].isin(ids))).sum())

    # Pair split consistency
    pos_bad = int((pos["subject_a"] != pos["subject_b"]).sum())
    neg_bad = int((neg["subject_a"] == neg["subject_b"]).sum())

    return {
        "split_subjects_disjoint": disjoint_ok,
        "manifest_split_leak_rows": leak_rows,
        "positive_pairs_subject_mismatch_rows": pos_bad,
        "negative_pairs_same_subject_rows": neg_bad,
        "ok": bool(disjoint_ok and leak_rows == 0 and pos_bad == 0 and neg_bad == 0),
    }


def main() -> None:
    rr = repo_root_from_here()
    dataset = "nist_sd300b"

    ap = argparse.ArgumentParser(description="Prepare SD300B processed manifests/pairs for Unit 1.")
    ap.add_argument("--plain_dir", type=str, required=True, help="Path to SD300B plain images directory")
    ap.add_argument("--roll_dir", type=str, required=True, help="Path to SD300B roll images directory")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(rr / "data" / "processed" / dataset),
        help="Output directory (default: <repo>/data/processed/nist_sd300b)",
    )
    ap.add_argument("--ppi", type=int, default=1000, help="Target PPI to include (default: 1000)")
    ap.add_argument("--ext", type=str, default="png", help="File extension(s), comma-separated (default: png)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--neg_per_pos", type=int, default=3)
    args = ap.parse_args()

    plain_dir = Path(args.plain_dir).expanduser().resolve()
    roll_dir = Path(args.roll_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple([e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()])
    if not exts:
        raise ValueError("No valid extensions parsed from --ext")

    print("Repo root :", rr)
    print("Plain dir :", plain_dir)
    print("Roll dir  :", roll_dir)
    print("Out dir   :", out_dir)
    print("Target PPI:", args.ppi)
    print("Exts      :", exts)

    # 1) Manifest
    print("\nBuilding manifest...")
    df = build_manifest(plain_dir, roll_dir, dataset=dataset, target_ppi=args.ppi, exts=exts)
    print("Parsed rows:", len(df))
    print(df.head(3))

    # 2) Split by subject (no leakage)
    split = split_by_subject(df, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    df = assign_split(df, split)
    df.to_csv(out_dir / "manifest.csv", index=False)

    # 3) Choose one sample per (subject, finger, capture, split)
    df_one = choose_one(df)

    # 4) Pairs
    print("\nCreating positive pairs...")
    pos = make_positive_pairs(df_one)
    pos.to_csv(out_dir / "pairs_pos.csv", index=False)

    print("Creating negative pairs...")
    neg = make_negative_pairs(df_one, pos, seed=args.seed, neg_per_pos=args.neg_per_pos)
    neg.to_csv(out_dir / "pairs_neg.csv", index=False)

    # 5) Combined per-split pair files (handy for downstream scoring)
    print("Writing per-split combined pair files...")
    for sp in ["train", "val", "test"]:
        pairs_sp = build_split_pairs(pos, neg, sp)
        pairs_sp.to_csv(out_dir / f"pairs_{sp}.csv", index=False)

    # 6) Meta + stats + sanity
    meta = {
        "dataset": dataset,
        "plain_dir": str(plain_dir),
        "roll_dir": str(roll_dir),
        "out_dir": str(out_dir),
        "ppi": int(args.ppi),
        "exts": list(exts),
        "seed": int(args.seed),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
        "neg_per_pos": int(args.neg_per_pos),
        "choose_one_policy": "lexicographically smallest path per (subject_id, frgp, capture, split)",
    }
    (out_dir / "pairs_split_build.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stats = {
        "manifest_rows": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()),
        "unique_frgp": sorted([int(x) for x in df["frgp"].unique().tolist()]),
        "plain_rows": int((df["capture"] == "plain").sum()),
        "roll_rows": int((df["capture"] == "roll").sum()),
        "pos_pairs": int(len(pos)),
        "neg_pairs": int(len(neg)),
        "pos_by_split": pos["split"].value_counts().to_dict(),
        "neg_by_split": neg["split"].value_counts().to_dict(),
        "pairs_by_split": {
            sp: int(pd.read_csv(out_dir / f"pairs_{sp}.csv").shape[0]) for sp in ["train", "val", "test"]
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sanity = sanity_checks(df, split, pos, neg)
    (out_dir / "sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    print("\nDONE.")
    print("Stats:\n", json.dumps(stats, indent=2))
    print("Sanity:\n", json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
