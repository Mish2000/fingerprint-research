from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from pipelines.ingest.pair_bundle_utils import (
    build_pairs_split_build_meta,
    build_split_subjects_metadata,
    validate_canonical_pairs_df,
    validate_pairs_split_build_meta,
    validate_split_subjects_metadata,
    write_json,
)


DATASET = "nist_sd300c"
DEFAULT_SEED = 42
DEFAULT_NEG_PER_POS = 3
DEFAULT_FINGER_COL = "frgp"
DEFAULT_POSITIVE_POLICY = "same_subject_same_finger_plain_to_roll"
DEFAULT_NEGATIVE_POLICY = "same_finger_other_subject_same_split"


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def load_sd300b_helpers():
    src = repo_root_from_here() / "pipelines" / "ingest" / "prepare_data_sd300b.py"
    if not src.exists():
        raise FileNotFoundError(f"Missing helper source: {src}")

    spec = importlib.util.spec_from_file_location("_prepare_data_sd300b_ref", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {src}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def write_nested_pairs_bundle(
    out_dir: Path,
    split_subjects: dict,
    *,
    seed: int,
    neg_per_pos: int,
    manifest_path: Path,
) -> None:
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for sp in ("train", "val", "test"):
        src = out_dir / f"pairs_{sp}.csv"
        if src.exists():
            shutil.copy2(src, pairs_dir / src.name)

    split_meta = build_split_subjects_metadata(
        splits=split_subjects,
        seed=seed,
        neg_per_pos=neg_per_pos,
        impostors_per_pos=neg_per_pos,
        same_finger_policy=True,
        negative_pair_policy=DEFAULT_NEGATIVE_POLICY,
        positive_pair_policy=DEFAULT_POSITIVE_POLICY,
        finger_col=DEFAULT_FINGER_COL,
        resolved_data_dir=out_dir,
        manifest_path=manifest_path,
        max_pos_per_subject=5000,
        max_pos_per_finger=500,
        pair_mode="plain_to_roll",
    )
    validate_split_subjects_metadata(split_meta, context=f"{DATASET} split_subjects metadata")
    write_json(pairs_dir / "split_subjects.json", split_meta)


def main() -> None:
    rr = repo_root_from_here()
    ref = load_sd300b_helpers()

    ap = argparse.ArgumentParser(
        description="Prepare SD300C manifest/splits/pairs bundle under data/manifests."
    )
    ap.add_argument(
        "--plain_dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--roll_dir",
        type=str,
        default=None,
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(rr / "data" / "manifests" / DATASET),
    )
    ap.add_argument("--ppi", type=int, default=2000)
    ap.add_argument("--ext", type=str, default="png")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--train_ratio", type=float, default=0.80)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--neg_per_pos", type=int, default=DEFAULT_NEG_PER_POS)
    args = ap.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train_ratio must be in (0, 1)")
    if not (0.0 <= args.val_ratio < 1.0):
        raise ValueError("--val_ratio must be in [0, 1)")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    plain_dir, roll_dir = ref.resolve_sd300_dirs(
        rr,
        dataset_name='sd300c',
        ppi=int(args.ppi),
        plain_dir=args.plain_dir,
        roll_dir=args.roll_dir,
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip())
    if not exts:
        raise ValueError("No valid extensions parsed from --ext")

    print("Repo root :", rr)
    print("Plain dir :", plain_dir)
    print("Roll dir  :", roll_dir)
    print("Out dir   :", out_dir)
    print("Target PPI:", args.ppi)
    print("Exts      :", exts)

    print("\nBuilding manifest.")
    df = ref.build_manifest(
        plain_dir,
        roll_dir,
        dataset=DATASET,
        target_ppi=args.ppi,
        exts=exts,
    )
    print("Parsed rows:", len(df))
    if len(df) == 0:
        raise RuntimeError("No SD300C rows parsed. Check --plain_dir / --roll_dir / --ext / --ppi.")

    split = ref.split_by_subject(
        df,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    df = ref.assign_split(df, split)
    manifest_path = out_dir / "manifest.csv"
    df.to_csv(manifest_path, index=False)

    df_one = ref.choose_one(df)

    print("\nCreating positive pairs.")
    pos = ref.make_positive_pairs(df_one)
    pos.to_csv(out_dir / "pairs_pos.csv", index=False)

    print("Creating negative pairs.")
    neg = ref.make_negative_pairs(df_one, pos, seed=args.seed, neg_per_pos=args.neg_per_pos)
    neg.to_csv(out_dir / "pairs_neg.csv", index=False)

    print("Writing per-split combined pair files.")
    for sp in ("train", "val", "test"):
        pairs_sp = ref.build_split_pairs(pos, neg, sp)
        pairs_sp = validate_canonical_pairs_df(
            pairs_sp,
            context=f"{DATASET}/{sp} canonical pairs",
            expected_split=sp,
            require_exact_columns=True,
            require_non_empty=True,
        )
        pairs_sp.to_csv(out_dir / f"pairs_{sp}.csv", index=False)

    write_nested_pairs_bundle(
        out_dir,
        split,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        manifest_path=manifest_path,
    )

    meta = build_pairs_split_build_meta(
        dataset=DATASET,
        seed=int(args.seed),
        neg_per_pos=int(args.neg_per_pos),
        impostors_per_pos=int(args.neg_per_pos),
        finger_col=DEFAULT_FINGER_COL,
        positive_pair_policy=DEFAULT_POSITIVE_POLICY,
        negative_pair_policy=DEFAULT_NEGATIVE_POLICY,
        extra={
            "plain_dir": str(plain_dir),
            "roll_dir": str(roll_dir),
            "out_dir": str(out_dir),
            "ppi": int(args.ppi),
            "exts": list(exts),
            "train_ratio": float(args.train_ratio),
            "val_ratio": float(args.val_ratio),
            "test_ratio": float(1.0 - args.train_ratio - args.val_ratio),
            "helper_source": "pipelines/ingest/prepare_data_sd300b.py",
            "choose_one_policy": "lexicographically smallest path per (subject_id, frgp, capture, split)",
        },
    )
    validate_pairs_split_build_meta(meta, context=f"{DATASET} pairs_split_build metadata")
    write_json(out_dir / "pairs_split_build.meta.json", meta)

    stats = {
        "manifest_rows": int(len(df)),
        "unique_subjects": int(df["subject_id"].nunique()),
        "unique_frgp": sorted(int(x) for x in df["frgp"].unique().tolist()),
        "plain_rows": int((df["capture"] == "plain").sum()),
        "roll_rows": int((df["capture"] == "roll").sum()),
        "pos_pairs": int(len(pos)),
        "neg_pairs": int(len(neg)),
        "pos_by_split": pos["split"].value_counts().to_dict(),
        "neg_by_split": neg["split"].value_counts().to_dict(),
        "pairs_by_split": {
            sp: int(pd.read_csv(out_dir / f"pairs_{sp}.csv").shape[0])
            for sp in ("train", "val", "test")
        },
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    sanity = ref.sanity_checks(df, split, pos, neg)
    (out_dir / "sanity_report.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")

    print("\nDONE.")
    print("Stats:\n", json.dumps(stats, indent=2))
    print("Sanity:\n", json.dumps(sanity, indent=2))
    print("Wrote nested pairs bundle to:", out_dir / "pairs")


if __name__ == "__main__":
    main()
