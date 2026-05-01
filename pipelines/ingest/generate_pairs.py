import pandas as pd

from pipelines.ingest.pair_bundle_utils import (
    build_split_subjects_metadata,
    canonicalize_pairs_df,
    validate_canonical_pairs_df,
    validate_split_subjects_metadata,
    write_json,
)
import numpy as np
import argparse
from pathlib import Path
import random
import re
import json
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser("Generate matching pairs from manifest (NIST protocol)")
    p.add_argument("--dataset", type=str, default="nist_sd300b")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--neg_per_pos", "--impostors_per_pos",
        dest="neg_per_pos",
        type=int,
        default=3,
        help="Number of impostor (negative) pairs per positive pair.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    p.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Dataset dir. Supports either data/processed/<dataset> or data/manifests/<dataset>.",
    )

    # targets are POS counts (like your current script)
    p.add_argument("--train_limit", type=int, default=2000)
    p.add_argument("--val_limit", type=int, default=700)
    p.add_argument("--test_limit", type=int, default=700)

    # safety caps so one subject/finger doesn't dominate
    p.add_argument("--max_pos_per_subject", type=int, default=5000)
    p.add_argument("--max_pos_per_finger", type=int, default=500)

    return p.parse_args()


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_data_dir(input_dir: Optional[Path], dataset: str) -> Path:
    """
    Supports both:
      - data/processed/<dataset>
      - data/manifests/<dataset>

    Returns the first directory that actually contains manifest.csv.
    """
    root = repo_root_from_here()
    candidates = []

    if input_dir is not None:
        candidates.append(input_dir)

        try:
            parent_name = input_dir.parent.name.lower()
            if parent_name == "processed":
                candidates.append(input_dir.parent.parent / "manifests" / input_dir.name)
            elif parent_name == "manifests":
                candidates.append(input_dir.parent.parent / "processed" / input_dir.name)
        except Exception:
            pass

    candidates.append(root / "data" / "processed" / dataset)
    candidates.append(root / "data" / "manifests" / dataset)

    uniq_candidates = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq_candidates.append(c)

    for c in uniq_candidates:
        if (c / "manifest.csv").exists():
            return c

    checked = [str(c) for c in uniq_candidates]
    raise FileNotFoundError(
        "Could not locate dataset directory containing manifest.csv. "
        f"Checked: {checked}"
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def _infer_finger_col(df: pd.DataFrame) -> str:
    for c in ["finger_id", "frgp", "finger", "finger_idx"]:
        if c in df.columns:
            return c

    # fallback: parse from path column like "..._f05_..."
    path_col = "path_col" if "path_col" in df.columns else "path" if "path" in df.columns else "file_path"
    if path_col in df.columns:
        pat = re.compile(r"_f(\d{2})_")

        def parse_f(path: str):
            m = pat.search(str(path))
            return int(m.group(1)) if m else None

        df["_finger_from_path"] = df[path_col].map(parse_f)
        if df["_finger_from_path"].notna().any():
            return "_finger_from_path"

    raise ValueError(
        "Could not find finger column. Expected finger_id/frgp/finger/finger_idx "
        "or a path pattern like _fXX_."
    )


def _validate_manifest(df: pd.DataFrame):
    capture_col = "capture_col" if "capture_col" in df.columns else "capture" if "capture" in df.columns else None
    path_col = "path_col" if "path_col" in df.columns else "path" if "path" in df.columns else "file_path"

    required = {"subject_id", "split"}
    if capture_col is not None:
        required.add(capture_col)
    if path_col is not None:
        required.add(path_col)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest.csv missing columns: {missing}")

    capture_values = set(df[capture_col].astype(str).str.lower().unique())
    if not capture_values.issuperset({"plain", "roll"}):
        raise ValueError(f"{capture_col} must include 'plain' and 'roll' for NIST protocol.")


def generate_split_pairs(
        df: pd.DataFrame,
        split_subjects: np.ndarray,
        pos_limit: int,
        neg_per_pos: int,
        rng: np.random.Generator,
        finger_col: str,
        max_pos_per_subject: int,
        max_pos_per_finger: int,
):
    # restrict to split subjects only
    sdf = df[df["subject_id"].isin(split_subjects)].copy()

    # pre-index rolls by (finger -> subject -> list(paths))
    rolls = sdf[sdf["capture_col"] == "roll"]
    roll_map = {}
    for (fid, sid), g in rolls.groupby([finger_col, "subject_id"]):
        roll_map.setdefault(fid, {})[sid] = g["path_col"].tolist()

    pos_pairs = []

    # build positives subject-by-subject, finger-by-finger, with caps
    for sid in split_subjects:
        subj_rows = sdf[sdf["subject_id"] == sid]
        plains = subj_rows[subj_rows["capture_col"] == "plain"]
        rolls_s = subj_rows[subj_rows["capture_col"] == "roll"]
        if plains.empty or rolls_s.empty:
            continue

        subj_pos_count = 0
        for fid, pgrp in plains.groupby(finger_col):
            rgrp = rolls_s[rolls_s[finger_col] == fid]
            if rgrp.empty:
                continue

            # all plain x roll combos, but capped
            combos = []
            for p_path in pgrp["path_col"].tolist():
                for r_path in rgrp["path_col"].tolist():
                    combos.append((p_path, r_path))

            if not combos:
                continue

            rng.shuffle(combos)
            combos = combos[:max_pos_per_finger]

            for (a, b) in combos:
                pos_pairs.append(
                    {
                        "capture_a": a,
                        "capture_b": b,
                        "label": 1,
                        "subject_a": sid,
                        "subject_b": sid,
                        "finger_id": int(fid) if fid is not None else fid,
                    }
                )
                subj_pos_count += 1
                if subj_pos_count >= max_pos_per_subject:
                    break

            if subj_pos_count >= max_pos_per_subject:
                break

        if len(pos_pairs) >= pos_limit:
            break

    pos_pairs = pos_pairs[:pos_limit]

    # negatives: for each positive, keep A (plain) and replace B with roll from other subject, same finger
    neg_pairs = []

    for p in pos_pairs:
        sid = p["subject_a"]
        fid = p["finger_id"]

        # candidate subjects in THIS split with rolls for this finger
        subj_to_rolls = roll_map.get(fid, {})
        candidates = [s for s in subj_to_rolls.keys() if s != sid]
        if not candidates:
            continue

        for _ in range(neg_per_pos):
            s2 = rng.choice(candidates)
            b_path = rng.choice(subj_to_rolls[s2])
            neg_pairs.append(
                {
                    "capture_a": p["capture_a"],  # keep the plain-side path_a value
                    "capture_b": b_path,  # use the other-subject roll path as path_b before canonical renaming
                    "label": 0,
                    "subject_a": sid,
                    "subject_b": s2,
                    "finger_id": fid,
                }
            )

    # prefix-safe interleave
    final_pairs = []
    p_idx, n_idx = 0, 0
    total_len = len(pos_pairs) + len(neg_pairs)

    for i in range(total_len):
        if i % (neg_per_pos + 1) == 0:
            if p_idx < len(pos_pairs):
                final_pairs.append(pos_pairs[p_idx])
                p_idx += 1
            elif n_idx < len(neg_pairs):
                final_pairs.append(neg_pairs[n_idx])
                n_idx += 1
        else:
            if n_idx < len(neg_pairs):
                final_pairs.append(neg_pairs[n_idx])
                n_idx += 1
            elif p_idx < len(pos_pairs):
                final_pairs.append(pos_pairs[p_idx])
                p_idx += 1

    return final_pairs, len(pos_pairs), len(neg_pairs)


def main():
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.data_dir:
        input_dir = Path(args.data_dir).expanduser()
    else:
        input_dir = repo_root_from_here() / "data" / "processed" / args.dataset

    data_dir = _resolve_data_dir(input_dir, args.dataset)
    manifest_path = data_dir / "manifest.csv"
    out_dir = data_dir / "pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)

    # backward compatibility for older column names
    if "path_col" not in df.columns and "path" in df.columns:
        df["path_col"] = df["path"]

    if "capture_col" not in df.columns and "capture" in df.columns:
        df["capture_col"] = df["capture"]

    # normalize capture values if needed
    if "capture_col" in df.columns:
        df["capture_col"] = df["capture_col"].astype(str).str.lower()
        df["capture_col"] = df["capture_col"].replace({"p": "plain", "r": "roll"})

    _validate_manifest(df)
    finger_col = _infer_finger_col(df)

    # subject shuffle for split allocation
    subjects = df["subject_id"].unique()
    rng.shuffle(subjects)

    if args.dry_run:
        print("Dry run OK.")
        print("resolved_data_dir:", data_dir)
        print("manifest_path:", manifest_path)
        print("subjects:", len(subjects))
        print("finger_col:", finger_col)
        return

    splits = [("train", args.train_limit), ("val", args.val_limit), ("test", args.test_limit)]
    split_subjects_meta = {}

    for split_name, pos_limit in splits:
        # take the split exactly as defined in the manifest
        df_split = df[df["split"] == split_name].copy()
        split_subjects = df_split["subject_id"].unique()
        split_subjects_meta[split_name] = split_subjects.tolist()

        # generate pairs only from this split
        pairs, npos, nneg = generate_split_pairs(
            df=df_split,
            split_subjects=split_subjects,
            pos_limit=pos_limit,
            neg_per_pos=args.neg_per_pos,
            rng=rng,
            finger_col=finger_col,
            max_pos_per_subject=args.max_pos_per_subject,
            max_pos_per_finger=args.max_pos_per_finger,
        )

        out_df = pd.DataFrame(pairs)

        # align with downstream expectations: path_a/path_b
        if "capture_a" in out_df.columns:
            out_df = out_df.rename(columns={"capture_a": "path_a", "capture_b": "path_b"})

        split_seed_offset = {"train": 0, "val": 1, "test": 2}.get(split_name, 0)
        if not out_df.empty:
            out_df = out_df.sample(frac=1.0, random_state=int(args.seed) + split_seed_offset).reset_index(drop=True)

        csv_path = data_dir / f"pairs_{split_name}.csv"
        if csv_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{csv_path} already exists. Use --overwrite to replace existing files."
            )

        out_df = canonicalize_pairs_df(out_df, split=split_name)
        out_df = validate_canonical_pairs_df(
            out_df,
            context=f"{dataset}/{split_name} canonical pairs",
            expected_split=split_name,
            require_exact_columns=True,
            require_non_empty=True,
        )
        out_df.to_csv(csv_path, index=False)
        print(f"{split_name}: saved {len(out_df)} pairs ({npos} pos, {nneg} neg) -> {csv_path.name}")

    # write split manifest for reproducibility
    meta = build_split_subjects_metadata(
        splits=split_subjects_meta,
        seed=args.seed,
        neg_per_pos=args.neg_per_pos,
        impostors_per_pos=args.neg_per_pos,
        same_finger_policy=True,
        negative_pair_policy="same_finger_other_subject_same_split",
        positive_pair_policy="same_subject_same_finger_plain_to_roll",
        finger_col=finger_col,
        resolved_data_dir=data_dir,
        manifest_path=manifest_path,
        max_pos_per_subject=args.max_pos_per_subject,
        max_pos_per_finger=args.max_pos_per_finger,
    )

    split_json_path = out_dir / "split_subjects.json"
    if split_json_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{split_json_path} already exists. Use --overwrite to replace existing files."
        )

    validate_split_subjects_metadata(meta, context=f"{dataset} split_subjects metadata")
    write_json(split_json_path, meta)


if __name__ == "__main__":
    main()
