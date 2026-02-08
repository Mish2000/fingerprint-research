from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def _interleave_pos_neg(pos_df: pd.DataFrame, neg_df: pd.DataFrame, neg_per_pos: int) -> pd.DataFrame:
    pos_df = pos_df.reset_index(drop=True)
    neg_df = neg_df.reset_index(drop=True)

    rows = []
    j = 0
    for i in range(len(pos_df)):
        rows.append(pos_df.iloc[i])
        j_end = min(j + neg_per_pos, len(neg_df))
        for k in range(j, j_end):
            rows.append(neg_df.iloc[k])
        j += neg_per_pos

    # append remaining negatives (if any)
    for k in range(j, len(neg_df)):
        rows.append(neg_df.iloc[k])

    return pd.DataFrame(rows).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 5: regenerate pairs_<split>.csv from pairs_pos/neg.csv")
    ap.add_argument("--dataset", type=str, default="nist_sd300b")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle pairs within each split deterministically.")
    ap.add_argument("--neg_per_pos", type=int, default=3, help="Used when not shuffling: interleave pattern pos + K*neg.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing pairs_<split>.csv files.")
    ap.add_argument("--dry_run", action="store_true", help="Print stats only (do not write files).")
    args = ap.parse_args()

    root = project_root()
    data_dir = root / "data" / "processed" / args.dataset

    pos_path = data_dir / "pairs_pos.csv"
    neg_path = data_dir / "pairs_neg.csv"

    if not pos_path.exists():
        raise FileNotFoundError(f"Missing: {pos_path}")
    if not neg_path.exists():
        raise FileNotFoundError(f"Missing: {neg_path}")

    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)

    required = {"label", "split", "path_a", "path_b"}
    if not required.issubset(set(pos.columns)):
        raise ValueError(f"pairs_pos.csv missing columns: {sorted(required - set(pos.columns))}")
    if not required.issubset(set(neg.columns)):
        raise ValueError(f"pairs_neg.csv missing columns: {sorted(required - set(neg.columns))}")

    out_meta = {}

    for split in ["train", "val", "test"]:
        out_path = data_dir / f"pairs_{split}.csv"

        pos_s = pos[pos["split"] == split][["label", "path_a", "path_b"]].copy()
        neg_s = neg[neg["split"] == split][["label", "path_a", "path_b"]].copy()

        if args.shuffle:
            df = pd.concat([pos_s, neg_s], ignore_index=True)
            df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
            order_mode = "shuffle"
        else:
            # Interleave is the new safe default (prefix will contain both classes)
            df = _interleave_pos_neg(pos_s, neg_s, neg_per_pos=args.neg_per_pos)
            order_mode = f"interleave_k{args.neg_per_pos}"

        df.insert(0, "pair_id", range(len(df)))

        pos_n = int((df["label"] == 1).sum())
        neg_n = int((df["label"] == 0).sum())
        out_meta[split] = {
            "pairs": int(len(df)),
            "pos": pos_n,
            "neg": neg_n,
            "pos_rate": float(pos_n / max(len(df), 1)),
            "out_csv": str(out_path),
            "order_mode": order_mode,
        }

        print(f"[{split}] N={len(df)} | pos={pos_n} | neg={neg_n} | pos_rate={pos_n/max(len(df),1):.4f}")

        if args.dry_run:
            continue

        if out_path.exists() and not args.overwrite:
            print(f"  -> exists, skipping (use --overwrite): {out_path}")
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  -> wrote: {out_path}")

    meta_path = data_dir / "pairs_split_build.meta.json"
    if not args.dry_run:
        meta = {
            "dataset": args.dataset,
            "seed": args.seed,
            "shuffle": bool(args.shuffle),
            "neg_per_pos": int(args.neg_per_pos),
            "root": str(root),
            "data_dir": str(data_dir),
            "splits": out_meta,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
