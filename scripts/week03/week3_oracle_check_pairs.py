from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

def auc_roc(y_true, y_score):
    # simple AUC without sklearn (rank-based)
    # returns ~0.5 if random, 1.0 if perfect
    df = pd.DataFrame({"y": y_true, "s": y_score}).dropna()
    df = df.sort_values("s")
    # average ranks for ties
    df["rank"] = df["s"].rank(method="average")
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    sum_ranks_pos = pos["rank"].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--root", default=r"C:\fingerprint-research")
    args = ap.parse_args()

    root = Path(args.root)
    data_dir = root / "data" / "processed" / "nist_sd300b"
    manifest_path = data_dir / "manifest.csv"
    pairs_path = data_dir / f"pairs_{args.split}.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Missing pairs: {pairs_path}")

    dfm = pd.read_csv(manifest_path)
    dfp = pd.read_csv(pairs_path)

    # Expect columns path_a/path_b/label
    needed = {"path_a", "path_b", "label"}
    missing = needed - set(dfp.columns)
    if missing:
        raise ValueError(f"Pairs file missing columns: {missing}. Columns={list(dfp.columns)}")

    # Join metadata for A and B
    dfm_small = dfm[["path", "subject_id", "frgp", "capture"]].copy()
    dfm_small = dfm_small.rename(columns={"path": "path_a", "subject_id": "sid_a", "frgp": "frgp_a", "capture": "cap_a"})
    dfp = dfp.merge(dfm_small, on="path_a", how="left")

    dfm_small = dfm[["path", "subject_id", "frgp", "capture"]].copy()
    dfm_small = dfm_small.rename(columns={"path": "path_b", "subject_id": "sid_b", "frgp": "frgp_b", "capture": "cap_b"})
    dfp = dfp.merge(dfm_small, on="path_b", how="left")

    null_meta = dfp[["sid_a","sid_b","frgp_a","frgp_b"]].isna().any(axis=1).mean()
    print(f"Pairs: {len(dfp)} | missing metadata rows: {null_meta*100:.2f}%")

    # Oracle rule: same sid+frgp => 1 else 0
    oracle = ((dfp["sid_a"] == dfp["sid_b"]) & (dfp["frgp_a"] == dfp["frgp_b"])).astype(int)
    y = dfp["label"].astype(int)

    auc = auc_roc(y, oracle)
    pos_rate = y.mean()
    oracle_pos_rate = oracle.mean()

    # also show capture composition
    same_cap = (dfp["cap_a"] == dfp["cap_b"]).mean()

    print(f"label positive rate: {pos_rate:.4f}")
    print(f"oracle positive rate: {oracle_pos_rate:.4f}")
    print(f"same-capture fraction: {same_cap:.4f}")
    print(f"ORACLE AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
