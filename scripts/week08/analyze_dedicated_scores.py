import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def parse_path(s: str) -> Path:
    s = str(s)
    if s.startswith("file:"):
        s = s[len("file:"):]
        while s.startswith("/"):
            s = s[1:]
    return Path(s)

def q(x):
    return {
        "min": float(np.min(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "med": float(np.quantile(x, 0.50)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_dedicated_scores.py <scores_csv_or_file_uri>")
        sys.exit(1)

    p = parse_path(sys.argv[1])
    df = pd.read_csv(p)

    # required cols
    for c in ["label", "score", "inliers", "tentative"]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Found: {list(df.columns)}")

    y = df["label"].astype(int).values
    score = df["score"].astype(float).values
    inl = df["inliers"].astype(int).values
    ten = df["tentative"].astype(int).values

    print(f"[FILE] {p}")
    print(f"[N] {len(df)}  pos={int((y==1).sum())}  neg={int((y==0).sum())}")
    print()

    # AUC of different signals
    def safe_auc(a):
        return float(roc_auc_score(y, a)) if len(np.unique(a)) > 1 else float("nan")

    ratio = inl / np.maximum(1, ten)
    print("[AUC] score:", safe_auc(score))
    print("[AUC] inliers:", safe_auc(inl))
    print("[AUC] tentative:", safe_auc(ten))
    print("[AUC] inliers/tentative:", safe_auc(ratio))
    print()

    for lab in [1, 0]:
        sub = df[df["label"] == lab]
        print(f"[LABEL {lab}] n={len(sub)}")
        print("  score:", q(sub["score"].values))
        print("  inliers:", q(sub["inliers"].values))
        print("  tentative:", q(sub["tentative"].values))
        z_inl = float((sub["inliers"].values == 0).mean())
        z_ten = float((sub["tentative"].values < 3).mean())
        print(f"  frac(inliers==0)={z_inl:.4f}  frac(tentative<3)={z_ten:.4f}")
        print()

if __name__ == "__main__":
    main()
