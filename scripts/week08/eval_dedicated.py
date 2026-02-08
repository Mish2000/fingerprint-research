import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.dedicated_matcher import DedicatedMatcher, _norm_path


def parse_file_uri(p: str) -> Path:
    if p.startswith("file:"):
        p = p[len("file:"):]
        while p.startswith("/"):
            p = p[1:]
    p = p.replace("/", "\\")
    return Path(p)



def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def compute_auc_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    return auc, eer


def load_capture_map(manifest_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(manifest_csv)
    m: Dict[str, str] = {}
    for _, r in df.iterrows():
        p = str(r["path"])
        cap = str(r["capture"])
        m[_norm_path(p)] = cap
    return m


def infer_capture_from_name(path: str) -> Optional[str]:
    name = Path(path).name.lower()
    if "_plain_" in name:
        return "plain"
    if "_roll_" in name:
        return "roll"
    return None


def resolve_capture(path: str, cap_map: Dict[str, str], fallback: Optional[str]) -> str:
    key = _norm_path(path)
    if key in cap_map:
        return str(cap_map[key])
    if fallback is not None:
        return str(fallback)
    inferred = infer_capture_from_name(path)
    if inferred is not None:
        return inferred
    raise ValueError(
        f"Could not find capture type for:\n  {path}\n"
        f"Either:\n"
        f"  1) pass --fallback_capture manually (NOT recommended for full eval), or\n"
        f"  2) ensure this exact path exists in manifest.csv, or\n"
        f"  3) include '_plain_' or '_roll_' in the filename."
    )


def pick_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of columns {candidates} in pairs CSV. Found: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser(description="Week 08: Dedicated matcher evaluation on pairs_<split>.csv")
    ap.add_argument("out_csv", type=str, help="Output scores CSV. Example: file:/C:/.../scores_dedicated_val.csv")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--pairs", type=str, default="", help="Optional explicit pairs CSV path.")
    ap.add_argument("--limit", type=int, default=0, help="If >0 evaluate only first N pairs (prefix-safe generator).")

    ap.add_argument("--manifest", type=str, default="data/processed/nist_sd300b/manifest.csv")
    ap.add_argument("--ckpt", type=str, default=None)

    # Optional tuning (defaults match src.dedicated_matcher constants)
    ap.add_argument("--patch", type=int, default=48)
    ap.add_argument("--max_kpts", type=int, default=800)
    ap.add_argument("--mask_cov_thr", type=float, default=0.70)
    ap.add_argument("--max_matches", type=int, default=200)
    ap.add_argument("--ransac_thresh", type=float, default=4.0)

    # Only for non-manifest paths (like Week02 visual samples). For full eval you should NOT use this.
    ap.add_argument("--fallback_capture", type=str, default=None, choices=[None, "plain", "roll"])

    args = ap.parse_args()

    out_csv = parse_file_uri(args.out_csv)
    ensure_parent(out_csv)
    meta_path = out_csv.with_suffix(".meta.json")

    root = Path.cwd()
    data_dir = root / "data" / "processed" / "nist_sd300b"
    pairs_path = Path(args.pairs) if args.pairs else (data_dir / f"pairs_{args.split}.csv")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs CSV not found: {pairs_path}")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

    df = pd.read_csv(pairs_path)
    col_a = pick_col(df, ["path_a", "img_a", "a", "left", "path1"])
    col_b = pick_col(df, ["path_b", "img_b", "b", "right", "path2"])
    col_y = pick_col(df, ["label", "y", "is_match"])

    if args.limit and args.limit > 0:
        df = df.iloc[: int(args.limit)].copy()

    cap_map = load_capture_map(manifest_path)

    matcher = DedicatedMatcher(
        ckpt_path=args.ckpt,
        patch=args.patch,
        max_kpts=args.max_kpts,
        mask_cov_thr=args.mask_cov_thr,
        max_matches=args.max_matches,
        ransac_thresh=args.ransac_thresh,
    )

    rows = []
    pair_total_ms = []
    embed_a_ms = []
    embed_b_ms = []
    match_ms = []
    ransac_ms = []

    t0 = time.time()
    for i, r in df.iterrows():
        pa = str(r[col_a])
        pb = str(r[col_b])
        y = int(r[col_y])

        cap_a = resolve_capture(pa, cap_map, args.fallback_capture)
        cap_b = resolve_capture(pb, cap_map, args.fallback_capture)

        res = matcher.score_pair(pa, pb, capture_a=cap_a, capture_b=cap_b)

        lat = res.latency_ms or {}
        ea = float(lat.get("embed_a_total", 0.0))
        eb = float(lat.get("embed_b_total", 0.0))
        mm = float(lat.get("match_ms", 0.0))
        rr = float(lat.get("ransac_ms", 0.0))
        pt = float(lat.get("pair_total_ms", 0.0))

        # Additional diagnostics (does not affect scoring)
        if args.max_matches and args.max_matches > 0:
            denom = float(args.max_matches)
        else:
            denom = float(max(1, int(res.tentative_count)))
        inlier_ratio = float(res.inliers_count) / denom

        mean_inlier_sim = float(getattr(res, "mean_inlier_sim", 0.0))
        median_inlier_sim = float(getattr(res, "median_inlier_sim", 0.0))
        mean_tentative_sim = float(getattr(res, "mean_tentative_sim", 0.0))
        median_tentative_sim = float(getattr(res, "median_tentative_sim", 0.0))
        inlier_sim_factor = float(np.clip(mean_inlier_sim, 0.0, 1.0))
        max_tentative_sim = float(getattr(res, "max_tentative_sim", 0.0))
        mean_top10_tentative_sim = float(getattr(res, "mean_top10_tentative_sim", 0.0))
        mean_top20_tentative_sim = float(getattr(res, "mean_top20_tentative_sim", 0.0))


        rows.append(
            {
                "path_a": pa,
                "path_b": pb,
                "capture_a": cap_a,
                "capture_b": cap_b,
                "label": y,
                "score": float(res.score),
                "inliers": int(res.inliers_count),
                "tentative": int(res.tentative_count),
                "inlier_ratio": float(inlier_ratio),
                "mean_inlier_sim": float(mean_inlier_sim),
                "median_inlier_sim": float(median_inlier_sim),
                "mean_tentative_sim": float(mean_tentative_sim),
                "median_tentative_sim": float(median_tentative_sim),
                "inlier_sim_factor": float(inlier_sim_factor),
                "embed_a_total_ms": ea,
                "embed_b_total_ms": eb,
                "match_ms": mm,
                "ransac_ms": rr,
                "pair_total_ms": pt,
                "max_tentative_sim": float(max_tentative_sim),
                "mean_top10_tentative_sim": float(mean_top10_tentative_sim),
                "mean_top20_tentative_sim": float(mean_top20_tentative_sim),
            }
        )

        pair_total_ms.append(pt)
        embed_a_ms.append(ea)
        embed_b_ms.append(eb)
        match_ms.append(mm)
        ransac_ms.append(rr)

    t1 = time.time()

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    y_true = out_df["label"].astype(int).values
    scores = out_df["score"].astype(float).values
    auc, eer = compute_auc_eer(y_true, scores)

    avg_ms_pair = float(np.mean(pair_total_ms)) if len(pair_total_ms) else 0.0

    meta = {
        "time_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "split": args.split,
        "pairs_csv": str(pairs_path),
        "n_pairs": int(len(out_df)),
        "limit": int(args.limit),
        "manifest": str(manifest_path),
        "ckpt": str(matcher.ckpt_path),
        "config": {
            "patch": int(args.patch),
            "max_kpts": int(args.max_kpts),
            "mask_cov_thr": float(args.mask_cov_thr),
            "max_matches": int(args.max_matches),
            "ransac_thresh": float(args.ransac_thresh),
        },
        "avg_ms_pair": avg_ms_pair,
        "avg_embed_a_total_ms": float(np.mean(embed_a_ms)) if len(embed_a_ms) else 0.0,
        "avg_embed_b_total_ms": float(np.mean(embed_b_ms)) if len(embed_b_ms) else 0.0,
        "avg_match_ms": float(np.mean(match_ms)) if len(match_ms) else 0.0,
        "avg_ransac_ms": float(np.mean(ransac_ms)) if len(ransac_ms) else 0.0,
        "wall_ms_pair": float((t1 - t0) * 1000.0 / max(len(out_df), 1)),
        "cache_size": int(len(getattr(matcher, "_cache", {}))),
        "note": "Caching is enabled inside DedicatedMatcher (embed_b_total may be 0.0 for repeated images).",
        "metrics": {"auc": float(auc), "eer": float(eer)},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Split={args.split} | N={len(out_df)} | AUC={auc:.4f} | EER~{eer:.4f}")
    print(f"Avg pair time: {avg_ms_pair:.2f} ms (cached embeddings help as N grows)")
    print(f"Saved: {out_csv}")
    print(f"Meta : {meta_path}")


if __name__ == "__main__":
    main()
