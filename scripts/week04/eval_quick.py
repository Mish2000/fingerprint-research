from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# --- robust project root + imports ---
def project_root() -> Path:
    env = os.environ.get("FPRJ_ROOT", "").strip()
    if env:
        return Path(env)
    # scripts/week04/eval_quick.py -> parents[2] == repo root
    return Path(__file__).resolve().parents[2]

ROOT = project_root()
sys.path.insert(0, str(ROOT))

from src.baseline_dl import BaselineDL, DLBaselineConfig
from src.preprocess import PreprocessConfig


def parse_file_uri(p: str) -> Path:
    # Supports: file:/C:/... or normal path
    if p.startswith("file:"):
        p = p[len("file:"):]
        if p.startswith("/"):
            p = p[1:]
    p = p.replace("/", "\\")  # Windows-friendly
    return Path(p)


def compute_auc_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[i] + fnr[i]) / 2)
    return auc, eer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", type=str, help="Output scores CSV. Example: file:/C:/.../scores_val_dl_quick.csv")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--pairs", type=str, default="",
                    help="Optional explicit pairs CSV. If empty uses data/processed/nist_sd300b/pairs_<split>.csv")
    ap.add_argument("--limit", type=int, default=200, help="Quick ROC uses first N pairs. 0 = all.")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu. Empty = auto")
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50"])
    ap.add_argument("--no_mask", action="store_true", help="Disable ROI/gate masking (ablation)")
    args = ap.parse_args()

    out_path = parse_file_uri(args.out_csv)

    data_dir = ROOT / "data" / "processed" / "nist_sd300b"
    pairs_path = Path(args.pairs) if args.pairs else (data_dir / f"pairs_{args.split}.csv")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = pd.read_csv(pairs_path)

    # Hard requirement: these columns must exist (Week 3 established this schema)
    for col in ["path_a", "path_b", "label"]:
        if col not in pairs.columns:
            raise ValueError(f"Pairs CSV missing column '{col}'. Columns: {list(pairs.columns)}")

    if args.limit and args.limit > 0:
        pairs = pairs.head(int(args.limit)).copy()

    dl_cfg = DLBaselineConfig(backbone=args.backbone, use_mask=(not args.no_mask))
    prep_cfg = PreprocessConfig(target_size=512)

    device = args.device.strip() or None
    model = BaselineDL(dl_cfg=dl_cfg, prep_cfg=prep_cfg, device=device)

    # Cache embeddings per image path (critical for speed)
    cache: dict[str, tuple[np.ndarray, float]] = {}

    def get_emb(path: str) -> tuple[np.ndarray, float]:
        if path in cache:
            return cache[path]
        emb, ms = model.embed_path(path)
        cache[path] = (emb, ms)
        return emb, ms

    t_all0 = time.perf_counter()

    rows = []
    ms_a_list, ms_b_list, ms_pair_list = [], [], []

    for _, r in pairs.iterrows():
        pa = str(r["path_a"])
        pb = str(r["path_b"])
        y = int(r["label"])

        t0 = time.perf_counter()
        ea, ms_a = get_emb(pa)
        eb, ms_b = get_emb(pb)
        score = model.cosine(ea, eb)
        ms_pair = (time.perf_counter() - t0) * 1000.0

        ms_a_list.append(ms_a)
        ms_b_list.append(ms_b)
        ms_pair_list.append(ms_pair)

        rows.append({
            "label": y,
            "split": args.split,
            "path_a": pa,
            "path_b": pb,
            "score": float(score),
            "ms_embed_a": float(ms_a),
            "ms_embed_b": float(ms_b),
            "ms_pair_total": float(ms_pair),
            "backbone": args.backbone,
            "use_mask": int(not args.no_mask),
            "device": model.device,
        })

    df = pd.DataFrame(rows)
    auc, eer = compute_auc_eer(df["label"].values, df["score"].values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    total_ms = (time.perf_counter() - t_all0) * 1000.0

    # Save a small run metadata JSON next to CSV (helps reproducibility)
    meta = {
        "split": args.split,
        "pairs_csv": str(pairs_path),
        "N": int(len(df)),
        "AUC": float(auc),
        "EER": float(eer),
        "total_ms": float(total_ms),
        "avg_ms_pair": float(np.mean(ms_pair_list)) if ms_pair_list else None,
        "model_cfg": model.config_dict(),
        "root": str(ROOT),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Split={args.split} | N={len(df)} | AUC={auc:.4f} | EER~{eer:.4f}")
    print(f"Avg pair time: {np.mean(ms_pair_list):.2f} ms (cached embeddings help as N grows)")
    print("Saved:", out_path)
    print("Meta :", meta_path)


if __name__ == "__main__":
    main()
