from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import json
import hashlib

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

from src.fpbench.matchers.baseline_dl import BaselineDL, DLBaselineConfig
from src.fpbench.preprocess.preprocess import PreprocessConfig


def parse_file_uri(p: str) -> Path:
    # Supports: file:/C:/... or normal path
    if p.startswith("file:"):
        p = p[len("file:"):]
        if p.startswith("/"):
            p = p[1:]
    p = p.replace("/", "\\")  # Windows-friendly
    return Path(p)




def normalize_capture(raw: str | None) -> str | None:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip().lower()
    aliases = {
        "plain": "plain",
        "roll": "roll",
        "rolled": "roll",
        "contactless": "contactless",
        "contact-less": "contactless",
        "contact_less": "contactless",
        "contact_based": "contact_based",
        "contact-based": "contact_based",
        "contactbased": "contact_based",
    }
    if not s:
        return None
    return aliases.get(s, s)


def infer_capture_from_path(path: str) -> str | None:
    s = str(path).lower()
    for token, cap in [("contactless", "contactless"), ("contact-less", "contactless"), ("contact_less", "contactless"), ("contact_based", "contact_based"), ("contact-based", "contact_based"), ("contactbased", "contact_based"), ("plain", "plain"), ("roll", "roll")]:
        if token in s:
            return cap
    return None


def resolve_pair_capture(row: pd.Series, side: str) -> str | None:
    keys = [f"capture_{side}", f"capture{side}", f"{side}_capture", "capture"]
    for key in keys:
        if key in row.index:
            cap = normalize_capture(row[key])
            if cap is not None:
                return cap
    path_key = f"path_{side}"
    if path_key in row.index:
        return infer_capture_from_path(str(row[path_key]))
    return None

def compute_auc_eer(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)

    if y_true.size == 0 or scores.size == 0:
        return float("nan"), float("nan")

    valid = np.isfinite(scores)
    if not np.any(valid):
        return float("nan"), float("nan")

    y_true = y_true[valid]
    scores = scores[valid]

    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")

    try:
        auc = float(roc_auc_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
    except ValueError:
        return float("nan"), float("nan")

    fnr = 1 - tpr
    delta = np.abs(fpr - fnr)
    if delta.size == 0 or np.isnan(delta).all():
        return auc, float("nan")

    i = int(np.nanargmin(delta))
    eer = float((fpr[i] + fnr[i]) / 2)
    return auc, eer


def balanced_limit_by_label(df: pd.DataFrame, label_col: str, limit: int) -> pd.DataFrame:
    limit = int(limit)
    if limit <= 0 or len(df) <= limit:
        return df.copy()

    if label_col not in df.columns:
        return df.head(limit).copy()

    labels = [x for x in sorted(df[label_col].dropna().unique().tolist())]
    if len(labels) < 2:
        return df.head(limit).copy()

    per_label = limit // len(labels)
    remainder = limit % len(labels)

    parts = []
    for idx, label in enumerate(labels):
        want = per_label + (1 if idx < remainder else 0)
        part = df[df[label_col] == label].head(want)
        if not part.empty:
            parts.append(part)

    if not parts:
        return df.head(limit).copy()

    out = pd.concat(parts, axis=0)
    if len(out) < limit:
        extra = df.drop(index=out.index, errors="ignore").head(limit - len(out))
        out = pd.concat([out, extra], axis=0)

    return out.sort_index().reset_index(drop=True).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", type=str, help="Output scores CSV. Example: file:/C:/.../scores_val_dl_quick.csv")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--pairs", type=str, default="",
                    help="Optional explicit pairs CSV. If empty resolves the dataset bundle and uses canonical pairs_<split>.csv (flat preferred, nested copy supported).")
    ap.add_argument("--limit", type=int, default=200, help="Quick ROC uses first N pairs. 0 = all.")
    ap.add_argument("--device", type=str, default="", help="cuda|cpu. Empty = auto")
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet18", "resnet50", "vit_base"])
    ap.add_argument("--no_mask", action="store_true", help="Disable ROI/gate masking (ablation)")
    ap.add_argument("--emb_cache_dir", type=str, default="",
                    help="Optional persistent embedding cache directory. Empty disables.")
    ap.add_argument("--cache_write", action="store_true",
                    help="If set, write missing embeddings to disk cache.")
    ap.add_argument("--cache_strip_prefix", type=str, default="",
                    help="Optional prefix to strip from paths before hashing (portability).")
    ap.add_argument("--dataset", type=str, default="nist_sd300b")
    args = ap.parse_args()

    out_path = parse_file_uri(args.out_csv)

    data_dir = ROOT / "data" / "processed" / args.dataset
    pairs_path = parse_file_uri(args.pairs) if args.pairs else (data_dir / f"pairs_{args.split}.csv")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = pd.read_csv(pairs_path)

    # Hard requirement: these columns must exist (Week 3 established this schema)
    for col in ["path_a", "path_b", "label"]:
        if col not in pairs.columns:
            raise ValueError(f"Pairs CSV missing column '{col}'. Columns: {list(pairs.columns)}")

    if args.limit and args.limit > 0:
        pairs = balanced_limit_by_label(pairs, "label", int(args.limit))

    dl_cfg = DLBaselineConfig(backbone=args.backbone, use_mask=(not args.no_mask))
    prep_cfg = PreprocessConfig(target_size=512)

    device = args.device.strip() or None
    model = BaselineDL(dl_cfg=dl_cfg, prep_cfg=prep_cfg, device=device)

    cache_root = parse_file_uri(args.emb_cache_dir) if args.emb_cache_dir else None
    disk_hits = 0
    disk_misses = 0

    # config for cache key (exclude device so cpu/cuda use the same cache)
    cfg_for_key = model.config_dict()
    cfg_for_key.pop("device", None)
    cfg_json = json.dumps(cfg_for_key, sort_keys=True, ensure_ascii=False)

    def canonical_path(p: str) -> str:
        p = str(p).replace("/", "\\")
        if args.cache_strip_prefix:
            pref = args.cache_strip_prefix.replace("/", "\\")
            if p.lower().startswith(pref.lower()):
                p = p[len(pref):]
        else:
            # nice default for your repo layout
            marker = "fingerprint collected data\\"
            i = p.lower().find(marker)
            if i != -1:
                p = p[i + len(marker):]
        return p.lower()

    def cache_file_for(path: str) -> Path:
        assert cache_root is not None
        key_src = canonical_path(path) + "|" + cfg_json
        h = hashlib.sha1(key_src.encode("utf-8")).hexdigest()
        # shard to avoid huge single folder
        return cache_root / h[:2] / f"{h}.npz"

    # Cache embeddings per image path (critical for speed)
    cache: dict[str, tuple[np.ndarray, float]] = {}

    def get_emb(path: str, capture: str | None = None) -> tuple[np.ndarray, float]:
        nonlocal disk_hits, disk_misses

        if path in cache:
            return cache[path]

        # disk cache
        if cache_root is not None:
            cf = cache_file_for(path)
            if cf.exists():
                d = np.load(str(cf))
                emb = d["emb"].astype(np.float32)
                cache[path] = (emb, 0.0)
                disk_hits += 1
                return cache[path]

        # compute
        emb, ms = model.embed_path(path, capture=capture)
        cache[path] = (emb, ms)

        # write
        if cache_root is not None and args.cache_write:
            cf = cache_file_for(path)
            cf.parent.mkdir(parents=True, exist_ok=True)
            tmp = cf.with_suffix(".tmp.npz")
            np.savez_compressed(str(tmp), emb=emb)
            os.replace(str(tmp), str(cf))
            disk_misses += 1

        return cache[path]

    t_all0 = time.perf_counter()

    rows = []
    ms_a_list, ms_b_list, ms_pair_list = [], [], []

    for _, r in pairs.iterrows():
        pa = str(r["path_a"])
        pb = str(r["path_b"])
        y = int(r["label"])

        t0 = time.perf_counter()
        cap_a = resolve_pair_capture(r, "a")
        cap_b = resolve_pair_capture(r, "b")
        ea, ms_a = get_emb(pa, cap_a)
        eb, ms_b = get_emb(pb, cap_b)
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
    meta = {"split": args.split, "pairs_csv": str(pairs_path), "N": int(len(df)), "AUC": float(auc), "EER": float(eer),
            "total_ms": float(total_ms), "avg_ms_pair": float(np.mean(ms_pair_list)) if ms_pair_list else None,
            "model_cfg": model.config_dict(), "root": str(ROOT),
            "disk_cache_dir": str(cache_root) if cache_root is not None else "", "disk_cache_hits": int(disk_hits),
            "disk_cache_misses": int(disk_misses)}
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Split={args.split} | N={len(df)} | AUC={auc:.4f} | EER~{eer:.4f}")
    print(f"Avg pair time: {np.mean(ms_pair_list):.2f} ms (cached embeddings help as N grows)")
    print("Saved:", out_path)
    print("Meta :", meta_path)


if __name__ == "__main__":
    main()