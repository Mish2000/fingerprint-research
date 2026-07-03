#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:  # pragma: no cover
    roc_auc_score = None
    roc_curve = None

Image.MAX_IMAGE_PIXELS = None
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".wsq"}
METHOD = "deep_pair_reranker_fast_ddp"


# -----------------------------
# Model: must match train_fast_pair_ddp.py
# -----------------------------
class ConvEncoder(nn.Module):
    def __init__(self, width: int = 32, embedding_dim: int = 512):
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.SiLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            block(1, width),
            block(width, width * 2),
            block(width * 2, width * 4),
            block(width * 4, width * 8),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width * 8, embedding_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairModel(nn.Module):
    def __init__(self, width: int = 32, embedding_dim: int = 512, hidden_dim: int = 768):
        super().__init__()
        self.encoder = ConvEncoder(width=width, embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 4, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Match the optimized training forward: one encoder call for both sides.
        x = torch.cat([x1, x2], dim=0)
        e = self.encoder(x)
        e1, e2 = e.chunk(2, dim=0)
        z = torch.cat([e1, e2, torch.abs(e1 - e2), e1 * e2], dim=1)
        return self.head(z).squeeze(1)


# -----------------------------
# Path and image IO
# -----------------------------
def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def normalize_path_string(value: str) -> str:
    s = str(value).strip().replace("\\", "/")
    marker = "fingerprint-research/"
    if marker in s:
        s = s.split(marker, 1)[1]
    low = s.lower()
    marker2 = "data/raw/"
    if marker2 in low:
        idx = low.index(marker2)
        s = s[idx:]
    return s.lstrip("/")


def build_path_index(repo_root: Path) -> Dict[str, Path]:
    raw_root = repo_root / "data" / "raw"
    if not raw_root.exists():
        raise FileNotFoundError(f"Missing raw root: {raw_root}")
    index: Dict[str, Path] = {}
    count = 0
    for p in raw_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        resolved = p.resolve()
        rel_raw = p.relative_to(raw_root).as_posix()
        rel_repo = p.relative_to(repo_root).as_posix()
        keys = {
            p.name, p.name.lower(),
            rel_raw, rel_raw.lower(),
            rel_repo, rel_repo.lower(),
            f"data/raw/{rel_raw}", f"data/raw/{rel_raw}".lower(),
        }
        parts = Path(rel_raw).parts
        for k in range(2, min(6, len(parts)) + 1):
            suffix = "/".join(parts[-k:])
            keys.add(suffix)
            keys.add(suffix.lower())
        for key in keys:
            index.setdefault(key, resolved)
        count += 1
    print(f"[index] Indexed {count} raw image files under {raw_root}", flush=True)
    return index


def resolve_path(value: str, repo_root: Path, path_index: Dict[str, Path]) -> Path:
    s = normalize_path_string(value)
    candidates = [Path(s), repo_root / s, repo_root / "data" / s, repo_root / "data" / "raw" / s]
    for c in candidates:
        if c.exists():
            return c.resolve()
    for key in [s, s.lower(), Path(s).name, Path(s).name.lower()]:
        if key in path_index:
            return path_index[key]
    raise FileNotFoundError(f"Could not resolve image path: original={value!r}, normalized={s!r}")


def load_image_u8(path: Path, size: int) -> torch.Tensor:
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            im = im.resize((size, size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.uint8).copy()
            return torch.from_numpy(arr).unsqueeze(0).contiguous()
    except Exception as pil_err:
        try:
            import cv2
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise RuntimeError("cv2.imread returned None")
            arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
            arr = arr.astype(np.uint8, copy=False)
            return torch.from_numpy(arr).unsqueeze(0).contiguous()
        except Exception as cv_err:
            raise RuntimeError(f"Failed reading image: {path}\nPIL error: {pil_err}\ncv2 error: {cv_err}")


# -----------------------------
# Pair tables and cache
# -----------------------------
def pair_csv_path(repo_root: Path, dataset: str, split: str) -> Path:
    ds_root = repo_root / "data" / "manifests" / dataset
    candidates = [ds_root / f"pairs_{split}.csv", ds_root / "pairs" / f"pairs_{split}.csv"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing pairs CSV for {dataset}/{split}. Tried: {[str(p) for p in candidates]}")


def read_pair_table(repo_root: Path, dataset: str, split: str, *, max_pairs: int | None = None) -> pd.DataFrame:
    path = pair_csv_path(repo_root, dataset, split)
    df = pd.read_csv(path)
    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
    if "path_a" not in df.columns or "path_b" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected path_a,path_b,label in {path}; found {list(df.columns)}")
    df = df.copy()
    df["dataset"] = dataset
    df["split"] = split
    df["_pairs_csv"] = str(path)
    if "pair_id" not in df.columns:
        df["pair_id"] = [f"{dataset}_{split}_{i:06d}" for i in range(len(df))]
    return df


def build_eval_cache(repo_root: Path, tables: Dict[Tuple[str, str], pd.DataFrame], input_size: int, cache_file: Path | None):
    if cache_file is not None and cache_file.exists():
        print(f"[cache] Reusing eval cache: {cache_file}", flush=True)
        return safe_torch_load(cache_file)

    path_index = build_path_index(repo_root)
    all_paths: List[Path] = []
    resolved_tables: Dict[Tuple[str, str], pd.DataFrame] = {}

    for key, df in tables.items():
        out = df.copy()
        resolved_a = []
        resolved_b = []
        for i, row in out.iterrows():
            if i % 2000 == 0:
                print(f"[resolve] {key} {i}/{len(out)}", flush=True)
            pa = resolve_path(row["path_a"], repo_root, path_index)
            pb = resolve_path(row["path_b"], repo_root, path_index)
            resolved_a.append(str(pa))
            resolved_b.append(str(pb))
            all_paths.extend([pa, pb])
        out["_resolved_path_a"] = resolved_a
        out["_resolved_path_b"] = resolved_b
        resolved_tables[key] = out

    unique_paths = sorted({str(p) for p in all_paths})
    path_to_idx = {p: i for i, p in enumerate(unique_paths)}
    print(f"[cache] Unique eval images: {len(unique_paths)}", flush=True)
    print(f"[cache] Tensor estimate uint8: {len(unique_paths) * input_size * input_size / 1024**3:.2f} GiB", flush=True)
    images = torch.empty((len(unique_paths), 1, input_size, input_size), dtype=torch.uint8)
    t0 = time.time()
    for i, p in enumerate(unique_paths):
        if i % 500 == 0:
            print(f"[cache] Loading eval image {i}/{len(unique_paths)} elapsed={time.time() - t0:.1f}s", flush=True)
        images[i] = load_image_u8(Path(p), input_size)

    pair_indices: Dict[Tuple[str, str], torch.Tensor] = {}
    for key, df in resolved_tables.items():
        idx = torch.empty((len(df), 2), dtype=torch.long)
        for i, row in df.iterrows():
            idx[i, 0] = path_to_idx[row["_resolved_path_a"]]
            idx[i, 1] = path_to_idx[row["_resolved_path_b"]]
        pair_indices[key] = idx

    payload = {"images": images, "paths": unique_paths, "tables": resolved_tables, "pair_indices": pair_indices, "input_size": input_size}
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, cache_file)
        meta = {"cache_file": str(cache_file), "num_images": len(unique_paths), "input_size": input_size,
                "splits": {f"{k[0]}/{k[1]}": len(v) for k, v in resolved_tables.items()}}
        cache_file.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[cache] Wrote eval cache: {cache_file}", flush=True)
    return payload


class EvalPairDataset(Dataset):
    def __init__(self, images: torch.Tensor, pair_idx: torch.Tensor):
        self.images = images
        self.pair_idx = pair_idx

    def __len__(self) -> int:
        return int(self.pair_idx.shape[0])

    def __getitem__(self, i: int):
        a = self.pair_idx[i, 0]
        b = self.pair_idx[i, 1]
        return self.images[a], self.images[b], i


# -----------------------------
# Metrics
# -----------------------------
def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    if roc_auc_score is None or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def eer_score(labels: np.ndarray, scores: np.ndarray) -> float:
    if roc_curve is None or len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0)


def select_threshold_from_val_negatives(labels: np.ndarray, scores: np.ndarray, target_far: float) -> float:
    labels = labels.astype(int)
    neg = scores[labels == 0]
    if len(neg) == 0:
        raise ValueError("No negative labels available for threshold selection")
    # Candidate thresholds are exact negative scores plus a strict threshold above max.
    candidates = np.unique(neg)
    candidates = np.concatenate([[np.nextafter(float(np.max(neg)), float("inf"))], candidates])
    valid = []
    for thr in candidates:
        far = float(np.mean(neg >= thr))
        if far <= target_far + 1e-12:
            valid.append(float(thr))
    if not valid:
        return float(np.nextafter(float(np.max(neg)), float("inf")))
    # Most permissive = lowest threshold that still satisfies the FAR target on VAL negatives.
    return float(min(valid))


def confusion_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    labels = labels.astype(int)
    pred = (scores >= threshold).astype(int)
    pos = labels == 1
    neg = labels == 0
    ta = int(np.sum(pred[pos] == 1))
    fr = int(np.sum(pred[pos] == 0))
    fa = int(np.sum(pred[neg] == 1))
    tr = int(np.sum(pred[neg] == 0))
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    tar = ta / n_pos if n_pos else float("nan")
    far = fa / n_neg if n_neg else float("nan")
    frr = fr / n_pos if n_pos else float("nan")
    return {"n_pairs": int(len(labels)), "n_pos": n_pos, "n_neg": n_neg, "TA": ta, "FR": fr, "FA": fa, "TR": tr,
            "TAR": tar, "FAR": far, "FRR": frr}


@torch.no_grad()
def score_table(model: nn.Module, images: torch.Tensor, pair_idx: torch.Tensor, *, device: torch.device,
                batch_size: int, num_workers: int, amp: bool) -> Tuple[np.ndarray, np.ndarray]:
    ds = EvalPairDataset(images, pair_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))
    logits_out = np.empty(len(ds), dtype=np.float32)
    model.eval()
    for x1, x2, idx in loader:
        x1 = x1.to(device, non_blocking=True).float().div_(255.0).contiguous(memory_format=torch.channels_last)
        x2 = x2.to(device, non_blocking=True).float().div_(255.0).contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            logits = model(x1, x2)
        logits_out[idx.numpy()] = logits.detach().float().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits_out))
    return logits_out, probs.astype(np.float32)


def save_scores(df: pd.DataFrame, logits: np.ndarray, probs: np.ndarray, out_csv: Path, checkpoint: Path, input_size: int) -> None:
    out = pd.DataFrame()
    for col in ["dataset", "split", "pair_id", "label", "subject_a", "subject_b", "finger_position", "frgp", "path_a", "path_b"]:
        if col in df.columns:
            out[col] = df[col]
    out.insert(0, "method", METHOD)
    if "dataset" not in out.columns:
        out["dataset"] = df["dataset"]
    if "split" not in out.columns:
        out["split"] = df["split"]
    out["label"] = df["label"].astype(int).values
    out["score"] = probs
    out["logit"] = logits
    out["probability"] = probs
    out["score_semantics"] = "match_probability_sigmoid_logit"
    out["higher_is_more_similar"] = True
    out["model_checkpoint"] = str(checkpoint)
    out["input_size"] = int(input_size)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--outdir", default="artifacts/reports/benchmark/deep_pair_reranker_fast_ddp_scores")
    ap.add_argument("--eval-cache-file", default="artifacts/cache/fp_pair_eval_cache_384.pt")
    ap.add_argument("--datasets", default="nist_sd300b,nist_sd300c")
    ap.add_argument("--splits", default="train")
    ap.add_argument("--target-fars", default="")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--max-pairs-per-split", type=int, default=None)
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    checkpoint = Path(args.checkpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    scores_dir = outdir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    ckpt = safe_torch_load(checkpoint)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    width = int(ckpt_args.get("width", 32))
    embedding_dim = int(ckpt_args.get("embedding_dim", 512))
    hidden_dim = int(ckpt_args.get("hidden_dim", 768))
    input_size = int(ckpt_args.get("input_size", 384))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    torch.backends.cudnn.benchmark = True
    model = PairModel(width=width, embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[model] loaded {checkpoint} model_type={ckpt.get('model_type')} epoch={ckpt.get('epoch')} metrics={ckpt.get('metrics')}", flush=True)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    target_fars = [float(x.strip()) for x in args.target_fars.split(",") if x.strip()]

    tables: Dict[Tuple[str, str], pd.DataFrame] = {}
    for ds in datasets:
        for split in splits:
            tables[(ds, split)] = read_pair_table(repo_root, ds, split, max_pairs=args.max_pairs_per_split)
            print(f"[pairs] {ds}/{split}: {len(tables[(ds, split)])} rows", flush=True)

    eval_cache_file = Path(args.eval_cache_file) if args.eval_cache_file else None
    payload = build_eval_cache(repo_root, tables, input_size, eval_cache_file)
    images = payload["images"]
    resolved_tables = payload["tables"]
    pair_indices = payload["pair_indices"]

    score_frames: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key in [(ds, split) for ds in datasets for split in splits]:
        ds, split = key
        print(f"[score] {ds}/{split}", flush=True)
        t0 = time.time()
        logits, probs = score_table(model, images, pair_indices[key], device=device, batch_size=args.batch_size,
                                    num_workers=args.num_workers, amp=args.amp)
        elapsed = time.time() - t0
        df = resolved_tables[key].copy()
        out_csv = scores_dir / f"scores_{ds}_{METHOD}_{split}.csv"
        save_scores(df, logits, probs, out_csv, checkpoint, input_size)
        sf = pd.read_csv(out_csv)
        score_frames[key] = sf
        print(f"[score] wrote {out_csv} rows={len(sf)} elapsed={elapsed:.1f}s", flush=True)

    # Always write a lightweight score summary for every scored split.
    summary_rows = []
    for key, sf in score_frames.items():
        ds, split = key
        labels = sf["label"].astype(int).to_numpy()
        scores = sf["score"].astype(float).to_numpy()
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        summary_rows.append({
            "method": METHOD,
            "dataset": ds,
            "split": split,
            "auc": auc_score(labels, scores),
            "eer": eer_score(labels, scores),
            "n_pairs": int(len(sf)),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "scores_csv": str(scores_dir / f"scores_{ds}_{METHOD}_{split}.csv"),
            "checkpoint": str(checkpoint),
        })

    score_summary_df = pd.DataFrame(summary_rows)
    score_summary_csv = outdir / "score_summary.csv"
    score_summary_df.to_csv(score_summary_csv, index=False)

    # Optional biometric threshold metrics. These require a VAL split for calibration.
    target_fars = [float(x.strip()) for x in args.target_fars.split(",") if x.strip()]
    threshold_rows = []
    metric_rows = []
    if target_fars:
        for ds in datasets:
            if (ds, "val") not in score_frames:
                print(f"[metrics] Skipping threshold metrics for {ds}: split 'val' was not scored.", flush=True)
                continue
            val = score_frames[(ds, "val")]
            val_labels = val["label"].astype(int).to_numpy()
            val_scores = val["score"].astype(float).to_numpy()
            for target_far in target_fars:
                thr = select_threshold_from_val_negatives(val_labels, val_scores, target_far)
                threshold_rows.append({"method": METHOD, "dataset": ds, "target_far": target_far, "threshold": thr,
                                       "calibration_split": "val", "calibration_negatives": int(np.sum(val_labels == 0))})
                for split in splits:
                    sf = score_frames[(ds, split)]
                    labels = sf["label"].astype(int).to_numpy()
                    scores = sf["score"].astype(float).to_numpy()
                    conf = confusion_at_threshold(labels, scores, thr)
                    metric_rows.append({
                        "method": METHOD,
                        "dataset": ds,
                        "split": split,
                        "target_far": target_far,
                        "threshold": thr,
                        "auc": auc_score(labels, scores),
                        "eer": eer_score(labels, scores),
                        **conf,
                        "scores_csv": str(scores_dir / f"scores_{ds}_{METHOD}_{split}.csv"),
                        "checkpoint": str(checkpoint),
                    })

    metrics_df = pd.DataFrame(metric_rows)
    thresholds_df = pd.DataFrame(threshold_rows)
    metrics_csv = outdir / "plain_roll_final_metrics.csv"
    thresholds_csv = outdir / "plain_roll_final_thresholds.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    thresholds_df.to_csv(thresholds_csv, index=False)

    summary_md = outdir / "plain_roll_final_summary.md"
    with summary_md.open("w", encoding="utf-8") as f:
        f.write(f"# {METHOD} score generation / benchmark\n\n")
        f.write(f"Checkpoint: `{checkpoint}`\n\n")
        f.write(f"Model type: `{ckpt.get('model_type')}`; epoch: `{ckpt.get('epoch')}`; checkpoint metrics: `{ckpt.get('metrics')}`\n\n")
        f.write("## Score summary\n\n")
        f.write(score_summary_df.to_markdown(index=False))
        if len(metrics_df):
            f.write("\n\n## Threshold metrics\n\n")
            f.write(metrics_df.to_markdown(index=False))
        if len(thresholds_df):
            f.write("\n\n## Thresholds\n\n")
            f.write(thresholds_df.to_markdown(index=False))
        f.write("\n")

    manifest = {
        "method": METHOD,
        "checkpoint": str(checkpoint),
        "checkpoint_model_type": ckpt.get("model_type"),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": ckpt.get("metrics"),
        "checkpoint_args": ckpt_args,
        "datasets": datasets,
        "splits": splits,
        "target_fars": target_fars,
        "input_size": input_size,
        "scores_dir": str(scores_dir),
        "score_summary_csv": str(score_summary_csv),
        "metrics_csv": str(metrics_csv),
        "thresholds_csv": str(thresholds_csv),
    }
    (outdir / "plain_roll_final_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("[done]", flush=True)
    print(metrics_df, flush=True)
    print("metrics:", metrics_csv, flush=True)
    print("thresholds:", thresholds_csv, flush=True)
    print("summary:", summary_md, flush=True)


if __name__ == "__main__":
    main()
