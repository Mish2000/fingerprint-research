from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
import json
import time
from dataclasses import asdict
from typing import Iterator, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preprocess import (
    PreprocessConfig,
    load_gray,
    preprocess_image,
    fingerprint_roi_mask,
    suppress_header_and_borders,
    rectangular_gate_mask,
    gftt_keypoints,
)

# -------------------------
# Patch extraction (frozen)
# -------------------------

def build_final_mask(img_u8: np.ndarray, capture: str) -> np.ndarray:
    roi = fingerprint_roi_mask(img_u8)
    roi = suppress_header_and_borders(roi, top_ratio=0.12, border=10)
    gate = rectangular_gate_mask(img_u8.shape[:2], capture=capture, top_plain=0.18, top_roll=0.05, border=12)
    final = ((roi > 0) & (gate > 0)).astype(np.uint8) * 255
    if (final > 0).mean() < 0.005:
        final = (gate > 0).astype(np.uint8) * 255
    return final

def filter_pts_inbounds(pts_xy: np.ndarray, h: int, w: int, r: int) -> np.ndarray:
    if len(pts_xy) == 0:
        return pts_xy
    x = pts_xy[:, 0]
    y = pts_xy[:, 1]
    ok = (x >= r) & (x < w - r) & (y >= r) & (y < h - r)
    return pts_xy[ok]

def mask_majority_ok(mask_u8: np.ndarray, x: int, y: int, r: int, thr: float = 0.70) -> bool:
    patch_m = mask_u8[y - r:y + r, x - r:x + r]
    if patch_m.size == 0:
        return False
    return (patch_m > 0).mean() >= thr

def extract_patch(img_u8: np.ndarray, x: int, y: int, patch: int) -> np.ndarray:
    r = patch // 2
    return img_u8[y - r:y + r, x - r:x + r].copy()

# -------------------------
# Augmentations (OpenCV)
# -------------------------

def aug_patch(patch_u8: np.ndarray, rng: np.random.Generator) -> torch.Tensor:
    """
    Input: uint8 (H,W) 0..255
    Output: float tensor (1,H,W) roughly normalized to 0..1
    """
    x = patch_u8.astype(np.float32) / 255.0

    # Random small rotation + translation
    h, w = x.shape
    ang = float(rng.uniform(-15.0, 15.0))
    tx = float(rng.uniform(-3.0, 3.0))
    ty = float(rng.uniform(-3.0, 3.0))

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    x = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    # Contrast / brightness jitter (mild)
    c = float(rng.uniform(0.80, 1.20))
    b = float(rng.uniform(-0.08, 0.08))
    x = x * c + b

    # Gaussian noise (mild)
    if rng.random() < 0.9:
        sigma = float(rng.uniform(0.0, 0.04))
        if sigma > 0:
            x = x + rng.normal(0.0, sigma, size=x.shape).astype(np.float32)

    # Occasional blur (very mild)
    if rng.random() < 0.2:
        x = cv2.GaussianBlur(x, (3, 3), 0)

    x = np.clip(x, 0.0, 1.0)
    t = torch.from_numpy(x).unsqueeze(0)  # (1,H,W)
    return t

# -------------------------
# Dataset: stream patches
# -------------------------

class PatchPairStream(torch.utils.data.IterableDataset):
    def __init__(
        self,
        manifest_csv: str,
        split: str,
        patch: int,
        max_kpts: int,
        patches_per_image: int,
        seed: int,
        limit_images: int = 0,
    ):
        super().__init__()
        self.manifest_csv = manifest_csv
        self.split = split
        self.patch = patch
        self.max_kpts = max_kpts
        self.patches_per_image = patches_per_image
        self.seed = seed
        self.limit_images = limit_images

        df = pd.read_csv(self.manifest_csv)
        df = df[df["split"] == self.split].copy()
        if self.limit_images and self.limit_images > 0:
            df = df.head(self.limit_images)
        if len(df) == 0:
            raise RuntimeError(f"No rows for split={split} in {manifest_csv}")
        self.df = df.reset_index(drop=True)

        self.cfg = PreprocessConfig(target_size=512, clahe_clip=2.0, clahe_grid=(8, 8), blur_ksize=3)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        base_seed = self.seed + 1337 * worker_id
        rng = np.random.default_rng(base_seed)

        patch = int(self.patch)
        r = patch // 2

        # Stream forever
        while True:
            order = rng.permutation(len(self.df))
            for idx in order:
                row = self.df.iloc[int(idx)]
                path = str(row["path"])
                capture = str(row["capture"])

                gray = load_gray(path)
                img = preprocess_image(gray, self.cfg)
                mask = build_final_mask(img, capture=capture)

                pts = gftt_keypoints(img, max_points=int(self.max_kpts), quality=0.01, min_dist=5.0, mask=mask)
                pts = filter_pts_inbounds(pts, img.shape[0], img.shape[1], r)

                if len(pts) == 0:
                    continue

                # keep only pts whose patch is mostly in mask
                valid = []
                for (x, y) in pts:
                    x, y = int(x), int(y)
                    if mask_majority_ok(mask, x, y, r, thr=0.70):
                        valid.append((x, y))
                if len(valid) < 8:
                    continue

                # yield multiple patches from this single image (speed)
                for _ in range(int(self.patches_per_image)):
                    x, y = valid[int(rng.integers(0, len(valid)))]
                    p = extract_patch(img, x, y, patch)
                    v1 = aug_patch(p, rng)
                    v2 = aug_patch(p, rng)
                    yield v1, v2

# -------------------------
# Model (small CNN + projection head)
# -------------------------

class SmallEncoder(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24x24

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12x12

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # 1x1
        )
        self.fc = nn.Linear(128, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

class SimCLRModel(nn.Module):
    def __init__(self, emb_dim: int = 256, proj_dim: int = 128):
        super().__init__()
        self.encoder = SmallEncoder(emb_dim=emb_dim)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        h = F.normalize(h, dim=1)
        return h, z

# -------------------------
# NT-Xent loss
# -------------------------

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    """
    z1,z2: (B,D) normalized
    Force float32 logits to avoid AMP float16 overflow on masking.
    """
    z1 = z1.float()
    z2 = z2.float()

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B,D)
    sim = (z @ z.t()) / float(temp)  # (2B,2B) float32

    diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, torch.finfo(sim.dtype).min)  # safe in float32

    pos = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, pos)
    return loss


# -------------------------
# Train
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/processed/nist_sd300b/manifest.csv")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--patch", type=int, default=48)
    ap.add_argument("--max_kpts", type=int, default=800)
    ap.add_argument("--patches_per_image", type=int, default=32)

    ap.add_argument("--limit_images", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--out_dir", type=str, default="reports/week06+07+07/patch_descriptor/run_smoke")
    args = ap.parse_args()

    if args.patch % 2 != 0:
        raise ValueError("--patch must be even (e.g., 32, 48).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save meta
    meta = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = PatchPairStream(
        manifest_csv=args.manifest,
        split=args.split,
        patch=args.patch,
        max_kpts=args.max_kpts,
        patches_per_image=args.patches_per_image,
        seed=args.seed,
        limit_images=args.limit_images,
    )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    it = iter(loader)

    model = SimCLRModel(emb_dim=256, proj_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    log_path = out_dir / "train_log.csv"
    if not log_path.exists():
        log_path.write_text("step,loss\n", encoding="utf-8")

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        x1, x2 = next(it)
        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = nt_xent(z1, z2, temp=float(args.temp))

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        # log
        if step % 20 == 0 or step == 1:
            dt = time.time() - t0
            print(f"[step {step:04d}/{args.steps}] loss={loss.item():.4f}  time={dt:.1f}s")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{step},{loss.item():.6f}\n")

        # checkpoint occasionally
        if step % 200 == 0 or step == args.steps:
            ckpt = {
                "step": step,
                "model": model.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "ckpt_last.pth")

    print(f"\n[DONE] Wrote run to: {out_dir.resolve()}")
    print(f"[DONE] Checkpoint: {str((out_dir / 'ckpt_last.pth').resolve())}")

if __name__ == "__main__":
    main()
