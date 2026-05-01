from __future__ import annotations

import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd

from src.fpbench.preprocess.preprocess import (
    PreprocessConfig,
    load_gray,
    preprocess_image,
    fingerprint_roi_mask,
    suppress_header_and_borders,
    rectangular_gate_mask,
    gftt_keypoints,
    draw_points,
)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def build_final_mask(img_u8: np.ndarray, capture: str) -> np.ndarray:
    roi = fingerprint_roi_mask(img_u8)  # 0/255
    roi = suppress_header_and_borders(roi, top_ratio=0.12, border=10)
    gate = rectangular_gate_mask(img_u8.shape[:2], capture=capture, top_plain=0.18, top_roll=0.05, border=12)
    final = ((roi > 0) & (gate > 0)).astype(np.uint8) * 255

    # Fallback: if mask is empty (rare), just use the gate
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

def make_mosaic(patches: list[np.ndarray], grid: int, scale: int = 2) -> np.ndarray:
    if not patches:
        return np.zeros((64, 64), dtype=np.uint8)

    p = patches[0].shape[0]
    canvas = np.zeros((grid * p, grid * p), dtype=np.uint8)

    for i, patch in enumerate(patches[: grid * grid]):
        rr = i // grid
        cc = i % grid
        canvas[rr * p:(rr + 1) * p, cc * p:(cc + 1) * p] = patch

    if scale != 1:
        canvas = cv2.resize(canvas, (canvas.shape[1] * scale, canvas.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/processed/nist_sd300b/manifest.csv")
    ap.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    ap.add_argument("--n_images", type=int, default=10)
    ap.add_argument("--patch", type=int, default=48)  # must be even (we use r=patch//2)
    ap.add_argument("--max_kpts", type=int, default=800)
    ap.add_argument("--patches_per_image", type=int, default=36)
    ap.add_argument("--out_dir", type=str, default="reports/week06+07+07/patch_sanity")
    args = ap.parse_args()

    patch = int(args.patch)
    if patch % 2 != 0:
        raise ValueError("--patch must be even (e.g., 32, 48).")

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.manifest)
    df = df[df["split"] == args.split].copy()
    if len(df) == 0:
        raise RuntimeError(f"No rows found for split={args.split} in {args.manifest}")

    df = df.head(int(args.n_images)).reset_index(drop=True)

    cfg = PreprocessConfig(target_size=512, clahe_clip=2.0, clahe_grid=(8, 8), blur_ksize=3)

    r = patch // 2
    grid = int(np.ceil(np.sqrt(args.patches_per_image)))

    print(f"[INFO] split={args.split} | n_images={len(df)} | patch={patch} | grid={grid}x{grid}")

    for i, row in df.iterrows():
        path = str(row["path"])
        capture = str(row["capture"])  # plain / roll

        gray = load_gray(path)
        img = preprocess_image(gray, cfg)

        mask = build_final_mask(img, capture=capture)

        pts = gftt_keypoints(img, max_points=int(args.max_kpts), quality=0.01, min_dist=5.0, mask=mask)
        # deterministic order
        pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))] if len(pts) else pts

        pts = filter_pts_inbounds(pts, img.shape[0], img.shape[1], r)

        patches = []
        for (x, y) in pts:
            x, y = int(x), int(y)
            if not mask_majority_ok(mask, x, y, r, thr=0.70):
                continue
            patches.append(extract_patch(img, x, y, patch))
            if len(patches) >= int(args.patches_per_image):
                break

        # Save visuals
        base = f"{i:02d}_{capture}_sid{int(row['subject_id'])}_f{int(row['frgp']):02d}"
        overlay = draw_points(img, pts[: min(len(pts), 500)], radius=2)
        cv2.imwrite(str(out_dir / f"{base}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{base}_mask.png"), mask)

        mosaic = make_mosaic(patches, grid=grid, scale=2)
        cv2.imwrite(str(out_dir / f"{base}_patches.png"), mosaic)

        print(f"[OK] {base} | kpts_inbounds={len(pts)} | patches_saved={len(patches)} | path={path}")

    print(f"\n[DONE] Wrote outputs to: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
