from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ---- Make sure imports work even if you run from anywhere ----
PROJECT_ROOT = Path(r"/")
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    PreprocessConfig,
    load_gray,
    preprocess_image,
    draw_points,
    fingerprint_roi_mask,   # your ROI function
)

# ---- Paths ----
DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "nist_sd300b"
OUT_DIR = PROJECT_ROOT / "reports" / "visual_samples" / "week02"


def rectangular_gate_mask(shape_hw, capture: str, top_plain: float = 0.18, top_roll: float = 0.05, border: int = 12) -> np.ndarray:
    """
    Deterministic gate that removes top header strip + borders.
    Works even if ROI segmentation is imperfect.
    """
    h, w = shape_hw
    m = np.zeros((h, w), dtype=np.uint8)

    top = int(round((top_plain if capture == "plain" else top_roll) * h))
    y0 = top
    y1 = h - border
    x0 = border
    x1 = w - border

    if y1 > y0 and x1 > x0:
        m[y0:y1, x0:x1] = 255
    return m


def harris_keypoints_masked(img_u8: np.ndarray, mask: np.ndarray,
                            max_points: int = 500,
                            block_size: int = 2, ksize: int = 3, k: float = 0.04,
                            thresh_rel: float = 0.01) -> np.ndarray:
    """
    Harris corner response with strict masking.
    Returns (N,2) points in (x,y).
    """
    img_f = np.float32(img_u8) / 255.0
    resp = cv2.cornerHarris(img_f, blockSize=block_size, ksize=ksize, k=k)

    # enforce mask
    if mask is not None:
        resp = resp.copy()
        resp[mask == 0] = 0.0

    if resp.max() <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    thr = thresh_rel * resp.max()
    m = resp > thr

    # NMS
    resp_dil = cv2.dilate(resp, None)
    nms = (resp == resp_dil) & m

    ys, xs = np.where(nms)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    scores = resp[ys, xs]
    idx = np.argsort(scores)[::-1][:max_points]
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)
    return pts


def gftt_keypoints_masked(img_u8: np.ndarray, mask: np.ndarray,
                          max_points: int = 500,
                          quality: float = 0.01,
                          min_dist: float = 5.0) -> np.ndarray:
    """
    GFTT (Shi-Tomasi) with OpenCV mask parameter.
    Returns (N,2) points in (x,y).
    """
    corners = cv2.goodFeaturesToTrack(
        img_u8,
        maxCorners=max_points,
        qualityLevel=quality,
        minDistance=min_dist,
        mask=mask,                 # <-- crucial
        useHarrisDetector=False
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.int32)
    return corners.reshape(-1, 2).astype(np.int32)


def inside_ratio(pts_xy: np.ndarray, mask: np.ndarray) -> float:
    if len(pts_xy) == 0:
        return 0.0
    xs = np.clip(pts_xy[:, 0], 0, mask.shape[1] - 1)
    ys = np.clip(pts_xy[:, 1], 0, mask.shape[0] - 1)
    return float((mask[ys, xs] > 0).mean())


def pts_in_top_strip(pts_xy: np.ndarray, top_px: int) -> int:
    if len(pts_xy) == 0:
        return 0
    return int((pts_xy[:, 1] < top_px).sum())


def main():
    # ---- Settings ----
    n_plain = 25
    n_roll = 25
    seed_plain = 42
    seed_roll = 43

    cfg = PreprocessConfig(target_size=512, clahe_clip=2.0, blur_ksize=3)

    # keypoint params
    harris_params = {"max_points": 500, "thresh_rel": 0.01, "block_size": 2, "ksize": 3, "k": 0.04}
    gftt_params = {"max_points": 500, "quality": 0.01, "min_dist": 5.0}

    # ROI policy thresholds
    ROI_TOO_BIG = 0.70
    ROI_TOO_SMALL = 0.10
    FINAL_TOO_SMALL = 0.12

    # ---- Prepare folders ----
    (OUT_DIR / "before").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "after_preprocess").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "roi_mask").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "harris_overlay").mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "gftt_overlay").mkdir(parents=True, exist_ok=True)

    manifest_path = DATASET_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found at: {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Sample balanced plain/roll
    df_plain = df[df["capture"] == "plain"].sample(n=n_plain, random_state=seed_plain)
    df_roll = df[df["capture"] == "roll"].sample(n=n_roll, random_state=seed_roll)
    sample = pd.concat([df_plain, df_roll], ignore_index=True)

    summary = {
        "n": int(len(sample)),
        "cfg": cfg.__dict__,
        "roi_policy": {
            "roi_too_big": ROI_TOO_BIG,
            "roi_too_small": ROI_TOO_SMALL,
            "final_too_small": FINAL_TOO_SMALL
        },
        "harris": harris_params,
        "gftt": gftt_params,
        "items": []
    }

    print(f"Running Week-2 examples on {len(sample)} images...")
    print(f"Output dir: {OUT_DIR}")

    for i, row in sample.iterrows():
        path = row["path"]
        sid = int(row["subject_id"])
        frgp = int(row["frgp"])
        cap = row["capture"]
        split = row.get("split", "na")

        stem = f"{i:03d}_sid{sid:04d}_f{frgp:02d}_{cap}_{split}"

        before = load_gray(path)
        after = preprocess_image(before, cfg)

        # --- Build masks ---
        mask_roi = fingerprint_roi_mask(after)  # 0/255 expected
        mask_gate = rectangular_gate_mask(after.shape[:2], capture=cap, top_plain=0.18, top_roll=0.05, border=12)

        roi_white = float((mask_roi > 0).mean())
        roi_bad = (roi_white > ROI_TOO_BIG) or (roi_white < ROI_TOO_SMALL)

        if roi_bad:
            mask_final = mask_gate
        else:
            mask_final = cv2.bitwise_and(mask_roi, mask_gate)

        final_white = float((mask_final > 0).mean())
        if final_white < FINAL_TOO_SMALL:
            mask_final = mask_gate
            final_white = float((mask_final > 0).mean())

        # top strip height used for debug
        top_px = int(round((0.18 if cap == "plain" else 0.05) * after.shape[0]))
        mask_top_mean = int(mask_final[:top_px, :].mean())  # should be 0 for plain

        # --- Keypoints (masked) ---
        h_pts = harris_keypoints_masked(
            after, mask_final,
            max_points=harris_params["max_points"],
            thresh_rel=harris_params["thresh_rel"],
            block_size=harris_params["block_size"],
            ksize=harris_params["ksize"],
            k=harris_params["k"],
        )

        g_pts = gftt_keypoints_masked(
            after, mask_final,
            max_points=gftt_params["max_points"],
            quality=gftt_params["quality"],
            min_dist=gftt_params["min_dist"],
        )

        # Debug proof: points in top strip should be ~0 for plain
        H_top = pts_in_top_strip(h_pts, top_px)
        G_top = pts_in_top_strip(g_pts, top_px)

        print(
            f"{stem} | cap={cap} roi_white={roi_white:.3f} roi_bad={roi_bad} "
            f"final_white={final_white:.3f} mask_top_mean={mask_top_mean} "
            f"H={len(h_pts)} H_top={H_top} | G={len(g_pts)} G_top={G_top}"
        )

        # --- Save outputs ---
        cv2.imwrite(str(OUT_DIR / "before" / f"{stem}.png"), before)
        cv2.imwrite(str(OUT_DIR / "after_preprocess" / f"{stem}.png"), after)

        cv2.imwrite(str(OUT_DIR / "roi_mask" / f"{stem}_roi.png"), mask_roi)
        cv2.imwrite(str(OUT_DIR / "roi_mask" / f"{stem}_gate.png"), mask_gate)
        cv2.imwrite(str(OUT_DIR / "roi_mask" / f"{stem}_final.png"), mask_final)

        cv2.imwrite(str(OUT_DIR / "harris_overlay" / f"{stem}_MASKED.png"), draw_points(after, h_pts))
        cv2.imwrite(str(OUT_DIR / "gftt_overlay" / f"{stem}_MASKED.png"), draw_points(after, g_pts))

        summary["items"].append({
            "stem": stem,
            "orig_path": path,
            "capture": cap,
            "split": split,
            "num_harris": int(len(h_pts)),
            "num_gftt": int(len(g_pts)),
            "harris_inside_ratio": inside_ratio(h_pts, mask_final),
            "gftt_inside_ratio": inside_ratio(g_pts, mask_final),
            "roi_white_frac": roi_white,
            "final_white_frac": final_white,
            "roi_bad": bool(roi_bad),
            "mask_top_mean": mask_top_mean,
            "harris_top_pts": H_top,
            "gftt_top_pts": G_top,
        })

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDONE")
    print("Wrote:", OUT_DIR)
    print("Summary:", OUT_DIR / "summary.json")


if __name__ == "__main__":
    main()
