from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO_ROOT = Path(os.environ.get("FPRJ_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.fpbench.matchers.matching_baseline import (
    HarrisConfig,
    SIFTConfig,
    harris_extract,
    match_harris,
    match_sift,
    ransac_inliers,
    sift_extract,
)
from src.fpbench.preprocess.preprocess import PreprocessConfig, preprocess_image


# ----------------------------
# Path helpers
# ----------------------------
def parse_file_uri(p: str) -> Path:
    # Supports: file:/C:/... or normal path
    if p.startswith("file:"):
        p = p[len("file:"):]
        if p.startswith("/"):
            p = p[1:]
    p = p.replace("/", "\\")  # Windows-friendly
    return Path(p)


def project_root_default() -> Path:
    return Path(os.environ.get("FPRJ_ROOT", r"/"))


# ----------------------------
# Simple ROI mask (robust-ish to letters)
# ----------------------------
def largest_cc(binary_255: np.ndarray) -> np.ndarray:
    # binary_255: uint8 0/255
    num, labels, stats, _ = cv2.connectedComponentsWithStats((binary_255 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return binary_255
    # skip background=0
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_255)
    out[labels == idx] = 255
    return out


def make_paper_mask(img: np.ndarray) -> np.ndarray:
    # img: uint8 gray
    # paper/background is typically > 10, black borders are near 0
    paper = (img > 10).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, k, iterations=1)
    paper = largest_cc(paper)
    return paper


def make_roi_mask(img: np.ndarray) -> np.ndarray:
    """
    Goal: find the fingerprint textured region.
    Heuristic:
      1) paper mask to remove black borders
      2) gradient magnitude -> blur to "texture energy"
      3) threshold energy -> pick largest CC (fingerprint usually dominates letters)
    """
    paper = make_paper_mask(img)

    # gradient magnitude
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    # normalize & blur to get energy blobs
    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    energy = cv2.GaussianBlur(mag_u8, (0, 0), sigmaX=7, sigmaY=7)

    # threshold inside paper only
    vals = energy[paper > 0]
    if vals.size == 0:
        return paper

    # use a high percentile to focus on ridges texture, not flat paper
    thr = np.percentile(vals, 80)
    roi = (energy >= thr).astype(np.uint8) * 255
    roi[paper == 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, k, iterations=1)

    roi = largest_cc(roi)

    # sanity: if ROI is insane (too big/small), fall back to paper
    frac = (roi > 0).mean()
    if frac < 0.01 or frac > 0.90:
        return paper

    return roi


# ----------------------------
# Preprocess + feature extraction
# ----------------------------
def clahe(img: np.ndarray, clip=2.0, grid=(8, 8)) -> np.ndarray:
    c = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(grid))
    return c.apply(img)


def resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return img
    scale = long_edge / m
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def gftt_keypoints(img: np.ndarray, mask: np.ndarray, max_points=800, quality=0.01, min_dist=5.0) -> list[cv2.KeyPoint]:
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=int(max_points),
        qualityLevel=float(quality),
        minDistance=float(min_dist),
        mask=(mask > 0).astype(np.uint8),
        blockSize=3,
        useHarrisDetector=False
    )
    if corners is None:
        return []
    corners = corners.reshape(-1, 2)
    return [cv2.KeyPoint(float(x), float(y), 31) for x, y in corners]


@lru_cache(maxsize=4096)
def extract(
    path_str: str,
    detector: str,
    nfeatures: int,
    long_edge: int,
    target_size: int,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None, np.ndarray | None]:
    path = Path(path_str)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None, np.zeros((1, 1), dtype=np.uint8)

    if detector in {"harris_orb", "sift"}:
        img = preprocess_image(img, PreprocessConfig(target_size=int(target_size)))
        roi = None
    else:
        img = resize_long_edge(img, long_edge=long_edge)
        img = clahe(img, clip=2.0, grid=(8, 8))
        roi = make_roi_mask(img)

    if detector == "orb":
        orb = cv2.ORB_create(nfeatures=int(nfeatures))
        kps, des = orb.detectAndCompute(img, (roi > 0).astype(np.uint8))
        return kps or [], des, roi

    if detector == "gftt_orb":
        orb = cv2.ORB_create(nfeatures=int(nfeatures))
        kps = gftt_keypoints(img, roi, max_points=min(1200, nfeatures))
        if len(kps) == 0:
            return [], None, roi
        kps, des = orb.compute(img, kps)
        return kps or [], des, roi

    if detector == "harris_orb":
        cfg = HarrisConfig(
            nfeatures=int(nfeatures),
            max_points=min(1200, int(nfeatures)),
        )
        kps, des = harris_extract(img, None, cfg)
        return kps or [], des, roi

    if detector == "sift":
        cfg = SIFTConfig(nfeatures=int(nfeatures))
        kps, des = sift_extract(img, None, cfg)
        return kps or [], des, roi

    raise ValueError(f"Unknown detector: {detector}")


# ----------------------------
# Matching / scoring
# ----------------------------
def match_and_score(
    des1,
    des2,
    kps1,
    kps2,
    score_mode: str,
    ratio: float,
    ransac_thresh: float,
    detector: str,
    normalization_k: int,
) -> tuple[float, int, int]:
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return 0.0, 0, 0

    if detector == "harris_orb":
        good = match_harris(des1, des2, ratio=ratio)
    elif detector == "sift":
        good = match_sift(des1, des2, ratio=ratio)
    else:
        norm_type = cv2.NORM_L2 if detector == "sift" else cv2.NORM_HAMMING
        bf = cv2.BFMatcher(norm_type, crossCheck=False)
        knn = bf.knnMatch(des1, des2, k=2)

        good = []
        for match in knn:
            if len(match) == 2:
                m, n = match
                if m.distance < ratio * n.distance:
                    good.append(m)

    matches = len(good)
    if matches < 8:
        if score_mode == "matches":
            return float(matches), 0, matches
        return 0.0, 0, matches

    if detector in {"harris_orb", "sift"}:
        inliers, _ = ransac_inliers(kps1, kps2, good, reproj=float(ransac_thresh))
    else:
        pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in good])

        _, inlier_mask = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99
        )

        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0

    if score_mode == "inliers":
        return float(inliers), inliers, matches
    if score_mode == "matches":
        return float(matches), inliers, matches
    if score_mode == "inliers_over_matches":
        return float(inliers) / float(matches), inliers, matches
    if score_mode == "inliers_over_k":
        return float(inliers) / float(max(1, int(normalization_k))), inliers, matches
    if score_mode == "inliers_over_min_keypoints":
        denom = max(1, min(len(kps1), len(kps2)))
        return float(inliers) / float(denom), inliers, matches

    raise ValueError(f"Unknown score_mode: {score_mode}")


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


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_csv", type=str, help="Output scores CSV. Example: file:/C:/.../scores_test.csv")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--pairs", type=str, default="", help="Optional explicit pairs CSV path. If empty, uses data/processed/nist_sd300b/pairs_<split>.csv")
    ap.add_argument("--detector", type=str, default="gftt_orb", choices=["orb", "gftt_orb", "harris_orb", "sift"])
    ap.add_argument("--score_mode", type=str, default="inliers_over_k",
                    choices=["inliers_over_k", "inliers", "matches", "inliers_over_matches", "inliers_over_min_keypoints"])
    ap.add_argument("--nfeatures", type=int, default=1500)
    ap.add_argument("--long_edge", type=int, default=512)
    ap.add_argument("--target_size", type=int, default=512)
    ap.add_argument("--ratio", type=float, default=0.75)
    ap.add_argument("--ransac_thresh", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=0, help="If >0, evaluate only first N pairs (for quick tests).")
    args = ap.parse_args()

    out_path = parse_file_uri(args.out_csv)
    root = project_root_default()
    data_dir = root / "data" / "processed" / "nist_sd300b"

    pairs_path = Path(args.pairs) if args.pairs else (data_dir / f"pairs_{args.split}.csv")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    pairs = pd.read_csv(pairs_path)
    if args.limit and args.limit > 0:
        pairs = balanced_limit_by_label(pairs, label_col="label", limit=int(args.limit))

    rows = []
    for _, r in pairs.iterrows():
        pa = str(r["path_a"])
        pb = str(r["path_b"])
        label = int(r["label"])

        kps1, des1, _ = extract(pa, args.detector, args.nfeatures, args.long_edge, args.target_size)
        kps2, des2, _ = extract(pb, args.detector, args.nfeatures, args.long_edge, args.target_size)

        score, inliers, matches = match_and_score(
            des1, des2, kps1, kps2,
            score_mode=args.score_mode,
            ratio=args.ratio,
            ransac_thresh=args.ransac_thresh,
            detector=args.detector,
            normalization_k=args.nfeatures,
        )

        rows.append({
            "label": label,
            "split": args.split,
            "path_a": pa,
            "path_b": pb,
            "score": float(score),
            "inliers": int(inliers),
            "matches": int(matches),
            "k1": len(kps1),
            "k2": len(kps2),
        })

    df = pd.DataFrame(rows)
    auc, eer = compute_auc_eer(df["label"].values, df["score"].values)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Split={args.split} | N={len(df)} | AUC={auc:.4f} | EER~{eer:.4f}")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
