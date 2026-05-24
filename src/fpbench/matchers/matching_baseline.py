from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

@dataclass
class HarrisConfig:
    nfeatures: int = 1500
    block_size: int = 2
    ksize: int = 3
    k: float = 0.04
    thresh_rel: float = 0.01
    max_points: int = 1200
    min_distance: float = 5.0
    orb_edge_threshold: int = 31
    orb_fast_threshold: int = 10

@dataclass
class ORBConfig:
    nfeatures: int = 1500
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    fastThreshold: int = 10

def orb_extract(img_u8: np.ndarray, mask: Optional[np.ndarray], cfg: ORBConfig):
    orb = cv2.ORB_create(
        nfeatures=cfg.nfeatures,
        scaleFactor=cfg.scaleFactor,
        nlevels=cfg.nlevels,
        edgeThreshold=cfg.edgeThreshold,
        fastThreshold=cfg.fastThreshold
    )
    kps, desc = orb.detectAndCompute(img_u8, mask)
    if desc is None or len(kps) == 0:
        return [], None
    return kps, desc

def match_orb(desc1, desc2, ratio: float = 0.75):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

@dataclass
class SIFTConfig:
    nfeatures: int = 1500
    nOctaveLayers: int = 3
    contrastThreshold: float = 0.04
    edgeThreshold: float = 10
    sigma: float = 1.6

def sift_extract(img_u8: np.ndarray, mask: Optional[np.ndarray], cfg: SIFTConfig):
    sift = cv2.SIFT_create(
        nfeatures=cfg.nfeatures,
        nOctaveLayers=cfg.nOctaveLayers,
        contrastThreshold=cfg.contrastThreshold,
        edgeThreshold=cfg.edgeThreshold,
        sigma=cfg.sigma
    )
    kps, desc = sift.detectAndCompute(img_u8, mask)
    if desc is None or len(kps) == 0:
        return [], None
    return kps, desc

def match_sift(desc1, desc2, ratio: float = 0.75):
    # SIFT features are float32, so we MUST use NORM_L2 (Euclidean) instead of Hamming
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = []
    for match in knn:
        # Check if knn found 2 matches (sometimes it only finds 1)
        if len(match) == 2:
            m, n = match
            if m.distance < ratio * n.distance:
                good.append(m)
    return good

SIFT_PLAIN_ROLL_V2_SCORE_MODE = "inliers_times_inlier_ratio_times_log1p_matches"
LOCAL_FEATURE_SCORE_MODES = (
    "inliers_over_k",
    "inliers",
    "matches",
    "inliers_over_matches",
    "inliers_over_min_keypoints",
    SIFT_PLAIN_ROLL_V2_SCORE_MODE,
)
RANSAC_MODELS = ("homography", "affine_partial_2d", "affine_full_2d")


def score_local_feature_counts(
    *,
    score_mode: str,
    inliers: int,
    matches: int,
    k1: int,
    k2: int,
    normalization_k: int,
) -> float:
    matches = int(matches)
    inliers = int(inliers)
    if score_mode == "inliers":
        return float(inliers)
    if score_mode == "matches":
        return float(matches)
    if score_mode == "inliers_over_matches":
        return float(inliers) / float(max(matches, 1))
    if score_mode == "inliers_over_k":
        return float(inliers) / float(max(1, int(normalization_k)))
    if score_mode == "inliers_over_min_keypoints":
        denom = max(1, min(int(k1), int(k2)))
        return float(inliers) / float(denom)
    if score_mode == SIFT_PLAIN_ROLL_V2_SCORE_MODE:
        return score_sift_plain_roll_v2_counts(matches=matches, inliers=inliers)
    raise ValueError(f"Unknown score_mode: {score_mode}")


def score_sift_plain_roll_v2_counts(*, matches: int, inliers: int) -> float:
    matches = int(matches)
    inliers = int(inliers)
    if matches <= 0 or inliers <= 0:
        return 0.0
    inlier_ratio = float(inliers) / float(max(matches, 1))
    return float(inliers) * inlier_ratio * math.log1p(float(matches))


def ransac_inliers_for_model(
    kps1,
    kps2,
    matches,
    *,
    ransac_model: str = "homography",
    ransac_thresh: float = 3.0,
) -> Tuple[int, Optional[np.ndarray]]:
    model = str(ransac_model)
    min_matches = 8 if model == "homography" else 3
    if len(matches) < min_matches:
        return 0, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    mask = None
    if model == "homography":
        _, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
        )
    elif model == "affine_partial_2d":
        cv2.setRNGSeed(0)
        _, mask = cv2.estimateAffinePartial2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99,
        )
    elif model == "affine_full_2d":
        cv2.setRNGSeed(0)
        _, mask = cv2.estimateAffine2D(
            pts1,
            pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_thresh),
            maxIters=2000,
            confidence=0.99,
        )
    else:
        raise ValueError(f"Unsupported ransac_model: {ransac_model}")

    if mask is None:
        return 0, None
    flat_mask = mask.reshape(-1).astype(np.uint8)
    return int(flat_mask.sum()), flat_mask

def _harris_points(
    img_u8: np.ndarray,
    mask: Optional[np.ndarray],
    cfg: HarrisConfig,
) -> List[Tuple[float, float]]:
    img_f32 = np.float32(img_u8)
    resp = cv2.cornerHarris(
        img_f32,
        blockSize=int(cfg.block_size),
        ksize=int(cfg.ksize),
        k=float(cfg.k),
    )
    resp = cv2.dilate(resp, None)

    max_resp = float(resp.max()) if resp.size else 0.0
    if max_resp <= 0:
        return []

    thr = max_resp * float(cfg.thresh_rel)
    ys, xs = np.where(resp >= thr)

    if len(xs) == 0:
        return []

    if mask is not None:
        keep = mask[ys, xs] > 0
        ys = ys[keep]
        xs = xs[keep]
        if len(xs) == 0:
            return []

    vals = resp[ys, xs]
    order = np.argsort(vals)[::-1]

    pts: List[Tuple[float, float]] = []
    min_d2 = float(cfg.min_distance) ** 2

    for idx in order:
        x0 = float(xs[idx])
        y0 = float(ys[idx])

        good = True
        for px, py in pts:
            dx = x0 - px
            dy = y0 - py
            if dx * dx + dy * dy < min_d2:
                good = False
                break

        if good:
            pts.append((x0, y0))

        if len(pts) >= int(cfg.max_points):
            break

    return pts


def harris_extract(
    img_u8: np.ndarray,
    mask: Optional[np.ndarray],
    cfg: HarrisConfig,
):
    pts = _harris_points(img_u8, mask, cfg)
    if not pts:
        return [], None

    kps = [cv2.KeyPoint(float(x), float(y), 31) for x, y in pts]

    orb = cv2.ORB_create(
        nfeatures=int(cfg.nfeatures),
        edgeThreshold=int(cfg.orb_edge_threshold),
        fastThreshold=int(cfg.orb_fast_threshold),
    )

    kps, desc = orb.compute(img_u8, kps)
    if desc is None or not kps:
        return [], None

    return kps, desc


def match_harris(desc1, desc2, ratio: float = 0.75):
    # Harris אצלנו משתמש ב-ORB descriptors, לכן matching כמו ORB
    return match_orb(desc1, desc2, ratio=ratio)


def score_pair_harris(
    img1_u8: np.ndarray,
    img2_u8: np.ndarray,
    mask1: Optional[np.ndarray],
    mask2: Optional[np.ndarray],
    harris_cfg: HarrisConfig,
    ratio: float = 0.75,
    reproj: float = 3.0,
) -> Dict:
    kps1, desc1 = harris_extract(img1_u8, mask1, harris_cfg)
    kps2, desc2 = harris_extract(img2_u8, mask2, harris_cfg)

    if desc1 is None or desc2 is None:
        return {
            "score": 0.0,
            "inliers": 0,
            "matches": 0,
            "k1": len(kps1) if kps1 else 0,
            "k2": len(kps2) if kps2 else 0,
        }

    matches = match_harris(desc1, desc2, ratio=ratio)
    inliers, _ = ransac_inliers(kps1, kps2, matches, reproj=reproj)

    denom = max(1, min(len(kps1), len(kps2)))
    score = inliers / denom

    return {
        "score": float(score),
        "inliers": int(inliers),
        "matches": int(len(matches)),
        "k1": len(kps1),
        "k2": len(kps2),
    }

# ==========================================
# Shared Geometry & Scoring
# ==========================================
def ransac_inliers(kps1, kps2, matches, reproj: float = 3.0) -> Tuple[int, Optional[np.ndarray]]:
    if len(matches) < 8:
        return 0, None
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=reproj)
    if mask is None:
        return 0, None
    inliers = int(mask.ravel().sum())
    return inliers, H

def score_pair(img1_u8: np.ndarray, img2_u8: np.ndarray,
               mask1: Optional[np.ndarray], mask2: Optional[np.ndarray],
               orb_cfg: ORBConfig,
               ratio: float = 0.75, reproj: float = 3.0) -> Dict:
    kps1, desc1 = orb_extract(img1_u8, mask1, orb_cfg)
    kps2, desc2 = orb_extract(img2_u8, mask2, orb_cfg)

    if desc1 is None or desc2 is None:
        return {"score": 0.0, "inliers": 0, "matches": 0, "k1": len(kps1), "k2": len(kps2)}

    matches = match_orb(desc1, desc2, ratio=ratio)
    inliers, _ = ransac_inliers(kps1, kps2, matches, reproj=reproj)

    denom = max(1, min(len(kps1), len(kps2)))
    score = inliers / denom

    return {"score": float(score), "inliers": int(inliers), "matches": int(len(matches)), "k1": len(kps1), "k2": len(kps2)}

def score_pair_sift(img1_u8: np.ndarray, img2_u8: np.ndarray,
                    mask1: Optional[np.ndarray], mask2: Optional[np.ndarray],
                    sift_cfg: SIFTConfig,
                    ratio: float = 0.75, reproj: float = 3.0) -> Dict:
    kps1, desc1 = sift_extract(img1_u8, mask1, sift_cfg)
    kps2, desc2 = sift_extract(img2_u8, mask2, sift_cfg)

    if desc1 is None or desc2 is None:
        return {"score": 0.0, "inliers": 0, "matches": 0, "k1": len(kps1) if kps1 else 0, "k2": len(kps2) if kps2 else 0}

    matches = match_sift(desc1, desc2, ratio=ratio)
    inliers, _ = ransac_inliers(kps1, kps2, matches, reproj=reproj)

    denom = max(1, min(len(kps1), len(kps2)))
    score = inliers / denom

    return {"score": float(score), "inliers": int(inliers), "matches": int(len(matches)), "k1": len(kps1), "k2": len(kps2)}
