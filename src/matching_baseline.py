from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import cv2
import numpy as np


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

    # Normalized score (stable across varying kp counts)
    denom = max(1, min(len(kps1), len(kps2)))
    score = inliers / denom

    return {"score": float(score), "inliers": int(inliers), "matches": int(len(matches)), "k1": len(kps1), "k2": len(kps2)}
