from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    target_size: int = 512          # output is target_size x target_size
    clahe_clip: float = 2.0
    clahe_grid: Tuple[int, int] = (8, 8)
    blur_ksize: int = 3             # must be odd; 0 disables blur


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_pad_to_square(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((target, target), dtype=resized.dtype)
    y0 = (target - new_h) // 2
    x0 = (target - new_w) // 2
    out[y0:y0 + new_h, x0:x0 + new_w] = resized
    return out


def preprocess_image(gray: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=cfg.clahe_clip, tileGridSize=cfg.clahe_grid)
    x = clahe.apply(gray)

    # Resize + pad to uniform size
    x = resize_pad_to_square(x, cfg.target_size)

    # Optional light blur
    if cfg.blur_ksize and cfg.blur_ksize >= 3:
        k = cfg.blur_ksize if cfg.blur_ksize % 2 == 1 else cfg.blur_ksize + 1
        x = cv2.GaussianBlur(x, (k, k), 0)

    # Normalize to 0..255 uint8 (stable)
    x = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX)
    x = x.astype(np.uint8)
    return x


def harris_keypoints(img_u8: np.ndarray, max_points: int = 500,
                     block_size: int = 2, ksize: int = 3, k: float = 0.04,
                     thresh_rel: float = 0.01,
                     mask: np.ndarray | None = None) -> np.ndarray:
    img_f = np.float32(img_u8) / 255.0
    resp = cv2.cornerHarris(img_f, blockSize=block_size, ksize=ksize, k=k)

    if mask is not None:
        resp = resp.copy()
        resp[mask == 0] = 0.0

    thr = thresh_rel * resp.max()
    if thr <= 0:
        return np.zeros((0, 2), dtype=np.int32)

    m = resp > thr
    resp_dil = cv2.dilate(resp, None)
    nms = (resp == resp_dil) & m

    ys, xs = np.where(nms)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    scores = resp[ys, xs]
    idx = np.argsort(scores)[::-1][:max_points]
    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.int32)
    return pts


def gftt_keypoints(img_u8: np.ndarray, max_points: int = 500,
                   quality: float = 0.01, min_dist: float = 5.0,
                   mask: np.ndarray | None = None) -> np.ndarray:
    corners = cv2.goodFeaturesToTrack(
        img_u8,
        maxCorners=max_points,
        qualityLevel=quality,
        minDistance=min_dist,
        mask=mask,
        useHarrisDetector=False
    )
    if corners is None:
        return np.zeros((0, 2), dtype=np.int32)
    return corners.reshape(-1, 2).astype(np.int32)



def draw_points(img_u8: np.ndarray, pts_xy: np.ndarray, radius: int = 2) -> np.ndarray:
    out = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    for (x, y) in pts_xy:
        cv2.circle(out, (int(x), int(y)), radius, (0, 255, 0), -1)
    return out

def fingerprint_roi_mask(img_u8: np.ndarray) -> np.ndarray:
    """
    Binary ROI mask (0/255) targeting ridge-dense fingerprint region.
    Uses black-hat to emphasize dark ridges on light background.
    """
    # emphasize ridges (dark thin structures)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    rid = cv2.morphologyEx(img_u8, cv2.MORPH_BLACKHAT, k)

    rid = cv2.GaussianBlur(rid, (5, 5), 0)
    rid = cv2.normalize(rid, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu threshold
    _, bw = cv2.threshold(rid, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphology to form one solid blob
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)), iterations=2)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),  iterations=1)

    # keep largest connected component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return np.full_like(img_u8, 255, dtype=np.uint8)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8) * 255

    # erode to avoid touching nearby artifacts
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    return mask


def suppress_header_and_borders(mask: np.ndarray, top_ratio: float = 0.12, border: int = 10) -> np.ndarray:
    m = mask.copy()
    h, w = m.shape[:2]

    # remove top strip (text/header)
    top_h = int(round(top_ratio * h))
    m[:top_h, :] = 0

    # remove borders (scanner/frame artifacts)
    if border > 0:
        m[:border, :] = 0
        m[-border:, :] = 0
        m[:, :border] = 0
        m[:, -border:] = 0

    return m

def rectangular_gate_mask(shape_hw, capture: str, top_plain: float = 0.18, top_roll: float = 0.05, border: int = 12):
    """
    A deterministic gate that removes the top header strip + borders.
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