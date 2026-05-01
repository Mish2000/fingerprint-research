"""
Outputs:
- score: float in [0,1] = geometry-aware composite score
- matches: list of tentative matches (for visualization)
- inliers_mask: list[bool] aligned with matches
- latency_ms: dict of timing components
- mean_inlier_sim / median_inlier_sim: similarity stats over inlier matches
- mean_tentative_sim / median_tentative_sim: similarity stats over tentative matches
- max_tentative_sim / mean_top10_tentative_sim / mean_top20_tentative_sim: extra tentative stats
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fpbench.preprocess.preprocess import (
    PreprocessConfig,
    extract_fingerprint_roi,
    load_gray,
    preprocess_image,
    suppress_header_and_borders,
    rectangular_gate_mask,
    gftt_keypoints,
)

# -------------------------
# Frozen "contract" defaults (Weeks 6-7)
# -------------------------
DEFAULT_CFG = PreprocessConfig(target_size=512, clahe_clip=2.0, clahe_grid=(8, 8), blur_ksize=3)
PATCH_SIZE = 48
MAX_KPTS = 800
MASK_COV_THR = 0.70

# For RANSAC
DEFAULT_RANSAC_THRESH = 4.0
DEFAULT_MAX_MATCHES = 200  # keep strongest matches before RANSAC (stability + speed)


def _norm_path(p: str) -> str:
    # normalize for caching / manifest matching
    return str(Path(p).resolve()).replace("\\", "/").lower()


def build_final_mask(img_u8: np.ndarray, *, capture: str) -> np.ndarray:
    """
    Final mask = roi AND gate; with fallback to gate if roi becomes too small.
    (Matches Week 6-7 contract summary.)
    """
    roi_result = extract_fingerprint_roi(img_u8)
    roi = suppress_header_and_borders(roi_result.mask, top_ratio=0.12, border=10)

    gate = rectangular_gate_mask(
        img_u8.shape[:2],
        capture=capture,
        top_plain=0.18,
        top_roll=0.05,
        border=12,
    )

    final = cv2.bitwise_and(roi, gate) if roi_result.is_valid else np.zeros_like(gate)

    # fallback to the deterministic gate only when ROI failure / collapse is explicit
    if (not roi_result.is_valid) or float(np.mean(final > 0)) < 0.005:
        final = gate

    return final.astype(np.uint8)


def filter_pts_inbounds(pts: np.ndarray, h: int, w: int, r: int) -> np.ndarray:
    if pts.size == 0:
        return pts
    x = pts[:, 0]
    y = pts[:, 1]
    ok = (x >= r) & (x < (w - r)) & (y >= r) & (y < (h - r))
    return pts[ok]


def mask_majority_ok(mask_u8: np.ndarray, x: int, y: int, r: int, thr: float = MASK_COV_THR) -> bool:
    y0, y1 = y - r, y + r
    x0, x1 = x - r, x + r
    win = mask_u8[y0:y1, x0:x1]
    if win.size == 0:
        return False
    return float(np.mean(win > 0)) >= float(thr)


def extract_patch(img_u8: np.ndarray, x: int, y: int, patch: int = PATCH_SIZE) -> np.ndarray:
    r = patch // 2
    return img_u8[y - r: y + r, x - r: x + r].copy()


# -------------------------
# Model definition (must match train_patch_descriptor.py)
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


def default_ckpt_path(root: Path) -> Path:
    candidates = [
        root / "artifacts" / "checkpoints" / "patch_descriptor" / "final" / "patch_descriptor_ckpt.pth",
        root / "reports" / "week06+07" / "patch_descriptor" / "final" / "patch_descriptor_ckpt.pth",
        root / "reports" / "week06" / "patch_descriptor" / "final" / "patch_descriptor_ckpt.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


@dataclass
class MatchResult:
    score: float
    matches: List[Dict[str, Any]]  # each: {i, j, sim, pt_a, pt_b}
    inliers_mask: List[bool]  # aligned with matches
    inliers_count: int
    tentative_count: int
    latency_ms: Dict[str, float]

    mean_inlier_sim: float = 0.0
    median_inlier_sim: float = 0.0
    mean_tentative_sim: float = 0.0
    median_tentative_sim: float = 0.0

    max_tentative_sim: float = 0.0
    mean_top10_tentative_sim: float = 0.0
    mean_top20_tentative_sim: float = 0.0
    inlier_ratio: float = 0.0
    stability_term: float = 0.0
    score_components: Dict[str, float] | None = None


def _geometry_aware_score(mean_tentative_sim: float, mean_inlier_sim: float, inlier_ratio: float, stability_term: float) -> tuple[float, Dict[str, float]]:
    weights = {
        "mean_tentative_sim": 0.35,
        "mean_inlier_sim": 0.25,
        "inlier_ratio": 0.25,
        "stability_term": 0.15,
    }
    raw = (weights["mean_tentative_sim"] * float(mean_tentative_sim) + weights["mean_inlier_sim"] * float(mean_inlier_sim) + weights["inlier_ratio"] * float(inlier_ratio) + weights["stability_term"] * float(stability_term))
    score = float(np.clip(raw, 0.0, 1.0))
    components = {**weights, "raw_score": float(raw)}
    return score, components


class DedicatedMatcher:
    def __init__(
            self,
            *,
            ckpt_path: Optional[str] = None,
            device: Optional[str] = None,
            cfg: PreprocessConfig = DEFAULT_CFG,
            patch: int = PATCH_SIZE,
            max_kpts: int = MAX_KPTS,
            mask_cov_thr: float = MASK_COV_THR,
            max_matches: int = DEFAULT_MAX_MATCHES,
            ransac_thresh: float = DEFAULT_RANSAC_THRESH,
            nn_delta: float = 0.00,
            nn_sim_min: float = -1.0,

    ):
        self.root = Path.cwd()
        self.cfg = cfg
        self.patch = int(patch)
        self.r = self.patch // 2
        self.max_kpts = int(max_kpts)
        self.mask_cov_thr = float(mask_cov_thr)
        self.max_matches = int(max_matches)
        self.ransac_thresh = float(ransac_thresh)
        self.nn_delta = float(nn_delta)
        self.nn_sim_min = float(nn_sim_min)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if ckpt_path is None:
            ckpt_path = str(default_ckpt_path(self.root))
        self.ckpt_path = Path(ckpt_path)

        self._model = SimCLRModel(emb_dim=256, proj_dim=128).to(self.device).eval()
        self._load_ckpt(self.ckpt_path)

        # cache: norm_path -> (pts_xy float32 Nx2, emb float32 NxD)
        self._cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def _load_ckpt(self, ckpt_path: Path) -> None:
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Descriptor checkpoint not found: {ckpt_path}\n"
                "Checked canonical path first: "
                "artifacts/checkpoints/patch_descriptor/final/patch_descriptor_ckpt.pth\n"
                "Legacy fallbacks: "
                "reports/week06+07/patch_descriptor/final/patch_descriptor_ckpt.pth, "
                "reports/week06/patch_descriptor/final/patch_descriptor_ckpt.pth"
            )
        try:
            ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location=self.device)

        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict):
            # allow direct state_dict save
            state = ckpt
        else:
            raise ValueError("Unexpected checkpoint format. Expected dict with key 'model'.")

        self._model.load_state_dict(state, strict=True)

    @torch.no_grad()
    def embed_image(self, img_path: str, *, capture: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Returns:
          pts_xy: (N,2) float32 in image coordinates
          emb:    (N,D) float32 L2-normalized (encoder space h)
          latency_ms: dict
        """
        t0 = time.perf_counter()

        key = _norm_path(img_path)
        if key in self._cache:
            pts_xy, emb = self._cache[key]
            return pts_xy, emb, {"total": 0.0, "cached": 1.0}

        # Load + preprocess
        gray = load_gray(img_path)
        img = preprocess_image(gray, self.cfg)  # uint8, 512x512
        t_pre = (time.perf_counter() - t0) * 1000.0

        # Mask + keypoints
        t1 = time.perf_counter()
        mask = build_final_mask(img, capture=capture)
        pts = gftt_keypoints(img, max_points=self.max_kpts, quality=0.01, min_dist=5.0, mask=mask)
        pts = filter_pts_inbounds(pts, img.shape[0], img.shape[1], self.r)

        valid: List[Tuple[int, int]] = []
        for (x, y) in pts:
            xi, yi = int(x), int(y)
            if mask_majority_ok(mask, xi, yi, self.r, thr=self.mask_cov_thr):
                valid.append((xi, yi))

        if len(valid) == 0:
            pts_xy = np.zeros((0, 2), dtype=np.float32)
            emb = np.zeros((0, 256), dtype=np.float32)
            self._cache[key] = (pts_xy, emb)
            t_total = (time.perf_counter() - t0) * 1000.0
            return pts_xy, emb, {"preprocess": t_pre, "kpts": (time.perf_counter() - t1) * 1000.0, "embed": 0.0,
                                 "total": t_total}

        pts_xy = np.array(valid, dtype=np.float32)

        t_kpts = (time.perf_counter() - t1) * 1000.0

        # Extract patches -> tensor batch (N,1,H,W) in [0,1]
        t2 = time.perf_counter()
        patches = np.stack([extract_patch(img, x, y, self.patch) for (x, y) in valid], axis=0)  # (N,48,48)
        patches_f = patches.astype(np.float32) / 255.0
        x = torch.from_numpy(patches_f).unsqueeze(1)  # (N,1,48,48)
        x = x.to(self.device, non_blocking=True)

        # Embed in batches to be safe on VRAM
        bs = 256
        embs = []
        for i in range(0, x.shape[0], bs):
            h, _ = self._model(x[i:i + bs])
            embs.append(h.detach().float().cpu().numpy())
        emb = np.concatenate(embs, axis=0).astype(np.float32)  # already normalized

        t_embed = (time.perf_counter() - t2) * 1000.0
        t_total = (time.perf_counter() - t0) * 1000.0

        self._cache[key] = (pts_xy, emb)
        return pts_xy, emb, {"preprocess": t_pre, "kpts": t_kpts, "embed": t_embed, "total": t_total}

    def _mutual_nn_matches(self, emb_a: np.ndarray, emb_b: np.ndarray) -> List[Tuple[int, int, float]]:
        if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
            return []

        sim = emb_a @ emb_b.T  # (Na, Nb)
        Na, Nb = sim.shape

        # Best & second-best per i (requires Nb>=2; otherwise skip ratio)
        if Nb >= 2:
            idx2 = np.argpartition(-sim, kth=1, axis=1)[:, :2]  # (Na,2) indices (unordered)
            vals2 = sim[np.arange(Na)[:, None], idx2]  # (Na,2) values

            order = np.argsort(-vals2, axis=1)  # sort desc within each row
            j1 = idx2[np.arange(Na), order[:, 0]]
            s1 = vals2[np.arange(Na), order[:, 0]]
            s2 = vals2[np.arange(Na), order[:, 1]]

            keep = (s1 >= self.nn_sim_min) & ((s1 - s2) >= self.nn_delta)
        else:
            j1 = sim.argmax(axis=1)
            s1 = sim[np.arange(Na), j1]
            keep = (s1 >= self.nn_sim_min)

        # Mutual check
        i_for_j = sim.argmax(axis=0)  # (Nb,)
        keep = keep & (i_for_j[j1] == np.arange(Na))

        isel = np.where(keep)[0]
        matches = [(int(i), int(j1[i]), float(s1[i])) for i in isel]
        return matches

    def score_pair(
            self,
            path_a: str,
            path_b: str,
            *,
            capture_a: str,
            capture_b: str,
    ) -> MatchResult:
        t0 = time.perf_counter()

        pts_a, emb_a, lat_a = self.embed_image(path_a, capture=capture_a)
        pts_b, emb_b, lat_b = self.embed_image(path_b, capture=capture_b)

        t_match0 = time.perf_counter()
        matches = self._mutual_nn_matches(emb_a, emb_b)

        # Keep strongest matches (stability)
        matches.sort(key=lambda x: x[2], reverse=True)
        if self.max_matches > 0:
            matches = matches[: self.max_matches]

        tentative_count = len(matches)
        t_match = (time.perf_counter() - t_match0) * 1000.0

        # RANSAC
        t_r0 = time.perf_counter()
        inliers_mask: List[bool] = []
        inliers_count = 0

        if tentative_count >= 3:
            src2 = np.float32([pts_a[i] for (i, _, _) in matches]).reshape(-1, 2)
            dst2 = np.float32([pts_b[j] for (_, j, _) in matches]).reshape(-1, 2)

            # Deterministic RANSAC per pair (OpenCV uses its own RNG)
            if hasattr(cv2, "setRNGSeed"):
                key = (_norm_path(path_a) + "|" + _norm_path(path_b)).encode("utf-8")
                seed = int(zlib.adler32(key) & 0x7fffffff)
                cv2.setRNGSeed(seed)

            M, mask = cv2.estimateAffinePartial2D(
                src2, dst2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_thresh,
                maxIters=2000,
                confidence=0.99,
                refineIters=10,
            )
            if mask is not None:
                mask = mask.astype(np.uint8).reshape(-1)
                inliers_mask = [bool(m) for m in mask.tolist()]
                inliers_count = int(mask.sum())
            else:
                inliers_mask = [False] * tentative_count
                inliers_count = 0
        else:
            inliers_mask = [False] * tentative_count
            inliers_count = 0

        t_ransac = (time.perf_counter() - t_r0) * 1000.0

        # Similarity stats (tentative and inlier-only)
        tentative_sims = [float(s) for (_, _, s) in matches]
        if tentative_sims:
            mean_tentative_sim = float(np.mean(tentative_sims))
            median_tentative_sim = float(np.median(tentative_sims))

            s_sorted = sorted(tentative_sims, reverse=True)
            max_tentative_sim = float(s_sorted[0])
            mean_top10_tentative_sim = float(np.mean(s_sorted[: min(10, len(s_sorted))]))
            mean_top20_tentative_sim = float(np.mean(s_sorted[: min(20, len(s_sorted))]))
        else:
            mean_tentative_sim = 0.0
            median_tentative_sim = 0.0
            max_tentative_sim = 0.0
            mean_top10_tentative_sim = 0.0
            mean_top20_tentative_sim = 0.0

        inlier_sims = [
            float(s)
            for k, (_, _, s) in enumerate(matches)
            if k < len(inliers_mask) and inliers_mask[k]
        ]
        if inlier_sims:
            mean_inlier_sim = float(np.mean(inlier_sims))
            median_inlier_sim = float(np.median(inlier_sims))
        else:
            mean_inlier_sim = 0.0
            median_inlier_sim = 0.0

        inlier_ratio = float(inliers_count / tentative_count) if tentative_count > 0 else 0.0
        stability_term = float(min(1.0, tentative_count / max(1, self.max_matches))) * inlier_ratio
        score, score_components = _geometry_aware_score(
            mean_tentative_sim=mean_tentative_sim,
            mean_inlier_sim=mean_inlier_sim,
            inlier_ratio=inlier_ratio,
            stability_term=stability_term,
        )

        # pack match dicts for visualization
        match_dicts: List[Dict[str, Any]] = []
        for k, (i, j, s) in enumerate(matches):
            match_dicts.append({
                "i": int(i),
                "j": int(j),
                "sim": float(s),
                "pt_a": [float(pts_a[i, 0]), float(pts_a[i, 1])] if len(pts_a) else [0.0, 0.0],
                "pt_b": [float(pts_b[j, 0]), float(pts_b[j, 1])] if len(pts_b) else [0.0, 0.0],
                "inlier": bool(inliers_mask[k]) if k < len(inliers_mask) else False,
            })

        total = (time.perf_counter() - t0) * 1000.0
        latency = {
            "embed_a_total": float(lat_a.get("total", 0.0)),
            "embed_b_total": float(lat_b.get("total", 0.0)),
            "match_ms": float(t_match),
            "ransac_ms": float(t_ransac),
            "pair_total_ms": float(total),
        }

        return MatchResult(
            score=score,
            matches=match_dicts,
            inliers_mask=inliers_mask,
            inliers_count=inliers_count,
            tentative_count=tentative_count,
            latency_ms=latency,
            mean_inlier_sim=mean_inlier_sim,
            median_inlier_sim=median_inlier_sim,
            mean_tentative_sim=mean_tentative_sim,
            median_tentative_sim=median_tentative_sim,
            max_tentative_sim=max_tentative_sim,
            mean_top10_tentative_sim=mean_top10_tentative_sim,
            mean_top20_tentative_sim=mean_top20_tentative_sim,
            inlier_ratio=inlier_ratio,
            stability_term=stability_term,
            score_components=score_components,
        )


def draw_match_viz(
        img_a_u8: np.ndarray,
        img_b_u8: np.ndarray,
        matches: List[Dict[str, Any]],
        *,
        max_draw: int = 100,
) -> np.ndarray:
    """
    Simple side-by-side visualization. Green = inlier, Red = outlier.
    """
    h1, w1 = img_a_u8.shape[:2]
    h2, w2 = img_b_u8.shape[:2]
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = cv2.cvtColor(img_a_u8, cv2.COLOR_GRAY2BGR)
    canvas[:h2, w1:w1 + w2] = cv2.cvtColor(img_b_u8, cv2.COLOR_GRAY2BGR)

    for m in matches[:max_draw]:
        x1, y1 = m["pt_a"]
        x2, y2 = m["pt_b"]
        x2s = x2 + w1
        inl = bool(m.get("inlier", False))
        color = (0, 255, 0) if inl else (0, 0, 255)
        cv2.circle(canvas, (int(x1), int(y1)), 2, color, -1)
        cv2.circle(canvas, (int(x2s), int(y2)), 2, color, -1)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2s), int(y2)), color, 1)

    return canvas
