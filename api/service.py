from __future__ import annotations

import time
from dataclasses import asdict
from typing import Optional, Tuple, Dict, Any, List

import cv2
import numpy as np

from api.schemas import MatchMethod, MatchResponse, Overlay, OverlayMatch

# Project imports (your existing code)
from src.preprocess import PreprocessConfig, load_gray, preprocess_image
from src.matching_baseline import ORBConfig, orb_extract, match_orb
from src.baseline_dl import BaselineDL
from src.dedicated_matcher import DedicatedMatcher


def _infer_capture_from_filename(name: str) -> Optional[str]:
    s = (name or "").lower()
    if "roll" in s:
        return "roll"
    if "plain" in s:
        return "plain"
    return None


def _default_threshold(method: MatchMethod) -> float:
    if method == MatchMethod.classic:
        return 0.15
    if method == MatchMethod.dl:
        return 0.50
    return 0.50  # dedicated


class MatchService:
    def __init__(self):
        # Keep preprocessing consistent with your baselines
        self.prep_cfg = PreprocessConfig(target_size=512)

        # Classic ORB defaults match your matching_baseline.py
        self.orb_cfg = ORBConfig()

        # Models (loaded once)
        self.dl = BaselineDL(prep_cfg=self.prep_cfg)
        self.dedicated = DedicatedMatcher(cfg=self.prep_cfg)

    def _preprocess_path(self, path: str) -> np.ndarray:
        gray = load_gray(path)
        return preprocess_image(gray, self.prep_cfg)

    def _classic_score_and_overlay(
        self,
        path_a: str,
        path_b: str,
        *,
        return_overlay: bool,
        max_draw: int = 200,
        ratio: float = 0.75,
        reproj: float = 3.0,
    ) -> Tuple[float, Dict[str, Any], Optional[Overlay]]:
        img1 = self._preprocess_path(path_a)
        img2 = self._preprocess_path(path_b)

        kps1, desc1 = orb_extract(img1, None, self.orb_cfg)
        kps2, desc2 = orb_extract(img2, None, self.orb_cfg)

        if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
            meta = {"inliers": 0, "matches": 0, "k1": len(kps1), "k2": len(kps2)}
            return 0.0, meta, (Overlay(matches=[]) if return_overlay else None)

        matches = match_orb(desc1, desc2, ratio=ratio)

        mask = None
        inliers = 0
        if len(matches) >= 8:
            pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=reproj)
            if mask is not None:
                mask = mask.reshape(-1).astype(np.uint8)
                inliers = int(mask.sum())

        denom = max(1, min(len(kps1), len(kps2)))
        score = float(inliers / denom)

        meta = {"inliers": int(inliers), "matches": int(len(matches)), "k1": len(kps1), "k2": len(kps2)}

        ov = None
        if return_overlay:
            out: List[OverlayMatch] = []
            take = matches[:max_draw]
            for i, m in enumerate(take):
                a = kps1[m.queryIdx].pt
                b = kps2[m.trainIdx].pt
                if mask is None:
                    kind = "tentative"
                else:
                    kind = "inlier" if int(mask[i]) == 1 else "outlier"
                out.append(OverlayMatch(a=(float(a[0]), float(a[1])), b=(float(b[0]), float(b[1])), kind=kind))
            ov = Overlay(matches=out)

        return score, meta, ov

    def match(
        self,
        *,
        method: MatchMethod,
        path_a: str,
        path_b: str,
        threshold: Optional[float],
        return_overlay: bool,
        capture_a: Optional[str],
        capture_b: Optional[str],
        filename_a: Optional[str],
        filename_b: Optional[str],
    ) -> MatchResponse:
        th = float(threshold) if threshold is not None else _default_threshold(method)

        if method == MatchMethod.classic:
            t0 = time.perf_counter()
            score, meta, ov = self._classic_score_and_overlay(path_a, path_b, return_overlay=return_overlay)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
            )

        if method == MatchMethod.dl:
            t0 = time.perf_counter()
            emb_a, ms_a = self.dl.embed_path(path_a)
            emb_b, ms_b = self.dl.embed_path(path_b)
            score = float(self.dl.cosine(emb_a, emb_b))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            meta = {
                "embed_ms_a": float(ms_a),
                "embed_ms_b": float(ms_b),
                "dl_config": self.dl.config_dict(),
            }
            return MatchResponse(
                method=method,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=None,
            )

        # dedicated
        cap_a = (capture_a or _infer_capture_from_filename(filename_a or "") or "plain").lower()
        cap_b = (capture_b or _infer_capture_from_filename(filename_b or "") or "plain").lower()
        if cap_a not in ("plain", "roll") or cap_b not in ("plain", "roll"):
            raise ValueError("capture_a/capture_b must be 'plain' or 'roll'")

        t0 = time.perf_counter()
        res = self.dedicated.score_pair(path_a, path_b, capture_a=cap_a, capture_b=cap_b)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ov = None
        if return_overlay:
            out = []
            for m in res.matches:
                a = m["pt_a"]
                b = m["pt_b"]
                kind = "inlier" if bool(m.get("inlier", False)) else "outlier"
                out.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind=kind,
                        sim=float(m.get("sim", 0.0)),
                    )
                )
            ov = Overlay(matches=out)

        meta = {
            "tentative_count": int(res.tentative_count),
            "inliers_count": int(res.inliers_count),
            "stats": {
                "mean_tentative_sim": float(res.mean_tentative_sim),
                "median_tentative_sim": float(res.median_tentative_sim),
                "mean_inlier_sim": float(res.mean_inlier_sim),
                "median_inlier_sim": float(res.median_inlier_sim),
                "max_tentative_sim": float(res.max_tentative_sim),
                "mean_top10_tentative_sim": float(res.mean_top10_tentative_sim),
                "mean_top20_tentative_sim": float(res.mean_top20_tentative_sim),
            },
            "latency_breakdown_ms": dict(res.latency_ms),
        }

        return MatchResponse(
            method=method,
            score=float(res.score),
            decision=bool(res.score >= th),
            threshold=th,
            latency_ms=float(latency_ms),
            meta=meta,
            overlay=ov,
        )
