from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from apps.api.method_registry import (
    ApiMethodRegistry,
    MethodRegistryError,
    ResolvedApiMethod,
    load_api_method_registry,
)
from apps.api.schemas import MatchMethod, MatchResponse, Overlay, OverlayMatch
from src.fpbench.matchers.baseline_dl import BaselineDL, DLBaselineConfig
from src.fpbench.identification.classic_vectorizers import (
    orb_aggregated_descriptor_vector,
    sift_aggregated_descriptor_vector,
)
from src.fpbench.matchers.dedicated_matcher import DedicatedMatcher
from src.fpbench.matchers.minutiae_matcher import (
    MinutiaeConfig,
    extract_minutiae_template,
    minutiae_aggregate_vector,
    score_pair_minutiae,
)
from src.fpbench.matchers.matching_baseline import (
    HarrisConfig,
    ORBConfig,
    SIFTConfig,
    harris_extract,
    match_harris,
    match_orb,
    match_sift,
    orb_extract,
    ransac_inliers_for_model,
    score_local_feature_counts,
    sift_extract,
)
from src.fpbench.preprocess.preprocess import PreprocessConfig, load_gray, preprocess_image


class MethodUnavailableError(RuntimeError):
    """Raised when a method is known to exist but is not currently usable."""


_DL_CONFIG_KEYS = (
    "backbone",
    "input_size",
    "use_mask",
    "roi_min_frac",
    "roi_max_frac",
    "gate_top_plain",
    "gate_top_roll",
    "gate_border",
)


def _infer_capture_from_filename(name: str) -> Optional[str]:
    s = (name or "").lower()
    if "roll" in s:
        return "roll"
    if "plain" in s:
        return "plain"
    return None


def _normalize_capture_label(raw: Optional[str], *, fallback_name: str = "") -> str:
    aliases = {
        "plain": "plain",
        "roll": "roll",
        "rolled": "roll",
        "contactless": "contactless",
        "contact-less": "contactless",
        "contact_less": "contactless",
        "contact_based": "contact_based",
        "contact-based": "contact_based",
        "contactbased": "contact_based",
    }
    s = str(raw or "").strip().lower()
    if not s and fallback_name:
        s = str(_infer_capture_from_filename(fallback_name) or "").strip().lower()
    if not s:
        return "plain"
    if s not in aliases:
        raise ValueError(f"Unsupported capture label: {raw}")
    return aliases[s]


def _copy_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(payload)


def _extract_kwargs(source: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
    return {key: copy.deepcopy(source[key]) for key in keys if key in source}


def _tuple_grid(value: Any, default: tuple[int, int] = (8, 8)) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return default


def _resize_long_edge(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    current = max(h, w)
    if current <= int(long_edge):
        return img
    scale = float(long_edge) / float(current)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _clahe_only(img: np.ndarray, *, clip: float, grid: tuple[int, int]) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=grid)
    return clahe.apply(img)


def _largest_cc(binary_255: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_255 > 0).astype(np.uint8),
        connectivity=8,
    )
    if num <= 1:
        return binary_255
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_255)
    out[labels == idx] = 255
    return out


def _make_paper_mask(img: np.ndarray) -> np.ndarray:
    paper = (img > 10).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    paper = cv2.morphologyEx(paper, cv2.MORPH_CLOSE, kernel, iterations=1)
    return _largest_cc(paper)


def _make_benchmark_roi_mask(img: np.ndarray) -> np.ndarray:
    paper = _make_paper_mask(img)

    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    energy = cv2.GaussianBlur(mag_u8, (0, 0), sigmaX=7, sigmaY=7)

    vals = energy[paper > 0]
    if vals.size == 0:
        return paper

    thr = np.percentile(vals, 80)
    roi = (energy >= thr).astype(np.uint8) * 255
    roi[paper == 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    roi = _largest_cc(roi)

    frac = float((roi > 0).mean())
    if frac < 0.01 or frac > 0.90:
        return paper

    return roi


def _gftt_keypoints(
    img: np.ndarray,
    mask: np.ndarray,
    *,
    max_points: int,
    quality: float = 0.01,
    min_dist: float = 5.0,
) -> list[cv2.KeyPoint]:
    corners = cv2.goodFeaturesToTrack(
        img,
        maxCorners=int(max_points),
        qualityLevel=float(quality),
        minDistance=float(min_dist),
        mask=(mask > 0).astype(np.uint8),
        blockSize=3,
        useHarrisDetector=False,
    )
    if corners is None:
        return []
    corners = corners.reshape(-1, 2)
    return [cv2.KeyPoint(float(x), float(y), 31) for x, y in corners]


def _affine_partial_inlier_mask(
    kps1: list[cv2.KeyPoint],
    kps2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
    *,
    ransac_thresh: float,
) -> tuple[int, np.ndarray | None]:
    if len(matches) < 8:
        return 0, None

    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    _, inlier_mask = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(ransac_thresh),
        maxIters=2000,
        confidence=0.99,
    )
    if inlier_mask is None:
        return 0, None
    flat_mask = inlier_mask.reshape(-1).astype(np.uint8)
    return int(flat_mask.sum()), flat_mask


def _build_preprocess_config(registry: ApiMethodRegistry) -> PreprocessConfig:
    defaults = _copy_dict(registry.preprocess_defaults)
    dl_runtime = _copy_dict(registry.definition_for("dl").runtime_defaults)
    preprocess_fallback = _copy_dict(dl_runtime.get("preprocess", {}) or {})
    for key, value in preprocess_fallback.items():
        defaults.setdefault(key, value)
    if "clahe_grid" in defaults and isinstance(defaults["clahe_grid"], list):
        defaults["clahe_grid"] = tuple(defaults["clahe_grid"])
    return PreprocessConfig(**defaults)


class MatchService:
    def __init__(
        self,
        *,
        method_registry: ApiMethodRegistry | None = None,
        dedicated_factory: Callable[..., DedicatedMatcher] | None = None,
        dl_factory: Callable[..., BaselineDL] | None = None,
    ):
        self.method_registry = method_registry or load_api_method_registry()
        self.prep_cfg = _build_preprocess_config(self.method_registry)

        classic_orb_defaults = _copy_dict(self.method_registry.definition_for("classic_orb").runtime_defaults)
        classic_gftt_orb_defaults = _copy_dict(self.method_registry.definition_for("classic_gftt_orb").runtime_defaults)
        minutiae_defaults = _copy_dict(self.method_registry.definition_for("minutiae").runtime_defaults)
        harris_defaults = _copy_dict(self.method_registry.definition_for("harris").runtime_defaults)
        sift_defaults = _copy_dict(self.method_registry.definition_for("sift").runtime_defaults)
        sift_plain_roll_v2_defaults = _copy_dict(
            self.method_registry.definition_for("sift_plain_roll_v2").runtime_defaults
        )
        dl_defaults = _copy_dict(self.method_registry.definition_for("dl").runtime_defaults)
        vit_defaults = _copy_dict(self.method_registry.definition_for("vit").runtime_defaults)
        dedicated_defaults = _copy_dict(self.method_registry.definition_for("dedicated").runtime_defaults)

        self.orb_cfg = ORBConfig(
            **_extract_kwargs(
                classic_orb_defaults,
                ("nfeatures", "scaleFactor", "nlevels", "edgeThreshold", "fastThreshold"),
            )
        )
        self.harris_cfg = HarrisConfig(
            **_extract_kwargs(
                harris_defaults,
                (
                    "nfeatures",
                    "block_size",
                    "ksize",
                    "k",
                    "thresh_rel",
                    "max_points",
                    "min_distance",
                    "orb_edge_threshold",
                    "orb_fast_threshold",
                ),
            )
        )
        self.sift_cfg = SIFTConfig(
            **_extract_kwargs(
                sift_defaults,
                ("nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold", "sigma"),
            )
        )
        self.sift_plain_roll_v2_cfg = SIFTConfig(
            **_extract_kwargs(
                sift_plain_roll_v2_defaults,
                ("nfeatures", "nOctaveLayers", "contrastThreshold", "edgeThreshold", "sigma"),
            )
        )
        self.sift_plain_roll_v2_prep_cfg = PreprocessConfig(
            target_size=int(sift_plain_roll_v2_defaults.get("target_size", 768)),
            clahe_clip=float(sift_plain_roll_v2_defaults.get("clahe_clip", self.prep_cfg.clahe_clip)),
            clahe_grid=_tuple_grid(sift_plain_roll_v2_defaults.get("clahe_grid"), self.prep_cfg.clahe_grid),
            blur_ksize=int(sift_plain_roll_v2_defaults.get("blur_ksize", 0)),
        )
        if "clahe_grid" in minutiae_defaults and isinstance(minutiae_defaults["clahe_grid"], list):
            minutiae_defaults["clahe_grid"] = tuple(minutiae_defaults["clahe_grid"])
        self.minutiae_cfg = MinutiaeConfig(
            **_extract_kwargs(
                minutiae_defaults,
                (
                    "target_size",
                    "clahe_clip",
                    "clahe_grid",
                    "blur_ksize",
                    "preprocess",
                    "ridge_kernel_size",
                    "min_ridge_response",
                    "min_binary_component_area",
                    "min_skeleton_component_area",
                    "border_px",
                    "suppress_top_ratio",
                    "roi_border_px",
                    "min_distance",
                    "orientation_radius",
                    "max_minutiae",
                    "max_thinning_iterations",
                    "candidate_quality_floor",
                    "dense_bifurcation_radius",
                    "max_dense_bifurcation_density",
                    "bifurcation_suppression_scale",
                    "spatial_tolerance",
                    "angle_tolerance_deg",
                    "anchor_limit",
                    "min_required_minutiae",
                    "allow_kind_mismatch",
                    "kind_mismatch_weight",
                    "orientation_periodic_pi",
                    "max_tentative_return",
                ),
            )
        )

        self._classic_orb_match_defaults = {
            "ratio": float(classic_orb_defaults.get("ratio", 0.75)),
            "reproj": float(classic_orb_defaults.get("reproj_threshold", 3.0)),
        }
        self._classic_gftt_orb_defaults = {
            "nfeatures": int(classic_gftt_orb_defaults.get("nfeatures", 1500)),
            "long_edge": int(classic_gftt_orb_defaults.get("long_edge", 512)),
            "ratio": float(classic_gftt_orb_defaults.get("ratio", 0.75)),
            "ransac_thresh": float(classic_gftt_orb_defaults.get("ransac_thresh", 4.0)),
            "clahe_clip": float(classic_gftt_orb_defaults.get("clahe_clip", 2.0)),
            "clahe_grid": _tuple_grid(classic_gftt_orb_defaults.get("clahe_grid")),
        }
        self._harris_match_defaults = {
            "ratio": float(harris_defaults.get("ratio", 0.75)),
            "reproj": float(harris_defaults.get("reproj_threshold", 3.0)),
        }
        self._sift_match_defaults = {
            "ratio": float(sift_defaults.get("ratio", 0.75)),
            "reproj": float(sift_defaults.get("reproj_threshold", 3.0)),
        }
        self._sift_plain_roll_v2_match_defaults = {
            "ratio": float(sift_plain_roll_v2_defaults.get("ratio", 0.75)),
            "reproj": float(
                sift_plain_roll_v2_defaults.get(
                    "reproj_threshold",
                    sift_plain_roll_v2_defaults.get("ransac_thresh", 3.0),
                )
            ),
            "ransac_model": str(sift_plain_roll_v2_defaults.get("ransac_model", "affine_full_2d")),
            "score_mode": str(
                sift_plain_roll_v2_defaults.get(
                    "score_mode",
                    "inliers_times_inlier_ratio_times_log1p_matches",
                )
            ),
        }

        self.dl_resnet: BaselineDL | None = None
        self.dl_vit: BaselineDL | None = None
        self._dl_factory = dl_factory or BaselineDL
        self._dedicated_factory = dedicated_factory or DedicatedMatcher
        self._dedicated_defaults = _extract_kwargs(
            dedicated_defaults,
            ("patch", "max_kpts", "mask_cov_thr", "max_matches", "ransac_thresh", "nn_delta", "nn_sim_min"),
        )
        self.dedicated: DedicatedMatcher | None = None
        self._method_availability: Dict[str, Dict[str, Optional[str] | bool]] = {
            definition.canonical_api_name: {"available": True, "error": None}
            for definition in self.method_registry.list_methods()
        }
        self._probe_dl("dl", dl_defaults)
        self._probe_dl("vit", vit_defaults)
        self._probe_dedicated()

    def _probe_dl(self, method: str, defaults: Dict[str, Any]) -> None:
        target_attr = "dl_vit" if method == "vit" else "dl_resnet"
        try:
            model = self._dl_factory(
                dl_cfg=DLBaselineConfig(**_extract_kwargs(defaults, _DL_CONFIG_KEYS)),
                prep_cfg=self.prep_cfg,
            )
        except Exception as exc:
            setattr(self, target_attr, None)
            self._method_availability[method] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            setattr(self, target_attr, model)
            self._method_availability[method] = {"available": True, "error": None}

    def _probe_dedicated(self) -> None:
        try:
            self.dedicated = self._dedicated_factory(cfg=self.prep_cfg, **self._dedicated_defaults)
        except Exception as exc:
            self.dedicated = None
            self._method_availability["dedicated"] = {
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            self._method_availability["dedicated"] = {"available": True, "error": None}

    def method_availability(self) -> Dict[str, Dict[str, Optional[str] | bool]]:
        return copy.deepcopy(self._method_availability)

    def _resolve_method(self, method: MatchMethod | str) -> ResolvedApiMethod:
        try:
            return self.method_registry.resolve(method)
        except MethodRegistryError as exc:
            raise ValueError(str(exc)) from exc

    def _ensure_method_available(self, resolved: ResolvedApiMethod) -> None:
        availability = self._method_availability.get(
            resolved.canonical_api_name,
            {"available": False, "error": "No availability information is registered."},
        )
        if bool(availability.get("available", False)):
            return
        error = str(availability.get("error") or "Unknown runtime initialization error.")
        raise MethodUnavailableError(
            f"Method {resolved.canonical_api_name!r} ({resolved.ui_label}) is unavailable: {error}"
        )

    def ensure_method_available(self, method: MatchMethod | str) -> None:
        self._ensure_method_available(self._resolve_method(method))

    def _preprocess_path(self, path: str) -> np.ndarray:
        gray = load_gray(path)
        return preprocess_image(gray, self.prep_cfg)

    def _benchmark_classic_preprocess_path(self, path: str) -> np.ndarray:
        gray = load_gray(path)
        resized = _resize_long_edge(gray, self._classic_gftt_orb_defaults["long_edge"])
        return _clahe_only(
            resized,
            clip=self._classic_gftt_orb_defaults["clahe_clip"],
            grid=self._classic_gftt_orb_defaults["clahe_grid"],
        )

    def _classic_gftt_orb_extract(
        self,
        img_u8: np.ndarray,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None, np.ndarray]:
        roi = _make_benchmark_roi_mask(img_u8)
        keypoints = _gftt_keypoints(
            img_u8,
            roi,
            max_points=min(1200, int(self._classic_gftt_orb_defaults["nfeatures"])),
        )
        if not keypoints:
            return [], None, roi

        orb = cv2.ORB_create(nfeatures=int(self._classic_gftt_orb_defaults["nfeatures"]))
        keypoints, desc = orb.compute(img_u8, keypoints)
        return keypoints or [], desc, roi

    def embed_classic_orb_path(self, path: str, capture: Optional[str] = None) -> np.ndarray:
        del capture
        img = self._preprocess_path(path)
        keypoints, descriptors = orb_extract(img, None, self.orb_cfg)
        return orb_aggregated_descriptor_vector(keypoints, descriptors, img.shape)

    def embed_classic_gftt_orb_path(self, path: str, capture: Optional[str] = None) -> np.ndarray:
        del capture
        img = self._benchmark_classic_preprocess_path(path)
        keypoints, descriptors, _roi = self._classic_gftt_orb_extract(img)
        return orb_aggregated_descriptor_vector(keypoints, descriptors, img.shape)

    def embed_harris_path(self, path: str, capture: Optional[str] = None) -> np.ndarray:
        del capture
        img = self._preprocess_path(path)
        keypoints, descriptors = harris_extract(img, None, self.harris_cfg)
        return orb_aggregated_descriptor_vector(keypoints, descriptors, img.shape)

    def embed_sift_path(self, path: str, capture: Optional[str] = None) -> np.ndarray:
        del capture
        img = self._preprocess_path(path)
        _keypoints, descriptors = sift_extract(img, None, self.sift_cfg)
        return sift_aggregated_descriptor_vector(descriptors)

    def embed_minutiae_path(self, path: str, capture: Optional[str] = None) -> np.ndarray:
        del capture
        gray = load_gray(path)
        template = extract_minutiae_template(gray, mask=None, cfg=self.minutiae_cfg)
        return minutiae_aggregate_vector(template, template.image_shape)

    def _classic_score_and_overlay(
        self,
        path_a: str,
        path_b: str,
        *,
        return_overlay: bool,
        max_draw: int = 200,
        ratio: float,
        reproj: float,
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
            for i, match in enumerate(take):
                a = kps1[match.queryIdx].pt
                b = kps2[match.trainIdx].pt
                if mask is None:
                    kind = "tentative"
                else:
                    kind = "inlier" if int(mask[i]) == 1 else "outlier"
                out.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind=kind,
                    )
                )
            ov = Overlay(matches=out)

        return score, meta, ov

    def _classic_gftt_orb_score_and_overlay(
        self,
        path_a: str,
        path_b: str,
        *,
        return_overlay: bool,
        max_draw: int = 200,
        ratio: float,
        ransac_thresh: float,
    ) -> Tuple[float, Dict[str, Any], Optional[Overlay]]:
        img1 = self._benchmark_classic_preprocess_path(path_a)
        img2 = self._benchmark_classic_preprocess_path(path_b)

        kps1, desc1, roi1 = self._classic_gftt_orb_extract(img1)
        kps2, desc2, roi2 = self._classic_gftt_orb_extract(img2)

        if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
            meta = {
                "inliers": 0,
                "matches": 0,
                "k1": len(kps1),
                "k2": len(kps2),
                "score_mode": "inliers_over_k",
                "normalization_k": int(self._classic_gftt_orb_defaults["nfeatures"]),
                "detector": "gftt_orb",
                "geometry_model": "affine_partial_2d",
                "roi_fraction_a": float((roi1 > 0).mean()),
                "roi_fraction_b": float((roi2 > 0).mean()),
            }
            return 0.0, meta, (Overlay(matches=[]) if return_overlay else None)

        matches = match_orb(desc1, desc2, ratio=ratio)
        inliers, mask = _affine_partial_inlier_mask(
            kps1,
            kps2,
            matches,
            ransac_thresh=ransac_thresh,
        )

        norm_k = max(1, int(self._classic_gftt_orb_defaults["nfeatures"]))
        score = float(inliers) / float(norm_k)
        meta = {
            "inliers": int(inliers),
            "matches": int(len(matches)),
            "k1": len(kps1),
            "k2": len(kps2),
            "score_mode": "inliers_over_k",
            "normalization_k": norm_k,
            "detector": "gftt_orb",
            "geometry_model": "affine_partial_2d",
            "roi_fraction_a": float((roi1 > 0).mean()),
            "roi_fraction_b": float((roi2 > 0).mean()),
        }

        overlay = None
        if return_overlay:
            rendered_matches: List[OverlayMatch] = []
            for index, match in enumerate(matches[:max_draw]):
                a = kps1[match.queryIdx].pt
                b = kps2[match.trainIdx].pt
                if mask is None:
                    kind = "tentative"
                else:
                    kind = "inlier" if int(mask[index]) == 1 else "outlier"
                rendered_matches.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind=kind,
                    )
                )
            overlay = Overlay(matches=rendered_matches)

        return score, meta, overlay

    def _harris_score_and_overlay(
        self,
        path_a: str,
        path_b: str,
        *,
        return_overlay: bool,
        max_draw: int = 200,
        ratio: float,
        reproj: float,
    ) -> Tuple[float, Dict[str, Any], Optional[Overlay]]:
        img1 = self._preprocess_path(path_a)
        img2 = self._preprocess_path(path_b)

        kps1, desc1 = harris_extract(img1, None, self.harris_cfg)
        kps2, desc2 = harris_extract(img2, None, self.harris_cfg)

        if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
            meta = {
                "inliers": 0,
                "matches": 0,
                "k1": len(kps1) if kps1 else 0,
                "k2": len(kps2) if kps2 else 0,
            }
            return 0.0, meta, (Overlay(matches=[]) if return_overlay else None)

        matches = match_harris(desc1, desc2, ratio=ratio)

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
            for i, match in enumerate(take):
                a = kps1[match.queryIdx].pt
                b = kps2[match.trainIdx].pt
                if mask is None:
                    kind = "tentative"
                else:
                    kind = "inlier" if int(mask[i]) == 1 else "outlier"
                out.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind=kind,
                    )
                )
            ov = Overlay(matches=out)

        return score, meta, ov

    def _sift_score_and_overlay(
        self,
        path_a: str,
        path_b: str,
        *,
        return_overlay: bool,
        max_draw: int = 200,
        ratio: float,
        reproj: float,
        sift_cfg: SIFTConfig | None = None,
        preprocess_cfg: PreprocessConfig | None = None,
        score_mode: str = "inliers_over_min_keypoints",
        ransac_model: str = "homography",
    ) -> Tuple[float, Dict[str, Any], Optional[Overlay]]:
        img1 = (
            preprocess_image(load_gray(path_a), preprocess_cfg)
            if preprocess_cfg is not None
            else self._preprocess_path(path_a)
        )
        img2 = (
            preprocess_image(load_gray(path_b), preprocess_cfg)
            if preprocess_cfg is not None
            else self._preprocess_path(path_b)
        )

        cfg = sift_cfg or self.sift_cfg
        kps1, desc1 = sift_extract(img1, None, cfg)
        kps2, desc2 = sift_extract(img2, None, cfg)

        if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
            meta = {
                "inliers": 0,
                "matches": 0,
                "k1": len(kps1) if kps1 else 0,
                "k2": len(kps2) if kps2 else 0,
            }
            return 0.0, meta, (Overlay(matches=[]) if return_overlay else None)

        matches = match_sift(desc1, desc2, ratio=ratio)

        mask = None
        inliers = 0
        min_matches = 8 if str(ransac_model) == "homography" else 3
        if len(matches) >= min_matches:
            inliers, mask = ransac_inliers_for_model(
                kps1,
                kps2,
                matches,
                ransac_model=str(ransac_model),
                ransac_thresh=float(reproj),
            )

        score = score_local_feature_counts(
            score_mode=str(score_mode),
            inliers=int(inliers),
            matches=int(len(matches)),
            k1=len(kps1),
            k2=len(kps2),
            normalization_k=int(cfg.nfeatures),
        )
        meta = {
            "inliers": int(inliers),
            "matches": int(len(matches)),
            "k1": len(kps1),
            "k2": len(kps2),
        }
        if str(score_mode) != "inliers_over_min_keypoints" or str(ransac_model) != "homography":
            meta.update(
                {
                    "score_mode": str(score_mode),
                    "geometry_model": str(ransac_model),
                    "inlier_ratio": float(inliers) / float(max(len(matches), 1)),
                    "nfeatures": int(cfg.nfeatures),
                    "target_size": int(preprocess_cfg.target_size) if preprocess_cfg is not None else self.prep_cfg.target_size,
                    "blur_ksize": int(preprocess_cfg.blur_ksize) if preprocess_cfg is not None else self.prep_cfg.blur_ksize,
                }
            )

        ov = None
        if return_overlay:
            out: List[OverlayMatch] = []
            take = matches[:max_draw]
            for i, match in enumerate(take):
                a = kps1[match.queryIdx].pt
                b = kps2[match.trainIdx].pt
                if mask is None:
                    kind = "tentative"
                else:
                    kind = "inlier" if int(mask[i]) == 1 else "outlier"
                out.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind=kind,
                    )
                )
            ov = Overlay(matches=out)

        return score, meta, ov

    def _minutiae_score_and_overlay(
        self,
        path_a: str,
        path_b: str,
        *,
        return_overlay: bool,
        max_draw: int = 200,
    ) -> Tuple[float, Dict[str, Any], Optional[Overlay]]:
        img1 = load_gray(path_a)
        img2 = load_gray(path_b)
        result = score_pair_minutiae(img1, img2, mask1=None, mask2=None, cfg=self.minutiae_cfg)

        meta = {
            "matched_minutiae": int(result.matched_count),
            "tentative_minutiae": int(result.tentative_count),
            "minutiae_count_a": int(result.minutiae_count_a),
            "minutiae_count_b": int(result.minutiae_count_b),
            "endings_a": int(result.endings_a),
            "endings_b": int(result.endings_b),
            "bifurcations_a": int(result.bifurcations_a),
            "bifurcations_b": int(result.bifurcations_b),
            "spatial_tolerance": float(result.spatial_tolerance),
            "angle_tolerance_deg": float(result.angle_tolerance_deg),
            "transform_angle_deg": result.transform_angle_deg,
            "transform_dx": result.transform_dx,
            "transform_dy": result.transform_dy,
            "raw_alignment_score": float(result.raw_alignment_score),
            "score_multiplier": float(result.score_multiplier),
            "score_components": dict(result.score_components),
            "skeleton_foreground_pixels_a": int(result.skeleton_foreground_pixels_a),
            "skeleton_foreground_pixels_b": int(result.skeleton_foreground_pixels_b),
            "skeleton_density_a": float(result.skeleton_density_a),
            "skeleton_density_b": float(result.skeleton_density_b),
            "raw_candidate_endings_a": int(result.raw_candidate_endings_a),
            "raw_candidate_endings_b": int(result.raw_candidate_endings_b),
            "raw_candidate_bifurcations_a": int(result.raw_candidate_bifurcations_a),
            "raw_candidate_bifurcations_b": int(result.raw_candidate_bifurcations_b),
            "saturated_by_max_minutiae_a": bool(result.saturated_by_max_minutiae_a),
            "saturated_by_max_minutiae_b": bool(result.saturated_by_max_minutiae_b),
            "ridge_polarity_a": result.ridge_polarity_a,
            "ridge_polarity_b": result.ridge_polarity_b,
            "extraction_quality_flags_a": list(result.extraction_quality_flags_a),
            "extraction_quality_flags_b": list(result.extraction_quality_flags_b),
            "score_mode": "quality_penalized_matched_minutiae_over_min_template_count_floor",
            "method_semantics_epoch": "minutiae_crossing_number_aligned_v2",
        }

        overlay = None
        if return_overlay:
            rendered = []
            for match in result.matched_minutiae[:max_draw]:
                a = match["a"]
                b = match["b"]
                sim = 1.0 - min(float(match.get("distance", 0.0)) / max(float(result.spatial_tolerance), 1.0), 1.0)
                rendered.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind="inlier",
                        sim=float(sim),
                    )
                )
            overlay = Overlay(matches=rendered)

        return float(result.score), meta, overlay

    def match(
        self,
        *,
        method: MatchMethod | str,
        path_a: str,
        path_b: str,
        threshold: Optional[float],
        return_overlay: bool,
        capture_a: Optional[str],
        capture_b: Optional[str],
        filename_a: Optional[str],
        filename_b: Optional[str],
    ) -> MatchResponse:
        resolved_method = self._resolve_method(method)
        self._ensure_method_available(resolved_method)

        method_enum = MatchMethod(resolved_method.canonical_api_name)
        th = float(threshold) if threshold is not None else float(resolved_method.decision_threshold)
        method_metadata = resolved_method.to_metadata()

        if method_enum == MatchMethod.classic_orb:
            t0 = time.perf_counter()
            score, meta, ov = self._classic_score_and_overlay(
                path_a,
                path_b,
                return_overlay=return_overlay,
                ratio=float(self._classic_orb_match_defaults["ratio"]),
                reproj=float(self._classic_orb_match_defaults["reproj"]),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
                method_metadata=method_metadata,
            )

        if method_enum == MatchMethod.classic_gftt_orb:
            t0 = time.perf_counter()
            score, meta, ov = self._classic_gftt_orb_score_and_overlay(
                path_a,
                path_b,
                return_overlay=return_overlay,
                ratio=float(self._classic_gftt_orb_defaults["ratio"]),
                ransac_thresh=float(self._classic_gftt_orb_defaults["ransac_thresh"]),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
                method_metadata=method_metadata,
            )

        if method_enum == MatchMethod.harris:
            t0 = time.perf_counter()
            score, meta, ov = self._harris_score_and_overlay(
                path_a,
                path_b,
                return_overlay=return_overlay,
                ratio=float(self._harris_match_defaults["ratio"]),
                reproj=float(self._harris_match_defaults["reproj"]),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
                method_metadata=method_metadata,
            )

        if method_enum == MatchMethod.sift:
            t0 = time.perf_counter()
            score, meta, ov = self._sift_score_and_overlay(
                path_a,
                path_b,
                return_overlay=return_overlay,
                ratio=float(self._sift_match_defaults["ratio"]),
                reproj=float(self._sift_match_defaults["reproj"]),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
                method_metadata=method_metadata,
            )

        if method_enum == MatchMethod.sift_plain_roll_v2:
            t0 = time.perf_counter()
            score, meta, ov = self._sift_score_and_overlay(
                path_a,
                path_b,
                return_overlay=return_overlay,
                ratio=float(self._sift_plain_roll_v2_match_defaults["ratio"]),
                reproj=float(self._sift_plain_roll_v2_match_defaults["reproj"]),
                sift_cfg=self.sift_plain_roll_v2_cfg,
                preprocess_cfg=self.sift_plain_roll_v2_prep_cfg,
                score_mode=str(self._sift_plain_roll_v2_match_defaults["score_mode"]),
                ransac_model=str(self._sift_plain_roll_v2_match_defaults["ransac_model"]),
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
                method_metadata=method_metadata,
            )

        if method_enum == MatchMethod.minutiae:
            t0 = time.perf_counter()
            score, meta, ov = self._minutiae_score_and_overlay(
                path_a,
                path_b,
                return_overlay=return_overlay,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=ov,
                method_metadata=method_metadata,
            )

        if method_enum in (MatchMethod.dl, MatchMethod.vit):
            t0 = time.perf_counter()
            model = self.dl_vit if method_enum == MatchMethod.vit else self.dl_resnet
            if model is None:
                self._ensure_method_available(resolved_method)
                raise MethodUnavailableError(f"Method {method_enum.value!r} is unavailable.")
            cap_a = _normalize_capture_label(capture_a, fallback_name=filename_a or path_a)
            cap_b = _normalize_capture_label(capture_b, fallback_name=filename_b or path_b)
            emb_a, ms_a = model.embed_path(path_a, capture=cap_a)
            emb_b, ms_b = model.embed_path(path_b, capture=cap_b)
            score = float(model.cosine(emb_a, emb_b))
            latency_ms = (time.perf_counter() - t0) * 1000.0

            meta = {
                "embed_ms_a": float(ms_a),
                "embed_ms_b": float(ms_b),
                "dl_config": model.config_dict(),
            }
            return MatchResponse(
                method=method_enum,
                score=score,
                decision=bool(score >= th),
                threshold=th,
                latency_ms=float(latency_ms),
                meta=meta,
                overlay=None,
                method_metadata=method_metadata,
            )

        cap_a = _normalize_capture_label(capture_a, fallback_name=filename_a or path_a)
        cap_b = _normalize_capture_label(capture_b, fallback_name=filename_b or path_b)
        if self.dedicated is None:
            self._ensure_method_available(resolved_method)

        t0 = time.perf_counter()
        res = self.dedicated.score_pair(path_a, path_b, capture_a=cap_a, capture_b=cap_b)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        ov = None
        if return_overlay:
            out = []
            for match in res.matches:
                a = match["pt_a"]
                b = match["pt_b"]
                kind = "inlier" if bool(match.get("inlier", False)) else "outlier"
                out.append(
                    OverlayMatch(
                        a=(float(a[0]), float(a[1])),
                        b=(float(b[0]), float(b[1])),
                        kind=kind,
                        sim=float(match.get("sim", 0.0)),
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
            method=method_enum,
            score=float(res.score),
            decision=bool(res.score >= th),
            threshold=th,
            latency_ms=float(latency_ms),
            meta=meta,
            overlay=ov,
            method_metadata=method_metadata,
        )
