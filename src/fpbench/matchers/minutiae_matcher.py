from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

from src.fpbench.preprocess.preprocess import (
    PreprocessConfig,
    extract_fingerprint_roi,
    preprocess_image,
    suppress_header_and_borders,
)

TAU = 2.0 * math.pi
RETRIEVAL_VECTOR_DIM = 512


@dataclass(frozen=True)
class MinutiaeConfig:
    target_size: int = 512
    clahe_clip: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)
    blur_ksize: int = 3
    preprocess: bool = True
    ridge_kernel_size: int = 15
    min_ridge_response: float = 5.0
    min_binary_component_area: int = 12
    min_skeleton_component_area: int = 8
    border_px: int = 16
    suppress_top_ratio: float = 0.03
    roi_border_px: int = 10
    min_distance: float = 8.0
    orientation_radius: int = 9
    max_minutiae: int = 96
    max_thinning_iterations: int = 80
    candidate_quality_floor: float = 0.20
    dense_bifurcation_radius: int = 5
    max_dense_bifurcation_density: float = 0.24
    bifurcation_suppression_scale: float = 1.35
    spatial_tolerance: float = 14.0
    angle_tolerance_deg: float = 30.0
    anchor_limit: int = 28
    min_required_minutiae: int = 12
    allow_kind_mismatch: bool = False
    kind_mismatch_weight: float = 0.65
    orientation_periodic_pi: bool = True
    max_tentative_return: int = 256


@dataclass(frozen=True)
class MinutiaePoint:
    x: float
    y: float
    theta: float
    kind: str
    quality: float = 1.0


@dataclass(frozen=True)
class MinutiaeTemplate:
    points: tuple[MinutiaePoint, ...]
    image_shape: tuple[int, int]
    roi_fraction: float = 0.0
    roi_bounds: tuple[int, int, int, int] | None = None
    skeleton_pixels: int = 0
    skeleton_foreground_pixels: int = 0
    skeleton_density: float = 0.0
    raw_candidate_endings: int = 0
    raw_candidate_bifurcations: int = 0
    kept_endings: int = 0
    kept_bifurcations: int = 0
    total_kept_minutiae: int = 0
    saturated_by_max_minutiae: bool = False
    ridge_polarity: str = "unknown"
    polarity_used: str = "unknown"
    extraction_quality_flags: tuple[str, ...] = field(default_factory=tuple)
    extraction_warnings: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MinutiaeMatchResult:
    score: float
    matched_count: int
    tentative_count: int
    minutiae_count_a: int
    minutiae_count_b: int
    endings_a: int
    endings_b: int
    bifurcations_a: int
    bifurcations_b: int
    spatial_tolerance: float
    angle_tolerance_deg: float
    transform_angle_deg: float | None
    transform_dx: float | None
    transform_dy: float | None
    matched_minutiae: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    tentative_minutiae: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    raw_alignment_score: float = 0.0
    score_multiplier: float = 1.0
    score_components: dict[str, float] = field(default_factory=dict)
    skeleton_foreground_pixels_a: int = 0
    skeleton_foreground_pixels_b: int = 0
    skeleton_density_a: float = 0.0
    skeleton_density_b: float = 0.0
    raw_candidate_endings_a: int = 0
    raw_candidate_endings_b: int = 0
    raw_candidate_bifurcations_a: int = 0
    raw_candidate_bifurcations_b: int = 0
    saturated_by_max_minutiae_a: bool = False
    saturated_by_max_minutiae_b: bool = False
    ridge_polarity_a: str = "unknown"
    ridge_polarity_b: str = "unknown"
    extraction_quality_flags_a: tuple[str, ...] = field(default_factory=tuple)
    extraction_quality_flags_b: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class _CandidateExtraction:
    points: tuple[MinutiaePoint, ...]
    raw_candidate_endings: int
    raw_candidate_bifurcations: int
    rejected_dense_bifurcations: int
    rejected_low_quality: int


@dataclass(frozen=True)
class _ExtractionProfile:
    polarity: str
    skeleton: np.ndarray
    extraction: _CandidateExtraction
    skeleton_foreground_pixels: int
    skeleton_density: float
    saturated_by_max_minutiae: bool
    quality_flags: tuple[str, ...]
    plausibility_score: float


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.dtype == np.uint8:
        return arr.copy()
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _normalized_angle(theta: float) -> float:
    return float(theta % TAU)


def _angle_delta(a: float, b: float, *, periodic_pi: bool = False) -> float:
    diff = abs((float(a) - float(b) + math.pi) % TAU - math.pi)
    if periodic_pi:
        diff = min(diff, abs(diff - math.pi))
    return float(diff)


def _odd_kernel_size(value: int, minimum: int = 3) -> int:
    size = max(int(value), minimum)
    return size if size % 2 == 1 else size + 1


def _largest_cc(binary_255: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_255 > 0).astype(np.uint8),
        connectivity=8,
    )
    if num <= 1:
        return np.zeros_like(binary_255, dtype=np.uint8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == idx).astype(np.uint8) * 255).astype(np.uint8, copy=False)


def _roi_bounds(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _texture_roi_mask(img_u8: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if float(np.max(mag)) <= 0.0:
        return np.zeros_like(img_u8, dtype=np.uint8)

    mag_u8 = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    energy = cv2.GaussianBlur(mag_u8, (0, 0), sigmaX=7, sigmaY=7)
    positive = energy[energy > 0]
    if positive.size == 0:
        return np.zeros_like(img_u8, dtype=np.uint8)

    threshold = float(np.percentile(positive, 60))
    roi = (energy >= threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=1)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_cc(roi)


def _prepare_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    m = np.asarray(mask)
    if m.ndim == 3:
        m = cv2.cvtColor(m.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if m.shape[:2] != shape:
        m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return ((m > 0).astype(np.uint8) * 255).astype(np.uint8, copy=False)


def _make_roi_mask(img_u8: np.ndarray, mask: np.ndarray | None, cfg: MinutiaeConfig) -> tuple[np.ndarray, list[str]]:
    warnings: list[str] = []
    supplied = _prepare_mask(mask, img_u8.shape[:2])

    roi_result = extract_fingerprint_roi(img_u8)
    roi = roi_result.mask if roi_result.is_valid else np.zeros_like(img_u8, dtype=np.uint8)
    if not roi_result.is_valid:
        warnings.append(f"roi_helper:{roi_result.failure_reason or 'invalid'}")

    texture_roi = _texture_roi_mask(img_u8)
    texture_fraction = float(np.mean(texture_roi > 0)) if texture_roi.size else 0.0
    if texture_fraction >= 0.005:
        if float(np.mean(roi > 0)) >= 0.005:
            roi = cv2.bitwise_or(roi, texture_roi)
            roi = _largest_cc(roi)
        else:
            roi = texture_roi

    roi_fraction = float(np.mean(roi > 0)) if roi.size else 0.0
    if roi_fraction < 0.005 or roi_fraction > 0.95:
        warnings.append("roi_fallback:rectangular_gate")
        roi = np.full_like(img_u8, 255, dtype=np.uint8)

    roi = suppress_header_and_borders(
        roi,
        top_ratio=float(cfg.suppress_top_ratio),
        border=int(cfg.border_px),
    )
    if supplied is not None:
        roi = cv2.bitwise_and(roi, supplied)

    if float(np.mean(roi > 0)) < 0.005:
        warnings.append("roi_fallback:empty_after_gate")
        roi = np.zeros_like(img_u8, dtype=np.uint8)

    return roi, warnings


def _remove_small_components(binary_255: np.ndarray, min_area: int) -> np.ndarray:
    binary = (binary_255 > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return np.zeros_like(binary_255, dtype=np.uint8)
    out = np.zeros_like(binary_255, dtype=np.uint8)
    for idx in range(1, num):
        if int(stats[idx, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == idx] = 255
    return out


def _threshold_ridge_response(response: np.ndarray, roi: np.ndarray, cfg: MinutiaeConfig) -> np.ndarray:
    enhanced = np.asarray(response, dtype=np.uint8).copy()
    enhanced[roi == 0] = 0

    roi_values = enhanced[roi > 0]
    if roi_values.size == 0 or float(np.max(roi_values)) <= 0.0:
        return np.zeros_like(enhanced, dtype=np.uint8)

    otsu_input = roi_values.astype(np.uint8).reshape(-1, 1)
    otsu_threshold, _ = cv2.threshold(
        otsu_input,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    threshold = max(float(otsu_threshold), float(cfg.min_ridge_response))
    binary = ((enhanced >= threshold) & (roi > 0)).astype(np.uint8) * 255

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return _remove_small_components(binary, cfg.min_binary_component_area)


def _binarize_ridge_polarity_candidates(
    img_u8: np.ndarray,
    roi: np.ndarray,
    cfg: MinutiaeConfig,
) -> dict[str, np.ndarray]:
    """Return deterministic ridge foreground candidates for polarity validation.

    The v1 extractor merged blackhat and tophat responses before skeletonization,
    which made ridge/valley polarity ambiguous and could produce dense skeletons
    whose crossing-number profile was dominated by spurious bifurcations.  v2
    evaluates dark-ridge, bright-ridge, and legacy mixed foregrounds separately;
    an inverse of the mixed foreground is considered only as a fallback when all
    direct polarity candidates are degenerate.
    """

    if float(np.mean(roi > 0)) <= 0.0:
        return {"empty": np.zeros_like(img_u8, dtype=np.uint8)}

    blurred = cv2.GaussianBlur(img_u8, (3, 3), 0)
    ksize = _odd_kernel_size(cfg.ridge_kernel_size, minimum=7)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dark_response = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    bright_response = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, kernel)
    mixed_response = cv2.max(dark_response, bright_response)

    candidates = {
        "dark": _threshold_ridge_response(dark_response, roi, cfg),
        "bright": _threshold_ridge_response(bright_response, roi, cfg),
        "mixed": _threshold_ridge_response(mixed_response, roi, cfg),
    }
    return candidates


def _invert_binary_inside_roi(binary: np.ndarray, roi: np.ndarray, cfg: MinutiaeConfig) -> np.ndarray:
    inverse = (((binary <= 0) & (roi > 0)).astype(np.uint8) * 255).astype(np.uint8, copy=False)
    inverse = cv2.morphologyEx(
        inverse,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return _remove_small_components(inverse, cfg.min_binary_component_area)


def zhang_suen_thinning(binary: np.ndarray, *, max_iterations: int = 80) -> np.ndarray:
    """Return a deterministic Zhang-Suen skeleton for a binary ridge image."""

    img = (np.asarray(binary) > 0).astype(np.uint8)
    if img.ndim != 2 or img.size == 0:
        return np.zeros_like(img, dtype=np.uint8)

    def _neighbors(x: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            x[:-2, 1:-1],
            x[:-2, 2:],
            x[1:-1, 2:],
            x[2:, 2:],
            x[2:, 1:-1],
            x[2:, :-2],
            x[1:-1, :-2],
            x[:-2, :-2],
        )

    for _ in range(max(1, int(max_iterations))):
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = _neighbors(img)
            center = img[1:-1, 1:-1]
            neighbor_count = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            ordered = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = np.zeros_like(center, dtype=np.uint8)
            for left, right in zip(ordered[:-1], ordered[1:]):
                transitions += ((left == 0) & (right == 1)).astype(np.uint8)

            common = (
                (center == 1)
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
            )
            if step == 0:
                marker = common & ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                marker = common & ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)

            if bool(np.any(marker)):
                interior = img[1:-1, 1:-1]
                interior[marker] = 0
                changed = True

        if not changed:
            break

    return (img.astype(np.uint8) * 255).astype(np.uint8, copy=False)


def _estimate_orientation(skeleton: np.ndarray, img_u8: np.ndarray, x: int, y: int, kind: str, cfg: MinutiaeConfig) -> float:
    radius = max(3, int(cfg.orientation_radius))
    y0 = max(0, y - radius)
    y1 = min(skeleton.shape[0], y + radius + 1)
    x0 = max(0, x - radius)
    x1 = min(skeleton.shape[1], x + radius + 1)
    patch = skeleton[y0:y1, x0:x1] > 0
    ys, xs = np.where(patch)
    if len(xs) > 1:
        xs_global = xs.astype(np.float32) + float(x0)
        ys_global = ys.astype(np.float32) + float(y0)
        keep = (xs_global != float(x)) | (ys_global != float(y))
        xs_global = xs_global[keep]
        ys_global = ys_global[keep]
        if xs_global.size:
            if kind == "ending":
                dx = float(np.mean(xs_global) - float(x))
                dy = float(np.mean(ys_global) - float(y))
                if dx * dx + dy * dy > 1e-6:
                    return _normalized_angle(math.atan2(dy, dx))

            coords = np.column_stack([xs_global - float(x), ys_global - float(y)]).astype(np.float32)
            cov = np.cov(coords, rowvar=False)
            if np.all(np.isfinite(cov)):
                eigvals, eigvecs = np.linalg.eigh(cov)
                vec = eigvecs[:, int(np.argmax(eigvals))]
                dx = float(vec[0])
                dy = float(vec[1])
                if dx < 0 or (abs(dx) <= 1e-9 and dy < 0):
                    dx = -dx
                    dy = -dy
                if dx * dx + dy * dy > 1e-6:
                    return _normalized_angle(math.atan2(dy, dx))

    gx = cv2.Sobel(img_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_u8, cv2.CV_32F, 0, 1, ksize=3)
    theta = math.atan2(float(gy[y, x]), float(gx[y, x])) + math.pi / 2.0
    return _normalized_angle(theta)


def _crossing_number(neighbors: Sequence[int]) -> int:
    total = 0
    for idx, value in enumerate(neighbors):
        total += abs(int(value) - int(neighbors[(idx + 1) % len(neighbors)]))
    return int(total // 2)


def _extract_candidates(
    skeleton: np.ndarray,
    img_u8: np.ndarray,
    roi: np.ndarray,
    cfg: MinutiaeConfig,
) -> _CandidateExtraction:
    skel = (skeleton > 0).astype(np.uint8)
    roi_binary = (roi > 0).astype(np.uint8)
    if int(np.sum(skel)) == 0 or int(np.sum(roi_binary)) == 0:
        return _CandidateExtraction(
            points=(),
            raw_candidate_endings=0,
            raw_candidate_bifurcations=0,
            rejected_dense_bifurcations=0,
            rejected_low_quality=0,
        )

    distance = cv2.distanceTransform(roi_binary, cv2.DIST_L2, 3)
    ys, xs = np.where(skel > 0)
    candidates: list[MinutiaePoint] = []
    h, w = skel.shape[:2]
    border = int(cfg.border_px)
    roi_border = float(cfg.roi_border_px)
    raw_endings = 0
    raw_bifurcations = 0
    rejected_dense_bifurcations = 0
    rejected_low_quality = 0

    for x_raw, y_raw in zip(xs, ys):
        x = int(x_raw)
        y = int(y_raw)
        if x <= border or y <= border or x >= w - border - 1 or y >= h - border - 1:
            continue
        if float(distance[y, x]) < roi_border:
            continue

        ordered = [
            skel[y - 1, x],
            skel[y - 1, x + 1],
            skel[y, x + 1],
            skel[y + 1, x + 1],
            skel[y + 1, x],
            skel[y + 1, x - 1],
            skel[y, x - 1],
            skel[y - 1, x - 1],
        ]
        cn = _crossing_number(ordered)
        neighbor_count = int(sum(int(value) for value in ordered))
        if cn == 1:
            kind = "ending"
            raw_endings += 1
            if neighbor_count > 2:
                continue
        elif cn == 3:
            kind = "bifurcation"
            raw_bifurcations += 1
            if neighbor_count < 3 or neighbor_count > 4:
                rejected_dense_bifurcations += 1
                continue
        else:
            continue

        radius = max(3, int(cfg.dense_bifurcation_radius))
        local = skel[max(0, y - radius): min(h, y + radius + 1), max(0, x - radius): min(w, x + radius + 1)]
        local_density = float(np.mean(local > 0)) if local.size else 0.0
        if kind == "bifurcation" and local_density > float(cfg.max_dense_bifurcation_density):
            rejected_dense_bifurcations += 1
            continue

        density_target = 0.13 if kind == "bifurcation" else 0.09
        density_score = 1.0 - min(abs(local_density - density_target) / max(density_target, 1e-6), 1.0)
        branch_score = 1.0
        if kind == "ending":
            branch_score = 1.0 if neighbor_count == 1 else 0.62
        elif kind == "bifurcation":
            branch_score = 1.0 if neighbor_count == 3 else 0.72

        border_quality = min(float(distance[y, x]) / max(roi_border * 3.0, 1.0), 1.0)
        quality = 0.52 * border_quality + 0.30 * density_score + 0.18 * branch_score
        if quality < float(cfg.candidate_quality_floor):
            rejected_low_quality += 1
            continue

        theta = _estimate_orientation(skeleton, img_u8, x, y, kind, cfg)
        candidates.append(
            MinutiaePoint(
                x=float(x),
                y=float(y),
                theta=theta,
                kind=kind,
                quality=float(np.clip(quality, 0.0, 1.0)),
            )
        )

    raw_total = int(raw_endings + raw_bifurcations)
    roi_pixels = int(np.sum(roi_binary > 0))
    skeleton_density = float(np.sum(skel > 0) / max(roi_pixels, 1))
    raw_pressure = max(raw_total / float(max(int(cfg.max_minutiae), 1)) - 1.0, 0.0)
    density_pressure = max(skeleton_density / 0.055 - 1.0, 0.0)
    adaptive_scale = 1.0 + min(0.95, 0.18 * raw_pressure + 0.35 * density_pressure)
    adaptive_min_distance = float(cfg.min_distance) * adaptive_scale
    if raw_total <= int(cfg.max_minutiae) and density_pressure <= 0.0:
        target_keep = int(cfg.max_minutiae)
    else:
        target_keep = int(round(float(cfg.max_minutiae) / min(adaptive_scale, 1.90)))
        target_keep = max(int(cfg.min_required_minutiae), target_keep)
        target_keep = min(int(cfg.max_minutiae), target_keep)

    candidates.sort(key=lambda p: (-p.quality, p.y, p.x, p.kind))
    kept: list[MinutiaePoint] = []
    min_d2 = adaptive_min_distance ** 2
    for candidate in candidates:
        duplicate = False
        for existing in kept:
            dx = candidate.x - existing.x
            dy = candidate.y - existing.y
            suppression_scale = (
                float(cfg.bifurcation_suppression_scale)
                if candidate.kind == "bifurcation" or existing.kind == "bifurcation"
                else 1.0
            )
            if dx * dx + dy * dy < min_d2 * suppression_scale * suppression_scale:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
        if len(kept) >= target_keep:
            break

    kept.sort(key=lambda p: (p.y, p.x, p.kind, -p.quality))
    return _CandidateExtraction(
        points=tuple(kept),
        raw_candidate_endings=int(raw_endings),
        raw_candidate_bifurcations=int(raw_bifurcations),
        rejected_dense_bifurcations=int(rejected_dense_bifurcations),
        rejected_low_quality=int(rejected_low_quality),
    )


def _preprocess_for_minutiae(img_u8: np.ndarray, cfg: MinutiaeConfig) -> np.ndarray:
    if not bool(cfg.preprocess):
        return img_u8
    prep_cfg = PreprocessConfig(
        target_size=int(cfg.target_size),
        clahe_clip=float(cfg.clahe_clip),
        clahe_grid=tuple(cfg.clahe_grid),
        blur_ksize=int(cfg.blur_ksize),
    )
    return preprocess_image(img_u8, prep_cfg)


def _quality_flags(
    *,
    skeleton_density: float,
    raw_candidate_endings: int,
    raw_candidate_bifurcations: int,
    kept_endings: int,
    kept_bifurcations: int,
    saturated_by_max_minutiae: bool,
    cfg: MinutiaeConfig,
) -> tuple[str, ...]:
    total_kept = int(kept_endings + kept_bifurcations)
    flags: list[str] = []
    if total_kept == 0:
        flags.append("no_minutiae")
    if bool(saturated_by_max_minutiae):
        flags.append("saturated")

    if total_kept >= max(int(cfg.min_required_minutiae), 8):
        bif_ratio = float(kept_bifurcations) / float(total_kept)
        ending_ratio = float(kept_endings) / float(total_kept)
        if bif_ratio >= 0.82:
            flags.append("bifurcation_dominated")
        if ending_ratio >= 0.92:
            flags.append("ending_dominated")

    if float(skeleton_density) < 0.0015:
        flags.append("sparse_skeleton")
    if float(skeleton_density) > 0.090:
        flags.append("dense_skeleton")

    raw_total = int(raw_candidate_endings + raw_candidate_bifurcations)
    if raw_total >= max(int(cfg.max_minutiae) * 4, 160) and raw_candidate_bifurcations > raw_candidate_endings * 2:
        if "bifurcation_dominated" not in flags:
            flags.append("bifurcation_dominated")

    return tuple(flags)


def _density_plausibility(skeleton_density: float) -> float:
    density = max(float(skeleton_density), 1e-9)
    if density < 0.0015:
        return max(0.05, density / 0.0015 * 0.35)
    if density > 0.090:
        return max(0.05, 1.0 - min((density - 0.090) / 0.090, 0.95))
    target = 0.020
    spread = 0.055
    return float(np.clip(1.0 - abs(density - target) / spread, 0.25, 1.0))


def _profile_plausibility(
    extraction: _CandidateExtraction,
    skeleton_density: float,
    saturated_by_max_minutiae: bool,
    flags: tuple[str, ...],
    cfg: MinutiaeConfig,
) -> float:
    total_kept = len(extraction.points)
    kept_endings, kept_bifurcations = _kind_counts(extraction.points)
    raw_total = extraction.raw_candidate_endings + extraction.raw_candidate_bifurcations

    count_score = min(float(total_kept) / float(max(int(cfg.min_required_minutiae), 1)), 1.0)
    density_score = _density_plausibility(skeleton_density)
    if total_kept <= 0:
        balance_score = 0.0
    else:
        minority = float(min(kept_endings, kept_bifurcations))
        majority = float(max(kept_endings, kept_bifurcations, 1))
        balance_score = 0.35 + 0.65 * math.sqrt(minority / majority) if minority > 0 else 0.18

    saturation_score = 0.25 if saturated_by_max_minutiae else 1.0
    raw_pressure = min(raw_total / float(max(int(cfg.max_minutiae) * 3, 1)), 1.0)
    raw_score = 1.0 - 0.35 * raw_pressure

    penalty = 1.0
    for flag in flags:
        if flag == "no_minutiae":
            penalty *= 0.05
        elif flag == "saturated":
            penalty *= 0.55
        elif flag in {"bifurcation_dominated", "ending_dominated"}:
            penalty *= 0.70
        elif flag in {"sparse_skeleton", "dense_skeleton"}:
            penalty *= 0.72

    return float(
        np.clip(
            (0.28 * density_score + 0.27 * balance_score + 0.25 * count_score + 0.12 * saturation_score + 0.08 * raw_score)
            * penalty,
            0.0,
            1.0,
        )
    )


def _profile_binary_candidate(
    polarity: str,
    binary: np.ndarray,
    gray: np.ndarray,
    roi: np.ndarray,
    cfg: MinutiaeConfig,
) -> _ExtractionProfile:
    skeleton = zhang_suen_thinning(binary, max_iterations=cfg.max_thinning_iterations)
    skeleton = _remove_small_components(skeleton, cfg.min_skeleton_component_area)
    extraction = _extract_candidates(skeleton, gray, roi, cfg)

    skeleton_pixels = int(np.sum(skeleton > 0))
    roi_pixels = int(np.sum(roi > 0))
    skeleton_density = float(skeleton_pixels / max(roi_pixels, 1))
    kept_endings, kept_bifurcations = _kind_counts(extraction.points)
    raw_total = int(extraction.raw_candidate_endings + extraction.raw_candidate_bifurcations)
    saturated = len(extraction.points) >= int(cfg.max_minutiae) and raw_total > int(cfg.max_minutiae)
    flags = _quality_flags(
        skeleton_density=skeleton_density,
        raw_candidate_endings=extraction.raw_candidate_endings,
        raw_candidate_bifurcations=extraction.raw_candidate_bifurcations,
        kept_endings=kept_endings,
        kept_bifurcations=kept_bifurcations,
        saturated_by_max_minutiae=saturated,
        cfg=cfg,
    )
    plausibility = _profile_plausibility(extraction, skeleton_density, saturated, flags, cfg)
    return _ExtractionProfile(
        polarity=str(polarity),
        skeleton=skeleton,
        extraction=extraction,
        skeleton_foreground_pixels=skeleton_pixels,
        skeleton_density=skeleton_density,
        saturated_by_max_minutiae=bool(saturated),
        quality_flags=flags,
        plausibility_score=plausibility,
    )


def _needs_inverse_polarity_fallback(profiles: Sequence[_ExtractionProfile]) -> bool:
    if not profiles:
        return True
    for profile in profiles:
        flags = set(profile.quality_flags)
        if profile.plausibility_score >= 0.35 and "no_minutiae" not in flags and "saturated" not in flags:
            return False
    return True


def _select_extraction_profile(
    gray: np.ndarray,
    roi: np.ndarray,
    cfg: MinutiaeConfig,
) -> _ExtractionProfile:
    binaries = _binarize_ridge_polarity_candidates(gray, roi, cfg)
    profiles = [
        _profile_binary_candidate(polarity, binary, gray, roi, cfg)
        for polarity, binary in binaries.items()
    ]

    if _needs_inverse_polarity_fallback(profiles):
        mixed = binaries.get("mixed")
        if mixed is not None:
            inverse = _invert_binary_inside_roi(mixed, roi, cfg)
            profiles.append(_profile_binary_candidate("mixed_inverse", inverse, gray, roi, cfg))

    profiles.sort(
        key=lambda item: (
            item.plausibility_score + {"dark": 0.03, "bright": 0.03, "mixed": 0.0, "mixed_inverse": -0.08}.get(
                item.polarity,
                -0.08,
            ),
            -int(item.saturated_by_max_minutiae),
            len(item.extraction.points),
            -len(item.quality_flags),
            {"dark": 3, "bright": 2, "mixed": 1, "mixed_inverse": 0}.get(item.polarity, -1),
        ),
        reverse=True,
    )
    if profiles:
        return profiles[0]

    empty = np.zeros_like(gray, dtype=np.uint8)
    return _profile_binary_candidate("empty", empty, gray, roi, cfg)


def extract_minutiae_template(
    img_u8: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: MinutiaeConfig | None = None,
) -> MinutiaeTemplate:
    cfg = cfg or MinutiaeConfig()
    gray = _preprocess_for_minutiae(_as_gray_u8(img_u8), cfg)
    roi, warnings = _make_roi_mask(gray, mask, cfg)
    profile = _select_extraction_profile(gray, roi, cfg)
    points = tuple(profile.extraction.points)
    kept_endings, kept_bifurcations = _kind_counts(points)

    return MinutiaeTemplate(
        points=points,
        image_shape=(int(gray.shape[0]), int(gray.shape[1])),
        roi_fraction=float(np.mean(roi > 0)) if roi.size else 0.0,
        roi_bounds=_roi_bounds(roi),
        skeleton_pixels=int(profile.skeleton_foreground_pixels),
        skeleton_foreground_pixels=int(profile.skeleton_foreground_pixels),
        skeleton_density=float(profile.skeleton_density),
        raw_candidate_endings=int(profile.extraction.raw_candidate_endings),
        raw_candidate_bifurcations=int(profile.extraction.raw_candidate_bifurcations),
        kept_endings=int(kept_endings),
        kept_bifurcations=int(kept_bifurcations),
        total_kept_minutiae=int(len(points)),
        saturated_by_max_minutiae=bool(profile.saturated_by_max_minutiae),
        ridge_polarity=str(profile.polarity),
        polarity_used=str(profile.polarity),
        extraction_quality_flags=tuple(profile.quality_flags),
        extraction_warnings=tuple(warnings),
    )


def _kind_counts(points: Sequence[MinutiaePoint]) -> tuple[int, int]:
    endings = sum(1 for point in points if point.kind == "ending")
    bifurcations = sum(1 for point in points if point.kind == "bifurcation")
    return int(endings), int(bifurcations)


def _compatible_kind(a: MinutiaePoint, b: MinutiaePoint, cfg: MinutiaeConfig) -> bool:
    return a.kind == b.kind or bool(cfg.allow_kind_mismatch)


def _transform_point(point: MinutiaePoint, angle_rad: float, dx: float, dy: float) -> tuple[float, float, float]:
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    x = cos_a * point.x - sin_a * point.y + dx
    y = sin_a * point.x + cos_a * point.y + dy
    theta = _normalized_angle(point.theta + angle_rad)
    return float(x), float(y), theta


def _match_for_transform(
    points_a: Sequence[MinutiaePoint],
    points_b: Sequence[MinutiaePoint],
    *,
    angle_rad: float,
    dx: float,
    dy: float,
    cfg: MinutiaeConfig,
) -> tuple[float, int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    spatial_tol = float(cfg.spatial_tolerance)
    angle_tol = math.radians(float(cfg.angle_tolerance_deg))
    transformed = [_transform_point(point, angle_rad, dx, dy) for point in points_a]

    pair_candidates: list[tuple[float, float, float, int, int, float, float, float]] = []
    for idx_a, (tx, ty, ttheta) in enumerate(transformed):
        point_a = points_a[idx_a]
        for idx_b, point_b in enumerate(points_b):
            if not _compatible_kind(point_a, point_b, cfg):
                continue
            dist = math.hypot(tx - point_b.x, ty - point_b.y)
            if dist > spatial_tol:
                continue
            angle_delta = _angle_delta(ttheta, point_b.theta, periodic_pi=cfg.orientation_periodic_pi)
            if angle_delta > angle_tol:
                continue
            kind_penalty = 0.0 if point_a.kind == point_b.kind else 1.0
            pair_candidates.append((kind_penalty, dist, angle_delta, idx_a, idx_b, tx, ty, ttheta))

    pair_candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[dict[str, Any]] = []
    weighted_count = 0.0
    for kind_penalty, dist, angle_delta, idx_a, idx_b, tx, ty, _ttheta in pair_candidates:
        if idx_a in used_a or idx_b in used_b:
            continue
        point_a = points_a[idx_a]
        point_b = points_b[idx_b]
        used_a.add(idx_a)
        used_b.add(idx_b)
        weighted_count += 1.0 if kind_penalty == 0.0 else float(cfg.kind_mismatch_weight)
        matched.append(
            {
                "a": (float(point_a.x), float(point_a.y)),
                "b": (float(point_b.x), float(point_b.y)),
                "a_aligned": (float(tx), float(ty)),
                "kind": "inlier",
                "kind_a": point_a.kind,
                "kind_b": point_b.kind,
                "distance": float(dist),
                "angle_delta_deg": float(math.degrees(angle_delta)),
                "quality_a": float(point_a.quality),
                "quality_b": float(point_b.quality),
            }
        )

    tentative: list[dict[str, Any]] = []
    for _kind_penalty, dist, angle_delta, idx_a, idx_b, tx, ty, _ttheta in pair_candidates[: int(cfg.max_tentative_return)]:
        point_a = points_a[idx_a]
        point_b = points_b[idx_b]
        tentative.append(
            {
                "a": (float(point_a.x), float(point_a.y)),
                "b": (float(point_b.x), float(point_b.y)),
                "a_aligned": (float(tx), float(ty)),
                "kind": "tentative",
                "kind_a": point_a.kind,
                "kind_b": point_b.kind,
                "distance": float(dist),
                "angle_delta_deg": float(math.degrees(angle_delta)),
            }
        )

    denom = float(max(min(len(points_a), len(points_b)), int(cfg.min_required_minutiae), 1))
    score = float(np.clip(weighted_count / denom, 0.0, 1.0))
    return score, len(matched), len(pair_candidates), matched, tentative


def _template_quality_multiplier(template: MinutiaeTemplate) -> float:
    flags = set(template.extraction_quality_flags)
    if "no_minutiae" in flags:
        return 0.0

    multiplier = 1.0
    if "saturated" in flags:
        multiplier *= 0.70
    if "bifurcation_dominated" in flags:
        multiplier *= 0.68
    if "ending_dominated" in flags:
        multiplier *= 0.82
    if "dense_skeleton" in flags:
        multiplier *= 0.75
    if "sparse_skeleton" in flags:
        multiplier *= 0.78

    total = max(int(template.total_kept_minutiae), len(template.points), 1)
    if total >= 16:
        minority = min(int(template.kept_endings), int(template.kept_bifurcations))
        if minority == 0:
            multiplier *= 0.58

    return float(np.clip(multiplier, 0.0, 1.0))


def _ambiguity_multiplier(matched_count: int, tentative_count: int) -> float:
    matched = max(int(matched_count), 0)
    tentative = max(int(tentative_count), 0)
    if matched <= 0:
        return 0.0
    generous_limit = max(matched * 3, matched + 8)
    if tentative <= generous_limit:
        return 1.0
    return float(np.clip(math.sqrt(generous_limit / max(float(tentative), 1.0)), 0.35, 1.0))


def _transform_plausibility_multiplier(
    template_a: MinutiaeTemplate,
    template_b: MinutiaeTemplate,
    dx: float,
    dy: float,
) -> float:
    h = max(float(template_a.image_shape[0]), float(template_b.image_shape[0]), 1.0)
    w = max(float(template_a.image_shape[1]), float(template_b.image_shape[1]), 1.0)
    diag = math.hypot(w, h)
    shift_ratio = math.hypot(float(dx), float(dy)) / max(diag, 1.0)
    if shift_ratio <= 0.35:
        return 1.0
    if shift_ratio >= 0.85:
        return 0.55
    return float(1.0 - 0.45 * ((shift_ratio - 0.35) / 0.50))


def _score_components(
    template_a: MinutiaeTemplate,
    template_b: MinutiaeTemplate,
    *,
    raw_score: float,
    matched_count: int,
    tentative_count: int,
    dx: float,
    dy: float,
) -> dict[str, float]:
    quality_a = _template_quality_multiplier(template_a)
    quality_b = _template_quality_multiplier(template_b)
    template_quality = math.sqrt(max(quality_a * quality_b, 0.0))
    ambiguity = _ambiguity_multiplier(matched_count, tentative_count)
    transform = _transform_plausibility_multiplier(template_a, template_b, dx, dy)
    final_multiplier = float(np.clip(template_quality * ambiguity * transform, 0.0, 1.0))
    return {
        "raw_alignment_score": float(raw_score),
        "template_quality_a": float(quality_a),
        "template_quality_b": float(quality_b),
        "template_quality_multiplier": float(template_quality),
        "ambiguity_multiplier": float(ambiguity),
        "transform_plausibility_multiplier": float(transform),
        "final_multiplier": float(final_multiplier),
    }


def _template_result_fields(
    template_a: MinutiaeTemplate,
    template_b: MinutiaeTemplate,
) -> dict[str, Any]:
    return {
        "skeleton_foreground_pixels_a": int(template_a.skeleton_foreground_pixels or template_a.skeleton_pixels),
        "skeleton_foreground_pixels_b": int(template_b.skeleton_foreground_pixels or template_b.skeleton_pixels),
        "skeleton_density_a": float(template_a.skeleton_density),
        "skeleton_density_b": float(template_b.skeleton_density),
        "raw_candidate_endings_a": int(template_a.raw_candidate_endings),
        "raw_candidate_endings_b": int(template_b.raw_candidate_endings),
        "raw_candidate_bifurcations_a": int(template_a.raw_candidate_bifurcations),
        "raw_candidate_bifurcations_b": int(template_b.raw_candidate_bifurcations),
        "saturated_by_max_minutiae_a": bool(template_a.saturated_by_max_minutiae),
        "saturated_by_max_minutiae_b": bool(template_b.saturated_by_max_minutiae),
        "ridge_polarity_a": str(template_a.ridge_polarity),
        "ridge_polarity_b": str(template_b.ridge_polarity),
        "extraction_quality_flags_a": tuple(template_a.extraction_quality_flags),
        "extraction_quality_flags_b": tuple(template_b.extraction_quality_flags),
    }


def _empty_match(template_a: MinutiaeTemplate, template_b: MinutiaeTemplate, cfg: MinutiaeConfig) -> MinutiaeMatchResult:
    endings_a, bifurcations_a = _kind_counts(template_a.points)
    endings_b, bifurcations_b = _kind_counts(template_b.points)
    return MinutiaeMatchResult(
        score=0.0,
        matched_count=0,
        tentative_count=0,
        minutiae_count_a=len(template_a.points),
        minutiae_count_b=len(template_b.points),
        endings_a=endings_a,
        endings_b=endings_b,
        bifurcations_a=bifurcations_a,
        bifurcations_b=bifurcations_b,
        spatial_tolerance=float(cfg.spatial_tolerance),
        angle_tolerance_deg=float(cfg.angle_tolerance_deg),
        transform_angle_deg=None,
        transform_dx=None,
        transform_dy=None,
        raw_alignment_score=0.0,
        score_multiplier=0.0,
        score_components={
            "raw_alignment_score": 0.0,
            "template_quality_a": _template_quality_multiplier(template_a),
            "template_quality_b": _template_quality_multiplier(template_b),
            "template_quality_multiplier": 0.0,
            "ambiguity_multiplier": 0.0,
            "transform_plausibility_multiplier": 0.0,
            "final_multiplier": 0.0,
        },
        **_template_result_fields(template_a, template_b),
    )


def match_minutiae_templates(
    template_a: MinutiaeTemplate,
    template_b: MinutiaeTemplate,
    cfg: MinutiaeConfig | None = None,
) -> MinutiaeMatchResult:
    cfg = cfg or MinutiaeConfig()
    points_a = tuple(template_a.points)
    points_b = tuple(template_b.points)
    if not points_a or not points_b:
        return _empty_match(template_a, template_b, cfg)

    anchors_a = sorted(points_a, key=lambda p: (-p.quality, p.y, p.x, p.kind))[: int(cfg.anchor_limit)]
    anchors_b = sorted(points_b, key=lambda p: (-p.quality, p.y, p.x, p.kind))[: int(cfg.anchor_limit)]

    best: tuple[
        float,
        float,
        int,
        int,
        float,
        float,
        float,
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, float],
    ] | None = None
    for anchor_a in anchors_a:
        for anchor_b in anchors_b:
            if not _compatible_kind(anchor_a, anchor_b, cfg):
                continue
            base_angle = _normalized_angle(anchor_b.theta - anchor_a.theta)
            candidate_angles = [base_angle]
            if cfg.orientation_periodic_pi:
                candidate_angles.append(_normalized_angle(base_angle + math.pi))

            for angle_rad in candidate_angles:
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)
                dx = anchor_b.x - (cos_a * anchor_a.x - sin_a * anchor_a.y)
                dy = anchor_b.y - (sin_a * anchor_a.x + cos_a * anchor_a.y)
                raw_score, matched_count, tentative_count, matched, tentative = _match_for_transform(
                    points_a,
                    points_b,
                    angle_rad=angle_rad,
                    dx=dx,
                    dy=dy,
                    cfg=cfg,
                )
                components = _score_components(
                    template_a,
                    template_b,
                    raw_score=raw_score,
                    matched_count=matched_count,
                    tentative_count=tentative_count,
                    dx=float(dx),
                    dy=float(dy),
                )
                final_score = float(np.clip(raw_score * components["final_multiplier"], 0.0, 1.0))
                candidate = (
                    final_score,
                    raw_score,
                    matched_count,
                    tentative_count,
                    angle_rad,
                    float(dx),
                    float(dy),
                    matched,
                    tentative,
                    components,
                )
                if best is None:
                    best = candidate
                    continue
                best_key = (best[0], best[2], best[1], best[3] * -1.0)
                candidate_key = (candidate[0], candidate[2], candidate[1], candidate[3] * -1.0)
                if candidate_key > best_key:
                    best = candidate

    if best is None:
        return _empty_match(template_a, template_b, cfg)

    score, raw_score, matched_count, tentative_count, angle_rad, dx, dy, matched, tentative, components = best
    endings_a, bifurcations_a = _kind_counts(points_a)
    endings_b, bifurcations_b = _kind_counts(points_b)
    angle_deg = ((math.degrees(angle_rad) + 180.0) % 360.0) - 180.0

    return MinutiaeMatchResult(
        score=float(np.clip(score, 0.0, 1.0)),
        matched_count=int(matched_count),
        tentative_count=int(tentative_count),
        minutiae_count_a=len(points_a),
        minutiae_count_b=len(points_b),
        endings_a=endings_a,
        endings_b=endings_b,
        bifurcations_a=bifurcations_a,
        bifurcations_b=bifurcations_b,
        spatial_tolerance=float(cfg.spatial_tolerance),
        angle_tolerance_deg=float(cfg.angle_tolerance_deg),
        transform_angle_deg=float(angle_deg),
        transform_dx=float(dx),
        transform_dy=float(dy),
        matched_minutiae=tuple(matched),
        tentative_minutiae=tuple(tentative),
        raw_alignment_score=float(raw_score),
        score_multiplier=float(components["final_multiplier"]),
        score_components=dict(components),
        **_template_result_fields(template_a, template_b),
    )


def score_pair_minutiae(
    img1_u8: np.ndarray,
    img2_u8: np.ndarray,
    mask1: np.ndarray | None = None,
    mask2: np.ndarray | None = None,
    cfg: MinutiaeConfig | None = None,
) -> MinutiaeMatchResult:
    cfg = cfg or MinutiaeConfig()
    template_a = extract_minutiae_template(img1_u8, mask=mask1, cfg=cfg)
    template_b = extract_minutiae_template(img2_u8, mask=mask2, cfg=cfg)
    return match_minutiae_templates(template_a, template_b, cfg=cfg)


def _sentinel_vector() -> np.ndarray:
    vec = np.zeros(RETRIEVAL_VECTOR_DIM, dtype=np.float32)
    vec[-1] = 1.0
    return vec


def _l2_normalized(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != RETRIEVAL_VECTOR_DIM:
        raise ValueError(f"minutiae retrieval vectors must be {RETRIEVAL_VECTOR_DIM}D, got {arr.size}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        return _sentinel_vector()
    return (arr / norm).astype(np.float32, copy=False)


def _unit_hist(values: np.ndarray, bins: int, *, value_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(bins, dtype=np.float32)
    low, high = value_range
    finite = np.clip(finite, low, high)
    hist, _ = np.histogram(finite, bins=bins, range=value_range)
    return (hist.astype(np.float32) / float(finite.size)).astype(np.float32, copy=False)


def _points_and_shape(
    template_or_points: MinutiaeTemplate | Sequence[MinutiaePoint],
    image_shape: Sequence[int] | None,
) -> tuple[tuple[MinutiaePoint, ...], tuple[float, float], float]:
    roi_fraction = 0.0
    if isinstance(template_or_points, MinutiaeTemplate):
        points = tuple(template_or_points.points)
        shape = template_or_points.image_shape
        roi_fraction = float(template_or_points.roi_fraction)
    else:
        points = tuple(template_or_points)
        shape = image_shape if image_shape is not None else (1, 1)
    if shape is None or len(shape) < 2:
        height, width = 1.0, 1.0
    else:
        height, width = max(1.0, float(shape[0])), max(1.0, float(shape[1]))
    return points, (height, width), roi_fraction


def _spatial_hist(xs: np.ndarray, ys: np.ndarray, bins_xy: tuple[int, int]) -> np.ndarray:
    if xs.size == 0 or ys.size == 0:
        return np.zeros(bins_xy[0] * bins_xy[1], dtype=np.float32)
    hist, _, _ = np.histogram2d(
        ys,
        xs,
        bins=[bins_xy[1], bins_xy[0]],
        range=[[0.0, 1.0], [0.0, 1.0]],
    )
    hist = hist.astype(np.float32).reshape(-1)
    total = float(np.sum(hist))
    return hist / total if total > 0 else hist


def minutiae_aggregate_vector(
    template_or_points: MinutiaeTemplate | Sequence[MinutiaePoint],
    image_shape: Sequence[int] | None = None,
) -> np.ndarray:
    """Return a deterministic 512D aggregate used only for 1:N shortlist retrieval."""

    points, (height, width), roi_fraction = _points_and_shape(template_or_points, image_shape)
    if not points:
        return _sentinel_vector()

    xs = np.asarray([point.x for point in points], dtype=np.float32)
    ys = np.asarray([point.y for point in points], dtype=np.float32)
    theta = np.asarray([_normalized_angle(point.theta) / TAU for point in points], dtype=np.float32)
    qualities = np.asarray([point.quality for point in points], dtype=np.float32)
    is_ending = np.asarray([point.kind == "ending" for point in points], dtype=bool)
    is_bif = np.asarray([point.kind == "bifurcation" for point in points], dtype=bool)

    x_norm = np.clip(xs / max(width - 1.0, 1.0), 0.0, 1.0)
    y_norm = np.clip(ys / max(height - 1.0, 1.0), 0.0, 1.0)
    coords = np.column_stack([x_norm, y_norm]).astype(np.float32)

    pairwise = np.zeros(0, dtype=np.float32)
    nearest = np.zeros(0, dtype=np.float32)
    local_density = np.zeros(0, dtype=np.float32)
    if len(points) >= 2:
        deltas = coords[:, None, :] - coords[None, :, :]
        distances = np.sqrt(np.sum(deltas * deltas, axis=2)).astype(np.float32) / math.sqrt(2.0)
        tri = np.triu_indices(len(points), k=1)
        pairwise = distances[tri]
        masked = distances.copy()
        np.fill_diagonal(masked, np.inf)
        nearest = np.min(masked, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        local_density = (
            np.sum((distances > 0.0) & (distances <= 0.08), axis=1).astype(np.float32)
            / float(max(len(points) - 1, 1))
        )

    centroid_x = float(np.mean(x_norm)) if x_norm.size else 0.0
    centroid_y = float(np.mean(y_norm)) if y_norm.size else 0.0
    radial = np.sqrt((x_norm - centroid_x) ** 2 + (y_norm - centroid_y) ** 2) / math.sqrt(2.0)

    ending_theta = theta[is_ending]
    bif_theta = theta[is_bif]
    ending_ratio = float(np.mean(is_ending)) if len(points) else 0.0
    bif_ratio = float(np.mean(is_bif)) if len(points) else 0.0

    bbox_area = 0.0
    if x_norm.size and y_norm.size:
        bbox_area = float((np.max(x_norm) - np.min(x_norm)) * (np.max(y_norm) - np.min(y_norm)))

    meta = np.zeros(32, dtype=np.float32)
    meta_values = np.asarray(
        [
            min(np.log1p(float(len(points))) / np.log1p(512.0), 1.0),
            ending_ratio,
            bif_ratio,
            float(np.clip(roi_fraction, 0.0, 1.0)),
            centroid_x,
            centroid_y,
            float(np.std(x_norm)) if x_norm.size else 0.0,
            float(np.std(y_norm)) if y_norm.size else 0.0,
            min(width / height, height / width),
            bbox_area,
            float(np.mean(np.cos(theta * TAU))) if theta.size else 0.0,
            float(np.mean(np.sin(theta * TAU))) if theta.size else 0.0,
            float(np.mean(qualities)) if qualities.size else 0.0,
            float(np.std(qualities)) if qualities.size else 0.0,
            float(np.mean(nearest)) if nearest.size else 0.0,
            float(np.mean(pairwise)) if pairwise.size else 0.0,
        ],
        dtype=np.float32,
    )
    meta[: meta_values.size] = meta_values

    vec = np.concatenate(
        [
            _unit_hist(x_norm, 64),
            _unit_hist(y_norm, 64),
            _unit_hist(theta, 64),
            _unit_hist(ending_theta, 32),
            _unit_hist(bif_theta, 32),
            _unit_hist(pairwise, 64),
            _unit_hist(nearest, 32),
            _unit_hist(radial, 32),
            _spatial_hist(x_norm, y_norm, (8, 8)),
            _unit_hist(local_density, 32),
            meta,
        ]
    )
    return _l2_normalized(vec.astype(np.float32, copy=False))
