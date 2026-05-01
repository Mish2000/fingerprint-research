from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fpbench.visualization.deterministic_preview import (
    SURFACE_FINAL_VERTICAL_MARGIN_RATIO,
    SURFACE_MAX_CROP_ASPECT_RATIO,
    SURFACE_MIN_CROP_HEIGHT_RATIO,
    SURFACE_THIN_MASK_HEIGHT_RATIO,
    SURFACE_THIN_MASK_VERTICAL_EXPANSION_RATIO,
    _crop_to_mask_bbox,
    normalize_numeric_array_to_u8,
    render_numeric_array,
)


def _make_surface(shape: tuple[int, int] = (240, 320), *, nan_border: bool = False) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[:height, :width].astype(np.float32)
    nx = (x - (width - 1) / 2.0) / (width / 2.0)
    ny = (y - (height - 1) / 2.0) / (height / 2.0)

    envelope = np.exp(-((nx / 0.72) ** 2 + (ny / 0.58) ** 2) * 2.2)
    plane = 180.0 + 95.0 * nx - 75.0 * ny + 26.0 * (nx**2 - 0.35 * ny**2)
    phase = 2.0 * np.pi * (11.5 * nx + 2.8 * ny + 0.8 * nx * ny)
    ridges = envelope * (9.0 * np.sin(phase) + 3.0 * np.sin(2.0 * phase + 0.6))
    bowl = 18.0 * np.exp(-((nx / 0.95) ** 2 + (ny / 0.8) ** 2))
    surface = (plane + bowl + ridges).astype(np.float32)

    if nan_border:
        finite_foreground = (np.abs(nx) <= 0.84) & (np.abs(ny) <= 0.78)
        surface = surface.copy()
        surface[~finite_foreground] = np.nan

    return surface


def _resize_like(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _edge_energy(image: np.ndarray) -> float:
    image_f32 = np.asarray(image, dtype=np.float32)
    grad_x = cv2.Sobel(image_f32, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_f32, cv2.CV_32F, 0, 1, ksize=3)
    return float(cv2.magnitude(grad_x, grad_y).mean())


def _detail_energy(image: np.ndarray) -> float:
    image_f32 = np.asarray(image, dtype=np.float32)
    return float(cv2.Laplacian(image_f32, cv2.CV_32F, ksize=3).var())


def _background_drift(image: np.ndarray) -> float:
    image_f32 = np.asarray(image, dtype=np.float32)
    band = max(4, min(image_f32.shape) // 24)
    horizontal_drift = abs(float(image_f32[:, :band].mean()) - float(image_f32[:, -band:].mean()))
    vertical_drift = abs(float(image_f32[:band, :].mean()) - float(image_f32[-band:, :].mean()))
    return horizontal_drift + vertical_drift


def test_render_numeric_array_crops_to_finite_foreground_and_preserves_ridge_detail() -> None:
    surface = _make_surface(nan_border=True)
    baseline = normalize_numeric_array_to_u8(surface)
    rendered = render_numeric_array(surface)
    baseline_resized = _resize_like(baseline, rendered.shape)

    assert rendered.dtype == np.uint8
    assert rendered.ndim == 2
    assert rendered.shape[0] < int(surface.shape[0] * 0.8)
    assert rendered.shape[1] < int(surface.shape[1] * 0.8)
    assert _edge_energy(rendered) > _edge_energy(baseline_resized) * 3.0
    assert _detail_energy(rendered) > _detail_energy(baseline_resized) * 4.0


def test_render_numeric_array_boosts_edges_and_suppresses_low_frequency_drift() -> None:
    surface = _make_surface()
    baseline = normalize_numeric_array_to_u8(surface)
    rendered = render_numeric_array(surface)
    rendered_resized = _resize_like(rendered, baseline.shape)

    assert _edge_energy(rendered_resized) > _edge_energy(baseline) * 2.0
    assert _detail_energy(rendered_resized) > _detail_energy(baseline) * 5.0
    assert _background_drift(rendered_resized) < _background_drift(baseline) * 0.25


def test_crop_to_mask_bbox_expands_thin_surface_masks_vertically() -> None:
    height, width = 320, 420
    mask = np.zeros((height, width), dtype=bool)
    mask[146:162, 30:390] = True
    array = np.zeros((height, width), dtype=np.float32)

    _, cropped_mask = _crop_to_mask_bbox(
        array,
        mask,
        margin_ratio=0.06,
        vertical_margin_ratio=SURFACE_FINAL_VERTICAL_MARGIN_RATIO,
        min_margin=4,
        max_aspect_ratio=SURFACE_MAX_CROP_ASPECT_RATIO,
        min_height_ratio=SURFACE_MIN_CROP_HEIGHT_RATIO,
        reference_height=height,
        thin_mask_height_ratio=SURFACE_THIN_MASK_HEIGHT_RATIO,
        thin_vertical_expansion_ratio=SURFACE_THIN_MASK_VERTICAL_EXPANSION_RATIO,
    )

    assert cropped_mask.shape[0] >= int(np.ceil(height * SURFACE_MIN_CROP_HEIGHT_RATIO))
    assert cropped_mask.shape[1] / cropped_mask.shape[0] <= SURFACE_MAX_CROP_ASPECT_RATIO + 1e-6
