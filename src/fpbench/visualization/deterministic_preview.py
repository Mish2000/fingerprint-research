from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from scipy import io as scipy_io

try:
    import h5py
except ImportError:
    h5py = None


RASTER_SOURCE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
ARRAY_SOURCE_SUFFIXES = frozenset({".mat", ".npy", ".npz"})
DETERMINISTIC_SOURCE_SUFFIXES = frozenset((*RASTER_SOURCE_SUFFIXES, *ARRAY_SOURCE_SUFFIXES))
DEFAULT_NUMERIC_FILL_VALUE = 245
SURFACE_CLAHE_CLIP_LIMIT = 2.0
SURFACE_CLAHE_TILE_GRID = (8, 8)
SURFACE_FOREGROUND_BORDER_RATIO = 0.012
SURFACE_FOREGROUND_MIN_AREA_RATIO = 0.015
SURFACE_FINAL_VERTICAL_MARGIN_RATIO = 0.12
SURFACE_MIN_CROP_HEIGHT_RATIO = 0.30
SURFACE_THIN_MASK_HEIGHT_RATIO = 0.18
SURFACE_THIN_MASK_VERTICAL_EXPANSION_RATIO = 0.08
SURFACE_MAX_CROP_ASPECT_RATIO = 2.5


def can_render_deterministic_source(path: Optional[Path]) -> bool:
    return path is not None and path.suffix.lower() in DETERMINISTIC_SOURCE_SUFFIXES


def deterministic_render_strategy(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in ARRAY_SOURCE_SUFFIXES:
        return "deterministic_numeric_artifact"
    if suffix in RASTER_SOURCE_SUFFIXES:
        return "deterministic_raster_source"
    raise ValueError(f"unsupported deterministic preview source suffix: {suffix}")


def normalize_numeric_array_to_u8(
    array: np.ndarray,
    *,
    fill_value: int = DEFAULT_NUMERIC_FILL_VALUE,
) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr.copy()

    arr_f32 = np.asarray(arr, dtype=np.float32)
    finite_mask = np.isfinite(arr_f32)
    if not finite_mask.any():
        raise ValueError("numeric artifact contains no finite values")

    valid = arr_f32[finite_mask]
    lo = float(valid.min())
    hi = float(valid.max())
    if valid.size >= 32:
        lo = float(np.percentile(valid, 1.0))
        hi = float(np.percentile(valid, 99.0))
    if hi <= lo:
        hi = lo + 1.0

    out = np.full(arr_f32.shape, float(fill_value), dtype=np.float32)
    clipped = np.clip(arr_f32[finite_mask], lo, hi)
    out[finite_mask] = (clipped - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _normalize_masked_numeric_array_to_u8(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    fill_value: int = DEFAULT_NUMERIC_FILL_VALUE,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    arr_f32 = np.asarray(array, dtype=np.float32)
    valid_mask = np.asarray(mask, dtype=bool) & np.isfinite(arr_f32)
    if not valid_mask.any():
        raise ValueError("numeric artifact contains no finite values inside the render mask")

    valid = arr_f32[valid_mask]
    lo = float(np.percentile(valid, lower_percentile))
    hi = float(np.percentile(valid, upper_percentile))
    if hi <= lo:
        hi = lo + 1.0

    out = np.full(arr_f32.shape, float(fill_value), dtype=np.float32)
    clipped = np.clip(arr_f32[valid_mask], lo, hi)
    out[valid_mask] = (clipped - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _crop_to_mask_bbox(
    array: np.ndarray,
    mask: np.ndarray,
    *,
    margin_ratio: float = 0.03,
    vertical_margin_ratio: Optional[float] = None,
    min_margin: int = 2,
    max_aspect_ratio: Optional[float] = None,
    min_height_ratio: Optional[float] = None,
    reference_height: Optional[int] = None,
    thin_mask_height_ratio: Optional[float] = None,
    thin_vertical_expansion_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    valid_y, valid_x = np.where(mask)
    if valid_y.size == 0 or valid_x.size == 0:
        raise ValueError("render mask is empty")

    height, width = mask.shape
    margin_y_ratio = vertical_margin_ratio if vertical_margin_ratio is not None else margin_ratio
    margin_y = max(min_margin, int(round(height * margin_y_ratio)))
    margin_x = max(min_margin, int(round(width * margin_ratio)))
    top = max(0, int(valid_y.min()) - margin_y)
    bottom = min(height, int(valid_y.max()) + margin_y + 1)
    left = max(0, int(valid_x.min()) - margin_x)
    right = min(width, int(valid_x.max()) + margin_x + 1)

    crop_reference_height = reference_height if reference_height is not None else height
    bbox_height = int(valid_y.max()) - int(valid_y.min()) + 1
    if thin_mask_height_ratio is not None and bbox_height < int(np.ceil(crop_reference_height * thin_mask_height_ratio)):
        extra_height = int(np.ceil(crop_reference_height * thin_vertical_expansion_ratio))
        top, bottom = _expand_interval(
            top,
            bottom,
            target=min(height, max(bottom - top + extra_height, bbox_height + extra_height)),
            limit=height,
        )

    if min_height_ratio is not None:
        min_height = int(np.ceil(crop_reference_height * min_height_ratio))
        top, bottom = _expand_interval(top, bottom, target=min(height, min_height), limit=height)

    if max_aspect_ratio is not None:
        crop_height = max(1, bottom - top)
        crop_width = max(1, right - left)
        if crop_width / crop_height > max_aspect_ratio:
            target_height = min(height, int(np.ceil(crop_width / max_aspect_ratio)))
            top, bottom = _expand_interval(top, bottom, target=target_height, limit=height)
        elif crop_height / crop_width > max_aspect_ratio:
            target_width = min(width, int(np.ceil(crop_height / max_aspect_ratio)))
            left, right = _expand_interval(left, right, target=target_width, limit=width)

    return array[top:bottom, left:right], mask[top:bottom, left:right]


def _clip_border_from_mask(
    mask: np.ndarray,
    *,
    border_ratio: float = SURFACE_FOREGROUND_BORDER_RATIO,
    min_border: int = 3,
) -> np.ndarray:
    height, width = mask.shape
    border = max(min_border, int(round(min(height, width) * border_ratio)))
    if border * 2 >= min(height, width):
        return np.asarray(mask, dtype=bool).copy()

    trimmed = np.asarray(mask, dtype=bool).copy()
    trimmed[:border, :] = False
    trimmed[-border:, :] = False
    trimmed[:, :border] = False
    trimmed[:, -border:] = False
    return trimmed


def _expand_interval(start: int, end: int, *, target: int, limit: int) -> tuple[int, int]:
    span = max(1, end - start)
    target = min(limit, max(span, target))
    if span >= target:
        return start, end

    extra = target - span
    start = max(0, start - extra // 2)
    end = min(limit, end + extra - extra // 2)
    if end - start < target:
        if start == 0:
            end = min(limit, target)
        elif end == limit:
            start = max(0, limit - target)
    return start, end


def _fill_surface_holes(surface: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(surface, dtype=np.float32).copy()
    if finite_mask.all():
        return arr

    median = float(np.median(arr[finite_mask]))
    arr[~finite_mask] = median
    return arr


def _fit_surface_plane(surface: np.ndarray, finite_mask: np.ndarray) -> np.ndarray:
    height, width = surface.shape
    coords_y, coords_x = np.where(finite_mask)
    values = surface[coords_y, coords_x]

    if values.size == 0:
        raise ValueError("surface contains no finite values")

    if values.size > 200_000:
        step = int(np.ceil(values.size / 200_000))
        coords_y = coords_y[::step]
        coords_x = coords_x[::step]
        values = values[::step]

    centered_x = coords_x.astype(np.float32) - float(width - 1) / 2.0
    centered_y = coords_y.astype(np.float32) - float(height - 1) / 2.0
    design = np.column_stack([centered_x, centered_y, np.ones_like(centered_x, dtype=np.float32)])
    coeffs, _, _, _ = np.linalg.lstsq(design, values.astype(np.float32), rcond=None)

    grid_x = np.arange(width, dtype=np.float32)[None, :] - float(width - 1) / 2.0
    grid_y = np.arange(height, dtype=np.float32)[:, None] - float(height - 1) / 2.0
    return coeffs[0] * grid_x + coeffs[1] * grid_y + coeffs[2]


def _masked_gaussian_blur(image: np.ndarray, mask: np.ndarray, *, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return np.asarray(image, dtype=np.float32).copy()

    image_f32 = np.asarray(image, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32)
    blurred_image = cv2.GaussianBlur(image_f32 * weights, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    blurred_weights = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    return blurred_image / np.maximum(blurred_weights, 1e-6)


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if component_count <= 1:
        return mask_u8.astype(bool)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest_label


def _surface_clahe_tile_grid(shape: tuple[int, int]) -> tuple[int, int]:
    height, width = shape
    tile_x = int(np.clip(round(width / 192.0), SURFACE_CLAHE_TILE_GRID[0], 14))
    tile_y = int(np.clip(round(height / 192.0), SURFACE_CLAHE_TILE_GRID[1], 14))
    return (tile_x, tile_y)


def _refine_surface_foreground_mask(
    surface: np.ndarray,
    highpass_surface: np.ndarray,
    finite_mask: np.ndarray,
) -> np.ndarray:
    height, width = finite_mask.shape
    if min(height, width) < 64:
        return finite_mask

    surface_f32 = np.asarray(surface, dtype=np.float32)
    detail = np.abs(np.asarray(highpass_surface, dtype=np.float32))
    detail = _masked_gaussian_blur(detail, finite_mask, sigma=max(1.0, min(height, width) / 320.0))

    valid_surface = surface_f32[finite_mask]
    surface_lo = float(np.percentile(valid_surface, 15.0))
    surface_hi = float(np.percentile(valid_surface, 85.0))
    surface_threshold = surface_lo + max(1.0, (surface_hi - surface_lo) * 0.12)
    support_seed = finite_mask & (surface_f32 >= surface_threshold)
    support_seed = cv2.morphologyEx(
        support_seed.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    ) > 0
    support_map = _masked_gaussian_blur(
        support_seed.astype(np.float32),
        finite_mask,
        sigma=max(2.0, min(height, width) / 48.0),
    )

    detail_threshold = float(np.percentile(detail[finite_mask], 68.0))
    support_threshold = max(0.2, float(np.percentile(support_map[finite_mask], 55.0)))
    signal_mask = finite_mask & (detail >= detail_threshold) & (support_map >= support_threshold)
    trimmed_signal_mask = _clip_border_from_mask(signal_mask)
    if trimmed_signal_mask.any():
        signal_mask = trimmed_signal_mask
    if not signal_mask.any():
        signal_mask = finite_mask & (detail >= float(np.percentile(detail[finite_mask], 75.0)))
        trimmed_signal_mask = _clip_border_from_mask(signal_mask)
        if trimmed_signal_mask.any():
            signal_mask = trimmed_signal_mask
    if not signal_mask.any():
        return finite_mask

    close_radius = max(3, int(round(min(height, width) / 90.0)))
    open_radius = max(1, close_radius // 2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_radius + 1, 2 * open_radius + 1))
    signal_mask = cv2.morphologyEx(signal_mask.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
    signal_mask = cv2.morphologyEx(signal_mask.astype(np.uint8), cv2.MORPH_OPEN, open_kernel) > 0
    signal_mask = _largest_connected_component(signal_mask)

    if signal_mask.sum() < max(512, int(finite_mask.sum() * SURFACE_FOREGROUND_MIN_AREA_RATIO)):
        return finite_mask

    return signal_mask


def _render_surface_ridge_map(array: np.ndarray) -> np.ndarray:
    surface = np.asarray(array, dtype=np.float32)
    finite_mask = np.isfinite(surface)
    if not finite_mask.any():
        raise ValueError("numeric artifact contains no finite values")

    cropped_surface, cropped_mask = _crop_to_mask_bbox(surface, finite_mask, margin_ratio=0.015, min_margin=2)
    filled_surface = _fill_surface_holes(cropped_surface, cropped_mask)
    detrended_surface = filled_surface - _fit_surface_plane(filled_surface, cropped_mask)
    low_frequency_background = _masked_gaussian_blur(
        detrended_surface,
        cropped_mask,
        sigma=max(2.0, min(cropped_mask.shape) / 30.0),
    )
    highpass_surface = detrended_surface - low_frequency_background

    refined_mask = _refine_surface_foreground_mask(filled_surface, highpass_surface, cropped_mask)
    if not np.array_equal(refined_mask, cropped_mask):
        highpass_surface, refined_mask = _crop_to_mask_bbox(
            highpass_surface,
            refined_mask,
            margin_ratio=0.06,
            vertical_margin_ratio=SURFACE_FINAL_VERTICAL_MARGIN_RATIO,
            min_margin=4,
            max_aspect_ratio=SURFACE_MAX_CROP_ASPECT_RATIO,
            min_height_ratio=SURFACE_MIN_CROP_HEIGHT_RATIO,
            reference_height=surface.shape[0],
            thin_mask_height_ratio=SURFACE_THIN_MASK_HEIGHT_RATIO,
            thin_vertical_expansion_ratio=SURFACE_THIN_MASK_VERTICAL_EXPANSION_RATIO,
        )
        cropped_mask = refined_mask

    highpass_surface = _masked_gaussian_blur(
        highpass_surface,
        cropped_mask,
        sigma=max(0.6, min(cropped_mask.shape) / 1024.0),
    )
    grad_x = cv2.Sobel(highpass_surface, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(highpass_surface, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    ridge_strength = np.sqrt(np.maximum(gradient_magnitude, 0.0))

    edge_mask = cv2.erode(cropped_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1) > 0
    if not edge_mask.any():
        edge_mask = cropped_mask

    normalized = _normalize_masked_numeric_array_to_u8(
        ridge_strength,
        edge_mask,
        fill_value=0,
        lower_percentile=10.0,
        upper_percentile=99.7,
    )
    clahe = cv2.createCLAHE(
        clipLimit=SURFACE_CLAHE_CLIP_LIMIT,
        tileGridSize=_surface_clahe_tile_grid(normalized.shape),
    )
    enhanced = clahe.apply(normalized)
    ridge_map = np.full(enhanced.shape, DEFAULT_NUMERIC_FILL_VALUE, dtype=np.uint8)
    ridge_map[cropped_mask] = 255 - enhanced[cropped_mask]
    return ridge_map


def render_numeric_array(array: np.ndarray) -> np.ndarray:
    arr = np.squeeze(np.asarray(array))
    if arr.ndim == 0:
        raise ValueError("scalar arrays are not renderable as preview assets")

    if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 2:
        return _render_surface_ridge_map(arr)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rendered = normalize_numeric_array_to_u8(arr)
        if rendered.shape[-1] == 4:
            return cv2.cvtColor(rendered, cv2.COLOR_BGRA2BGR)
        return rendered

    raise ValueError(f"unsupported numeric artifact shape: {arr.shape}")


def _collect_numeric_arrays(value: Any, *, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []

    if h5py is not None and isinstance(value, h5py.Group):
        for key in value.keys():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            candidates.extend(_collect_numeric_arrays(value[key], prefix=child_prefix))
        return candidates

    if h5py is not None and isinstance(value, h5py.Dataset):
        return _collect_numeric_arrays(value[()], prefix=prefix)

    if isinstance(value, np.ndarray):
        if value.dtype.names:
            for field_name in value.dtype.names:
                child_prefix = f"{prefix}.{field_name}" if prefix else str(field_name)
                candidates.extend(_collect_numeric_arrays(value[field_name], prefix=child_prefix))
            return candidates

        if value.dtype == object:
            for index, item in np.ndenumerate(value):
                child_prefix = f"{prefix}[{','.join(str(part) for part in index)}]" if prefix else "object_item"
                candidates.extend(_collect_numeric_arrays(item, prefix=child_prefix))
            return candidates

        squeezed = np.squeeze(value)
        if squeezed.size > 0 and np.issubdtype(squeezed.dtype, np.number):
            candidates.append((prefix or "array", squeezed))
        return candidates

    if hasattr(value, "__dict__") and not isinstance(value, (str, bytes)):
        for key, child in vars(value).items():
            if key.startswith("_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            candidates.extend(_collect_numeric_arrays(child, prefix=child_prefix))

    return candidates


def _numeric_array_score(name: str, array: np.ndarray) -> tuple[int, int, int]:
    lowered = name.lower()
    if any(token in lowered for token in ("surface", "depth", "height", "z", "fc")):
        name_rank = 0
    elif any(token in lowered for token in ("image", "img", "data")):
        name_rank = 1
    else:
        name_rank = 2

    shape_rank = 0 if array.ndim == 2 else 1 if array.ndim == 3 else 2
    return (name_rank, shape_rank, -int(array.size))


def _best_numeric_array(candidates: list[tuple[str, np.ndarray]]) -> np.ndarray:
    if not candidates:
        raise ValueError("no numeric arrays were found in artifact")
    _, array = min(candidates, key=lambda item: _numeric_array_score(item[0], item[1]))
    return array


def load_numeric_artifact(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            candidates = [(str(key), archive[key]) for key in archive.files]
        return _best_numeric_array(candidates)
    if suffix == ".mat":
        try:
            payload = scipy_io.loadmat(path, squeeze_me=True, struct_as_record=False)
            numeric_candidates: list[tuple[str, np.ndarray]] = []
            for key, value in payload.items():
                if str(key).startswith("__"):
                    continue
                numeric_candidates.extend(_collect_numeric_arrays(value, prefix=key))
            return _best_numeric_array(numeric_candidates)
        except (NotImplementedError, ValueError):
            if h5py is None:
                raise ValueError("h5py is required to load HDF5-backed .mat artifacts") from None
            with h5py.File(path, "r") as handle:
                return _best_numeric_array(_collect_numeric_arrays(handle))
    raise ValueError(f"unsupported numeric artifact suffix: {suffix}")


def read_deterministic_preview_source(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in RASTER_SOURCE_SUFFIXES:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"failed to read raster source: {path}")
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim in {2, 3}:
            return image
        raise ValueError(f"unsupported raster shape: {image.shape}")

    if suffix in ARRAY_SOURCE_SUFFIXES:
        return render_numeric_array(load_numeric_artifact(path))

    raise ValueError(f"unsupported deterministic preview source suffix: {suffix}")
