from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


IMAGE_QUALITY_FEATURES = [
    "image_read_ok",
    "width",
    "height",
    "aspect_ratio",
    "mean_intensity",
    "std_intensity",
    "contrast_proxy",
    "foreground_ratio",
    "sharpness_laplacian_var",
    "edge_density",
]


def parse_file_uri(raw: str | Path, *, repo_root: Path | None = None) -> Path:
    value = str(raw)
    if value.startswith("file:"):
        value = value[len("file:") :]
        if len(value) >= 3 and value[0] == "/" and value[2] == ":":
            value = value[1:]
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = repo_root or Path.cwd()
    return (base / path).resolve()


def _empty_quality() -> dict[str, float]:
    return {
        "image_read_ok": 0.0,
        "width": float("nan"),
        "height": float("nan"),
        "aspect_ratio": float("nan"),
        "mean_intensity": float("nan"),
        "std_intensity": float("nan"),
        "contrast_proxy": float("nan"),
        "foreground_ratio": float("nan"),
        "sharpness_laplacian_var": float("nan"),
        "edge_density": float("nan"),
    }


def _finite_or_nan(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


@lru_cache(maxsize=8192)
def _extract_image_quality_cached(path_text: str) -> tuple[tuple[str, float], ...]:
    path = Path(path_text)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return tuple(_empty_quality().items())

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    height, width = img.shape[:2]
    values = img.astype(np.float32)
    mean = float(np.mean(values))
    std = float(np.std(values))
    p05, p95 = np.quantile(values, [0.05, 0.95])

    foreground_ratio = float("nan")
    try:
        otsu_threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_fraction = float(np.mean(img < otsu_threshold))
        light_fraction = float(np.mean(img >= otsu_threshold))
        foreground_ratio = min(max(dark_fraction, 0.0), 1.0)
        if foreground_ratio > 0.95:
            foreground_ratio = min(max(light_fraction, 0.0), 1.0)
    except cv2.error:
        foreground_ratio = float(np.mean(img < mean)) if img.size else float("nan")

    try:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = float(laplacian.var())
    except cv2.error:
        sharpness = float("nan")

    try:
        edges = cv2.Canny(img, 50, 150)
        edge_density = float(np.mean(edges > 0))
    except cv2.error:
        edge_density = float("nan")

    quality = {
        "image_read_ok": 1.0,
        "width": float(width),
        "height": float(height),
        "aspect_ratio": float(width / height) if height else float("nan"),
        "mean_intensity": mean,
        "std_intensity": std,
        "contrast_proxy": float((p95 - p05) / 255.0),
        "foreground_ratio": foreground_ratio,
        "sharpness_laplacian_var": sharpness,
        "edge_density": edge_density,
    }
    return tuple((name, _finite_or_nan(quality[name])) for name in IMAGE_QUALITY_FEATURES)


def extract_image_quality(path: str | Path, *, repo_root: Path | None = None) -> dict[str, float]:
    """Return lightweight deterministic quality features for a single image."""

    resolved = parse_file_uri(path, repo_root=repo_root)
    return dict(_extract_image_quality_cached(str(resolved)))


def prefixed_quality_features(path: str | Path, prefix: str, *, repo_root: Path | None = None) -> dict[str, float]:
    quality = extract_image_quality(path, repo_root=repo_root)
    return {f"{prefix}_{name}": value for name, value in quality.items()}
