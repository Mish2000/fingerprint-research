from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.fpbench.universal.quality import IMAGE_QUALITY_FEATURES, extract_image_quality


def test_extract_image_quality_returns_deterministic_lightweight_features(tmp_path: Path) -> None:
    image = np.full((64, 96), 220, dtype=np.uint8)
    cv2.line(image, (10, 10), (80, 50), 30, 3)
    cv2.circle(image, (42, 32), 12, 80, 2)
    path = tmp_path / "plain_quality.png"
    assert cv2.imwrite(str(path), image)

    first = extract_image_quality(path)
    second = extract_image_quality(path)

    assert list(first) == IMAGE_QUALITY_FEATURES
    assert first == second
    assert first["image_read_ok"] == 1.0
    assert first["width"] == 96
    assert first["height"] == 64
    assert first["aspect_ratio"] == 1.5
    assert first["std_intensity"] > 0
    assert 0 <= first["foreground_ratio"] <= 1
    assert first["sharpness_laplacian_var"] > 0
    assert 0 <= first["edge_density"] <= 1


def test_extract_image_quality_handles_missing_images() -> None:
    features = extract_image_quality("C:/definitely/missing/fingerprint.png")

    assert features["image_read_ok"] == 0.0
    assert np.isnan(features["width"])
