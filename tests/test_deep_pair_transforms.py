from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.fpbench.deep.transforms import FingerprintPairTransform, foreground_bbox, resize_with_padding


def test_resize_with_padding_outputs_square_uint8() -> None:
    img = np.full((20, 60), 255, dtype=np.uint8)
    img[5:15, 20:40] = 20
    out = resize_with_padding(img, size=64)
    assert out.shape == (64, 64)
    assert out.dtype == np.uint8


def test_pair_transform_outputs_tensor_shape() -> None:
    img = np.full((48, 96), 255, dtype=np.uint8)
    img[10:35, 20:75] = 30
    transform = FingerprintPairTransform(size=64, channels=1, foreground_crop=True)
    tensor = transform(img)
    assert tuple(tensor.shape) == (1, 64, 64)
    assert torch.isfinite(tensor).all()


def test_pair_transform_can_emit_three_channels() -> None:
    img = np.full((32, 32), 180, dtype=np.uint8)
    transform = FingerprintPairTransform(size=32, channels=3, foreground_crop=False)
    tensor = transform(img)
    assert tuple(tensor.shape) == (3, 32, 32)
