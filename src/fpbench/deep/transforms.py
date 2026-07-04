from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


def foreground_bbox(image: np.ndarray, *, margin: int = 16) -> tuple[int, int, int, int]:
    """Estimate a foreground bounding box for a grayscale fingerprint image.

    The implementation intentionally avoids a dataset-specific segmentation
    model. It compares pixels to a robust border-background estimate, then
    returns the bounding box of pixels that differ from the background.
    """
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"foreground_bbox expects grayscale 2D image; got shape={arr.shape}")
    h, w = arr.shape
    if h == 0 or w == 0:
        return 0, 0, w, h

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]).astype(np.float32)
    background = float(np.median(border))
    dynamic_range = float(np.percentile(arr, 98) - np.percentile(arr, 2))
    threshold = max(6.0, 0.08 * dynamic_range)
    mask = np.abs(arr.astype(np.float32) - background) > threshold

    if mask.sum() < max(16, int(0.002 * h * w)):
        return 0, 0, w, h

    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin + 1, w)
    y1 = min(int(ys.max()) + margin + 1, h)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, w, h
    return x0, y0, x1, y1


def crop_foreground(image: np.ndarray, *, margin: int = 16) -> np.ndarray:
    x0, y0, x1, y1 = foreground_bbox(image, margin=margin)
    return np.asarray(image)[y0:y1, x0:x1]


def resize_with_padding(image: np.ndarray, *, size: int, fill: int = 255) -> np.ndarray:
    """Resize preserving aspect ratio and pad to ``size x size``."""
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError(f"resize_with_padding expects grayscale 2D image; got shape={arr.shape}")
    h, w = arr.shape
    if h <= 0 or w <= 0:
        raise ValueError("Cannot resize an empty image")
    scale = min(float(size) / float(w), float(size) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    pil = Image.fromarray(arr, mode="L").resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (int(size), int(size)), color=int(fill))
    left = (int(size) - new_w) // 2
    top = (int(size) - new_h) // 2
    canvas.paste(pil, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


@dataclass(frozen=True)
class FingerprintPairTransform:
    """Deterministic preprocessing used for Phase 2B inference/evaluation.

    For the first prototype we keep augmentation out of the benchmark path.
    Training-time augmentation can be added later without changing the score
    schema.
    """

    size: int = 384
    channels: int = 1
    foreground_crop: bool = True
    foreground_margin: int = 16
    invert: bool = False

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 2:
            raise ValueError(f"Expected grayscale image, got shape={arr.shape}")
        if self.foreground_crop:
            arr = crop_foreground(arr, margin=int(self.foreground_margin))
        arr = resize_with_padding(arr, size=int(self.size), fill=255)
        tensor = torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)
        if self.invert:
            tensor = 1.0 - tensor
        tensor = (tensor - 0.5) / 0.5
        if int(self.channels) == 3:
            tensor = tensor.repeat(3, 1, 1)
        elif int(self.channels) != 1:
            raise ValueError(f"channels must be 1 or 3; got {self.channels!r}")
        return tensor.contiguous()
