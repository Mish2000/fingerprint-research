from __future__ import annotations

from typing import Any, Sequence

import numpy as np

RETRIEVAL_VECTOR_DIM = 512
SIFT_DESCRIPTOR_DIM = 128
ORB_DESCRIPTOR_BYTES = 32


def _sentinel_vector() -> np.ndarray:
    vec = np.zeros(RETRIEVAL_VECTOR_DIM, dtype=np.float32)
    vec[-1] = 1.0
    return vec


def _l2_normalized(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size != RETRIEVAL_VECTOR_DIM:
        raise ValueError(f"classic retrieval vectors must be {RETRIEVAL_VECTOR_DIM}D, got {arr.size}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        return _sentinel_vector()
    return (arr / norm).astype(np.float32, copy=False)


def _coerce_descriptor_matrix(descriptors: Any, *, width: int) -> np.ndarray | None:
    if descriptors is None:
        return None
    arr = np.asarray(descriptors)
    if arr.size == 0:
        return None
    if arr.ndim == 1:
        if arr.size % width == 0:
            arr = arr.reshape(-1, width)
        else:
            padded = np.zeros((1, width), dtype=arr.dtype)
            take = min(width, arr.size)
            padded[0, :take] = arr[:take]
            arr = padded
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[0] == 0:
        return None
    if arr.shape[1] != width:
        fixed = np.zeros((arr.shape[0], width), dtype=arr.dtype)
        take = min(width, arr.shape[1])
        fixed[:, :take] = arr[:, :take]
        arr = fixed
    return arr


def _image_height_width(image_shape: Sequence[int] | None) -> tuple[float, float]:
    if image_shape is None or len(image_shape) < 2:
        return 1.0, 1.0
    height = max(1.0, float(image_shape[0]))
    width = max(1.0, float(image_shape[1]))
    return height, width


def _keypoint_array(keypoints: Sequence[Any] | None, attr: str, default: float = 0.0) -> np.ndarray:
    if not keypoints:
        return np.zeros(0, dtype=np.float32)
    values = [float(getattr(kp, attr, default)) for kp in keypoints]
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=default, posinf=default, neginf=default)


def _keypoint_points(keypoints: Sequence[Any] | None) -> tuple[np.ndarray, np.ndarray]:
    if not keypoints:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    xs: list[float] = []
    ys: list[float] = []
    for kp in keypoints:
        pt = getattr(kp, "pt", (0.0, 0.0))
        xs.append(float(pt[0]))
        ys.append(float(pt[1]))
    x_arr = np.nan_to_num(np.asarray(xs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_arr = np.nan_to_num(np.asarray(ys, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return x_arr, y_arr


def _unit_hist(values: np.ndarray, bins: int, *, value_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.zeros(bins, dtype=np.float32)
    low, high = value_range
    finite = np.clip(finite, low, high)
    hist, _ = np.histogram(finite, bins=bins, range=value_range)
    return (hist.astype(np.float32) / float(finite.size)).astype(np.float32, copy=False)


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return finite
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    if hi <= lo:
        if hi == 0.0:
            return np.zeros_like(finite, dtype=np.float32)
        return np.full_like(finite, 0.5, dtype=np.float32)
    return np.clip((finite - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def sift_aggregated_descriptor_vector(descriptors: Any) -> np.ndarray:
    """Return a 512D mean/std/p10/p90 SIFT descriptor aggregate for shortlist retrieval."""
    desc = _coerce_descriptor_matrix(descriptors, width=SIFT_DESCRIPTOR_DIM)
    if desc is None:
        return _sentinel_vector()

    arr = np.asarray(desc, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape[0] == 0:
        return _sentinel_vector()

    stats = (
        np.mean(arr, axis=0),
        np.std(arr, axis=0),
        np.percentile(arr, 10, axis=0),
        np.percentile(arr, 90, axis=0),
    )
    return _l2_normalized(np.concatenate(stats).astype(np.float32, copy=False))


def orb_aggregated_descriptor_vector(
    keypoints: Sequence[Any] | None,
    descriptors: Any,
    image_shape: Sequence[int] | None,
) -> np.ndarray:
    """Return a deterministic 512D ORB-family aggregate for shortlist retrieval.

    Layout:
      0:256   bit occupancy from unpacked 32-byte ORB descriptors
      256:320 x-coordinate histogram over image width
      320:384 y-coordinate histogram over image height
      384:448 keypoint angle histogram
      448:480 keypoint response histogram
      480:496 keypoint size histogram
      496:504 keypoint octave histogram
      504:512 count/image/centroid metadata, with final slot reserved as sentinel flag
    """
    desc = _coerce_descriptor_matrix(descriptors, width=ORB_DESCRIPTOR_BYTES)
    if desc is None or not keypoints:
        return _sentinel_vector()

    desc_u8 = np.nan_to_num(np.asarray(desc, dtype=np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    desc_u8 = np.clip(desc_u8, 0.0, 255.0).astype(np.uint8, copy=False)
    if desc_u8.shape[0] == 0:
        return _sentinel_vector()

    bits = np.unpackbits(desc_u8, axis=1).astype(np.float32)
    bit_occupancy = np.mean(bits, axis=0)

    height, width = _image_height_width(image_shape)
    xs, ys = _keypoint_points(keypoints)
    x_norm = np.clip(xs / max(width - 1.0, 1.0), 0.0, 1.0)
    y_norm = np.clip(ys / max(height - 1.0, 1.0), 0.0, 1.0)

    angles = _keypoint_array(keypoints, "angle", default=-1.0)
    valid_angles = angles[np.isfinite(angles) & (angles >= 0.0)]
    angle_norm = np.mod(valid_angles, 360.0) / 360.0 if valid_angles.size else valid_angles

    responses = _robust_unit_interval(_keypoint_array(keypoints, "response", default=0.0))
    sizes = _keypoint_array(keypoints, "size", default=0.0)
    size_norm = np.clip(sizes / max(height, width, 1.0), 0.0, 1.0)
    octaves = np.clip(_keypoint_array(keypoints, "octave", default=0.0), 0.0, 7.0)

    keypoint_count = float(len(keypoints))
    descriptor_count = float(desc_u8.shape[0])
    meta = np.array(
        [
            min(np.log1p(keypoint_count) / np.log1p(4096.0), 1.0),
            min(np.log1p(descriptor_count) / np.log1p(4096.0), 1.0),
            min(descriptor_count / max(keypoint_count, 1.0), 1.0),
            min(np.log1p(height * width) / np.log1p(2048.0 * 2048.0), 1.0),
            min(width / height, height / width),
            float(np.mean(x_norm)) if x_norm.size else 0.0,
            float(np.mean(y_norm)) if y_norm.size else 0.0,
            0.0,
        ],
        dtype=np.float32,
    )

    vec = np.concatenate(
        [
            bit_occupancy,
            _unit_hist(x_norm, 64),
            _unit_hist(y_norm, 64),
            _unit_hist(angle_norm, 64),
            _unit_hist(responses, 32),
            _unit_hist(size_norm, 16),
            _unit_hist(octaves, 8, value_range=(0.0, 8.0)),
            meta,
        ]
    )
    return _l2_normalized(vec)
