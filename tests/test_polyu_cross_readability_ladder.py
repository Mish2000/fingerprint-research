"""Tests for the SourceAFIS readability-ladder transforms (Phase 4A.2A).

Hermetic: verifies deterministic pure transforms, contact-side immutability,
and metric computation. No sidecar, torch, or biometric images required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from scripts.diagnostics import run_polyu_cross_sourceafis_readability_ladder as lad


def _img(seed: int, h: int = 64, w: int = 80) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(30, 210, w, dtype=np.float64)
    return (np.tile(base, (h, 1)) + rng.normal(0, 6, (h, w))).clip(0, 255).astype(np.uint8)


@pytest.mark.parametrize("variant", lad.VARIANT_ORDER)
def test_transforms_deterministic_and_uint8(variant):
    img = _img(1)
    fn = lad.VARIANTS[variant]
    a = fn(img.copy())
    b = fn(img.copy())
    assert a.dtype == np.uint8
    assert np.array_equal(a, b)  # deterministic
    # raw input not mutated
    assert np.array_equal(img, _img(1))


def test_p1_invert_correct():
    img = _img(2)
    assert np.array_equal(lad.p1_invert(img), 255 - img)


def test_p2_robust_norm_and_degenerate():
    img = _img(3)
    out = lad.p2_robust_intensity_norm(img)
    assert out.min() >= 0 and out.max() <= 255
    # Degenerate constant image -> identity passthrough (no divide-by-zero).
    const = np.full((32, 32), 128, np.uint8)
    assert np.array_equal(lad.p2_robust_intensity_norm(const), const)


def test_p4_canvas_shape():
    out = lad.p4_roi_crop_pad(_img(4))
    assert out.shape == (lad.P4_CANVAS, lad.P4_CANVAS)


def test_build_transformed_leaves_contact_untouched(tmp_path):
    root = tmp_path / "root"
    (root / "cl").mkdir(parents=True)
    (root / "ct").mkdir(parents=True)
    cl_path = root / "cl" / "probe.bmp"
    ct_path = root / "ct" / "gallery.jpg"
    cv2.imwrite(str(cl_path), _img(10))
    cv2.imwrite(str(ct_path), _img(11))
    ct_before = hashlib.sha256(ct_path.read_bytes()).hexdigest()

    resolved = pd.DataFrame(
        {
            "pair_id": ["0"], "label": [1],
            "modality_a": [lad.CONTACTLESS], "modality_b": [lad.CONTACT],
            "sample_uid_a": ["u_cl"], "sample_uid_b": ["u_ct"],
            "resolved_path_a": [str(cl_path)], "resolved_path_b": [str(ct_path)],
        }
    )
    tmp = tmp_path / "scratch"; tmp.mkdir()
    out = lad.build_transformed_resolved(resolved, "P1_invert", tmp)

    # Contact side path unchanged and file byte-identical.
    assert out.iloc[0]["resolved_path_b"] == str(ct_path)
    assert hashlib.sha256(ct_path.read_bytes()).hexdigest() == ct_before
    # Contactless side redirected to a transformed scratch file (inverted).
    assert out.iloc[0]["resolved_path_a"] != str(cl_path)
    transformed = cv2.imread(out.iloc[0]["resolved_path_a"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(transformed, 255 - _img(10))


def test_p0_keeps_original_paths(tmp_path):
    root = tmp_path / "root"; (root / "cl").mkdir(parents=True)
    cl_path = root / "cl" / "p.bmp"; cv2.imwrite(str(cl_path), _img(5))
    resolved = pd.DataFrame(
        {"pair_id": ["0"], "label": [1], "modality_a": [lad.CONTACTLESS], "modality_b": [lad.CONTACTLESS],
         "sample_uid_a": ["a"], "sample_uid_b": ["a"], "resolved_path_a": [str(cl_path)], "resolved_path_b": [str(cl_path)]}
    )
    out = lad.build_transformed_resolved(resolved, "P0_raw", tmp_path / "sc")
    assert out.iloc[0]["resolved_path_a"] == str(cl_path)
    assert out.iloc[0]["resolved_path_b"] == str(cl_path)


def test_compute_metrics_fraction_nonzero():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([5.0, 0.0, 0.0, 3.0])
    m = lad.compute_metrics(labels, scores)
    assert m["fraction_nonzero_all"] == 0.5
    assert m["fraction_nonzero_genuine"] == 0.5
    assert m["fraction_nonzero_impostor"] == 0.5
    assert m["scored_count"] == 4 and m["failed_count"] == 0
