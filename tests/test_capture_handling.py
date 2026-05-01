from __future__ import annotations

import cv2
import numpy as np
import pytest

from apps.api.service import _normalize_capture_label
from src.fpbench.matchers.baseline_dl import _build_final_mask, DLBaselineConfig, _normalize_capture
from src.fpbench.matchers.dedicated_matcher import _geometry_aware_score, build_final_mask as build_dedicated_mask
from src.fpbench.preprocess.preprocess import extract_fingerprint_roi, rectangular_gate_mask


def _synthetic_fingerprint_like() -> np.ndarray:
    img = np.full((256, 256), 245, dtype=np.uint8)
    mask = np.zeros_like(img)
    cv2.ellipse(mask, (128, 148), (70, 95), 0, 0, 360, 255, -1)
    for i in range(10):
        cv2.ellipse(img, (128, 148), (68 - i * 5, 92 - i * 7), 0, 200, 340, 80 if i % 2 == 0 else 200, 2)
    img = np.where(mask > 0, img, 245).astype(np.uint8)
    cv2.line(img, (96, 170), (156, 100), 70, 3)
    return img


def test_normalize_capture_accepts_explicit_contact_variants():
    assert _normalize_capture("plain") == "plain"
    assert _normalize_capture("roll") == "roll"
    assert _normalize_capture("contactless") == "contactless"
    assert _normalize_capture("contact-based") == "contact_based"


def test_unknown_capture_does_not_silently_fallback_to_plain():
    with pytest.raises(ValueError):
        _normalize_capture("latent")
    with pytest.raises(ValueError):
        _normalize_capture_label("latent")


def test_build_final_mask_supports_contactless_and_contact_based():
    img = np.full((64, 64), 180, dtype=np.uint8)
    cfg = DLBaselineConfig()
    mask_contactless = _build_final_mask(img, cfg, "contactless")
    mask_contact_based = _build_final_mask(img, cfg, "contact_based")
    assert mask_contactless.shape == img.shape
    assert mask_contact_based.shape == img.shape
    assert mask_contactless.dtype == np.uint8
    assert mask_contact_based.dtype == np.uint8


def test_roi_extraction_returns_valid_mask_for_fingerprint_like_image():
    img = _synthetic_fingerprint_like()
    result = extract_fingerprint_roi(img)
    assert result.is_valid is True
    assert result.failure_reason is None
    assert result.mask.shape == img.shape
    assert float(np.mean(result.mask > 0)) > 0.02


def test_roi_extraction_returns_empty_mask_on_blank_image():
    img = np.full((128, 128), 255, dtype=np.uint8)
    result = extract_fingerprint_roi(img)
    assert result.is_valid is False
    assert result.mask.shape == img.shape
    assert int(result.mask.sum()) == 0
    assert result.failure_reason is not None


def test_dl_mask_falls_back_to_gate_when_roi_fails():
    img = np.full((128, 128), 255, dtype=np.uint8)
    cfg = DLBaselineConfig()
    mask = _build_final_mask(img, cfg, "plain")
    gate = rectangular_gate_mask(
        img.shape[:2],
        capture="plain",
        top_plain=cfg.gate_top_plain,
        top_roll=cfg.gate_top_roll,
        border=cfg.gate_border,
    )
    assert np.array_equal(mask, gate)


def test_dedicated_mask_falls_back_to_gate_when_roi_fails():
    img = np.full((128, 128), 255, dtype=np.uint8)
    mask = build_dedicated_mask(img, capture="plain")
    gate = rectangular_gate_mask(img.shape[:2], capture="plain", top_plain=0.18, top_roll=0.05, border=12)
    assert np.array_equal(mask, gate)


def test_geometry_aware_score_rewards_better_geometry():
    score_lo, _ = _geometry_aware_score(0.8, 0.8, 0.1, 0.1)
    score_hi, _ = _geometry_aware_score(0.8, 0.8, 0.8, 0.8)
    assert score_hi > score_lo
