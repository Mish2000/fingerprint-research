from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from apps.api.method_registry import load_api_method_registry
from pipelines.benchmark import eval_minutiae
from src.fpbench.matchers.minutiae_matcher import (
    MinutiaeConfig,
    extract_minutiae_template,
    minutiae_aggregate_vector,
    score_pair_minutiae,
)


MINUTIAE_METHOD_SEMANTICS_EPOCH = "minutiae_crossing_number_aligned_v2"


def _synthetic_minutiae_image() -> np.ndarray:
    img = np.full((256, 256), 255, dtype=np.uint8)
    for y, x0, branch_x in [
        (60, 38, 102),
        (95, 52, 150),
        (130, 34, 116),
        (165, 56, 174),
        (200, 44, 138),
    ]:
        cv2.line(img, (x0, y), (218, y), 0, 7, cv2.LINE_AA)
        cv2.line(img, (branch_x, y), (branch_x + 42, y - 30), 0, 7, cv2.LINE_AA)
    return img


def _synthetic_different_finger_image() -> np.ndarray:
    img = np.full((256, 256), 255, dtype=np.uint8)
    for y, x0 in [
        (50, 40),
        (90, 60),
        (140, 35),
        (190, 70),
    ]:
        cv2.line(img, (x0, y), (x0 + 50, y + 20), 0, 7, cv2.LINE_AA)
    return img


def _synthetic_bifurcation_heavy_image() -> np.ndarray:
    img = np.full((256, 256), 255, dtype=np.uint8)
    for y, x0, branch_x in [
        (46, 28, 70),
        (78, 36, 105),
        (110, 30, 84),
        (142, 46, 132),
        (174, 34, 100),
        (206, 52, 150),
    ]:
        cv2.line(img, (x0, y), (226, y), 0, 7, cv2.LINE_AA)
        cv2.line(img, (branch_x, y), (branch_x + 44, y - 30), 0, 7, cv2.LINE_AA)
        cv2.line(img, (branch_x + 18, y), (branch_x + 56, y + 26), 0, 7, cv2.LINE_AA)
    return img


def _test_cfg() -> MinutiaeConfig:
    return MinutiaeConfig(
        target_size=256,
        max_minutiae=128,
        spatial_tolerance=18.0,
        angle_tolerance_deg=35.0,
        min_required_minutiae=8,
    )


def _default_threshold_fast_cfg() -> MinutiaeConfig:
    return MinutiaeConfig(
        target_size=256,
        max_minutiae=96,
        spatial_tolerance=14.0,
        angle_tolerance_deg=30.0,
        min_required_minutiae=12,
    )


def test_synthetic_ridge_image_produces_non_empty_minutiae_template() -> None:
    cfg = _test_cfg()
    template = extract_minutiae_template(_synthetic_minutiae_image(), cfg=cfg)

    assert len(template.points) >= 8
    assert len(template.points) < cfg.max_minutiae
    assert any(point.kind == "ending" for point in template.points)
    assert any(point.kind == "bifurcation" for point in template.points)
    assert template.kept_endings > 0
    assert template.kept_bifurcations > 0
    assert template.total_kept_minutiae == len(template.points)
    assert template.raw_candidate_endings >= template.kept_endings
    assert template.raw_candidate_bifurcations >= template.kept_bifurcations
    assert template.skeleton_foreground_pixels == template.skeleton_pixels
    assert template.skeleton_density > 0.0
    assert template.ridge_polarity in {"dark", "bright", "mixed", "mixed_inverse", "empty"}
    assert "saturated" not in template.extraction_quality_flags
    assert template.saturated_by_max_minutiae is False
    assert template.skeleton_pixels > 0
    assert template.roi_fraction > 0.0


def test_bifurcation_heavy_synthetic_pattern_is_not_all_bifurcations() -> None:
    template = extract_minutiae_template(_synthetic_bifurcation_heavy_image(), cfg=_test_cfg())

    assert len(template.points) > 0
    assert template.kept_bifurcations > 0
    assert template.kept_endings > 0
    assert template.kept_bifurcations < len(template.points)


def test_rotated_translated_same_finger_scores_above_mismatch() -> None:
    img = _synthetic_minutiae_image()
    transform = cv2.getRotationMatrix2D((128, 128), 5.0, 1.0)
    transform[:, 2] += [5.0, -4.0]
    moved = cv2.warpAffine(img, transform, (256, 256), flags=cv2.INTER_LINEAR, borderValue=255)
    blank = np.full_like(img, 255)

    same = score_pair_minutiae(img, moved, cfg=_test_cfg())
    mismatch = score_pair_minutiae(img, blank, cfg=_test_cfg())

    assert 0.0 <= same.score <= 1.0
    assert 0.0 <= mismatch.score <= 1.0
    assert same.score > mismatch.score
    assert same.score >= 0.25
    assert same.matched_count > mismatch.matched_count


def test_synthetic_different_finger_scores_below_same_and_default_threshold() -> None:
    img = _synthetic_minutiae_image()
    transform = cv2.getRotationMatrix2D((128, 128), 5.0, 1.0)
    transform[:, 2] += [5.0, -4.0]
    moved = cv2.warpAffine(img, transform, (256, 256), flags=cv2.INTER_LINEAR, borderValue=255)
    different = _synthetic_different_finger_image()
    cfg = _default_threshold_fast_cfg()

    same = score_pair_minutiae(img, moved, cfg=cfg)
    mismatch = score_pair_minutiae(img, different, cfg=cfg)
    mismatch_template = extract_minutiae_template(different, cfg=cfg)
    default_threshold = load_api_method_registry().definition_for("minutiae").decision_threshold

    assert len(mismatch_template.points) > 0
    assert 0.0 <= same.score <= 1.0
    assert 0.0 <= mismatch.score <= 1.0
    assert same.score > mismatch.score
    assert same.score - mismatch.score >= 0.25
    assert mismatch.score <= same.score * 0.75
    assert same.matched_count > mismatch.matched_count
    assert mismatch.score < default_threshold


def test_minutiae_match_result_metadata_contract_is_type_stable() -> None:
    img = _synthetic_minutiae_image()
    transform = cv2.getRotationMatrix2D((128, 128), 5.0, 1.0)
    transform[:, 2] += [5.0, -4.0]
    moved = cv2.warpAffine(img, transform, (256, 256), flags=cv2.INTER_LINEAR, borderValue=255)

    result = score_pair_minutiae(img, moved, cfg=_test_cfg())

    assert isinstance(result.matched_minutiae, tuple)
    assert isinstance(result.tentative_minutiae, tuple)
    assert isinstance(result.minutiae_count_a, int)
    assert isinstance(result.minutiae_count_b, int)
    assert isinstance(result.endings_a, int)
    assert isinstance(result.endings_b, int)
    assert isinstance(result.bifurcations_a, int)
    assert isinstance(result.bifurcations_b, int)
    assert isinstance(result.transform_angle_deg, float)
    assert isinstance(result.transform_dx, float)
    assert isinstance(result.transform_dy, float)
    assert isinstance(result.raw_alignment_score, float)
    assert isinstance(result.score_multiplier, float)
    assert isinstance(result.score_components, dict)
    assert isinstance(result.skeleton_foreground_pixels_a, int)
    assert isinstance(result.skeleton_foreground_pixels_b, int)
    assert isinstance(result.skeleton_density_a, float)
    assert isinstance(result.skeleton_density_b, float)
    assert isinstance(result.raw_candidate_endings_a, int)
    assert isinstance(result.raw_candidate_bifurcations_a, int)
    assert isinstance(result.saturated_by_max_minutiae_a, bool)
    assert isinstance(result.ridge_polarity_a, str)
    assert isinstance(result.extraction_quality_flags_a, tuple)

    assert result.matched_minutiae
    assert result.tentative_minutiae
    for item in result.matched_minutiae[:3] + result.tentative_minutiae[:3]:
        assert isinstance(item, dict)
        assert isinstance(item["a"], tuple)
        assert isinstance(item["b"], tuple)
        assert isinstance(item["a_aligned"], tuple)
        assert len(item["a"]) == 2
        assert len(item["b"]) == 2
        assert len(item["a_aligned"]) == 2
        assert isinstance(item["kind"], str)
        assert isinstance(item["kind_a"], str)
        assert isinstance(item["kind_b"], str)
        assert isinstance(item["distance"], float)
        assert isinstance(item["angle_delta_deg"], float)


def test_empty_image_returns_zero_score_and_sentinel_vector() -> None:
    blank = np.full((256, 256), 255, dtype=np.uint8)
    template = extract_minutiae_template(blank, cfg=_test_cfg())
    result = score_pair_minutiae(blank, blank, cfg=_test_cfg())
    vec = minutiae_aggregate_vector(template, template.image_shape)

    assert template.points == ()
    assert result.score == 0.0
    assert result.matched_count == 0
    assert vec.shape == (512,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) == pytest.approx(1.0)
    assert vec[-1] == pytest.approx(1.0)


def test_minutiae_aggregate_vector_contract_is_deterministic() -> None:
    template = extract_minutiae_template(_synthetic_minutiae_image(), cfg=_test_cfg())

    first = minutiae_aggregate_vector(template, template.image_shape)
    second = minutiae_aggregate_vector(template, template.image_shape)

    assert first.shape == (512,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.linalg.norm(first) == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_eval_minutiae_smoke_writes_scores_and_meta(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    img = _synthetic_minutiae_image()
    transform = cv2.getRotationMatrix2D((128, 128), 5.0, 1.0)
    transform[:, 2] += [5.0, -4.0]
    moved = cv2.warpAffine(img, transform, (256, 256), flags=cv2.INTER_LINEAR, borderValue=255)
    different = _synthetic_different_finger_image()

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_c = tmp_path / "c.png"
    assert cv2.imwrite(str(image_a), img)
    assert cv2.imwrite(str(image_b), moved)
    assert cv2.imwrite(str(image_c), different)

    pairs_csv = tmp_path / "pairs_val.csv"
    pairs_csv.write_text(
        "\n".join(
            [
                "path_a,path_b,label",
                f"{image_a},{image_b},1",
                f"{image_a},{image_c},0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "scores.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_minutiae.py",
            str(out_csv),
            "--pairs",
            str(pairs_csv),
            "--split",
            "val",
            "--limit",
            "2",
            "--target_size",
            "256",
        ],
    )

    eval_minutiae.main()

    meta_path = out_csv.with_suffix(".meta.json")
    assert out_csv.exists()
    assert meta_path.exists()

    with out_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert "score" in rows[0]
    for column in [
        "matched_minutiae",
        "tentative_minutiae",
        "minutiae_count_a",
        "minutiae_count_b",
        "endings_a",
        "endings_b",
            "bifurcations_a",
            "bifurcations_b",
            "skeleton_foreground_pixels_a",
            "skeleton_foreground_pixels_b",
            "skeleton_density_a",
            "skeleton_density_b",
            "raw_candidate_endings_a",
            "raw_candidate_endings_b",
            "raw_candidate_bifurcations_a",
            "raw_candidate_bifurcations_b",
            "saturated_by_max_minutiae_a",
            "saturated_by_max_minutiae_b",
            "ridge_polarity_a",
            "ridge_polarity_b",
            "extraction_quality_flags_a",
            "extraction_quality_flags_b",
            "raw_alignment_score",
            "score_multiplier",
            "transform_angle_deg",
            "transform_dx",
            "transform_dy",
    ]:
        assert column in rows[0]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["method"] == "minutiae"
    assert meta["method_semantics_epoch"] == MINUTIAE_METHOD_SEMANTICS_EPOCH
    assert meta["config"]["method_semantics_epoch"] == MINUTIAE_METHOD_SEMANTICS_EPOCH
    assert meta["template_cache"]["enabled"] is True
    assert meta["template_cache"]["hits"] >= 1


def test_eval_minutiae_template_cache_does_not_change_score_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    img = _synthetic_minutiae_image()
    transform = cv2.getRotationMatrix2D((128, 128), 5.0, 1.0)
    transform[:, 2] += [5.0, -4.0]
    moved = cv2.warpAffine(img, transform, (256, 256), flags=cv2.INTER_LINEAR, borderValue=255)
    different = _synthetic_different_finger_image()

    image_a = tmp_path / "a.png"
    image_b = tmp_path / "b.png"
    image_c = tmp_path / "c.png"
    assert cv2.imwrite(str(image_a), img)
    assert cv2.imwrite(str(image_b), moved)
    assert cv2.imwrite(str(image_c), different)

    pairs_csv = tmp_path / "pairs_val.csv"
    pairs_csv.write_text(
        "\n".join(
            [
                "path_a,path_b,label",
                f"{image_a},{image_b},1",
                f"{image_a},{image_c},0",
                f"{image_a},{image_b},1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cached_csv = tmp_path / "scores_cached.csv"
    uncached_csv = tmp_path / "scores_uncached.csv"

    for out_csv, extra in ((cached_csv, []), (uncached_csv, ["--disable_template_cache"])):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "eval_minutiae.py",
                str(out_csv),
                "--pairs",
                str(pairs_csv),
                "--split",
                "val",
                "--target_size",
                "256",
                "--progress_every",
                "0",
                *extra,
            ],
        )
        eval_minutiae.main()

    with cached_csv.open("r", encoding="utf-8", newline="") as handle:
        cached_rows = list(csv.DictReader(handle))
    with uncached_csv.open("r", encoding="utf-8", newline="") as handle:
        uncached_rows = list(csv.DictReader(handle))

    assert len(cached_rows) == len(uncached_rows)
    for cached, uncached in zip(cached_rows, uncached_rows):
        assert cached["score"] == uncached["score"]
        assert cached["raw_alignment_score"] == uncached["raw_alignment_score"]
        assert cached["minutiae_count_a"] == uncached["minutiae_count_a"]
        assert cached["minutiae_count_b"] == uncached["minutiae_count_b"]

    meta = json.loads(cached_csv.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["template_cache"]["hits"] > 0
