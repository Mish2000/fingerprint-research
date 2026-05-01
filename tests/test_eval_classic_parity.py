from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pipelines.benchmark import eval_classic
from src.fpbench.matchers.matching_baseline import (
    HarrisConfig,
    SIFTConfig,
    harris_extract,
    match_harris,
    match_sift,
    ransac_inliers,
    sift_extract,
)
from src.fpbench.preprocess.preprocess import PreprocessConfig, load_gray, preprocess_image

PREP_CFG = PreprocessConfig(target_size=512)
HARRIS_CFG = HarrisConfig(nfeatures=1500, max_points=1200)
SIFT_CFG = SIFTConfig(nfeatures=1500)


def _synthetic_person(person: str, *, angle: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    h = w = 512
    img = np.full((h, w), 245, dtype=np.uint8)
    cx, cy = w // 2, h // 2 + 20

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (150, 210), 0, 0, 360, 255, -1)

    for i in range(16):
        ax = 145 - i * 7
        ay = 205 - i * 10
        if ax <= 0 or ay <= 0:
            break
        offset = -25 if person == "alice" else 25
        center = (cx + offset, cy - 20 if person == "alice" else cy + 10)
        cv2.ellipse(img, center, (ax, ay), 0, 200, 340, 90 if i % 2 == 0 else 200, 2)

    if person == "alice":
        cv2.line(img, (190, 310), (320, 180), 60, 4)
        cv2.circle(img, (250, 240), 16, 80, 2)
        cv2.line(img, (250, 140), (250, 360), 120, 1)
    else:
        cv2.line(img, (170, 210), (350, 330), 60, 4)
        cv2.circle(img, (290, 275), 22, 80, 2)
        cv2.line(img, (180, 260), (350, 260), 120, 1)

    img = np.where(mask > 0, img, 245).astype(np.uint8)
    transform = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    transform[:, 2] += shift
    img = cv2.warpAffine(img, transform, (w, h), flags=cv2.INTER_LINEAR, borderValue=245)

    rng = np.random.default_rng(123 if person == "alice" else 456)
    noise = rng.normal(0, 4, (h, w)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _write_png(path: Path, img: np.ndarray) -> Path:
    ok = cv2.imwrite(str(path), img)
    assert ok
    return path


def _runtime_harris_result(path_a: str, path_b: str) -> tuple[float, int, int, int, int]:
    img1 = preprocess_image(load_gray(path_a), PREP_CFG)
    img2 = preprocess_image(load_gray(path_b), PREP_CFG)
    kps1, desc1 = harris_extract(img1, None, HARRIS_CFG)
    kps2, desc2 = harris_extract(img2, None, HARRIS_CFG)
    if desc1 is None or desc2 is None:
        return 0.0, 0, 0, len(kps1), len(kps2)
    matches = match_harris(desc1, desc2, ratio=0.75)
    inliers, _ = ransac_inliers(kps1, kps2, matches, reproj=3.0)
    denom = max(1, min(len(kps1), len(kps2)))
    return float(inliers) / float(denom), inliers, len(matches), len(kps1), len(kps2)


def _runtime_sift_result(path_a: str, path_b: str) -> tuple[float, int, int, int, int]:
    img1 = preprocess_image(load_gray(path_a), PREP_CFG)
    img2 = preprocess_image(load_gray(path_b), PREP_CFG)
    kps1, desc1 = sift_extract(img1, None, SIFT_CFG)
    kps2, desc2 = sift_extract(img2, None, SIFT_CFG)
    if desc1 is None or desc2 is None:
        return 0.0, 0, 0, len(kps1), len(kps2)
    matches = match_sift(desc1, desc2, ratio=0.75)
    inliers, _ = ransac_inliers(kps1, kps2, matches, reproj=3.0)
    denom = max(1, min(len(kps1), len(kps2)))
    return float(inliers) / float(denom), inliers, len(matches), len(kps1), len(kps2)


def _benchmark_result(path_a: str, path_b: str, detector: str) -> tuple[float, int, int, int, int]:
    eval_classic.extract.cache_clear()
    kps1, desc1, _ = eval_classic.extract(path_a, detector, 1500, 512, 512)
    kps2, desc2, _ = eval_classic.extract(path_b, detector, 1500, 512, 512)
    score, inliers, matches = eval_classic.match_and_score(
        desc1,
        desc2,
        kps1,
        kps2,
        score_mode="inliers_over_min_keypoints",
        ratio=0.75,
        ransac_thresh=3.0,
        detector=detector,
        normalization_k=1500,
    )
    return score, inliers, matches, len(kps1), len(kps2)


def test_harris_benchmark_matches_runtime_scoring_on_same_pair(tmp_path: Path) -> None:
    path_a = _write_png(tmp_path / "alice.png", _synthetic_person("alice"))
    path_b = _write_png(
        tmp_path / "alice_probe.png",
        _synthetic_person("alice", angle=7.0, shift=(8.0, -6.0)),
    )

    benchmark_result = _benchmark_result(str(path_a), str(path_b), "harris_orb")
    runtime_result = _runtime_harris_result(str(path_a), str(path_b))

    assert benchmark_result[0] == pytest.approx(runtime_result[0])
    assert benchmark_result[1:] == runtime_result[1:]


def test_sift_benchmark_matches_runtime_scoring_on_same_pair(tmp_path: Path) -> None:
    path_a = _write_png(tmp_path / "alice.png", _synthetic_person("alice"))
    path_b = _write_png(
        tmp_path / "alice_probe.png",
        _synthetic_person("alice", angle=7.0, shift=(8.0, -6.0)),
    )

    benchmark_result = _benchmark_result(str(path_a), str(path_b), "sift")
    runtime_result = _runtime_sift_result(str(path_a), str(path_b))

    assert benchmark_result[0] == pytest.approx(runtime_result[0])
    assert benchmark_result[1:] == runtime_result[1:]
