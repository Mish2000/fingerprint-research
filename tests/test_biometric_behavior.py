from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from apps.api.identification_service import IdentificationService
from apps.api.schemas import MatchMethod
from src.fpbench.matchers.matching_baseline import SIFTConfig, match_sift, sift_extract
from src.fpbench.preprocess.preprocess import PreprocessConfig, load_gray, preprocess_image
from tests.test_identification_pipeline import InMemoryStore


PREP_CFG = PreprocessConfig(target_size=512)
SIFT_CFG = SIFTConfig()


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
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[:, 2] += shift
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=245)

    rng = np.random.default_rng(123 if person == "alice" else 456)
    noise = rng.normal(0, 4, (h, w)).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _write_png(path: Path, img: np.ndarray) -> Path:
    ok = cv2.imwrite(str(path), img)
    assert ok
    return path


def _sift_pair_score(path_a: str, path_b: str) -> float:
    img1 = preprocess_image(load_gray(path_a), PREP_CFG)
    img2 = preprocess_image(load_gray(path_b), PREP_CFG)
    kps1, desc1 = sift_extract(img1, None, SIFT_CFG)
    kps2, desc2 = sift_extract(img2, None, SIFT_CFG)
    if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
        return 0.0
    matches = match_sift(desc1, desc2, ratio=0.75)
    if len(matches) < 8:
        return 0.0
    pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=3.0)
    inliers = int(mask.reshape(-1).sum()) if mask is not None else 0
    denom = max(1, min(len(kps1), len(kps2)))
    return float(inliers / denom)


def _embedding_vectorizer(path: str, capture: str | None = None) -> np.ndarray:
    img = load_gray(path)
    x = preprocess_image(img, PreprocessConfig(target_size=256))
    x = cv2.GaussianBlur(x, (5, 5), 0)
    x = cv2.resize(x, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    vec = x.reshape(-1)
    return vec / (np.linalg.norm(vec) + 1e-6)


def test_local_feature_matcher_scores_same_person_above_mismatch(tmp_path: Path) -> None:
    alice = _write_png(tmp_path / 'alice.png', _synthetic_person('alice'))
    alice_probe = _write_png(tmp_path / 'alice_probe.png', _synthetic_person('alice', angle=7.0, shift=(8.0, -6.0)))
    bob = _write_png(tmp_path / 'bob.png', _synthetic_person('bob'))

    same_score = _sift_pair_score(str(alice), str(alice_probe))
    diff_score = _sift_pair_score(str(alice), str(bob))

    assert same_score > diff_score


def test_embedding_vectorizer_scores_same_person_above_mismatch(tmp_path: Path) -> None:
    alice = _write_png(tmp_path / 'alice.png', _synthetic_person('alice'))
    alice_probe = _write_png(tmp_path / 'alice_probe.png', _synthetic_person('alice', angle=7.0, shift=(8.0, -6.0)))
    bob = _write_png(tmp_path / 'bob.png', _synthetic_person('bob'))

    vec_alice = _embedding_vectorizer(str(alice))
    vec_probe = _embedding_vectorizer(str(alice_probe))
    vec_bob = _embedding_vectorizer(str(bob))

    same_score = float(np.dot(vec_alice, vec_probe))
    diff_score = float(np.dot(vec_alice, vec_bob))
    assert same_score > diff_score


def test_identification_shortlist_and_rerank_returns_correct_candidate(tmp_path: Path) -> None:
    alice = _write_png(tmp_path / 'alice.png', _synthetic_person('alice'))
    alice_probe = _write_png(tmp_path / 'alice_probe.png', _synthetic_person('alice', angle=7.0, shift=(8.0, -6.0)))
    bob = _write_png(tmp_path / 'bob.png', _synthetic_person('bob'))

    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={'dl': _embedding_vectorizer, 'vit': _embedding_vectorizer},
        rerank_callable=lambda method, probe_path, candidate_path, probe_capture, candidate_capture: _sift_pair_score(probe_path, candidate_path),
    )

    service.enroll_from_path(
        path=str(alice),
        full_name='Alice Levi',
        national_id='111111111',
        capture='plain',
        vector_methods=('dl',),
    )
    service.enroll_from_path(
        path=str(bob),
        full_name='Bob Cohen',
        national_id='222222222',
        capture='plain',
        vector_methods=('dl',),
    )

    result = service.identify_from_path(
        path=str(alice_probe),
        capture='plain',
        retrieval_method='dl',
        rerank_method=MatchMethod.sift,
        shortlist_size=2,
    )

    assert result.top_candidate is not None
    assert result.top_candidate.full_name == 'Alice Levi'
    assert result.candidates[0].rerank_score is not None
    assert result.candidates[0].rerank_score > result.candidates[1].rerank_score
