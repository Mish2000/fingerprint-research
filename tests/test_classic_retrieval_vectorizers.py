from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from apps.api.service import MatchService
from src.fpbench.identification.classic_vectorizers import (
    RETRIEVAL_VECTOR_DIM,
    orb_aggregated_descriptor_vector,
    sift_aggregated_descriptor_vector,
)


class _FakeDL:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def embed_path(self, path: str, capture: str | None = None):
        del path, capture
        return np.ones(512, dtype=np.float32), 0.0


class _FakeDedicatedMatcher:
    def __init__(self, *args, **kwargs) -> None:
        pass


@pytest.fixture(scope="module")
def match_service() -> MatchService:
    return MatchService(dl_factory=_FakeDL, dedicated_factory=_FakeDedicatedMatcher)


@pytest.fixture()
def feature_image(tmp_path: Path) -> Path:
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (24, 24), (232, 232), 180, thickness=3)
    cv2.circle(img, (128, 128), 56, 220, thickness=2)
    cv2.line(img, (42, 190), (214, 62), 255, thickness=2)
    cv2.line(img, (50, 70), (210, 200), 130, thickness=2)
    cv2.putText(img, "FP", (86, 144), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 200, 2, cv2.LINE_AA)
    path = tmp_path / "feature.png"
    assert cv2.imwrite(str(path), img)
    return path


def _assert_vector_contract(vec: np.ndarray) -> None:
    assert vec.shape == (RETRIEVAL_VECTOR_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()
    assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert np.count_nonzero(vec) > 0


@pytest.mark.parametrize(
    "method_name",
    [
        "embed_classic_orb_path",
        "embed_classic_gftt_orb_path",
        "embed_harris_path",
        "embed_sift_path",
    ],
)
def test_classic_match_service_vectorizers_return_512d_finite_float32(
    match_service: MatchService,
    feature_image: Path,
    method_name: str,
) -> None:
    vec = getattr(match_service, method_name)(str(feature_image), capture="plain")

    _assert_vector_contract(vec)


@pytest.mark.parametrize(
    "method_name",
    [
        "embed_classic_orb_path",
        "embed_classic_gftt_orb_path",
        "embed_harris_path",
        "embed_sift_path",
    ],
)
def test_classic_match_service_vectorizers_are_deterministic(
    match_service: MatchService,
    feature_image: Path,
    method_name: str,
) -> None:
    embed = getattr(match_service, method_name)

    first = embed(str(feature_image), capture="plain")
    second = embed(str(feature_image), capture="plain")

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)


def test_no_feature_inputs_return_nonzero_finite_sentinel_vectors() -> None:
    orb_vec = orb_aggregated_descriptor_vector([], None, (256, 256))
    sift_vec = sift_aggregated_descriptor_vector(None)

    _assert_vector_contract(orb_vec)
    _assert_vector_contract(sift_vec)
    assert orb_vec[-1] == pytest.approx(1.0)
    assert sift_vec[-1] == pytest.approx(1.0)
