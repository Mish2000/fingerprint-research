from __future__ import annotations

import cv2
import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from apps.api.main import app
from apps.api.method_registry import METHODS_CONFIG_PATH, THRESHOLDS_CONFIG_PATH, load_api_method_registry

DIRECT_RETRIEVAL_METHODS = ["classic_orb", "classic_gftt_orb", "harris", "sift", "dl", "vit"]
RERANK_ONLY_METHODS = ["dedicated"]
DEDICATED_RETRIEVAL_UNAVAILABLE_REASON = "experimental_rerank_only_no_validated_global_retrieval_vector_yet"
DEDICATED_FUTURE_ADAPTER_HINT = (
    "A future dedicated_aggregated_patch_descriptor_v1 adapter can be added once global pooling is validated."
)
CLASSIC_RETRIEVAL_KINDS = {
    "classic_orb": "classic_orb_aggregated_descriptor_v1",
    "classic_gftt_orb": "classic_gftt_orb_aggregated_descriptor_v1",
    "harris": "harris_orb_aggregated_descriptor_v1",
    "sift": "sift_aggregated_descriptor_v1",
}


def _png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(512, 512), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok, "cv2.imencode failed"
    return buf.tobytes()


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _post_match(client: TestClient, method: str):
    files = {
        "img_a": ("a.png", _png_bytes(11), "image/png"),
        "img_b": ("b.png", _png_bytes(12), "image/png"),
    }
    data = {
        "method": method,
        "return_overlay": "false",
        "capture_a": "plain",
        "capture_b": "plain",
    }
    return client.post("/match", data=data, files=files)


@pytest.mark.parametrize(
    ("method", "label", "benchmark_method"),
    [
        ("classic_orb", "Classic (ORB)", "classic_orb"),
        ("classic_gftt_orb", "Classic (ROI GFTT+ORB)", "classic_v2"),
    ],
)
def test_match_supports_split_public_classic_methods(
    client: TestClient,
    method: str,
    label: str,
    benchmark_method: str,
) -> None:
    response = _post_match(client, method)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["method"] == method
    assert payload["method_metadata"]["canonical_method"] == method
    assert payload["method_metadata"]["requested_method"] == method
    assert payload["method_metadata"]["benchmark_method"] == benchmark_method
    assert payload["method_metadata"]["method_label"] == label
    assert payload["method_metadata"]["resolved_from_alias"] is False


@pytest.mark.parametrize(
    ("requested_method", "canonical_method", "label", "benchmark_method"),
    [
        ("classic", "classic_orb", "Classic (ORB)", "classic_orb"),
        ("classic_v2", "classic_gftt_orb", "Classic (ROI GFTT+ORB)", "classic_v2"),
    ],
)
def test_match_accepts_classic_aliases(
    client: TestClient,
    requested_method: str,
    canonical_method: str,
    label: str,
    benchmark_method: str,
) -> None:
    response = _post_match(client, requested_method)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["method"] == canonical_method
    assert payload["method_metadata"]["canonical_method"] == canonical_method
    assert payload["method_metadata"]["requested_method"] == requested_method
    assert payload["method_metadata"]["benchmark_method"] == benchmark_method
    assert payload["method_metadata"]["method_label"] == label
    assert payload["method_metadata"]["resolved_from_alias"] is True


def test_match_accepts_dl_quick_alias(client: TestClient) -> None:
    response = _post_match(client, "dl_quick")

    if response.status_code == 503:
        assert "Method 'dl'" in response.json()["detail"]
        assert "unavailable" in response.json()["detail"]
        return

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["method"] == "dl"
    assert payload["method_metadata"]["canonical_method"] == "dl"
    assert payload["method_metadata"]["requested_method"] == "dl_quick"
    assert payload["method_metadata"]["benchmark_method"] == "dl_quick"
    assert payload["method_metadata"]["method_label"] == "Deep Learning (ResNet18)"
    assert payload["method_metadata"]["family"] == "global_embedding"
    assert payload["method_metadata"]["status"] == "active"
    assert payload["method_metadata"]["embedding_dim"] == 512
    assert payload["method_metadata"]["resolved_from_alias"] is True


def test_methods_endpoint_returns_structured_catalog(client: TestClient) -> None:
    response = client.get("/methods")

    assert response.status_code == 200, response.text
    payload = response.json()
    entries = {entry["id"]: entry for entry in payload["methods"]}

    assert set(entries) >= {"classic_orb", "classic_gftt_orb", "harris", "sift", "dl", "dedicated", "vit"}
    assert payload["direct_vector_retrieval_methods"] == DIRECT_RETRIEVAL_METHODS
    assert payload["rerank_only_methods"] == RERANK_ONLY_METHODS
    assert set(payload["method_capabilities"]) >= set(entries)

    classic_orb = entries["classic_orb"]
    assert classic_orb["label"] == "Classic (ORB)"
    assert classic_orb["benchmark_name"] == "classic_orb"
    assert "classic" in classic_orb["aliases"]
    assert classic_orb["thresholds"]["decision"] == pytest.approx(0.01)
    assert classic_orb["identification_roles"]["retrieval_capable"] is True
    assert classic_orb["identification_roles"]["rerank_capable"] is True
    assert classic_orb["identification_roles"]["supports_direct_vector_retrieval"] is True
    assert classic_orb["identification_roles"]["supports_pairwise_rerank"] is True
    assert classic_orb["identification_roles"]["retrieval_vector_dim"] == 512
    assert classic_orb["identification_roles"]["retrieval_vector_kind"] == CLASSIC_RETRIEVAL_KINDS["classic_orb"]
    assert classic_orb["identification_roles"]["retrieval_distance_metric"] == "cosine"
    assert classic_orb["identification_roles"]["retrieval_unavailable_reason"] is None

    classic_gftt_orb = entries["classic_gftt_orb"]
    assert classic_gftt_orb["label"] == "Classic (ROI GFTT+ORB)"
    assert classic_gftt_orb["benchmark_name"] == "classic_v2"
    assert "classic_v2" in classic_gftt_orb["aliases"]
    assert classic_gftt_orb["thresholds"]["decision"] == pytest.approx(0.01)
    assert classic_gftt_orb["identification_roles"]["retrieval_capable"] is True
    assert classic_gftt_orb["identification_roles"]["rerank_capable"] is True
    assert classic_gftt_orb["retrieval_capability"]["supports_direct_vector_retrieval"] is True
    assert classic_gftt_orb["retrieval_capability"]["retrieval_vector_dim"] == 512
    assert (
        classic_gftt_orb["retrieval_capability"]["retrieval_vector_kind"]
        == CLASSIC_RETRIEVAL_KINDS["classic_gftt_orb"]
    )
    assert classic_gftt_orb["retrieval_capability"]["retrieval_distance_metric"] == "cosine"

    dl_entry = entries["dl"]
    assert dl_entry["label"] == "Deep Learning (ResNet18)"
    assert dl_entry["benchmark_name"] == "dl_quick"
    assert "dl_quick" in dl_entry["aliases"]
    assert dl_entry["family"] == "global_embedding"
    assert dl_entry["status"] == "active"
    assert dl_entry["embedding_dim"] == 512
    assert dl_entry["runtime_defaults"]["backbone"] == "resnet18"
    assert dl_entry["benchmark_defaults"]["backbone"] == "resnet18"
    assert dl_entry["benchmark_defaults"]["use_mask"] is True
    assert dl_entry["identification_roles"]["retrieval_capable"] is True
    assert dl_entry["identification_roles"]["rerank_capable"] is True
    assert dl_entry["identification_roles"]["retrieval_vector_dim"] == 512
    assert dl_entry["identification_roles"]["retrieval_vector_kind"] == "deep_embedding_resnet"
    assert dl_entry["identification_roles"]["retrieval_distance_metric"] == "cosine"
    assert dl_entry["retrieval_capability"] == payload["method_capabilities"]["dl"]
    assert isinstance(dl_entry["availability"]["available"], bool)
    if not dl_entry["availability"]["available"]:
        assert dl_entry["availability"]["error"]

    vit_entry = entries["vit"]
    assert vit_entry["label"] == "Deep Learning (ViT)"
    assert vit_entry["benchmark_name"] == "vit"
    assert vit_entry["family"] == "global_embedding"
    assert vit_entry["status"] == "active"
    assert vit_entry["embedding_dim"] == 768
    assert vit_entry["benchmark_defaults"]["use_mask"] is True
    assert vit_entry["identification_roles"]["retrieval_vector_dim"] == 768
    assert vit_entry["identification_roles"]["retrieval_vector_kind"] == "vit_embedding"
    assert vit_entry["identification_roles"]["retrieval_distance_metric"] == "cosine"

    harris_entry = entries["harris"]
    assert harris_entry["benchmark_defaults"]["score_mode"] == "inliers_over_min_keypoints"
    assert harris_entry["benchmark_defaults"]["geometry_model"] == "homography"
    assert harris_entry["benchmark_defaults"]["method_semantics_epoch"] == "harris_runtime_aligned_v1"
    assert harris_entry["retrieval_capability"]["supports_direct_vector_retrieval"] is True
    assert harris_entry["retrieval_capability"]["retrieval_vector_dim"] == 512
    assert harris_entry["retrieval_capability"]["retrieval_vector_kind"] == CLASSIC_RETRIEVAL_KINDS["harris"]
    assert harris_entry["retrieval_capability"]["retrieval_distance_metric"] == "cosine"

    sift_entry = entries["sift"]
    assert sift_entry["benchmark_defaults"]["score_mode"] == "inliers_over_min_keypoints"
    assert sift_entry["benchmark_defaults"]["geometry_model"] == "homography"
    assert sift_entry["benchmark_defaults"]["method_semantics_epoch"] == "sift_runtime_aligned_v1"
    assert sift_entry["retrieval_capability"]["supports_direct_vector_retrieval"] is True
    assert sift_entry["retrieval_capability"]["retrieval_vector_dim"] == 512
    assert sift_entry["retrieval_capability"]["retrieval_vector_kind"] == CLASSIC_RETRIEVAL_KINDS["sift"]
    assert sift_entry["retrieval_capability"]["retrieval_distance_metric"] == "cosine"

    dedicated_entry = entries["dedicated"]
    assert dedicated_entry["status"] == "experimental"
    assert dedicated_entry["presentation_tier"] == "research"
    assert dedicated_entry["showcase_eligible"] is False
    assert dedicated_entry["benchmark_default"] is False
    assert dedicated_entry["canonical_default"] is False
    assert dedicated_entry["research_track"] is True
    assert dedicated_entry["promotion_required"] is True
    assert dedicated_entry["is_experimental"] is True
    assert dedicated_entry["is_showcase_eligible"] is False
    assert dedicated_entry["promotion_criteria"]
    assert "must not be promoted" in dedicated_entry["showcase_exclusion_note"]
    assert dedicated_entry["identification_roles"]["experimental"] is True
    assert dedicated_entry["identification_roles"]["retrieval_capability_status"] == "experimental_rerank_only"
    assert dedicated_entry["identification_roles"]["direct_retrieval_exclusion"] == "intentional_rerank_only"
    assert dedicated_entry["identification_roles"]["supports_direct_vector_retrieval"] is False
    assert dedicated_entry["identification_roles"]["supports_pairwise_rerank"] is True
    assert dedicated_entry["identification_roles"]["retrieval_unavailable_reason"] == DEDICATED_RETRIEVAL_UNAVAILABLE_REASON
    assert dedicated_entry["identification_roles"]["future_adapter_hint"] == DEDICATED_FUTURE_ADAPTER_HINT
    assert dedicated_entry["retrieval_capability"]["experimental"] is True
    assert dedicated_entry["retrieval_capability"]["presentation_tier"] == "research"
    assert dedicated_entry["retrieval_capability"]["retrieval_capability_status"] == "experimental_rerank_only"
    assert dedicated_entry["retrieval_capability"]["direct_retrieval_exclusion"] == "intentional_rerank_only"


def test_registry_exposes_method_level_retrieval_capability_contract() -> None:
    registry = load_api_method_registry()

    assert registry.direct_vector_retrieval_methods() == tuple(DIRECT_RETRIEVAL_METHODS)
    assert registry.rerank_only_methods() == tuple(RERANK_ONLY_METHODS)

    specs = registry.retrieval_vector_specs()
    for method, vector_kind in CLASSIC_RETRIEVAL_KINDS.items():
        assert specs[method].dim == 512
        assert specs[method].vector_kind == vector_kind
        assert specs[method].distance_metric == "cosine"
    assert specs["dl"].dim == 512
    assert specs["dl"].vector_kind == "deep_embedding_resnet"
    assert specs["dl"].distance_metric == "cosine"
    assert specs["vit"].dim == 768
    assert specs["vit"].vector_kind == "vit_embedding"
    assert specs["vit"].distance_metric == "cosine"

    capabilities = registry.method_capabilities()
    assert set(capabilities) >= set(registry.supported_method_names())
    for method, vector_kind in CLASSIC_RETRIEVAL_KINDS.items():
        capability = capabilities[method]
        assert capability["supports_pairwise_rerank"] is True
        assert capability["supports_direct_vector_retrieval"] is True
        assert capability["retrieval_vector_dim"] == 512
        assert capability["retrieval_vector_kind"] == vector_kind
        assert capability["retrieval_distance_metric"] == "cosine"
        assert capability["retrieval_unavailable_reason"] is None

    dedicated_capability = capabilities["dedicated"]
    assert dedicated_capability["supports_pairwise_rerank"] is True
    assert dedicated_capability["supports_direct_vector_retrieval"] is False
    assert dedicated_capability["retrieval_vector_dim"] is None
    assert dedicated_capability["retrieval_vector_kind"] is None
    assert dedicated_capability["retrieval_distance_metric"] is None
    assert dedicated_capability["experimental"] is True
    assert dedicated_capability["research_track"] is True
    assert dedicated_capability["retrieval_capability_status"] == "experimental_rerank_only"
    assert dedicated_capability["direct_retrieval_exclusion"] == "intentional_rerank_only"
    assert dedicated_capability["retrieval_unavailable_reason"] == DEDICATED_RETRIEVAL_UNAVAILABLE_REASON
    assert dedicated_capability["future_adapter_hint"] == DEDICATED_FUTURE_ADAPTER_HINT


def test_threshold_registry_marks_dl_and_vit_benchmark_defaults_as_masked() -> None:
    payload = yaml.safe_load(THRESHOLDS_CONFIG_PATH.read_text(encoding="utf-8"))

    assert "multimodal_frozen" not in payload
    assert payload["dl_baseline"]["benchmark_defaults"]["use_mask_default"] is True
    assert payload["dl_baseline"]["benchmark_defaults"]["backbone_default"] == "resnet18"
    assert payload["vit_baseline"]["benchmark_defaults"]["use_mask_default"] is True
    assert payload["classic_harris"]["benchmark_defaults"]["score_mode"] == "inliers_over_min_keypoints"
    assert payload["classic_harris"]["benchmark_defaults"]["geometry_model"] == "homography"
    assert payload["classic_sift"]["benchmark_defaults"]["score_mode"] == "inliers_over_min_keypoints"
    assert payload["classic_sift"]["benchmark_defaults"]["geometry_model"] == "homography"


def test_active_method_registry_is_fingerprint_only() -> None:
    methods_payload = yaml.safe_load(METHODS_CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds_payload = yaml.safe_load(THRESHOLDS_CONFIG_PATH.read_text(encoding="utf-8"))
    registry = load_api_method_registry()

    assert "frozen_or_archived" not in methods_payload
    assert "multimodal_frozen" not in thresholds_payload
    assert set(registry.supported_method_names()) == {
        "classic_orb",
        "classic_gftt_orb",
        "harris",
        "sift",
        "dl",
        "dedicated",
        "vit",
    }

    blocked_tokens = ("face", "multimodal")
    for method_name in registry.supported_method_names():
        assert all(token not in method_name for token in blocked_tokens)

    api_runtime = set(methods_payload["namespaces"]["api_runtime"])
    benchmark_runtime = set(methods_payload["namespaces"]["benchmark_runtime"])
    assert "fusion_balanced_v1" not in api_runtime
    assert "fusion_balanced_v1" in benchmark_runtime
    assert registry.canonical_benchmark_methods() == ("classic_v2", "harris", "sift", "dl_quick", "vit")
    assert "dedicated" not in registry.canonical_benchmark_methods()
    assert "dedicated" in registry.research_methods()

    fusion_payload = methods_payload["methods"]["fusion_balanced_v1"]
    fusion_text = " ".join(
        [
            str(fusion_payload["ui_label"]),
            str(fusion_payload["family"]),
            *[str(note) for note in fusion_payload.get("notes", [])],
        ]
    ).lower()
    assert "fingerprint" in fusion_text
    assert "cross-biometric" in fusion_text
