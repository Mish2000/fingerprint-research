from __future__ import annotations

from pathlib import Path

import pytest

import apps.api.main as api_main
from apps.api.service import MatchService, MethodUnavailableError


class _NoopDedicatedMatcher:
    def __init__(self, *args, **kwargs):
        pass


class _FakeDL:
    def __init__(self, *args, **kwargs):
        pass

    def config_dict(self):
        return {"embed_dim": 512}


def _match_kwargs(method: str, tmp_path: Path) -> dict[str, object]:
    return {
        "method": method,
        "path_a": str(tmp_path / "a.png"),
        "path_b": str(tmp_path / "b.png"),
        "threshold": None,
        "return_overlay": False,
        "capture_a": "plain",
        "capture_b": "plain",
        "filename_a": "a.png",
        "filename_b": "b.png",
    }


def test_dl_unavailable_is_reported_and_dl_aliases_raise(tmp_path: Path) -> None:
    def dl_factory(*, dl_cfg, prep_cfg):
        if dl_cfg.backbone == "resnet18":
            raise RuntimeError("resnet18 pretrained weights unavailable")
        return _FakeDL()

    service = MatchService(dl_factory=dl_factory, dedicated_factory=_NoopDedicatedMatcher)
    availability = service.method_availability()

    assert availability["dl"]["available"] is False
    assert "resnet18 pretrained weights unavailable" in str(availability["dl"]["error"])
    assert availability["vit"]["available"] is True

    with pytest.raises(MethodUnavailableError, match="Method 'dl'.*unavailable"):
        service.match(**_match_kwargs("dl", tmp_path))

    with pytest.raises(MethodUnavailableError, match="Method 'dl'.*unavailable"):
        service.match(**_match_kwargs("dl_quick", tmp_path))


def test_vit_unavailable_is_reported_and_vit_raises(tmp_path: Path) -> None:
    def dl_factory(*, dl_cfg, prep_cfg):
        if dl_cfg.backbone == "vit_base":
            raise RuntimeError("vit_base pretrained weights unavailable")
        return _FakeDL()

    service = MatchService(dl_factory=dl_factory, dedicated_factory=_NoopDedicatedMatcher)
    availability = service.method_availability()

    assert availability["dl"]["available"] is True
    assert availability["vit"]["available"] is False
    assert "vit_base pretrained weights unavailable" in str(availability["vit"]["error"])

    with pytest.raises(MethodUnavailableError, match="Method 'vit'.*unavailable"):
        service.match(**_match_kwargs("vit", tmp_path))


def test_health_and_methods_payload_include_dl_vit_availability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def dl_factory(*, dl_cfg, prep_cfg):
        if dl_cfg.backbone == "resnet18":
            raise RuntimeError("resnet18 pretrained weights unavailable")
        if dl_cfg.backbone == "vit_base":
            raise RuntimeError("vit_base pretrained weights unavailable")
        return _FakeDL()

    service = MatchService(dl_factory=dl_factory, dedicated_factory=_NoopDedicatedMatcher)

    monkeypatch.setattr(api_main, "_service", service)
    monkeypatch.setattr(api_main, "_service_init_error", None)
    monkeypatch.setattr(api_main, "_ident_service", None)
    monkeypatch.setattr(api_main, "_ident_service_init_error", None)
    monkeypatch.setattr(api_main, "_initialize_identification_service", lambda: None)
    monkeypatch.setattr(api_main, "_browser_health_fields", lambda: {})

    health_payload = api_main.health()
    methods_payload = api_main.methods()
    entries = {entry["id"]: entry for entry in methods_payload["methods"]}

    assert health_payload["methods"]["dl"]["available"] is False
    assert health_payload["methods"]["vit"]["available"] is False
    assert "resnet18 pretrained weights unavailable" in health_payload["methods"]["dl"]["error"]
    assert health_payload["direct_vector_retrieval_methods"] == [
        "classic_orb",
        "classic_gftt_orb",
        "harris",
        "sift",
        "dl",
        "vit",
    ]
    assert health_payload["rerank_only_methods"] == ["dedicated"]
    assert health_payload["method_capabilities"]["dl"]["retrieval_vector_dim"] == 512
    assert health_payload["method_capabilities"]["sift"]["retrieval_vector_dim"] == 512
    assert health_payload["method_capabilities"]["sift"]["retrieval_vector_kind"] == "sift_aggregated_descriptor_v1"
    dedicated_capability = health_payload["method_capabilities"]["dedicated"]
    assert dedicated_capability["retrieval_unavailable_reason"] == (
        "experimental_rerank_only_no_validated_global_retrieval_vector_yet"
    )
    assert dedicated_capability["retrieval_capability_status"] == "experimental_rerank_only"
    assert dedicated_capability["direct_retrieval_exclusion"] == "intentional_rerank_only"
    assert dedicated_capability["experimental"] is True
    assert entries["dl"]["availability"]["available"] is False
    assert entries["vit"]["availability"]["available"] is False
    assert methods_payload["direct_vector_retrieval_methods"] == [
        "classic_orb",
        "classic_gftt_orb",
        "harris",
        "sift",
        "dl",
        "vit",
    ]
    assert methods_payload["method_capabilities"]["vit"]["retrieval_vector_dim"] == 768
