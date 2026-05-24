from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import apps.api.main as api_main


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FPBENCH_API_LAZY_STARTUP", "1")
    monkeypatch.delenv("FINGERPRINT_ENGINE_PROVIDER", raising=False)
    api_main._shutdown_services()
    with TestClient(api_main.app) as test_client:
        yield test_client
    api_main._shutdown_services()


def test_health_includes_safe_fingerprint_engine_metadata(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200, response.text
    payload = response.json()
    engine_payload = payload["fingerprint_engine"]
    engines = {engine["provider_id"]: engine for engine in engine_payload["engines"]}

    assert engine_payload["selected_provider"] == "null"
    assert engine_payload["selection_error"] is None
    assert set(engine_payload["registered_provider_ids"]) >= {"null", "cots_afis_primary_stub"}
    assert engines["null"]["available"] is True
    assert engines["null"]["capabilities"]["supports_verification"] is True
    assert engines["null"]["capabilities"]["supports_identification"] is True
    assert engines["cots_afis_primary_stub"]["available"] is False
    assert "real SDK adapter" in engines["cots_afis_primary_stub"]["unavailable_reason"]
    assert engines["cots_afis_primary_stub"]["capabilities"]["supports_quality"] is True
    assert "template_bytes" not in json.dumps(engine_payload)


def test_fingerprint_engine_metadata_route_respects_configured_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FPBENCH_API_LAZY_STARTUP", "1")
    monkeypatch.setenv("FINGERPRINT_ENGINE_PROVIDER", "cots_afis_primary_stub")
    api_main._shutdown_services()

    with TestClient(api_main.app) as test_client:
        response = test_client.get("/fingerprint-engine/metadata")

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["selected_provider"] == "cots_afis_primary_stub"
    assert payload["selected_engine"]["available"] is False
    assert payload["selected_engine"]["name"] == "COTS AFIS provider stub"
    assert payload["selected_engine"]["capabilities"]["supports_identification"] is True
    assert "real SDK adapter" in payload["selected_engine"]["unavailable_reason"]
    assert "template_bytes" not in json.dumps(payload)
