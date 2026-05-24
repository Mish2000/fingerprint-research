from __future__ import annotations

import json

import pytest
import yaml
from fastapi.testclient import TestClient

import apps.api.main as api_main
from apps.api.method_registry import METHODS_CONFIG_PATH
from src.fpbench.fingerprint_engine.registry import get_engine, list_engine_metadata, list_engines


def _clear_sourceafis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCEAFIS_ENABLED", raising=False)
    monkeypatch.delenv("SOURCEAFIS_SERVICE_URL", raising=False)
    monkeypatch.delenv("SOURCEAFIS_CLI_PATH", raising=False)


def test_sourceafis_provider_is_registered_but_unavailable_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sourceafis_env(monkeypatch)

    provider_ids = list_engines()
    metadata_by_provider = {metadata.provider_id: metadata for metadata in list_engine_metadata()}

    assert "sourceafis_open" in provider_ids
    sourceafis = metadata_by_provider["sourceafis_open"]
    assert sourceafis.provider_id == "sourceafis_open"
    assert sourceafis.name == "SourceAFIS Open Matcher"
    assert sourceafis.available is False
    assert sourceafis.unavailable_reason
    assert "SOURCEAFIS_SERVICE_URL" in sourceafis.unavailable_reason
    assert sourceafis.capabilities.supports_template_extraction is True
    assert sourceafis.capabilities.supports_verification is True
    assert sourceafis.capabilities.supports_identification is True
    assert sourceafis.capabilities.supports_quality is False
    assert sourceafis.capabilities.supports_template_storage is True
    assert sourceafis.template_format == "sourceafis"


def test_get_engine_returns_sourceafis_optional_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sourceafis_env(monkeypatch)

    engine = get_engine("sourceafis_open")

    assert engine.metadata().provider_id == "sourceafis_open"
    assert engine.metadata().available is False


def test_api_metadata_includes_unavailable_sourceafis_without_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FPBENCH_API_LAZY_STARTUP", "1")
    monkeypatch.delenv("FINGERPRINT_ENGINE_PROVIDER", raising=False)
    _clear_sourceafis_env(monkeypatch)
    api_main._shutdown_services()

    with TestClient(api_main.app) as test_client:
        response = test_client.get("/fingerprint-engine/metadata")

    api_main._shutdown_services()
    assert response.status_code == 200, response.text
    payload = response.json()
    engines = {engine["provider_id"]: engine for engine in payload["engines"]}

    assert payload["selected_provider"] == "null"
    assert "sourceafis_open" in payload["registered_provider_ids"]
    assert engines["sourceafis_open"]["available"] is False
    assert "SOURCEAFIS_SERVICE_URL" in engines["sourceafis_open"]["unavailable_reason"]
    assert engines["sourceafis_open"]["capabilities"]["supports_identification"] is True
    assert engines["sourceafis_open"]["capabilities"]["supports_quality"] is False
    assert "template_bytes" not in json.dumps(payload)


def test_methods_config_documents_sourceafis_without_enabling_runtime_namespace() -> None:
    payload = yaml.safe_load(METHODS_CONFIG_PATH.read_text(encoding="utf-8"))
    entry = payload["methods"]["sourceafis_open"]

    assert entry["label"] == "SourceAFIS Open Matcher"
    assert entry["family"] == "open_source_afis"
    assert entry["track"] == "open_source_baseline"
    assert entry["showcase_eligible"] is True
    assert entry["benchmark_default"] is False
    assert entry["canonical_default"] is False
    assert entry["supports_verification"] is True
    assert entry["supports_identification"] is True
    assert entry["supports_quality"] is False
    assert entry["supports_template_storage"] is True
    assert entry["provider_id"] == "sourceafis_open"
    assert "sourceafis_open" not in payload["namespaces"]["api_runtime"]
    assert "sourceafis_open" not in payload["namespaces"]["benchmark_runtime"]
