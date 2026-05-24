from __future__ import annotations

import pytest

from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError
from src.fpbench.fingerprint_engine.providers.sourceafis_provider import SourceAfisFingerprintEngine
from src.fpbench.fingerprint_engine.types import FingerprintImage, FingerprintTemplate


def _clear_sourceafis_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOURCEAFIS_ENABLED", raising=False)
    monkeypatch.delenv("SOURCEAFIS_SERVICE_URL", raising=False)
    monkeypatch.delenv("SOURCEAFIS_CLI_PATH", raising=False)


def _placeholder_template() -> FingerprintTemplate:
    return FingerprintTemplate(
        provider_id="sourceafis_open",
        provider_version="3.18.1",
        template_format="sourceafis",
        template_version="3.18.1",
        template_bytes=b"serialized-sourceafis-template",
    )


def test_sourceafis_operations_raise_provider_unavailable_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sourceafis_env(monkeypatch)
    engine = SourceAfisFingerprintEngine()
    template = _placeholder_template()

    with pytest.raises(ProviderUnavailableError, match="SOURCEAFIS_SERVICE_URL"):
        engine.extract_template(FingerprintImage(image_bytes=b"synthetic fingerprint", image_id="probe"))
    with pytest.raises(ProviderUnavailableError, match="SOURCEAFIS_SERVICE_URL"):
        engine.verify(template, template)
    with pytest.raises(ProviderUnavailableError, match="SOURCEAFIS_SERVICE_URL"):
        engine.identify(template, [])

    assert engine.assess_quality(FingerprintImage(image_bytes=b"synthetic fingerprint")) is None


def test_sourceafis_enabled_true_without_url_has_actionable_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sourceafis_env(monkeypatch)
    monkeypatch.setenv("SOURCEAFIS_ENABLED", "true")

    metadata = SourceAfisFingerprintEngine().metadata()

    assert metadata.available is False
    assert metadata.unavailable_reason
    assert "SOURCEAFIS_ENABLED=true" in metadata.unavailable_reason
    assert "SOURCEAFIS_SERVICE_URL" in metadata.unavailable_reason


def test_sourceafis_enabled_false_disables_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_sourceafis_env(monkeypatch)
    monkeypatch.setenv("SOURCEAFIS_ENABLED", "false")
    monkeypatch.setenv("SOURCEAFIS_SERVICE_URL", "http://127.0.0.1:8765")

    metadata = SourceAfisFingerprintEngine().metadata()

    assert metadata.available is False
    assert metadata.unavailable_reason == "SOURCEAFIS_ENABLED=false disables the SourceAFIS provider."
