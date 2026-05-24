from __future__ import annotations

import pytest

from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError
from src.fpbench.fingerprint_engine.registry import get_default_engine, get_engine, list_engine_metadata, list_engines


def test_registry_default_is_null_without_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINGERPRINT_ENGINE_PROVIDER", raising=False)

    engine = get_default_engine()

    assert engine.metadata().provider_id == "null"
    assert engine.metadata().available is True


def test_registry_respects_environment_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINGERPRINT_ENGINE_PROVIDER", "cots_afis_primary_stub")

    engine = get_default_engine()

    assert engine.metadata().provider_id == "cots_afis_primary_stub"
    assert engine.metadata().available is False


def test_explicit_provider_overrides_environment_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINGERPRINT_ENGINE_PROVIDER", "cots_afis_primary_stub")

    engine = get_engine("null")

    assert engine.metadata().provider_id == "null"


def test_registered_engines_include_null_and_cots_stub() -> None:
    provider_ids = list_engines()
    metadata_by_provider = {metadata.provider_id: metadata for metadata in list_engine_metadata()}

    assert "null" in provider_ids
    assert "cots_afis_primary_stub" in provider_ids
    assert metadata_by_provider["null"].available is True
    assert metadata_by_provider["cots_afis_primary_stub"].available is False


def test_unknown_provider_raises_provider_unavailable() -> None:
    with pytest.raises(ProviderUnavailableError, match="not registered"):
        get_engine("missing_provider")
