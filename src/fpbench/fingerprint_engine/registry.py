from __future__ import annotations

import os
from collections.abc import Callable

from src.fpbench.fingerprint_engine.base import FingerprintEngine
from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError
from src.fpbench.fingerprint_engine.providers import (
    CotsStubFingerprintEngine,
    NullFingerprintEngine,
    SourceAfisFingerprintEngine,
)
from src.fpbench.fingerprint_engine.types import EngineMetadata

EngineFactory = Callable[[], FingerprintEngine]

_DEFAULT_PROVIDER_ID = "null"
_PROVIDER_ENV_VAR = "FINGERPRINT_ENGINE_PROVIDER"
_REGISTRY: dict[str, EngineFactory] = {}


def register_engine(provider_id: str, factory: EngineFactory) -> None:
    normalized_provider_id = _normalize_provider_id(provider_id)
    if not callable(factory):
        raise TypeError(f"Fingerprint engine factory for {normalized_provider_id!r} must be callable.")
    _REGISTRY[normalized_provider_id] = factory


def get_engine(provider_id: str | None = None) -> FingerprintEngine:
    selected_provider_id = _select_provider_id(provider_id)
    factory = _REGISTRY.get(selected_provider_id)
    if factory is None:
        raise ProviderUnavailableError(
            f"Fingerprint engine provider {selected_provider_id!r} is not registered. "
            f"Registered providers: {list_engines()}."
        )
    return factory()


def get_default_engine() -> FingerprintEngine:
    return get_engine()


def list_engines() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def list_engine_metadata() -> list[EngineMetadata]:
    return [get_engine(provider_id).metadata() for provider_id in list_engines()]


def configured_provider_id() -> str:
    return _select_provider_id(None)


def _select_provider_id(provider_id: str | None) -> str:
    explicit_provider_id = _normalize_optional_provider_id(provider_id)
    if explicit_provider_id:
        return explicit_provider_id

    configured = _normalize_optional_provider_id(os.getenv(_PROVIDER_ENV_VAR))
    if configured:
        return configured

    return _DEFAULT_PROVIDER_ID


def _normalize_provider_id(provider_id: str) -> str:
    normalized = str(provider_id or "").strip().lower()
    if not normalized:
        raise ValueError("Fingerprint engine provider_id must be non-empty.")
    return normalized


def _normalize_optional_provider_id(provider_id: str | None) -> str | None:
    if provider_id is None:
        return None
    normalized = str(provider_id).strip().lower()
    return normalized or None


register_engine("null", NullFingerprintEngine)
register_engine("cots_afis_primary_stub", CotsStubFingerprintEngine)
register_engine("sourceafis_open", SourceAfisFingerprintEngine)
