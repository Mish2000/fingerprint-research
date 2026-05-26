from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.fpbench.fingerprint_engine.errors import (
    FingerprintEngineError,
    IdentificationError,
    InvalidTemplateError,
    ProviderUnavailableError,
    TemplateExtractionError,
    VerificationError,
)
from src.fpbench.fingerprint_engine.providers.sourceafis_client import SourceAfisClient
from src.fpbench.fingerprint_engine.types import (
    CandidateResult,
    EngineCapabilities,
    EngineMetadata,
    FingerprintImage,
    FingerprintTemplate,
    GalleryTemplate,
    IdentificationResult,
    MatchResult,
    QualityResult,
)


SOURCEAFIS_ENABLED_ENV = "SOURCEAFIS_ENABLED"
SOURCEAFIS_SERVICE_URL_ENV = "SOURCEAFIS_SERVICE_URL"
SOURCEAFIS_CLI_PATH_ENV = "SOURCEAFIS_CLI_PATH"


class SourceAfisClientProtocol(Protocol):
    service_url: str

    def health(self) -> Any:
        ...

    def extract_template(self, image: FingerprintImage) -> dict[str, Any]:
        ...

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> dict[str, Any]:
        ...

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: Sequence[GalleryTemplate],
        top_k: int = 10,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class _RuntimeStatus:
    available: bool
    client: SourceAfisClientProtocol | None = None
    provider_version: str = "sidecar-unavailable"
    template_version: str = "unknown"
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAfisFingerprintEngine:
    """Optional SourceAFIS provider behind the fingerprint engine abstraction.

    To connect a real runtime, run a SourceAFIS sidecar that implements the
    contract documented in sourceafis_client.py, then set:

    SOURCEAFIS_ENABLED=true
    SOURCEAFIS_SERVICE_URL=http://127.0.0.1:8765
    """

    provider_id = "sourceafis_open"
    display_name = "SourceAFIS Open Matcher"
    template_format = "sourceafis"
    provider_version = "sidecar-unavailable"
    template_version = "unknown"

    _raw_score_warning = (
        "SourceAFIS raw scores are not normalized by this provider; calibrate thresholds on a "
        "validation split before comparing them with other matcher families."
    )

    def __init__(self, client: SourceAfisClientProtocol | None = None) -> None:
        self._client = client

    def metadata(self) -> EngineMetadata:
        status = self._runtime_status()
        return EngineMetadata(
            provider_id=self.provider_id,
            provider_version=status.provider_version,
            name=self.display_name,
            description=(
                "Open-source AFIS provider used as a real non-proprietary baseline behind "
                "the fingerprint engine abstraction. It communicates with a SourceAFIS "
                "HTTP sidecar instead of linking the Python API directly to the matcher runtime."
            ),
            available=status.available,
            unavailable_reason=status.unavailable_reason,
            template_format=self.template_format,
            template_version=status.template_version,
            sdk_name="SourceAFIS",
            sdk_required=False,
            capabilities=EngineCapabilities(
                supports_template_extraction=True,
                supports_verification=True,
                supports_identification=True,
                supports_quality=False,
                supports_template_storage=True,
                template_formats=[self.template_format],
                score_range=None,
                normalized_score_range=None,
                metadata={
                    "score_semantics": "sourceafis_raw_similarity_score",
                    "score_normalization": "not_calibrated",
                },
                warnings=[self._raw_score_warning],
            ),
            metadata={
                "runtime": "sourceafis_http_sidecar",
                "service_url_configured": bool(status.metadata.get("service_url_configured")),
                "env": {
                    "enabled": SOURCEAFIS_ENABLED_ENV,
                    "service_url": SOURCEAFIS_SERVICE_URL_ENV,
                    "cli_path": SOURCEAFIS_CLI_PATH_ENV,
                },
                "sidecar_contract": {
                    "health": "GET /health",
                    "extract_template": "POST /extract-template",
                    "verify": "POST /verify",
                    "identify": "POST /identify",
                },
                **status.metadata,
            },
            warnings=[] if status.available else ["SourceAFIS provider is registered but no runtime is available."],
        )

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        status = self._require_runtime()
        try:
            payload = status.client.extract_template(image) if status.client is not None else {}
        except FingerprintEngineError:
            raise
        except Exception as exc:
            raise TemplateExtractionError(f"SourceAFIS template extraction failed: {exc}") from exc

        template_bytes = payload.get("template_bytes")
        if not isinstance(template_bytes, bytes) or not template_bytes:
            raise TemplateExtractionError("SourceAFIS template extraction returned no template bytes.")

        provider_version = _optional_str(payload.get("provider_version")) or status.provider_version
        template_version = _optional_str(payload.get("template_version")) or provider_version or status.template_version
        return FingerprintTemplate(
            provider_id=self.provider_id,
            provider_version=provider_version,
            template_format=_optional_str(payload.get("template_format")) or self.template_format,
            template_version=template_version,
            template_bytes=template_bytes,
            image_id=image.image_id,
            quality_score=_optional_float(payload.get("quality_score")),
            metadata={
                "runtime": "sourceafis_http_sidecar",
                "biometric_template": True,
                **_metadata_dict(payload.get("metadata")),
            },
        )

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> MatchResult:
        self._validate_template(probe_template)
        self._validate_template(candidate_template)
        status = self._require_runtime()
        try:
            payload = status.client.verify(probe_template, candidate_template) if status.client is not None else {}
        except FingerprintEngineError:
            raise
        except Exception as exc:
            raise VerificationError(f"SourceAFIS verification failed: {exc}") from exc

        score = _required_score(payload, VerificationError)
        normalized_score = _optional_float(payload.get("normalized_score"))
        warnings = [] if normalized_score is not None else [self._raw_score_warning]
        return MatchResult(
            provider_id=self.provider_id,
            provider_version=_optional_str(payload.get("provider_version")) or status.provider_version,
            score=score,
            normalized_score=normalized_score,
            threshold=_optional_float(payload.get("threshold")),
            decision=_optional_bool(payload.get("decision")),
            latency_ms=_optional_float(payload.get("latency_ms")),
            metadata={
                "score_semantics": "sourceafis_raw_similarity_score",
                "score_normalization": "not_calibrated" if normalized_score is None else "provided_by_sidecar",
                **_metadata_dict(payload.get("metadata")),
            },
            warnings=warnings,
        )

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: Sequence[GalleryTemplate],
        top_k: int = 10,
    ) -> IdentificationResult:
        self._validate_template(probe_template)
        status = self._require_runtime()
        limit = max(int(top_k), 0)
        gallery_items = list(gallery)
        for item in gallery_items:
            self._validate_template(item.template)

        if not gallery_items or limit == 0:
            return IdentificationResult(
                provider_id=self.provider_id,
                provider_version=status.provider_version,
                candidates=[],
                top_candidate=None,
                decision=False,
                latency_ms=0.0,
                metadata={"gallery_size": len(gallery_items), "top_k": limit},
                warnings=[self._raw_score_warning],
            )

        try:
            payload = status.client.identify(probe_template, gallery_items, top_k=limit) if status.client is not None else {}
        except FingerprintEngineError:
            raise
        except Exception as exc:
            raise IdentificationError(f"SourceAFIS identification failed: {exc}") from exc

        gallery_by_id = {item.gallery_id: item for item in gallery_items}
        scored = [
            self._candidate_from_payload(candidate_payload, gallery_by_id, status, payload)
            for candidate_payload in payload.get("candidates", [])
        ]
        ranked = sorted(scored, key=lambda candidate: (-candidate.score, candidate.gallery_id))
        candidates = [
            CandidateResult(
                gallery_id=candidate.gallery_id,
                rank=index + 1,
                score=candidate.score,
                normalized_score=candidate.normalized_score,
                threshold=candidate.threshold,
                decision=candidate.decision,
                subject_id=candidate.subject_id,
                provider_id=self.provider_id,
                provider_version=candidate.provider_version or status.provider_version,
                latency_ms=candidate.latency_ms,
                metadata=candidate.metadata,
                warnings=candidate.warnings,
                errors=candidate.errors,
            )
            for index, candidate in enumerate(ranked[:limit])
        ]
        top_candidate = candidates[0] if candidates else None
        normalized_available = all(candidate.normalized_score is not None for candidate in candidates)
        warnings = [] if normalized_available else [self._raw_score_warning]
        return IdentificationResult(
            provider_id=self.provider_id,
            provider_version=_optional_str(payload.get("provider_version")) or status.provider_version,
            candidates=candidates,
            top_candidate=top_candidate,
            threshold=_optional_float(payload.get("threshold")),
            decision=_optional_bool(payload.get("decision")) if top_candidate is not None else False,
            latency_ms=_optional_float(payload.get("latency_ms")),
            metadata={
                "gallery_size": len(gallery_items),
                "top_k": limit,
                "score_semantics": "sourceafis_raw_similarity_score",
                "score_normalization": "not_calibrated" if not normalized_available else "provided_by_sidecar",
                **_metadata_dict(payload.get("metadata")),
            },
            warnings=warnings,
        )

    def assess_quality(self, image: FingerprintImage) -> QualityResult | None:
        del image
        return None

    def _candidate_from_payload(
        self,
        payload: dict[str, Any],
        gallery_by_id: dict[str, GalleryTemplate],
        status: _RuntimeStatus,
        result_payload: dict[str, Any],
    ) -> CandidateResult:
        gallery_id = _optional_str(payload.get("gallery_id"))
        if gallery_id is None:
            raise IdentificationError("SourceAFIS identification candidate is missing gallery_id.")
        score = _required_score(payload, IdentificationError)
        gallery_item = gallery_by_id.get(gallery_id)
        normalized_score = _optional_float(payload.get("normalized_score"))
        return CandidateResult(
            gallery_id=gallery_id,
            rank=0,
            score=score,
            normalized_score=normalized_score,
            threshold=_optional_float(payload.get("threshold") or result_payload.get("threshold")),
            decision=_optional_bool(payload.get("decision")),
            subject_id=_optional_str(payload.get("subject_id")) or (gallery_item.subject_id if gallery_item else None),
            provider_id=self.provider_id,
            provider_version=_optional_str(payload.get("provider_version")) or status.provider_version,
            latency_ms=_optional_float(payload.get("latency_ms")),
            metadata={
                **(dict(gallery_item.metadata) if gallery_item else {}),
                **_metadata_dict(payload.get("metadata")),
            },
            warnings=[] if normalized_score is not None else [self._raw_score_warning],
        )

    def _validate_template(self, template: FingerprintTemplate) -> None:
        if template.provider_id != self.provider_id:
            raise InvalidTemplateError(
                f"Template provider {template.provider_id!r} is not compatible with provider {self.provider_id!r}."
            )
        if template.template_format != self.template_format:
            raise InvalidTemplateError(
                f"Template format {template.template_format!r} is not compatible with {self.template_format!r}."
            )
        if not template.template_bytes:
            raise InvalidTemplateError("SourceAFIS template bytes must be non-empty.")

    def _require_runtime(self) -> _RuntimeStatus:
        status = self._runtime_status()
        if not status.available or status.client is None:
            raise ProviderUnavailableError(status.unavailable_reason or "SourceAFIS runtime is not available.")
        return status

    def _runtime_status(self) -> _RuntimeStatus:
        client = self._client
        service_url_configured = True
        if client is None:
            client, unavailable_reason, service_url_configured = self._client_from_environment()
            if client is None:
                return _RuntimeStatus(
                    available=False,
                    unavailable_reason=unavailable_reason,
                    metadata={"service_url_configured": service_url_configured},
                )
            self._client = client

        try:
            health = client.health()
        except ProviderUnavailableError as exc:
            return _RuntimeStatus(
                available=False,
                client=client,
                unavailable_reason=str(exc),
                metadata={"service_url_configured": service_url_configured},
            )
        except Exception as exc:
            return _RuntimeStatus(
                available=False,
                client=client,
                unavailable_reason=f"SourceAFIS health check failed: {type(exc).__name__}: {exc}",
                metadata={"service_url_configured": service_url_configured},
            )

        available = bool(getattr(health, "available", False))
        reason = _optional_str(getattr(health, "unavailable_reason", None))
        version = _optional_str(getattr(health, "version", None)) or "sourceafis-sidecar"
        health_metadata = _metadata_dict(getattr(health, "metadata", {}))
        health_metadata["service_url_configured"] = service_url_configured
        health_metadata["service_url"] = _optional_str(getattr(client, "service_url", None)) or ""
        timeout_settings = getattr(client, "timeout_settings", None)
        if timeout_settings is not None and hasattr(timeout_settings, "as_dict"):
            health_metadata["timeout_settings"] = timeout_settings.as_dict()
        return _RuntimeStatus(
            available=available,
            client=client if available else None,
            provider_version=version,
            template_version=version,
            unavailable_reason=None if available else reason or "SourceAFIS sidecar reported unavailable.",
            metadata=health_metadata,
        )

    def _client_from_environment(self) -> tuple[SourceAfisClientProtocol | None, str | None, bool]:
        enabled = _optional_bool(os.getenv(SOURCEAFIS_ENABLED_ENV))
        service_url = str(os.getenv(SOURCEAFIS_SERVICE_URL_ENV) or "").strip()
        cli_path = str(os.getenv(SOURCEAFIS_CLI_PATH_ENV) or "").strip()

        if enabled is False:
            return None, f"{SOURCEAFIS_ENABLED_ENV}=false disables the SourceAFIS provider.", bool(service_url)

        if not service_url:
            if cli_path:
                return (
                    None,
                    f"{SOURCEAFIS_CLI_PATH_ENV} is set, but this phase expects an HTTP SourceAFIS sidecar. "
                    f"Start the sidecar and set {SOURCEAFIS_SERVICE_URL_ENV}=http://127.0.0.1:8765.",
                    False,
                )
            return (
                None,
                f"SourceAFIS runtime is not configured. Set {SOURCEAFIS_ENABLED_ENV}=true and "
                f"{SOURCEAFIS_SERVICE_URL_ENV}=http://127.0.0.1:8765 after starting a SourceAFIS sidecar.",
                False,
            )

        return SourceAfisClient(service_url), None, True


def _required_score(payload: dict[str, Any], error_cls: type[Exception]) -> float:
    raw_value = payload.get("score", payload.get("raw_score"))
    value = _optional_float(raw_value)
    if value is None:
        raise error_cls("SourceAFIS response missing required numeric score.")
    return value


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _metadata_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    blocked_keys = {
        "template_bytes",
        "template_bytes_b64",
        "probe_template_bytes",
        "probe_template_bytes_b64",
        "candidate_template_bytes",
        "candidate_template_bytes_b64",
    }
    return {
        str(key): item
        for key, item in value.items()
        if str(key).lower() not in blocked_keys
    }
