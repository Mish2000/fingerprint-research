from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import os
from typing import Any

import httpx

from src.fpbench.fingerprint_engine.errors import (
    IdentificationError,
    ProviderUnavailableError,
    TemplateExtractionError,
    VerificationError,
)
from src.fpbench.fingerprint_engine.types import FingerprintImage, FingerprintTemplate, GalleryTemplate


DEFAULT_SOURCEAFIS_CONNECT_TIMEOUT_SECONDS = 5.0
DEFAULT_SOURCEAFIS_READ_TIMEOUT_SECONDS = 60.0
DEFAULT_SOURCEAFIS_EXTRACT_TIMEOUT_SECONDS = 120.0
DEFAULT_SOURCEAFIS_VERIFY_TIMEOUT_SECONDS = 60.0
DEFAULT_SOURCEAFIS_TIMEOUT_SECONDS = DEFAULT_SOURCEAFIS_READ_TIMEOUT_SECONDS

SOURCEAFIS_CONNECT_TIMEOUT_SECONDS_ENV = "SOURCEAFIS_CONNECT_TIMEOUT_SECONDS"
SOURCEAFIS_READ_TIMEOUT_SECONDS_ENV = "SOURCEAFIS_READ_TIMEOUT_SECONDS"
SOURCEAFIS_EXTRACT_TIMEOUT_SECONDS_ENV = "SOURCEAFIS_EXTRACT_TIMEOUT_SECONDS"
SOURCEAFIS_VERIFY_TIMEOUT_SECONDS_ENV = "SOURCEAFIS_VERIFY_TIMEOUT_SECONDS"


@dataclass(frozen=True)
class SourceAfisTimeoutSettings:
    connect_timeout_seconds: float
    read_timeout_seconds: float
    extract_timeout_seconds: float
    verify_timeout_seconds: float

    def as_dict(self) -> dict[str, float]:
        return {
            "connect_timeout_seconds": float(self.connect_timeout_seconds),
            "read_timeout_seconds": float(self.read_timeout_seconds),
            "extract_timeout_seconds": float(self.extract_timeout_seconds),
            "verify_timeout_seconds": float(self.verify_timeout_seconds),
        }


@dataclass(frozen=True)
class SourceAfisHealth:
    available: bool
    version: str | None = None
    service_url: str | None = None
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAfisClient:
    """HTTP sidecar adapter for the SourceAFIS runtime.

    The Python backend intentionally does not link directly against SourceAFIS.
    The sidecar exposes JSON endpoints that accept base64-encoded image and
    template bytes and return SourceAFIS raw matching scores.
    """

    def __init__(
        self,
        service_url: str,
        *,
        timeout_seconds: float | None = None,
        connect_timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        extract_timeout_seconds: float | None = None,
        verify_timeout_seconds: float | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        normalized_url = str(service_url or "").strip().rstrip("/")
        if not normalized_url:
            raise ProviderUnavailableError("SOURCEAFIS_SERVICE_URL must be set to a non-empty HTTP sidecar URL.")
        self.service_url = normalized_url
        self.timeout_settings = sourceafis_timeout_settings_from_env(
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            extract_timeout_seconds=extract_timeout_seconds,
            verify_timeout_seconds=verify_timeout_seconds,
        )
        self.timeout_seconds = float(self.timeout_settings.read_timeout_seconds)
        self._client = httpx.Client(timeout=self._httpx_timeout(self.timeout_seconds), transport=transport)

    def health(self) -> SourceAfisHealth:
        payload = self._request_json(
            "GET",
            "/health",
            error_cls=ProviderUnavailableError,
            operation="SourceAFIS health check",
        )
        available = _optional_bool(payload.get("available"))
        status = _optional_str(payload.get("status"))
        if available is None:
            available = status is None or status.lower() in {"ok", "ready", "healthy"}

        version = _optional_str(
            payload.get("sourceafis_version")
            or payload.get("engine_version")
            or payload.get("version")
            or payload.get("provider_version")
        )
        reason = _optional_str(
            payload.get("unavailable_reason")
            or payload.get("reason")
            or payload.get("message")
        )
        if not available and reason is None:
            reason = f"SourceAFIS sidecar at {self.service_url} reported status={status!r}."

        return SourceAfisHealth(
            available=bool(available),
            version=version,
            service_url=self.service_url,
            unavailable_reason=reason,
            metadata=_sanitize_metadata(payload.get("metadata", {})),
        )

    def extract_template(self, image: FingerprintImage) -> dict[str, Any]:
        payload = self._request_json(
            "POST",
            "/extract-template",
            error_cls=TemplateExtractionError,
            operation="SourceAFIS template extraction",
            json_payload=_image_payload(image),
            read_timeout_seconds=self.timeout_settings.extract_timeout_seconds,
        )
        encoded_template = _required_str(
            payload,
            ("template_base64", "template_bytes_b64"),
            TemplateExtractionError,
        )
        return {
            "provider_version": _optional_str(
                payload.get("provider_version")
                or payload.get("engine_version")
                or payload.get("sourceafis_version")
                or payload.get("template_version")
            ),
            "template_format": _optional_str(payload.get("template_format")) or "sourceafis",
            "template_version": _optional_str(
                payload.get("template_version")
                or payload.get("engine_version")
                or payload.get("sourceafis_version")
                or payload.get("provider_version")
            ),
            "template_bytes": _decode_base64(encoded_template, TemplateExtractionError, "template_base64"),
            "quality_score": _optional_float(payload.get("quality_score")),
            "metadata": _sanitize_metadata(payload.get("metadata", {})),
        }

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> dict[str, Any]:
        payload = self._request_json(
            "POST",
            "/verify",
            error_cls=VerificationError,
            operation="SourceAFIS verification",
            json_payload={
                "probe_template_base64": _template_base64(probe_template),
                "candidate_template_base64": _template_base64(candidate_template),
            },
            read_timeout_seconds=self.timeout_settings.verify_timeout_seconds,
        )
        return {
            "provider_version": _optional_str(
                payload.get("provider_version") or payload.get("engine_version") or payload.get("sourceafis_version")
            ),
            "score": _required_float(payload, VerificationError),
            "normalized_score": _optional_float(payload.get("normalized_score")),
            "threshold": _optional_float(payload.get("threshold")),
            "decision": _optional_bool(payload.get("decision")),
            "latency_ms": _optional_float(payload.get("latency_ms")),
            "metadata": _sanitize_metadata(payload.get("metadata", {})),
        }

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: Sequence[GalleryTemplate],
        top_k: int = 10,
    ) -> dict[str, Any]:
        payload = self._request_json(
            "POST",
            "/identify",
            error_cls=IdentificationError,
            operation="SourceAFIS identification",
            json_payload={
                "probe_template_base64": _template_base64(probe_template),
                "gallery": [
                    {
                        "candidate_id": item.gallery_id,
                        "template_base64": _template_base64(item.template),
                        "metadata": _gallery_metadata(item),
                    }
                    for item in gallery
                ],
                "top_k": max(int(top_k), 0),
            },
        )
        candidates = payload.get("candidates", [])
        if not isinstance(candidates, list):
            raise IdentificationError("SourceAFIS identification response field 'candidates' must be a list.")

        return {
            "provider_version": _optional_str(
                payload.get("provider_version") or payload.get("engine_version") or payload.get("sourceafis_version")
            ),
            "candidates": [_candidate_payload(candidate) for candidate in candidates],
            "threshold": _optional_float(payload.get("threshold")),
            "decision": _optional_bool(payload.get("decision")),
            "latency_ms": _optional_float(payload.get("latency_ms")),
            "metadata": _sanitize_metadata(payload.get("metadata", {})),
        }

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        error_cls: type[Exception],
        operation: str,
        json_payload: Mapping[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self.service_url}{path}"
        try:
            response = self._client.request(
                method,
                url,
                json=json_payload,
                timeout=self._httpx_timeout(read_timeout_seconds or self.timeout_settings.read_timeout_seconds),
            )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(f"{operation} timed out contacting SourceAFIS sidecar at {self.service_url}.") from exc
        except httpx.RequestError as exc:
            raise ProviderUnavailableError(f"{operation} could not reach SourceAFIS sidecar at {self.service_url}: {exc}") from exc

        if response.status_code >= 500:
            raise ProviderUnavailableError(
                f"{operation} failed because SourceAFIS sidecar at {self.service_url} returned HTTP "
                f"{response.status_code}: {_response_detail(response)}"
            )
        if response.status_code >= 400:
            raise error_cls(
                f"{operation} was rejected by SourceAFIS sidecar at {self.service_url} with HTTP "
                f"{response.status_code}: {_response_detail(response)}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise error_cls(f"{operation} returned non-JSON response from SourceAFIS sidecar.") from exc
        if not isinstance(payload, dict):
            raise error_cls(f"{operation} returned JSON {type(payload).__name__}; expected an object.")
        return payload

    def _httpx_timeout(self, read_timeout_seconds: float) -> httpx.Timeout:
        return httpx.Timeout(
            connect=float(self.timeout_settings.connect_timeout_seconds),
            read=float(read_timeout_seconds),
            write=float(read_timeout_seconds),
            pool=float(self.timeout_settings.connect_timeout_seconds),
        )


def sourceafis_timeout_settings_from_env(
    *,
    timeout_seconds: float | None = None,
    connect_timeout_seconds: float | None = None,
    read_timeout_seconds: float | None = None,
    extract_timeout_seconds: float | None = None,
    verify_timeout_seconds: float | None = None,
) -> SourceAfisTimeoutSettings:
    fallback = _positive_float(timeout_seconds, "timeout_seconds") if timeout_seconds is not None else None
    connect = _timeout_value(
        connect_timeout_seconds,
        fallback,
        SOURCEAFIS_CONNECT_TIMEOUT_SECONDS_ENV,
        DEFAULT_SOURCEAFIS_CONNECT_TIMEOUT_SECONDS,
    )
    read = _timeout_value(
        read_timeout_seconds,
        fallback,
        SOURCEAFIS_READ_TIMEOUT_SECONDS_ENV,
        DEFAULT_SOURCEAFIS_READ_TIMEOUT_SECONDS,
    )
    extract = _timeout_value(
        extract_timeout_seconds,
        fallback,
        SOURCEAFIS_EXTRACT_TIMEOUT_SECONDS_ENV,
        DEFAULT_SOURCEAFIS_EXTRACT_TIMEOUT_SECONDS,
    )
    verify = _timeout_value(
        verify_timeout_seconds,
        fallback,
        SOURCEAFIS_VERIFY_TIMEOUT_SECONDS_ENV,
        DEFAULT_SOURCEAFIS_VERIFY_TIMEOUT_SECONDS,
    )
    return SourceAfisTimeoutSettings(
        connect_timeout_seconds=connect,
        read_timeout_seconds=read,
        extract_timeout_seconds=extract,
        verify_timeout_seconds=verify,
    )


def _timeout_value(
    explicit: float | None,
    fallback: float | None,
    env_name: str,
    default: float,
) -> float:
    if explicit is not None:
        return _positive_float(explicit, env_name)
    if fallback is not None:
        return fallback
    env_value = os.getenv(env_name)
    if env_value not in (None, ""):
        return _positive_float(env_value, env_name)
    return float(default)


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderUnavailableError(f"{name} must be a positive number of seconds.") from exc
    if not parsed > 0:
        raise ProviderUnavailableError(f"{name} must be a positive number of seconds.")
    return float(parsed)


def _image_payload(image: FingerprintImage) -> dict[str, Any]:
    if image.image_bytes is None:
        raise TemplateExtractionError("SourceAFIS sidecar extraction requires in-memory image bytes.")
    metadata = _sanitize_metadata(image.metadata)
    for key, value in {
        "sha256": image.sha256,
        "image_id": image.image_id,
        "mime_type": image.mime_type,
        "width": image.width,
        "height": image.height,
        "dpi": image.dpi,
        "capture_type": image.capture_type,
    }.items():
        if value is not None and key not in metadata:
            metadata[key] = value
    return {
        "image_base64": base64.b64encode(image.image_bytes).decode("ascii"),
        "image_format": _image_format(image),
        "metadata": metadata,
    }


def _template_base64(template: FingerprintTemplate) -> str:
    return base64.b64encode(template.template_bytes).decode("ascii")


def _gallery_metadata(item: GalleryTemplate) -> dict[str, Any]:
    metadata = _sanitize_metadata(item.metadata)
    if item.subject_id is not None and "subject_id" not in metadata:
        metadata["subject_id"] = item.subject_id
    return metadata


def _candidate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise IdentificationError("SourceAFIS identification candidate entries must be JSON objects.")
    gallery_id = _required_str(payload, ("candidate_id", "gallery_id"), IdentificationError)
    return {
        "gallery_id": gallery_id,
        "subject_id": _optional_str(payload.get("subject_id"))
        or _optional_str(_sanitize_metadata(payload.get("metadata", {})).get("subject_id")),
        "score": _required_float(payload, IdentificationError),
        "normalized_score": _optional_float(payload.get("normalized_score")),
        "threshold": _optional_float(payload.get("threshold")),
        "decision": _optional_bool(payload.get("decision")),
        "latency_ms": _optional_float(payload.get("latency_ms")),
        "metadata": _sanitize_metadata(payload.get("metadata", {})),
    }


def _decode_base64(value: str, error_cls: type[Exception], field_name: str) -> bytes:
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise error_cls(f"SourceAFIS response field {field_name!r} is not valid base64.") from exc


def _required_str(payload: Mapping[str, Any], field_names: str | tuple[str, ...], error_cls: type[Exception]) -> str:
    names = (field_names,) if isinstance(field_names, str) else field_names
    for field_name in names:
        value = payload.get(field_name)
        text = _optional_str(value)
        if text is not None:
            return text
    expected = names[0] if len(names) == 1 else " or ".join(repr(name) for name in names)
    raise error_cls(f"SourceAFIS response missing required field {expected}.")


def _required_float(payload: Mapping[str, Any], error_cls: type[Exception]) -> float:
    raw_value = payload.get("score", payload.get("raw_score"))
    value = _optional_float(raw_value)
    if value is None:
        raise error_cls("SourceAFIS response missing required numeric field 'score'.")
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


def _image_format(image: FingerprintImage) -> str:
    mime_type = _optional_str(image.mime_type)
    if mime_type:
        normalized = mime_type.lower().split(";", 1)[0].strip()
        if normalized in {"image/png", "image/x-png"}:
            return "png"
        if normalized in {"image/jpeg", "image/jpg"}:
            return "jpg"
        if normalized in {"image/bmp", "image/x-ms-bmp"}:
            return "bmp"
        if normalized in {"image/tiff", "image/tif"}:
            return "tif"

    path = _optional_str(image.path)
    if path:
        suffix = path.rsplit(".", 1)[-1].lower() if "." in path else ""
        if suffix in {"png", "jpg", "jpeg", "bmp", "tif", "tiff"}:
            return "jpg" if suffix == "jpeg" else "tif" if suffix == "tiff" else suffix
    return "unknown"


def _sanitize_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    blocked_keys = {
        "template_bytes",
        "template_bytes_b64",
        "probe_template_bytes",
        "probe_template_bytes_b64",
        "candidate_template_bytes",
        "candidate_template_bytes_b64",
        "image_base64",
        "image_bytes_b64",
        "template_base64",
        "probe_template_base64",
        "candidate_template_base64",
    }
    return {
        str(key): item
        for key, item in value.items()
        if str(key).lower() not in blocked_keys
    }


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500]
    if isinstance(payload, Mapping):
        for key in ("detail", "error", "message"):
            if key in payload:
                return str(payload[key])
    return str(payload)[:500]
