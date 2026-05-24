from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FingerprintImage:
    """Image input for a fingerprint engine.

    Tests and SDK adapters can pass raw bytes, a precomputed hash, or an opaque
    path/reference. Engine implementations decide which inputs they support.
    """

    image_bytes: bytes | None = None
    sha256: str | None = None
    path: str | None = None
    image_id: str | None = None
    mime_type: str | None = None
    width: int | None = None
    height: int | None = None
    dpi: int | None = None
    capture_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FingerprintTemplate:
    provider_id: str
    provider_version: str
    template_format: str
    template_version: str
    template_bytes: bytes
    image_id: str | None = None
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GalleryTemplate:
    gallery_id: str
    template: FingerprintTemplate
    subject_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchResult:
    provider_id: str
    provider_version: str
    score: float
    normalized_score: float | None = None
    threshold: float | None = None
    decision: bool | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CandidateResult:
    gallery_id: str
    rank: int
    score: float
    normalized_score: float | None = None
    threshold: float | None = None
    decision: bool | None = None
    subject_id: str | None = None
    provider_id: str | None = None
    provider_version: str | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IdentificationResult:
    provider_id: str
    provider_version: str
    candidates: list[CandidateResult] = field(default_factory=list)
    top_candidate: CandidateResult | None = None
    threshold: float | None = None
    decision: bool | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QualityResult:
    provider_id: str
    provider_version: str
    quality_score: float
    normalized_score: float | None = None
    threshold: float | None = None
    decision: bool | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EngineCapabilities:
    supports_template_extraction: bool = True
    supports_verification: bool = False
    supports_identification: bool = False
    supports_quality: bool = False
    supports_template_storage: bool = False
    template_formats: list[str] = field(default_factory=list)
    score_range: tuple[float, float] | None = None
    normalized_score_range: tuple[float, float] | None = (0.0, 1.0)
    max_gallery_size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EngineMetadata:
    provider_id: str
    provider_version: str
    name: str
    description: str
    available: bool
    capabilities: EngineCapabilities
    unavailable_reason: str | None = None
    template_format: str | None = None
    template_version: str | None = None
    sdk_name: str | None = None
    sdk_required: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
