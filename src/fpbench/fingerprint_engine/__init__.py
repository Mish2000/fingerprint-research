from __future__ import annotations

from src.fpbench.fingerprint_engine.base import FingerprintEngine
from src.fpbench.fingerprint_engine.errors import (
    FingerprintEngineError,
    IdentificationError,
    InvalidTemplateError,
    ProviderUnavailableError,
    TemplateExtractionError,
    UnsupportedEngineOperationError,
    VerificationError,
)
from src.fpbench.fingerprint_engine.registry import (
    configured_provider_id,
    get_default_engine,
    get_engine,
    list_engine_metadata,
    list_engines,
    register_engine,
)
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

__all__ = [
    "CandidateResult",
    "EngineCapabilities",
    "EngineMetadata",
    "FingerprintEngine",
    "FingerprintEngineError",
    "FingerprintImage",
    "FingerprintTemplate",
    "GalleryTemplate",
    "IdentificationError",
    "IdentificationResult",
    "InvalidTemplateError",
    "MatchResult",
    "ProviderUnavailableError",
    "QualityResult",
    "TemplateExtractionError",
    "UnsupportedEngineOperationError",
    "VerificationError",
    "configured_provider_id",
    "get_default_engine",
    "get_engine",
    "list_engine_metadata",
    "list_engines",
    "register_engine",
]
