from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from src.fpbench.fingerprint_engine.types import (
    EngineMetadata,
    FingerprintImage,
    FingerprintTemplate,
    GalleryTemplate,
    IdentificationResult,
    MatchResult,
    QualityResult,
)


@runtime_checkable
class FingerprintEngine(Protocol):
    def metadata(self) -> EngineMetadata:
        """Return public, non-secret provider metadata and capabilities."""

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        """Extract a provider-specific template from a fingerprint image."""

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> MatchResult:
        """Run 1:1 fingerprint verification."""

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: Sequence[GalleryTemplate],
        top_k: int = 10,
    ) -> IdentificationResult:
        """Run 1:N fingerprint identification."""

    def assess_quality(self, image: FingerprintImage) -> QualityResult | None:
        """Return image quality when the provider supports quality assessment."""
