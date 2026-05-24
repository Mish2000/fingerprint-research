from __future__ import annotations

from collections.abc import Sequence

from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError
from src.fpbench.fingerprint_engine.types import (
    EngineCapabilities,
    EngineMetadata,
    FingerprintImage,
    FingerprintTemplate,
    GalleryTemplate,
    IdentificationResult,
    MatchResult,
    QualityResult,
)


class CotsStubFingerprintEngine:
    """Placeholder for a future commercial AFIS SDK adapter."""

    provider_id = "cots_afis_primary_stub"
    provider_version = "0.1.0"
    template_format = "cots-afis-primary-template"
    template_version = "stub"
    unavailable_reason = (
        "COTS AFIS provider stub is not connected to a real SDK adapter. Install and configure "
        "a licensed real SDK adapter in a later phase before using this provider for operations."
    )

    def metadata(self) -> EngineMetadata:
        return EngineMetadata(
            provider_id=self.provider_id,
            provider_version=self.provider_version,
            name="COTS AFIS provider stub",
            description="COTS AFIS provider stub for the future commercial matcher integration slot.",
            available=False,
            unavailable_reason=self.unavailable_reason,
            template_format=self.template_format,
            template_version=self.template_version,
            sdk_name="commercial_afis_sdk",
            sdk_required=True,
            capabilities=EngineCapabilities(
                supports_template_extraction=True,
                supports_verification=True,
                supports_identification=True,
                supports_quality=True,
                supports_template_storage=True,
                template_formats=[self.template_format],
                score_range=None,
                normalized_score_range=(0.0, 1.0),
            ),
            metadata={"adapter_status": "stub"},
            warnings=["No proprietary SDK is loaded by this stub."],
        )

    def extract_template(self, image: FingerprintImage) -> FingerprintTemplate:
        del image
        raise self._unavailable()

    def verify(
        self,
        probe_template: FingerprintTemplate,
        candidate_template: FingerprintTemplate,
    ) -> MatchResult:
        del probe_template, candidate_template
        raise self._unavailable()

    def identify(
        self,
        probe_template: FingerprintTemplate,
        gallery: Sequence[GalleryTemplate],
        top_k: int = 10,
    ) -> IdentificationResult:
        del probe_template, gallery, top_k
        raise self._unavailable()

    def assess_quality(self, image: FingerprintImage) -> QualityResult | None:
        del image
        raise self._unavailable()

    def _unavailable(self) -> ProviderUnavailableError:
        return ProviderUnavailableError(self.unavailable_reason)
