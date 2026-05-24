from __future__ import annotations

import pytest

from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError
from src.fpbench.fingerprint_engine.providers.cots_stub_provider import CotsStubFingerprintEngine
from src.fpbench.fingerprint_engine.providers.null_provider import NullFingerprintEngine
from src.fpbench.fingerprint_engine.types import FingerprintImage, FingerprintTemplate, GalleryTemplate


def test_null_provider_extract_verify_identify_and_quality_are_deterministic() -> None:
    engine = NullFingerprintEngine()
    image = FingerprintImage(image_bytes=b"synthetic fingerprint bytes", image_id="probe")
    alternate_image = FingerprintImage(image_bytes=b"synthetic alternate fingerprint bytes", image_id="candidate")

    probe_a = engine.extract_template(image)
    probe_b = engine.extract_template(image)
    alternate = engine.extract_template(alternate_image)

    assert probe_a == probe_b
    assert probe_a.template_bytes
    assert probe_a.template_bytes != alternate.template_bytes
    assert probe_a.metadata["biometric_template"] is False

    same_match_a = engine.verify(probe_a, probe_b)
    same_match_b = engine.verify(probe_a, probe_b)
    different_match_a = engine.verify(probe_a, alternate)
    different_match_b = engine.verify(probe_a, alternate)

    assert same_match_a == same_match_b
    assert same_match_a.score == pytest.approx(1.0)
    assert same_match_a.decision is True
    assert different_match_a == different_match_b
    assert 0.0 <= different_match_a.normalized_score <= 1.0

    gallery = [
        GalleryTemplate(gallery_id="alternate", subject_id="candidate", template=alternate),
        GalleryTemplate(gallery_id="probe", subject_id="probe", template=probe_b),
    ]
    identification_a = engine.identify(probe_a, gallery, top_k=2)
    identification_b = engine.identify(probe_a, gallery, top_k=2)

    assert identification_a == identification_b
    assert [candidate.rank for candidate in identification_a.candidates] == [1, 2]
    assert identification_a.candidates[0].gallery_id == "probe"
    assert identification_a.top_candidate == identification_a.candidates[0]
    assert identification_a.decision is True

    quality_a = engine.assess_quality(image)
    quality_b = engine.assess_quality(image)

    assert quality_a == quality_b
    assert 0.0 <= quality_a.normalized_score <= 1.0
    assert 0.0 <= quality_a.quality_score <= 100.0


def test_cots_stub_is_unavailable_and_operations_raise_helpful_error() -> None:
    engine = CotsStubFingerprintEngine()
    metadata = engine.metadata()
    placeholder_template = FingerprintTemplate(
        provider_id="cots_afis_primary_stub",
        provider_version="0.1.0",
        template_format="cots-afis-primary-template",
        template_version="stub",
        template_bytes=b"placeholder",
    )

    assert metadata.provider_id == "cots_afis_primary_stub"
    assert metadata.available is False
    assert metadata.unavailable_reason
    assert "real SDK adapter" in metadata.unavailable_reason
    assert metadata.capabilities.supports_verification is True
    assert metadata.capabilities.supports_identification is True
    assert metadata.capabilities.supports_quality is True
    assert metadata.capabilities.supports_template_storage is True

    with pytest.raises(ProviderUnavailableError, match="real SDK adapter"):
        engine.extract_template(FingerprintImage(image_bytes=b"synthetic"))
    with pytest.raises(ProviderUnavailableError, match="real SDK adapter"):
        engine.verify(placeholder_template, placeholder_template)
    with pytest.raises(ProviderUnavailableError, match="real SDK adapter"):
        engine.identify(placeholder_template, [])
    with pytest.raises(ProviderUnavailableError, match="real SDK adapter"):
        engine.assess_quality(FingerprintImage(image_bytes=b"synthetic"))
