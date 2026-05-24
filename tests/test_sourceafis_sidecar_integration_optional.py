from __future__ import annotations

import binascii
import math
import os
import struct
import zlib
from collections.abc import Iterator

import httpx
import pytest

from src.fpbench.fingerprint_engine.providers.sourceafis_provider import SourceAfisFingerprintEngine
from src.fpbench.fingerprint_engine.types import FingerprintImage, GalleryTemplate


def _integration_enabled() -> bool:
    return str(os.getenv("SOURCEAFIS_INTEGRATION_TESTS") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@pytest.fixture()
def sourceafis_service_url() -> Iterator[str]:
    if not _integration_enabled():
        pytest.skip("Set SOURCEAFIS_INTEGRATION_TESTS=true to run SourceAFIS sidecar integration tests.")

    service_url = str(os.getenv("SOURCEAFIS_SERVICE_URL") or "http://127.0.0.1:8765").rstrip("/")
    try:
        response = httpx.get(f"{service_url}/health", timeout=2.0)
    except httpx.RequestError as exc:
        pytest.skip(f"SourceAFIS sidecar is not reachable at {service_url}: {exc}")
    if response.status_code != 200:
        pytest.skip(f"SourceAFIS sidecar at {service_url} returned HTTP {response.status_code}.")
    yield service_url


def test_sourceafis_sidecar_health_and_python_engine_roundtrip(
    sourceafis_service_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOURCEAFIS_ENABLED", "true")
    monkeypatch.setenv("SOURCEAFIS_SERVICE_URL", sourceafis_service_url)

    health = httpx.get(f"{sourceafis_service_url}/health", timeout=2.0)
    assert health.status_code == 200
    health_payload = health.json()
    assert health_payload["status"] == "ok"
    assert health_payload["provider_id"] == "sourceafis_open"
    assert health_payload["template_format"] == "sourceafis"

    engine = SourceAfisFingerprintEngine()
    metadata = engine.metadata()
    assert metadata.available is True
    assert metadata.provider_id == "sourceafis_open"

    probe_image = FingerprintImage(
        image_bytes=_synthetic_fingerprint_png(0),
        image_id="synthetic-probe",
        mime_type="image/png",
        metadata={"dpi": 500, "fixture": "synthetic_non_sensitive"},
    )
    candidate_image = FingerprintImage(
        image_bytes=_synthetic_fingerprint_png(0),
        image_id="synthetic-candidate",
        mime_type="image/png",
        metadata={"dpi": 500, "fixture": "synthetic_non_sensitive"},
    )

    probe = engine.extract_template(probe_image)
    candidate = engine.extract_template(candidate_image)
    assert probe.template_bytes
    assert candidate.template_bytes
    assert probe.metadata["biometric_template"] is True

    match = engine.verify(probe, candidate)
    assert math.isfinite(match.score)
    assert match.normalized_score is None
    assert match.decision is None
    assert match.warnings

    identification = engine.identify(
        probe,
        [GalleryTemplate(gallery_id="subject-1", subject_id="subject-1", template=candidate)],
        top_k=1,
    )
    assert len(identification.candidates) == 1
    assert identification.top_candidate is not None
    assert identification.top_candidate.gallery_id == "subject-1"
    assert math.isfinite(identification.top_candidate.score)


def _synthetic_fingerprint_png(variant: int) -> bytes:
    width = 360
    height = 460
    period = 10.5
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            dx = (x - width / 2) / (width * 0.46)
            dy = (y - height / 2) / (height * 0.43)
            if dx * dx + dy * dy > 1.0:
                row.append(255)
                continue
            warped_y = y + 18.0 * math.sin(x * 0.035 + variant * 0.35) + 5.0 * math.sin(y * 0.02)
            ridge_pos = (warped_y - 52.0) % period
            distance = min(ridge_pos, period - ridge_pos)
            row.append(22 if distance < 1.7 else 245)
        rows.append(bytes([0]) + bytes(row))
    return _png_bytes(width, height, b"".join(rows))


def _png_bytes(width: int, height: int, filtered_rows: bytes) -> bytes:
    def chunk(name: bytes, data: bytes) -> bytes:
        checksum = binascii.crc32(name)
        checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", checksum)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(filtered_rows)) + chunk(b"IEND", b"")
