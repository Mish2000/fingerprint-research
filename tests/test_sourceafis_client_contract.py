from __future__ import annotations

import base64
import json

import httpx
import pytest

from src.fpbench.fingerprint_engine.errors import ProviderUnavailableError, TemplateExtractionError, VerificationError
from src.fpbench.fingerprint_engine.providers.sourceafis_client import SourceAfisClient, SourceAfisHealth
from src.fpbench.fingerprint_engine.providers.sourceafis_provider import SourceAfisFingerprintEngine
from src.fpbench.fingerprint_engine.types import FingerprintImage, FingerprintTemplate, GalleryTemplate


def _sourceafis_template(template_bytes: bytes, *, image_id: str | None = None) -> FingerprintTemplate:
    return FingerprintTemplate(
        provider_id="sourceafis_open",
        provider_version="3.18.1",
        template_format="sourceafis",
        template_version="3.18.1",
        template_bytes=template_bytes,
        image_id=image_id,
    )


def test_sourceafis_http_client_contract_encodes_binary_inputs_and_decodes_responses() -> None:
    requests: list[tuple[str, str, dict[str, object] | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8")) if request.content else None
        requests.append((request.method, request.url.path, payload))
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "available": True,
                    "sourceafis_version": "3.18.1",
                    "metadata": {"service": "mock", "template_bytes_b64": "redacted"},
                },
            )
        if request.url.path == "/extract-template":
            assert payload is not None
            assert payload["image_base64"] == base64.b64encode(b"fake png").decode("ascii")
            assert payload["image_format"] == "png"
            assert payload["metadata"]["image_id"] == "probe"
            return httpx.Response(
                200,
                json={
                    "provider_version": "3.18.1",
                    "template_format": "sourceafis",
                    "template_version": "3.18.1",
                    "template_base64": base64.b64encode(b"template-probe").decode("ascii"),
                    "metadata": {"extractor": "mock", "template_bytes_b64": "redacted"},
                },
            )
        if request.url.path == "/verify":
            assert payload is not None
            assert payload["probe_template_base64"] == base64.b64encode(b"template-probe").decode("ascii")
            assert payload["candidate_template_base64"] == base64.b64encode(b"template-candidate").decode("ascii")
            return httpx.Response(
                200,
                json={"score": 42.5, "latency_ms": 3.0, "metadata": {"matcher": "mock"}},
            )
        if request.url.path == "/identify":
            assert payload is not None
            assert payload["probe_template_base64"] == base64.b64encode(b"template-probe").decode("ascii")
            assert payload["gallery"][0]["candidate_id"] == "g1"
            assert payload["gallery"][0]["template_base64"] == base64.b64encode(b"template-candidate").decode("ascii")
            assert payload["gallery"][0]["metadata"]["subject_id"] == "s1"
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"candidate_id": "g2", "metadata": {"subject_id": "s2"}, "score": 25.0},
                        {"candidate_id": "g1", "metadata": {"subject_id": "s1"}, "score": 30.0},
                    ],
                    "latency_ms": 4.0,
                },
            )
        return httpx.Response(404, json={"detail": "missing endpoint"})

    client = SourceAfisClient(
        "http://sourceafis.test",
        transport=httpx.MockTransport(handler),
    )

    health = client.health()
    extracted = client.extract_template(
        FingerprintImage(image_bytes=b"fake png", image_id="probe", mime_type="image/png")
    )
    probe = _sourceafis_template(b"template-probe", image_id="probe")
    candidate = _sourceafis_template(b"template-candidate", image_id="candidate")
    match = client.verify(probe, candidate)
    identification = client.identify(
        probe,
        [
            GalleryTemplate(gallery_id="g1", subject_id="s1", template=candidate),
            GalleryTemplate(gallery_id="g2", subject_id="s2", template=_sourceafis_template(b"other")),
        ],
        top_k=2,
    )

    assert health.available is True
    assert health.version == "3.18.1"
    assert health.metadata == {"service": "mock"}
    assert extracted["template_bytes"] == b"template-probe"
    assert extracted["metadata"] == {"extractor": "mock"}
    assert match["score"] == pytest.approx(42.5)
    assert match["normalized_score"] is None
    assert identification["candidates"][0]["gallery_id"] == "g2"
    assert [request[:2] for request in requests] == [
        ("GET", "/health"),
        ("POST", "/extract-template"),
        ("POST", "/verify"),
        ("POST", "/identify"),
    ]


def test_sourceafis_client_serializes_image_dpi_as_metadata() -> None:
    seen_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payload.update(payload)
        return httpx.Response(
            200,
            json={
                "template_base64": base64.b64encode(b"template-with-dpi").decode("ascii"),
                "template_format": "sourceafis",
                "template_version": "3.18.1",
            },
        )

    client = SourceAfisClient(
        "http://sourceafis.test",
        transport=httpx.MockTransport(handler),
    )

    client.extract_template(
        FingerprintImage(image_bytes=b"fake png", image_id="probe", mime_type="image/png", dpi=1000)
    )

    assert seen_payload["metadata"]["dpi"] == 1000


def test_sourceafis_client_maps_sidecar_errors_to_domain_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/extract-template":
            return httpx.Response(400, json={"detail": "image decode failed"})
        if request.url.path == "/verify":
            return httpx.Response(400, json={"detail": "template mismatch"})
        return httpx.Response(503, json={"detail": "service warming up"})

    client = SourceAfisClient(
        "http://sourceafis.test",
        transport=httpx.MockTransport(handler),
    )
    template = _sourceafis_template(b"template")

    with pytest.raises(ProviderUnavailableError, match="503"):
        client.health()
    with pytest.raises(TemplateExtractionError, match="image decode failed"):
        client.extract_template(FingerprintImage(image_bytes=b"not an image"))
    with pytest.raises(VerificationError, match="template mismatch"):
        client.verify(template, template)


def test_sourceafis_client_timeout_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCEAFIS_CONNECT_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("SOURCEAFIS_READ_TIMEOUT_SECONDS", "11")
    monkeypatch.setenv("SOURCEAFIS_EXTRACT_TIMEOUT_SECONDS", "17")
    monkeypatch.setenv("SOURCEAFIS_VERIFY_TIMEOUT_SECONDS", "19")

    client = SourceAfisClient("http://sourceafis.test", transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))

    assert client.timeout_settings.connect_timeout_seconds == pytest.approx(3.0)
    assert client.timeout_settings.read_timeout_seconds == pytest.approx(11.0)
    assert client.timeout_settings.extract_timeout_seconds == pytest.approx(17.0)
    assert client.timeout_settings.verify_timeout_seconds == pytest.approx(19.0)


def test_sourceafis_client_timeout_seconds_keeps_backward_compatible_single_value() -> None:
    client = SourceAfisClient(
        "http://sourceafis.test",
        timeout_seconds=7,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
    )

    assert client.timeout_settings.connect_timeout_seconds == pytest.approx(7.0)
    assert client.timeout_settings.read_timeout_seconds == pytest.approx(7.0)
    assert client.timeout_settings.extract_timeout_seconds == pytest.approx(7.0)
    assert client.timeout_settings.verify_timeout_seconds == pytest.approx(7.0)


def test_sourceafis_provider_uses_mock_client_for_successful_operations() -> None:
    class MockSourceAfisClient:
        service_url = "mock://sourceafis"

        def health(self) -> SourceAfisHealth:
            return SourceAfisHealth(
                available=True,
                version="3.18.1",
                metadata={"service": "mock", "template_bytes_b64": "redacted"},
            )

        def extract_template(self, image: FingerprintImage) -> dict[str, object]:
            return {
                "provider_version": "3.18.1",
                "template_format": "sourceafis",
                "template_version": "3.18.1",
                "template_bytes": f"template:{image.image_id}".encode("ascii"),
                "metadata": {"extractor": "mock", "template_bytes_b64": "redacted"},
            }

        def verify(
            self,
            probe_template: FingerprintTemplate,
            candidate_template: FingerprintTemplate,
        ) -> dict[str, object]:
            assert probe_template.template_bytes
            assert candidate_template.template_bytes
            return {"score": 55.25, "latency_ms": 1.5, "metadata": {"matcher": "mock"}}

        def identify(
            self,
            probe_template: FingerprintTemplate,
            gallery: list[GalleryTemplate],
            top_k: int = 10,
        ) -> dict[str, object]:
            assert probe_template.template_bytes
            assert top_k == 2
            return {
                "candidates": [
                    {"gallery_id": gallery[1].gallery_id, "score": 19.0},
                    {"gallery_id": gallery[0].gallery_id, "score": 19.0},
                    {"gallery_id": gallery[2].gallery_id, "score": 7.0},
                ],
                "latency_ms": 2.5,
            }

    engine = SourceAfisFingerprintEngine(client=MockSourceAfisClient())
    metadata = engine.metadata()
    probe = engine.extract_template(FingerprintImage(image_bytes=b"probe", image_id="probe"))
    candidate_a = engine.extract_template(FingerprintImage(image_bytes=b"a", image_id="a"))
    candidate_b = engine.extract_template(FingerprintImage(image_bytes=b"b", image_id="b"))
    candidate_c = engine.extract_template(FingerprintImage(image_bytes=b"c", image_id="c"))

    match = engine.verify(probe, candidate_a)
    identification = engine.identify(
        probe,
        [
            GalleryTemplate(gallery_id="gallery-a", subject_id="subject-a", template=candidate_a),
            GalleryTemplate(gallery_id="gallery-b", subject_id="subject-b", template=candidate_b),
            GalleryTemplate(gallery_id="gallery-c", subject_id="subject-c", template=candidate_c),
        ],
        top_k=2,
    )

    assert metadata.available is True
    assert metadata.provider_version == "3.18.1"
    assert probe.provider_id == "sourceafis_open"
    assert probe.template_format == "sourceafis"
    assert probe.template_bytes == b"template:probe"
    assert "template_bytes_b64" not in probe.metadata
    assert match.score == pytest.approx(55.25)
    assert match.normalized_score is None
    assert "not normalized" in match.warnings[0]
    assert [candidate.gallery_id for candidate in identification.candidates] == ["gallery-a", "gallery-b"]
    assert [candidate.rank for candidate in identification.candidates] == [1, 2]
    assert identification.top_candidate == identification.candidates[0]
    assert identification.candidates[0].score == pytest.approx(19.0)
    assert identification.candidates[0].subject_id == "subject-a"
