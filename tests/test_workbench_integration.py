"""Real API + split PostgreSQL + Java, using only synthetic software fixtures."""
import os
import uuid
import math

import pytest
from fastapi.testclient import TestClient

from src.fpbench.demo_fixtures import synthetic_fingerprint_png


@pytest.fixture()
def client(monkeypatch, tmp_path):
    if os.getenv("FPBENCH_RUNTIME_INTEGRATION_TESTS") != "true":
        pytest.skip("Set FPBENCH_RUNTIME_INTEGRATION_TESTS=true for real API/DB/Java integration")
    for key in ("IDENTIFICATION_TEST_BIOMETRIC_DATABASE_URL", "IDENTIFICATION_TEST_IDENTITY_DATABASE_URL"):
        if not os.getenv(key):
            pytest.fail(f"Required integration configuration is missing: {key}")
    monkeypatch.setenv("FPBENCH_TABLE_PREFIX", f"workbench_{uuid.uuid4().hex[:8]}_")
    monkeypatch.setenv("FPBENCH_ENROLLMENT_DIR", str(tmp_path / "retained-inputs"))
    monkeypatch.setenv("FPBENCH_API_LAZY_STARTUP", "false")
    monkeypatch.setenv("SOURCEAFIS_ENABLED", "true")
    from apps.api import main
    main._shutdown_services()
    with TestClient(main.app) as test_client:
        yield test_client


def test_real_upload_enrollment_retrieval_rerank_and_delete(client):
    image = synthetic_fingerprint_png(0)
    upload = {"img": ("synthetic.png", image, "image/png")}
    search_data = {"retrieval_method": "classic_orb", "rerank_method": "sift"}
    empty = client.post("/identify/search", data=search_data, files=upload)
    assert empty.status_code == 200, empty.text
    assert empty.json()["top_candidate"] is None
    enrollment = client.post("/identify/enroll", data={
        "full_name": "Synthetic Demo Person", "national_id": "900000002", "vector_methods": "classic_orb",
    }, files=upload)
    assert enrollment.status_code == 200, enrollment.text
    random_id = enrollment.json()["random_id"]
    search = client.post("/identify/search", data=search_data, files=upload)
    assert search.status_code == 200, search.text
    assert search.json()["top_candidate"]["random_id"] == random_id
    assert search.json()["rerank_status"] == "rerank_performed"
    assert math.isfinite(search.json()["top_candidate"]["rerank_score"])
    from apps.api import main
    retained = list(main._ident_service.enrollment_sources.root.glob("*.image"))
    assert len(retained) == 1
    deleted = client.delete(f"/identify/person/{random_id}")
    assert deleted.status_code == 200, deleted.text
    assert deleted.json()["removed"]
    assert not retained[0].exists()
    assert client.post("/identify/search", data=search_data, files=upload).json()["top_candidate"] is None


def test_real_engine_endpoint_and_input_failures(client):
    image = synthetic_fingerprint_png(0)
    pair = {"img_a": ("a.png", image, "image/png"), "img_b": ("b.png", image, "image/png")}
    missing_dpi = client.post("/fingerprint-engine/verify", files=pair)
    assert missing_dpi.status_code == 422
    response = client.post("/fingerprint-engine/verify", data={"dpi_a": "500", "dpi_b": "500"}, files=pair)
    assert response.status_code == 200, response.text
    assert response.json()["provider"] == "sourceafis_open"
    assert math.isfinite(response.json()["score"])
    assert response.json()["decision"] is None
    corrupt = {"img_a": ("bad.png", b"not an image", "image/png"), "img_b": pair["img_b"]}
    invalid = client.post("/match", data={"method": "sift"}, files=corrupt)
    assert invalid.status_code == 400
    assert "decode" in invalid.json()["detail"]
    retired = client.post("/match", data={"method": "dedicated"}, files=pair)
    assert retired.status_code == 400
    assert "retired" in retired.json()["detail"]


@pytest.mark.parametrize("rejected_id", ["900000002", "no-digits"])
def test_rejected_enrollment_cleans_only_the_new_image(client, rejected_id):
    original = synthetic_fingerprint_png(0)
    enrollment = client.post("/identify/enroll", data={
        "full_name": "Synthetic Original", "national_id": "900000002", "vector_methods": "classic_orb",
    }, files={"img": ("original.png", original, "image/png")})
    assert enrollment.status_code == 200, enrollment.text
    from apps.api import main
    service = main._ident_service
    original_id = enrollment.json()["random_id"]
    original_digest = enrollment.json()["image_sha256"]
    retained_before = set(service.enrollment_sources.root.iterdir())
    assert len(retained_before) == 1
    rejected = client.post("/identify/enroll", data={
        "full_name": "Synthetic Rejected", "national_id": rejected_id, "vector_methods": "classic_orb",
    }, files={"img": ("different.png", synthetic_fingerprint_png(1), "image/png")})
    assert rejected.status_code == 400, rejected.text
    assert set(service.enrollment_sources.root.iterdir()) == retained_before
    assert service.enrollment_sources.resolve(original_digest).read_bytes() == original
    assert service.store.total_people() == 1
    assert service.store.get_person(original_id).national_id == "900000002"
    assert client.delete(f"/identify/person/{original_id}").json()["removed"]
