"""Failure compensation uses real files and controlled store failure states."""
import hashlib
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from apps.api.identification_service import IdentificationService
from apps.api.enrollment_sources import EnrollmentSources


@pytest.fixture()
def failed_enrollment(monkeypatch, tmp_path):
    monkeypatch.setenv("FPBENCH_ENROLLMENT_DIR", str(tmp_path / "retained"))
    source = tmp_path / "synthetic.png"
    source.write_bytes(b"synthetic unit-test input")
    error = ValueError("national_id already enrolled")
    store = SimpleNamespace(enroll=Mock(side_effect=error), has_image_reference=Mock(return_value=False))
    service = IdentificationService(store=store, vectorizers={"dl": lambda path, capture=None: np.ones(512)},
                                    rerank_callable=lambda *args: 0.0)
    return service, store, source, error


def enroll(service, source):
    return service.enroll_from_path(path=str(source), full_name="Synthetic Person", national_id="900000001",
                                    capture="plain", vector_methods=["dl"])


def test_rejected_enrollment_removes_its_new_unreferenced_image(failed_enrollment):
    service, store, source, original_error = failed_enrollment
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with pytest.raises(ValueError) as raised:
        enroll(service, source)
    assert raised.value is original_error
    store.has_image_reference.assert_called_once_with(digest)
    assert not service.enrollment_sources.path(digest).exists()
    assert list(service.enrollment_sources.root.iterdir()) == []
    assert source.is_file()


@pytest.mark.parametrize("referenced", [False, True])
def test_rejected_enrollment_never_deletes_an_existing_input(failed_enrollment, referenced):
    service, store, source, original_error = failed_enrollment
    digest = service.enrollment_sources.save(source)
    store.has_image_reference.return_value = referenced
    with pytest.raises(ValueError) as raised:
        enroll(service, source)
    assert raised.value is original_error
    assert service.enrollment_sources.resolve(digest).read_bytes() == source.read_bytes()
    store.has_image_reference.assert_not_called()


def test_late_error_preserves_a_new_image_with_a_committed_reference(failed_enrollment):
    service, store, source, _ = failed_enrollment
    store.has_image_reference.return_value = True
    with pytest.raises(ValueError):
        enroll(service, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert service.enrollment_sources.resolve(digest).read_bytes() == source.read_bytes()


@pytest.mark.parametrize("failure_stage", ["reference_lookup", "remove"])
def test_cleanup_failure_preserves_the_original_error_and_reports_reconciliation(failed_enrollment, caplog, failure_stage):
    service, store, source, original_error = failed_enrollment
    if failure_stage == "reference_lookup":
        store.has_image_reference.side_effect = RuntimeError("database unavailable")
    else:
        service.enrollment_sources.remove = Mock(side_effect=OSError("input is locked"))
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as raised:
        enroll(service, source)
    assert raised.value is original_error
    assert "reconciliation required" in original_error.__notes__[0]
    assert "cleanup requires reconciliation" in caplog.text
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert service.enrollment_sources.resolve(digest).is_file()


def test_atomic_publication_does_not_claim_a_file_another_writer_created(monkeypatch, tmp_path):
    import apps.api.enrollment_sources as module
    source = tmp_path / "synthetic.png"
    source.write_bytes(b"synthetic unit-test input")
    sources = EnrollmentSources(tmp_path / "retained")
    def competing_writer(temporary, destination):
        destination.write_bytes(source.read_bytes())
        raise FileExistsError()
    monkeypatch.setattr(module.os, "link", competing_writer)
    saved = sources.save_with_status(source)
    assert saved.created is False
    assert sources.resolve(saved.digest).read_bytes() == source.read_bytes()
