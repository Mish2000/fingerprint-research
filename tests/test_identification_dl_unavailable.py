from __future__ import annotations

from pathlib import Path

import pytest

from apps.api.candidate_source_resolver import CandidateSourceResolver
from apps.api.identification_service import IdentificationService
from apps.api.schemas import MatchMethod
from apps.api.service import MethodUnavailableError
from tests.test_identification_pipeline import InMemoryStore, _fake_rerank, _fake_vectorizer


class _UnavailableRetrievalMatchService:
    dl_resnet = None
    dl_vit = None

    def __init__(self, unavailable: set[str]):
        self.unavailable = set(unavailable)

    def ensure_method_available(self, method):
        raw = getattr(method, "value", method)
        canonical = str(raw).strip().lower()
        if canonical in self.unavailable:
            raise MethodUnavailableError(f"Method {canonical!r} is unavailable: pretrained weights unavailable")


def _write_probe(path: Path) -> Path:
    path.write_bytes(b"fingerprint")
    return path


def test_enroll_rejects_unavailable_dl_without_storing_fake_vector(tmp_path: Path) -> None:
    store = InMemoryStore()
    service = IdentificationService(
        store=store,
        match_service=_UnavailableRetrievalMatchService({"dl"}),
    )
    probe = _write_probe(tmp_path / "enroll.bin")

    with pytest.raises(MethodUnavailableError, match="Method 'dl' is unavailable"):
        service.enroll_from_path(
            path=str(probe),
            full_name="Alice Levi",
            national_id="111111111",
            capture="plain",
            vector_methods=("dl",),
        )

    assert store.people == {}
    assert store.vectors == {}


def test_enroll_resolves_dl_quick_alias_before_availability_check(tmp_path: Path) -> None:
    store = InMemoryStore()
    service = IdentificationService(
        store=store,
        match_service=_UnavailableRetrievalMatchService({"dl"}),
    )
    probe = _write_probe(tmp_path / "enroll_alias.bin")

    with pytest.raises(MethodUnavailableError, match="Method 'dl' is unavailable"):
        service.enroll_from_path(
            path=str(probe),
            full_name="Alice Levi",
            national_id="111111111",
            capture="plain",
            vector_methods=("dl_quick",),
        )

    assert store.vectors == {}


def test_identify_rejects_unavailable_vit_before_vectorization(tmp_path: Path) -> None:
    store = InMemoryStore()
    service = IdentificationService(
        store=store,
        match_service=_UnavailableRetrievalMatchService({"vit"}),
    )
    probe = _write_probe(tmp_path / "probe.bin")

    with pytest.raises(MethodUnavailableError, match="Method 'vit' is unavailable"):
        service.identify_from_path(
            path=str(probe),
            capture="plain",
            retrieval_method="vit",
            shortlist_size=1,
        )


def test_identify_marks_unavailable_rerank_method_without_dropping_shortlist(tmp_path: Path) -> None:
    store = InMemoryStore()
    resolver = CandidateSourceResolver()
    service = IdentificationService(
        store=store,
        match_service=_UnavailableRetrievalMatchService({"sift"}),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
        candidate_source_resolver=resolver,
    )
    candidate = tmp_path / "candidate.bin"
    candidate.write_bytes(b"A_candidate")
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"A_probe")

    receipt = service.enroll_from_path(
        path=str(candidate),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    resolver.register_test_source(receipt.random_id, candidate)

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="dl",
        rerank_method=MatchMethod.sift,
        shortlist_size=1,
    )

    assert result.top_candidate is not None
    assert result.top_candidate.full_name == "Alice Levi"
    assert result.top_candidate.rerank_score is None
    assert result.top_candidate.rerank_status == "skipped_method_unavailable"
    assert result.top_candidate.candidate_source_status == "test_fixture_source"
    assert result.decision is False
    assert result.decision_status == "candidate_found_rerank_skipped"
    assert result.rerank_status == "skipped_method_unavailable"
