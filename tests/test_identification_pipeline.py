from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from fastapi import HTTPException, UploadFile

import apps.api.main as api_main
from apps.api.candidate_source_resolver import CandidateSourceResolver
from apps.api.identification_service import IdentificationService, default_enrollment_vector_methods
from apps.api.schemas import MatchMethod
from src.fpbench.identification.secure_split_store import (
    EnrollmentReceipt,
    IdentifyHints,
    PersonDirectoryRecord,
    RawFingerprintRecord,
)


def _safe_capture(raw: str | None) -> str:
    s = str(raw or "plain").strip().lower()
    return s or "plain"


def _normalize_name(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


def _normalize_national_id(national_id: str) -> str:
    return "".join(ch for ch in str(national_id).strip() if ch.isdigit())


def _pattern_to_like(raw: str) -> str:
    s = str(raw).strip().replace("*", "%")
    if "%" not in s:
        s += "%"
    return s


class InMemoryStore:
    def __init__(self):
        self.people: dict[str, PersonDirectoryRecord] = {}
        self.raw: dict[str, RawFingerprintRecord] = {}
        self.vectors: dict[tuple[str, str], np.ndarray] = {}
        self.legacy_bytes: dict[str, bytes] = {}
        self.shortlist_calls: list[str] = []

    def enroll(
        self,
        *,
        full_name: str,
        national_id: str,
        capture: str,
        ext: str,
        vectors: dict[str, np.ndarray],
        image_bytes: bytes | None = None,
        image_sha256: str | None = None,
        byte_size: int | None = None,
        random_id: str | None = None,
        created_at: str | None = None,
        replace_existing: bool = False,
    ) -> EnrollmentReceipt:
        national_id_norm = _normalize_national_id(national_id)
        existing = None
        for rec in self.people.values():
            if rec.national_id == national_id_norm:
                existing = rec.random_id
                break
        if existing is not None and not replace_existing:
            raise ValueError("national_id already enrolled; pass replace_existing=True to rotate the template")
        if existing is not None and replace_existing:
            self.purge(existing)

        rid = random_id or uuid.uuid4().hex
        created_at = created_at or "2026-03-18T00:00:00+00:00"
        if image_sha256 is None:
            if image_bytes is None:
                raise ValueError("image_bytes or image_sha256 must be provided")
            image_hash = hashlib.sha256(image_bytes).hexdigest()
        else:
            image_hash = image_sha256
        byte_size_value = byte_size if byte_size is not None else (len(image_bytes) if image_bytes is not None else None)
        self.people[rid] = PersonDirectoryRecord(
            random_id=rid,
            full_name=full_name,
            name_norm=_normalize_name(full_name),
            national_id=national_id_norm,
            created_at=created_at,
        )
        self.raw[rid] = RawFingerprintRecord(
            random_id=rid,
            capture=_safe_capture(capture),
            ext=ext,
            sha256=image_hash,
            byte_size=byte_size_value,
            created_at=created_at,
            legacy_image_bytes_present=False,
        )
        for method, vec in vectors.items():
            self.vectors[(rid, method)] = np.asarray(vec, dtype=np.float32).reshape(-1)
        return EnrollmentReceipt(
            random_id=rid,
            created_at=created_at,
            vector_methods=sorted(vectors.keys()),
            image_sha256=image_hash,
        )

    def purge(self, random_id: str) -> bool:
        existed = random_id in self.people
        self.people.pop(random_id, None)
        self.raw.pop(random_id, None)
        self.legacy_bytes.pop(random_id, None)
        for key in [key for key in list(self.vectors.keys()) if key[0] == random_id]:
            self.vectors.pop(key, None)
        return existed

    def total_people(self) -> int:
        return len(self.people)

    def count_vectors(self, method: str) -> int:
        return sum(1 for _, m in self.vectors.keys() if m == method)

    def search_people(self, hints: IdentifyHints, *, limit: int | None = None):
        rows = list(self.people.values())
        if hints.name_pattern:
            pat = _pattern_to_like(_normalize_name(hints.name_pattern)).replace("%", "")
            rows = [r for r in rows if r.name_norm.startswith(pat)]
        if hints.national_id_pattern:
            pat = _pattern_to_like(_normalize_national_id(hints.national_id_pattern)).replace("%", "")
            rows = [r for r in rows if r.national_id.startswith(pat)]
        rows.sort(key=lambda item: item.created_at, reverse=True)
        if limit is not None:
            rows = rows[:limit]
        return rows

    def get_person(self, random_id: str):
        return self.people.get(random_id)

    def load_raw_fingerprint(self, random_id: str):
        return self.raw.get(random_id)

    def load_legacy_raw_fingerprint_image_bytes(self, random_id: str):
        return self.legacy_bytes.get(random_id)

    def shortlist_by_vector(self, *, method: str, probe_vector: np.ndarray, limit: int, candidate_ids=None):
        self.shortlist_calls.append(method)
        probe = np.asarray(probe_vector, dtype=np.float32).reshape(-1)
        ids = list(candidate_ids) if candidate_ids is not None else [rid for rid, m in self.vectors.keys() if m == method]
        rows = []
        for rid in ids:
            key = (rid, method)
            if key not in self.vectors:
                continue
            score = float(np.dot(probe, self.vectors[key]))
            rows.append((rid, score))
        rows.sort(key=lambda item: item[1], reverse=True)
        return rows[:limit]

    def dump_layout(self):
        return {
            "backend": "memory",
            "dual_database_enabled": "false",
            "person_table": "memory_biometric_db.memory_person_directory",
            "raw_fingerprints_table": "memory_biometric_db.memory_raw_fingerprints",
            "feature_vectors_table": "memory_biometric_db.memory_feature_vectors",
            "identity_map_table": "memory_identity_db.memory_identity_map",
        }


def _fake_vectorizer(path: str) -> np.ndarray:
    payload = Path(path).read_bytes()
    key = payload[:1]
    mapping = {
        b"A": np.array([1.0, 0.0], dtype=np.float32),
        b"B": np.array([0.0, 1.0], dtype=np.float32),
        b"C": np.array([0.7, 0.7], dtype=np.float32),
    }
    return mapping.get(key, np.array([0.0, 0.0], dtype=np.float32))


class _FakeVectorModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def embed_path(self, path: str, capture: str | None = None):
        del path, capture
        return np.ones(self.dim, dtype=np.float32), 0.0


class _FakeMatchServiceForVectorizers:
    def __init__(self) -> None:
        self.dl_resnet = _FakeVectorModel(512)
        self.dl_vit = _FakeVectorModel(768)

    def ensure_method_available(self, method) -> None:
        assert str(method) in {"classic_orb", "classic_gftt_orb", "minutiae", "harris", "sift", "dl", "vit"}

    def embed_classic_orb_path(self, path: str, capture: str | None = None) -> np.ndarray:
        del path, capture
        return np.ones(512, dtype=np.float32)

    def embed_classic_gftt_orb_path(self, path: str, capture: str | None = None) -> np.ndarray:
        del path, capture
        return np.ones(512, dtype=np.float32)

    def embed_minutiae_path(self, path: str, capture: str | None = None) -> np.ndarray:
        del path, capture
        return np.ones(512, dtype=np.float32)

    def embed_harris_path(self, path: str, capture: str | None = None) -> np.ndarray:
        del path, capture
        return np.ones(512, dtype=np.float32)

    def embed_sift_path(self, path: str, capture: str | None = None) -> np.ndarray:
        del path, capture
        return np.ones(512, dtype=np.float32)


def _fake_rerank(method: MatchMethod, probe_path: str, candidate_path: str, probe_capture: str, candidate_capture: str) -> float:
    probe = Path(probe_path).read_bytes()[:1]
    cand = Path(candidate_path).read_bytes()[:1]
    return 0.95 if probe == cand else 0.05


def _write_probe(path: Path, first_byte: bytes) -> Path:
    path.write_bytes(first_byte + b"_probe")
    return path


def _recording_vectorizer(calls: list[tuple[str, str]]) -> Callable[..., np.ndarray]:
    def _vectorizer(path: str, capture: str | None = None) -> np.ndarray:
        calls.append((Path(path).name, _safe_capture(capture)))
        return _fake_vectorizer(path)

    return _vectorizer


def _all_direct_vectorizers() -> dict[str, Callable[..., np.ndarray]]:
    return {method: _fake_vectorizer for method in default_enrollment_vector_methods()}


def test_identification_service_builds_vectorizers_from_registry_capability_contract() -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        match_service=_FakeMatchServiceForVectorizers(),
        rerank_callable=_fake_rerank,
    )

    assert set(service.vectorizers) == set(service.method_registry.direct_vector_retrieval_methods())
    assert set(service.vectorizers) == {"classic_orb", "classic_gftt_orb", "minutiae", "harris", "sift", "dl", "vit"}


def test_default_enrollment_vector_methods_follow_direct_retrieval_registry() -> None:
    assert default_enrollment_vector_methods() == (
        "classic_orb",
        "classic_gftt_orb",
        "minutiae",
        "harris",
        "sift",
        "dl",
        "vit",
    )
    assert "dedicated" not in default_enrollment_vector_methods()


def test_enroll_from_path_default_vector_methods_enrolls_all_direct_retrieval_methods(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers=_all_direct_vectorizers(),
        rerank_callable=_fake_rerank,
    )
    alice = _write_probe(tmp_path / "alice_defaults.bin", b"A")

    receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=None,
    )

    assert set(receipt.vector_methods) == set(default_enrollment_vector_methods())
    assert set(service.store.vectors) == {
        (receipt.random_id, method)
        for method in default_enrollment_vector_methods()
    }


def test_enroll_from_path_explicit_legacy_subset_stays_limited(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers=_all_direct_vectorizers(),
        rerank_callable=_fake_rerank,
    )
    alice = _write_probe(tmp_path / "alice_legacy_subset.bin", b"A")

    receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl", "vit"),
    )

    assert receipt.vector_methods == ["dl", "vit"]
    assert set(service.store.vectors) == {
        (receipt.random_id, "dl"),
        (receipt.random_id, "vit"),
    }


def test_enroll_from_path_explicit_classic_subset_stays_limited(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers=_all_direct_vectorizers(),
        rerank_callable=_fake_rerank,
    )
    alice = _write_probe(tmp_path / "alice_classic_subset.bin", b"A")

    receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("classic_orb", "sift"),
    )

    assert receipt.vector_methods == ["classic_orb", "sift"]
    assert set(service.store.vectors) == {
        (receipt.random_id, "classic_orb"),
        (receipt.random_id, "sift"),
    }


def test_identification_pipeline_keeps_metadata_only_shortlist_with_explicit_skip(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )

    alice = _write_probe(tmp_path / "alice.bin", b"A")
    bob = _write_probe(tmp_path / "bob.bin", b"B")
    probe = _write_probe(tmp_path / "probe.bin", b"A")

    service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    service.enroll_from_path(
        path=str(bob),
        full_name="Bob Cohen",
        national_id="222222222",
        capture="roll",
        vector_methods=("dl",),
    )

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="dl",
        rerank_method=MatchMethod.sift,
        shortlist_size=2,
    )

    assert result.total_enrolled == 2
    assert result.candidate_pool_size == 2
    assert result.top_candidate is not None
    assert result.top_candidate.full_name == "Alice Levi"
    assert result.top_candidate.rank == 1
    assert result.top_candidate.rerank_score is None
    assert result.top_candidate.rerank_status == "skipped_no_candidate_source"
    assert result.top_candidate.candidate_source_status == "no_candidate_source"
    assert result.top_candidate.decision is None
    assert result.decision is False
    assert result.decision_status == "candidate_found_rerank_skipped"
    assert result.decision_basis == "vector_shortlist_only"
    assert result.rerank_status == "skipped_no_candidate_source"
    assert result.rerank_summary["performed_count"] == 0
    assert result.rerank_summary["skipped_count"] == 2


def test_identification_pipeline_reranks_with_safe_test_candidate_source(tmp_path: Path) -> None:
    resolver = CandidateSourceResolver()
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
        candidate_source_resolver=resolver,
    )

    alice = _write_probe(tmp_path / "alice.bin", b"A")
    bob = _write_probe(tmp_path / "bob.bin", b"B")
    probe = _write_probe(tmp_path / "probe.bin", b"A")

    alice_receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    bob_receipt = service.enroll_from_path(
        path=str(bob),
        full_name="Bob Cohen",
        national_id="222222222",
        capture="roll",
        vector_methods=("dl",),
    )
    resolver.register_test_source(alice_receipt.random_id, alice, capture="plain")
    resolver.register_test_source(bob_receipt.random_id, bob, capture="roll")

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="dl",
        rerank_method=MatchMethod.sift,
        shortlist_size=2,
    )

    assert result.top_candidate is not None
    assert result.top_candidate.full_name == "Alice Levi"
    assert result.top_candidate.rerank_score == 0.95
    assert result.top_candidate.rerank_status == "rerank_performed"
    assert result.top_candidate.candidate_source_status == "test_fixture_source"
    assert result.top_candidate.decision is True
    assert result.decision is True
    assert result.decision_status == "rerank_match"
    assert result.decision_basis == "rerank"
    assert result.rerank_summary["performed_count"] == 2


def test_identification_pipeline_legacy_bytes_rerank_is_explicit(tmp_path: Path) -> None:
    store = InMemoryStore()
    service = IdentificationService(
        store=store,
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )

    alice = _write_probe(tmp_path / "alice_legacy.bin", b"A")
    probe = _write_probe(tmp_path / "probe_legacy.bin", b"A")

    receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    assert store.raw[receipt.random_id].legacy_image_bytes_present is False
    assert receipt.random_id not in store.legacy_bytes

    store.raw[receipt.random_id] = replace(
        store.raw[receipt.random_id],
        legacy_image_bytes_present=True,
    )
    store.legacy_bytes[receipt.random_id] = alice.read_bytes()

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="dl",
        rerank_method=MatchMethod.sift,
        shortlist_size=1,
    )

    assert result.top_candidate is not None
    assert result.top_candidate.rerank_score == 0.95
    assert result.top_candidate.rerank_status == "rerank_performed"
    assert result.top_candidate.candidate_source_status == "legacy_db_image_bytes"
    assert result.decision is True


def test_identification_pipeline_missing_candidate_source_file_keeps_shortlist(tmp_path: Path) -> None:
    resolver = CandidateSourceResolver()
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
        candidate_source_resolver=resolver,
    )

    alice = _write_probe(tmp_path / "alice_missing_source.bin", b"A")
    probe = _write_probe(tmp_path / "probe_missing_source.bin", b"A")
    receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    resolver.register_test_source(receipt.random_id, tmp_path / "deleted_candidate.bin")

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
    assert result.top_candidate.rerank_status == "skipped_no_candidate_source"
    assert result.top_candidate.candidate_source_status == "candidate_source_missing"
    assert result.decision is False
    assert result.decision_status == "candidate_found_rerank_skipped"


def test_identification_pipeline_matcher_exception_marks_candidate_failed(tmp_path: Path) -> None:
    resolver = CandidateSourceResolver()

    def _broken_rerank(
        method: MatchMethod,
        probe_path: str,
        candidate_path: str,
        probe_capture: str,
        candidate_capture: str,
    ) -> float:
        _ = (method, probe_path, candidate_path, probe_capture, candidate_capture)
        raise RuntimeError("simulated matcher failure")

    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_broken_rerank,
        candidate_source_resolver=resolver,
    )

    alice = _write_probe(tmp_path / "alice_broken_rerank.bin", b"A")
    probe = _write_probe(tmp_path / "probe_broken_rerank.bin", b"A")
    receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    resolver.register_test_source(receipt.random_id, alice)

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
    assert result.top_candidate.rerank_status == "failed_rerank_error"
    assert result.top_candidate.candidate_source_status == "test_fixture_source"
    assert result.decision is False
    assert result.decision_status == "candidate_found_rerank_failed"


def test_identification_pipeline_respects_indexed_hints(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )

    alice = _write_probe(tmp_path / "alice.bin", b"A")
    bob = _write_probe(tmp_path / "bob.bin", b"B")
    probe = _write_probe(tmp_path / "probe.bin", b"B")

    service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    service.enroll_from_path(
        path=str(bob),
        full_name="Bob Cohen",
        national_id="222222222",
        capture="roll",
        vector_methods=("dl",),
    )

    result = service.identify_from_path(
        path=str(probe),
        capture="roll",
        retrieval_method="dl",
        rerank_method=MatchMethod.sift,
        shortlist_size=5,
        hints=IdentifyHints(name_pattern="bob*"),
    )

    assert result.candidate_pool_size == 1
    assert result.top_candidate is not None
    assert result.top_candidate.full_name == "Bob Cohen"
    assert result.hints_applied == {"name_pattern": "bob*"}



@pytest.mark.parametrize("retrieval_method", ["classic_orb", "minutiae", "sift"])
def test_identification_classic_retrieval_methods_reach_vectorization_and_shortlist(
    tmp_path: Path,
    retrieval_method: str,
) -> None:
    calls: list[tuple[str, str]] = []
    store = InMemoryStore()
    service = IdentificationService(
        store=store,
        vectorizers={retrieval_method: _recording_vectorizer(calls)},
        rerank_callable=_fake_rerank,
    )
    alice = _write_probe(tmp_path / f"alice_{retrieval_method}.bin", b"A")
    probe = _write_probe(tmp_path / f"probe_{retrieval_method}.bin", b"A")

    service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=(retrieval_method,),
    )

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method=retrieval_method,
        rerank_method=MatchMethod.sift,
        shortlist_size=1,
    )

    assert result.retrieval_method == retrieval_method
    assert result.top_candidate is not None
    assert result.top_candidate.full_name == "Alice Levi"
    assert calls == [(alice.name, "plain"), (probe.name, "plain")]
    assert store.shortlist_calls == [retrieval_method]


def test_identification_rejects_unsupported_shortlist_retrieval_method(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )
    probe = _write_probe(tmp_path / "probe.bin", b"A")

    with pytest.raises(ValueError, match="experimental rerank-only method") as excinfo:
        service.identify_from_path(
            path=str(probe),
            capture="plain",
            retrieval_method="dedicated",
            rerank_method=MatchMethod.sift,
            shortlist_size=2,
        )
    assert "does not have a validated fixed-size direct retrieval vector adapter yet" in str(excinfo.value)
    assert "dedicated_aggregated_patch_descriptor_v1" in str(excinfo.value)
    assert "Supported retrieval methods: ['classic_orb', 'classic_gftt_orb', 'minutiae', 'harris', 'sift', 'dl', 'vit']" in str(
        excinfo.value
    )


def test_identify_search_api_rejects_dedicated_retrieval_method_with_capability_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers=_all_direct_vectorizers(),
        rerank_callable=_fake_rerank,
    )
    monkeypatch.setattr(api_main, "_service_for_scope", lambda store_scope="operational": service)

    upload = UploadFile(filename="probe.png", file=BytesIO(b"A"))
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            api_main.identify_search(
                img=upload,
                capture="plain",
                retrieval_method="dedicated",
                rerank_method=MatchMethod.sift,
                shortlist_size=2,
                threshold=None,
                name_pattern=None,
                national_id_pattern=None,
                created_from=None,
                created_to=None,
                store_scope="operational",
            )
        )

    assert excinfo.value.status_code == 400
    detail = str(excinfo.value.detail)
    assert (
        "Method 'dedicated' is currently an experimental rerank-only method and does not have a "
        "validated fixed-size direct retrieval vector adapter yet."
    ) in detail
    assert "Supported retrieval methods: ['classic_orb', 'classic_gftt_orb', 'minutiae', 'harris', 'sift', 'dl', 'vit']" in detail


def test_identification_accepts_valid_retrieval_and_rerank_combination(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )
    alice = _write_probe(tmp_path / "alice.bin", b"A")
    probe = _write_probe(tmp_path / "probe.bin", b"A")

    service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("vit",),
    )

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="vit",
        rerank_method=MatchMethod.dedicated,
        shortlist_size=1,
    )

    assert result.top_candidate is not None
    assert result.retrieval_method == "vit"
    assert result.rerank_method == MatchMethod.dedicated
    assert result.top_candidate.full_name == "Alice Levi"


def test_identification_normalizes_aliases_for_retrieval_and_rerank(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )

    alice = _write_probe(tmp_path / "alice_alias.bin", b"A")
    probe = _write_probe(tmp_path / "probe_alias.bin", b"A")

    service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl_quick",),
    )

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="dl_quick",
        rerank_method="classic_v2",
        shortlist_size=1,
    )

    assert result.retrieval_method == "dl"
    assert result.rerank_method == MatchMethod.classic_gftt_orb
    assert result.retrieval_method_metadata["requested_method"] == "dl_quick"
    assert result.retrieval_method_metadata["canonical_method"] == "dl"
    assert result.rerank_method_metadata["requested_method"] == "classic_v2"
    assert result.rerank_method_metadata["canonical_method"] == "classic_gftt_orb"


@pytest.mark.parametrize(
    ("identify_kwargs", "expected_rerank_calls", "expected_skipped_count"),
    [
        ({"skip_rerank": True}, 0, 3),
        ({"rerank_limit": 0}, 0, 3),
        ({"rerank_limit": 1}, 1, 2),
        ({}, 3, 0),
    ],
)
def test_identification_service_rerank_policy_controls_pairwise_call_count(
    tmp_path: Path,
    identify_kwargs: dict[str, object],
    expected_rerank_calls: int,
    expected_skipped_count: int,
) -> None:
    resolver = CandidateSourceResolver()
    calls: list[str] = []

    def _counting_rerank(
        method: MatchMethod,
        probe_path: str,
        candidate_path: str,
        probe_capture: str,
        candidate_capture: str,
    ) -> float:
        del method, probe_path, probe_capture, candidate_capture
        calls.append(Path(candidate_path).name)
        return 0.95 if Path(candidate_path).read_bytes()[:1] == b"A" else 0.05

    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer},
        rerank_callable=_counting_rerank,
        candidate_source_resolver=resolver,
    )

    alice = _write_probe(tmp_path / "alice_policy.bin", b"A")
    bob = _write_probe(tmp_path / "bob_policy.bin", b"B")
    casey = _write_probe(tmp_path / "casey_policy.bin", b"C")
    probe = _write_probe(tmp_path / "probe_policy.bin", b"A")

    alice_receipt = service.enroll_from_path(
        path=str(alice),
        full_name="Alice Levi",
        national_id="111111111",
        capture="plain",
        vector_methods=("dl",),
    )
    bob_receipt = service.enroll_from_path(
        path=str(bob),
        full_name="Bob Cohen",
        national_id="222222222",
        capture="plain",
        vector_methods=("dl",),
    )
    casey_receipt = service.enroll_from_path(
        path=str(casey),
        full_name="Casey Dan",
        national_id="333333333",
        capture="plain",
        vector_methods=("dl",),
    )
    resolver.register_test_source(alice_receipt.random_id, alice, capture="plain")
    resolver.register_test_source(bob_receipt.random_id, bob, capture="plain")
    resolver.register_test_source(casey_receipt.random_id, casey, capture="plain")

    result = service.identify_from_path(
        path=str(probe),
        capture="plain",
        retrieval_method="dl",
        rerank_method=MatchMethod.sift,
        shortlist_size=3,
        **identify_kwargs,
    )

    assert len(calls) == expected_rerank_calls
    assert result.rerank_summary["performed_count"] == expected_rerank_calls
    assert result.rerank_summary["skipped_count"] == expected_skipped_count
    assert [candidate.retrieval_rank for candidate in result.candidates] == [candidate.rank for candidate in result.candidates]
    if expected_rerank_calls == 0:
        assert all(candidate.rerank_status == "skipped_vector_only_mode" for candidate in result.candidates)
