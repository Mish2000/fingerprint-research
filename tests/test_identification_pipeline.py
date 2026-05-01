from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pytest

from apps.api.identification_service import IdentificationService
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

    def enroll(
        self,
        *,
        full_name: str,
        national_id: str,
        image_bytes: bytes,
        capture: str,
        ext: str,
        vectors: dict[str, np.ndarray],
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
            sha256=hashlib.sha256(image_bytes).hexdigest(),
            created_at=created_at,
            image_bytes=image_bytes,
        )
        for method, vec in vectors.items():
            self.vectors[(rid, method)] = np.asarray(vec, dtype=np.float32).reshape(-1)
        return EnrollmentReceipt(
            random_id=rid,
            created_at=created_at,
            vector_methods=sorted(vectors.keys()),
            image_sha256=hashlib.sha256(image_bytes).hexdigest(),
        )

    def purge(self, random_id: str) -> bool:
        existed = random_id in self.people
        self.people.pop(random_id, None)
        self.raw.pop(random_id, None)
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

    def shortlist_by_vector(self, *, method: str, probe_vector: np.ndarray, limit: int, candidate_ids=None):
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


def _fake_rerank(method: MatchMethod, probe_path: str, candidate_path: str, probe_capture: str, candidate_capture: str) -> float:
    probe = Path(probe_path).read_bytes()[:1]
    cand = Path(candidate_path).read_bytes()[:1]
    return 0.95 if probe == cand else 0.05


def _write_probe(path: Path, first_byte: bytes) -> Path:
    path.write_bytes(first_byte + b"_probe")
    return path


def test_identification_pipeline_shortlist_then_rerank(tmp_path: Path) -> None:
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
    assert result.top_candidate.rerank_score == 0.95
    assert result.decision is True


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



def test_identification_rejects_unsupported_shortlist_retrieval_method(tmp_path: Path) -> None:
    service = IdentificationService(
        store=InMemoryStore(),
        vectorizers={"dl": _fake_vectorizer, "vit": _fake_vectorizer},
        rerank_callable=_fake_rerank,
    )
    probe = _write_probe(tmp_path / "probe.bin", b"A")

    with pytest.raises(ValueError, match="shortlist retrieval"):
        service.identify_from_path(
            path=str(probe),
            capture="plain",
            retrieval_method="sift",
            rerank_method=MatchMethod.sift,
            shortlist_size=2,
        )


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
