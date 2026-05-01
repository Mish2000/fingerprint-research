from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from apps.api.method_registry import MethodRegistryError, load_api_method_registry
from apps.api.schemas import MatchMethod
from src.fpbench.identification.secure_split_store import (
    EnrollmentReceipt,
    IdentifyHints,
    RawFingerprintRecord,
    SecureSplitFingerprintStore,
)

Vectorizer = Callable[..., np.ndarray]
RerankCallable = Callable[[MatchMethod, str, str, str, str], float]


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return arr
    return arr / norm


def _safe_capture(raw: Optional[str]) -> str:
    s = str(raw or "plain").strip().lower()
    return s or "plain"


def _mask_national_id(raw: str) -> str:
    s = str(raw)
    if len(s) <= 4:
        return s
    return f"{'*' * max(0, len(s) - 4)}{s[-4:]}"


@dataclass(frozen=True)
class IdentifyCandidateResult:
    rank: int
    random_id: str
    full_name: str
    national_id_masked: str
    created_at: str
    capture: str
    retrieval_score: float
    rerank_score: Optional[float]
    decision: Optional[bool]


@dataclass(frozen=True)
class IdentifyRunResult:
    retrieval_method: str
    rerank_method: MatchMethod
    threshold: float
    decision: bool
    total_enrolled: int
    candidate_pool_size: int
    shortlist_size: int
    hints_applied: Dict[str, str]
    top_candidate: Optional[IdentifyCandidateResult]
    candidates: List[IdentifyCandidateResult]
    latency_ms: Dict[str, float]
    storage_layout: Dict[str, str]
    retrieval_method_metadata: Dict[str, Any]
    rerank_method_metadata: Dict[str, Any]


if TYPE_CHECKING:
    from apps.api.service import MatchService


class IdentificationService:
    """
    1:N identification built on top of the existing 1:1 MatchService.

    Search plan:
      1. Optional indexed hint filtering on the PostgreSQL person table
      2. Fast pgvector shortlist over stored templates (dl / vit)
      3. Existing 1:1 matcher rerank over the shortlist
    """

    def __init__(
        self,
        *,
        database_url: str | None = None,
        identity_database_url: str | None = None,
        table_prefix: str = "",
        store: SecureSplitFingerprintStore | None = None,
        match_service: "MatchService" | None = None,
        vectorizers: Dict[str, Vectorizer] | None = None,
        rerank_callable: RerankCallable | None = None,
    ):
        self.method_registry = load_api_method_registry()
        self.store = store or SecureSplitFingerprintStore(
            database_url=database_url,
            identity_database_url=identity_database_url,
            table_prefix=table_prefix,
        )

        need_match_service = (match_service is None) and (vectorizers is None or rerank_callable is None)
        if need_match_service:
            from apps.api.service import MatchService as _MatchService

            self.match_service = _MatchService()
        else:
            self.match_service = match_service

        if vectorizers is not None:
            self.vectorizers = vectorizers
        else:
            if self.match_service is None:
                raise ValueError("vectorizers must be provided when match_service is not available")
            self.vectorizers = {
                "dl": lambda path, capture=None: self.match_service.dl_resnet.embed_path(path, capture=capture)[0],
                "vit": lambda path, capture=None: self.match_service.dl_vit.embed_path(path, capture=capture)[0],
            }

        if rerank_callable is not None:
            self.rerank_callable = rerank_callable
        else:
            if self.match_service is None:
                raise ValueError("rerank_callable must be provided when match_service is not available")
            self.rerank_callable = self._rerank_with_match_service

    def _vectorize_with_capture(self, method: str, path: str, capture: Optional[str]) -> np.ndarray:
        fn = self.vectorizers[method]
        try:
            return fn(path, capture=capture)
        except TypeError:
            return fn(path)

    def _resolve_retrieval_method(self, retrieval_method: str):
        try:
            return self.method_registry.resolve_retrieval_method(retrieval_method)
        except MethodRegistryError as exc:
            raise ValueError(str(exc)) from exc

    def _resolve_rerank_method(self, rerank_method: MatchMethod | str):
        try:
            return self.method_registry.resolve_rerank_method(rerank_method)
        except MethodRegistryError as exc:
            raise ValueError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll_from_path(
        self,
        *,
        path: str,
        full_name: str,
        national_id: str,
        capture: str,
        vector_methods: Sequence[str] = ("dl", "vit"),
        replace_existing: bool = False,
        random_id: str | None = None,
        created_at: str | None = None,
    ) -> EnrollmentReceipt:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing enrollment image: {file_path}")

        raw_methods = [str(m).strip() for m in vector_methods if str(m).strip()]
        if not raw_methods:
            raise ValueError("vector_methods must not be empty")

        methods: List[str] = []
        seen_methods: set[str] = set()
        for raw_method in raw_methods:
            resolved_method = self._resolve_retrieval_method(raw_method)
            canonical_method = resolved_method.canonical_api_name
            if canonical_method in seen_methods:
                continue
            methods.append(canonical_method)
            seen_methods.add(canonical_method)

        vectors: Dict[str, np.ndarray] = {}
        for method in methods:
            if method not in self.vectorizers:
                raise ValueError(f"Vectorizer for retrieval_method={method} is not configured")
            vectors[method] = _l2_normalize(
                self._vectorize_with_capture(method, str(file_path), _safe_capture(capture))
            )

        return self.store.enroll(
            full_name=full_name,
            national_id=national_id,
            image_bytes=file_path.read_bytes(),
            capture=_safe_capture(capture),
            ext=file_path.suffix or ".png",
            vectors=vectors,
            replace_existing=replace_existing,
            random_id=random_id,
            created_at=created_at,
        )

    # ------------------------------------------------------------------
    # Identification
    # ------------------------------------------------------------------
    def identify_from_path(
        self,
        *,
        path: str,
        capture: str,
        retrieval_method: str = "dl",
        rerank_method: MatchMethod | str = MatchMethod.sift,
        shortlist_size: int = 25,
        threshold: float | None = None,
        hints: IdentifyHints | None = None,
    ) -> IdentifyRunResult:
        hints = hints or IdentifyHints()
        resolved_retrieval = self._resolve_retrieval_method(retrieval_method)
        resolved_rerank = self._resolve_rerank_method(rerank_method)

        retrieval = resolved_retrieval.canonical_api_name
        rerank_method_enum = MatchMethod(resolved_rerank.canonical_api_name)

        if retrieval not in self.vectorizers:
            raise ValueError(f"Vectorizer for retrieval_method={retrieval} is not configured")
        if self.match_service is not None:
            self.match_service.ensure_method_available(rerank_method_enum)

        probe_path = Path(path)
        if not probe_path.is_file():
            raise FileNotFoundError(f"Missing probe image: {probe_path}")

        threshold = float(threshold) if threshold is not None else float(resolved_rerank.decision_threshold)
        capture_norm = _safe_capture(capture)
        shortlist_size = max(1, int(shortlist_size))

        total_t0 = time.perf_counter()
        t0 = time.perf_counter()
        probe_vec = _l2_normalize(self._vectorize_with_capture(retrieval, str(probe_path), capture_norm))
        embed_ms = (time.perf_counter() - t0) * 1000.0

        hint_fields = {
            "name_pattern": hints.name_pattern,
            "national_id_pattern": hints.national_id_pattern,
            "created_from": hints.created_from,
            "created_to": hints.created_to,
        }
        hints_applied = {k: str(v) for k, v in hint_fields.items() if v}

        if hints_applied:
            people = self.store.search_people(hints)
            candidate_ids = [p.random_id for p in people]
            person_map = {p.random_id: p for p in people}
            candidate_pool_size = len(candidate_ids)
        else:
            people = []
            candidate_ids = None
            person_map = {}
            candidate_pool_size = self.store.count_vectors(retrieval)

        empty_result_kwargs = {
            "retrieval_method_metadata": resolved_retrieval.to_metadata(),
            "rerank_method_metadata": resolved_rerank.to_metadata(),
        }

        if candidate_ids is not None and not candidate_ids:
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            return IdentifyRunResult(
                retrieval_method=retrieval,
                rerank_method=rerank_method_enum,
                threshold=threshold,
                decision=False,
                total_enrolled=self.store.total_people(),
                candidate_pool_size=0,
                shortlist_size=0,
                hints_applied=hints_applied,
                top_candidate=None,
                candidates=[],
                latency_ms={
                    "probe_embed_ms": float(embed_ms),
                    "shortlist_scan_ms": 0.0,
                    "rerank_ms": 0.0,
                    "total_ms": float(total_ms),
                },
                storage_layout=self.store.dump_layout(),
                **empty_result_kwargs,
            )

        t1 = time.perf_counter()
        shortlist = self.store.shortlist_by_vector(
            method=retrieval,
            probe_vector=probe_vec,
            limit=shortlist_size,
            candidate_ids=candidate_ids,
        )
        shortlist_ms = (time.perf_counter() - t1) * 1000.0

        if not shortlist:
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            return IdentifyRunResult(
                retrieval_method=retrieval,
                rerank_method=rerank_method_enum,
                threshold=threshold,
                decision=False,
                total_enrolled=self.store.total_people(),
                candidate_pool_size=candidate_pool_size,
                shortlist_size=0,
                hints_applied=hints_applied,
                top_candidate=None,
                candidates=[],
                latency_ms={
                    "probe_embed_ms": float(embed_ms),
                    "shortlist_scan_ms": float(shortlist_ms),
                    "rerank_ms": 0.0,
                    "total_ms": float(total_ms),
                },
                storage_layout=self.store.dump_layout(),
                **empty_result_kwargs,
            )

        t2 = time.perf_counter()
        candidates: List[IdentifyCandidateResult] = []
        for retrieval_rank, (random_id, retrieval_score) in enumerate(shortlist, start=1):
            person = person_map.get(random_id) or self.store.get_person(random_id)
            raw = self.store.load_raw_fingerprint(random_id)
            if person is None or raw is None:
                continue

            rerank_score = self._rerank_probe_against_record(
                probe_path=str(probe_path),
                probe_capture=capture_norm,
                rerank_method=rerank_method_enum,
                raw=raw,
            )
            decision = bool(rerank_score >= threshold)
            candidates.append(
                IdentifyCandidateResult(
                    rank=retrieval_rank,
                    random_id=random_id,
                    full_name=person.full_name,
                    national_id_masked=_mask_national_id(person.national_id),
                    created_at=person.created_at,
                    capture=raw.capture,
                    retrieval_score=float(retrieval_score),
                    rerank_score=float(rerank_score),
                    decision=decision,
                )
            )
        rerank_ms = (time.perf_counter() - t2) * 1000.0

        candidates.sort(
            key=lambda item: (
                float(item.rerank_score if item.rerank_score is not None else item.retrieval_score),
                item.retrieval_score,
            ),
            reverse=True,
        )
        candidates = [replace(item, rank=rank) for rank, item in enumerate(candidates, start=1)]
        top_candidate = candidates[0] if candidates else None
        total_ms = (time.perf_counter() - total_t0) * 1000.0

        return IdentifyRunResult(
            retrieval_method=retrieval,
            rerank_method=rerank_method_enum,
            threshold=threshold,
            decision=bool(top_candidate.decision) if top_candidate else False,
            total_enrolled=self.store.total_people(),
            candidate_pool_size=candidate_pool_size,
            shortlist_size=len(candidates),
            hints_applied=hints_applied,
            top_candidate=top_candidate,
            candidates=candidates,
            latency_ms={
                "probe_embed_ms": float(embed_ms),
                "shortlist_scan_ms": float(shortlist_ms),
                "rerank_ms": float(rerank_ms),
                "total_ms": float(total_ms),
            },
            storage_layout=self.store.dump_layout(),
            retrieval_method_metadata=resolved_retrieval.to_metadata(),
            rerank_method_metadata=resolved_rerank.to_metadata(),
        )

    def stats(self) -> Dict[str, object]:
        return {
            "total_enrolled": self.store.total_people(),
            "storage_layout": self.store.dump_layout(),
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _rerank_probe_against_record(
        self,
        *,
        probe_path: str,
        probe_capture: str,
        rerank_method: MatchMethod,
        raw: RawFingerprintRecord,
    ) -> float:
        suffix = raw.ext if raw.ext.startswith(".") else f".{raw.ext}"
        with tempfile.NamedTemporaryFile(delete=False, prefix=f"cand_{raw.capture}_", suffix=suffix) as tmp:
            tmp.write(raw.image_bytes)
            tmp.flush()
            candidate_path = tmp.name

        try:
            return float(
                self.rerank_callable(
                    rerank_method,
                    probe_path,
                    candidate_path,
                    probe_capture,
                    _safe_capture(raw.capture),
                )
            )
        finally:
            Path(candidate_path).unlink(missing_ok=True)

    def _rerank_with_match_service(
        self,
        method: MatchMethod,
        probe_path: str,
        candidate_path: str,
        probe_capture: str,
        candidate_capture: str,
    ) -> float:
        response = self.match_service.match(
            method=method,
            path_a=str(probe_path),
            path_b=str(candidate_path),
            threshold=None,
            return_overlay=False,
            capture_a=probe_capture,
            capture_b=candidate_capture,
            filename_a=Path(probe_path).name,
            filename_b=Path(candidate_path).name,
        )
        return float(response.score)
