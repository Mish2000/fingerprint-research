from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from apps.api.candidate_source_resolver import (
    CandidateSourceResolver,
    SOURCE_CANDIDATE_SOURCE_MISSING,
    SOURCE_NO_CANDIDATE_SOURCE,
)
from apps.api.method_registry import ApiMethodRegistry, MethodRegistryError, load_api_method_registry
from apps.api.schemas import MatchMethod
from src.fpbench.identification.secure_split_store import (
    EnrollmentReceipt,
    IdentifyHints,
    RawFingerprintRecord,
    SecureSplitFingerprintStore,
)

Vectorizer = Callable[..., np.ndarray]
RerankCallable = Callable[[MatchMethod, str, str, str, str], float]

_MATCH_SERVICE_MODEL_ATTRS = {
    "dl": "dl_resnet",
    "vit": "dl_vit",
}
_MATCH_SERVICE_EMBED_METHODS = {
    "classic_orb": "embed_classic_orb_path",
    "classic_gftt_orb": "embed_classic_gftt_orb_path",
    "minutiae": "embed_minutiae_path",
    "harris": "embed_harris_path",
    "sift": "embed_sift_path",
}

RERANK_PERFORMED = "rerank_performed"
RERANK_SKIPPED_NO_CANDIDATE_SOURCE = "skipped_no_candidate_source"
RERANK_SKIPPED_METHOD_UNAVAILABLE = "skipped_method_unavailable"
RERANK_SKIPPED_VECTOR_ONLY_MODE = "skipped_vector_only_mode"
RERANK_SKIPPED_LEGACY_BYTES_DISABLED = "skipped_legacy_bytes_disabled"
RERANK_FAILED_ERROR = "failed_rerank_error"
RERANK_NOT_APPLICABLE_NO_CANDIDATES = "not_applicable_no_candidates"

CANDIDATE_SOURCE_NO_RAW_METADATA = "no_raw_metadata"
CANDIDATE_SOURCE_LEGACY_DB_IMAGE_BYTES = "legacy_db_image_bytes"
CANDIDATE_SOURCE_LEGACY_DB_IMAGE_BYTES_UNAVAILABLE = "legacy_db_image_bytes_unavailable"

DECISION_STATUS_NO_MATCH = "no_match"
DECISION_STATUS_CANDIDATE_FOUND_RERANK_SKIPPED = "candidate_found_rerank_skipped"
DECISION_STATUS_CANDIDATE_FOUND_RERANK_FAILED = "candidate_found_rerank_failed"
DECISION_STATUS_RERANK_MATCH = "rerank_match"
DECISION_STATUS_RERANK_NO_MATCH = "rerank_no_match"

DECISION_BASIS_NONE = "none"
DECISION_BASIS_RERANK = "rerank"
DECISION_BASIS_VECTOR_SHORTLIST_ONLY = "vector_shortlist_only"

DEFAULT_ENROLLMENT_VECTOR_METHOD_ALIASES = frozenset({"all", "direct", "retrieval"})


def default_enrollment_vector_methods(
    registry: ApiMethodRegistry | None = None,
) -> tuple[str, ...]:
    resolved_registry = registry or load_api_method_registry()
    return tuple(resolved_registry.direct_vector_retrieval_methods())


def resolve_enrollment_vector_methods(
    vector_methods: Sequence[str] | str | None,
    *,
    registry: ApiMethodRegistry | None = None,
) -> list[str]:
    resolved_registry = registry or load_api_method_registry()
    defaults = default_enrollment_vector_methods(resolved_registry)

    if vector_methods is None:
        raw_methods = list(defaults)
    elif isinstance(vector_methods, str):
        stripped = vector_methods.strip()
        if not stripped or stripped.lower() in DEFAULT_ENROLLMENT_VECTOR_METHOD_ALIASES:
            raw_methods = list(defaults)
        else:
            raw_methods = [item.strip() for item in stripped.split(",") if item.strip()]
    else:
        raw_methods = [str(method).strip() for method in vector_methods if str(method).strip()]
        if not raw_methods:
            raw_methods = list(defaults)
        elif len(raw_methods) == 1 and raw_methods[0].lower() in DEFAULT_ENROLLMENT_VECTOR_METHOD_ALIASES:
            raw_methods = list(defaults)

    methods: list[str] = []
    seen_methods: set[str] = set()
    for raw_method in raw_methods:
        resolved_method = resolved_registry.resolve_retrieval_method(raw_method)
        canonical_method = resolved_method.canonical_api_name
        if canonical_method in seen_methods:
            continue
        methods.append(canonical_method)
        seen_methods.add(canonical_method)
    return methods


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
    retrieval_rank: int
    random_id: str
    full_name: str
    national_id_masked: str
    created_at: str
    capture: str
    retrieval_score: float
    vector_score: float
    rerank_score: Optional[float]
    rerank_status: str
    candidate_source_status: str
    decision: Optional[bool]


@dataclass(frozen=True)
class IdentifyRunResult:
    retrieval_method: str
    rerank_method: MatchMethod
    threshold: float
    decision: bool
    decision_status: str
    decision_basis: str
    rerank_status: str
    rerank_summary: Dict[str, Any]
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


@dataclass(frozen=True)
class RerankAttemptResult:
    score: Optional[float]
    rerank_status: str
    candidate_source_status: str


if TYPE_CHECKING:
    from apps.api.service import MatchService


class IdentificationService:
    """
    1:N identification built on top of the existing 1:1 MatchService.

    Search plan:
      1. Optional indexed hint filtering on the PostgreSQL person table
      2. Fast pgvector shortlist over registry-declared retrieval vectors
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
        candidate_source_resolver: CandidateSourceResolver | None = None,
    ):
        self.method_registry = load_api_method_registry()
        self.store = store or SecureSplitFingerprintStore(
            database_url=database_url,
            identity_database_url=identity_database_url,
            table_prefix=table_prefix,
        )
        self.candidate_source_resolver = candidate_source_resolver or CandidateSourceResolver()

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
                method: self._match_service_vectorizer(method)
                for method in self.method_registry.direct_vector_retrieval_methods()
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

    def _match_service_vectorizer(self, method: str) -> Vectorizer:
        def _vectorizer(path: str, capture: Optional[str] = None) -> np.ndarray:
            if self.match_service is None:
                raise RuntimeError("match_service is not configured")
            self.match_service.ensure_method_available(method)

            embed_method_name = _MATCH_SERVICE_EMBED_METHODS.get(method)
            if embed_method_name is not None:
                embed_method = getattr(self.match_service, embed_method_name, None)
                if not callable(embed_method):
                    raise RuntimeError(f"No MatchService vectorizer adapter is registered for {method!r}")
                return np.asarray(embed_method(path, capture=capture), dtype=np.float32)

            model_attr = _MATCH_SERVICE_MODEL_ATTRS.get(method)
            if model_attr is None:  # pragma: no cover - future vector adapters should register here.
                raise RuntimeError(f"No MatchService vectorizer adapter is registered for {method!r}")

            model = getattr(self.match_service, model_attr, None)
            if model is None:
                self.match_service.ensure_method_available(method)
                raise RuntimeError(f"MatchService reported {method!r} available but no vectorizer model is loaded")
            return model.embed_path(path, capture=capture)[0]

        return _vectorizer

    def _ensure_retrieval_available(self, method: str) -> None:
        if self.match_service is not None:
            self.match_service.ensure_method_available(method)

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
        vector_methods: Sequence[str] | None = None,
        replace_existing: bool = False,
        random_id: str | None = None,
        created_at: str | None = None,
    ) -> EnrollmentReceipt:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing enrollment image: {file_path}")

        try:
            methods = resolve_enrollment_vector_methods(
                vector_methods,
                registry=self.method_registry,
            )
        except MethodRegistryError as exc:
            raise ValueError(str(exc)) from exc

        vectors: Dict[str, np.ndarray] = {}
        for method in methods:
            if method not in self.vectorizers:
                raise ValueError(f"Vectorizer for retrieval_method={method} is not configured")
            self._ensure_retrieval_available(method)
            vectors[method] = _l2_normalize(
                self._vectorize_with_capture(method, str(file_path), _safe_capture(capture))
            )
        file_stat = file_path.stat()

        return self.store.enroll(
            full_name=full_name,
            national_id=national_id,
            capture=_safe_capture(capture),
            ext=file_path.suffix or ".png",
            vectors=vectors,
            image_sha256=self._sha256_file(file_path),
            byte_size=int(file_stat.st_size),
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
        skip_rerank: bool = False,
        rerank_limit: int | None = None,
    ) -> IdentifyRunResult:
        hints = hints or IdentifyHints()
        resolved_retrieval = self._resolve_retrieval_method(retrieval_method)
        resolved_rerank = self._resolve_rerank_method(rerank_method)

        retrieval = resolved_retrieval.canonical_api_name
        rerank_method_enum = MatchMethod(resolved_rerank.canonical_api_name)

        if retrieval not in self.vectorizers:
            raise ValueError(f"Vectorizer for retrieval_method={retrieval} is not configured")
        self._ensure_retrieval_available(retrieval)

        probe_path = Path(path)
        if not probe_path.is_file():
            raise FileNotFoundError(f"Missing probe image: {probe_path}")

        threshold = float(threshold) if threshold is not None else float(resolved_rerank.decision_threshold)
        capture_norm = _safe_capture(capture)
        shortlist_size = max(1, int(shortlist_size))
        rerank_limit_value = None if rerank_limit is None else int(rerank_limit)
        skip_rerank = bool(skip_rerank) or (
            rerank_limit_value is not None and rerank_limit_value <= 0
        )
        controlled_rerank_policy = skip_rerank or rerank_limit_value is not None

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
            "decision_status": DECISION_STATUS_NO_MATCH,
            "decision_basis": DECISION_BASIS_NONE,
            "rerank_status": RERANK_NOT_APPLICABLE_NO_CANDIDATES,
            "rerank_summary": self._empty_rerank_summary(),
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
        controlled_rerank_ms = 0.0
        candidates: List[IdentifyCandidateResult] = []
        for retrieval_rank, (random_id, retrieval_score) in enumerate(shortlist, start=1):
            person = person_map.get(random_id) or self.store.get_person(random_id)
            if person is None:
                continue

            should_rerank = (
                not skip_rerank
                and (rerank_limit_value is None or retrieval_rank <= rerank_limit_value)
            )
            if should_rerank:
                rerank_item_t0 = time.perf_counter() if controlled_rerank_policy else None
                raw = self.store.load_raw_fingerprint(random_id)
                rerank_attempt = (
                    self._rerank_probe_against_record(
                        probe_path=str(probe_path),
                        probe_capture=capture_norm,
                        rerank_method=rerank_method_enum,
                        raw=raw,
                    )
                    if raw is not None
                    else RerankAttemptResult(
                        score=None,
                        rerank_status=RERANK_SKIPPED_NO_CANDIDATE_SOURCE,
                        candidate_source_status=CANDIDATE_SOURCE_NO_RAW_METADATA,
                    )
                )
                if rerank_item_t0 is not None:
                    controlled_rerank_ms += (time.perf_counter() - rerank_item_t0) * 1000.0
            else:
                raw = None
                rerank_attempt = RerankAttemptResult(
                    score=None,
                    rerank_status=RERANK_SKIPPED_VECTOR_ONLY_MODE,
                    candidate_source_status=RERANK_SKIPPED_VECTOR_ONLY_MODE,
                )
            rerank_score = rerank_attempt.score
            decision = bool(rerank_score >= threshold) if rerank_score is not None else None
            candidates.append(
                IdentifyCandidateResult(
                    rank=retrieval_rank,
                    retrieval_rank=retrieval_rank,
                    random_id=random_id,
                    full_name=person.full_name,
                    national_id_masked=_mask_national_id(person.national_id),
                    created_at=person.created_at,
                    capture=raw.capture if raw is not None else "unknown",
                    retrieval_score=float(retrieval_score),
                    vector_score=float(retrieval_score),
                    rerank_score=None if rerank_score is None else float(rerank_score),
                    rerank_status=rerank_attempt.rerank_status,
                    candidate_source_status=rerank_attempt.candidate_source_status,
                    decision=decision,
                )
            )
        rerank_ms = (
            controlled_rerank_ms
            if controlled_rerank_policy
            else (time.perf_counter() - t2) * 1000.0
        )

        candidates.sort(
            key=lambda item: (
                item.rerank_score is not None,
                float(item.rerank_score if item.rerank_score is not None else item.retrieval_score),
                item.retrieval_score,
            ),
            reverse=True,
        )
        candidates = [replace(item, rank=rank) for rank, item in enumerate(candidates, start=1)]
        top_candidate = candidates[0] if candidates else None
        rerank_summary = self._build_rerank_summary(candidates)
        rerank_status = self._top_level_rerank_status(rerank_summary)
        decision, decision_status, decision_basis = self._decision_fields(
            candidates=candidates,
            top_candidate=top_candidate,
            rerank_summary=rerank_summary,
        )
        total_ms = (time.perf_counter() - total_t0) * 1000.0

        return IdentifyRunResult(
            retrieval_method=retrieval,
            rerank_method=rerank_method_enum,
            threshold=threshold,
            decision=decision,
            decision_status=decision_status,
            decision_basis=decision_basis,
            rerank_status=rerank_status,
            rerank_summary=rerank_summary,
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
    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _legacy_candidate_image_bytes(self, raw: RawFingerprintRecord) -> bytes | None:
        if not bool(getattr(raw, "legacy_image_bytes_present", False)):
            return None
        loader = getattr(self.store, "load_legacy_raw_fingerprint_image_bytes", None)
        if callable(loader):
            return loader(raw.random_id)
        legacy_bytes = getattr(raw, "image_bytes", None)
        return bytes(legacy_bytes) if legacy_bytes is not None else None

    def _rerank_probe_against_record(
        self,
        *,
        probe_path: str,
        probe_capture: str,
        rerank_method: MatchMethod,
        raw: RawFingerprintRecord,
    ) -> RerankAttemptResult:
        source = self.candidate_source_resolver.resolve(raw.random_id)
        if source.available and source.path is not None:
            return self._rerank_candidate_path(
                probe_path=probe_path,
                probe_capture=probe_capture,
                rerank_method=rerank_method,
                candidate_path=str(source.path),
                candidate_capture=_safe_capture(source.capture or raw.capture),
                candidate_source_status=source.status,
            )

        if source.status == SOURCE_CANDIDATE_SOURCE_MISSING:
            return RerankAttemptResult(
                score=None,
                rerank_status=RERANK_SKIPPED_NO_CANDIDATE_SOURCE,
                candidate_source_status=SOURCE_CANDIDATE_SOURCE_MISSING,
            )

        # Transitional/deprecated path for rows that predate metadata-only enrollment.
        # New enrollments never write these bytes, and the explicit adapter above is
        # the only DB-byte route kept for legacy compatibility.
        legacy_image_bytes = self._legacy_candidate_image_bytes(raw)
        if legacy_image_bytes is None:
            if bool(getattr(raw, "legacy_image_bytes_present", False)):
                return RerankAttemptResult(
                    score=None,
                    rerank_status=RERANK_SKIPPED_LEGACY_BYTES_DISABLED,
                    candidate_source_status=CANDIDATE_SOURCE_LEGACY_DB_IMAGE_BYTES_UNAVAILABLE,
                )
            return RerankAttemptResult(
                score=None,
                rerank_status=RERANK_SKIPPED_NO_CANDIDATE_SOURCE,
                candidate_source_status=source.status or SOURCE_NO_CANDIDATE_SOURCE,
            )

        suffix = raw.ext if raw.ext.startswith(".") else f".{raw.ext}"
        with tempfile.NamedTemporaryFile(delete=False, prefix=f"cand_{raw.capture}_", suffix=suffix) as tmp:
            tmp.write(legacy_image_bytes)
            tmp.flush()
            candidate_path = tmp.name

        try:
            return self._rerank_candidate_path(
                probe_path=probe_path,
                probe_capture=probe_capture,
                rerank_method=rerank_method,
                candidate_path=candidate_path,
                candidate_capture=_safe_capture(raw.capture),
                candidate_source_status=CANDIDATE_SOURCE_LEGACY_DB_IMAGE_BYTES,
            )
        finally:
            Path(candidate_path).unlink(missing_ok=True)

    def _rerank_candidate_path(
        self,
        *,
        probe_path: str,
        probe_capture: str,
        rerank_method: MatchMethod,
        candidate_path: str,
        candidate_capture: str,
        candidate_source_status: str,
    ) -> RerankAttemptResult:
        try:
            self._ensure_rerank_available(rerank_method)
            score = float(
                self.rerank_callable(
                    rerank_method,
                    probe_path,
                    candidate_path,
                    probe_capture,
                    _safe_capture(candidate_capture),
                )
            )
        except Exception as exc:
            if self._is_method_unavailable_error(exc):
                return RerankAttemptResult(
                    score=None,
                    rerank_status=RERANK_SKIPPED_METHOD_UNAVAILABLE,
                    candidate_source_status=candidate_source_status,
                )
            return RerankAttemptResult(
                score=None,
                rerank_status=RERANK_FAILED_ERROR,
                candidate_source_status=candidate_source_status,
            )
        return RerankAttemptResult(
            score=score,
            rerank_status=RERANK_PERFORMED,
            candidate_source_status=candidate_source_status,
        )

    def _ensure_rerank_available(self, method: MatchMethod) -> None:
        if self.match_service is not None:
            self.match_service.ensure_method_available(method)

    @staticmethod
    def _is_method_unavailable_error(exc: Exception) -> bool:
        return exc.__class__.__name__ == "MethodUnavailableError"

    @staticmethod
    def _status_counts(items: Sequence[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return counts

    @classmethod
    def _empty_rerank_summary(cls) -> Dict[str, Any]:
        return {
            "candidate_count": 0,
            "performed_count": 0,
            "skipped_count": 0,
            "failed_count": 0,
            "rerank_status_counts": {},
            "candidate_source_status_counts": {},
        }

    @classmethod
    def _build_rerank_summary(cls, candidates: Sequence[IdentifyCandidateResult]) -> Dict[str, Any]:
        rerank_statuses = [candidate.rerank_status for candidate in candidates]
        candidate_source_statuses = [candidate.candidate_source_status for candidate in candidates]
        performed_count = sum(1 for status in rerank_statuses if status == RERANK_PERFORMED)
        failed_count = sum(1 for status in rerank_statuses if status == RERANK_FAILED_ERROR)
        skipped_count = sum(1 for status in rerank_statuses if status.startswith("skipped_"))
        return {
            "candidate_count": len(candidates),
            "performed_count": performed_count,
            "skipped_count": skipped_count,
            "failed_count": failed_count,
            "rerank_status_counts": cls._status_counts(rerank_statuses),
            "candidate_source_status_counts": cls._status_counts(candidate_source_statuses),
        }

    @staticmethod
    def _top_level_rerank_status(summary: Dict[str, Any]) -> str:
        candidate_count = int(summary.get("candidate_count") or 0)
        if candidate_count <= 0:
            return RERANK_NOT_APPLICABLE_NO_CANDIDATES
        if int(summary.get("performed_count") or 0) > 0:
            return RERANK_PERFORMED
        if int(summary.get("failed_count") or 0) >= candidate_count:
            return RERANK_FAILED_ERROR

        counts = dict(summary.get("rerank_status_counts") or {})
        for status in (
            RERANK_SKIPPED_NO_CANDIDATE_SOURCE,
            RERANK_SKIPPED_METHOD_UNAVAILABLE,
            RERANK_SKIPPED_LEGACY_BYTES_DISABLED,
        ):
            if int(counts.get(status) or 0) >= candidate_count:
                return status
        if int(summary.get("failed_count") or 0) > 0:
            return RERANK_FAILED_ERROR
        return RERANK_SKIPPED_VECTOR_ONLY_MODE

    @staticmethod
    def _decision_fields(
        *,
        candidates: Sequence[IdentifyCandidateResult],
        top_candidate: Optional[IdentifyCandidateResult],
        rerank_summary: Dict[str, Any],
    ) -> tuple[bool, str, str]:
        if not candidates or top_candidate is None:
            return False, DECISION_STATUS_NO_MATCH, DECISION_BASIS_NONE

        if int(rerank_summary.get("performed_count") or 0) > 0:
            decision = bool(top_candidate.decision)
            status = DECISION_STATUS_RERANK_MATCH if decision else DECISION_STATUS_RERANK_NO_MATCH
            return decision, status, DECISION_BASIS_RERANK

        if int(rerank_summary.get("failed_count") or 0) > 0:
            return (
                False,
                DECISION_STATUS_CANDIDATE_FOUND_RERANK_FAILED,
                DECISION_BASIS_VECTOR_SHORTLIST_ONLY,
            )

        return (
            False,
            DECISION_STATUS_CANDIDATE_FOUND_RERANK_SKIPPED,
            DECISION_BASIS_VECTOR_SHORTLIST_ONLY,
        )

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
